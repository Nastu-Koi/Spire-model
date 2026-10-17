"""The sampling side of an overlapped run (`iterate --overlap`): a process that plays
the rounds and evaluations while the driver's process updates the policy.

The driver forks it before touching the GPU, so that it has a CUDA context of its
own. It holds one model object, loaded from the first job's checkpoint, and the
sampling processes forked around that object (`sampling.Samplers`); every later job
puts another archive's weights into the same object in place (`checkpoint.load_weights`).
A job names the policy and asks for a round of act groups, an evaluation, or both;
each result goes back over the pipe as it is made, then `done`. A failure ends the
side: the driver raises what it reported.
"""

import gc
import multiprocessing
import os
import traceback
from pathlib import Path

import torch

from main.env.processes import end, end_with
from main.env.runtime import configure_gpu
from main.iterate import play_evaluation, play_round, round_result, samplers_for
from main.network.checkpoint import load_weights
from main.network.config import LearningRates
from main.play.rollout import RolloutRunner
from main.play.sampling import Samplers


class SamplingSide:
    """The driver's handle on the sampling side."""

    def __init__(self, engine_factory, settings, device):
        if str(device).startswith("cuda") and torch.cuda.is_initialized():
            raise RuntimeError("The sampling side forks before the driver's process initializes CUDA")
        context = multiprocessing.get_context("fork")
        self.connection, theirs = context.Pipe()
        self.process = context.Process(target=_serve, args=(theirs, os.getpid(), engine_factory, settings, device))
        self.process.start()
        theirs.close()
        self.pending = False

    def dispatch(self, job):
        """Hand the side a job; `collect` waits for its results."""
        if self.pending:
            raise RuntimeError("The sampling side is on a job already")
        self.connection.send(job)
        self.pending = True

    def collect(self):
        """The results of the job under way, by kind (`round`, `evaluation`)."""
        if not self.pending:
            raise RuntimeError("The sampling side has no job to collect")
        results = {}
        try:
            while True:
                kind, value = self.connection.recv()
                if kind == "done":
                    break
                if kind == "error":
                    raise RuntimeError("The sampling side failed:\n" + value)
                results[kind] = value
        except EOFError:
            raise RuntimeError("The sampling side exited without finishing its job") from None
        finally:
            self.pending = False
        return results

    def run(self, job):
        self.dispatch(job)
        return self.collect()

    def close(self):
        self.connection.close()
        end([self.process])


def _serve(connection, driver, engine_factory, settings, device):
    from main.cli import _learner, _profile

    status = 1
    model = vocabulary = config = pool = None
    try:
        # The side leads a process group with its sampling processes and their
        # engines, so that the driver can end them together.
        os.setsid()
        end_with(driver)
        # The parent's intra-op pool does not survive the fork: a child that computes
        # with it waits forever. The side's network runs on the GPU; the host needs
        # one thread.
        torch.set_num_threads(1)
        configure_gpu(device)
        while True:
            try:
                job = connection.recv()
            except EOFError:
                break
            if job is None:
                break
            policy = Path(job["policy"])
            if model is None:
                # Loaded as the driver loads (`Driver.load`): the profile's sampling settings,
                # the rates file's rates, so that the strict resume accepts the profile.
                rates = settings.learning_rates
                learner, manifest = _learner(policy, device, inference=True, training=_profile(settings.config),
                                             rates=LearningRates.load(rates) if rates else None)
                model, vocabulary, config = learner.model, learner.vocabulary, learner.config
                # Forked around the model just loaded, before anything else is allocated.
                pool = Samplers(engine_factory, model, vocabulary, engines=settings.workers,
                                processes=settings.processes)
            else:
                manifest = load_weights(model, policy, device)
            progress = manifest.get("progress", {})
            version = progress.get("policy_version", 0)
            metadata = {key: value for key, value in progress.items() if key not in {"policy_version", "updates"}}

            def runner(iteration=None):
                return RolloutRunner(model, vocabulary, precision=config.precision, version=version,
                                     max_steps=settings.max_steps, ascension=config.ascension,
                                     temperatures=config.temperatures, iteration=iteration)

            def samplers(engines):
                return samplers_for(pool, engines, settings, engine_factory, model, vocabulary)

            if "round" in job:
                asked = job["round"]
                summary = play_round(samplers(settings.workers), runner(asked["iteration"]), asked["seeds"],
                                     asked["directory"], settings, Path(asked["directory"]).parents[1])
                connection.send(("round", round_result(summary, asked["directory"], settings)))
            if "evaluation" in job:
                asked = job["evaluation"]
                result = play_evaluation(samplers(settings.eval_workers or settings.workers), runner(),
                                         asked["seeds"], asked["directory"], asked["iteration"], metadata,
                                         Path(asked["directory"]).parents[1])
                connection.send(("evaluation", result))
            gc.collect()
            if torch.cuda.is_available() and str(device).startswith("cuda"):
                torch.cuda.empty_cache()
            connection.send(("done", None))
        status = 0
    except BaseException:
        try:
            connection.send(("error", traceback.format_exc()))
        except OSError:
            # The driver has gone: there is nobody left to tell.
            pass
    finally:
        if pool is not None:
            pool.close()
        os._exit(status)
