"""Rounds sampled by several processes around one GPU.

Between two passes of the network a decision costs milliseconds of Python: the
engine's reply is parsed, the journal written, the frame turned into tokens and the
tokens into tensors. In one process all of that shares an interpreter lock with the
thread that runs the network, and the lock, not the GPU or the engines, bounds the
round. Here the engines and that work live in sampling processes. The collecting
process keeps the model: it joins what the processes have packed since its last
pass, chooses for all of it in one pass and sends the choices back.

A pass takes about as long for one decision as for a hundred, and a decision waits
for the pass under way to end before its own begins. So a round is as fast as the
decisions each pass carries: every engine should be waiting for nothing but the
network, which is why a sampling process is best given a single engine.

Sampling processes are forked: they share the vocabulary with the collecting process
without copying, and never touch its CUDA state. What they pack crosses in the flat
form of `wire`. They stay from round to round with their engines: an engine process
begins its next run without loading the game again.

The update that follows a round needs every decision packed once more. The collecting
process has all of them in hand as they pass: asked to, it keeps them, and hands the
round over as runs whose macro-actions hold packed decisions in place of frames.
"""

import gc
import multiprocessing
import os
import queue
import threading
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from multiprocessing.connection import wait

import torch

from .policy import Choice, SessionPolicy, prepare_decisions
from .processes import end, end_with, release_memory
from .protocol import ProtocolError
from .representation import forget
from .rollout import RolloutRunner, play, precision_context
from .trainer import climb
from .wire import pack, unpack

# A sampling process says something whenever a decision or a run is ready. Silence
# this long is checked against the processes being alive.
SILENCE_SECONDS = 60


class _Waiting:
    """The decisions the engines of one sampling process wait for."""

    def __init__(self, connection, send, vocabulary, version):
        self.connection, self.send = connection, send
        self.vocabulary, self.version = vocabulary, version
        self.requests = queue.Queue()

    def policy(self, job):
        """The policy of one run: `job` says whose decisions they are."""
        session = SessionPolicy(None, self.vocabulary, version=self.version)
        requests = self.requests

        class Remote:
            def reset(self):
                session.reset()

            def choose(self, frame, *, sample=True):
                candidates = session.advance(frame)
                # A forced action needs no network, hence no trip to it.
                if len(candidates) == 1:
                    return Choice(candidates[0]["candidate_ref"])
                result = Future()
                requests.put((job, frame, result))
                return result.result()

        return Remote()

    def serve(self):
        """Send the decisions that wait, if any, and hand out their choices. While the
        collecting process works on them, the other engines queue the next ones. Each
        decision is packed on its own: an update takes a round apart by macro-action."""
        try:
            waiting = [self.requests.get(timeout=0.01)]
        except queue.Empty:
            return
        while True:
            try:
                waiting.append(self.requests.get_nowait())
            except queue.Empty:
                break
        observations, messages = [], []
        for job, frame, _ in waiting:
            (obs,), decisions = prepare_decisions(self.vocabulary, [frame])
            observations.append(obs)
            messages.append((job, pack(decisions)))
        self.send(("decisions", messages))
        chosen = self.connection.recv()
        for (_, frame, result), obs, (slot, log_prob, value) in zip(waiting, observations, chosen):
            slot = obs.slot_refs[int(slot)]
            result.set_result(Choice(
                next(c["candidate_ref"] for c in frame["legal"]["candidates"]
                     if c["decoder_slot_ref"] == slot),
                log_prob, None, value))


def _without_frames(run):
    """A run as its sampling process hands it over: everything but its frames. Each
    macro-action says how many decisions it holds, for the packed ones kept of them."""
    handed = {key: value for key, value in run.items() if key not in ("macros", "environment_actions")}
    handed["climb"] = climb(run)
    handed["macros"] = [
        dict({key: value for key, value in macro.items() if key != "steps"},
             branches=sum(not step["forced"] for step in macro["steps"]))
        for macro in run["macros"]
    ]
    return handed


def _sample(connection, inherited, collector, engine_factory, model, vocabulary, taken, count):
    """A sampling process: round after round, its `count` engines play the jobs they
    take, each engine one run after another."""
    status = 1
    sending = threading.Lock()
    engines = [None] * count

    def send(message):
        # Engine threads report their runs on the connection the decisions go out on.
        with sending:
            connection.send(message)

    try:
        # The process leads a group with its engines, so that the collecting process can
        # end them together when it stops early.
        os.setsid()
        end_with(collector)
        for other in inherited:
            other.close()
        # The parent's intra-op pool does not survive the fork; packing needs none.
        torch.set_num_threads(1)
        while True:
            try:
                settings, digest, jobs, sample, save = connection.recv()
            except EOFError:
                # The collecting process has no more rounds.
                break
            runner = RolloutRunner(model, vocabulary, **settings)
            # Hashed once by the collecting process, not here from weights that are
            # not this process's to read.
            runner._digest = digest
            waiting = _Waiting(connection, send, vocabulary, runner.version)
            finished = []

            def work(slot):
                while True:
                    with taken.get_lock():
                        index = taken.value
                        taken.value += 1
                    if index >= len(jobs):
                        return
                    if engines[slot] is None:
                        engines[slot] = engine_factory()
                    run = play(engines[slot], runner, jobs[index], sample=sample,
                               policy=waiting.policy(index), save=save)
                    send(("run", (index, _without_frames(run))))
                    finished.append(index)

            # No `with`: after a failure the engine threads wait for choices that will
            # not come, and the process must leave without them.
            pool = ThreadPoolExecutor(max_workers=count, thread_name_prefix="engine")
            working = [pool.submit(work, slot) for slot in range(count)]
            while working:
                waiting.serve()
                if finished:
                    # A run leaves its frames and what parsing cached of it: tens of
                    # megabytes that this process would otherwise keep, like every
                    # other sampling process, for as long as it lives.
                    finished.clear()
                    forget()
                    release_memory()
                for future in [future for future in working if future.done()]:
                    future.result()
                    working.remove(future)
            pool.shutdown()
            # Nothing of the round is held while the collecting process updates.
            forget()
            release_memory()
            send(("done", None))
        status = 0
    except BaseException:
        try:
            send(("error", traceback.format_exc()))
        except OSError:
            # The collecting process has gone: there is nobody left to tell.
            pass
    finally:
        # The model this process inherited is not its own to release. Its engines end
        # with its process group.
        os._exit(status)


class Samplers:
    """Sampling processes holding `engines` engines between them, for the rounds of
    one model and vocabulary. With `keep`, every round's packed decisions are kept for
    the update that follows it (see `runs`)."""

    def __init__(self, engine_factory, model, vocabulary, *, engines, processes, keep=False):
        if type(engines) is not int or type(processes) is not int or min(engines, processes) < 1:
            raise ValueError("Sampling needs positive numbers of engines and processes")
        self.model, self.vocabulary, self.keep = model, vocabulary, keep
        self.handed = None
        processes = min(processes, engines)
        context = multiprocessing.get_context("fork")
        self.taken = context.Value("i", 0)
        self.samplers, self.connections = [], []
        # A collection in a sampling process would write to every object it inherits
        # and so copy it: what exists now is set aside from the collector.
        gc.collect()
        gc.freeze()
        try:
            for index in range(processes):
                ours, theirs = context.Pipe()
                process = context.Process(target=_sample, args=(
                    theirs, [*self.connections, ours], os.getpid(), engine_factory, model,
                    vocabulary, self.taken, engines // processes + (index < engines % processes)))
                process.start()
                theirs.close()
                self.samplers.append(process)
                self.connections.append(ours)
        except BaseException:
            self.close()
            raise
        finally:
            gc.unfreeze()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        """End the processes and their engines."""
        for connection in self.connections:
            connection.close()
        end(self.samplers)
        self.samplers, self.connections = [], []

    def runs(self):
        """The runs of the last round, in the order of its jobs, as their sampling
        processes handed them over: without frames, each macro-action holding the
        packed decisions (`packed`, one message each) and the slots chosen (`slots`)."""
        if self.handed is None:
            raise RuntimeError("No round was kept")
        runs = []
        for summary, decisions in zip(*self.handed):
            run, taken = dict(summary, macros=[]), 0
            for macro in summary["macros"]:
                macro = dict(macro)
                mine = decisions[taken:taken + macro.pop("branches")]
                taken += len(mine)
                # The run added up the same choices in the same order.
                if "old_log_prob" in macro and sum(row[1] for _, row in mine) != macro["old_log_prob"]:
                    raise ProtocolError("Kept decisions are not the ones a macro-action was played with")
                macro["packed"] = [message for message, _ in mine]
                macro["slots"] = [int(row[0]) for _, row in mine]
                run["macros"].append(macro)
            if taken != len(decisions):
                raise ProtocolError("A run's kept decisions do not match its macro-actions")
            runs.append(run)
        return runs

    def collect(self, runner, jobs, *, sample=True, on_run=None, save=True):
        """Play `jobs`, each written where and as `rollout.play` writes it. Returns the
        runs without their frames, in the order of the jobs. A failure ends the
        processes: there is no telling what their engines are in the middle of."""
        if runner.model is not self.model or runner.vocabulary is not self.vocabulary:
            raise ValueError("The round's model and vocabulary differ from the sampling processes'")
        if not self.samplers:
            raise RuntimeError("The sampling processes have ended")
        try:
            return self._collect(runner, jobs, sample, on_run, save)
        except BaseException:
            self.close()
            raise

    def _collect(self, runner, jobs, sample, on_run, save):
        self.model.eval()
        self.taken.value = 0
        settings = {name: getattr(runner, name) for name in (
            "precision", "version", "max_steps", "timeout", "ascension", "claim_rewards",
            "temperature")}
        for connection in self.connections:
            connection.send((settings, runner.digest, jobs, sample, save))
        active = dict(enumerate(self.connections))
        completed = 0
        # What is kept of the round: every run without its frames, and with `keep` its
        # decisions as they passed, each (packed message, (slot, log-probability, value)).
        self.handed = None
        summaries, kept = [None] * len(jobs), [[] for _ in jobs] if self.keep else None
        while active:
            ready = wait(list(active.values()), timeout=SILENCE_SECONDS)
            if not ready:
                # A process that died without its last word would be waited for forever.
                dead = [index for index in active if not self.samplers[index].is_alive()]
                if dead:
                    raise RuntimeError(f"Sampling process {dead[0]} exited without finishing")
                continue
            asked = []
            for connection in ready:
                index = next(i for i, c in active.items() if c is connection)
                try:
                    # A process has one batch of decisions out at a time; the runs its
                    # engines finished meanwhile are read with it.
                    while index in active and connection.poll():
                        kind, value = connection.recv()
                        if kind == "decisions":
                            asked.append((connection, value))
                        elif kind == "run":
                            job, summaries[job] = value
                            completed += 1
                            if on_run:
                                on_run(summaries[job]["character"], completed, len(jobs))
                        elif kind == "error":
                            raise RuntimeError("Sampling process failed:\n" + value)
                        else:
                            del active[index]
                except EOFError:
                    raise RuntimeError(f"Sampling process {index} exited without finishing") from None
            if asked:
                self._answer(runner, asked, sample, kept)
        if kept is not None:
            self.handed = summaries, kept
        return summaries

    def _answer(self, runner, asked, sample, kept):
        """Choose for the decisions of several sampling processes in one pass. Each
        process sent a list of (job, packed decision)."""
        tagged = [pair for _, pairs in asked for pair in pairs]
        decisions, unregistered = unpack([message for _, message in tagged], self.model.device)
        self.vocabulary.report(unregistered)
        with torch.no_grad(), precision_context(self.model, runner.precision):
            chosen = self.model.choose(decisions, sample=sample, temperature=runner.temperature)
            # One transfer for the whole pass: (slot, log-probability, value) rows.
            rows = torch.stack([column.float() for column in chosen], dim=1).cpu().tolist()
        if kept is not None:
            for (job, message), row in zip(tagged, rows):
                kept[job].append((message, row))
        start = 0
        for connection, pairs in asked:
            connection.send(rows[start:start + len(pairs)])
            start += len(pairs)
