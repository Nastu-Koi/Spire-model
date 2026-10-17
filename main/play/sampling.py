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
form of `wire`, each decision with the temperature it is drawn at. They stay from round
to round with their engines: an engine process begins its next run without loading
the game again. Between rounds that something else needs the memory for, the engines
can end while the processes stay (`Samplers.rest`). Every run is written where it was played, or handed over compressed
for the wave's shards (`main/play/groups.py`) as it ends; nothing of a round stays in
memory for whatever learns from it.
"""

import gc
import multiprocessing
import os
import queue
import threading
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from multiprocessing.connection import wait

import torch

from main.network.policy import Choice, SessionPolicy, prepare_decisions
from main.env.processes import end, end_with, release_memory
from main.encoding.representation import forget
from main.play.rollout import FULL_RUNS, RolloutRunner, packed_line, play, precision_context
from main.play.evaluation import climb
from main.play.wire import pack, unpack

# A sampling process says something whenever a decision or a run is ready. Silence
# this long is checked against the processes being alive.
SILENCE_SECONDS = 60


class _Waiting:
    """The decisions the engines of one sampling process wait for."""

    def __init__(self, connection, send, vocabulary, runner):
        self.connection, self.send = connection, send
        self.vocabulary, self.runner = vocabulary, runner
        self.requests = queue.Queue()

    def policy(self, job):
        """The policy of one run: `job` says whose decisions they are."""
        session = SessionPolicy(None, self.vocabulary, version=self.runner.version)
        requests = self.requests

        class Remote:
            def reset(self):
                session.reset()

            def choose(self, frame, *, sample=True, temperature=1.0):
                candidates = session.advance(frame)
                # A forced action needs no network, hence no trip to it.
                if len(candidates) == 1:
                    return Choice(candidates[0]["candidate_ref"])
                result = Future()
                requests.put((job, frame, temperature, result))
                return result.result()

        return Remote()

    def serve(self):
        """Send the decisions that wait, if any, and hand out their choices. While the
        collecting process works on them, the other engines queue the next ones."""
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
        for job, frame, temperature, _ in waiting:
            (obs,), decisions = prepare_decisions(self.vocabulary, [frame])
            observations.append(obs)
            messages.append((job, pack(decisions), temperature))
        self.send(("decisions", messages))
        chosen = self.connection.recv()
        for (_, frame, _, result), obs, (slot, log_prob, base) in zip(waiting, observations, chosen):
            slot = obs.slot_refs[int(slot)]
            result.set_result(Choice(
                next(c["candidate_ref"] for c in frame["legal"]["candidates"]
                     if c["decoder_slot_ref"] == slot),
                log_prob, base_log_prob=base))


def _without_frames(run):
    """A run as its sampling process hands it over: everything but its frames. Each
    macro-action says how many decisions it holds, as the fights of a sampled run do
    already."""
    handed = {key: value for key, value in run.items() if key not in ("macros", "environment_actions", "after")}
    handed["macros"] = [
        macro if "steps" not in macro else
        dict({key: value for key, value in macro.items() if key != "steps"},
             branches=sum(not step["forced"] for step in macro["steps"]), decisions=len(macro["steps"]))
        for macro in run["macros"]
    ]
    return handed


def _handed(run, layout):
    """What crosses to the collecting process of a run: its summary and, where the
    round keeps the run for training rather than in a file of its own, the run
    compressed as the line it makes in a shard, with what its index row counts."""
    # A segment's continue save goes to the collecting process beside the run, not in
    # the shard line it makes.
    save = run.pop("save", None)
    handed = _without_frames(run)
    if save is not None:
        handed["save"] = save
    if not layout["trajectory"]:
        handed.update(_shard_line(run))
    return handed


def _shard_line(run):
    """A run, its frames still packed, as the line it makes in a shard, with what its
    index row counts."""
    line, phases = packed_line(run)
    return dict(member=line, phases=phases,
                stored_steps=sum(len(macro["steps"]) for macro in run["macros"] if "steps" in macro))


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
                message = connection.recv()
            except EOFError:
                # The collecting process has no more rounds.
                break
            if message is None:
                # A rest (`Samplers.rest`): the engines end, the process stays.
                for slot, engine in enumerate(engines):
                    if engine is not None and hasattr(engine, "close"):
                        engine.close()
                    engines[slot] = None
                forget()
                release_memory()
                send(("rested", None))
                continue
            settings, digest, jobs, sample, layout = message
            runner = RolloutRunner(model, vocabulary, **settings)
            # Hashed once by the collecting process, not here from weights that are
            # not this process's to read.
            runner._digest = digest
            waiting = _Waiting(connection, send, vocabulary, runner)
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
                               policy=waiting.policy(index), **layout)
                    send(("run", (index, _handed(run, layout))))
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
            # Nothing of the round is held between rounds.
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


class _Timing:
    """The collecting process's account of one round."""

    def __init__(self):
        self.started = time.monotonic()
        self.seconds = 0.0
        self.passes = self.decisions = 0
        self.waiting = self.unpacking = self.choosing = self.sending = 0.0

    def report(self):
        accounted = self.waiting + self.unpacking + self.choosing + self.sending
        return dict(
            seconds=round(self.seconds, 3), passes=self.passes, decisions=self.decisions,
            decisions_per_pass=round(self.decisions / self.passes, 3) if self.passes else None,
            decisions_per_second=round(self.decisions / self.seconds, 3) if self.seconds else None,
            seconds_per_pass=round(self.seconds / self.passes, 6) if self.passes else None,
            waiting_seconds=round(self.waiting, 3), unpacking_seconds=round(self.unpacking, 3),
            choosing_seconds=round(self.choosing, 3), sending_seconds=round(self.sending, 3),
            other_seconds=round(max(self.seconds - accounted, 0.0), 3))


class Samplers:
    """Sampling processes holding `engines` engines between them, for the rounds of
    one model and vocabulary. `processes` of them, at most one per engine."""

    def __init__(self, engine_factory, model, vocabulary, *, engines, processes=None):
        processes = engines if processes is None else processes
        if type(engines) is not int or type(processes) is not int or min(engines, processes) < 1:
            raise ValueError("Sampling needs positive numbers of engines and processes")
        self.model, self.vocabulary = model, vocabulary
        # Where the collecting process's time went in the last round (`timing`).
        self._timing = _Timing()
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

    def rest(self):
        """End the engines and keep the processes, until the next round starts the
        engines again. A process forked once, before anything is trained, can serve
        every round: one forked later inherits the trainer's heap, whose free gaps it
        fills page by page with copies of its own."""
        if not self.samplers:
            return
        try:
            for connection in self.connections:
                connection.send(None)
            for index, connection in enumerate(self.connections):
                kind, value = connection.recv()
                if kind == "error":
                    raise RuntimeError("Sampling process failed:\n" + value)
                if kind != "rested":
                    raise RuntimeError(f"Sampling process {index} answered a rest with {kind}")
        except EOFError:
            self.close()
            raise RuntimeError("A sampling process exited while its engines were ending") from None
        except BaseException:
            self.close()
            raise

    def close(self):
        """End the processes and their engines."""
        for connection in self.connections:
            connection.close()
        end(self.samplers)
        self.samplers, self.connections = [], []

    def collect(self, runner, jobs, *, sample=True, on_run=None, layout=None):
        """Play `jobs`, each written where and as `rollout.play` writes it under
        `layout` (`rollout.FULL_RUNS` unless told otherwise). Returns the runs without
        their frames, in the order of the jobs; `on_run(job index, run)` is told of
        each as it arrives. A failure ends the processes: there is no telling what
        their engines are in the middle of."""
        if runner.model is not self.model or runner.vocabulary is not self.vocabulary:
            raise ValueError("The round's model and vocabulary differ from the sampling processes'")
        if not self.samplers:
            raise RuntimeError("The sampling processes have ended")
        try:
            return self._collect(runner, jobs, sample, on_run, FULL_RUNS if layout is None else dict(layout))
        except BaseException:
            self.close()
            raise

    def _collect(self, runner, jobs, sample, on_run, layout):
        self.model.eval()
        # A pass of a few decisions is bound by the host: the global layers replay as
        # CUDA graphs (`PolicyNetwork.graphed`); training's replay is untouched.
        if hasattr(self.model, "graphed"):
            self.model.graphed(True)
        self.taken.value = 0
        settings = {name: getattr(runner, name) for name in (
            "precision", "version", "max_steps", "timeout", "ascension", "claim_rewards",
            "temperatures", "iteration")}
        for connection in self.connections:
            connection.send((settings, runner.digest, jobs, sample, layout))
        active = dict(enumerate(self.connections))
        completed = 0
        summaries = [None] * len(jobs)
        timing = self._timing = _Timing()
        while active:
            waited = time.monotonic()
            ready = wait(list(active.values()), timeout=SILENCE_SECONDS)
            timing.waiting += time.monotonic() - waited
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
                            job, run = value
                            completed += 1
                            if on_run:
                                on_run(job, run)
                            # The shard line went to `on_run`'s writer; a round of
                            # thousands of segments is not held here whole.
                            summaries[job] = {key: v for key, v in run.items() if key != "member"}
                        elif kind == "error":
                            raise RuntimeError("Sampling process failed:\n" + value)
                        else:
                            del active[index]
                except EOFError:
                    raise RuntimeError(f"Sampling process {index} exited without finishing") from None
            if asked:
                self._answer(runner, asked, sample)
        timing.seconds = time.monotonic() - timing.started
        return summaries

    @property
    def timing(self):
        """Where the collecting process's time went in the last round: how many passes
        of the network it made, the decisions they carried, and the seconds spent
        waiting for the sampling processes, unpacking their decisions, choosing (the
        pass and the copy of its choices back to the host) and sending the choices."""
        return self._timing.report()

    def _answer(self, runner, asked, sample):
        """Choose for the decisions of several sampling processes in one pass. Each
        process sent a list of (job, packed decision, temperature)."""
        timing, began = self._timing, time.monotonic()
        tagged = [triple for _, triples in asked for triple in triples]
        decisions, unregistered = unpack([message for _, message, _ in tagged], self.model.device)
        self.vocabulary.report(unregistered)
        unpacked = time.monotonic()
        with torch.no_grad(), precision_context(self.model, runner.precision):
            chosen = self.model.choose(
                decisions, sample=sample,
                temperature=torch.tensor([temperature for _, _, temperature in tagged],
                                         dtype=torch.float32, device=self.model.device))
            # One transfer for the whole pass: (slot, log-probability, at temperature 1) rows.
            rows = torch.stack([column.float() for column in chosen], dim=1).cpu().tolist()
        chose = time.monotonic()
        start = 0
        for connection, triples in asked:
            connection.send(rows[start:start + len(triples)])
            start += len(triples)
        timing.passes += 1
        timing.decisions += len(tagged)
        timing.unpacking += unpacked - began
        timing.choosing += chose - unpacked
        timing.sending += time.monotonic() - chose
