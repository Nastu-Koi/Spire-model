from contextlib import nullcontext
from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import queue
import threading
import time

import torch

from .policy import SessionPolicy, choose_batch
from .protocol import CHARACTERS, SCHEMA, ProtocolError, clean_frame, execution_command, fingerprint, segment_key, validate_frame
from .rewards import MilestoneLedger


def precision_context(model, precision):
    return torch.autocast(model.device.type, dtype=torch.bfloat16) if precision == "bf16" else nullcontext()


def numeric_backend(model):
    return {"device": model.device.type, "attention": model.config.backend,
            "torch": str(torch.__version__), "tf32": torch.backends.cuda.matmul.allow_tf32}


class RolloutRunner:
    def __init__(self, model, vocabulary, *, precision="no", version=0, max_steps=10000, timeout=30):
        if max_steps < 1 or timeout <= 0:
            raise ValueError("Rollout limits must be positive")
        self.model, self.vocabulary = model, vocabulary
        self.precision, self.version = precision, version
        self.max_steps, self.timeout = max_steps, timeout

    def run(self, engine, character, seed, *, sample=True, journal=None, _policy=None):
        if character not in CHARACTERS:
            raise ValueError("Unknown character")
        trace = {"schema": SCHEMA, "character": character, "seed": str(seed), "ascension": 10,
                 "policy_version": self.version, "precision": self.precision, "vocabulary_hash": self.vocabulary.digest,
                 "numeric_backend": numeric_backend(self.model),
                 "source": "on_policy" if sample else "evaluation", "sampling": "full_distribution" if sample else "argmax",
                 "status": "unresolved", "macros": [], "automatic_steps": 0, "initial_reward": 0.0}
        ledger = MilestoneLedger()
        policy = _policy or SessionPolicy(self.model, self.vocabulary, version=self.version)
        pending_steps, current_segment, active_macro = [], None, None
        waiting_since = None
        sink = Path(journal).open("w") if journal else nullcontext()
        self.model.eval()
        try:
            with sink as stream, torch.no_grad(), precision_context(self.model, self.precision):
                frame = engine.reset(character, seed)
                trace["run_id"] = frame.get("routing", {}).get("episode_id", fingerprint([character, seed]))
                trace["contract"] = deepcopy(frame.get("contract", {}))
                steps = 0
                while True:
                    validate_frame(frame)
                    if frame["contract"] != trace["contract"]:
                        raise ProtocolError("Engine contract changed during run")
                    reward = ledger.apply(frame.get("events", []))
                    if trace["macros"]:
                        trace["macros"][-1]["reward"] += reward
                    else:
                        trace["initial_reward"] += reward
                    if stream:
                        stream.write(json.dumps({"frame": frame, "reward": reward}, allow_nan=False) + "\n")
                        stream.flush()
                    boundary = frame["boundary"]
                    if boundary == "terminal":
                        victory = bool(frame.get("public", {}).get("outcome", {}).get("victory"))
                        if victory != ledger.victory_paid:
                            raise ProtocolError("Terminal outcome disagrees with milestone events")
                        trace.update(status="complete", victory=victory)
                        break
                    if boundary == "waiting":
                        waiting_since = waiting_since or time.monotonic()
                        if time.monotonic() - waiting_since > self.timeout:
                            raise TimeoutError("unresolved_wait_timeout")
                        time.sleep(.01)
                        frame = engine.send({"cmd": "advance_to_boundary"})
                        continue
                    waiting_since = None
                    if steps >= self.max_steps:
                        trace["error"] = "unresolved_step_limit"
                        break
                    frame = clean_frame(frame)
                    entities = frame["public"]["entities"]
                    act = next((int(x["act"]) for x in entities if "act" in x), 1)
                    frame["public"]["memory"].append(ledger.public(act))
                    key = segment_key(frame)
                    if key != current_segment:
                        current_segment, pending_steps, active_macro = key, [], None
                    branching = len(frame["legal"]["candidates"]) > 1
                    choice = policy.choose(frame, sample=sample)
                    step = {"frame": frame, "candidate_ref": choice.candidate_ref, "forced": not branching}
                    if branching:
                        if active_macro is None:
                            active_macro = {"steps": pending_steps, "old_log_prob": 0.0,
                                            "old_value": float(choice.value), "reward": 0.0,
                                            "phase": frame["public"]["phase"]}
                            trace["macros"].append(active_macro)
                        active_macro["old_log_prob"] += float(choice.log_prob)
                    else:
                        trace["automatic_steps"] += 1
                    if active_macro is None:
                        pending_steps.append(step)
                    else:
                        active_macro["steps"].append(step)
                    steps += 1
                    frame = engine.send(execution_command(frame, choice.candidate_ref))
        except (ProtocolError, TimeoutError, OSError, ValueError) as exc:
            trace["status"] = "unresolved" if isinstance(exc, TimeoutError) else "error"
            trace["error"] = str(exc)
        trace["ledger"] = ledger.state_dict()
        if trace["status"] == "complete":
            future = 0.0
            for macro in reversed(trace["macros"]):
                future += macro["reward"]
                macro["return"] = future
                macro["advantage"] = future - macro["old_value"]
        return trace


def write_run(path, trace):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(trace, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)


class _InferenceQueue:
    """Engine threads wait here; the collecting thread owns all GPU execution."""

    def __init__(self, runner):
        self.runner = runner
        self.requests = queue.Queue()
        self.lock = threading.Lock()
        self.closed = False

    def policy(self):
        session = SessionPolicy(self.runner.model, self.runner.vocabulary, version=self.runner.version)
        owner = self

        class QueuedPolicy:
            def choose(self, frame, *, sample=True):
                result = Future()
                with owner.lock:
                    if owner.closed:
                        raise RuntimeError("Rollout inference queue closed")
                    owner.requests.put((session, frame, sample, result))
                return result.result()

        return QueuedPolicy()

    def process(self, workers):
        try:
            first = self.requests.get(timeout=.01)
        except queue.Empty:
            return
        requests = [first]
        # Briefly coalesce ready decisions, without waiting for every engine.
        deadline = time.monotonic() + .002
        while len(requests) < workers:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                requests.append(self.requests.get(timeout=remaining))
            except queue.Empty:
                break
        try:
            if len({r[2] for r in requests}) != 1:
                raise ValueError("A rollout batch cannot mix sampling and evaluation")
            with torch.no_grad(), precision_context(self.runner.model, self.runner.precision):
                choices = choose_batch([r[0] for r in requests], [r[1] for r in requests], sample=first[2])
            for request, choice in zip(requests, choices):
                request[3].set_result(choice)
        except BaseException as exc:
            for request in requests:
                request[3].set_exception(exc)
            raise

    def close(self):
        with self.lock:
            self.closed = True
            while True:
                try:
                    request = self.requests.get_nowait()
                except queue.Empty:
                    break
                request[3].set_exception(RuntimeError("Rollout inference queue closed"))


def collect_round(engine_factory, runner, seeds, directory, on_run=None, *, workers=1, sample=True):
    """Freeze the assigned queue before running it; any failure blocks the update."""
    counts = {c: len(seeds.get(c, [])) for c in CHARACTERS}
    if type(workers) is not int or workers < 1:
        raise ValueError("Rollout workers must be a positive integer")
    if len(set(counts.values())) != 1 or not all(counts.values()):
        raise ValueError("Allocate equal nonzero run counts to all five characters")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.glob("*.json")):
        raise FileExistsError("Rollout directory already contains runs; choose a new round directory")
    # Save the full assignment before starting any engine, including queues
    # which later fail. Keep metadata outside the trajectory JSON glob.
    assignment = directory / "metadata" / "seeds.json"
    if assignment.exists():
        raise FileExistsError("Rollout seed assignment already exists; choose a new round directory")
    write_run(assignment, seeds)
    jobs = [(character, seed, directory / f"{character}-{i}.json")
            for character in CHARACTERS for i, seed in enumerate(seeds[character])]
    paths = [path for _, _, path in jobs]
    broker = _InferenceQueue(runner) if workers > 1 else None

    def collect(job):
        character, seed, path = job
        with engine_factory() as engine:
            trace = runner.run(engine, character, seed, sample=sample, journal=path.with_suffix(".jsonl"),
                               _policy=broker.policy() if broker else None)
        write_run(path, trace)
        return character

    if broker is None:
        for completed, job in enumerate(jobs, 1):
            character = collect(job)
            if on_run:
                on_run(character, completed, len(jobs))
    else:
        runner.model.eval()
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="engine") as pool:
            pending = {pool.submit(collect, job) for job in jobs}
            completed = 0
            try:
                while pending:
                    broker.process(workers)
                    for future in tuple(pending):
                        if future.done():
                            character = future.result()
                            pending.remove(future)
                            completed += 1
                            if on_run:
                                on_run(character, completed, len(jobs))
            finally:
                broker.close()
                for future in pending:
                    future.cancel()
    failures = [str(p) for p in paths if json.loads(p.read_text())["status"] != "complete"]
    if failures and sample:
        raise ProtocolError(f"Round incomplete; no PPO update is permitted. Inspect: {failures}")
    return paths
