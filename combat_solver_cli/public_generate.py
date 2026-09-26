"""Generate public-policy victories with independent native replay evidence."""

import argparse
import hashlib
import json
import math
import os
import secrets
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from functools import partial
from pathlib import Path

from model.data import validate_run
from model.engine import CliEngine
from model.protocol import (
    SCHEMA,
    ProtocolError,
    action_semantics,
    clean_frame,
    execution_command,
    segment_key,
    validate_frame,
)
from model.rewards import MilestoneLedger

from .astar import ReplayMismatch, resolve, state_key
from .trajectory import native_replay_error


class GenerationLimit(RuntimeError):
    """A bounded attempt ended without a game outcome."""


def _check_deadline(deadline, stop_event=None):
    if stop_event is not None and stop_event.is_set():
        raise GenerationLimit("coordinator_stop")
    if time.monotonic() >= deadline:
        raise GenerationLimit("deadline_exceeded")


def _settle(engine, frame, ledger, contract, deadline, stop_event=None):
    """Account for every visible event exactly once, including waiting frames."""
    while True:
        _check_deadline(deadline, stop_event)
        validate_frame(frame)
        if frame["contract"] != contract:
            raise ReplayMismatch("Engine contract changed during run")
        ledger.apply(frame.get("events", []))
        if frame["boundary"] != "waiting":
            return frame
        frame = engine.send({"cmd": "advance_to_boundary"})


def _victory_error(frame, ledger, engine):
    if frame["boundary"] != "terminal":
        return "Run did not reach a terminal frame"
    if frame["public"]["outcome"]["victory"] is not True:
        return "Run ended without victory"
    if not ledger.victory_paid or ledger.bosses != {1, 2, 3}:
        return "Victory lacks all three native boss milestones"
    native_error = native_replay_error(engine)
    if native_error:
        return "Native engine logged an error: " + native_error
    return None


def _public_choice_frame(clean):
    return {key: clean[key] for key in ("boundary", "public", "legal")}


def play_attempt(
    engine,
    planner,
    seed,
    deadline,
    max_steps,
    *,
    abandon=True,
    stop_event=None,
    progress=None,
    journal=None,
):
    """Act once per real observation; never branch or reset the live engine."""
    frame = engine.reset("Ironclad", seed, 0)
    contract = frame["contract"]
    run_id = frame["routing"]["episode_id"]
    ledger = MilestoneLedger()
    records, diagnostics = [], []
    if progress is not None:
        progress.update(
            records=records,
            diagnostics=diagnostics,
            contract=contract,
            run_id=run_id,
            planning_seconds=0.0,
        )
    while True:
        frame = _settle(engine, frame, ledger, contract, deadline, stop_event)
        if progress is not None:
            progress["last_frame"] = frame
            progress["last_ledger"] = ledger.state_dict()
        if frame["boundary"] == "terminal":
            error = _victory_error(frame, ledger, engine)
            if error is None:
                return {
                    "status": "candidate_victory",
                    "records": records,
                    "terminal_hash": state_key(frame),
                    "contract": contract,
                    "run_id": run_id,
                    "diagnostics": diagnostics,
                    "terminal": frame,
                    "ledger": ledger.state_dict(),
                }
            if (
                frame["public"]["outcome"]["victory"] is False
                and error == "Run ended without victory"
            ):
                if ledger.victory_paid:
                    raise ReplayMismatch(
                        "Terminal defeat disagrees with victory milestone"
                    )
                native_error = native_replay_error(engine)
                if native_error:
                    raise ReplayMismatch(
                        "Native engine logged an error: " + native_error
                    )
                return {
                    "status": "defeat",
                    "steps": len(records),
                    "ledger": ledger.state_dict(),
                    "terminal": frame,
                }
            raise ReplayMismatch(error)
        decision_frame = clean_frame(frame)
        if progress is not None:
            progress["last_decision_frame"] = decision_frame
        if len(records) >= max_steps:
            raise GenerationLimit("step_limit")
        visible = _public_choice_frame(decision_frame)
        planning_started = time.monotonic()
        try:
            reason = planner.abandon_reason(visible) if abandon else None
            if reason:
                return {
                    "status": "abandoned",
                    "reason": reason,
                    "steps": len(records),
                    "frame": frame,
                    "ledger": ledger.state_dict(),
                }
            candidate, detail = planner.choose(visible)
        finally:
            if progress is not None:
                progress["planning_seconds"] += time.monotonic() - planning_started
        matches = [
            item
            for item in frame["legal"]["candidates"]
            if item["candidate_ref"] == candidate.get("candidate_ref")
        ]
        if len(matches) != 1:
            raise ReplayMismatch("Public planner chose a non-legal candidate")
        chosen = matches[0]
        if progress is not None:
            progress["pending_decision"] = {
                "action": action_semantics(chosen),
                "diagnostics": detail,
            }
        _check_deadline(deadline, stop_event)
        if progress is not None:
            progress.pop("pending_decision", None)
        records.append(
            {"before_hash": state_key(frame), "action": action_semantics(chosen)}
        )
        diagnostics.append(detail)
        if journal is not None:
            with journal.open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "frame": decision_frame,
                            "record": records[-1],
                            "diagnostics": detail,
                        },
                        allow_nan=False,
                    )
                    + "\n"
                )
        frame = engine.send(execution_command(frame, chosen["candidate_ref"]))


def verify_attempt(engine, seed, candidate, deadline, stop_event=None):
    """Replay recorded semantics in a fresh engine and build the training sample."""
    frame = engine.reset("Ironclad", seed, 0)
    contract = frame["contract"]
    if contract != candidate["contract"]:
        raise ReplayMismatch("Independent replay contract differs")
    replay_id = frame["routing"]["episode_id"]
    ledger = MilestoneLedger()
    macros, current_key, automatic = [], None, 0
    for record in candidate["records"]:
        frame = _settle(engine, frame, ledger, contract, deadline, stop_event)
        if frame["boundary"] != "decision" or state_key(frame) != record["before_hash"]:
            raise ReplayMismatch("Independent native replay diverged before a decision")
        chosen = resolve(frame, record["action"])
        key = segment_key(frame)
        if key != current_key:
            macros.append({"phase": frame["public"]["phase"], "steps": []})
            current_key = key
        forced = len(frame["legal"]["candidates"]) == 1
        macros[-1]["steps"].append(
            {
                "frame": clean_frame(frame),
                "candidate_ref": chosen["candidate_ref"],
                "forced": forced,
            }
        )
        automatic += forced
        frame = engine.send(execution_command(frame, chosen["candidate_ref"]))
    frame = _settle(engine, frame, ledger, contract, deadline, stop_event)
    if state_key(frame) != candidate["terminal_hash"]:
        raise ReplayMismatch("Independent native replay terminal state diverged")
    error = _victory_error(frame, ledger, engine)
    if error:
        raise ReplayMismatch(error)
    macros = [
        macro for macro in macros if any(not step["forced"] for step in macro["steps"])
    ]
    run = {
        "schema": SCHEMA,
        "source": "demonstration",
        "teacher_visibility": "public",
        "status": "complete",
        "victory": True,
        "ascension": 0,
        "character": "Ironclad",
        "seed": str(seed),
        "run_id": replay_id,
        "contract": contract,
        "macros": macros,
        "automatic_steps": automatic,
        "provenance": {
            "generator": "public_search_v1",
            "verified_by": "fresh_native_seed_replay",
            "bosses": sorted(ledger.bosses),
            "search_diagnostics": candidate["diagnostics"],
        },
    }
    evidence = {
        "seed": str(seed),
        "source_run_id": candidate["run_id"],
        "replay_run_id": replay_id,
        "source_terminal": candidate["terminal"],
        "source_ledger": candidate["ledger"],
        "replay_terminal": frame,
        "replay_ledger": ledger.state_dict(),
        "public_terminal_hash": candidate["terminal_hash"],
    }
    return validate_run(run), evidence


def _stderr_tail(engine, limit=65536):
    """Read a bounded diagnostic tail without moving the worker's stderr offset."""
    stream = getattr(engine, "stderr", None)
    if stream is None:
        return None
    try:
        stream.flush()
        descriptor = stream.fileno()
        size = os.fstat(descriptor).st_size
        start = max(0, size - limit)
        if hasattr(os, "pread"):
            data = os.pread(descriptor, size - start, start)
        else:
            binary = getattr(stream, "buffer", stream)
            position = binary.tell()
            try:
                binary.seek(start)
                data = binary.read(size - start)
            finally:
                binary.seek(position)
        return (
            data.decode("utf-8", errors="replace") if isinstance(data, bytes) else data
        )
    except (OSError, ValueError, AttributeError):
        # The worker can close stderr itself on a transport failure.
        return None


def run_attempt(
    seed,
    deadline,
    max_steps,
    *,
    planner_factory=None,
    engine_factory=CliEngine,
    abandon=True,
    stop_event=None,
    attempt_dir=None,
    planner_config=None,
):
    if planner_factory is None:
        from .public_search import PublicPlanner

        planner_factory = PublicPlanner
    started = time.monotonic()
    progress = {"acting_stderr_tail": None, "verification_stderr_tail": None}
    evidence = None
    outcome = None
    run = None
    verification_seconds = 0.0
    stage = "initialization"
    try:
        _check_deadline(deadline, stop_event)
        planner = planner_factory()
        stage = "acting"
        with engine_factory() as engine:
            try:
                result = play_attempt(
                    engine,
                    planner,
                    seed,
                    deadline,
                    max_steps,
                    abandon=abandon,
                    stop_event=stop_event,
                    progress=progress,
                    journal=attempt_dir / "decisions.jsonl" if attempt_dir else None,
                )
            finally:
                progress["acting_stderr_tail"] = _stderr_tail(engine)
        if result["status"] != "candidate_victory":
            outcome = dict(result, seed=seed)
        else:
            _check_deadline(deadline, stop_event)
            verification_started = time.monotonic()
            try:
                stage = "verification"
                with engine_factory() as verifier:
                    try:
                        run, evidence = verify_attempt(
                            verifier, seed, result, deadline, stop_event
                        )
                    finally:
                        progress["verification_stderr_tail"] = _stderr_tail(verifier)
            finally:
                verification_seconds = time.monotonic() - verification_started
            outcome = {
                "status": "verified_victory",
                "seed": seed,
                "steps": len(result["records"]),
            }
    except GenerationLimit as exc:
        outcome = {"status": "limit", "seed": seed, "reason": str(exc)}
    except Exception as exc:  # noqa: BLE001 - unexpected planner/engine faults are batch errors
        reason = str(exc)
        if (
            reason.startswith("Engine exited (")
            or "Native engine logged an error:" in reason
            or isinstance(exc, (OSError, TimeoutError, EOFError))
        ):
            category = "native_error"
        elif isinstance(exc, ProtocolError):
            category = "protocol_error"
        elif isinstance(exc, ReplayMismatch):
            category = (
                "replay_rejected" if stage == "verification" else "protocol_error"
            )
        else:
            category = "other_error"
        outcome = {
            "status": "error",
            "seed": seed,
            "error_type": type(exc).__name__,
            "error_category": category,
            "error_stage": stage,
            "reason": reason,
        }
    finished = time.monotonic()
    outcome["wall_seconds"] = finished - started
    outcome["budget_overrun_seconds"] = max(0.0, finished - deadline)
    outcome["planning_seconds"] = progress.get("planning_seconds", 0.0)
    outcome["verification_seconds"] = verification_seconds
    if attempt_dir is not None:
        if evidence is not None:
            _atomic_json(attempt_dir / "evidence.json", evidence)
            digest = hashlib.sha256(
                (attempt_dir / "evidence.json").read_bytes()
            ).hexdigest()
            run["provenance"].update(
                evidence_sha256=digest, evidence_path=str(attempt_dir / "evidence.json")
            )
        _atomic_json(
            attempt_dir / "result.json",
            {"outcome": outcome, "prefix": progress, "planner_config": planner_config},
        )
    return outcome, run


def _atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
    temporary.replace(path)


def _atomic_jsonl(path, values):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def _source_hashes(engine_factory):
    planner_path = Path(__file__).with_name("public_search.py")
    result = {}
    for key, path in (
        ("planner_sha256", planner_path),
        ("heuristics_sha256", Path(__file__).with_name("astar.py")),
        ("effects_sha256", Path(__file__).with_name("public_effects.py")),
        ("resources_sha256", Path(__file__).with_name("public_resources.py")),
        ("rewards_sha256", Path(__file__).with_name("public_rewards.py")),
        ("future_sha256", Path(__file__).with_name("public_future.py")),
        ("generator_sha256", Path(__file__)),
        (
            "representation_sha256",
            Path(__file__).resolve().parents[1] / "model/representation.py",
        ),
    ):
        if path.is_file():
            result[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    if engine_factory is CliEngine:
        root = Path(__file__).resolve().parents[1] / "sts2-cli"
        for key, path in (
            (
                "engine_sha256",
                root / "src/Sts2Headless/bin/Debug/net9.0/Sts2Headless.dll",
            ),
            ("game_sha256", root / "lib/sts2.dll"),
        ):
            if path.is_file():
                result[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def generate(
    output,
    *,
    target_trajectories=1,
    workers=None,
    max_attempts=100,
    max_seconds=300,
    max_steps=10000,
    depth=3,
    beam_width=16,
    abandon=True,
    planner_factory=None,
    engine_factory=CliEngine,
    seed_factory=None,
    planner_options=None,
):
    """Run bounded independent seeds; wall time includes defeats and replay checks."""
    if workers is None:
        workers = os.cpu_count() or 1
    if any(
        type(value) is not int or value < 1
        for value in (
            target_trajectories,
            workers,
            max_attempts,
            max_steps,
            depth,
            beam_width,
        )
    ):
        raise ValueError(
            "Targets, workers, attempts and steps must be positive integers"
        )
    if (
        type(max_seconds) not in (int, float)
        or not math.isfinite(max_seconds)
        or max_seconds <= 0
    ):
        raise ValueError("max_seconds must be positive and finite")
    if planner_options is None:
        planner_options = {}
    if not isinstance(planner_options, dict) or set(planner_options) - {
        "sampling_seed",
        "effect_profile",
        "route_resources",
        "dynamic_legality",
        "draw_samples",
        "future_value",
        "deck_rewards",
        "adaptive_budget",
        "decision_node_limit",
        "decision_time_ms",
    }:
        raise ValueError("Unsupported public planner option")
    if planner_factory is None:
        from .public_search import PublicPlanner

        planner_factory = partial(
            PublicPlanner, depth=depth, beam_width=beam_width, **planner_options
        )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "attempts").mkdir()
    (output / "accepted-parts").mkdir()
    started = time.monotonic()
    deadline = started + max_seconds
    new_seed = seed_factory or (lambda: secrets.token_hex(8).upper())
    seen, results, accepted = set(), [], []
    stop_event = threading.Event()
    submitted = 0
    fatal_error = False
    planner_config = {
        "depth": depth,
        "beam_width": beam_width,
        "abandon": abandon,
        **planner_options,
    }
    hashes = _source_hashes(engine_factory)

    def summary():
        elapsed = time.monotonic() - started
        outcome_counts = {
            status: sum(result["status"] == status for result in results)
            for status in ("verified_victory", "defeat", "abandoned", "limit", "error")
        }
        outcome_counts.update(
            {
                category: sum(
                    result.get("error_category") == category for result in results
                )
                for category in (
                    "native_error",
                    "protocol_error",
                    "replay_rejected",
                    "other_error",
                )
            }
        )
        return {
            "mode": "public_search_bootstrap",
            "status": "error"
            if fatal_error
            else ("complete" if len(accepted) >= target_trajectories else "incomplete"),
            "character": "Ironclad",
            "ascension": 0,
            "target_trajectories": target_trajectories,
            "verified_victories": len(accepted),
            "attempts": submitted,
            "workers": workers,
            "max_attempts": max_attempts,
            "max_seconds": max_seconds,
            "planner": planner_config,
            "outcome_counts": outcome_counts,
            "source_hashes": hashes,
            "wall_seconds": elapsed,
            "budget_overrun_seconds": max(0.0, elapsed - max_seconds),
            "unfinished_attempt_seconds": sum(
                result["wall_seconds"]
                for result in results
                if result["status"] in {"limit", "error"}
            ),
            "wall_seconds_per_verified_victory": elapsed / len(accepted)
            if accepted
            else None,
            "results": results,
        }

    def persist():
        _atomic_jsonl(output / "accepted.jsonl", accepted)
        snapshot = summary()
        _atomic_json(output / "summary.json", snapshot)
        return snapshot

    persist()

    def submit(pool, pending):
        nonlocal submitted
        if (
            fatal_error
            or submitted >= max_attempts
            or time.monotonic() >= deadline
            or len(accepted) >= target_trajectories
        ):
            return
        seed = new_seed()
        if seed in seen:
            raise ValueError("Seed factory produced a duplicate seed")
        seen.add(seed)
        index = submitted + 1
        attempt_dir = output / "attempts" / f"{index:06d}"
        attempt_dir.mkdir()
        future = pool.submit(
            run_attempt,
            seed,
            deadline,
            max_steps,
            planner_factory=planner_factory,
            engine_factory=engine_factory,
            abandon=abandon,
            stop_event=stop_event,
            attempt_dir=attempt_dir,
            planner_config=planner_config,
        )
        pending[future] = index
        submitted += 1

    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {}
        for _ in range(min(workers, max_attempts)):
            submit(pool, pending)
        while pending:
            done, _ = wait(
                pending,
                timeout=max(0, deadline - time.monotonic()),
                return_when=FIRST_COMPLETED,
            )
            if not done:
                # In-flight native requests finish at the engine's own I/O timeout.
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                index = pending.pop(future)
                outcome, run = future.result()
                if outcome["status"] == "error":
                    fatal_error = True
                    stop_event.set()
                if run is not None:
                    run["provenance"].update(hashes, planner=planner_config)
                    validate_run(run)
                    _atomic_jsonl(
                        output / "accepted-parts" / f"{index:06d}.jsonl", [run]
                    )
                    accepted.append(run)
                    outcome["accepted_part"] = f"accepted-parts/{index:06d}.jsonl"
                    if len(accepted) >= target_trajectories:
                        stop_event.set()
                outcome["attempt_dir"] = f"attempts/{index:06d}"
                results.append(outcome)
                persist()
            while len(pending) < workers and len(accepted) < target_trajectories:
                before = submitted
                submit(pool, pending)
                if submitted == before:
                    break
    return persist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-trajectories", type=int, default=1)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--max-attempts", type=int, default=100)
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=300,
        help="Whole-batch wall-clock limit in seconds; the five-minute goal is measured per verified victory",
    )
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--beam-width", type=int, default=16)
    parser.add_argument(
        "--abandon", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--route-resources", action="store_true")
    parser.add_argument("--deck-rewards", action="store_true")
    parser.add_argument("--dynamic-legality", action="store_true")
    parser.add_argument("--draw-samples", type=int, default=0)
    parser.add_argument("--future-value", action="store_true")
    parser.add_argument("--adaptive-budget", action="store_true")
    parser.add_argument("--decision-node-limit", type=int, default=512)
    parser.add_argument("--decision-time-ms", type=float, default=50)
    args = vars(parser.parse_args())
    args["planner_options"] = {
        key: args.pop(key)
        for key in (
            "route_resources",
            "deck_rewards",
            "dynamic_legality",
            "draw_samples",
            "future_value",
            "adaptive_budget",
            "decision_node_limit",
            "decision_time_ms",
        )
    }
    summary = generate(**args)
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
    raise SystemExit(0 if summary["status"] == "complete" else 2)


if __name__ == "__main__":
    main()
