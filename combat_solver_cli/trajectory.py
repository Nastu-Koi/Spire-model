"""Independent native replay gate for Bootstrap-only winning demonstrations."""

import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path

from model.data import validate_run
from model.protocol import (
    SCHEMA,
    clean_frame,
    execution_command,
    segment_key,
    validate_frame,
)
from model.rewards import MilestoneLedger

from .astar import ReplayMismatch, resolve, state_key
from .client import SolverEngine, configuration


def native_replay_error(engine):
    """Reject native faults even when the engine still emits a terminal frame."""
    stream = engine.stderr
    stream.flush()
    size = os.fstat(stream.fileno()).st_size
    offset, pending = 0, b""
    while offset < size:
        if hasattr(os, "pread"):
            block = os.pread(stream.fileno(), min(65536, size - offset), offset)
        else:
            # Verification calls this only after the final reply, with the worker idle.
            binary = getattr(stream, "buffer", stream)
            position = binary.tell()
            binary.seek(offset)
            block = binary.read(min(65536, size - offset))
            binary.seek(position)
        if not block:
            break
        offset += len(block)
        lines = (pending + block).split(b"\n")
        pending = lines.pop()
        for line in lines:
            if b"[ERROR]" in line:
                return line[line.index(b"[ERROR]") :].decode("utf-8", errors="replace")[
                    :2000
                ]
    if b"[ERROR]" in pending:
        return pending[pending.index(b"[ERROR]") :].decode("utf-8", errors="replace")[
            :2000
        ]
    return None


def verify_and_export(config, prefix_path, output, timeout=600):
    data = json.loads(Path(prefix_path).read_text())
    ascension = data.get("ascension", 10)
    if type(ascension) is not int or not 0 <= ascension <= 10:
        raise ValueError("ascension must be an integer from 0 to 10")
    pinned = configuration(config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    accepted, raw = output / "accepted.jsonl", output / "verified_trace.jsonl"
    if accepted.exists() or raw.exists():
        raise FileExistsError("Verified outputs already exist")
    raw_temp, accepted_temp = (
        output / "verified_trace.jsonl.tmp",
        output / "accepted.jsonl.tmp",
    )
    deadline = time.monotonic() + timeout
    ledger, actors, macros = MilestoneLedger(), Counter(), []
    try:
        with SolverEngine(config) as engine, raw_temp.open("x") as stream:
            frame = engine.reset(data["character"], data["seed"], ascension)
            contract = frame["contract"]
            run_id = frame["routing"]["episode_id"]

            def settle(frame):
                while True:
                    if time.monotonic() >= deadline:
                        raise ReplayMismatch("Verification deadline exceeded")
                    validate_frame(frame)
                    if frame["contract"] != contract:
                        raise ReplayMismatch("Contract changed during verification")
                    ledger.apply(frame.get("events", []))
                    if frame["boundary"] != "waiting":
                        return frame
                    frame = engine.send({"cmd": "advance_to_boundary"})

            previous = None
            for record in data["records"]:
                frame = settle(frame)
                if (
                    frame["boundary"] != "decision"
                    or state_key(frame) != record["before_hash"]
                ):
                    raise ReplayMismatch("Independent native replay diverged")
                candidate = resolve(frame, record["action"])
                step = {
                    "frame": clean_frame(frame),
                    "candidate_ref": candidate["candidate_ref"],
                    "forced": len(frame["legal"]["candidates"]) == 1,
                }
                key = segment_key(frame)
                if key != previous:
                    macros.append({"phase": frame["public"]["phase"], "steps": []})
                    previous = key
                macros[-1]["steps"].append(step)
                actors[record["actor"]] += 1
                stream.write(
                    json.dumps(
                        dict(kind="decision", actor=record["actor"], **step),
                        allow_nan=False,
                    )
                    + "\n"
                )
                frame = engine.send(
                    execution_command(frame, candidate["candidate_ref"])
                )
            frame = settle(frame)
            if (
                frame["boundary"] != "terminal"
                or frame["public"]["outcome"]["victory"] is not True
                or not ledger.victory_paid
                or ledger.bosses != {1, 2, 3}
            ):
                raise ReplayMismatch("Prefix does not prove native final-boss victory")
            error = native_replay_error(engine)
            if error:
                raise ReplayMismatch("Native replay logged an engine error: " + error)
            stream.write(
                json.dumps(
                    {
                        "kind": "terminal",
                        "frame": frame,
                        "milestones": ledger.state_dict(),
                    },
                    allow_nan=False,
                )
                + "\n"
            )
        automatic = sum(
            len(m["steps"]) for m in macros if all(s["forced"] for s in m["steps"])
        )
        macros = [m for m in macros if any(not s["forced"] for s in m["steps"])]
        # Existing recorder_bc schema deliberately does not claim public teacher visibility.
        run = {
            "schema": SCHEMA,
            "source": "recorder_bc",
            "teacher_visibility": "unverified",
            "status": "partial",
            "victory": None,
            "ascension": ascension,
            "character": data["character"],
            "seed": data["seed"],
            "run_id": run_id,
            "contract": contract,
            "macros": macros,
            "automatic_steps": automatic,
            "provenance": {
                "bc_only": True,
                "importer": "combat-solver-cli-v1",
                "verified_outcome": f"A{ascension}_final_boss_victory",
                "verified_by": "fresh_native_seed_replay",
                "bosses": sorted(ledger.bosses),
                "actors": dict(actors),
                "search": data.get("search", {}),
                "raw_sha256": hashlib.sha256(raw_temp.read_bytes()).hexdigest(),
                "solver_sha256": pinned["solver_dll_sha256"],
                "game_sha256": pinned["game_dll_sha256"],
                "engine_sha256": hashlib.sha256(
                    Path(pinned["worker_dll"])
                    .with_name("Sts2Headless.dll")
                    .read_bytes()
                ).hexdigest(),
                "godot_stubs_sha256": hashlib.sha256(
                    Path(pinned["worker_dll"]).with_name("GodotSharp.dll").read_bytes()
                ).hexdigest(),
                "worker_sha256": hashlib.sha256(
                    Path(pinned["worker_dll"]).read_bytes()
                ).hexdigest(),
            },
        }
        validate_run(run)
        with accepted_temp.open("x") as stream:
            stream.write(json.dumps(run, allow_nan=False) + "\n")
        raw_temp.replace(raw)
        accepted_temp.replace(accepted)
        return run
    except BaseException:
        raw_temp.unlink(missing_ok=True)
        accepted_temp.unlink(missing_ok=True)
        raise
