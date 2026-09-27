"""Bounded independent seed jobs; publish only verified final-boss victories."""

import json
import math
import sys
import threading
import time
from pathlib import Path

from model.data import load_runs
from model.protocol import CHARACTERS, fingerprint

from .mcts import search
from .search_support import write_json


def _category(exc):
    from model.protocol import ProtocolError

    from .client import InfrastructureError

    if isinstance(exc, InfrastructureError):
        return "native_error"
    if isinstance(exc, ProtocolError):
        return "protocol_error"
    if isinstance(exc, ValueError) and "verified" in str(exc).lower():
        return "replay_rejected"
    return "other_error"


def generate(
    config,
    jobs,
    output,
    *,
    workers=1,
    progress=True,
    progress_interval=10,
    search_fn=search,
    **options,
):
    options.setdefault("ascension", 0)
    ascension = options["ascension"]
    if type(ascension) is not int or not 0 <= ascension <= 10:
        raise ValueError("ascension must be an integer from 0 to 10")
    if workers != 1:
        raise ValueError("Only one game may be searched at a time")
    if (
        type(progress_interval) not in (int, float)
        or not math.isfinite(progress_interval)
        or progress_interval <= 0
    ):
        raise ValueError("progress_interval must be positive and finite")
    identities = [(j["character"], str(j["seed"])) for j in jobs]
    if not identities or len(set(identities)) != len(identities):
        raise ValueError("Jobs must be nonempty and unique")
    if any(c not in CHARACTERS for c, s in identities):
        raise ValueError("Unsupported character")
    if any(j.get("resume") and j.get("prefix_path") for j in jobs):
        raise ValueError("A job cannot use both resume and prefix_path")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    write_json(
        output / "manifest.json",
        {
            "mode": "mcts_batch_v1",
            "config": str(config),
            "jobs": jobs,
            "workers": workers,
            "options": options,
        },
    )
    results, accepted = [], []
    if progress:
        print(f"[0.0s] started jobs={len(jobs)}", file=sys.stderr, flush=True)
    for index, (job, (character, seed)) in enumerate(zip(jobs, identities), 1):
        directory = output / "jobs" / (character + "-" + fingerprint(seed)[:16])
        job_options = dict(options)
        for key in ("resume", "prefix_path"):
            if job.get(key):
                job_options[key] = Path(job[key])
        job_started = time.monotonic()
        done = threading.Event()

        def report_waiting(done=done, index=index, seed=seed):
            while not done.wait(progress_interval):
                print(
                    f"[{time.monotonic() - started:.1f}s] waiting "
                    f"job={index}/{len(jobs)} seed={seed}",
                    file=sys.stderr,
                    flush=True,
                )

        monitor = None
        if progress:
            monitor = threading.Thread(target=report_waiting, daemon=True)
            monitor.start()
        try:
            result = search_fn(config, character, seed, directory, **job_options)
            if result["status"] == "verified_victory":
                runs = load_runs(directory / "accepted.jsonl")
                if (
                    len(runs) != 1
                    or runs[0]["provenance"].get("verified_outcome")
                    != f"A{ascension}_final_boss_victory"
                ):
                    raise ValueError("Missing verified outcome")
                accepted.extend(runs)
        except Exception as exc:  # noqa: BLE001 - record each failed independent job
            result = {
                "character": character,
                "seed": seed,
                "status": "job_error",
                "error": str(exc),
                "error_category": _category(exc),
            }
        finally:
            done.set()
            if monitor:
                monitor.join(timeout=1)
        if result["status"] == "infrastructure_error":
            result.setdefault("error_category", "native_error")
        elif result["status"] == "replay_failed":
            result.setdefault("error_category", "replay_rejected")
        elif result["status"] in {"lane_error", "job_error"}:
            result.setdefault("error_category", "other_error")
        result["job_wall_seconds"] = time.monotonic() - job_started
        results.append(result)
        write_json(output / "progress.json", results)
        if progress:
            print(
                f"[{time.monotonic() - started:.1f}s] finished "
                f"job={index}/{len(jobs)} seed={seed} status={result['status']} "
                f"duration={result['job_wall_seconds']:.1f}s",
                file=sys.stderr,
                flush=True,
            )
    accepted.sort(key=lambda r: (r["character"], r["seed"]))
    temporary = output / "accepted.jsonl.tmp"
    with temporary.open("x") as stream:
        for run in accepted:
            stream.write(json.dumps(run, allow_nan=False) + "\n")
    temporary.replace(output / "accepted.jsonl")
    elapsed = time.monotonic() - started
    counts = {
        status: sum(r["status"] == status for r in results)
        for status in sorted({r["status"] for r in results})
    }
    counts.update(
        {
            category: sum(r.get("error_category") == category for r in results)
            for category in (
                "native_error",
                "protocol_error",
                "replay_rejected",
                "other_error",
            )
        }
    )
    summary = {
        "ascension": ascension,
        "verified_victories": len(accepted),
        "unresolved_or_defeated": len(jobs) - len(accepted),
        "attempts": len(results),
        "outcome_counts": counts,
        "wall_seconds": elapsed,
        "wall_seconds_per_verified_victory": elapsed / len(accepted)
        if accepted
        else None,
        "unfinished_attempt_seconds": sum(
            r["job_wall_seconds"] for r in results if r["status"] != "verified_victory"
        ),
        "jobs": results,
    }
    write_json(output / "summary.json", summary)
    if progress:
        print(
            f"[{elapsed:.1f}s] stopped wins={len(accepted)}/{len(jobs)}",
            file=sys.stderr,
            flush=True,
        )
    return summary
