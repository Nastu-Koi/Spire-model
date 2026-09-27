"""Profile bounded MCTS, separating actual solver searches from plan execution."""

import argparse
import hashlib
import json
import threading
import time
from collections import defaultdict
from functools import wraps
from pathlib import Path
from unittest.mock import patch

from combat_solver_cli.client import DEFAULT_CONFIG, SolverEngine
from combat_solver_cli.evaluate import source_hashes
from combat_solver_cli.mcts import search
from combat_solver_cli.search_support import ReplayWorker, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--lanes", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--budget-ms", type=int, default=1000)
    parser.add_argument("--boss-budget-ms", type=int, default=5000)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    metrics = defaultdict(lambda: {"calls": 0, "seconds": 0.0})
    lock = threading.Lock()

    def measured(original, label):
        @wraps(original)
        def wrapper(*a, **kw):
            started = time.monotonic()
            try:
                result = original(*a, **kw)
                if label == "solver_step" and isinstance(result, dict):
                    key = (
                        "native_search"
                        if result.get("search") is not None
                        else "plan_or_selection"
                    )
                    with lock:
                        metrics[key]["calls"] += 1
                return result
            finally:
                with lock:
                    metrics[label]["calls"] += 1
                    metrics[label]["seconds"] += time.monotonic() - started

        return wrapper

    hashes = source_hashes(args.config)
    hashes["profile_search.py"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    started = time.monotonic()
    with (
        patch.object(
            ReplayWorker, "restore", measured(ReplayWorker.restore, "restore")
        ),
        patch.object(SolverEngine, "step", measured(SolverEngine.step, "solver_step")),
    ):
        result = search(
            args.config,
            "Ironclad",
            args.seed,
            args.output,
            ascension=0,
            max_seconds=args.seconds,
            search_lanes=args.lanes,
            budget_ms=args.budget_ms,
            boss_budget_ms=args.boss_budget_ms,
            reuse_turn_plan=True,
            resume=args.resume,
        )
    report = {
        "algorithm": "mcts",
        "seed": args.seed,
        "status": result["status"],
        "total_seconds": time.monotonic() - started,
        "timings": dict(metrics),
        "source_hashes": hashes,
        "note": "Parallel worker durations overlap; they are not total wall time.",
    }
    write_json(args.output / "profile.json", report)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
