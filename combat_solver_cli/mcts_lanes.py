"""Independent MCTS trees on disjoint same-seed opening shards."""

import json
import math
import os
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from .astar import load_frontier, save_frontier, write_json
from .astar import search as astar_search
from .diversity import partition_routes
from .lanes import _copy_verified


def search_parallel(
    config,
    character,
    seed,
    output,
    *,
    search_lanes=4,
    resume=None,
    prefix_path=None,
    **options,
):
    from .mcts import search

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    warm_expanded = 0
    identity = {
        "character": character,
        "seed": seed,
        "ascension": options["ascension"],
        "search_lanes": search_lanes,
    }
    paths = []
    warm = None
    if resume:
        data = json.loads(Path(resume).read_text())
        if data["schema"] == "mcts-lanes-v1":
            if data["identity"] != identity:
                raise ValueError("MCTS lane identity differs")
            paths = [(Path(resume).parent / p).resolve() for p in data["checkpoints"]]
            if len(paths) != search_lanes:
                raise ValueError("MCTS lane count differs")
            if any(not p.is_file() for p in paths):
                raise FileNotFoundError("MCTS lane checkpoint missing")
        elif data["schema"] != "astar-frontier-v2":
            raise ValueError(
                "Parallel MCTS requires a lane manifest or initial A* frontier"
            )
    if not paths:
        if resume:
            source = resume
        else:
            warm_options = {
                k: v
                for k, v in options.items()
                if k
                not in (
                    "exploration",
                    "rollout_epsilon",
                    "local_repair",
                    "risk_aware_rollout",
                )
            }
            warm_options.update(
                max_expansions=min(options["max_expansions"], search_lanes - 1),
                rollout_decisions=0,
            )
            warm = astar_search(
                config,
                character,
                seed,
                output / "warmup",
                search_lanes=1,
                prefix_path=prefix_path,
                **warm_options,
            )
            warm_expanded = warm["expanded"]
            if warm["status"] == "verified_victory":
                _copy_verified(output / "warmup", output)
                warm.update(
                    algorithm="mcts_opening_warmup",
                    search_lanes=search_lanes,
                    seconds=time.monotonic() - started,
                )
                write_json(output / "summary.json", warm)
                return warm
            source = output / "warmup" / "frontier.json"
        frontier = load_frontier(
            source, character, seed, options["weight"], options["ascension"]
        )
        if options["early_route_diversity"]:
            shards = partition_routes(frontier, search_lanes)
        else:
            ordered = sorted(frontier, key=lambda n: (n[0], n[1]))
            shards = [ordered[i::search_lanes] for i in range(search_lanes)]
        inputs = output / "lane-inputs"
        inputs.mkdir()
        for i, shard in enumerate(shards):
            path = inputs / f"lane-{i}.json"
            save_frontier(
                path, shard, character, seed, options["weight"], options["ascension"]
            )
            paths.append(path.resolve())

    def manifest(checkpoints):
        write_json(
            output / "tree.json",
            {
                "schema": "mcts-lanes-v1",
                "identity": identity,
                "checkpoints": [
                    os.path.relpath(p, output.resolve()) for p in checkpoints
                ],
            },
        )

    manifest(paths)
    if warm and (
        warm["status"] in ("infrastructure_error", "stopped", "replay_failed")
        or warm_expanded >= options["max_expansions"]
    ):
        warm.update(
            algorithm="mcts_opening_warmup",
            search_lanes=search_lanes,
            seconds=time.monotonic() - started,
        )
        write_json(output / "summary.json", warm)
        return warm
    directories = [output / f"lane-{i}" for i in range(search_lanes)]
    run_options = dict(options)
    run_options.update(
        max_expansions=max(
            1, math.ceil((options["max_expansions"] - warm_expanded) / search_lanes)
        ),
        max_seconds=max(0.001, options["max_seconds"] - (time.monotonic() - started)),
    )
    stop_monitor = threading.Event()
    halt = threading.Event()

    def monitor():
        while not stop_monitor.wait(0.25):
            if halt.is_set() or (output / "STOP").exists():
                for d in directories:
                    if d.is_dir():
                        (d / "STOP").touch()

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    results = [None] * search_lanes
    first_verified = None
    try:
        with ThreadPoolExecutor(
            max_workers=search_lanes, thread_name_prefix="mcts-lane"
        ) as pool:
            futures = {
                pool.submit(
                    search,
                    config,
                    character,
                    seed,
                    directories[i],
                    search_lanes=1,
                    resume=paths[i],
                    lane_index=i,
                    **run_options,
                ): i
                for i in range(search_lanes)
            }
            pending = set(futures)
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    i = futures[future]
                    try:
                        results[i] = future.result()
                    except Exception as exc:
                        results[i] = {"status": "lane_error", "error": str(exc)}
                    if (
                        results[i]["status"] == "verified_victory"
                        and first_verified is None
                    ):
                        first_verified = time.monotonic() - started
                        halt.set()
    finally:
        stop_monitor.set()
        thread.join(timeout=1)
    checkpoints = []
    recovered = []
    for i, result in enumerate(results):
        path = directories[i] / "tree.json"
        if result["status"] == "lane_error" or not path.exists():
            path = paths[i]
            recovered.append(i)
        checkpoints.append(path.resolve())
    manifest(checkpoints)
    winner = next(
        (i for i, r in enumerate(results) if r["status"] == "verified_victory"), None
    )
    statuses = {r["status"] for r in results}
    if winner is not None:
        _copy_verified(directories[winner], output)
        status = "verified_victory"
    elif recovered or statuses & {"infrastructure_error", "lane_error"}:
        status = "infrastructure_error"
    elif "replay_failed" in statuses:
        status = "replay_failed"
    elif "stopped" in statuses:
        status = "stopped"
    elif all(s.startswith("tree_exhausted") for s in statuses):
        status = (
            "tree_exhausted_with_unresolved"
            if any("unresolved" in s for s in statuses)
            else "tree_exhausted"
        )
    else:
        status = "budget_exhausted"
    summary = {
        "status": status,
        "character": character,
        "seed": seed,
        "ascension": options["ascension"],
        "algorithm": "mcts_uct_repair_risk_v2"
        if options["local_repair"] or options["risk_aware_rollout"]
        else "mcts_uct_history_v1",
        "local_repair": options["local_repair"],
        "risk_aware_rollout": options["risk_aware_rollout"],
        "search_lanes": search_lanes,
        "expanded": warm_expanded + sum(r.get("expanded", 0) for r in results),
        "simulations": sum(r.get("simulations", 0) for r in results),
        "deaths": sum(r.get("deaths", 0) for r in results),
        "unresolved_branches": sum(r.get("unresolved_branches", 0) for r in results),
        "budget_ms": options["budget_ms"],
        "boss_budget_ms": options["boss_budget_ms"],
        "rollout_decisions": options["rollout_decisions"],
        "rollout_epsilon": options["rollout_epsilon"],
        "exploration": options["exploration"],
        "reuse_turn_plan": options["reuse_turn_plan"],
        "early_route_diversity": options["early_route_diversity"],
        "recovered_lanes": recovered,
        "first_verified_seconds": first_verified,
        "seconds": time.monotonic() - started,
        "lanes": results,
        "optimality_proven": False,
    }
    write_json(output / "summary.json", summary)
    return summary
