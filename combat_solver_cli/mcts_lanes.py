"""Parallel MCTS trees over disjoint native opening decisions."""

import json
import os
import shutil
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from model.protocol import fingerprint

from .diversity import partition_routes
from .search_support import write_json


def _copy_verified(source, output):
    for name in ("accepted.jsonl", "winning_prefix.json", "verified_trace.jsonl"):
        path = source / name
        if path.is_file():
            shutil.copy2(path, output / name)


def _opening_shards(
    config,
    character,
    seed,
    output,
    prefix_path,
    lanes,
    options,
    started,
    warm_resume=None,
):
    """Expand the native MCTS root once, then preserve every legal child in one lane."""
    from .mcts import Tree, load_tree, save_tree, search, search_identity

    warm_options = dict(options)
    warm_options.update(
        max_expansions=1,
        rollout_decisions=0,
        max_seconds=max(0.001, options["max_seconds"] - (time.monotonic() - started)),
    )
    stop_monitor = threading.Event()

    def propagate_stop():
        while not stop_monitor.wait(0.25):
            warmup = output / "warmup"
            if (output / "STOP").exists() and warmup.is_dir():
                (warmup / "STOP").touch()
                return

    thread = threading.Thread(target=propagate_stop, daemon=True)
    thread.start()
    try:
        warm = search(
            config,
            character,
            seed,
            output / "warmup",
            search_lanes=1,
            prefix_path=prefix_path,
            resume=warm_resume,
            **warm_options,
        )
    finally:
        stop_monitor.set()
        thread.join(timeout=1)
    if warm["status"] == "verified_victory":
        return warm, []
    warm_identity = search_identity(
        character,
        seed,
        options["ascension"],
        options["exploration"],
        options["rollout_epsilon"],
        0,
        options["local_repair"],
        options["risk_aware_rollout"],
    )
    tree, _ = load_tree(
        output / "warmup" / "tree.json", warm_identity, options["exploration"]
    )
    if len(tree.roots) != 1:
        raise ValueError("MCTS opening checkpoint must have one root")
    root = tree.nodes[tree.roots[0]]
    if not root.expanded or not root.children:
        return warm, []
    opening = [
        (-tree.nodes[index].prior, order, 0, root.prefix, tree.nodes[index].pending)
        for order, index in enumerate(root.children)
    ]
    if options["early_route_diversity"]:
        shards = partition_routes(opening, lanes)
    else:
        ordered = sorted(opening, key=lambda item: (item[0], item[1]))
        shards = [ordered[index::lanes] for index in range(lanes)]
    identity = search_identity(
        character,
        seed,
        options["ascension"],
        options["exploration"],
        options["rollout_epsilon"],
        options["rollout_decisions"],
        options["local_repair"],
        options["risk_aware_rollout"],
    )
    inputs = output / "lane-inputs"
    inputs.mkdir()
    paths = []
    for lane, shard in enumerate(shards):
        lane_tree = Tree(
            int(fingerprint([seed, "mcts", lane])[:16], 16), options["exploration"]
        )
        root_index = lane_tree.add(None, root.prefix, None)
        lane_root = lane_tree.nodes[root_index]
        lane_root.expanded = True
        lane_root.value = root.value
        lane_root.decision = root.decision
        for _, order, _, prefix, pending in shard:
            prior = tree.nodes[root.children[order]].prior
            lane_tree.add(root_index, prefix, pending, prior)
        if not lane_root.children:
            lane_root.closed = True
        path = inputs / f"lane-{lane}.json"
        save_tree(path, lane_tree, identity, {})
        paths.append(path.resolve())
    return warm, paths


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
    from .mcts import search, search_identity

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    identity = {
        "search_lanes": search_lanes,
        "early_route_diversity": options["early_route_diversity"],
        "tree": search_identity(
            character,
            seed,
            options["ascension"],
            options["exploration"],
            options["rollout_epsilon"],
            options["rollout_decisions"],
            options["local_repair"],
            options["risk_aware_rollout"],
        ),
    }
    warm, warm_expanded = None, 0
    next_lane = 0
    opening_resume = None
    if resume:
        data = json.loads(Path(resume).read_text())
        if data.get("schema") != "mcts-lanes-v2":
            raise ValueError(
                "Parallel MCTS checkpoint is incompatible; expected mcts-lanes-v2"
            )
        if data.get("identity") != identity:
            raise ValueError("MCTS lane identity or policy differs")
        stage = data.get("stage", "lanes")
        if stage == "opening":
            opening_resume = (
                Path(resume).parent / data["opening_checkpoint"]
            ).resolve()
            if not opening_resume.is_file():
                raise ValueError("MCTS opening checkpoint file is missing")
        elif stage == "lanes":
            paths = [
                (Path(resume).parent / name).resolve() for name in data["checkpoints"]
            ]
            if len(paths) != search_lanes or any(not path.is_file() for path in paths):
                raise ValueError("MCTS lane checkpoint count or files differ")
        else:
            raise ValueError("MCTS lane checkpoint stage is invalid")
        next_lane = data.get("next_lane", 0)
        if type(next_lane) is not int or not 0 <= next_lane < search_lanes:
            raise ValueError("MCTS lane checkpoint cursor is invalid")
    if not resume or opening_resume is not None:
        warm, paths = _opening_shards(
            config,
            character,
            seed,
            output,
            prefix_path,
            search_lanes,
            options,
            started,
            warm_resume=opening_resume,
        )
        warm_expanded = int(warm.get("expanded", 0))
        if not paths:
            if warm["status"] == "verified_victory":
                _copy_verified(output / "warmup", output)
            else:
                write_json(
                    output / "tree.json",
                    {
                        "schema": "mcts-lanes-v2",
                        "identity": identity,
                        "stage": "opening",
                        "next_lane": next_lane,
                        "opening_checkpoint": os.path.relpath(
                            output / "warmup" / "tree.json", output
                        ),
                    },
                )
            warm.update(search_lanes=search_lanes, seconds=time.monotonic() - started)
            write_json(output / "summary.json", warm)
            return warm

    def manifest(checkpoints):
        write_json(
            output / "tree.json",
            {
                "schema": "mcts-lanes-v2",
                "identity": identity,
                "stage": "lanes",
                "next_lane": next_lane,
                "checkpoints": [
                    os.path.relpath(path, output.resolve()) for path in checkpoints
                ],
            },
        )

    if warm and warm["status"] in ("stopped", "infrastructure_error", "replay_failed"):
        manifest(paths)
        warm.update(search_lanes=search_lanes, seconds=time.monotonic() - started)
        write_json(output / "summary.json", warm)
        return warm

    directories = [output / f"lane-{i}" for i in range(search_lanes)]
    remaining_expansions = max(0, options["max_expansions"] - warm_expanded)
    remaining_seconds = options["max_seconds"] - (time.monotonic() - started)
    live = []
    for lane, path in enumerate(paths):
        checkpoint = json.loads(path.read_text())
        if (
            checkpoint.get("schema") != "mcts-tree-v1"
            or checkpoint.get("identity") != identity["tree"]
        ):
            raise ValueError("MCTS lane tree checkpoint is incompatible")
        if any(not checkpoint["nodes"][root]["closed"] for root in checkpoint["roots"]):
            live.append(lane)
    shares = [0] * search_lanes
    for _ in range(remaining_expansions):
        if not live:
            break
        lane = next((index for index in live if index >= next_lane), live[0])
        shares[lane] += 1
        next_lane = (lane + 1) % search_lanes
    manifest(paths)
    if remaining_seconds <= 0 or not any(shares) or (output / "STOP").exists():
        status = (
            "stopped"
            if (output / "STOP").exists()
            else "tree_exhausted"
            if not live
            else "budget_exhausted"
        )
        summary = {
            "status": status,
            "character": character,
            "seed": seed,
            "ascension": options["ascension"],
            "search_lanes": search_lanes,
            "expanded": warm_expanded,
            "recovered_lanes": [],
            "seconds": time.monotonic() - started,
        }
        write_json(output / "summary.json", summary)
        return summary

    stop_monitor = threading.Event()
    halt = threading.Event()

    def monitor():
        while not stop_monitor.wait(0.25):
            if halt.is_set() or (output / "STOP").exists():
                for directory in directories:
                    if directory.is_dir():
                        (directory / "STOP").touch()

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    results = [
        {"status": "budget_exhausted", "expanded": 0} for _ in range(search_lanes)
    ]
    checkpoints = list(paths)
    recovered = []
    first_verified = None
    run_options = dict(options)
    run_options["max_seconds"] = max(0.001, remaining_seconds)
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
                    max_expansions=shares[i],
                    **{
                        key: value
                        for key, value in run_options.items()
                        if key != "max_expansions"
                    },
                ): i
                for i in range(search_lanes)
                if shares[i]
            }
            pending = set(futures)
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    i = futures[future]
                    try:
                        results[i] = future.result()
                    except Exception as exc:  # noqa: BLE001 - isolate failed lanes
                        results[i] = {
                            "status": "lane_error",
                            "error": str(exc),
                            "expanded": 0,
                        }
                    if (
                        results[i]["status"] == "verified_victory"
                        and first_verified is None
                    ):
                        first_verified = time.monotonic() - started
                        halt.set()
    finally:
        stop_monitor.set()
        thread.join(timeout=1)
    for i, result in enumerate(results):
        path = directories[i] / "tree.json"
        if shares[i] and result["status"] != "lane_error" and path.exists():
            checkpoints[i] = path.resolve()
        elif shares[i]:
            recovered.append(i)
    manifest(checkpoints)
    winner = next(
        (
            i
            for i, result in enumerate(results)
            if result["status"] == "verified_victory"
        ),
        None,
    )
    statuses = {result["status"] for result in results}
    if winner is not None:
        _copy_verified(directories[winner], output)
        status = "verified_victory"
    elif recovered or statuses & {"infrastructure_error", "lane_error"}:
        status = "infrastructure_error"
    elif "replay_failed" in statuses:
        status = "replay_failed"
    elif "stopped" in statuses or (output / "STOP").exists():
        status = "stopped"
    elif all(value.startswith("tree_exhausted") for value in statuses):
        status = (
            "tree_exhausted_with_unresolved"
            if any("unresolved" in value for value in statuses)
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
        "expanded": warm_expanded
        + sum(result.get("expanded", 0) for result in results),
        "simulations": sum(result.get("simulations", 0) for result in results),
        "deaths": sum(result.get("deaths", 0) for result in results),
        "unresolved_branches": sum(
            result.get("unresolved_branches", 0) for result in results
        ),
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
