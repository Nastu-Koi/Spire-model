"""Isolated search lanes over one seed and a partitioned A* frontier."""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import heapq
import math
from pathlib import Path
import shutil
import threading
import time

from .astar import load_frontier, save_frontier, write_json
from .diversity import partition_routes


def _copy_verified(source, output):
    for name in ('accepted.jsonl', 'winning_prefix.json', 'verified_trace.jsonl'):
        path = source/name
        if path.is_file(): shutil.copy2(path, output/name)


def _merge_checkpoints(outputs, inputs, results, character, seed, weight, ascension,
                       output, base_state):
    frontier, states, recovered = [], [], []
    for index, (directory, input_path, result) in enumerate(zip(outputs, inputs, results)):
        # A raised exception can leave a periodic checkpoint without its active node.
        # Replaying this lane's entire input shard is the safe recovery point.
        path = input_path if result['status'] == 'lane_error' else directory/'frontier.json'
        try:
            queue, state = load_frontier(path, character, seed, weight, ascension,
                                         with_state=True)
        except (OSError, ValueError, KeyError, TypeError, IndexError):
            queue, state = load_frontier(input_path, character, seed, weight, ascension,
                                         with_state=True)
            recovered.append(index)
        else:
            if path == input_path: recovered.append(index)
        frontier.extend(queue); states.append(state)
    # Local tie counters overlap between lanes; assign unique counters before heapifying.
    frontier = [(f, tie, g, prefix, action)
                for tie, (f, _, g, prefix, action) in enumerate(frontier)]
    heapq.heapify(frontier)
    base_failures = int(base_state.get('boss_failures', 0))
    failures = base_failures + sum(max(0, int(s.get('boss_failures', 0))-base_failures)
                                  for s in states)
    thresholds = [s.get('backtrack_before') for s in states
                  if s.get('backtrack_before') is not None]
    route_visits = dict(base_state.get('route_visits', {}))
    for state in states:
        for key, count in state.get('route_visits', {}).items():
            route_visits[key] = route_visits.get(key, 0) + max(0, count-base_state.get('route_visits', {}).get(key, 0))
    counts = dict(base_state.get('failure_counts', {}))
    for state in states:
        for key, count in state.get('failure_counts', {}).items():
            counts[key] = counts.get(key, 0) + max(0, count-base_state.get('failure_counts', {}).get(key, 0))
    state = {'boss_failures': failures, 'failure_counts': counts, 'route_visits': route_visits,
             'backtrack_before': min(thresholds) if thresholds else None}
    save_frontier(output/'frontier.json', frontier, character, seed, weight, ascension,
                  search_state=state)
    return len(frontier), state, recovered


def search_parallel(config, character, seed, output, *, budget_ms=1000, weight=5.,
                    max_expansions=2000, max_seconds=3600, max_steps=10000,
                    resume=None, prefix_path=None, reuse_turn_plan=False,
                    rollout_decisions=0, ascension=10, search_lanes=2,
                    serial_search=None, boss_budget_ms=5000, early_route_diversity=True):
    if search_lanes not in (2, 4):
        raise ValueError('Parallel search supports two or four lanes')
    if resume and prefix_path:
        raise ValueError('Choose either a frontier or a prefix')
    if serial_search is None:
        from .astar import search as serial_search
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()

    # Generate enough outside-combat alternatives to occupy up to four lanes.
    warm_expanded = 0
    if resume:
        frontier, base_state = load_frontier(resume, character, seed, weight, ascension,
                                             with_state=True)
    else:
        warmup = output/'warmup'
        warm = serial_search(config, character, seed, warmup, budget_ms=budget_ms,
                             weight=weight, max_expansions=min(max_expansions, search_lanes-1),
                             max_seconds=max_seconds, max_steps=max_steps,
                             prefix_path=prefix_path, reuse_turn_plan=reuse_turn_plan,
                             rollout_decisions=0, ascension=ascension, search_lanes=1,
                             boss_budget_ms=boss_budget_ms, early_route_diversity=early_route_diversity)
        warm_expanded = int(warm.get('expanded', 0))
        if warm.get('status') in ('verified_victory', 'infrastructure_error', 'replay_failed', 'stopped'):
            if warm.get('status') == 'verified_victory':
                _copy_verified(warmup, output)
            elif (warmup/'frontier.json').is_file():
                shutil.copy2(warmup/'frontier.json', output/'frontier.json')
            warm['search_lanes'] = search_lanes
            write_json(output/'summary.json', warm)
            return warm
        frontier, base_state = load_frontier(warmup/'frontier.json', character, seed,
                                             weight, ascension, with_state=True)
    if not frontier or warm_expanded >= max_expansions:
        status = 'frontier_exhausted' if not frontier else 'budget_exhausted'
        save_frontier(output/'frontier.json', frontier, character, seed, weight,
                      ascension, search_state=base_state)
        summary = dict(status=status, character=character, seed=seed,
                       ascension=ascension, search_lanes=search_lanes,
                       expanded=warm_expanded, frontier=len(frontier))
        write_json(output/'summary.json', summary)
        return summary

    ordered = sorted(frontier, key=lambda node: (node[0], node[1]))
    shards = partition_routes(frontier, search_lanes) if early_route_diversity else [ordered[index::search_lanes] for index in range(search_lanes)]
    inputs = output/'lane-inputs'; inputs.mkdir()
    lane_outputs = [output/f'lane-{index}' for index in range(search_lanes)]
    input_paths = []
    for index, shard in enumerate(shards):
        path = inputs/f'lane-{index}.json'
        save_frontier(path, shard, character, seed, weight, ascension,
                      search_state=base_state)
        input_paths.append(path)

    options = dict(budget_ms=budget_ms, weight=weight,
                   max_expansions=max(1, math.ceil((max_expansions-warm_expanded)/search_lanes)),
                   max_seconds=max(0.001, max_seconds-(time.monotonic()-started)),
                   max_steps=max_steps, reuse_turn_plan=reuse_turn_plan,
                   rollout_decisions=rollout_decisions, ascension=ascension,
                   search_lanes=1, boss_budget_ms=boss_budget_ms, early_route_diversity=early_route_diversity)
    stop_monitor = threading.Event()

    def propagate_stop():
        while not stop_monitor.wait(.25):
            if (output/'STOP').exists():
                for directory in lane_outputs:
                    if directory.is_dir(): (directory/'STOP').touch()
                return

    monitor = threading.Thread(target=propagate_stop, daemon=True)
    monitor.start()
    results = [None] * search_lanes
    try:
        with ThreadPoolExecutor(max_workers=search_lanes,
                                thread_name_prefix='same-seed-lane') as pool:
            futures = {pool.submit(serial_search, config, character, seed, lane_outputs[index],
                                   resume=input_paths[index], **options): index
                       for index in range(search_lanes)}
            pending = set(futures)
            winner = None
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    index = futures[future]
                    try:
                        results[index] = future.result()
                    except Exception as exc:
                        results[index] = dict(status='lane_error', character=character,
                                              seed=seed, error=str(exc))
                    if results[index].get('status') == 'verified_victory' and winner is None:
                        winner = index
                        for other in pending:
                            directory = lane_outputs[futures[other]]
                            if directory.is_dir(): (directory/'STOP').touch()
    finally:
        stop_monitor.set()
        monitor.join(timeout=1)

    expanded = warm_expanded + sum(int(r.get('expanded', 0)) for r in results)
    deaths = sum(int(r.get('deaths', 0)) for r in results)
    unresolved = sum(int(r.get('unresolved_branches', 0)) for r in results)
    winner = next((i for i, r in enumerate(results)
                   if r.get('status') == 'verified_victory'), None)
    recovered = []
    if winner is not None:
        _copy_verified(lane_outputs[winner], output)
        status, frontier_size, state = 'verified_victory', 0, {}
    else:
        frontier_size, state, recovered = _merge_checkpoints(
            lane_outputs, input_paths, results, character, seed, weight, ascension,
            output, base_state)
        statuses = {r.get('status') for r in results}
        if 'stopped' in statuses or (output/'STOP').exists(): status = 'stopped'
        elif recovered or statuses & {'lane_error', 'infrastructure_error'}:
            status = 'infrastructure_error'
        elif 'replay_failed' in statuses: status = 'replay_failed'
        elif frontier_size == 0: status = 'frontier_exhausted'
        else: status = 'budget_exhausted'
    summary = dict(status=status, character=character, seed=seed, ascension=ascension,
                   search_lanes=search_lanes, expanded=expanded, deaths=deaths,
                   unresolved_branches=unresolved, frontier=frontier_size,
                   algorithm=f'{search_lanes}_lane_early_routes_v2' if early_route_diversity else f'{search_lanes}_lane_sharded_weighted_astar_v1',
                   heuristic_weight=weight, budget_ms=budget_ms, boss_budget_ms=boss_budget_ms,
                   early_route_diversity=early_route_diversity,
                   reuse_turn_plan=reuse_turn_plan,
                   rollout_decisions=rollout_decisions,
                   boss_failures=state.get('boss_failures', max(
                       (int(r.get('boss_failures', 0)) for r in results), default=0)),
                   recovered_lanes=recovered, lanes=results)
    write_json(output/'lanes.json', dict(input_sizes=list(map(len, shards)), partition_strategy="early_routes" if early_route_diversity else "score_stripes", lanes=results))
    write_json(output/'summary.json', summary)
    return summary
