"""The segments of a round of act groups (`main/play/groups.py`) and what they came to.

A segment's return R is ±1 where its run ended (won or lost) and, for a segment that
reached the cut of its wave, the mean return of the group played on from its save
(the Monte-Carlo value of the state it reached, passed back up the chain from the
last wave); a segment that reached the cut but has no finished segment after it
has no return (`returns`). The segments of a group are laid along a tree (`tree`): a
node is a frame every segment that reached it saw alike (same public state and
candidates), and the node branches by the action chosen there; a node reached by at
least two segments has their mean return as its value.

A segment is read with what pricing its decisions needs and nothing else of its
frames (`read_waves`): from its wave's digest (`play.digests`), or from the shards.

The ridge regression on the decisions' state summaries, within groups (`fit`), is the
baseline a segment's later decisions are measured against (`main/learn/groups.py`).
"""

import gc
import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from main.data.shards import RunShards
from main.env.control import policy_macros
from main.learn.features import NAMES, summary
from main.play import digests

RETURNS = {"win": 1.0, "loss": -1.0}
RIDGE = 1.0


@dataclass(slots=True)
class Decision:
    index: int  # the macro-action's place in its segment
    key: str  # the frame as the segments compare it
    action: tuple  # what was chosen, a candidate per step
    features: list
    explored: bool = False  # drawn by exploration rather than from the policy (`RolloutRunner.explore_potions`)
    forked: bool = False  # taken by its rank in the policy's order (`RolloutRunner.run(fork=)`)
    fight: bool = False  # made inside a fight


@dataclass(slots=True)
class Segment:
    run_id: str
    group: str
    character: str
    outcome: str
    decisions: list
    ascension: int | None = None
    # Its macro-actions kept without frames, which have no decision here: the fights of a
    # segment played with greedy fights (`RolloutRunner.greedy_fights`, `digests.frameless`).
    frameless: int = 0


def _shard_segments(task):
    """The segments of one shard (`task` is the shard's part of the wave's data and
    its name), with what the advantages need of each decision and nothing else of
    its frames, as a digest has it (`play.digests`): a run at a time, so that a
    reader holds one decoded run."""
    shards, name = task
    found = []
    for run in shards.runs([name]):
        place = {id(macro): index for index, macro in enumerate(run["macros"])}
        found.append(_digested(dict(run, decisions=[
            digests.decision(place[id(macro)], macro, macro["steps"][0]["frame"]) for macro in policy_macros(run)],
            frameless=digests.frameless(run))))
    return found


def _digested(entry):
    """A segment as its wave's digest has it (`play.digests`)."""
    character = entry["character"]
    decisions = []
    for index, key, action, act, floor, *rest in entry["decisions"]:
        *state, fight, explored, forked = rest
        decisions.append(Decision(index, key, tuple(action),
                                  summary(character, act, floor, dict(zip(digests.STATE, state)), fight),
                                  explored, forked, bool(fight)))
    return Segment(entry["run_id"], entry["group"], character, entry["outcome"], decisions, entry.get("ascension"),
                   entry.get("frameless", 0))


def read_waves(directory, waves, processes=None):
    """{wave: its segments, in the order of its shards} of a round's `waves`. A wave
    whose digest (`play.digests`) holds every segment of its index is read from it.
    The others are read from their shards, which is decoding every frame they hold:
    their shards are read `processes` abreast (half the machine's CPUs by default: its
    cores, beyond which the decoding gains little and every worker holds a decoded
    run), each by a worker handed the index rows of its own shard alone."""
    tasks, found = [], {}
    for wave in waves:
        shards = RunShards(Path(directory) / wave)
        rows = sorted(shards.rows, key=lambda row: (row["shard"], row["line"]))
        path = Path(directory) / wave / digests.FILE
        # A line at a time, kept only as the segment it makes: the later line of a run stands.
        digested = {entry["run_id"]: _digested(entry) for entry in digests.entries(path)} if path.is_file() else {}
        if all(row["run_id"] in digested for row in rows):
            found[wave] = [digested[row["run_id"]] for row in rows]
            continue
        found[wave] = []
        by_shard = defaultdict(list)
        for row in rows:
            by_shard[row["shard"]].append(row)
        tasks += [(wave, RunShards(shards.directories, part), name) for name, part in sorted(by_shard.items())]
    if processes is None:
        processes = max(1, (os.cpu_count() or 1) // 2)
    if min(processes, len(tasks)) <= 1:
        for wave, shards, name in tasks:
            found[wave] += _shard_segments((shards, name))
        return found
    # A collection in a worker would write to every object it inherits from the
    # driver and so copy it: what exists now is set aside first.
    thawed = not gc.get_freeze_count()
    gc.collect()
    gc.freeze()
    try:
        with ProcessPoolExecutor(min(processes, len(tasks)), mp_context=get_context("fork")) as pool:
            read = pool.map(_shard_segments, [(shards, name) for _, shards, name in tasks])
            for (wave, _, _), segments in zip(tasks, read):
                found[wave] += segments
    finally:
        if thawed:
            gc.unfreeze()
    return found


def returns(waves):
    """{segment id: return} over the waves of a round, `waves` being (name, segments)
    pairs in order. A segment that reached its cut takes the mean return of the group
    played from its save in the next wave; one whose group has no finished segment
    has no return and is left out."""
    table, children, following = {}, defaultdict(list), None
    for name, wave_segments in reversed(list(waves)):
        for segment in wave_segments:
            if segment.outcome in RETURNS:
                value = RETURNS[segment.outcome]
            elif segment.outcome == "reached" and following is not None:
                played = children.get(f"{segment.run_id}:{following}")
                if not played:
                    continue
                value = sum(played) / len(played)
            else:
                continue
            table[segment.run_id] = value
            children[segment.group].append(value)
        following = name
    return table


def tree(members, table):
    """The advantages on the shared prefix of one group: {(segment id, macro index):
    (node value, advantage)} and the counts of the tree's nodes."""
    found, counts = {}, Counter()

    def visit(reached, depth):
        alive = [s for s in reached if depth < len(s.decisions)]
        if len(alive) < 2:
            return
        by_frame = defaultdict(list)
        for segment in alive:
            by_frame[segment.decisions[depth].key].append(segment)
        for same in by_frame.values():
            if len(same) < 2:
                continue
            value = sum(table[s.run_id] for s in same) / len(same)
            by_action = defaultdict(list)
            for segment in same:
                by_action[segment.decisions[depth].action].append(segment)
            counts["nodes"] += 1
            counts["decisions"] += len(same)
            if len(by_action) > 1:
                counts["branching_nodes"] += 1
                counts["branching_decisions"] += len(same)
            for chosen in by_action.values():
                quality = sum(table[s.run_id] for s in chosen) / len(chosen)
                for segment in chosen:
                    found[segment.run_id, segment.decisions[depth].index] = (value, quality - value)
                visit(chosen, depth + 1)

    visit(members, 0)
    return found, counts


def within(groups, table):
    """What a ridge regression of a wave is fitted on, within groups: every decision's
    features and its segment's return, less their group's means, as (x, y) arrays."""
    rows, targets = [], []
    for members in groups.values():
        decisions = [(d.features, table[s.run_id]) for s in members for d in s.decisions]
        if not decisions:
            continue
        mean_features = np.mean([f for f, _ in decisions], axis=0)
        mean_return = sum(r for _, r in decisions) / len(decisions)
        for features, value in decisions:
            rows.append(np.asarray(features) - mean_features)
            targets.append(value - mean_return)
    return np.asarray(rows).reshape(len(rows), len(NAMES)), np.asarray(targets, dtype=float)


def fit(groups, table, ridge=RIDGE):
    """The ridge regression of a wave, within groups (`within`): returns (weights,
    within-group R², decisions fitted)."""
    x, y = within(groups, table)
    if not len(y):
        return np.zeros(len(NAMES)), 0.0, 0
    weights = np.linalg.solve(x.T @ x + ridge * np.eye(x.shape[1]), x.T @ y)
    total = float((y ** 2).sum())
    residual = float(((y - x @ weights) ** 2).sum())
    return weights, (1.0 - residual / total) if total > 0 else 0.0, len(y)
