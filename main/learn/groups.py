"""Advantages of act groups (`main/play/groups.py`), from the groups themselves.

A segment's return R is ±1 where its run ended (won or lost) and, for a segment that
reached the cut of its wave, the mean return of the group played on from its save
(the Monte-Carlo value of the state it reached, passed back up the chain from the
last wave); a segment that reached the cut but has no finished segment after it
has no return and is left out. A group's segments share a start, so their mean is
the baseline each is measured against; it is refined two ways so that every
decision gets an advantage of its own:

- On the shared prefix. The segments of a group are laid along a tree: a node is a
  frame every segment that reached it saw alike (same public state and candidates),
  and the node branches by the action chosen there. At a node reached by at least two
  segments, a decision's advantage is the mean return of the segments that chose its
  action minus the mean return of all that reached the node: the same state, other
  actions, their real consequences. A node with one action teaches nothing (0).
- After the segments part. A decision's advantage is its segment's return minus a
  baseline of the group's mean return and a ridge regression, fitted within groups,
  on the decision's state summary (`features.summary`): where the segment stands
  and what the player has. A few dozen weights, nothing to memorize a run with; the
  corrections sum to zero within a group.

Written beside each wave's shards as a round's advantages are
(`main/learn/advantage.py`), with the fit and the counts of the tree. The waves are
fitted apart: their returns mean different things.
"""

import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from main.data.shards import RunShards
from main.env.control import policy_macros
from main.env.protocol import fingerprint
from main.learn.advantage import header, write
from main.learn.features import NAMES, summary, where
from main.play.groups import groups_summary

KIND = "act-groups"
RIDGE = 1.0
RETURNS = {"win": 1.0, "loss": -1.0}
# Processes that read a wave's shards: a decision's frame is fingerprinted once.
PROCESSES = 8


@dataclass(slots=True)
class Decision:
    index: int  # the macro-action's place in its segment
    key: str  # the frame as the segments compare it
    action: tuple  # what was chosen, a candidate per step
    features: list
    explored: bool = False  # drawn by exploration rather than from the policy (`RolloutRunner.explore_potions`)


@dataclass(slots=True)
class Segment:
    run_id: str
    group: str
    character: str
    outcome: str
    decisions: list


def frame_key(frame):
    """What makes two frames the same decision: the public state and the candidates."""
    return fingerprint({"public": frame["public"], "legal": frame["legal"]})


def _shard_segments(shards, names):
    found = []
    for run in shards.read(names):
        place = {id(macro): index for index, macro in enumerate(run["macros"])}
        decisions = []
        for macro in policy_macros(run):
            frame = macro["steps"][0]["frame"]
            act, floor, state = where(frame)
            decisions.append(Decision(
                place[id(macro)], frame_key(frame), tuple(step["candidate_ref"] for step in macro["steps"]),
                summary(run["character"], act, floor, state, macro.get("fight") is not None),
                any(step.get("explored") for step in macro["steps"])))
        found.append(Segment(run["run_id"], run["group"], run["character"], run["outcome"], decisions))
    return found


def segments(directory, processes=None):
    """The segments of a wave, with what the advantages need of each decision and
    nothing else of its frames: a shard at a time, `processes` shards abreast."""
    shards = RunShards(directory)
    names = [name for window in shards.window_names(shards=1) for name in window]
    if processes is None:
        processes = min(PROCESSES, os.cpu_count() or 1, len(names))
    if processes <= 1:
        for name in names:
            yield from _shard_segments(shards, [name])
        return
    with ProcessPoolExecutor(processes, mp_context=get_context("fork")) as pool:
        for found in pool.map(_shard_segments, [shards] * len(names), [[name] for name in names]):
            yield from found


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


def fit(groups, table, ridge=RIDGE):
    """The ridge regression of a wave, within groups: returns (weights, within-group
    R², decisions fitted). Each decision's features and its segment's return are taken
    less their group's means."""
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
    if not rows:
        return np.zeros(len(NAMES)), 0.0, 0
    x, y = np.asarray(rows), np.asarray(targets)
    weights = np.linalg.solve(x.T @ x + ridge * np.eye(x.shape[1]), x.T @ y)
    total = float((y ** 2).sum())
    residual = float(((y - x @ weights) ** 2).sum())
    return weights, (1.0 - residual / total) if total > 0 else 0.0, len(rows)


def share(values_by_group):
    """The between-group share of the returns' variance (one-way ANOVA, unbalanced),
    over the groups with at least two returns; and the mean return."""
    groups = [np.asarray(g, float) for g in values_by_group if len(g) >= 2]
    if len(groups) < 2:
        return None, None
    sizes = np.array([len(g) for g in groups])
    total, count = sizes.sum(), len(groups)
    grand = float(np.concatenate(groups).mean())
    within = sum(((g - g.mean()) ** 2).sum() for g in groups) / (total - count)
    between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups) / (count - 1)
    n0 = (total - (sizes ** 2).sum() / total) / (count - 1)
    tau = max((between - within) / n0, 0.0)
    return (tau / (tau + within) if tau + within > 0 else 0.0), grand


def wave_advantages(wave_segments, table, ridge=RIDGE):
    """{segment id: [(macro index, value, advantage)]} of one wave's segments that have
    a return, and the statistics of how they were computed."""
    groups = defaultdict(list)
    for segment in wave_segments:
        if segment.run_id in table:
            groups[segment.group].append(segment)
    weights, r2, fitted = fit(groups, table, ridge)
    counts, runs = Counter(), {}
    explored = []  # the advantages of the explored decisions
    prefix = {}
    for members in groups.values():
        found, node_counts = tree(members, table)
        prefix.update(found)
        counts.update(node_counts)
    for members in groups.values():
        decisions = [(d.features, table[s.run_id]) for s in members for d in s.decisions]
        if not decisions:
            continue
        mean_features = np.mean([f for f, _ in decisions], axis=0)
        mean_return = sum(r for _, r in decisions) / len(decisions)
        for segment in members:
            entries = []
            for decision in segment.decisions:
                if (segment.run_id, decision.index) in prefix:
                    value, advantage = prefix[segment.run_id, decision.index]
                else:
                    value = mean_return + float(np.dot(weights, np.asarray(decision.features) - mean_features))
                    advantage = table[segment.run_id] - value
                entries.append((decision.index, value, advantage))
                if decision.explored:
                    explored.append(advantage)
            runs[segment.run_id] = entries
    between, mean_return = share([[table[s.run_id] for s in members] for members in groups.values()])
    uniform = sum(len({table[s.run_id] for s in members}) == 1 for members in groups.values())
    decisions = sum(len(entries) for entries in runs.values())
    stats = dict(
        groups=len(groups), segments=len(runs), decisions=decisions,
        return_mean=None if mean_return is None else round(mean_return, 5),
        between_share=None if between is None else round(between, 5), uniform_groups=uniform,
        tree=dict({name: counts[name] for name in ("nodes", "decisions", "branching_nodes", "branching_decisions")},
                  share=round(counts["decisions"] / decisions, 5) if decisions else 0.0,
                  branching_share=round(counts["branching_decisions"] / decisions, 5) if decisions else 0.0),
        baseline=dict(ridge=ridge, r2_within=round(r2, 5), fitted=fitted,
                      weights={name: round(float(w), 5) for name, w in zip(NAMES, weights)}),
        # What exploring taught: the mean advantage of the explored decisions says whether
        # what they tried paid off on average; the signed counts how often.
        explored=dict(decisions=len(explored),
                      advantage_mean=round(float(np.mean(explored)), 5) if explored else None,
                      positive=sum(a > 0 for a in explored), negative=sum(a < 0 for a in explored)))
    return runs, stats


def advantages(directory, ridge=RIDGE, processes=None):
    """Compute and write the advantages of every wave of a round of act groups.
    Returns {wave: header}; a round holding advantages of this kind and ridge
    already is left as it is."""
    directory = Path(directory)
    summary_ = groups_summary(directory)
    if summary_ is None:
        raise FileNotFoundError(f"{directory} holds no finished round of act groups")
    identity = dict(kind=KIND, ridge=ridge, boundaries=summary_["boundaries"], bosses=summary_.get("bosses", []),
                    policy_version=summary_["policy_version"],
                    written={wave: summary_[wave]["written"] for wave in summary_["waves"]})
    found = headers(directory)
    if all(found[wave] is not None and all(found[wave].get(key) == value for key, value in identity.items())
           for wave in summary_["waves"]):
        return found
    waves = [(wave, list(segments(directory / wave, processes))) for wave in summary_["waves"]]
    table = returns(waves)
    written = {}
    for wave, wave_segments in waves:
        runs, stats = wave_advantages(wave_segments, table, ridge)
        written[wave] = write(directory / wave, dict(identity, wave=wave, stats=stats), runs)
    return written


def headers(directory):
    """The advantage headers of a round's waves, or None where a wave has none."""
    summary_ = groups_summary(directory)
    if summary_ is None:
        raise FileNotFoundError(f"{directory} holds no finished round of act groups")
    return {wave: header(Path(directory) / wave) for wave in summary_["waves"]}
