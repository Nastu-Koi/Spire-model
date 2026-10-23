"""Advantages of act groups (`main/play/groups.py`), from the groups themselves.

A segment's return R is what it came to, passed back up the chain of waves
(`segments.returns`); a segment without one is left out. A group's segments share a
start, so their mean is the baseline each is measured against; it is refined two
ways so that every decision gets an advantage of its own:

- On the shared prefix. The segments of a group are laid along a tree
  (`segments.tree`). At a node reached by at least two segments, a decision's
  advantage is the mean return of the segments that chose its action minus the mean
  return of all that reached the node: the same state, other actions, their real
  consequences. A node with one action teaches nothing (0).
- After the segments part. A decision's advantage is its segment's return minus a
  baseline of the group's mean return and a ridge regression, fitted within groups,
  on the decision's state summary (`features.summary`): where the segment stands
  and what the player has. A few dozen weights, nothing to memorize a run with; the
  corrections sum to zero within a group.

A group whose segments all came to the same return (a uniform group) has nothing to
compare: its tree gives 0 and after the parting only the regression's small noise is
left. With `drop_uniform` such groups are left out, as a segment without a return is:
their decisions get no advantage and do not enter the update, and the baseline is
fitted on the other groups (their targets, all zero, would pull it towards none).

A round played with greedy fights (`RolloutRunner.greedy_fights`) keeps no frames of
its fights: a segment's decisions are those outside them, and so are the tree, the
baseline and the advantages. The fights' macro-actions are counted (`fights.excluded`)
and enter no update.

Written beside each wave's shards as a round's advantages are
(`main/learn/advantage.py`), with the fit and the counts of the tree. The waves are
fitted apart: their returns mean different things.
"""

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from main.learn.advantage import header, write
from main.learn.features import NAMES
from main.learn.segments import RIDGE, fit, read_waves, returns, tree
from main.play.groups import groups_summary

KIND = "act-groups"


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


def uniform(members, table):
    """Whether every segment of a group came to the same return."""
    return len({table[s.run_id] for s in members}) == 1


def wave_advantages(wave_segments, table, ridge=RIDGE, drop_uniform=False):
    """{segment id: [(macro index, value, advantage)]} of one wave's segments that have
    a return, and the statistics of how they were computed. With `drop_uniform` the
    uniform groups' segments are left out, and the baseline is fitted without them;
    the counts of the groups, of their returns and of the uniform ones still take in
    every group, the advantages' and the tree's the groups kept."""
    groups = defaultdict(list)
    for segment in wave_segments:
        if segment.run_id in table:
            groups[segment.group].append(segment)
    kept = {name: members for name, members in groups.items() if not (drop_uniform and uniform(members, table))}
    weights, r2, fitted = fit(kept, table, ridge)
    counts, runs = Counter(), {}
    explored, forked = [], []  # the advantages of the explored and of the forked decisions
    explored_at = defaultdict(list)  # the explored decisions' advantages by ascension
    fights = 0  # the decisions priced that were made inside a fight
    prefix = {}
    for members in kept.values():
        found, node_counts = tree(members, table)
        prefix.update(found)
        counts.update(node_counts)
    for members in kept.values():
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
                    explored_at[segment.ascension].append(advantage)
                if decision.forked:
                    forked.append(advantage)
                fights += decision.fight
            runs[segment.run_id] = entries
    between, mean_return = share([[table[s.run_id] for s in members] for members in groups.values()])
    uniform_groups = sum(uniform(members, table) for members in groups.values())
    decisions = sum(len(entries) for entries in runs.values())
    stats = dict(
        groups=len(groups), segments=len(runs), decisions=decisions,
        return_mean=None if mean_return is None else round(mean_return, 5),
        between_share=None if between is None else round(between, 5), uniform_groups=uniform_groups,
        tree=dict({name: counts[name] for name in ("nodes", "decisions", "branching_nodes", "branching_decisions")},
                  share=round(counts["decisions"] / decisions, 5) if decisions else 0.0,
                  branching_share=round(counts["branching_decisions"] / decisions, 5) if decisions else 0.0),
        baseline=dict(ridge=ridge, r2_within=round(r2, 5), fitted=fitted,
                      weights={name: round(float(w), 5) for name, w in zip(NAMES, weights)}),
        # What exploring taught: the mean advantage of the explored decisions says whether
        # what they tried paid off on average; the signed counts how often.
        explored=signed(explored),
        # What forking taught: its decisions stand on the tree, each half against the other.
        forked=signed(forked),
        # The fights' decisions priced, and those left out: kept without frames, played
        # by argmax with greedy fights (`Segment.frameless`), so that no update learns
        # from them. With greedy fights none is priced.
        fights=dict(decisions=fights, excluded=sum(s.frameless for members in kept.values() for s in members)),
        # The same by ascension: a group is at one (`main.play.groups`).
        by_ascension=by_ascension(groups, table, explored_at))
    if drop_uniform:
        left = [members for name, members in groups.items() if name not in kept]
        stats["dropped"] = dict(groups=len(left), segments=sum(map(len, left)),
                                decisions=sum(len(s.decisions) for members in left for s in members))
    return runs, stats


def by_ascension(groups, table, explored_at):
    """Each ascension's groups, how many are uniform, their mean return and the
    advantages of their explored decisions."""
    levels = defaultdict(list)
    for members in groups.values():
        levels[members[0].ascension].append(members)
    found = {}
    for level, groups_at in sorted(levels.items(), key=lambda item: str(item[0])):
        returns = [table[s.run_id] for members in groups_at for s in members]
        found[str(level)] = dict(
            groups=len(groups_at), segments=len(returns),
            uniform_groups=sum(uniform(members, table) for members in groups_at),
            return_mean=round(sum(returns) / len(returns), 5) if returns else None,
            explored=signed(explored_at.get(level, [])))
    return found


def signed(values):
    """How many advantages there are, their mean and how many are of each sign."""
    return dict(decisions=len(values), advantage_mean=round(float(np.mean(values)), 5) if values else None,
                positive=sum(a > 0 for a in values), negative=sum(a < 0 for a in values))


def advantages(directory, ridge=RIDGE, processes=None, drop_uniform=False):
    """Compute and write the advantages of every wave of a round of act groups, the
    uniform groups left out with `drop_uniform` (`wave_advantages`). Returns {wave:
    header}; a round holding advantages of this kind, ridge and dropping already is
    left as it is."""
    directory = Path(directory)
    summary_ = groups_summary(directory)
    if summary_ is None:
        raise FileNotFoundError(f"{directory} holds no finished round of act groups")
    identity = dict(kind=KIND, ridge=ridge, boundaries=summary_["boundaries"], bosses=summary_.get("bosses", []),
                    policy_version=summary_["policy_version"],
                    written={wave: summary_[wave]["written"] for wave in summary_["waves"]})
    if drop_uniform:
        identity["drop_uniform_groups"] = True
    found = headers(directory)
    if all(found[wave] is not None and found[wave].get("drop_uniform_groups", False) == drop_uniform
           and all(found[wave].get(key) == value for key, value in identity.items()) for wave in summary_["waves"]):
        return found
    waves = list(read_waves(directory, summary_["waves"], processes).items())
    table = returns(waves)
    written = {}
    for wave, wave_segments in waves:
        runs, stats = wave_advantages(wave_segments, table, ridge, drop_uniform)
        written[wave] = write(directory / wave, dict(identity, wave=wave, stats=stats), runs)
    return written


def headers(directory):
    """The advantage headers of a round's waves, or None where a wave has none."""
    summary_ = groups_summary(directory)
    if summary_ is None:
        raise FileNotFoundError(f"{directory} holds no finished round of act groups")
    return {wave: header(Path(directory) / wave) for wave in summary_["waves"]}
