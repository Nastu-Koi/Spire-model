"""Advantages of act groups (`main/play/groups.py`), from the groups themselves.

A segment's return R is what it came to, passed back up the chain of waves
(`segments.returns`); a segment without one is left out. With a win's HP bonus
(`win_bonuses`) a won run adds to its 1 the bonus's weight times the share of its
maximum HP it ended with: a tie-break among wins, at most worth a few points of
winning, that parts the groups whose segments all won. A group's segments share a
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
from main.learn.segments import RIDGE, TREE, fit, read_waves, returns, tree
from main.play.fights import fight_rows
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


def wave_advantages(wave_segments, table, ridge=RIDGE, drop_uniform=False, explored_on_tree=False,
                    explored_centered=False):
    """{segment id: [(macro index, value, advantage)]} of one wave's segments that have
    a return, and the statistics of how they were computed. With `drop_uniform` the
    uniform groups' segments are left out, and the baseline is fitted without them;
    the counts of the groups, of their returns and of the uniform ones still take in
    every group, the advantages' and the tree's the groups kept. With
    `explored_on_tree` an explored decision outside fights that the tree does not price
    gets no advantage (0): off the tree its advantage is the segment's return against
    the baseline, mostly the luck of the rest of the segment, which an explored
    decision's unclipped score would push on in full (ticket 31's fifteenth segment).
    The statistics take its advantage before. With `explored_centered` the explored
    decisions outside fights that learn (those the tree prices with `explored_on_tree`)
    have their advantages less the mean of those of their phase in the wave: that an
    action the policy all but never takes does worse than its own on average is what
    the policy knows already, and learnt again it pushes every candidate but the most
    likely down through the shared weights (ticket 33: the sixteenth segment's fixed-set
    entropy fell every round, and rose with the explored decisions' advantages at 0);
    what is left is which proposals did better than the others. The statistics take
    the advantages before; `centered` has each phase's mean taken off."""
    groups = defaultdict(list)
    for segment in wave_segments:
        if segment.run_id in table:
            groups[segment.group].append(segment)
    kept = {name: members for name, members in groups.items() if not (drop_uniform and uniform(members, table))}
    weights, r2, fitted = fit(kept, table, ridge)
    counts, runs = Counter(), {}
    explored, forked = [], []  # the advantages of the explored and of the forked decisions
    explored_at = defaultdict(list)  # the explored decisions' advantages by ascension
    # The explored decisions outside fights (`RolloutRunner.explore_outside`), and how many
    # of them the tree prices: same frame, another segment's other action.
    outside, outside_on_tree = [], 0
    # The same by who proposed them (`explore_prior`), with the tree's alone apart.
    outside_by = defaultdict(lambda: dict(all=[], on_tree=[]))
    fights = 0  # the decisions priced that were made inside a fight
    learning = defaultdict(list)  # the explored decisions outside fights that learn, by phase: (segment, place)
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
                    if not decision.fight:
                        priced = (segment.run_id, decision.index) in prefix
                        outside.append(advantage)
                        outside_on_tree += priced
                        source = decision.explored if isinstance(decision.explored, str) else "rule"
                        outside_by[source]["all"].append(advantage)
                        if priced:
                            outside_by[source]["on_tree"].append(advantage)
                        elif explored_on_tree:
                            advantage = 0.0
                            entries[-1] = (decision.index, value, advantage)
                        if priced or not explored_on_tree:
                            learning[decision.phase].append((segment.run_id, len(entries) - 1))
                if decision.forked:
                    forked.append(advantage)
                fights += decision.fight
            runs[segment.run_id] = entries
    centered = {}
    if explored_centered:
        for phase, places in sorted(learning.items(), key=lambda item: str(item[0])):
            mean = sum(runs[run_id][place][2] for run_id, place in places) / len(places)
            centered[str(phase)] = dict(decisions=len(places), mean=round(mean, 5))
            for run_id, place in places:
                index, value, advantage = runs[run_id][place]
                runs[run_id][place] = (index, value, advantage - mean)
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
        explored_outside=dict(signed(outside), on_tree=outside_on_tree,
                              by_source={source: dict(signed(kept["all"]), on_tree=signed(kept["on_tree"]))
                                         for source, kept in sorted(outside_by.items())},
                              learned="on_tree" if explored_on_tree else "all",
                              **({"centered": centered} if explored_centered else {})),
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


def win_bonuses(directory, waves, weight):
    """{segment id: `weight` × the share of its maximum HP a segment's last fight left
    it} of the segments of a round's `waves` whose last fight they lived through, from
    the waves' fight records (`fights.fight_rows`, the round played with fight
    entries): the HP after the fight, or at its last decision where the run ended with
    it. `returns` adds it to a win alone; a won run's last fight is the one that won it."""
    last = {}
    for wave in waves:
        for row in fight_rows(directory, wave):
            if row["run_id"] not in last or row["fight"] > last[row["run_id"]]["fight"]:
                last[row["run_id"]] = row
    found = {}
    for run_id, row in last.items():
        hp = row["exit_hp"] if row.get("exit_hp") is not None else row.get("last_hp")
        top = row["exit_max_hp"] if row.get("exit_max_hp") is not None else row.get("last_max_hp")
        if not row["died"] and hp is not None and top:
            found[run_id] = weight * max(0, hp) / top
    return found


def advantages(directory, ridge=RIDGE, processes=None, drop_uniform=False, win_hp_bonus=0.0, explored_on_tree=False,
               explored_centered=False):
    """Compute and write the advantages of every wave of a round of act groups, the
    uniform groups left out with `drop_uniform` (`wave_advantages`), a win's return
    raised by `win_hp_bonus` times the share of its maximum HP it ended with
    (`win_bonuses`; the round played with fight entries), the explored decisions outside
    fights off the tree at 0 with `explored_on_tree`, those that learn centered by phase
    with `explored_centered`. Returns {wave: header}; a round
    holding advantages of this kind, ridge, dropping, bonus and explored rule already is
    left as it is."""
    directory = Path(directory)
    summary_ = groups_summary(directory)
    if summary_ is None:
        raise FileNotFoundError(f"{directory} holds no finished round of act groups")
    identity = dict(kind=KIND, ridge=ridge, tree=TREE, boundaries=summary_["boundaries"],
                    policy_version=summary_["policy_version"],
                    written={wave: summary_[wave]["written"] for wave in summary_["waves"]})
    if drop_uniform:
        identity["drop_uniform_groups"] = True
    if win_hp_bonus:
        if not summary_.get("fight_entries"):
            raise ValueError("A win's HP bonus reads the HP from the fights' records: play the round with fight entries")
        identity["win_hp_bonus"] = win_hp_bonus
    if explored_on_tree:
        identity["explored_on_tree"] = True
    if explored_centered:
        identity["explored_centered"] = True
    found = headers(directory)
    if all(found[wave] is not None and found[wave].get("drop_uniform_groups", False) == drop_uniform
           and found[wave].get("win_hp_bonus", 0.0) == win_hp_bonus
           and found[wave].get("explored_on_tree", False) == explored_on_tree
           and found[wave].get("explored_centered", False) == explored_centered
           and all(found[wave].get(key) == value for key, value in identity.items()) for wave in summary_["waves"]):
        return found
    waves = list(read_waves(directory, summary_["waves"], processes).items())
    bonuses = win_bonuses(directory, summary_["waves"], win_hp_bonus) if win_hp_bonus else {}
    table = returns(waves, bonuses)
    written = {}
    for wave, wave_segments in waves:
        runs, stats = wave_advantages(wave_segments, table, ridge, drop_uniform, explored_on_tree, explored_centered)
        if win_hp_bonus:
            won = [s.run_id for s in wave_segments if s.outcome == "win"]
            given = [bonuses[run_id] for run_id in won if run_id in bonuses]
            stats["win_bonus"] = dict(wins=len(won), missing=len(won) - len(given),
                                      mean=round(float(np.mean(given)), 5) if given else None,
                                      low=round(min(given), 5) if given else None,
                                      high=round(max(given), 5) if given else None)
        written[wave] = write(directory / wave, dict(identity, wave=wave, stats=stats), runs)
    return written


def headers(directory):
    """The advantage headers of a round's waves, or None where a wave has none."""
    summary_ = groups_summary(directory)
    if summary_ is None:
        raise FileNotFoundError(f"{directory} holds no finished round of act groups")
    return {wave: header(Path(directory) / wave) for wave in summary_["waves"]}
