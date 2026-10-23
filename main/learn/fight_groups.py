"""Advantages of fight groups (`main/play/fight_groups.py`), from the groups themselves.

A fight segment's return R is its fight's, counted from the fight's record
(`main/play/fights.py`) at the prices the caller gives (`costs.fight_return`): the
returns are not written anywhere, so that other prices (a fitted table) price the same
segments again without playing them again. A group's segments start from one entry
save, so their mean is the baseline each is measured against, as an act group's is
(`main/learn/groups.py`), and the shared prefix is refined the same way:

- On the shared prefix. The segments are laid along a tree (`segments.tree`): at a
  node reached by at least two segments, a decision's advantage is the mean return of
  the segments that chose its action minus the mean return of all that reached the
  node. Played from the same save with the same actions a fight comes out the same,
  so the segments share every frame up to their first different action.
- After the segments part, a decision's advantage is its segment's return minus the
  group's mean: no regression on the state, which within one fight hardly differs.

A group whose segments all came to the same return (a uniform group) has nothing to
compare: its advantages are 0. With `drop_uniform` its segments are left out, as an
act group's are: their decisions get no advantage and enter no update.

Written beside the wave's shards as an act wave's advantages are (`main/learn/advantage.py`,
the same file and format, so that an update reads it as it reads an act wave's), with
what they were computed from and the diagnostics of the groups (`diagnostics`): by
kind of fight, how many groups were uniform, how many split on dying, and within the
groups where some segments used a potion and others did not, how the users fared
against the others.
"""

import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import NamedTuple

from main.learn import costs
from main.learn.advantage import write
from main.learn.groups import share, signed, uniform
from main.learn.segments import read_waves, tree
from main.play.fight_groups import WAVE, wave_summary
from main.play.fights import fight_rows
from main.play.groups import groups_summary

KIND = "fight-groups"


class Fought(NamedTuple):
    """What the diagnostics take of a fight segment's record."""

    died: bool
    used: int  # the potions the policy used in the fight
    lost: int | None  # entry HP − exit HP, of a fight survived with a decision after it
    cost: float  # at the prices given


def fought(record, prices):
    died = bool(record["died"])
    lost = None if died or record.get("exit_hp") is None else record["entry_hp"] - record["exit_hp"]
    return Fought(died, len(record.get("potions_used") or ()), lost, costs.cost(record, prices).total)


def records_of(wave_segments, rows):
    """{segment id: its fight's record}: every segment of a fight group has the one
    record of the fight it played."""
    found = {}
    for row in rows:
        if row["run_id"] in found:
            raise ValueError(f"Fight segment {row['run_id']} has more than one fight record")
        found[row["run_id"]] = row
    missing = [segment.run_id for segment in wave_segments if segment.run_id not in found]
    if missing:
        raise ValueError(f"{len(missing)} fight segments have no fight record, {missing[0]} among them")
    return found


def describe(prices):
    """What the advantage header says of the prices: a `costs.Prices`' tables, what
    another price object says of itself (`describe()`), or its type's name."""
    if isinstance(prices, costs.Prices):
        return dict(tiers=list(prices.tiers), final=list(prices.final), free=sorted(prices.free))
    told = getattr(prices, "describe", None)
    return told() if callable(told) else type(prices).__name__


def within_variance(values_by_group):
    """The variance of the returns within groups, pooled over the groups with at least
    two (each group's squared deviations from its mean over its degrees of freedom)."""
    groups = [values for values in values_by_group if len(values) >= 2]
    freedom = sum(len(values) - 1 for values in groups)
    if not freedom:
        return None
    squares = 0.0
    for values in groups:
        mean = statistics.fmean(values)
        squares += sum((value - mean) ** 2 for value in values)
    return squares / freedom


def _mean(values, digits=5):
    return round(statistics.mean(values), digits) if values else None


def diagnostics(groups, facts, table):
    """Each kind of fight's groups (`groups` maps a group to its segments and their
    kind, `facts` a segment to what it came to): how many, of how many segments, how
    many were uniform, the mean return and its variance within groups; the mean
    within groups of the share of segments that died, and how many groups split on
    it; how many groups used a potion somewhere, the potions per segment (the mean
    within groups); and of the groups where some segments used a potion and others
    did not, the mean within them of the users' death rate, HP lost (of the segments
    that survived with a decision after the fight, where both sides have one) and
    cost less the others'."""
    found = {}
    for kind in costs.KINDS:
        cells = [members for fight_kind, members in groups.values() if fight_kind == kind]
        death, death_varies, somewhere, potions = [], 0, 0, []
        varies, d_death, d_lost, d_cost = 0, [], [], []
        for members in cells:
            those = [facts[s.run_id] for s in members]
            died = sum(f.died for f in those)
            death.append(died / len(those))
            death_varies += 0 < died < len(those)
            somewhere += any(f.used for f in those)
            potions.append(statistics.mean(f.used for f in those))
            users = [f for f in those if f.used]
            others = [f for f in those if not f.used]
            if users and others:
                varies += 1
                d_death.append(statistics.mean(f.died for f in users) - statistics.mean(f.died for f in others))
                d_cost.append(statistics.mean(f.cost for f in users) - statistics.mean(f.cost for f in others))
                lost_users = [f.lost for f in users if f.lost is not None]
                lost_others = [f.lost for f in others if f.lost is not None]
                if lost_users and lost_others:
                    d_lost.append(statistics.mean(lost_users) - statistics.mean(lost_others))
        returns = [[table[s.run_id] for s in members] for members in cells]
        spread = within_variance(returns)
        found[kind] = dict(
            groups=len(cells), segments=sum(map(len, cells)),
            uniform_groups=sum(uniform(members, table) for members in cells),
            return_mean=_mean([value for values in returns for value in values]),
            within_variance=None if spread is None else round(spread, 6),
            death_rate=_mean(death), death_varies=death_varies, potion_somewhere=somewhere,
            potions_per_segment=_mean(potions), potion_varies=varies,
            users_minus_others_death=_mean(d_death), users_minus_others_hp_lost=_mean(d_lost),
            n_hp_pairs=len(d_lost), users_minus_others_cost=_mean(d_cost))
    return found


def wave_advantages(wave_segments, rows, prices, drop_uniform=False):
    """{segment id: [(macro index, value, advantage)]} of a wave of fight groups, its
    segments' returns counted from their fights' records `rows` at `prices`, and the
    statistics of how they were computed. With `drop_uniform` the uniform groups'
    segments are left out; the counts of the groups, of their returns and the
    diagnostics by kind still take in every group, the advantages' and the tree's the
    groups kept."""
    records = records_of(wave_segments, rows)
    table = {segment.run_id: costs.fight_return(records[segment.run_id], prices) for segment in wave_segments}
    facts = {segment.run_id: fought(records[segment.run_id], prices) for segment in wave_segments}
    groups = defaultdict(list)
    for segment in wave_segments:
        groups[segment.group].append(segment)
    kinds = {name: (records[members[0].run_id]["kind"], members) for name, members in groups.items()}
    kept = {name: members for name, members in groups.items() if not (drop_uniform and uniform(members, table))}
    counts, runs, explored, forked = Counter(), {}, [], []
    for members in kept.values():
        found, node_counts = tree(members, table)
        counts.update(node_counts)
        returns = [table[s.run_id] for s in members]
        alike = uniform(members, table)
        mean = statistics.fmean(returns)
        for segment in members:
            entries = []
            for decision in segment.decisions:
                if alike:
                    # Nothing to compare: exactly 0, whatever the means' rounding.
                    value, advantage = table[segment.run_id], 0.0
                elif (segment.run_id, decision.index) in found:
                    value, advantage = found[segment.run_id, decision.index]
                else:
                    value, advantage = mean, table[segment.run_id] - mean
                entries.append((decision.index, value, advantage))
                if decision.explored:
                    explored.append(advantage)
                if decision.forked:
                    forked.append(advantage)
            runs[segment.run_id] = entries
    returns = [[table[s.run_id] for s in members] for members in groups.values()]
    between, _ = share(returns)
    spread = within_variance(returns)
    decisions = sum(len(entries) for entries in runs.values())
    stats = dict(
        groups=len(groups), segments=len(runs), decisions=decisions,
        return_mean=_mean([value for values in returns for value in values]),
        between_share=None if between is None else round(between, 5),
        within_variance=None if spread is None else round(spread, 6),
        uniform_groups=sum(uniform(members, table) for members in groups.values()),
        tree=dict({name: counts[name] for name in ("nodes", "decisions", "branching_nodes", "branching_decisions")},
                  share=round(counts["decisions"] / decisions, 5) if decisions else 0.0,
                  branching_share=round(counts["branching_decisions"] / decisions, 5) if decisions else 0.0),
        explored=signed(explored), forked=signed(forked),
        kinds=diagnostics(kinds, facts, table))
    if drop_uniform:
        left = [members for name, members in groups.items() if name not in kept]
        stats["dropped"] = dict(groups=len(left), segments=sum(map(len, left)),
                                decisions=sum(len(s.decisions) for members in left for s in members))
    return runs, stats


def advantages(directory, prices, drop_uniform=False, processes=None):
    """Compute and write the advantages of the fight groups of the round of act groups
    in `directory`, their returns at `prices` (`costs.Prices`, or anything with its
    `price(potion, fight)`), the uniform groups left out with `drop_uniform`
    (`wave_advantages`). Made anew every time, the prices being the caller's: reading
    the wave's digest and records takes seconds. Returns the header."""
    directory = Path(directory)
    summary = groups_summary(directory)
    if summary is None or not summary.get("fight_groups"):
        raise FileNotFoundError(f"{directory} holds no finished round with fight groups")
    wave = directory / WAVE
    if wave_summary(wave) is None:
        raise FileNotFoundError(f"{wave} holds no finished wave of fight groups")
    segments = read_waves(directory, [WAVE], processes)[WAVE]
    runs, stats = wave_advantages(segments, fight_rows(directory, WAVE), prices, drop_uniform)
    identity = dict(kind=KIND, wave=WAVE, prices=describe(prices), policy_version=summary["policy_version"],
                    written=summary["fight_groups"]["written"])
    if drop_uniform:
        identity["drop_uniform_groups"] = True
    return write(wave, dict(identity, stats=stats), runs)
