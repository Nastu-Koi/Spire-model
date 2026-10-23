"""Fight groups: fights a round of act groups played, played again from their entry saves.

A round of act groups with fight entries (`groups.sample_groups(fight_entries=True)`)
records every fight its segments played and writes the entry save of each one not
begun by an event (`main/play/fights.py`). After its waves some of those fights are
picked (`pick`), and each is played `size` times from its entry save (`jobs`): a fight
group, whose segments stop at the first decision after the fight, or with the run's
end inside it (`RolloutRunner.run(until_fight_end=True)`). A segment's return is its
fight's cost, counted from its record by the learning side (`main/learn/costs.py`);
the group's mean is the baseline each segment is measured against.

The pick is of the fights not begun by an event, `count` of them, shared among
normal, elite and boss fights 3:4:4 (`quotas`). Within a kind a `priority` share
(half by default) is taken by priority, deaths first and then by cost, highest
first; the rest is drawn evenly from the kind's other fights. A kind with fewer fights
than its quota gives what it has and is named short; no other kind makes up for it.
The cost is the caller's (`cost`): what a fight cost is for the learning side to say.
The draws are seeded by `key` (the run's seed root and the round), so that a round
picks the same fights whenever it is played.

The fights are played by the round's policy at its temperatures and drawn, not
greedy (`RolloutRunner.variant(greedy_fights=False)`): every frame is kept, potion
exploration and forking work as they do in act groups, at rates and depths of the
fight groups' own. The wave (`WAVE`) is written beside the round's act waves as one
of them is (`groups.WaveWriter`): shards and index, a digest, each segment's fight
record (`fights.FILE`, without saves), a `summary.json`, and the pick (`PICKS`),
written before anything is played. It is not one of the round's `waves`: the
advantages and the update of act groups do not read it. The round's summary takes
the wave's summary under `fight_groups`.
"""

import json
import random
import time
from pathlib import Path

from main.data.index import INDEX
from main.play.fights import fight_rows
from main.play.groups import WaveWriter, forked, groups_summary, set_aside, write_summary
from main.play.rollout import SEGMENTS

WAVE = "fight-groups"
FORMAT = "fight-groups-v1"
PICKS = "picks.json"
SIZE = 8  # segments from each fight
PRIORITY = 0.5  # the share of a kind's fights taken by priority
SHARES = {"normal": 3, "elite": 4, "boss": 4}


def quotas(count, shares=SHARES):
    """`count` fights shared among the kinds in proportion to `shares`, each its whole
    part and the rest one at a time by the largest remainders (ties in the kinds' order)."""
    total = sum(shares.values())
    quota = {kind: count * share // total for kind, share in shares.items()}
    remainders = sorted(shares, key=lambda kind: -(count * shares[kind] % total))
    for kind in remainders[:count - sum(quota.values())]:
        quota[kind] += 1
    return quota


def candidates(directory, waves):
    """The fights the round's `waves` recorded (`fights.fight_rows`), each with its
    wave, in an order of their own: by wave, segment and fight."""
    rows = []
    for wave in waves:
        rows += sorted((dict(row, wave=wave) for row in fight_rows(directory, wave)),
                       key=lambda row: (row["run_id"], row["fight"]))
    return rows


def pick(rows, count, cost, key, priority=PRIORITY, shares=SHARES):
    """The fights of `rows` (`candidates`) to play again, `count` of them shared among
    the kinds by `shares` (`quotas`), and an account of the pick. The fights begun by
    an event (no entry save, `path` None) are left out. Of a kind's quota a `priority`
    share (rounded half up) is the kind's deaths and then its costliest fights
    (`cost(row)`, in HP), the rest is drawn evenly from its other fights; a kind short
    of its quota gives what it has. Each fight picked is its row with its `cost` and
    `reason` (`priority` or `uniform`)."""
    if type(count) is not int or count < 0:
        raise ValueError("Fight groups: a number of fights, 0 or more")
    if isinstance(priority, bool) or not isinstance(priority, (int, float)) or not 0 <= priority <= 1:
        raise ValueError("The share of the fights picked by priority is in [0, 1]")
    entered = [row for row in rows if row["path"] is not None]
    picked, kinds = [], {}
    for kind, quota in quotas(count, shares).items():
        pool = [row for row in entered if row["kind"] == kind]
        draw = random.Random(f"{key}:{kind}")
        # Fights of equal priority are taken in the order of the draw.
        draw.shuffle(pool)
        ranked = sorted(((row, float(cost(row))) for row in pool), key=lambda item: (not item[0]["died"], -item[1]))
        taken = min(quota, len(pool))
        first = min(int(quota * priority + 0.5), taken)
        picked += [dict(row, cost=price, reason="priority") for row, price in ranked[:first]]
        picked += [dict(row, cost=price, reason="uniform") for row, price in draw.sample(ranked[first:], taken - first)]
        kinds[kind] = dict(fights=len(pool), died=sum(bool(row["died"]) for row in pool), quota=quota,
                           priority=first, uniform=taken - first, short=quota - taken)
    account = dict(fights=len(rows), events=len(rows) - len(entered), picked=len(picked), kinds=kinds,
                   short=[kind for kind, counts in kinds.items() if counts["short"]])
    return picked, account


def group_name(row):
    """The group of a fight picked: its segment and its place there."""
    return f"{row['run_id']}:fight{row['fight']}"


def jobs(picked, size=SIZE, fork_depth=0):
    """The jobs of the fight groups: `size` segments from the entry save of each fight
    picked, each to the fight's end, at the ascension the save was written at, forking
    at its first `fork_depth` decisions with a choice by its place in the group."""
    found = []
    for row in picked:
        group = group_name(row)
        level = {} if row.get("ascension") is None else dict(ascension=row["ascension"])
        for number in range(size):
            found.append((row["character"], dict(seed=row["seed"], run_id=f"{group}-{number}", group=group,
                                                 resume=row["path"], until_fight_end=True,
                                                 **forked(number, size, fork_depth), **level), None))
    return found


def read_picks(directory):
    path = Path(directory) / PICKS
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def wave_summary(directory):
    """The summary of a finished wave of fight groups, or None."""
    path = Path(directory) / "summary.json"
    if not path.is_file():
        return None
    summary = json.loads(path.read_text(encoding="utf-8"))
    return summary if summary.get("kind") == "fight_groups_wave" and summary.get("format") == FORMAT else None


def sample_fight_groups(samplers, runner, directory, count, *, cost, key, size=SIZE, priority=PRIORITY,
                        fork_depth=0, shard_samples=2000, on_run=None):
    """Pick `count` fights of the finished round of act groups in `directory` (`pick`,
    by `cost` and seeded by `key`) and play each `size` times from its entry save, to
    the fight's end, by the sampling processes `samplers` with `runner` (the round's
    policy, its fights drawn), into the round's `WAVE`. Returns the round's summary,
    the wave's under `fight_groups`. A wave cut off on the same pick goes on from the
    segments it finished; one on another pick is set aside and played again."""
    directory = Path(directory)
    summary = groups_summary(directory)
    if summary is None:
        raise FileNotFoundError(f"{directory} holds no finished round of act groups")
    if not summary.get("fight_entries"):
        raise ValueError("Fight groups are picked from the fights a round recorded: play it with fight entries")
    if runner.greedy_fights:
        raise ValueError("The segments of fight groups draw their decisions: not with greedy fights")
    if type(size) is not int or size < 2:
        raise ValueError("A fight group has at least two segments")
    if type(fork_depth) is not int or fork_depth < 0:
        raise ValueError("Fight groups fork at a depth of no decisions or more")
    plan = dict(count=count, size=size, priority=priority, fork_depth=fork_depth, key=key,
                temperature=dict(runner.temperatures), explore_potions=dict(runner.explore_potions),
                policy_digest=runner.digest)
    wave = directory / WAVE
    finished = wave_summary(wave)
    if finished is not None and finished["plan"] == plan:
        summary["fight_groups"] = finished
        write_summary(directory, summary)
        return summary
    stored = read_picks(wave)
    partial = stored is not None and stored["plan"] == plan and (wave / INDEX).is_file()
    if wave.exists() and not partial:
        set_aside(wave)
    if partial:
        picked, account = stored["picks"], stored["account"]
    else:
        picked, account = pick(candidates(directory, summary["waves"]), count, cost, key, priority)
        wave.mkdir(parents=True)
        # Before anything is played: a wave cut off goes on with the same fights.
        stage = wave / (PICKS + ".tmp")
        stage.write_text(json.dumps(dict(plan=plan, account=account, picks=picked), ensure_ascii=False,
                                    allow_nan=False) + "\n", encoding="utf-8")
        stage.replace(wave / PICKS)
    played = jobs(picked, size, fork_depth)
    begun = time.monotonic()
    writer = WaveWriter(wave, shard_samples, resume=partial, fight_entries=True)
    pending = [job for job in played if job[1]["run_id"] not in writer.done]

    def arrived(index, run):
        # A fight segment stops with no save: nothing goes on from it.
        writer.add(run)
        if on_run:
            on_run(WAVE, index, run)

    try:
        if pending:
            samplers.collect(runner, pending, sample=True, on_run=arrived, layout=SEGMENTS)
    finally:
        # A wave cut off keeps readable shards and index, and no summary.
        writer.close_files()
    collector = getattr(samplers, "timing", None) if pending else None
    finished = dict(kind="fight_groups_wave", format=FORMAT, wave=WAVE, plan=plan, picks=account,
                    segments=len(played), **writer.tally(), fights=writer.fights.summary(),
                    seconds=round(time.monotonic() - begun, 1), collector=collector)
    write_summary(wave, finished)
    summary["fight_groups"] = finished
    write_summary(directory, summary)
    return summary
