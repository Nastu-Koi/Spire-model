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
normal, elite and boss fights 3:4:4 by default (`quotas`, `shares`). Within a kind a `priority` share
(half by default) is taken by priority, deaths first and then by cost, highest
first; the rest is drawn evenly from the kind's other fights. A kind with fewer fights
than its quota gives what it has and is named short; no other kind makes up for it.
The cost is the caller's (`cost`): what a fight cost is for the learning side to say.
The draws are seeded by `key` (the run's seed root and the round), so that a round
picks the same fights whenever it is played. A kind's groups may be of a size of its
own (`size` as {kind: segments}, `group_size`).

A pool of entry saves from outside the round (`pool_rows`: fight evaluation sets, such
as human fights rebuilt from their entry states, `main/play/fight_eval.py`) may give a
`pool_share` of the fights: shared among the kinds as the round's are and drawn evenly
within a kind, none by priority (the policy has not played them; they have no cost).
The round's own fights give the rest.

The fights are played by the round's policy at its temperatures and drawn, not
greedy (`RolloutRunner.variant(greedy_fights=False)`): every frame is kept, nothing
is explored or forced. The wave (`WAVE`) is written beside the round's act waves as one
of them is (`groups.WaveWriter`): shards and index, a digest, each segment's fight
record (`fights.FILE`, without saves), a `summary.json`, and the pick (`PICKS`),
written before anything is played. It is not one of the round's `waves`: the
advantages and the update of act groups do not read it. The round's summary takes
the wave's summary under `fight_groups`.

Hindsight forks (`sample_hindsight`) go back over the fight groups' segments: `count`
of them a round (the trunks), shared among the kinds of fight 3:4:4 and drawn evenly
within a kind, whether they died or not. A trunk is replayed from its fight's entry
save by its own actions to a decision of its fight, the same actions bringing the
fight back to the same frame (`RolloutRunner.run(hindsight=)`), and there a hindsight
group of its own parts. The decisions it parts at (`branch_points`) are told by what
the decision shows, not by how the fight went: the first in danger with a potion to
drink (`danger`: one arm for each potion, made to drink it, and `CONTROLS` that draw),
the one from the second round on where the policy's two most likely distinct actions
are closest (`uncertain`: `FORCED` arms taking each of the two), and the first of a
round drawn from the second on (`turn`: `DRAWS` arms that draw). Every arm draws to
the fight's end. The wave (`HINDSIGHT`) is written as the fight groups' is, its forks
(`PICKS`) before anything is played; the round's summary takes its summary under
`fight_hindsight`.
"""

import json
import random
import time
from pathlib import Path

from main.data.index import INDEX
from main.data.shards import RunShards
from main.play.digests import frame_key
from main.play.fights import DANGER, drinks, fight_rows
from main.play.groups import WaveWriter, groups_summary, set_aside, write_summary
from main.play.rollout import SEGMENTS

WAVE = "fight-groups"
FORMAT = "fight-groups-v1"
PICKS = "picks.json"
SIZE = 8  # segments from each fight
PRIORITY = 0.5  # the share of a kind's fights taken by priority
THIN = 2  # a natural pick keeps this many times a kind's quota before its priority (`pick`)
SHARES = {"normal": 3, "elite": 4, "boss": 4}
HINDSIGHT = "fight-hindsight"
HINDSIGHT_FORMAT = "fight-hindsight-v2"
PLACES = ("danger", "uncertain", "turn")  # the kinds of decision a trunk parts at (`branch_points`)
CONTROLS = 2  # arms of a danger point that draw beside those that drink
FORCED = 2  # arms of an uncertain point taking each of its two actions
DRAWS = 4  # arms of a turn's first decision, all of them drawing
# Keys of an entity that say where it stands, not what it is (`same_action`).
PLACE_KEYS = frozenset({"ref", "position", "index", "slot", "order", "semantic_program"})


def quotas(count, shares=SHARES):
    """`count` fights shared among the kinds in proportion to `shares`, each its whole
    part and the rest one at a time by the largest remainders (ties in the kinds' order)."""
    total = sum(shares.values())
    # Whole numbers of fights whatever the shares are: the command line gives them as floats.
    quota = {kind: int(count * share // total) for kind, share in shares.items()}
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


def natural_weights(rows):
    """Each row's weight: one over the segments its run (character and seed) has in its
    wave, as many as its fights there name."""
    segments = {}
    for row in rows:
        segments.setdefault((row["wave"], row["character"], row["seed"]), set()).add(row["run_id"])
    return [1.0 / len(segments[(row["wave"], row["character"], row["seed"])]) for row in rows]


def _drawn(pairs, rng):
    """`(row, weight)` pairs in the order of a draw without replacement at those weights
    (each the larger of its key u ** (1 / weight) first)."""
    return [row for _, row in sorted(((rng.random() ** (1.0 / weight), row) for row, weight in pairs),
                                     key=lambda item: -item[0])]


def pick(rows, count, cost, key, priority=PRIORITY, shares=SHARES, natural=False):
    """The fights of `rows` (`candidates`) to play again, `count` of them shared among
    the kinds by `shares` (`quotas`), and an account of the pick. The fights begun by
    an event (no entry save, `path` None) are left out. Of a kind's quota a `priority`
    share (rounded half up) is the kind's deaths and then its costliest fights
    (`cost(row)`, in HP), the rest is drawn evenly from its other fights; a kind short
    of its quota gives what it has. Each fight picked is its row with its `cost` and
    `reason` (`priority` or `uniform`).

    With `natural` a kind's fights are first thinned to `THIN` times its quota by a
    draw at their natural weights (`natural_weights`: one over the segments their run
    has in their wave), so that every run counts once in each act; the priority and
    the even draw are then of the fights kept. Without it the act groups' fan-out
    makes the last act's fights outnumber the first's several times over."""
    if type(count) is not int or count < 0:
        raise ValueError("Fight groups: a number of fights, 0 or more")
    if isinstance(priority, bool) or not isinstance(priority, (int, float)) or not 0 <= priority <= 1:
        raise ValueError("The share of the fights picked by priority is in [0, 1]")
    entered = [row for row in rows if row["path"] is not None]
    picked, kinds = [], {}
    for kind, quota in quotas(count, shares).items():
        pool = [row for row in entered if row["kind"] == kind]
        kind_fights = len(pool)
        draw = random.Random(f"{key}:{kind}")
        if natural:
            pool = _drawn(list(zip(pool, natural_weights(pool))), draw)[:THIN * quota]
        # Fights of equal priority are taken in the order of the draw.
        draw.shuffle(pool)
        ranked = sorted(((row, float(cost(row))) for row in pool), key=lambda item: (not item[0]["died"], -item[1]))
        taken = min(quota, len(pool))
        first = min(int(quota * priority + 0.5), taken)
        picked += [dict(row, cost=price, reason="priority") for row, price in ranked[:first]]
        picked += [dict(row, cost=price, reason="uniform") for row, price in draw.sample(ranked[first:], taken - first)]
        kinds[kind] = dict(fights=kind_fights, died=sum(bool(row["died"]) for row in pool), quota=quota,
                           priority=first, uniform=taken - first, short=quota - taken)
    account = dict(fights=len(rows), events=len(rows) - len(entered), picked=len(picked), kinds=kinds,
                   short=[kind for kind, counts in kinds.items() if counts["short"]])
    if natural:
        account["natural"] = dict(thin=THIN)
    return picked, account


def group_name(row):
    """The group of a fight picked: its segment and its place there."""
    return f"{row['run_id']}:fight{row['fight']}"


def group_size(size, kind):
    """The segments of a fight group of `kind`: `size`, or its kind's of {kind: segments}
    (the kinds it does not name take `SIZE`)."""
    return size.get(kind, SIZE) if isinstance(size, dict) else size


def jobs(picked, size=SIZE):
    """The jobs of the fight groups: `size` segments (`group_size`) from the entry save
    of each fight picked, each to the fight's end, at the ascension the save was written
    at."""
    found = []
    for row in picked:
        group = group_name(row)
        level = {} if row.get("ascension") is None else dict(ascension=row["ascension"])
        segments = group_size(size, row["kind"])
        for number in range(segments):
            found.append((row["character"], dict(seed=row["seed"], run_id=f"{group}-{number}", group=group,
                                                 resume=row["path"], until_fight_end=True, **level), None))
    return found


def pool_rows(directories):
    """The fights of fight evaluation sets (`fight_eval`: `fights.json` and its saves)
    as rows a pick takes (`pick_pool`): each its set's fight id, character, seed,
    ascension, kind and the absolute path of its entry save."""
    rows = []
    for directory in map(Path, directories):
        document = json.loads((directory / "fights.json").read_text(encoding="utf-8"))
        for fight in document["fights"]:
            rows.append(dict(run_id=f"{directory.name}:{fight['id']}", fight=0, wave="pool",
                             character=fight["character"], seed=fight["seed"], ascension=fight.get("ascension"),
                             kind=fight["kind"], path=str((directory / fight["path"]).resolve()), died=False))
    return rows


def pick_pool(rows, count, key, shares=SHARES):
    """`count` fights of a pool (`pool_rows`) shared among the kinds by `shares`, drawn
    evenly within a kind by `key`, and an account of the pick."""
    picked, kinds = [], {}
    for kind, quota in quotas(count, shares).items():
        pool = sorted((row for row in rows if row["kind"] == kind), key=lambda row: row["run_id"])
        taken = min(quota, len(pool))
        picked += [dict(row, cost=None, reason="pool") for row in random.Random(f"{key}:pool:{kind}").sample(pool, taken)]
        kinds[kind] = dict(fights=len(pool), quota=quota, picked=taken, short=quota - taken)
    return picked, dict(fights=len(rows), picked=len(picked), kinds=kinds,
                        short=[kind for kind, counts in kinds.items() if counts["short"]])


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
                        shares=SHARES, pool=None, pool_share=0.0, natural=False, shard_samples=2000, on_run=None):
    """Pick `count` fights of the finished round of act groups in `directory` (`pick`,
    by `cost` and seeded by `key`) and play each `size` times from its entry save, to
    the fight's end, by the sampling processes `samplers` with `runner` (the round's
    policy, its fights drawn), into the round's `WAVE`; with `pool` (`pool_rows`) a
    `pool_share` of the fights from it (`pick_pool`); with `natural` the round's own are
    picked at their natural weights (`pick(natural=)`). Returns the round's summary,
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
    if any(type(n) is not int or n < 2 for n in (size.values() if isinstance(size, dict) else [size])):
        raise ValueError("A fight group has at least two segments")
    if not 0 <= pool_share <= 1 or (pool_share and not pool):
        raise ValueError("A pool's share of the fight groups is in [0, 1], and needs a pool")
    plan = dict(count=count, size=size, priority=priority, key=key, temperature=dict(runner.temperatures),
                policy_digest=runner.digest)
    # Named only when not the default, so that a round played before goes on as it was.
    if shares != SHARES:
        plan["shares"] = dict(shares)
    if pool_share:
        plan["pool"] = dict(share=pool_share, fights=sorted(row["run_id"] for row in pool))
    if natural:
        plan["natural"] = dict(thin=THIN)
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
        from_pool = round(count * pool_share)
        picked, account = pick(candidates(directory, summary["waves"]), count - from_pool, cost, key, priority, shares,
                               natural)
        if from_pool:
            pooled, account["pool"] = pick_pool(pool, from_pool, key, shares)
            picked += pooled
        wave.mkdir(parents=True)
        # Before anything is played: a wave cut off goes on with the same fights.
        stage = wave / (PICKS + ".tmp")
        stage.write_text(json.dumps(dict(plan=plan, account=account, picks=picked), ensure_ascii=False,
                                    allow_nan=False) + "\n", encoding="utf-8")
        stage.replace(wave / PICKS)
    played = jobs(picked, size)
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


# ---- Hindsight forks ----

def drawn_steps(run):
    """(candidate ref, frame) of every decision with a choice a segment took, in order."""
    return [(step["candidate_ref"], step["frame"]) for macro in run["macros"] for step in macro.get("steps", ())
            if not step["forced"]]


def combat_round(frame):
    """The round of the fight a frame stands in, as its player says; None without one."""
    player = next((e for e in frame["public"]["entities"] if e.get("entity_type") == "player"), None)
    value = None if player is None else player.get("round")
    return value if type(value) is int else None


def drawn_decisions(run):
    """The steps of the decisions with a choice a segment took, in order."""
    return [step for macro in run["macros"] for step in macro.get("steps", ()) if not step["forced"]]


def hp_ratio(frame):
    """The player's HP over their max HP in a frame, None without both."""
    player = next((e for e in frame["public"]["entities"] if e.get("entity_type") == "player"), None)
    if player is None or not player.get("max_hp") or player.get("hp") is None:
        return None
    return player["hp"] / player["max_hp"]


def same_action(frame, first, second):
    """Whether two candidates of a frame do the same: one verb, the same targets and
    sources that differ only in where they stand (two copies of one card in hand)."""
    if first["verb"] != second["verb"] or first.get("target_refs") != second.get("target_refs"):
        return False
    one, other = first.get("source_refs") or [], second.get("source_refs") or []
    if one == other:
        return True
    if len(one) != len(other):
        return False
    entities = {e["ref"]: e for e in frame["public"]["entities"] if e and e.get("ref")}

    def what(ref):
        return {k: v for k, v in entities[ref].items() if k not in PLACE_KEYS} if ref in entities else None

    return all(what(a) is not None and what(a) == what(b) for a, b in zip(one, other))


def distinct_pair(frame, log_probs):
    """The policy's two most likely distinct actions at a decision, from the
    distribution it was drawn from (`log_probs`, in the frame's order): (the refs of
    the first and the second, the gap of their log-probabilities), or None when every
    candidate does the same."""
    candidates = frame["legal"]["candidates"]
    order = sorted(range(len(candidates)), key=lambda i: (-log_probs[i], i))
    first = order[0]
    second = next((i for i in order[1:] if not same_action(frame, candidates[first], candidates[i])), None)
    if second is None:
        return None
    return candidates[first]["candidate_ref"], candidates[second]["candidate_ref"], log_probs[first] - log_probs[second]


def branch_points(run, key):
    """Where a trunk parts, each once: the first decision in danger (HP under `DANGER`
    of the max) where a potion can be drunk; from the second round on, the decision
    whose two most likely distinct actions are closest (`distinct_pair`, from the
    distributions the segment kept); and the first decision of a round from the second
    on, drawn by `key` and the segment. None of them looks at how the fight went. Each
    is (`place`, the round, the decision's index among those with a choice, its frame's
    key, its arms: {"potion": …}, {"force": …} or None for an arm that draws)."""
    steps = drawn_decisions(run)
    rounds = [combat_round(step["frame"]) for step in steps]
    points, taken = [], set()

    def add(place, at, arms):
        taken.add(at)
        points.append(dict(place=place, round=rounds[at], at=at, key=frame_key(steps[at]["frame"]), arms=arms))

    for at, step in enumerate(steps):
        ratio = hp_ratio(step["frame"])
        potions = sorted(potion for potion in drinks(step["frame"]) if potion is not None)
        if ratio is not None and ratio < DANGER and potions:
            add("danger", at, [*({"potion": potion} for potion in potions), *[None] * CONTROLS])
            break
    closest = None
    for at, step in enumerate(steps):
        if at in taken or rounds[at] is None or rounds[at] < 2 or step.get("log_probs") is None:
            continue
        pair = distinct_pair(step["frame"], step["log_probs"])
        if pair is not None and (closest is None or pair[2] < closest[1][2]):
            closest = (at, pair)
    if closest is not None:
        at, (first, second, _) = closest
        add("uncertain", at, [*[{"force": first}] * FORCED, *[{"force": second}] * FORCED])
    starts = {}
    for at, value in enumerate(rounds):
        if value is not None and value >= 2 and value not in starts:
            starts[value] = at
    starts = {value: at for value, at in starts.items() if at not in taken}
    if starts:
        value = random.Random(f"{key}:{run['run_id']}:turn").choice(sorted(starts))
        add("turn", starts[value], [None] * DRAWS)
    return sorted(points, key=lambda point: point["at"])


def segments_of(directory, chosen, take):
    """{run id: `take(run)`} of the fight groups' segments named in `chosen` of the round
    in `directory`, read from the wave's shards a shard at a time and a run at a time:
    a run built whole is megabytes of frames, and a round's trunks are thousands, so
    only what `take` keeps of each stays."""
    shards = RunShards(Path(directory) / WAVE)
    rows = [row for row in shards.rows if row["run_id"] in chosen]
    found = {}
    for name in sorted({row["shard"] for row in rows}):
        for run in RunShards(shards.directories, [row for row in rows if row["shard"] == name]).runs([name]):
            found[run["run_id"]] = take(run)
    return found


def pick_hindsight(directory, count, key, shares=SHARES):
    """The hindsight forks of the round in `directory` and an account of the pick: of
    the fight groups' segments `count` trunks, shared among the kinds by `shares` (`quotas`)
    and drawn evenly within a kind by `key`, died or not, each forked at its
    `branch_points`. Each fork is its group, the segment it comes from, where it stands
    (the entry save of the segment's fight, its character, seed and ascension, the
    fight's kind, the fork's round and place), the actions it replays, its frame's key
    and its arms."""
    if type(count) is not int or count < 0:
        raise ValueError("Hindsight forks: a number of segments, 0 or more")
    picks = read_picks(Path(directory) / WAVE)
    if picks is None:
        raise FileNotFoundError(f"{directory} holds no pick of fight groups")
    saves = {group_name(row): row["path"] for row in picks["picks"]}
    records = {row["run_id"]: row for row in fight_rows(directory, WAVE)}
    draw = random.Random(f"{key}:hindsight")
    chosen, short = [], []
    for kind, quota in quotas(count, shares).items():
        pool = sorted(run_id for run_id, row in records.items() if row["kind"] == kind)
        draw.shuffle(pool)
        chosen += pool[:quota]
        if len(pool) < quota:
            short.append(kind)

    def forks_of(run):
        record, steps = records[run["run_id"]], drawn_steps(run)
        return [dict(group=f"{run['run_id']}:{point['place']}", source=run["run_id"], character=run["character"],
                     seed=run["seed"], ascension=run.get("ascension"), path=saves[run["group"]], kind=record["kind"],
                     round=point["round"], place=point["place"], replay=[ref for ref, _ in steps[:point["at"]]],
                     key=point["key"], arms=point["arms"])
                for point in branch_points(run, key)]

    found = segments_of(directory, set(chosen), forks_of)
    forks = [fork for run_id in chosen for fork in found[run_id]]
    places = {place: sum(fork["place"] == place for fork in forks) for place in PLACES}
    account = dict(trunks=len(chosen), short=short, died=sum(records[run_id]["died"] for run_id in chosen),
                   forked=len({fork["source"] for fork in forks}), forks=len(forks), places=places,
                   kinds={kind: sum(fork["kind"] == kind for fork in forks) for kind in SHARES},
                   segments=sum(len(fork["arms"]) for fork in forks))
    return forks, account


def arm_name(fork, number):
    """The run id of a hindsight group's `number`-th segment: `p` and the potion's place
    among the fork's potions, `f` and the forced arm's among those forced, or `c` and
    the drawing arm's among those that draw."""
    arm = fork["arms"][number]
    kind = "c" if arm is None else "p" if "potion" in arm else "f"
    before = sum(("c" if other is None else "p" if "potion" in other else "f") == kind
                 for other in fork["arms"][:number])
    return f"{fork['group']}-{kind}{before}"


def hindsight_jobs(forks):
    """The jobs of the hindsight groups: each arm of each fork, from the entry save of
    its segment's fight to the fight's end, replaying the segment to the fork."""
    found = []
    for fork in forks:
        level = {} if fork.get("ascension") is None else dict(ascension=fork["ascension"])
        for number, arm in enumerate(fork["arms"]):
            arm = arm or {}
            found.append((fork["character"], dict(
                seed=fork["seed"], run_id=arm_name(fork, number), group=fork["group"], resume=fork["path"],
                until_fight_end=True, hindsight=dict(replay=fork["replay"], key=fork["key"], potion=arm.get("potion"),
                                                     force=arm.get("force")),
                **level), None))
    return found


def arms(forks):
    """{run id: (its group's fork, its arm)} of the hindsight groups' segments; an arm is
    {"potion": …}, {"force": …} or None for one that draws."""
    return {arm_name(fork, number): (fork, arm) for fork in forks for number, arm in enumerate(fork["arms"])}


def hindsight_summary(directory):
    """The summary of a finished wave of hindsight forks, or None."""
    path = Path(directory) / "summary.json"
    if not path.is_file():
        return None
    summary = json.loads(path.read_text(encoding="utf-8"))
    return summary if summary.get("kind") == "fight_hindsight_wave" and summary.get("format") == HINDSIGHT_FORMAT \
        else None


def sample_hindsight(samplers, runner, directory, count, *, key, shares=SHARES, shard_samples=2000, on_run=None):
    """Fork `count` trunks of the finished fight groups of the round in `directory`
    (`pick_hindsight`, seeded by `key`, shared among the kinds by `shares`) and play every arm of
    every fork by the sampling processes `samplers` with `runner` (the round's policy,
    its fights drawn), into the round's `HINDSIGHT` wave. Returns
    the round's summary, the wave's under `fight_hindsight`. A wave cut off on the
    same forks goes on from the segments it finished; one on other forks is set aside
    and played again."""
    directory = Path(directory)
    summary = groups_summary(directory)
    if summary is None or wave_summary(directory / WAVE) is None:
        raise FileNotFoundError(f"{directory} holds no finished fight groups to fork")
    if runner.greedy_fights:
        raise ValueError("Hindsight forks draw their fights")
    plan = dict(count=count, places=list(PLACES), arms=dict(controls=CONTROLS, forced=FORCED, draws=DRAWS), key=key,
                temperature=dict(runner.temperatures), policy_digest=runner.digest)
    if shares != SHARES:
        plan["shares"] = dict(shares)
    wave = directory / HINDSIGHT
    finished = hindsight_summary(wave)
    if finished is not None and finished["plan"] == plan:
        summary["fight_hindsight"] = finished
        write_summary(directory, summary)
        return summary
    stored = read_picks(wave)
    partial = stored is not None and stored["plan"] == plan and (wave / INDEX).is_file()
    if wave.exists() and not partial:
        set_aside(wave)
    if partial:
        forks, account = stored["forks"], stored["account"]
    else:
        forks, account = pick_hindsight(directory, count, key, shares)
        wave.mkdir(parents=True)
        stage = wave / (PICKS + ".tmp")
        stage.write_text(json.dumps(dict(plan=plan, account=account, forks=forks), ensure_ascii=False,
                                    allow_nan=False) + "\n", encoding="utf-8")
        stage.replace(wave / PICKS)
    played = hindsight_jobs(forks)
    begun = time.monotonic()
    writer = WaveWriter(wave, shard_samples, resume=partial, fight_entries=True)
    pending = [job for job in played if job[1]["run_id"] not in writer.done]

    def arrived(index, run):
        writer.add(run)
        if on_run:
            on_run(HINDSIGHT, index, run)

    try:
        if pending:
            samplers.collect(runner, pending, sample=True, on_run=arrived, layout=SEGMENTS)
    finally:
        writer.close_files()
    collector = getattr(samplers, "timing", None) if pending else None
    finished = dict(kind="fight_hindsight_wave", format=HINDSIGHT_FORMAT, wave=HINDSIGHT, plan=plan, picks=account,
                    segments=len(played), **writer.tally(), fights=writer.fights.summary(),
                    seconds=round(time.monotonic() - begun, 1), collector=collector)
    write_summary(wave, finished)
    summary["fight_hindsight"] = finished
    write_summary(directory, summary)
    return summary
