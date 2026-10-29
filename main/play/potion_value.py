"""Points and paired continuations for the potions' prices (`main/learn/potion_prices.py`;
`.scratch/training-plan` ticket 10).

1. `play_sources`: runs of the checkpoint on seeds of their own write a point at the
   first map decision after every fight that leaves a potion in the belt
   (`PointEngine`): how to come to it again from its act's anchor (the run's start, or
   the act's first map decision, whose save resumes there) and where the player stands
   (`fights.state`), with the run's character, seed and ascension. A continue save
   written at the point itself would resume before the room the point follows.
2. `draw`: points by act, belt full or not and HP band (`BANDS`), each act and belt
   taking as many pairs (potions held) as asked, the bands weighted by `BAND_WEIGHTS`:
   a potion saves a run where HP is low, so those points count more.
3. `branch_saves`, `play_branches`: from each point one continuation as it stood and
   one for each potion with that potion thrown away at the point, each coming to the
   point again from its anchor (`PointEngine.resume`), all played greedily to the
   next act's first decision or, in the last act, to the run's end; each keeps its
   fights' records and where it stopped (`RolloutRunner.run(fight_entries=True,
   entry_saves=False)`: `cut_state`). Some points' continuation as it stood is played
   twice, to see how far batching's near ties part two plays of the same point
   (`potion_prices.determinism`).

Every step is written to the output as it is made and taken from there when asked
again: a cut-off fit goes on.
"""

import copy
import gzip
import json
import random
import time
from collections import Counter
from pathlib import Path

from main.env.protocol import ProtocolError, execution_command
from main.env.saves import FightTracker, continue_save
from main.play.digests import frame_key
from main.play.fights import state as fight_state

POINTS, BRANCHES, RESULTS = "points", "branches", "branches.jsonl.gz"
# The acts' anchors, under the points; in a branch's save, how to come to its point again.
ANCHORS, REPLAY = "anchors", "potion_point_replay"
# How long a replay waits on the engine's waiting frames, as the runner does.
WAIT = 30.0
# The HP bands points are drawn by (share of max HP), and how much each weighs in an
# act and belt's pairs.
BANDS = ((0.0, 0.5), (0.5, 0.8), (0.8, 1.01))
BAND_WEIGHTS = (2, 1, 1)
# A continuation keeps its fights' records and where it stopped; no frames, no run file.
LAYOUT = dict(trajectory="summary", journal=False, combat_frames=False, segment=True, fight_entries=True,
              entry_saves=False)
# The player's fight fields a frame keeps after a fight until the next one; a run resumed
# from an act's map save has them unset. They are left out of the key a replay must meet.
STALE = ("energy", "max_energy", "stars", "round")


def point_key(frame):
    """What a replay must come to: the decision's public state and candidates
    (`digests.frame_key`) without the player's stale fight fields (`STALE`)."""
    frame = copy.deepcopy(frame)
    for entity in frame["public"]["entities"]:
        if entity and entity.get("entity_type") == "player":
            for name in STALE:
                entity.pop(name, None)
    return frame_key(frame)


class PointEngine:
    """An engine that writes a point at the first map decision after each fight of the
    runs started on `seeds` (character, seed) when the belt holds a potion, and resumes
    a continuation at its point.

    The engine's continue save written at that decision resumes before the room the
    point follows, and plays the room again; one written at an act's first map decision
    resumes to that decision. So a point is how to come to it again: its anchor (the
    run's start in the first act, else its act's first map decision, whose save is
    `anchors/<character>-<seed>-act<act>.json.gz` beside the points), the candidates
    taken from there and its decision's key (`point_key`), with where the player stands
    (`fights.state`), the run's character, seed and ascension. A save holding `REPLAY`
    (`branch_saves`) is resumed by starting the run again or resuming the anchor, taking
    the candidates again (waiting frames advanced as the runner does), checking the key
    and throwing the branch's potion away there."""

    def __init__(self, inner, directory, seeds):
        self.inner, self.directory, self.seeds = inner, Path(directory), set(seeds)
        self.active = None

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def reset(self, character, seed, ascension):
        frame = self.inner.reset(character, seed, ascension)
        self.active = (character, str(seed), ascension) if (character, str(seed)) in self.seeds else None
        self.tracker, self.after_fight, self.count = FightTracker(), False, 0
        self.anchor, self.refs, self.act = "start", [], 1
        return self._seen(frame)

    def resume(self, save):
        self.active = None
        if REPLAY not in save:
            return self.inner.resume(save)
        save = dict(save)
        replay = save.pop(REPLAY)
        frame = self._settled(self.inner.reset(*replay["start"]) if "start" in replay else self.inner.resume(save))
        for index, ref in enumerate(replay["refs"]):
            if all(candidate["candidate_ref"] != ref for candidate in frame["legal"]["candidates"]):
                raise ProtocolError(f"A potion point's replay diverged at its decision {index}")
            frame = self._settled(self.inner.send(execution_command(frame, ref)))
        if point_key(frame) != replay["key"]:
            raise ProtocolError("A potion point's replay came to another decision than its point's")
        if replay.get("drop") is not None:
            index, potion = replay["drop"]
            belt = sorted((e for e in frame["public"]["entities"]
                           if e and e.get("entity_type") == "potion" and e.get("owner_ref") == "player"),
                          key=lambda e: e.get("slot", 0))
            if belt[index].get("content_id") != potion:
                raise ProtocolError("A potion point's replay holds another potion than the one to throw away")
            thrown = next((c for c in frame["legal"]["candidates"]
                           if c["verb"] == "DISCARD_POTION" and c.get("source_refs") == [belt[index]["ref"]]), None)
            if thrown is None:
                raise ProtocolError("A potion point's replay cannot throw its potion away")
            frame = self._settled(self.inner.send(execution_command(frame, thrown["candidate_ref"])))
        # The run goes on from the point as from a resumed save: what led to it is not its own.
        return dict(frame, events=[])

    def resume_fight(self, row):
        self.active = None
        return self.inner.resume_fight(row)

    def send(self, command):
        if self.active and command.get("cmd") == "execute_candidate":
            self.refs.append(command["candidate_ref"])
        reply = self.inner.send(command)
        return self._seen(reply) if self.active else reply

    def _settled(self, frame):
        """The next decision: a waiting frame is advanced as the runner advances it."""
        began = time.monotonic()
        while frame.get("boundary") == "waiting":
            if time.monotonic() - began > WAIT:
                raise TimeoutError("unresolved_wait_timeout")
            time.sleep(0.01)
            frame = self.inner.send({"cmd": "advance_to_boundary"})
        return frame

    def _seen(self, frame):
        if not self.active or not isinstance(frame, dict) or "boundary" not in frame:
            return frame
        self.tracker.events(frame.get("events", []))
        if frame["boundary"] in ("terminal", "waiting"):
            return frame
        character, seed, ascension = self.active
        where = fight_state(frame) if frame["public"].get("phase") == "map" else None
        if where and (where["act"] or 1) > self.act:
            self.act = where["act"]
            self.anchor, self.refs = f"{character}-{seed}-act{self.act}", []
            write_gz(self.directory / ANCHORS / f"{self.anchor}.json.gz", continue_save(self.inner))
        fight, _ = self.tracker.decision()
        if fight is not None:
            self.after_fight = True
        elif self.after_fight and where:
            self.after_fight = False
            if where["potions"]:
                self.count += 1
                point = dict(id=f"{character}-{seed}-{self.count:03d}", character=character, seed=seed,
                             ascension=ascension, state=where, anchor=self.anchor, refs=list(self.refs),
                             key=point_key(frame))
                write_gz(self.directory / f"{point['id']}.json.gz", point)
        return frame


def write_gz(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as stream:
        json.dump(value, stream, separators=(",", ":"))
    temporary.replace(path)


def play_sources(samplers, runner, seeds, ascensions, output):
    """The source runs (`seeds` by character, each at its ascension in `ascensions`),
    played to their end by `runner` in the engines of `samplers`, whose engines are
    `PointEngine`s writing to `<output>/points`. Returns how the runs ended; a second
    call finds them played (`source.json`)."""
    output = Path(output)
    done = output / "source.json"
    if done.is_file():
        return json.loads(done.read_text())
    (output / POINTS).mkdir(parents=True, exist_ok=True)
    outcomes = Counter()
    jobs = [(character, dict(seed=seed, run_id=f"{character}:{seed}:source", group=f"{character}:{seed}",
                             ascension=ascensions[character][seed], until_act=4), None)
            for character in seeds for seed in seeds[character]]
    samplers.collect(runner, jobs, sample=True, layout=dict(LAYOUT, fight_entries=False),
                     on_run=lambda _, run: outcomes.update([run.get("outcome") or run["status"]]))
    found = dict(runs=len(jobs), outcomes=dict(outcomes), points=len(list((output / POINTS).glob("*.json.gz"))))
    done.write_text(json.dumps(found) + "\n")
    return found


def read_point(path, with_replay=False):
    """A point as written, without how to come to it again (its candidates and key)
    unless asked."""
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        point = json.load(stream)
    if not with_replay:
        point.pop("refs")
        point.pop("key")
    point["path"] = str(path)
    return point


def band(where):
    share = where["hp"] / where["max_hp"] if where.get("max_hp") else 1.0
    return next(number for number, (low, high) in enumerate(BANDS) if low <= share < high)


def draw(points, pairs_per_act, *, seed=20261023):
    """Points for `pairs_per_act` pairs in each act, half with the belt full and half
    not, split among the HP bands by `BAND_WEIGHTS`; where a stratum has too few
    points, all of them. Returns the points drawn and each stratum's pairs (asked,
    found)."""
    random_draw = random.Random(seed)
    strata = {}
    for point in sorted(points, key=lambda p: p["id"]):
        where = point["state"]
        full = where.get("capacity") is not None and len(where["potions"]) >= where["capacity"]
        strata.setdefault((where["act"], full, band(where)), []).append(point)
    drawn, account = [], {}
    for act in (1, 2, 3):
        for full in (False, True):
            for number, weight in enumerate(BAND_WEIGHTS):
                asked = round(pairs_per_act / 2 * weight / sum(BAND_WEIGHTS))
                found = list(strata.get((act, full, number), ()))
                random_draw.shuffle(found)
                pairs = 0
                for point in found:
                    if pairs >= asked:
                        break
                    drawn.append(point)
                    pairs += len(point["state"]["potions"])
                account[f"{act}-{'full' if full else 'room'}-{number}"] = dict(asked=asked, found=pairs)
    return drawn, account


def branch_saves(points, directory):
    """The saves of each point's continuations: as it stood (`keep`) and one for each
    potion with that potion thrown away at the point (`drop<slot>`, the potion's place in
    the belt's slot order), each its anchor's save, or none where the anchor is the run's
    start, holding how to come to the point again (`REPLAY`). Returns the branches."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    branches = []
    for point in points:
        found = read_point(point["path"], with_replay=True)
        replay = dict(refs=found["refs"], key=found["key"])
        if found["anchor"] == "start":
            anchor, replay["start"] = {}, [found["character"], found["seed"], found["ascension"]]
        else:
            with gzip.open(Path(point["path"]).parent / ANCHORS / f"{found['anchor']}.json.gz", "rt",
                           encoding="utf-8") as stream:
                anchor = json.load(stream)
        variants = [("keep", None)] + [(f"drop{index}", potion) for index, potion in enumerate(found["state"]["potions"])]
        for name, dropped in variants:
            path = directory / f"{point['id']}-{name}.json.gz"
            if not path.is_file():
                drop = None if dropped is None else [int(name[len("drop"):]), dropped]
                write_gz(path, dict(at="act", save=dict(anchor, **{REPLAY: dict(replay, drop=drop)})))
            branches.append(dict(point=point["id"], branch=name, dropped=dropped, path=str(path)))
    return branches


def read_results(path):
    path = Path(path)
    if not path.is_file():
        return []
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def play_branches(samplers, runner, points, branches, output, *, repeat=30):
    """Every continuation of `branches` not played yet, greedily, with the first
    `repeat` points' `keep` once more (`keep-again`); each result is a line of
    `<output>/branches.jsonl.gz` as it arrives: the branch, how it ended, its fights'
    records and where it stopped. Returns all the results."""
    path = Path(output) / RESULTS
    results = read_results(path)
    done = {(row["point"], row["branch"]) for row in results}
    named = {point["id"]: point for point in points}
    again = [dict(branch, branch="keep-again") for branch in branches if branch["branch"] == "keep"][:repeat]
    pending = [branch for branch in branches + again if (branch["point"], branch["branch"]) not in done]
    jobs = []
    for branch in pending:
        point = named[branch["point"]]
        act = point["state"]["act"]
        jobs.append((point["character"], dict(seed=point["seed"], run_id=f"{branch['point']}:{branch['branch']}",
                                              group=branch["point"], resume=branch["path"],
                                              ascension=point["ascension"], **({} if act >= 3 else
                                                                               dict(until_act=act + 1))), None))
    with gzip.open(path, "at", encoding="utf-8") as stream:
        def arrived(index, run):
            run.pop("save", None)  # where it stopped, as a save: not needed, and large
            branch = pending[index]
            row = dict(point=branch["point"], branch=branch["branch"], dropped=branch["dropped"],
                       status=run["status"], outcome=run.get("outcome"), error=run.get("error"),
                       records=[{key: value for key, value in record.items() if key != "save"}
                                for record in run.get("fight_records") or ()],
                       cut_state=run.get("cut_state"))
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            results.append(row)

        if jobs:
            samplers.collect(runner, jobs, sample=False, layout=LAYOUT, on_run=arrived)
    return results
