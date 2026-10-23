"""The fights of the segments of act groups: each fight's entry save and its record.

With `fight_entries` a segment (`rollout.RolloutRunner.run`) keeps a record of every
fight it plays (`FightLog`): which fight it was, the player's HP and max HP at its
first decision and at the first decision after it, the potions the policy used in it,
whether it set the fight up with a choice before its first play, whether the player
died there, how many decisions it took. At the first decision of
every fight not begun by an event it has the engine write its continue save, the
fight's entry save, in the format of a boss cut's (`{"at": "fight", "save": …,
"fight": …}`), from which `CliEngine.resume_fight` enters the same fight again. The
records are the run's facts as it was played; what a fight cost is for the learning
side to say (`main/learn/`). A segment played from an entry save to the fight's end
(`RolloutRunner.run(until_fight_end=True)`, the fight groups of
`main/play/fight_groups.py`) keeps its fight's record and writes no save
(`FightLog(saves=False)`).

A round of act groups with fight entries (`groups.sample_groups(fight_entries=True)`)
writes each wave's records to `<wave>/fights.jsonl` (`FILE`), a line per fight (`ROW`),
and the entry saves to `<wave>/fights/` (`SAVES`), a file per fight that
`groups.read_save` reads as it reads a cut's save. The sampling process compresses a
save as it hands its segment over (`handed_records`); the collecting process only
writes the bytes (`FightWriter`).

What the engine's frames do not say:

- The terminal frame has no entities: a run won at its last boss has no HP after the
  fight (`exit_hp`, `exit_max_hp` None); `last_hp`, `last_max_hp` are the player's at
  the fight's last decision. A run lost in a fight has `exit_hp` 0 and the max HP of
  its last decision.
- No event tells of a death prevented (a bottled fairy, a lizard tail): the engine
  sends `encounter_started`, `encounter_completed`, `run_completed` and
  `candidate_executed` only, and raises its own death event after every preventer
  declined. A revival shows only as the HP the fight ends at.
"""

import gzip
import json
import time
from collections import Counter
from pathlib import Path

from main.env.clock import timed
from main.env.saves import continue_save

FILE = "fights.jsonl"
SAVES = "fights"
# The decisions that play a fight; a choice before the first of them sets it up.
PLAYS = frozenset({"PLAY_CARD", "USE_POTION", "END_TURN"})
# A fight's line in its wave's `FILE`, in order: what the fight was, the player's HP
# and max HP at its first decision, at the first decision after it and at its last
# decision, the content ids of the potions the policy used in it in the order used,
# whether its first choice before any play was a selection (a gambling chip's, toasty
# mittens', a choice of one card of several: `opening_selection`), whether the player
# died in it, its decisions; then the segment it was played in and the file of its entry save (None
# for a fight an event began).
ROW = ("fight", "kind", "act", "floor", "encounter", "second_boss", "from_event", "entry_hp", "entry_max_hp",
       "exit_hp", "exit_max_hp", "last_hp", "last_max_hp", "potions_used", "opening_selection", "died", "decisions",
       "run_id", "group", "character", "seed", "ascension", "path")


def vitals(frame):
    """The player's HP and max HP in a frame; None for each where it has no player."""
    for entity in frame.get("public", {}).get("entities", ()):
        if entity.get("entity_type") == "player":
            return entity.get("hp"), entity.get("max_hp")
    return None, None


class FightLog:
    """The fights of one run or segment, from the decisions `RolloutRunner.run` plays.
    Told of every decision with the fight it belongs to (`saves.FightTracker.decision`),
    it opens a fight's record at the fight's first decision, writing the entry save
    unless an event began the fight, and closes it at the first decision after the
    fight or when the run ends. Without `saves` it keeps the records alone: a segment
    played from a fight's entry save to the fight's end (`RolloutRunner.run(
    until_fight_end=True)`) does not write the save it was resumed from again."""

    def __init__(self, saves=True):
        self.fights = []  # the records, in the order the fights began
        self.open = None  # the record of the fight under way
        self.saves = saves

    def passed(self, frame, fight):
        """A decision of `fight` (None outside a fight) is reached. The fight under way,
        if this is not one of its decisions, is over: this is the first decision after it."""
        if self.open is not None and fight != self.open["fight"]:
            self.open["exit_hp"], self.open["exit_max_hp"] = vitals(frame)
            self.open = None

    def entered(self, engine, frame, fight, entry, floor):
        """A decision the run plays, of `fight`; with `entry` (`FightTracker`'s
        description) the fight's first, on `floor`."""
        if entry is not None:
            save = None
            if self.saves and not entry["from_event"]:
                # The engine still stands at the fight's entry and saves the run as it was
                # before the room, as at a boss cut.
                with timed("fight_entry_save"):
                    save = dict(at="fight", save=continue_save(engine), fight=dict(entry, floor=floor))
            hp, max_hp = vitals(frame)
            self.open = dict(
                fight=entry["fight"], kind=entry["kind"], act=entry["act"], floor=floor, encounter=entry["encounter"],
                second_boss=entry["second_boss"], from_event=entry["from_event"], entry_hp=hp, entry_max_hp=max_hp,
                exit_hp=None, exit_max_hp=None, last_hp=hp, last_max_hp=max_hp, potions_used=[], opening_selection=False,
                died=False, decisions=0, save=save, played=False)
            self.fights.append(self.open)
        if self.open is not None and fight == self.open["fight"]:
            self.open["decisions"] += 1
            self.open["last_hp"], self.open["last_max_hp"] = vitals(frame)

    def chose(self, frame, fight, candidate_ref):
        """The policy took `candidate_ref` at a decision of `fight`: a potion it used is
        the fight's, named by the potion the candidate refers to in the frame. Before
        the fight's first play a choice among several candidates that is no play sets
        the fight up (`opening_selection`)."""
        if self.open is None or fight != self.open["fight"]:
            return
        candidates = frame["legal"]["candidates"]
        candidate = next((c for c in candidates if c["candidate_ref"] == candidate_ref), None)
        verb = candidate.get("verb") if candidate is not None else None
        if not self.open["played"]:
            if verb in PLAYS:
                self.open["played"] = True
            elif len(candidates) > 1:
                self.open["opening_selection"] = True
        if verb != "USE_POTION":
            return
        sources = candidate.get("source_refs") or [None]
        potion = next((e for e in frame["public"]["entities"] if e.get("ref") == sources[0]), {})
        self.open["potions_used"].append(potion.get("content_id"))

    def ended(self, frame, victory):
        """The run ended, in the fight under way if there is one: won (the last boss)
        with what the terminal frame says, or lost there."""
        if self.open is None:
            return
        hp, max_hp = vitals(frame)
        if victory:
            self.open["exit_hp"], self.open["exit_max_hp"] = hp, max_hp
        else:
            self.open.update(died=True, exit_hp=0, exit_max_hp=self.open["last_max_hp"] if max_hp is None else max_hp)
        self.open = None

    def records(self, run):
        """The records, each with the run's id, group, character, seed and ascension and
        its entry save (`save`, None for a fight an event began)."""
        where = {key: run.get(key) for key in ("run_id", "group", "character", "seed", "ascension")}
        return [dict({key: value for key, value in record.items() if key != "played"}, **where)
                for record in self.fights]


def pack_save(save):
    """An entry save as the bytes of its file: gzip-compressed JSON, which
    `groups.read_save` reads."""
    return gzip.compress(json.dumps(save, separators=(",", ":")).encode("utf-8"), compresslevel=6, mtime=0)


def handed_records(records):
    """A segment's records as its sampling process hands them over: each save
    compressed, so that the collecting process only writes it."""
    with timed("fight_entry_packing"):
        return [dict(record, save=None if record["save"] is None else pack_save(record["save"]))
                for record in records]


def save_name(run_id, fight):
    return f"{run_id.replace(':', '-')}-{fight:03d}.json.gz"


def read_rows(path):
    path = Path(path)
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def fight_rows(directory, wave):
    """The fight records a wave wrote (`<wave>/fights.jsonl`), oldest first. A segment
    written twice (a wave gone on with after it was cut off) has its later lines stand."""
    rows = {}
    for row in read_rows(Path(directory) / wave / FILE):
        rows.pop((row["run_id"], row["fight"]), None)
        rows[(row["run_id"], row["fight"])] = row
    return list(rows.values())


class FightWriter:
    """A wave's fight records (`FILE`) and entry saves (`SAVES`), written as its
    segments arrive; a record without a save (a fight an event began, or a fight
    segment's own, `FightLog(saves=False)`) has `path` None. With `done` (the segments
    of a cut-off wave that stand) the records of those stay and the rest are written
    after them."""

    def __init__(self, directory, done=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / FILE
        standing = [] if done is None else [row for row in read_rows(path) if row["run_id"] in done]
        self.rows, self.kinds, self.died = len(standing), Counter(row["kind"] for row in standing), 0
        self.saves = self.bytes = 0
        for row in standing:
            self.died += bool(row["died"])
            if row["path"] is not None and Path(row["path"]).is_file():
                self.saves += 1
                self.bytes += Path(row["path"]).stat().st_size
        self.seconds = 0.0
        self.stream = path.open("w" if done is None else "a", encoding="utf-8")

    def add(self, records):
        """The records of a segment written to its wave (`handed_records`)."""
        began = time.monotonic()
        for record in records:
            path = None
            if record["save"] is not None:
                path = self.directory / SAVES / save_name(record["run_id"], record["fight"])
                # Made with the first save: a wave of fight segments writes none (`FightLog(saves=False)`).
                path.parent.mkdir(exist_ok=True)
                path.write_bytes(record["save"])
                self.saves += 1
                self.bytes += len(record["save"])
            row = {key: record.get(key) for key in ROW}
            row["path"] = None if path is None else str(path)
            self.stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            self.rows += 1
            self.kinds[row["kind"]] += 1
            self.died += bool(row["died"])
        self.stream.flush()
        self.seconds += time.monotonic() - began

    def close(self):
        self.stream.close()

    def summary(self):
        """What a wave's summary says of its fights: the records, by kind, the deaths,
        the entry saves and their bytes, and the collecting process's seconds writing them."""
        return dict(records=self.rows, kinds=dict(sorted(self.kinds.items())), died=self.died, saves=self.saves,
                    save_bytes=self.bytes, write_seconds=round(self.seconds, 3))
