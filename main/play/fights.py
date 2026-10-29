"""The fights of the segments of act groups: each fight's entry save and its record.

With `fight_entries` a segment (`rollout.RolloutRunner.run`) keeps a record of every
fight it plays (`FightLog`): which fight it was, the player's HP and max HP at its
first decision and at the first decision after it, the potions the policy used in it,
whether it set the fight up with a choice before its first play, whether the player
died there, how many decisions it took, and where the player stood and what they had
at its first decision (`state`: belt, deck and relics, what a potion's price may
depend on). At the first decision of
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
DANGER = 0.3  # a decision below this share of the player's max HP is in danger
# A fight's line in its wave's `FILE`, in order: what the fight was, the player's HP
# and max HP at its first decision, at the first decision after it and at its last
# decision, the content ids of the potions the policy used in it in the order used,
# whether its first choice before any play was a selection (a gambling chip's, toasty
# mittens', a choice of one card of several: `opening_selection`), whether the player
# died in it, its decisions, where the player stood and what they had at its first
# decision (`entry_state`, `state`); then the segment it was played in and the file of
# its entry save (None for a fight an event began).
ROW = ("fight", "kind", "act", "floor", "encounter", "second_boss", "from_event", "entry_hp", "entry_max_hp",
       "exit_hp", "exit_max_hp", "last_hp", "last_max_hp", "potions_used", "opening_selection", "died", "decisions",
       "entry_state", "growth", "relic_counters", "run_id", "group", "character", "seed", "ascension", "path")
# The card types a deck is counted by (`state`), as the engine names them.
CARD_TYPES = {"Attack": "attacks", "Skill": "skills", "Power": "powers", "Curse": "curses", "Status": "statuses"}
# Cards that leave the run something after the fight (ticket 15 of .scratch/training-plan):
# grown each time they are played, or something gained when their blow is fatal. A
# record's `growth` counts, by card, the plays or the fatal blows, each copy of the deck
# at most once a fight (`FightLog.chose`).
PLAYED = frozenset({"CARD.GENETIC_ALGORITHM", "CARD.THE_SCYTHE"})
FATAL = frozenset({"CARD.FEED", "CARD.HAND_OF_GREED", "CARD.THE_HUNT"})
HUNT = "POWER.THE_HUNT_POWER"  # what a fatal blow of The Hunt leaves on the player while the fight goes on


def vitals(frame):
    """The player's HP and max HP in a frame; None for each where it has no player."""
    for entity in frame.get("public", {}).get("entities", ()):
        if entity.get("entity_type") == "player":
            return entity.get("hp"), entity.get("max_hp")
    return None, None


def state(frame):
    """Where the player stands in a frame and what they have, as a potion's price may
    depend on it (`main/learn/potion_prices.py`): the act, the floor and the act's
    floors (its map's rows, as `rollout.stand` counts them), HP and max HP, the
    potions in the belt in slot order and the belt's capacity, the deck in counts
    (cards, attacks, skills, powers, curses, statuses, upgraded cards) and the relics.
    None where the frame has no player."""
    player, potions, rows, relics = None, [], [], 0
    deck = dict(cards=0, **dict.fromkeys(CARD_TYPES.values(), 0), upgraded=0)
    for entity in frame.get("public", {}).get("entities", ()):
        kind = entity.get("entity_type")
        if kind == "player" and player is None:
            player = entity
        elif kind == "potion" and entity.get("owner_ref") == "player":
            potions.append((entity.get("slot", 0), entity.get("content_id")))
        elif kind == "card" and entity.get("zone") == "deck":
            deck["cards"] += 1
            if entity.get("card_type") in CARD_TYPES:
                deck[CARD_TYPES[entity["card_type"]]] += 1
            deck["upgraded"] += bool(entity.get("upgraded"))
        elif kind == "relic":
            relics += 1
        elif kind == "map_node" and type(entity.get("floor")) is int:
            rows.append(entity["floor"])
    if player is None:
        return None
    return dict(act=player.get("act"), floor=player.get("floor"), act_floors=max(rows) + 1 if rows else None,
                hp=player.get("hp"), max_hp=player.get("max_hp"), potions=[content for _, content in sorted(potions)],
                capacity=player.get("capacity"), deck=deck, relics=relics)


def player_marks(frame):
    """What shows a fatal blow of a `FATAL` card: the player's max HP (Feed), gold (Hand
    of Greed) and stacks of The Hunt's power."""
    marks = dict(max_hp=None, gold=None, hunt=0)
    for entity in frame.get("public", {}).get("entities", ()):
        if entity.get("entity_type") == "player" and marks["max_hp"] is None:
            marks.update(max_hp=entity.get("max_hp"), gold=entity.get("gold"))
        elif entity.get("entity_type") == "power" and entity.get("owner_ref") == "player" \
                and entity.get("content_id") == HUNT:
            marks["hunt"] += entity.get("stacks") or 0
    return marks


def relic_counters(frame):
    """{relic's content id: the count it shows} of the player's relics that show one."""
    return {entity["content_id"]: entity["counter"] for entity in frame.get("public", {}).get("entities", ())
            if entity.get("entity_type") == "relic" and type(entity.get("counter")) is int}


def potion_of(frame, candidate):
    """The content id of the potion a USE_POTION candidate drinks: the potion entity
    its first source refers to in the frame (None if the frame lacks it)."""
    sources = candidate.get("source_refs") or [None]
    return next((e for e in frame["public"]["entities"] if e.get("ref") == sources[0]), {}).get("content_id")


def drinks(frame):
    """{potion's content id: the refs of the USE_POTION candidates that drink it} of a
    frame, a potion that takes a target having a candidate for each."""
    found = {}
    for candidate in frame["legal"]["candidates"]:
        if candidate.get("verb") == "USE_POTION":
            found.setdefault(potion_of(frame, candidate), []).append(candidate["candidate_ref"])
    return found


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
        if this is not one of its decisions, is over: this is the first decision after it,
        whose relics' counts are those the fight leaves (`relic_counters`)."""
        if self.open is not None and self.open["pending"] is not None:
            self._judge(frame, going_on=fight == self.open["fight"])
        if self.open is not None and fight != self.open["fight"]:
            self.open["exit_hp"], self.open["exit_max_hp"] = vitals(frame)
            self.open["relic_counters"] = relic_counters(frame)
            self._close()

    def _judge(self, frame, going_on):
        """Whether the `FATAL` card played at the decision before this one dealt a fatal
        blow, from what it left on the player: more max HP (Feed), more gold (Hand of
        Greed), more stacks of The Hunt's power, or for The Hunt the fight over at once
        (its power goes with the fight; the blow that ends a fight kills)."""
        card, before = self.open["pending"]
        self.open["pending"] = None
        after = player_marks(frame)

        def grew(mark):
            return after[mark] is not None and before[mark] is not None and after[mark] > before[mark]

        if (card == "CARD.FEED" and grew("max_hp") or card == "CARD.HAND_OF_GREED" and grew("gold")
                or card == "CARD.THE_HUNT" and (grew("hunt") or not going_on)):
            self.open["events"][card] += 1

    def _close(self):
        self.open["growth"] = growth(self.open)
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
                died=False, decisions=0, entry_state=state(frame), growth={}, relic_counters=None, save=save,
                played=False, pending=None, events=Counter(),
                copies=Counter(entity["content_id"] for entity in frame.get("public", {}).get("entities", ())
                               if entity.get("entity_type") == "card" and entity.get("zone") == "deck"
                               and entity.get("content_id") in PLAYED | FATAL))
            self.fights.append(self.open)
        if self.open is not None and fight == self.open["fight"]:
            self.open["decisions"] += 1
            self.open["last_hp"], self.open["last_max_hp"] = vitals(frame)

    def chose(self, frame, fight, candidate_ref):
        """The policy took `candidate_ref` at a decision of `fight`: a potion it used is
        the fight's, named by the potion the candidate refers to in the frame. Before
        the fight's first play a choice among several candidates that is no play sets
        the fight up (`opening_selection`). A `PLAYED` card played counts for `growth`;
        a `FATAL` card played is judged at the next decision (`_judge`)."""
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
        if verb == "PLAY_CARD":
            card = potion_of(frame, candidate)  # the entity its source names: here a card
            if card in PLAYED:
                self.open["events"][card] += 1
            elif card in FATAL:
                self.open["pending"] = (card, player_marks(frame))
        if verb != "USE_POTION":
            return
        self.open["potions_used"].append(potion_of(frame, candidate))

    def ended(self, frame, victory):
        """The run ended, in the fight under way if there is one: won (the last boss)
        with what the terminal frame says, or lost there."""
        if self.open is None:
            return
        hp, max_hp = vitals(frame)
        if self.open["pending"] is not None:
            self._judge(frame, going_on=False)
        if victory:
            self.open["exit_hp"], self.open["exit_max_hp"] = hp, max_hp
        else:
            self.open.update(died=True, exit_hp=0, exit_max_hp=self.open["last_max_hp"] if max_hp is None else max_hp)
        self._close()

    def records(self, run):
        """The records, each with the run's id, group, character, seed and ascension and
        its entry save (`save`, None for a fight an event began)."""
        where = {key: run.get(key) for key in ("run_id", "group", "character", "seed", "ascension")}
        return [dict({key: value for key, value in record.items() if key not in INTERNAL},
                     growth=growth(record), **where) for record in self.fights]


# What a record keeps while its fight goes on, and drops.
INTERNAL = frozenset({"played", "pending", "events", "copies"})


def growth(record):
    """{card: how many times it grew} of a fight's record (`PLAYED`, `FATAL`): its plays
    or fatal blows, each copy of the deck at most once (no more than the deck's copies:
    a card's reference is the frame's own, so the copies are told apart by number only)."""
    return {card: min(count, record["copies"][card]) for card, count in sorted(record["events"].items())
            if min(count, record["copies"][card]) > 0}


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
