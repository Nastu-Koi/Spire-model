"""The engine's save of a run at the entry of each of its fights.

A fight the policy walked into is where the solver can teach it: from the state the
fight was entered with, the solver plays the fight and its actions become labels
(`.scratch/a0-training-baseline/issues/16-student-combat-labels.md`). What the solver
needs is the engine's continue save as it stands at the fight's first decision:
written from inside a room, the engine saves the run as it was before the room was
entered (the node is taken back out of the visited coordinates, the room cleared), and
`enter_anchor` with the fight's floor and map point type, then `anchor_room` with its
encounter, enters the same fight again: same HP, gold, deck, relics, potions and
enemies. `anchor_state` serializes the state with the room already entered and does
not serve.

One row per fight of a run, written beside the run's combat labels once the run has
ended, so that each row can name its label (`combat_index`, `result`). The save is
kept compressed until then.
"""

import gzip
import json
import os
import tempfile
import zlib
from collections import Counter
from pathlib import Path

from .protocol import ProtocolError

SCHEMA = "rollout-entry-saves-v1"
# The engine's kind of a fight, and the map point the fight's node has on the map.
MAP_POINTS = {"normal": "monster", "elite": "elite", "boss": "boss"}


def continue_save(engine):
    """The engine's continue save of the run as it stands, as the parsed JSON it writes.
    The engine writes to a file: a temporary one here, read back and removed."""
    descriptor, path = tempfile.mkstemp(prefix="spire-entry-", suffix=".json")
    os.close(descriptor)
    try:
        reply = engine.send({"cmd": "write_continue_save", "path": path})
        if reply.get("type") != "save_result" or reply.get("success") is not True:
            raise ProtocolError("write_continue_save: " + str(reply.get("message") or reply))
        return json.loads(Path(path).read_text(encoding="utf-8"))
    finally:
        Path(path).unlink(missing_ok=True)


class FightTracker:
    """Which fight a decision belongs to, from the encounter events the engine sends
    with its frames: a fight begins with `encounter_started` and ends at the first
    decision after its `encounter_completed`, or with the run."""

    def __init__(self):
        self.count = 0  # fights begun; the engine numbers encounters the same way
        self.current = None
        self.won = False
        self.bosses = Counter()  # boss fights begun, by act
        self.entry = None  # the fight whose first decision is the next one

    def events(self, events):
        for event in events:
            kind = event.get("type")
            if kind == "encounter_started":
                if not event.get("encounter") or event.get("kind") not in MAP_POINTS:
                    raise ProtocolError("Encounter start without identity")
                self.count += 1
                self.current, self.won = self.count, False
                act = event.get("act")
                boss = event["kind"] == "boss"
                self.entry = dict(
                    fight=self.count, encounter_id=event.get("encounter_id"), encounter=event["encounter"],
                    kind=event["kind"], act=act, from_event=bool(event.get("from_event")),
                    map_point_type=MAP_POINTS[event["kind"]],
                    # An act's second boss follows its first; the map names it apart.
                    second_boss=boss and self.bosses[act] > 0)
                self.bosses[act] += boss
            elif kind == "encounter_completed" and event.get("result") == "victory":
                self.won = True

    def decision(self):
        """At a decision frame: the fight it belongs to (None outside a fight) and, at
        the first decision of a fight, that fight's description."""
        if self.current is not None and self.won:
            self.current = None
        entry, self.entry = self.entry, None
        return self.current, entry


class EntrySaves:
    """The entry saves of one run's fights, written as `<run>.saves.jsonl.gz`."""

    def __init__(self, path, *, seed, character, run_id, ascension):
        self.path = Path(path)
        self.identity = dict(schema=SCHEMA, seed=str(seed), run_id=run_id, character=character, ascension=ascension)
        self.rows, self.failed = [], 0

    def take(self, engine, entry, floor):
        """The save at the first decision of a fight. A save the engine refuses is a row
        without one: the fight still has its label, and the refusal says why."""
        row = dict(self.identity, **entry, floor=floor, save=None)
        try:
            row["save"] = zlib.compress(json.dumps(continue_save(engine), separators=(",", ":")).encode(), 6)
        except ProtocolError as exc:
            row["error"] = str(exc)
            self.failed += 1
        self.rows.append(row)

    def close(self, labels):
        """Write the rows. `labels` maps an encounter id to its combat label's
        (combat_index, result); a fight without a label (cut off) is written without."""
        temporary = self.path.with_name(self.path.name + ".tmp")
        with gzip.open(temporary, "wt", encoding="utf-8") as stream:
            for row in self.rows:
                index, result = labels.get(row["encounter_id"], (None, None))
                row = dict(row, combat_index=index, result=result,
                           save=None if row["save"] is None else json.loads(zlib.decompress(row["save"])))
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        temporary.replace(self.path)
        return dict(path=self.path.name, schema=SCHEMA, fights=len(self.rows), failed=self.failed,
                    from_event=sum(row["from_event"] for row in self.rows))


def read_saves(path):
    """The rows of a saves file."""
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if row.get("schema") != SCHEMA:
                    raise ProtocolError(f"Unsupported entry-save schema in {path}")
                yield row


def anchor_commands(row):
    """The two engine commands that enter the fight of a saves row again: `enter_anchor`
    at the fight's floor and map point, then `anchor_room` with its encounter."""
    if row.get("save") is None:
        raise ProtocolError("The fight has no entry save")
    return (
        dict(cmd="enter_anchor", save=row["save"], act_floor=row["floor"],
             map_point_type=row["map_point_type"], second_boss=row["second_boss"]),
        dict(cmd="anchor_room", room=dict(type="combat", encounter=row["encounter"].removeprefix("ENCOUNTER."))),
    )
