"""The engine's continue save of a run, and which fight a decision belongs to.

A continue save written from inside a room holds the run as it was before the room
was entered (the node is taken back out of the visited coordinates, the room cleared).
Written during a fight it is the run as the engine kept it when the fight's room was
entered, before the fight's start acted on it (relics' heals, gold, potions and counts,
the enemies' HP, the opening shuffle), and enters the same fight again as it began:
the same actions play it the same way (`fight_resume_command`,
`CliEngine.resume_fight`). Written at a map decision it goes on from the map
(`CliEngine.resume`). Act groups cut runs at both
(`main/play/groups.py`). `anchor_state` serializes the state with the room already
entered and does not serve.

The engine returns with a continue save the public history its run had observed as
the save stands (`public_history/NativeHistory.Snapshot`): the odds of a potion
drop and of the question rooms, whether the last map point was a shop. It is kept in
the save under `PUBLIC_HISTORY`, so that it goes wherever the save goes (a segment's
save, a fight's entry save, a potion point's branches), and handed back when a run
resumes from the save (`resume_fields`): the run goes on knowing what its own run had
observed, where without it a loaded save starts the history unknown. A save from an
engine before it, or written by a run that was itself resumed without one, carries
unknowns on.
"""

import json
import os
import tempfile
from collections import Counter
from pathlib import Path

from main.env.protocol import ProtocolError

# The engine's kind of a fight, and the map point the fight's node has on the map.
MAP_POINTS = {"normal": "monster", "elite": "elite", "boss": "boss"}

# Where a save keeps the public history the engine returned with it.
PUBLIC_HISTORY = "spire_public_history"


def continue_save(engine):
    """The engine's continue save of the run as it stands, as the parsed JSON it writes,
    with the public history the engine returned beside it under `PUBLIC_HISTORY`. The
    engine writes to a file: a temporary one here, read back and removed."""
    descriptor, path = tempfile.mkstemp(prefix="spire-entry-", suffix=".json")
    os.close(descriptor)
    try:
        reply = engine.send({"cmd": "write_continue_save", "path": path})
        if reply.get("type") != "save_result" or reply.get("success") is not True:
            raise ProtocolError("write_continue_save: " + str(reply.get("message") or reply))
        save = json.loads(Path(path).read_text(encoding="utf-8"))
        if reply.get("public_history") is not None:
            save[PUBLIC_HISTORY] = reply["public_history"]
        return save
    finally:
        Path(path).unlink(missing_ok=True)


def resume_fields(save):
    """A save as a command that resumes from it takes it (`resume_save`, `resume_fight`):
    `save` without the public history it keeps, and that history as `public_history`
    when it has one. The save given is left as it is."""
    fields = dict(save={key: value for key, value in save.items() if key != PUBLIC_HISTORY})
    if save.get(PUBLIC_HISTORY) is not None:
        fields["public_history"] = save[PUBLIC_HISTORY]
    return fields


class FightTracker:
    """Which fight a decision belongs to, from the encounter events the engine sends
    with its frames: a fight begins with `encounter_started` and ends at the first
    decision after its `encounter_completed`, or with the run."""

    def __init__(self):
        self.count = 0  # fights begun; the engine numbers encounters the same way
        self.current = None
        self.won = False
        self.bosses = Counter()  # boss fights begun, by act
        self.kinds = {}  # each fight's kind: normal, elite or boss
        self.entry = None  # the fight whose first decision is the next one
        # How each fight ended for the policy: "win", "loss", or None while it goes on
        # and for a fight the run was cut off in.
        self.results = {}

    def events(self, events):
        for event in events:
            kind = event.get("type")
            if kind == "encounter_started":
                if not event.get("encounter") or event.get("kind") not in MAP_POINTS:
                    raise ProtocolError("Encounter start without identity")
                self.count += 1
                self.current, self.won = self.count, False
                self.results[self.count] = None
                self.kinds[self.count] = event["kind"]
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
                if self.current is not None:
                    self.results[self.current] = "win"

    def finish(self, victory):
        """The run ended: a defeat inside a fight is that fight's loss. A time limit or
        an engine fault is not a defeat, and never comes here."""
        if self.current is not None and not self.won and not victory:
            self.results[self.current] = "loss"

    def tally(self):
        """The fights of the run by how they ended, as a run records them."""
        results = list(self.results.values())
        return dict(wins=results.count("win"), losses=results.count("loss"), unfinished=results.count(None))

    def decision(self):
        """At a decision frame: the fight it belongs to (None outside a fight) and, at
        the first decision of a fight, that fight's description."""
        if self.current is not None and self.won:
            self.current = None
        entry, self.entry = self.entry, None
        return self.current, entry

    @property
    def kind(self):
        """The kind of the fight the last decision belongs to; None outside a fight."""
        return self.kinds.get(self.current)


def fight_resume_command(row):
    """The engine command that resumes a run at the fight of a saves row: `resume_fight`
    with the save, the encounter, the floor and the map point (`CliEngine.resume_fight`)."""
    if row.get("save") is None:
        raise ProtocolError("The fight has no entry save")
    return dict(cmd="resume_fight", **resume_fields(row["save"]), encounter=row["encounter"], act_floor=row["floor"],
                map_point_type=row["map_point_type"], second_boss=bool(row.get("second_boss")))

