"""Record the fights a policy plays in a rollout as combat-outcome labels.

One row per fight the policy finished: who fought it, the public decision frames
of the fight in order, and how it ended. The engine names the fight when it
begins and confirms a win; a death during a fight is its loss. A fight cut off
by a time limit or an engine fault is counted and never becomes a label. Rows
are written when the run ends, once it is known how the act of each fight ended.
"""

import gzip
import json
import zlib
from copy import deepcopy
from pathlib import Path

from combat_solver_cli.search_support import player
from model.protocol import ProtocolError, action_semantics

from .data import SCHEMA
from .frames import input_entities

KINDS = {"normal": "regular", "elite": "elite", "boss": "boss"}


def resources(frame):
    """What a fight can change besides HP, as the player sees it."""
    hero = player(frame)
    return dict(hp=hero["hp"], max_hp=hero["max_hp"], gold=hero.get("gold"),
                potions=sorted(e["content_id"] for e in frame["public"]["entities"]
                               if e.get("entity_type") == "potion" and e.get("content_id")))


def act_results(status, victory, passed, last_act):
    """How each act the run entered ended for this policy: passed, died, or not known."""
    results = {act: "passed" for act in passed}
    if last_act is not None and last_act not in results:
        results[last_act] = "died" if status == "complete" and victory is False else None
    return results


class CombatRecorder:
    def __init__(self, path, *, seed, character, run_id, policy, contract):
        self.path = Path(path)
        self.identity = dict(schema=SCHEMA, source="rollout", actor="policy", seed=str(seed), run_id=run_id,
                             character=character, policy=deepcopy(policy), contract=deepcopy(contract))
        self.pending, self.rows = None, []
        # Not labels: fights cut off before their end, and fights whose entry or
        # result was never observed at a decision.
        self.unfinished = self.unobserved = 0
        self.last_act = None

    def events(self, events):
        """Fight boundaries the engine reports with a frame, before the frame is read."""
        for event in events:
            kind = event.get("type")
            if kind == "encounter_started":
                if self.pending is not None:
                    raise ProtocolError("A fight began before the previous one ended")
                if not event.get("encounter") or event.get("kind") not in KINDS:
                    raise ProtocolError("Encounter start without identity")
                self.pending = dict(encounter_id=event["encounter_id"], encounter=event["encounter"],
                                    kind=KINDS[event["kind"]], from_event=bool(event.get("from_event")),
                                    act=event["act"], entry=None, won=False, frames=[], turns=set(),
                                    potions_used=[], potions_discarded=[])
            elif kind == "encounter_completed" and event.get("result") == "victory":
                if self.pending is None or self.pending["encounter_id"] != event["encounter_id"]:
                    raise ProtocolError("The engine completed a fight it never announced")
                self.pending["won"] = True

    def decision(self, frame):
        """A decision frame: the first of a fight is its entry, the first after a win its exit."""
        self.last_act = player(frame).get("act", self.last_act)
        fight = self.pending
        if fight is None:
            return
        if fight["won"]:
            self._finish(True, resources(frame))
        elif fight["entry"] is None:
            hero = player(frame)
            fight.update(entry=resources(frame), entities=input_entities(frame), floor=hero.get("floor"),
                         ascension=hero.get("ascension"))

    def step(self, frame, candidate, index):
        """The action taken at a decision frame of the fight in progress."""
        fight = self.pending
        if fight is None or fight["won"]:
            return
        if (turn := player(frame).get("round")) is not None:
            fight["turns"].add(turn)
        if candidate["verb"] in ("USE_POTION", "DISCARD_POTION"):
            refs = candidate.get("source_refs", [])
            fight["potions_used" if candidate["verb"] == "USE_POTION" else "potions_discarded"].extend(
                deepcopy(e) for e in frame["public"]["entities"]
                if e.get("ref") in refs and e.get("entity_type") == "potion")
        # A fight is tens of frames that differ little; they wait for the run's end compressed.
        fight["frames"].append(zlib.compress(json.dumps(dict(
            step=index, forced=len(frame["legal"]["candidates"]) == 1, public=frame["public"],
            action=action_semantics(candidate)), separators=(",", ":"), allow_nan=False).encode(), 1))

    def terminal(self, victory, final_hp):
        """The run ended. `final_hp()` reads the HP a won last fight left: no frame follows it."""
        fight = self.pending
        if fight is None:
            return
        if fight["won"]:
            hp = final_hp()
            self._finish(True, None if hp is None else dict(hp=hp))
        elif victory:
            raise ProtocolError("The run was won during a fight the engine did not complete")
        else:
            self._finish(False, None)

    def _finish(self, won, end):
        fight, self.pending = self.pending, None
        entry = fight.pop("entry")
        if entry is None or (won and end is None):
            self.unobserved += 1
            return
        frames = fight.pop("frames")
        # A lost fight costs all the HP it was entered with.
        end_hp = end["hp"] if won else 0
        self.rows.append(dict(
            self.identity, combat_index=len(self.rows) + 1, encounter_id=fight.pop("encounter_id"),
            encounter=fight.pop("encounter"), kind=fight.pop("kind"), from_event=fight.pop("from_event"),
            act=fight.pop("act"), floor=fight.pop("floor"), ascension=fight.pop("ascension"),
            result="win" if won else "loss", start_hp=entry["hp"], max_hp=entry["max_hp"], end_hp=end_hp,
            hp_lost=entry["hp"] - end_hp, start=entry, end=end if won and "max_hp" in end else None,
            turns=len(fight.pop("turns")), steps=len(frames), potions_used=fight.pop("potions_used"),
            potions_discarded=fight.pop("potions_discarded"), entities=fight.pop("entities"), frames=frames))

    def close(self, status, victory, passed):
        """Write the run's fights and return their counts. `passed` are the acts whose boss fell."""
        if self.pending is not None:
            # A time limit or an engine fault is not a defeat.
            self.pending, self.unfinished = None, self.unfinished + 1
        acts = act_results(status, victory, passed, self.last_act)
        temporary = self.path.with_name(self.path.name + ".tmp")
        with gzip.open(temporary, "wt", encoding="utf-8") as stream:
            for row in self.rows:
                row = dict(row, act_result=acts.get(row["act"]),
                           frames=[json.loads(zlib.decompress(f)) for f in row["frames"]])
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        temporary.replace(self.path)
        results = [row["result"] for row in self.rows]
        return dict(path=self.path.name, schema=SCHEMA, wins=results.count("win"), losses=results.count("loss"),
                    unfinished=self.unfinished, unobserved=self.unobserved,
                    acts={str(act): result for act, result in sorted(acts.items())})
