"""Record completed solver fights of a run as outcome labels; unfinished fights are never labels."""

from copy import deepcopy
import json
import time

from combat_solver_cli.search_support import player

from .data import SCHEMA
from .frames import input_entities


class CombatRecorder:
    def __init__(self, path, *, seed, character, solver):
        self.stream = path.open("x")
        self.identity = dict(seed=str(seed), character=character,
                             solver=solver, schema=SCHEMA, source="rollout")
        self.pending = None
        self.count = 0

    def begin(self, frame, info, prefix_length):
        if self.pending is not None:
            raise RuntimeError("Unfinished fight at new combat entry")
        if not info.get("encounter"):
            raise RuntimeError("Missing active encounter identity")
        p = player(frame)
        self.pending = dict(self.identity, entities=input_entities(frame),
                            encounter=info["encounter"],
                            kind="boss" if info["boss_combat"] else "elite" if info["elite_combat"] else "regular",
                            start_hp=p["hp"], max_hp=p["max_hp"], act=p["act"], floor=p["floor"],
                            ascension=p.get("ascension"),
                            prefix_length=prefix_length, steps=0, potions_used=[], turns=set())
        self.started = time.monotonic()

    def step(self, frame, candidate):
        row = self.pending
        row["steps"] += 1
        if (turn := player(frame).get("round")) is not None:
            row["turns"].add(turn)
        if candidate["verb"] == "USE_POTION":
            refs = candidate.get("source_refs", [])
            row["potions_used"].extend(deepcopy(e) for e in frame["public"]["entities"]
                                        if e.get("ref") in refs and e.get("entity_type") == "potion")

    def finish(self, frame, won):
        if self.pending is None:
            return
        row, self.pending = self.pending, None
        self.count += 1
        end_hp = player(frame)["hp"]
        row.update(combat_index=self.count, result="win" if won else "loss", end_hp=end_hp,
                   hp_lost=row["start_hp"] - end_hp, turns=len(row["turns"]),
                   seconds=time.monotonic() - self.started)
        self.stream.write(json.dumps(row, allow_nan=False) + "\n")
        self.stream.flush()

    def close(self):
        # A time limit or engine fault does not become a defeat training label.
        self.pending = None
        self.stream.close()
