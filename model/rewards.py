"""Versioned, restorable reward accounting from confirmed engine events only."""
from dataclasses import asdict, dataclass, field

from .protocol import ProtocolError


@dataclass
class MilestoneLedger:
    version: str = "milestones-v1"
    encounters: set[str] = field(default_factory=set)
    bosses: set[int] = field(default_factory=set)
    paid: dict[int, float] = field(default_factory=dict)
    victory_paid: bool = False
    components: dict[str, float] = field(default_factory=lambda: {"combat": 0., "boss": 0., "victory": 0.})

    def apply(self, events):
        total = 0.0
        for event in events:
            kind = event.get("type")
            if kind == "run_completed":
                if event.get("victory") is True and not self.victory_paid:
                    if event.get("act") != 3 or not event.get("final_boss_defeated"):
                        raise ProtocolError("Victory requires confirmation of the third-act boss")
                    self.victory_paid = True
                    self.components["victory"] += 1.0
                    total += 1.0
                continue
            if kind != "encounter_completed" or event.get("result") != "victory":
                continue
            encounter, act = event.get("encounter_id"), event.get("act")
            if not encounter or act not in (1, 2, 3):
                raise ProtocolError("Invalid encounter milestone")
            if encounter in self.encounters:
                continue
            self.encounters.add(encounter)
            encounter_kind = event.get("kind")
            if encounter_kind == "boss":
                if not event.get("final_in_act"):
                    continue
                if act in self.bosses:
                    continue
                self.bosses.add(act)
                reward = {1: .1, 2: .2, 3: 0.0}[act]
                self.components["boss"] += reward
            elif encounter_kind in {"normal", "event", "elite"}:
                reward = min(.02 if encounter_kind == "elite" else .005, max(0, .1 - self.paid.get(act, 0)))
                self.paid[act] = self.paid.get(act, 0) + reward
                self.components["combat"] += reward
            else:
                raise ProtocolError("Unknown encounter kind")
            total += reward
        return total

    def public(self, act):
        return {"entity_type": "reward_progress", "act": act, "nonboss_paid": self.paid.get(act, 0),
                "boss_1": 1 in self.bosses, "boss_2": 2 in self.bosses, "boss_3": 3 in self.bosses}

    def state_dict(self):
        data = asdict(self)
        data["encounters"] = sorted(self.encounters)
        data["bosses"] = sorted(self.bosses)
        return data

    @classmethod
    def restore(cls, data):
        data = dict(data)
        data["encounters"] = set(data["encounters"])
        data["bosses"] = set(data["bosses"])
        data["paid"] = {int(k): v for k, v in data["paid"].items()}
        if data.get("version") != "milestones-v1":
            raise ValueError("Reward version mismatch")
        return cls(**data)
