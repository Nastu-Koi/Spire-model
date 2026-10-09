"""Versioned, restorable reward accounting from confirmed engine events only."""

from dataclasses import asdict, dataclass, field

from .protocol import ProtocolError

REWARD_VERSION = "act-progress-v2"


def require_reward_version(version):
    if version != REWARD_VERSION:
        raise ProtocolError(
            f"Reward version mismatch: expected {REWARD_VERSION}, got {version!r}. "
            "Initialize a new checkpoint for this reward contract and collect fresh PPO data; "
            "old checkpoints remain available for inference and evaluation."
        )


@dataclass
class MilestoneLedger:
    version: str = REWARD_VERSION
    encounters: set[str] = field(default_factory=set)
    bosses: set[int] = field(default_factory=set)
    victory_paid: bool = False
    components: dict[str, float] = field(
        default_factory=lambda: {"combat": 0.0, "boss": 0.0, "victory": 0.0}
    )

    def __post_init__(self):
        require_reward_version(self.version)

    def _victory(self):
        if self.victory_paid:
            return 0.0
        self.victory_paid = True
        self.components["victory"] += 5.0
        return 5.0

    def apply(self, events):
        total = 0.0
        for event in events:
            kind = event.get("type")
            if kind == "run_completed":
                if event.get("victory") is True:
                    if event.get("act") != 3 or event.get("final_boss_defeated") is not True:
                        raise ProtocolError(
                            "Victory requires confirmation of the third-act boss"
                        )
                    total += self._victory()
                continue
            if kind != "encounter_completed" or event.get("result") != "victory":
                continue
            encounter, act = event.get("encounter_id"), event.get("act")
            if not isinstance(encounter, str) or not encounter or type(act) is not int or act not in (1, 2, 3):
                raise ProtocolError("Invalid encounter milestone")
            if encounter in self.encounters:
                continue
            self.encounters.add(encounter)
            encounter_kind = event.get("kind")
            if encounter_kind == "boss":
                if event.get("final_in_act") is not True:
                    continue
                if act in self.bosses:
                    continue
                self.bosses.add(act)
                if act == 3:
                    reward = self._victory()
                else:
                    reward = float(act)
                    self.components["boss"] += reward
            elif encounter_kind in {"normal", "event", "elite"}:
                reward = 0.0
            else:
                raise ProtocolError("Unknown encounter kind")
            total += reward
        return total

    def state_dict(self):
        data = asdict(self)
        data["encounters"] = sorted(self.encounters)
        data["bosses"] = sorted(self.bosses)
        return data

    @classmethod
    def restore(cls, data):
        data = dict(data)
        require_reward_version(data.get("version"))
        data["encounters"] = set(data["encounters"])
        data["bosses"] = set(data["bosses"])
        return cls(**data)
