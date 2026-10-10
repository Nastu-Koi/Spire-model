"""Versioned, restorable reward accounting from confirmed engine events only.

A run pays once, when it ends: +1 for a victory, -1 for a defeat. Nothing on the way
pays. A victory is paid on the events that confirm it; a defeat on its event or, where
the engine sends none, on the terminal state. The acts whose boss fell are still
recorded: reports say how far a run got.
"""

from dataclasses import asdict, dataclass, field

from .protocol import ProtocolError

REWARD_VERSION = "outcome-v1"


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
    defeat_paid: bool = False
    components: dict[str, float] = field(
        default_factory=lambda: {"victory": 0.0, "defeat": 0.0}
    )

    def __post_init__(self):
        require_reward_version(self.version)

    @property
    def outcome(self):
        """Whether the run was won, once it has ended; None until then."""
        return True if self.victory_paid else False if self.defeat_paid else None

    def _victory(self):
        if self.defeat_paid:
            raise ProtocolError("A lost run cannot be won")
        if self.victory_paid:
            return 0.0
        self.victory_paid = True
        self.components["victory"] += 1.0
        return 1.0

    def _defeat(self):
        if self.victory_paid:
            raise ProtocolError("A won run cannot be lost")
        if self.defeat_paid:
            return 0.0
        self.defeat_paid = True
        self.components["defeat"] -= 1.0
        return -1.0

    def finish(self, victory):
        """What is still owed when the terminal state says how the run ended. The engine
        confirms every victory with events, but ends some lost runs without one."""
        if victory:
            if not self.victory_paid:
                raise ProtocolError("Terminal victory was not confirmed by the run's events")
            return 0.0
        return self._defeat()

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
                elif event.get("victory") is False:
                    total += self._defeat()
                else:
                    raise ProtocolError("Run completion does not say whether the run was won")
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
                # The last act's boss falling is the victory, whichever event says so first.
                reward = self._victory() if act == 3 else 0.0
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
