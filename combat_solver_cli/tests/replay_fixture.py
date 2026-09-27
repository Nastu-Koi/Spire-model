"""Public decision fixtures for the production independent replay gate."""

import tempfile
from typing import ClassVar

from model.protocol import execution_command


class FakeEngine:
    serial = 0
    contract: ClassVar[dict] = {
        "adapter_version": "fake-v1",
        "observation_schema": "public-v1",
        "action_schema": "candidate-v1",
        "fixed_ascension": 0,
        "training_ready": True,
    }

    def __init__(
        self,
        *,
        victory=True,
        milestones=True,
        loop=False,
        divergent=False,
        native_error=False,
    ):
        type(self).serial += 1
        self.serial = type(self).serial
        self.victory = victory
        self.milestones = milestones
        self.loop = loop
        self.divergent = divergent
        self.native_error = native_error
        self.stderr = tempfile.TemporaryFile(mode="w+t")  # noqa: SIM115 - closed in __exit__

    def reset(self, character, seed, ascension):
        assert (character, ascension) == ("Ironclad", 0)
        self.seed, self.version = str(seed), 0
        return self._frame()

    def _frame(self):
        terminal = self.version > 0 and not self.loop
        choices = [
            {
                "candidate_ref": "win",
                "decoder_slot_ref": "win",
                "verb": "PLAY_CARD",
                "source_refs": [],
                "target_refs": [],
            },
            {
                "candidate_ref": "lose",
                "decoder_slot_ref": "lose",
                "verb": "END_TURN",
                "source_refs": [],
                "target_refs": [],
            },
        ]
        public = {
            "phase": "combat",
            "entities": [
                {
                    "ref": "player",
                    "entity_type": "player",
                    "character": "Ironclad",
                    "hp": 70 if not self.divergent else 69,
                    "max_hp": 80,
                    "act": 3,
                    "floor": 16,
                }
            ],
            "relations": [],
            "memory": [],
            "decoder_bank": [
                {key: value for key, value in item.items() if key != "candidate_ref"}
                for item in choices
            ],
        }
        events = []
        if terminal:
            public["outcome"] = {"victory": self.victory}
            if self.victory:
                if self.milestones:
                    events.extend(
                        {
                            "type": "encounter_completed",
                            "encounter_id": f"boss-{act}",
                            "act": act,
                            "kind": "boss",
                            "result": "victory",
                            "final_in_act": True,
                        }
                        for act in (1, 2, 3)
                    )
                events.append(
                    {
                        "type": "run_completed",
                        "victory": True,
                        "act": 3,
                        "final_boss_defeated": True,
                    }
                )
            else:
                events.append(
                    {
                        "type": "run_completed",
                        "victory": False,
                        "act": 3,
                        "final_boss_defeated": False,
                    }
                )
        return {
            "type": "decision_frame",
            "boundary": "terminal" if terminal else "decision",
            "contract": self.contract,
            "routing": {
                "episode_id": f"Ironclad:{self.seed}:{self.serial}",
                "decision_id": f"d{self.version}",
                "state_version": self.version,
            },
            "public": public,
            "legal": {"candidates": [] if terminal else choices},
            "events": events,
        }

    def send(self, command):
        current = self._frame()
        assert command == execution_command(current, command["candidate_ref"])
        self.version += 1
        if self.native_error:
            self.stderr.write("[ERROR] simulated engine fault\n")
        return self._frame()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.stderr.close()
