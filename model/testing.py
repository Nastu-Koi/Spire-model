"""Synthetic protocol engine for integration tests; NOT a game simulator."""

from copy import deepcopy

from .protocol import ProtocolError, execution_command


class SyntheticEngine:
    def __init__(self, count=10, select=5, *, forced_first=False):
        self.count, self.select, self.forced_first = count, select, forced_first

    def reset(self, character="Ironclad", seed="test"):
        self.character, self.seed = character, str(seed)
        self.selected, self.version, self.stage = [], 0, "selection"
        self.events = []
        self.commands = []
        return self.frame()

    @property
    def contract(self):
        return {
            "adapter_version": "synthetic-v1",
            "observation_schema": "public-v1",
            "action_schema": "candidate-v1",
            "fixed_ascension": 10,
            "training_ready": True,
            "domain": "synthetic_test_only",
            "effect_registry_hash": "test-only",
        }

    def frame(self):
        episode = self.character + ":" + self.seed
        routing = {
            "episode_id": episode,
            "decision_id": f"{episode}:{self.version}",
            "state_version": self.version,
            "base_public_version": 0 if self.stage == "selection" else self.version,
            "action_bank_version": self.stage,
        }
        public = {
            "phase": "card_select" if self.stage == "selection" else self.stage,
            "entities": [
                {
                    "ref": "player",
                    "entity_type": "player",
                    "character": self.character,
                    "hp": 70,
                    "max_hp": 80,
                    "act": 3,
                }
            ],
            "relations": [],
            "memory": [],
        }
        result = {
            "type": "decision_frame",
            "contract": self.contract,
            "boundary": "decision",
            "routing": routing,
            "public": public,
            "events": deepcopy(self.events),
        }
        self.events = []
        if self.stage == "terminal":
            result["boundary"] = "terminal"
            public["outcome"] = {"victory": self.won}
            result["legal"] = {"candidates": []}
            return result
        if self.stage == "selection":
            routing.update(
                selection_id=episode + ":selection",
                selection_revision=len(self.selected),
            )
            public["entities"] += [
                {
                    "ref": f"card:{i}",
                    "entity_type": "card",
                    "zone": "hand",
                    "content_id": "STRIKE" if i % 2 else "DEFEND",
                    "cost": i % 3,
                    "stats": {"damage": 6 + i},
                    "semantic_program": {
                        "kind": "effect",
                        "op": "DAMAGE",
                        "args": {"amount": {"kind": "literal", "value": 6 + i}},
                        "bindings": {"recipient": {"kind": "entity", "ref": "player"}},
                    },
                }
                for i in range(self.count)
            ]
            bank = [
                {
                    "verb": "SELECT_ONE",
                    "operation": "exhaust",
                    "decoder_slot_ref": f"card:{i}",
                    "source_refs": [f"card:{i}"],
                    "target_refs": [],
                    "effect_coverage": "complete",
                    "semantic_program": {
                        "kind": "effect",
                        "op": "EXHAUST",
                        "bindings": {"subject": {"kind": "entity", "ref": f"card:{i}"}},
                    },
                }
                for i in range(self.count)
            ]
            bank.append(
                {
                    "verb": "FINISH_SELECTION",
                    "decoder_slot_ref": "stop",
                    "source_refs": [],
                    "target_refs": [],
                    "effect_coverage": "complete",
                }
            )
            legal = (
                [x for i, x in enumerate(bank[:-1]) if i not in self.selected]
                if len(self.selected) < self.select
                else [bank[-1]]
            )
            if self.forced_first and not self.selected:
                legal = legal[:1]
            public["decoder_bank"] = bank
            public["selection_context"] = {
                "mode": "buffered",
                "operation": "exhaust",
                "selected_refs": [f"card:{i}" for i in self.selected],
                "selected_count": len(self.selected),
                "min_total": self.select,
                "max_total": self.select,
                "remaining_required": self.select - len(self.selected),
                "can_finish": len(self.selected) == self.select,
                "can_skip": False,
                "can_cancel": False,
                "order_matters": False,
            }
        else:
            legal = [
                {
                    "verb": verb,
                    "decoder_slot_ref": verb,
                    "source_refs": ["player"],
                    "target_refs": [],
                    "effect_coverage": "opaque",
                    "semantic_program": {"kind": "effect", "op": verb},
                }
                for verb in ("PLAY_CARD", "END_TURN")
            ]
            public["decoder_bank"] = legal
        result["legal"] = {
            "candidates": [dict(c, candidate_ref=f"c{i}") for i, c in enumerate(legal)]
        }
        return result

    def send(self, command):
        if command["cmd"] == "advance_to_boundary":
            return self.frame()
        before = self.frame()
        try:
            expected = execution_command(before, command["candidate_ref"])
            if command != expected:
                raise ProtocolError("stale_decision")
            candidate = next(
                c
                for c in before["legal"]["candidates"]
                if c["candidate_ref"] == command["candidate_ref"]
            )
        except (KeyError, StopIteration):
            raise ProtocolError("unknown_candidate") from None
        self.commands.append(deepcopy(command))
        self.version += 1
        if self.stage == "selection":
            if candidate["verb"] == "FINISH_SELECTION":
                self.stage = "combat_play"
                self.events = [
                    {
                        "type": "encounter_completed",
                        "encounter_id": "ordinary",
                        "act": 1,
                        "kind": "normal",
                        "result": "victory",
                    }
                ]
            else:
                self.selected.append(int(candidate["decoder_slot_ref"].split(":")[1]))
        else:
            self.stage = "terminal"
            self.won = candidate["verb"] == "PLAY_CARD"
            self.events = [
                {
                    "type": "run_completed",
                    "victory": self.won,
                    "act": 3,
                    "final_boss_defeated": self.won,
                }
            ]
        return self.frame()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


def demonstration(character="Ironclad", seed="demo", count=10, select=5):
    """Generate an explicit synthetic teacher trajectory, never an on-policy claim."""
    from .protocol import SCHEMA, clean_frame, segment_key

    engine = SyntheticEngine(count, select)
    frame = engine.reset(character, seed)
    macros, current = [], None
    while frame["boundary"] != "terminal":
        key = segment_key(frame)
        if key != current:
            current = key
            macros.append({"steps": [], "phase": frame["public"]["phase"]})
        candidate = frame["legal"]["candidates"][0]
        macros[-1]["steps"].append(
            {
                "frame": clean_frame(frame),
                "candidate_ref": candidate["candidate_ref"],
                "forced": len(frame["legal"]["candidates"]) == 1,
            }
        )
        frame = engine.send(execution_command(frame, candidate["candidate_ref"]))
    return {
        "schema": SCHEMA,
        "source": "demonstration",
        "teacher_visibility": "public",
        "status": "complete",
        "victory": True,
        "ascension": 10,
        "character": character,
        "seed": str(seed),
        "run_id": character + ":" + str(seed),
        "contract": engine.contract,
        "macros": [m for m in macros if any(not s["forced"] for s in m["steps"])],
    }
