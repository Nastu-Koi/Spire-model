"""Synthetic protocol engine for integration tests; NOT a game simulator."""

from copy import deepcopy

from main.env.protocol import ProtocolError, execution_command


class SyntheticEngine:
    def __init__(self, count=10, select=5, *, forced_first=False):
        self.count, self.select, self.forced_first = count, select, forced_first

    def reset(self, character="Ironclad", seed="test", ascension=10):
        self.character, self.seed, self.ascension = character, str(seed), ascension
        self.commands = []
        self.version = 0
        self.enter("ordinary")
        return self.frame()

    def enter(self, encounter, act=1, kind="normal"):
        """Begin a fight: the selection, which the engine's event names."""
        self.encounter = encounter
        self.selected, self.stage = [], "selection"
        self.events = [
            {
                "type": "encounter_started",
                "encounter_id": encounter,
                "encounter": "ENCOUNTER.SYNTHETIC",
                "act": act,
                "kind": kind,
                "from_event": False,
            }
        ]

    def leave(self, candidate):
        """The decision after the fight: it ends the run, won by playing the card."""
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

    @property
    def contract(self):
        return {
            "adapter_version": "synthetic-v1",
            "observation_schema": "public-state-v7",
            "public_history_version": "public-history-v1",
            "action_schema": "candidate-v1",
            "fixed_ascension": getattr(self, "ascension", 10),
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
        if command["cmd"] == "write_continue_save":
            # The engine writes its continue save to the path it is given; a stand-in
            # with the fields a reader looks at.
            import json
            from pathlib import Path

            save = {"players": [{"current_hp": 70, "max_hp": 80, "gold": 99}],
                    "seed": self.seed, "state_version": self.version, "synthetic": True}
            Path(command["path"]).write_text(json.dumps(save))
            return {"type": "save_result", "success": True, "path": command["path"]}
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
                        "encounter_id": self.encounter,
                        "act": 1,
                        "kind": "normal",
                        "result": "victory",
                    }
                ]
            else:
                self.selected.append(int(candidate["decoder_slot_ref"].split(":")[1]))
        else:
            self.leave(candidate)
        return self.frame()

    def reseed(self, seed):
        """Another future (`CliEngine.reseed`): nothing here is drawn at random, so
        nothing changes but the record of the seeds given, in order."""
        self.reseeds = getattr(self, "reseeds", []) + [int(seed)]
        return {"type": "ok"}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


def demonstration(character="Ironclad", seed="demo", count=10, select=5):
    """Generate an explicit synthetic teacher trajectory, never an on-policy claim."""
    from main.env.protocol import SCHEMA, segment_key
    from main.encoding.frames import clean_frame

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
