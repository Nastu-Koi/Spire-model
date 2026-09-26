"""Contract tests for public-policy generation; no local game assembly required."""

import json
import tempfile
import time
import unittest
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

from combat_solver_cli import public_generate
from combat_solver_cli.public_generate import generate, run_attempt
from model.protocol import execution_command


class FakePlanner:
    def abandon_reason(self, frame):
        assert set(frame) == {"boundary", "public", "legal"}

    def choose(self, frame):
        assert set(frame) == {"boundary", "public", "legal"}
        return frame["legal"]["candidates"][0], {"method": "test"}


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


class PublicGenerateTests(unittest.TestCase):
    def test_deadline_crossed_in_planning_retains_cost_without_executing(self):
        clock = [0.0]
        engines = []

        class SlowPlanner(FakePlanner):
            def choose(self, frame):
                clock[0] = 2.0
                return super().choose(frame)

        def engine_factory():
            engine = FakeEngine(victory=False)
            engines.append(engine)
            return engine

        with (
            tempfile.TemporaryDirectory() as temp,
            patch(
                "combat_solver_cli.public_generate.time.monotonic",
                side_effect=lambda: clock[0],
            ),
        ):
            output = Path(temp) / "run"
            report = generate(
                output,
                workers=1,
                max_attempts=1,
                max_seconds=1,
                planner_factory=SlowPlanner,
                engine_factory=engine_factory,
            )
            result = json.loads((output / "attempts/000001/result.json").read_text())
        self.assertEqual(report["outcome_counts"]["limit"], 1)
        self.assertEqual(engines[0].version, 0)
        self.assertEqual(result["prefix"]["records"], [])
        self.assertEqual(
            result["prefix"]["pending_decision"]["diagnostics"], {"method": "test"}
        )
        self.assertEqual(result["outcome"]["planning_seconds"], 2.0)
        self.assertEqual(report["budget_overrun_seconds"], 1.0)
        self.assertEqual(report["unfinished_attempt_seconds"], 2.0)

    def test_target_stops_other_workers_at_a_decision_boundary(self):
        class MixedEngine(FakeEngine):
            def reset(self, character, seed, ascension):
                self.loop = seed == "loop"
                return super().reset(character, seed, ascension)

            def send(self, command):
                if self.loop:
                    time.sleep(0.01)
                return super().send(command)

        seeds = iter(["win", "loop"])
        with tempfile.TemporaryDirectory() as temp:
            summary = generate(
                Path(temp) / "parallel",
                workers=2,
                max_attempts=2,
                max_seconds=10,
                planner_factory=FakePlanner,
                engine_factory=MixedEngine,
                seed_factory=lambda: next(seeds),
            )
        self.assertEqual(summary["verified_victories"], 1)
        self.assertEqual(summary["attempts"], 2)
        stopped = [r for r in summary["results"] if r["seed"] == "loop"]
        self.assertEqual(stopped[0]["reason"], "coordinator_stop")
        self.assertLess(summary["wall_seconds"], 5)

    def test_victory_requires_fresh_matching_replay_and_exports_demonstration(self):
        engines = []

        def factory():
            engine = FakeEngine()
            engines.append(engine)
            return engine

        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "batch"
            summary = generate(
                output,
                target_trajectories=1,
                workers=1,
                max_attempts=1,
                max_seconds=10,
                planner_factory=FakePlanner,
                engine_factory=factory,
                seed_factory=lambda: "seed-a",
            )
            self.assertEqual(len(engines), 2)
            self.assertEqual(summary["verified_victories"], 1)
            self.assertGreaterEqual(summary["wall_seconds"], 0)
            self.assertIn("generator_sha256", summary["source_hashes"])
            self.assertIn("effects_sha256", summary["source_hashes"])
            self.assertIn("representation_sha256", summary["source_hashes"])
            run = json.loads((output / "accepted.jsonl").read_text())
            self.assertEqual(
                (run["source"], run["teacher_visibility"]), ("demonstration", "public")
            )
            self.assertEqual(run["provenance"]["bosses"], [1, 2, 3])
            self.assertEqual(len(run["macros"]), 1)
            evidence = json.loads(
                (output / "attempts/000001/evidence.json").read_text()
            )
            self.assertNotEqual(evidence["source_run_id"], evidence["replay_run_id"])
            self.assertEqual(run["run_id"], evidence["replay_run_id"])
            self.assertEqual(evidence["source_ledger"]["bosses"], [1, 2, 3])
            self.assertTrue((output / "accepted-parts/000001.jsonl").is_file())
            with self.assertRaises(FileExistsError):
                generate(
                    output,
                    max_attempts=1,
                    planner_factory=FakePlanner,
                    engine_factory=factory,
                )

    def test_defeat_is_recorded_and_does_not_trigger_replay(self):
        engines = []

        def factory():
            engine = FakeEngine(victory=False)
            engines.append(engine)
            return engine

        result, run = run_attempt(
            "seed-b",
            time.monotonic() + 10,
            5,
            planner_factory=FakePlanner,
            engine_factory=factory,
        )
        self.assertIsNone(run)
        self.assertEqual(result["status"], "defeat")
        self.assertEqual(len(engines), 1)

    def test_terminal_victory_without_three_bosses_is_error(self):
        result, run = run_attempt(
            "seed-c",
            time.monotonic() + 10,
            5,
            planner_factory=FakePlanner,
            engine_factory=lambda: FakeEngine(milestones=False),
        )
        self.assertIsNone(run)
        self.assertEqual(result["status"], "error")
        self.assertIn("boss milestones", result["reason"])

    def test_replay_divergence_is_error(self):
        engines = []

        def factory():
            engine = FakeEngine(divergent=bool(engines))
            engines.append(engine)
            return engine

        result, run = run_attempt(
            "seed-d",
            time.monotonic() + 10,
            5,
            planner_factory=FakePlanner,
            engine_factory=factory,
        )
        self.assertIsNone(run)
        self.assertEqual(result["status"], "error")
        self.assertIn("diverged", result["reason"])
        self.assertEqual(result["error_category"], "replay_rejected")

    def test_native_and_protocol_errors_are_separate_from_defeat(self):
        from model.protocol import ProtocolError

        with tempfile.TemporaryDirectory() as temp:
            native, _ = run_attempt(
                "native",
                time.monotonic() + 10,
                5,
                planner_factory=FakePlanner,
                engine_factory=lambda: FakeEngine(native_error=True),
                attempt_dir=Path(temp),
            )
        self.assertEqual(native["error_category"], "native_error")

        class BrokenProtocol(FakeEngine):
            def send(self, command):
                raise ProtocolError("invalid decision frame")

        protocol, _ = run_attempt(
            "protocol",
            time.monotonic() + 10,
            5,
            planner_factory=FakePlanner,
            engine_factory=BrokenProtocol,
        )
        self.assertEqual(protocol["error_category"], "protocol_error")

        class CrashedNative(FakeEngine):
            def reset(self, character, seed, ascension):
                raise ProtocolError("Engine exited (1)")

        crashed, _ = run_attempt(
            "crashed",
            time.monotonic() + 10,
            5,
            planner_factory=FakePlanner,
            engine_factory=CrashedNative,
        )
        self.assertEqual(crashed["error_category"], "native_error")

    def test_step_limit_and_native_error_are_not_defeats(self):
        limited, _ = run_attempt(
            "seed-e",
            time.monotonic() + 10,
            1,
            planner_factory=FakePlanner,
            engine_factory=lambda: FakeEngine(loop=True),
        )
        self.assertEqual(
            (limited["status"], limited["reason"]), ("limit", "step_limit")
        )
        with tempfile.TemporaryDirectory() as temp:
            fault, _ = run_attempt(
                "seed-f",
                time.monotonic() + 10,
                5,
                planner_factory=FakePlanner,
                engine_factory=lambda: FakeEngine(victory=False, native_error=True),
                attempt_dir=Path(temp),
            )
            self.assertEqual(fault["status"], "error")
            progress = json.loads((Path(temp) / "result.json").read_text())["prefix"]
            self.assertIn(
                "[ERROR] simulated engine fault", progress["acting_stderr_tail"]
            )
            self.assertIsNone(progress["verification_stderr_tail"])

    def test_verification_error_preserves_verifier_stderr_tail(self):
        engines = []

        def factory():
            engine = FakeEngine(native_error=bool(engines))
            engines.append(engine)
            return engine

        with tempfile.TemporaryDirectory() as temp:
            result, run = run_attempt(
                "seed-verifier-fault",
                time.monotonic() + 10,
                5,
                planner_factory=FakePlanner,
                engine_factory=factory,
                attempt_dir=Path(temp),
            )
            self.assertIsNone(run)
            self.assertEqual(result["status"], "error")
            progress = json.loads((Path(temp) / "result.json").read_text())["prefix"]
            self.assertEqual(progress["acting_stderr_tail"], "")
            self.assertIn(
                "[ERROR] simulated engine fault", progress["verification_stderr_tail"]
            )

    def test_failed_batch_keeps_prefix_and_nullable_rate(self):
        class SparseTerminalEngine(FakeEngine):
            def _frame(self):
                frame = super()._frame()
                if frame["boundary"] == "terminal":
                    frame["public"] = {"outcome": frame["public"]["outcome"]}
                return frame

        class SlowPlanner(FakePlanner):
            def choose(self, frame):
                time.sleep(0.002)
                return super().choose(frame)

        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "batch"
            summary = generate(
                output,
                workers=1,
                max_attempts=1,
                max_seconds=10,
                planner_factory=SlowPlanner,
                engine_factory=lambda: SparseTerminalEngine(victory=False),
                seed_factory=lambda: "seed-g",
            )
            self.assertEqual(summary["status"], "incomplete")
            self.assertIsNone(summary["wall_seconds_per_verified_victory"])
            self.assertEqual((output / "accepted.jsonl").read_text(), "")
            result = json.loads((output / "attempts/000001/result.json").read_text())
            self.assertEqual(result["outcome"]["status"], "defeat")
            self.assertEqual(len(result["prefix"]["records"]), 1)
            self.assertEqual(len(result["prefix"]["diagnostics"]), 1)
            self.assertEqual(
                result["prefix"]["last_frame"]["public"].keys(), {"outcome"}
            )
            self.assertEqual(
                result["prefix"]["last_decision_frame"]["public"]["entities"][0]["hp"],
                70,
            )
            journal = json.loads(
                (output / "attempts/000001/decisions.jsonl").read_text()
            )
            self.assertEqual(journal["frame"]["routing"]["decision_id"], "d0")
            self.assertEqual(journal["frame"]["public"]["entities"][0]["hp"], 70)
            self.assertEqual(
                journal["record"]["before_hash"],
                result["prefix"]["records"][0]["before_hash"],
            )
            self.assertGreaterEqual(result["outcome"]["planning_seconds"], 0.002)
            self.assertEqual(result["outcome"]["verification_seconds"], 0.0)

    def test_abandon_keeps_earlier_decisions(self):
        class AbandonAfterOne(FakePlanner):
            def __init__(self):
                self.calls = 0

            def abandon_reason(self, frame):
                self.calls += 1
                return "low_chance" if self.calls > 1 else None

        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "batch"
            summary = generate(
                output,
                workers=1,
                max_attempts=1,
                max_seconds=10,
                planner_factory=AbandonAfterOne,
                engine_factory=lambda: FakeEngine(loop=True),
            )
            self.assertEqual(summary["results"][0]["status"], "abandoned")
            result = json.loads((output / "attempts/000001/result.json").read_text())
            self.assertEqual(len(result["prefix"]["records"]), 1)
            self.assertEqual(
                len(
                    (output / "attempts/000001/decisions.jsonl")
                    .read_text()
                    .splitlines()
                ),
                1,
            )

    def test_wall_clock_includes_accepted_export(self):
        original = public_generate._atomic_jsonl

        def slow_write(path, values):
            time.sleep(0.02)
            return original(path, values)

        with (
            tempfile.TemporaryDirectory() as temp,
            patch.object(public_generate, "_atomic_jsonl", side_effect=slow_write),
        ):
            summary = generate(
                Path(temp) / "batch",
                workers=1,
                max_attempts=1,
                max_seconds=10,
                planner_factory=FakePlanner,
                engine_factory=FakeEngine,
            )
            self.assertGreaterEqual(summary["wall_seconds"], 0.04)

    def test_engine_error_stops_batch_without_retrying_seeds(self):
        calls = []

        def broken():
            calls.append(1)
            raise FileNotFoundError("engine unavailable")

        with tempfile.TemporaryDirectory() as temp:
            summary = generate(
                Path(temp) / "batch",
                workers=1,
                max_attempts=5,
                max_seconds=10,
                planner_factory=FakePlanner,
                engine_factory=broken,
            )
            self.assertEqual(summary["status"], "error")
            self.assertEqual(summary["attempts"], 1)
            self.assertEqual(len(calls), 1)

    def test_invalid_budget_rejected_before_creating_output(self):
        with tempfile.TemporaryDirectory() as temp:
            for options in (
                {"workers": 0},
                {"max_seconds": float("nan")},
                {"max_seconds": float("inf")},
                {"max_seconds": True},
            ):
                output = Path(temp) / "batch"
                with self.assertRaises(ValueError):
                    generate(output, planner_factory=FakePlanner, **options)
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
