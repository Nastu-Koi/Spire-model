"""End-to-end contract tests for fixed-seed public-policy evaluation."""

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from combat_solver_cli import public_evaluate
from combat_solver_cli.public_evaluate import evaluate
from combat_solver_cli.tests.test_public_generate import FakeEngine, FakePlanner


class PublicEvaluationTests(unittest.TestCase):
    def test_reused_enemy_ref_is_not_comparable_but_player_delta_remains(self):
        class CompactedEnemies(FakeEngine):
            def _frame(self):
                frame = super()._frame()
                if self.version == 1:
                    frame["boundary"] = "decision"
                    frame["legal"]["candidates"] = [
                        {**item, "candidate_ref": item["decoder_slot_ref"]}
                        for item in frame["public"]["decoder_bank"]
                    ]
                    frame["public"].pop("outcome", None)
                    frame["events"] = []
                frame["public"]["entities"][0]["hp"] = 62 if self.version == 0 else 60
                if self.version == 0:
                    enemies = [
                        ("creature:1", "MONSTER.TWIG_SLIME_S", 5),
                        ("creature:2", "MONSTER.LEAF_SLIME_M", 35),
                    ]
                else:
                    enemies = [("creature:1", "MONSTER.LEAF_SLIME_M", 35)]
                frame["public"]["entities"].extend(
                    {
                        "ref": ref,
                        "entity_type": "enemy",
                        "content_id": content,
                        "hp": hp,
                        "block": 0,
                    }
                    for ref, content, hp in enemies
                )
                return frame

        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "evaluation"
            report = evaluate(
                self.manifest(),
                output,
                split="dev",
                engine_factory=lambda: CompactedEnemies(victory=False),
                planner_factory=FakePlanner,
            )
            row = json.loads(
                (output / "runs/000001/effects.jsonl").read_text().splitlines()[0]
            )
        self.assertEqual(row["observed_delta"]["player_hp"], -2)
        self.assertIsNone(row["observed_delta"]["enemy_hp"])
        self.assertIsNone(row["prediction_error"]["enemy_hp"])
        self.assertEqual(
            report["runs"][0]["effect_comparison"]["approximate"]["nonzero_error"], 1
        )

    def test_same_type_enemies_are_not_assumed_to_keep_ref_identity(self):
        class SameTypeEnemies(FakeEngine):
            def _frame(self):
                frame = super()._frame()
                health = (5, 10) if self.version == 0 else (10, 5)
                frame["public"]["entities"].extend(
                    {
                        "ref": f"creature:{index}",
                        "entity_type": "enemy",
                        "content_id": "MONSTER.TWIG_SLIME_S",
                        "hp": hp,
                        "block": 0,
                    }
                    for index, hp in enumerate(health, 1)
                )
                return frame

        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "evaluation"
            report = evaluate(
                self.manifest(),
                output,
                split="dev",
                engine_factory=lambda: SameTypeEnemies(victory=False),
                planner_factory=FakePlanner,
            )
            row = json.loads(
                (output / "runs/000001/effects.jsonl").read_text().splitlines()[0]
            )
        self.assertIsNone(row["observed_delta"]["enemy_hp"])
        self.assertEqual(
            report["runs"][0]["effect_comparison"]["approximate"]["zero_error"], 0
        )
        self.assertEqual(
            report["runs"][0]["effect_comparison"]["approximate"]["not_comparable"], 1
        )

    def test_zero_attempt_deadline_is_budget_stop_and_does_not_abort_matrix(self):
        manifest = self.manifest()
        manifest["budget"]["max_seconds_per_run"] = 1e-12
        manifest["profiles"].append(
            {
                "name": "candidate",
                "depth": 3,
                "beam_width": 16,
                "effect_profile": "second_wind",
            }
        )
        with tempfile.TemporaryDirectory() as temp:
            report = evaluate(
                manifest,
                Path(temp) / "evaluation",
                split="dev",
                engine_factory=FakeEngine,
                planner_factory=FakePlanner,
            )
        self.assertEqual(report["cases"], 2)
        self.assertEqual(report["counts"]["budget_stop"], 2)
        self.assertEqual([case["attempts"] for case in report["runs"]], [0, 0])
        self.assertEqual(
            [case["status"] for case in report["runs"]], ["limit", "limit"]
        )

    def test_report_cost_includes_repeatability_analysis(self):
        original = public_evaluate._first_difference

        def slow_compare(left, right):
            time.sleep(0.03)
            return original(left, right)

        with (
            tempfile.TemporaryDirectory() as temp,
            patch.object(
                public_evaluate, "_first_difference", side_effect=slow_compare
            ),
        ):
            report = evaluate(
                self.manifest(repetitions=2),
                Path(temp) / "evaluation",
                split="dev",
                engine_factory=FakeEngine,
                planner_factory=FakePlanner,
            )
        self.assertGreaterEqual(
            report["wall_seconds"],
            sum(run["wall_seconds"] for run in report["runs"]) + 0.03,
        )

    def test_final_action_effect_uses_native_post_observation(self):
        class FinalHpLoss(FakeEngine):
            def _frame(self):
                frame = super()._frame()
                if frame["boundary"] == "terminal":
                    frame["public"]["entities"][0]["hp"] = 62
                return frame

        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "evaluation"
            evaluate(
                self.manifest(),
                output,
                split="dev",
                engine_factory=lambda: FinalHpLoss(victory=False),
                planner_factory=FakePlanner,
            )
            rows = [
                json.loads(line)
                for line in (output / "runs/000001/effects.jsonl")
                .read_text()
                .splitlines()
            ]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["observed_delta"]["player_hp"], -8)

    def test_repeatability_includes_final_public_outcome_state(self):
        class TerminalVariation(FakeEngine):
            def _frame(self):
                frame = super()._frame()
                if frame["boundary"] == "terminal":
                    frame["public"]["entities"][0]["hp"] = 20 + self.serial % 2
                return frame

        with tempfile.TemporaryDirectory() as temp:
            report = evaluate(
                self.manifest(repetitions=2),
                Path(temp) / "evaluation",
                split="dev",
                engine_factory=lambda: TerminalVariation(victory=False),
                planner_factory=FakePlanner,
            )
        difference = report["repeatability"]["baseline"]["first_difference"]
        self.assertEqual(report["repeatability"]["baseline"]["status"], "diverged")
        self.assertIn("terminal_public", difference["fields"])

    def test_overlapping_seed_partitions_rejected_before_output_creation(self):
        manifest = self.manifest()
        manifest["seeds"]["holdout"] = ["defeat-a"]
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "evaluation"
            with self.assertRaisesRegex(ValueError, "overlap"):
                evaluate(manifest, output, split="dev", planner_factory=FakePlanner)
            self.assertFalse(output.exists())

    def test_error_accounting_and_zero_success_keep_cost_nullable(self):
        with tempfile.TemporaryDirectory() as temp:
            report = evaluate(
                self.manifest(),
                Path(temp) / "evaluation",
                split="dev",
                engine_factory=lambda: FakeEngine(native_error=True),
                planner_factory=FakePlanner,
            )
        self.assertEqual(report["counts"]["error"], 1)
        self.assertEqual(report["counts"]["native_error"], 1)
        self.assertEqual(report["counts"]["verified_victory"], 0)
        self.assertIsNone(report["wall_seconds_per_verified_victory"])
        self.assertGreaterEqual(
            report["wall_seconds"], report["runs"][0]["wall_seconds"]
        )

    def test_effect_report_compares_adjacent_real_public_observations(self):
        class TwoDecisionDefeat(FakeEngine):
            def _frame(self):
                frame = super()._frame()
                if self.version == 1:
                    frame["boundary"] = "decision"
                    frame["legal"]["candidates"] = list(frame["public"]["decoder_bank"])
                    for item in frame["legal"]["candidates"]:
                        item["candidate_ref"] = item["decoder_slot_ref"]
                    frame["public"].pop("outcome", None)
                    frame["events"] = []
                frame["public"]["entities"][0]["hp"] = 70 if self.version == 0 else 68
                frame["public"]["entities"].append(
                    {
                        "ref": "enemy:1",
                        "entity_type": "enemy",
                        "content_id": "MONSTER.TEST_ENEMY",
                        "hp": 10 if self.version == 0 else 5,
                        "max_hp": 10,
                        "block": 0,
                    }
                )
                return frame

        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "evaluation"
            report = evaluate(
                self.manifest(),
                output,
                split="dev",
                engine_factory=lambda: TwoDecisionDefeat(victory=False),
                planner_factory=FakePlanner,
            )
            rows = [
                json.loads(line)
                for line in (output / "runs/000001/effects.jsonl")
                .read_text()
                .splitlines()
            ]
        self.assertEqual(report["counts"]["defeat"], 1)
        self.assertGreaterEqual(
            report["runs"][0]["effect_comparison"]["approximate"]["nonzero_error"], 1
        )
        self.assertEqual(rows[0]["step"], 1)
        self.assertEqual(rows[0]["scope"], "next_public_boundary_same_combat_round")
        self.assertEqual(rows[0]["observed_delta"]["player_hp"], -2)
        self.assertEqual(rows[0]["observed_delta"]["enemy_hp"]["enemy:1"], -5)
        self.assertEqual(rows[0]["predicted_delta"]["player_hp"], 0)
        self.assertEqual(rows[0]["predicted_delta"]["enemy_hp"]["enemy:1"], 0)
        self.assertEqual(rows[0]["prediction_error"]["enemy_hp"]["enemy:1"], 5)

    def test_combat_net_hp_includes_end_turn_and_counts_potions(self):
        class ThreeObservations(FakeEngine):
            def _frame(self):
                frame = super()._frame()
                if self.version == 1:
                    frame["boundary"] = "decision"
                    frame["legal"]["candidates"] = [
                        {**item, "candidate_ref": item["decoder_slot_ref"]}
                        for item in frame["public"]["decoder_bank"]
                    ]
                    frame["public"].pop("outcome", None)
                    frame["events"] = []
                frame["public"]["entities"][0]["hp"] = (70, 60, 55)[self.version]
                if self.version == 0:
                    frame["legal"]["candidates"][0]["verb"] = "USE_POTION"
                    frame["public"]["decoder_bank"][0]["verb"] = "USE_POTION"
                return frame

        class ChoosePotionThenEnd(FakePlanner):
            def choose(self, frame):
                candidate = next(
                    item
                    for item in frame["legal"]["candidates"]
                    if item["verb"]
                    == (
                        "USE_POTION"
                        if frame["public"]["entities"][0]["hp"] == 70
                        else "END_TURN"
                    )
                )
                return candidate, {"method": "test"}

        with tempfile.TemporaryDirectory() as temp:
            report = evaluate(
                self.manifest(),
                Path(temp) / "evaluation",
                split="dev",
                engine_factory=lambda: ThreeObservations(victory=False),
                planner_factory=ChoosePotionThenEnd,
            )
        run = report["runs"][0]
        self.assertEqual(run["potions_used"], 1)
        self.assertEqual(run["combat_hp_net_loss_observed"], 15)
        self.assertEqual(run["combat_hp_observed_edges"], 2)
        self.assertEqual((run["final_act"], run["final_floor"]), (3, 16))

    def test_unknown_effect_prediction_never_counts_as_zero_error(self):
        class UnknownAction(FakeEngine):
            def _frame(self):
                frame = super()._frame()
                if frame["boundary"] == "decision":
                    frame["legal"]["candidates"][0]["verb"] = "WAIT"
                    frame["public"]["decoder_bank"][0]["verb"] = "WAIT"
                return frame

        with tempfile.TemporaryDirectory() as temp:
            report = evaluate(
                self.manifest(),
                Path(temp) / "evaluation",
                split="dev",
                engine_factory=lambda: UnknownAction(victory=False),
                planner_factory=FakePlanner,
            )
        comparison = report["runs"][0]["effect_comparison"]
        self.assertEqual(comparison["unknown"]["not_comparable"], 1)
        self.assertEqual(comparison["unknown"]["zero_error"], 0)

    def test_repeated_failure_reports_first_public_divergence(self):
        class NondeterministicDefeat(FakeEngine):
            resets = 0

            def reset(self, character, seed, ascension):
                type(self).resets += 1
                self.opening_hp = 70 + type(self).resets % 2
                return super().reset(character, seed, ascension)

            def _frame(self):
                frame = super()._frame()
                frame["public"]["entities"][0]["hp"] = self.opening_hp
                return frame

        manifest = self.manifest(repetitions=2)
        with tempfile.TemporaryDirectory() as temp:
            report = evaluate(
                manifest,
                Path(temp) / "evaluation",
                split="dev",
                engine_factory=lambda: NondeterministicDefeat(victory=False),
                planner_factory=FakePlanner,
            )
        self.assertEqual(report["counts"]["defeat"], 2)
        self.assertEqual(report["repeatability"]["baseline"]["status"], "diverged")
        self.assertEqual(
            report["repeatability"]["baseline"]["first_difference"]["step"], 1
        )
        self.assertIn(
            "public", report["repeatability"]["baseline"]["first_difference"]["fields"]
        )

    def test_sample_rng_is_reused_across_profiles_but_never_derived_from_seed(self):
        manifest = self.manifest(repetitions=2)
        manifest["profiles"].append(
            {
                "name": "candidate",
                "depth": 3,
                "beam_width": 16,
                "effect_profile": "second_wind",
            }
        )
        received = []

        class ConfigPlanner(FakePlanner):
            def __init__(self, *, depth, beam_width, sampling_seed, effect_profile):
                received.append((depth, beam_width, sampling_seed, effect_profile))

        with (
            tempfile.TemporaryDirectory() as temp,
            patch("combat_solver_cli.public_search.PublicPlanner", ConfigPlanner),
        ):
            report = evaluate(
                manifest,
                Path(temp) / "evaluation",
                split="dev",
                engine_factory=lambda: FakeEngine(victory=False),
            )
        self.assertEqual(report["cases"], 4)
        self.assertEqual([r[2] for r in received[:2]], [r[2] for r in received[2:]])
        self.assertEqual(received[0][2], received[1][2])
        self.assertTrue(all(type(r[2]) is int for r in received))
        self.assertEqual([r[3] for r in received], ["legacy"] * 2 + ["second_wind"] * 2)

    @staticmethod
    def manifest(*, repetitions=1):
        return {
            "schema": "public-evaluation-v1",
            "game_version": "test-engine",
            "seeds": {"dev": ["defeat-a"], "holdout": ["untouched-b"]},
            "profiles": [
                {
                    "name": "baseline",
                    "depth": 3,
                    "beam_width": 16,
                    "effect_profile": "legacy",
                }
            ],
            "repetitions": repetitions,
            "sampling_seed": 17,
            "budget": {"max_seconds_per_run": 10, "max_steps": 10},
            "stop": {"complete_all_cases": True, "abandon": False},
        }

    def test_fixed_seed_repetitions_run_to_completion_and_keep_frozen_manifest(self):
        manifest = {
            "schema": "public-evaluation-v1",
            "game_version": "test-engine",
            "seeds": {"dev": ["defeat-a"], "holdout": ["untouched-b"]},
            "profiles": [
                {
                    "name": "baseline",
                    "depth": 3,
                    "beam_width": 16,
                    "effect_profile": "legacy",
                }
            ],
            "repetitions": 2,
            "sampling_seed": 17,
            "budget": {"max_seconds_per_run": 10, "max_steps": 10},
            "stop": {"complete_all_cases": True, "abandon": False},
        }
        resets = []

        class RecordingEngine(FakeEngine):
            def reset(self, character, seed, ascension):
                resets.append(seed)
                return super().reset(character, seed, ascension)

        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "evaluation"
            report = evaluate(
                manifest,
                output,
                split="dev",
                engine_factory=lambda: RecordingEngine(),
                planner_factory=FakePlanner,
            )
            self.assertEqual(report["cases"], 2)
            self.assertEqual(report["counts"]["verified_victory"], 2)
            self.assertEqual(report["boss_pass_counts"], {"1": 2, "2": 2, "3": 2})
            self.assertEqual(report["runs"][0]["bosses"], [1, 2, 3])
            self.assertEqual(report["runs"][0]["potions_used"], 0)
            self.assertEqual(
                resets, ["defeat-a"] * 4
            )  # acting + strict replay per case
            self.assertEqual(
                json.loads((output / "manifest.json").read_text()), manifest
            )
            self.assertEqual(report["split"], "dev")
            self.assertEqual(report["game_version"], "test-engine")
            self.assertEqual(report["budget"], manifest["budget"])
            self.assertEqual(report["stop"], manifest["stop"])
            self.assertIn("generator_sha256", report["source_hashes"])
            self.assertEqual(report["repeatability"]["baseline"]["status"], "stable")
            self.assertNotIn("untouched-b", json.dumps(report))
            self.assertTrue(
                (output / "runs/000001/attempts/000001/evidence.json").is_file()
            )
            self.assertTrue(
                (output / "runs/000002/attempts/000001/evidence.json").is_file()
            )


if __name__ == "__main__":
    unittest.main()
