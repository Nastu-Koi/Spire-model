"""The parallel entry keeps every opening and resumes only compatible trees."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from combat_solver_cli import mcts
from combat_solver_cli.client import InfrastructureError
from combat_solver_cli.search_support import SearchLimit
from combat_solver_cli.tests.test_mcts import Worker, candidate


class ParallelMctsTests(unittest.TestCase):
    def run_search(self, output, **options):
        return mcts.search(
            None,
            "Ironclad",
            "fixed",
            output,
            ascension=0,
            search_lanes=2,
            max_expansions=options.pop("max_expansions", 3),
            rollout_decisions=options.pop("rollout_decisions", 2),
            rollout_epsilon=0,
            **options,
        )

    def test_opening_preserves_every_candidate_with_one_expansion(self):
        class WorkerWithAbandon(Worker):
            def restore(self, prefix):
                super().restore(prefix)
                if self.frame["boundary"] == "decision":
                    abandon = candidate("quit")
                    abandon["verb"] = "ABANDON_RUN"
                    self.frame["legal"]["candidates"].append(abandon)

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(mcts, "ReplayWorker", WorkerWithAbandon),
            patch.object(mcts, "preference", return_value=0),
        ):
            output = Path(directory) / "first"
            result = self.run_search(output, max_expansions=1)
            self.assertEqual(result["status"], "budget_exhausted")
            self.assertEqual(result["expanded"], 1)
            self.assertEqual(
                json.loads((output / "tree.json").read_text())["schema"],
                "mcts-lanes-v2",
            )
            actions = []
            for lane in range(2):
                checkpoint = json.loads(
                    (output / "lane-inputs" / f"lane-{lane}.json").read_text()
                )
                actions.extend(
                    node["pending"]["source_refs"][0]
                    for node in checkpoint["nodes"]
                    if node["pending"] is not None
                )
            self.assertCountEqual(actions, ["bad", "good", "quit"])

    def test_parallel_resume_from_opening_finds_verified_win(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(mcts, "ReplayWorker", Worker),
            patch.object(mcts, "preference", return_value=0),
            patch("combat_solver_cli.trajectory.verify_and_export") as verify,
        ):
            root = Path(directory)
            first = self.run_search(root / "first", max_expansions=1)
            self.assertEqual(first["status"], "budget_exhausted")
            resumed = self.run_search(
                root / "resumed",
                resume=root / "first" / "tree.json",
                max_expansions=2,
            )
            self.assertEqual(resumed["status"], "verified_victory")
            self.assertLessEqual(resumed["expanded"], 2)
            verify.assert_called_once()
            winner = json.loads((root / "resumed" / "winning_prefix.json").read_text())
            self.assertEqual(winner["records"][-1]["action"]["source_refs"], ["good"])

    def test_one_expansion_resumes_rotate_across_live_lanes(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(mcts, "ReplayWorker", Worker),
            patch.object(mcts, "preference", return_value=0),
            patch("combat_solver_cli.trajectory.verify_and_export"),
        ):
            root = Path(directory)
            result = self.run_search(root / "initial", max_expansions=1)
            self.assertEqual(result["status"], "budget_exhausted")
            previous = root / "initial" / "tree.json"
            for attempt in range(1, 4):
                output = root / f"resume-{attempt}"
                result = self.run_search(
                    output,
                    resume=previous,
                    max_expansions=1,
                )
                self.assertLessEqual(result["expanded"], 1)
                if result["status"] == "verified_victory":
                    break
                previous = output / "tree.json"
            self.assertEqual(result["status"], "verified_victory")
            self.assertLessEqual(attempt, 2)

    def test_interrupted_opening_has_resumable_top_level_checkpoint(self):
        class InterruptedOpening(Worker):
            failure = None

            def restore(self, prefix):
                if self.__class__.failure is not None:
                    failure = self.__class__.failure
                    self.__class__.failure = None
                    raise failure
                return super().restore(prefix)

        cases = (
            (SearchLimit("time budget exhausted"), "budget_exhausted"),
            (SearchLimit("STOP requested"), "stopped"),
            (InfrastructureError("worker disconnected"), "infrastructure_error"),
        )
        for failure, expected in cases:
            with (
                self.subTest(expected=expected),
                tempfile.TemporaryDirectory() as directory,
                patch.object(mcts, "ReplayWorker", InterruptedOpening),
                patch.object(mcts, "preference", return_value=0),
                patch("combat_solver_cli.trajectory.verify_and_export") as verify,
            ):
                root = Path(directory)
                InterruptedOpening.failure = failure
                first = self.run_search(root / "first", max_expansions=3)
                self.assertEqual(first["status"], expected)
                self.assertEqual(first["expanded"], 1)
                manifest = json.loads((root / "first" / "tree.json").read_text())
                self.assertEqual(manifest["stage"], "opening")
                checkpoint = root / "first" / manifest["opening_checkpoint"]
                self.assertFalse(
                    json.loads(checkpoint.read_text())["nodes"][0]["expanded"]
                )
                resumed = self.run_search(
                    root / "resumed",
                    resume=root / "first" / "tree.json",
                    max_expansions=3,
                )
                self.assertEqual(resumed["status"], "verified_victory")
                self.assertLessEqual(resumed["expanded"], 3)
                verify.assert_called_once()

    def test_old_lane_manifest_is_rejected_explicitly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = root / "old.json"
            old.write_text(json.dumps({"schema": "mcts-lanes-v1"}))
            with self.assertRaisesRegex(ValueError, "incompatible"):
                self.run_search(root / "new", resume=old)

    def test_old_astar_frontier_is_rejected_explicitly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = root / "old.json"
            old.write_text(json.dumps({"schema": "astar-frontier-v2"}))
            with self.assertRaisesRegex(ValueError, "incompatible"):
                mcts.search(
                    None, "Ironclad", "fixed", root / "new", ascension=0, resume=old
                )

    def test_stop_after_opening_keeps_resumable_manifest(self):
        from combat_solver_cli import mcts_lanes

        opening = mcts_lanes._opening_shards

        def stopped(*args, **kwargs):
            warm, paths = opening(*args, **kwargs)
            args[3].joinpath("STOP").touch()
            return warm, paths

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(mcts, "ReplayWorker", Worker),
            patch.object(mcts, "preference", return_value=0),
            patch.object(mcts_lanes, "_opening_shards", side_effect=stopped),
        ):
            output = Path(directory) / "stopped"
            result = self.run_search(output)
            self.assertEqual(result["status"], "stopped")
            self.assertTrue((output / "tree.json").is_file())

    def test_lane_exception_recovers_input_checkpoint(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(mcts, "ReplayWorker", Worker),
            patch.object(mcts, "preference", return_value=0),
        ):
            root = Path(directory)
            self.run_search(root / "first", max_expansions=1)
            real_search = mcts.search
            with patch.object(mcts, "search", side_effect=RuntimeError("lane failed")):
                result = real_search(
                    None,
                    "Ironclad",
                    "fixed",
                    root / "error",
                    ascension=0,
                    search_lanes=2,
                    max_expansions=2,
                    rollout_decisions=2,
                    rollout_epsilon=0,
                    resume=root / "first" / "tree.json",
                )
            self.assertEqual(result["status"], "infrastructure_error")
            self.assertEqual(result["recovered_lanes"], [0, 1])
            manifest = json.loads((root / "error" / "tree.json").read_text())
            self.assertEqual(len(manifest["checkpoints"]), 2)
            with patch("combat_solver_cli.trajectory.verify_and_export"):
                resumed = self.run_search(
                    root / "retry",
                    resume=root / "error" / "tree.json",
                    max_expansions=2,
                )
            self.assertEqual(resumed["status"], "verified_victory")


if __name__ == "__main__":
    unittest.main()
