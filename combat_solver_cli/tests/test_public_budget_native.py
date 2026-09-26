"""Opt-in native execution check for an adaptive planner root."""

import os
import unittest

from combat_solver_cli.public_search import PublicPlanner
from combat_solver_cli.tests.test_public_effects_native import start_combat
from model.engine import CliEngine
from model.protocol import clean_frame, execution_command


@unittest.skipUnless(
    os.environ.get("STS2_NATIVE_BUDGET_TESTS") == "1",
    "set STS2_NATIVE_BUDGET_TESTS=1 to launch the native game",
)
class NativeBudgetTests(unittest.TestCase):
    def test_chosen_native_combat_root_executes(self):
        with CliEngine() as engine:
            frame = start_combat(
                engine,
                ["STRIKE_IRONCLAD", "STRIKE_IRONCLAD", "DEFEND_IRONCLAD"] * 2,
                "native_adaptive_budget",
            )
            action, details = PublicPlanner(adaptive_budget=True).choose(
                clean_frame(frame)
            )
            refs = {c["candidate_ref"] for c in frame["legal"]["candidates"]}
            self.assertIn(action["candidate_ref"], refs)
            self.assertEqual(details["base_root_count"], len(refs))
            after = engine.send(execution_command(frame, action["candidate_ref"]))
            self.assertIn(after.get("type"), {"decision_frame", "terminal_frame"})


if __name__ == "__main__":
    unittest.main()
