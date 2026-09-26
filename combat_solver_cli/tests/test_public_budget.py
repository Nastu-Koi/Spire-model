"""Observable checks for optional per-decision public search budgets."""

import copy
import unittest
from unittest.mock import patch

from combat_solver_cli.public_search import PublicPlanner
from combat_solver_cli.tests.test_public_resources import frame as outside_frame
from combat_solver_cli.tests.test_public_resources import node
from combat_solver_cli.tests.test_public_search import candidate, card, combat_frame


def choice_frame(*, hp=50, incoming=0):
    hand = [
        card("strike:1", "CARD.STRIKE_IRONCLAD", damage=6),
        card("strike:2", "CARD.STRIKE_IRONCLAD", damage=6),
        card("defend", "CARD.DEFEND_IRONCLAD", block=5, card_type="Skill"),
    ]
    return combat_frame(hp=hp, incoming=incoming, hand=hand)


class PublicBudgetTests(unittest.TestCase):
    def test_fixed_path_has_exact_score_and_choice_parity(self):
        frame = choice_frame()
        before, old = PublicPlanner().choose(frame)
        after, new = PublicPlanner(adaptive_budget=False).choose(frame)
        self.assertEqual(before, after)
        self.assertEqual(old["candidate_scores"], new["candidate_scores"])
        self.assertEqual(new["adaptive_budget"], False)

    def test_danger_and_close_scores_receive_more_budget_than_simple_choice(self):
        simple = combat_frame(
            hand=[card("strike", "CARD.STRIKE_IRONCLAD", damage=6)], enemy_hp=6
        )
        danger = choice_frame(hp=8, incoming=12)
        _, simple_details = PublicPlanner(adaptive_budget=True).choose(simple)
        _, danger_details = PublicPlanner(adaptive_budget=True).choose(danger)
        self.assertEqual(simple_details["budget_tier"], "simple")
        self.assertIn(danger_details["budget_tier"], {"danger", "close"})
        self.assertGreater(
            danger_details["extra_evaluations"], simple_details["extra_evaluations"]
        )
        self.assertEqual(
            len(danger_details["base_candidate_scores"]),
            len(danger["legal"]["candidates"]),
        )
        close = combat_frame(
            hand=[
                card("one", "CARD.STRIKE_IRONCLAD", damage=6),
                card("two", "CARD.STRIKE_IRONCLAD", damage=6),
            ]
        )
        _, close_details = PublicPlanner(adaptive_budget=True).choose(close)
        self.assertEqual(close_details["budget_tier"], "close")

    def test_tiny_budget_still_scores_every_root_and_preserves_original_frame(self):
        frame = choice_frame()
        frame["public"]["hidden_deck_order"] = ["secret"]
        original = copy.deepcopy(frame)
        action, details = PublicPlanner(
            adaptive_budget=True, decision_node_limit=1
        ).choose(frame)
        self.assertIn(action, frame["legal"]["candidates"])
        self.assertEqual(frame, original)
        self.assertEqual(details["base_root_count"], len(frame["legal"]["candidates"]))
        self.assertEqual(
            len(details["candidate_scores"]), len(frame["legal"]["candidates"])
        )
        self.assertLessEqual(details["extra_evaluations"], 1)

    def test_extra_work_is_round_robin_across_live_roots(self):
        frame = choice_frame(hp=8, incoming=12)
        _, details = PublicPlanner(
            adaptive_budget=True, decision_node_limit=4, decision_time_ms=1000
        ).choose(frame)
        self.assertEqual(sum(details["extra_evaluations_by_root"]), 4)
        self.assertGreaterEqual(
            sum(count > 0 for count in details["extra_evaluations_by_root"]), 2
        )

    def test_future_samples_cannot_bypass_extra_transition_limit(self):
        frame = combat_frame(
            hp=30,
            incoming=18,
            hand=[card("hand", "CARD.STRIKE_IRONCLAD", damage=6)],
            extra=[
                card(f"draw:{i}", "CARD.STRIKE_IRONCLAD", damage=6, zone="draw_pile")
                for i in range(5)
            ],
        )
        _, details = PublicPlanner(
            adaptive_budget=True,
            depth=1,
            draw_samples=8,
            decision_node_limit=10,
            decision_time_ms=1000,
        ).choose(frame)
        self.assertLessEqual(details["extra_evaluations"], 10)
        self.assertEqual(
            sum(details["extra_evaluations_by_root"]), details["extra_evaluations"]
        )
        self.assertGreater(details["unfinished_future_samples"], 0)
        self.assertGreaterEqual(
            sum(count > 0 for count in details["extra_evaluations_by_root"]), 2
        )

    def test_unknown_root_remains_scored_and_public_equivalence_holds(self):
        frame = choice_frame()
        frame["legal"]["candidates"].append(candidate("unknown", "UNKNOWN_ACTION"))
        other = copy.deepcopy(frame)
        other["public"]["entities"].reverse()
        other["public"]["hidden_deck_order"] = ["secret"]
        planner = PublicPlanner(adaptive_budget=True, decision_node_limit=12)
        first, left = planner.choose(frame)
        second, right = planner.choose(other)
        self.assertEqual(first["candidate_ref"], second["candidate_ref"])
        self.assertEqual(left["candidate_scores"], right["candidate_scores"])
        self.assertIn("unknown", {x["candidate_ref"] for x in left["candidate_scores"]})

    def test_wall_clock_exhaustion_after_base_scores_is_reported(self):
        frame = choice_frame()
        with patch(
            "combat_solver_cli.public_search.time.monotonic", side_effect=[0, 1, 1, 1]
        ):
            _, details = PublicPlanner(adaptive_budget=True, decision_time_ms=1).choose(
                frame
            )
        self.assertEqual(details["base_root_count"], len(frame["legal"]["candidates"]))
        self.assertEqual(details["extra_evaluations"], 0)
        self.assertEqual(details["budget_exhausted"], "time")
        self.assertGreater(details["budget_overrun_seconds"], 0)

    def test_outside_combat_base_time_overrun_is_reported(self):
        state = outside_frame(
            "map",
            [node("rest", "RestSite")],
            [candidate("rest", "MOVE_TO_NODE", "rest")],
        )
        with patch(
            "combat_solver_cli.public_search.time.monotonic", side_effect=[0, 0.25]
        ):
            _, details = PublicPlanner(
                adaptive_budget=True, route_resources=True, decision_time_ms=1
            ).choose(state)
        self.assertEqual(details["base_root_count"], 1)
        self.assertEqual(details["extra_evaluations"], 0)
        self.assertEqual(details["base_elapsed_seconds"], 0.25)
        self.assertEqual(details["elapsed_seconds"], 0.25)
        self.assertAlmostEqual(details["budget_overrun_seconds"], 0.249)

    def test_invalid_limits_fail_at_configuration(self):
        for kwargs in (
            {"adaptive_budget": 1},
            {"decision_node_limit": 0},
            {"decision_node_limit": True},
            {"decision_time_ms": float("nan")},
            {"decision_time_ms": -1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                PublicPlanner(**kwargs)  # pyright: ignore[reportArgumentType]


if __name__ == "__main__":
    unittest.main()
