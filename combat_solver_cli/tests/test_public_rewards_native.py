"""Real reward offers and execution; debug fixtures never become demonstrations."""

import os
import unittest

from combat_solver_cli.public_search import PublicPlanner
from combat_solver_cli.tests.test_public_effects_native import start_combat
from model.engine import CliEngine
from model.protocol import execution_command


def reward_boundary(engine):
    state = start_combat(engine, ["BLUDGEON"] * 5, "native_reward_pair")
    planner = PublicPlanner()
    for _ in range(100):
        if state.get("type") != "decision_frame":
            raise AssertionError(state)
        if any(c["verb"] == "TAKE_CARD_REWARD" for c in state["legal"]["candidates"]):
            # Collect independent gold/etc. rewards first so the tested choice
            # is between the actual offered cards and leaving the reward menu.
            while True:
                other = next(
                    (
                        c
                        for c in state["legal"]["candidates"]
                        if c["verb"] == "TAKE_REWARD"
                    ),
                    None,
                )
                if other is None:
                    return state
                state = engine.send(execution_command(state, other["candidate_ref"]))
        action, _ = planner.choose(state)
        state = engine.send(execution_command(state, action["candidate_ref"]))
    raise AssertionError("Native fixture failed to reach a real reward offer")


@unittest.skipUnless(
    os.environ.get("STS2_NATIVE_REWARD_TESTS") == "1",
    "set STS2_NATIVE_REWARD_TESTS=1 to launch the native game",
)
class NativeDeckRewardTests(unittest.TestCase):
    def test_same_native_offer_is_taken_for_damage_gap_and_skipped_for_saturated_deck(
        self,
    ):
        offers = []
        for deck, expected in (
            (["DEFEND_IRONCLAD"] * 10, "TAKE_CARD_REWARD"),
            (["RAMPAGE"] * 10, "LEAVE_REWARDS"),
        ):
            with self.subTest(expected=expected), CliEngine(timeout=60) as engine:
                reward_boundary(engine)
                changed = engine.send({"cmd": "set_player", "deck": deck})
                self.assertEqual(changed["type"], "ok")
                state = engine.send({"cmd": "advance_to_boundary"})
                self.assertIs(state["contract"]["training_ready"], False)
                offered = sorted(
                    e["content_id"]
                    for e in state["public"]["entities"]
                    if e.get("zone") == "reward" and e.get("entity_type") == "card"
                )
                offers.append(offered)
                action, diagnostic = PublicPlanner(deck_rewards=True).choose(state)
                self.assertEqual(action["verb"], expected)
                self.assertEqual(
                    len(diagnostic["candidate_scores"]),
                    len(state["legal"]["candidates"]),
                )
                if expected == "TAKE_CARD_REWARD":
                    source = next(
                        e
                        for e in state["public"]["entities"]
                        if e.get("ref") in action["source_refs"]
                    )
                    self.assertEqual(source["content_id"], "CARD.RAMPAGE")
                after = engine.send(execution_command(state, action["candidate_ref"]))
                self.assertEqual(after["type"], "decision_frame")
                final_deck = [
                    e["content_id"]
                    for e in after["public"]["entities"]
                    if e.get("entity_type") == "card" and e.get("zone") == "deck"
                ]
                self.assertEqual(
                    len(final_deck), 11 if expected == "TAKE_CARD_REWARD" else 10
                )
                if expected == "TAKE_CARD_REWARD":
                    self.assertIn("CARD.RAMPAGE", final_deck)
                else:
                    self.assertEqual(after["public"]["phase"], "map")
        self.assertEqual(offers[0], offers[1])

    def test_native_upgrade_and_remove_follow_same_deck_need(self):
        from combat_solver_cli.tests.test_public_resources_native import (
            debug_room,
            source_entity,
        )

        planner = PublicPlanner(deck_rewards=True, route_resources=True)
        for room, deck, expected in (
            (
                "rest_site",
                ["STRIKE_IRONCLAD"] * 10 + ["DEFEND_IRONCLAD"],
                "CARD.DEFEND_IRONCLAD",
            ),
            (
                "shop",
                ["STRIKE_IRONCLAD"] * 10 + ["DEFEND_IRONCLAD", "SHAME"],
                "CARD.SHAME",
            ),
        ):
            with self.subTest(room=room), CliEngine(timeout=60) as engine:
                state = debug_room(engine, room, hp=80, deck=deck)
                action, _ = planner.choose(state)
                selection = engine.send(
                    execution_command(state, action["candidate_ref"])
                )
                self.assertEqual(selection["public"]["phase"], "card_select")
                chosen, diagnostics = planner.choose(selection)
                self.assertEqual(
                    source_entity(selection, chosen)["content_id"], expected
                )
                self.assertEqual(
                    len(diagnostics["candidate_scores"]),
                    len(selection["legal"]["candidates"]),
                )
                pending = engine.send(
                    execution_command(selection, chosen["candidate_ref"])
                )
                finish, _ = planner.choose(pending)
                self.assertEqual(finish["verb"], "FINISH_SELECTION")
                after = engine.send(execution_command(pending, finish["candidate_ref"]))
                deck_after = [
                    c
                    for c in after["public"]["entities"]
                    if c.get("zone") == "deck" and c.get("entity_type") == "card"
                ]
                if room == "shop":
                    self.assertNotIn(expected, [c["content_id"] for c in deck_after])
                else:
                    self.assertTrue(
                        next(c for c in deck_after if c["content_id"] == expected)[
                            "upgraded"
                        ]
                    )
