"""Opt-in native checks for hypothetical actions and real root execution."""

import os
import unittest

from combat_solver_cli.public_effects import CombatModel
from combat_solver_cli.public_search import PublicPlanner
from combat_solver_cli.tests.test_public_effects_native import (
    card_candidate,
    player_entity,
    power_stack,
    start_combat,
)
from model.engine import CliEngine
from model.protocol import clean_frame, execution_command


@unittest.skipUnless(
    os.environ.get("STS2_NATIVE_LOOKAHEAD_TESTS") == "1",
    "set STS2_NATIVE_LOOKAHEAD_TESTS=1 to launch the native game",
)
class NativeLookaheadTests(unittest.TestCase):
    def test_unmodeled_enlightenment_cost_change_stops_at_native_boundary(self):
        with CliEngine() as engine:
            frame = start_combat(
                engine,
                [
                    "ENLIGHTENMENT",
                    "BLUDGEON",
                    "STRIKE_IRONCLAD",
                    "STRIKE_IRONCLAD",
                    "STRIKE_IRONCLAD",
                ],
                "lookahead_enlightenment",
            )
            for _ in range(2):
                strike = card_candidate(frame, "CARD.STRIKE_IRONCLAD")
                frame = engine.send(execution_command(frame, strike["candidate_ref"]))
                self.assertEqual(frame.get("type"), "decision_frame", frame)
            self.assertEqual(player_entity(frame)["energy"], 1)
            self.assertFalse(
                any(
                    c["verb"] == "PLAY_CARD"
                    and any(
                        e.get("content_id") == "CARD.BLUDGEON"
                        for e in frame["public"]["entities"]
                        if e.get("ref") in c.get("source_refs", [])
                    )
                    for c in frame["legal"]["candidates"]
                )
            )
            enlightenment = card_candidate(frame, "CARD.ENLIGHTENMENT")
            self.assertIsNone(
                CombatModel(frame, dynamic_legality=True).predict(enlightenment).state
            )
            chosen, details = PublicPlanner(dynamic_legality=True).choose(
                clean_frame(frame)
            )
            self.assertIn(
                chosen["candidate_ref"],
                {c["candidate_ref"] for c in frame["legal"]["candidates"]},
            )
            self.assertEqual(
                len(details["candidate_scores"]), len(frame["legal"]["candidates"])
            )
            after = engine.send(
                execution_command(frame, enlightenment["candidate_ref"])
            )
            self.assertEqual(after.get("type"), "decision_frame", after)
            bludgeon = card_candidate(after, "CARD.BLUDGEON")
            source = bludgeon["source_refs"][0]
            self.assertEqual(
                next(
                    e["cost"]
                    for e in after["public"]["entities"]
                    if e.get("ref") == source
                ),
                1,
            )
            replanned, _ = PublicPlanner(dynamic_legality=True).choose(
                clean_frame(after)
            )
            self.assertEqual(replanned["source_refs"], [source])
            played = engine.send(execution_command(after, replanned["candidate_ref"]))
            self.assertIn(played.get("type"), {"decision_frame", "terminal_frame"})

    def test_sampled_future_selects_growth_and_executes_native_root(self):
        with CliEngine() as engine:
            frame = start_combat(
                engine,
                [
                    "INFLAME",
                    "STRIKE_IRONCLAD",
                    "STRIKE_IRONCLAD",
                    "STRIKE_IRONCLAD",
                    "STRIKE_IRONCLAD",
                ],
                "lookahead_growth_native",
            )
            planner = PublicPlanner(
                depth=2,
                dynamic_legality=True,
                draw_samples=2,
                future_value=True,
                sampling_seed=17,
            )
            chosen, details = planner.choose(clean_frame(frame))
            self.assertEqual(chosen["verb"], "PLAY_CARD")
            source = chosen["source_refs"][0]
            self.assertEqual(
                next(
                    e["content_id"]
                    for e in frame["public"]["entities"]
                    if e.get("ref") == source
                ),
                "CARD.INFLAME",
            )
            self.assertEqual(details["draw_samples"], 2)
            predicted = CombatModel(frame, dynamic_legality=True).predict(chosen)
            after = engine.send(execution_command(frame, chosen["candidate_ref"]))
            self.assertEqual(after.get("type"), "decision_frame", after)
            self.assertEqual(player_entity(after)["hp"], predicted.state.hp)
            self.assertEqual(player_entity(after)["block"], predicted.state.block)
            self.assertEqual(
                power_stack(after, "POWER.STRENGTH_POWER", "player"),
                predicted.state.strength,
            )

    def test_energy_potion_unlocks_bludgeon_and_planner_sends_native_root(self):
        with CliEngine() as engine:
            frame = start_combat(
                engine,
                [
                    "DEFEND_IRONCLAD",
                    "BLUDGEON",
                    "STRIKE_IRONCLAD",
                    "STRIKE_IRONCLAD",
                    "STRIKE_IRONCLAD",
                ],
                "lookahead_potion_bludgeon",
                potions=("ENERGY_POTION",),
            )
            defend = card_candidate(frame, "CARD.DEFEND_IRONCLAD")
            frame = engine.send(execution_command(frame, defend["candidate_ref"]))
            self.assertEqual(frame.get("type"), "decision_frame", frame)
            self.assertFalse(
                any(
                    c["verb"] == "PLAY_CARD"
                    and any(
                        e.get("content_id") == "CARD.BLUDGEON"
                        for e in frame["public"]["entities"]
                        if e.get("ref") in c.get("source_refs", [])
                    )
                    for c in frame["legal"]["candidates"]
                )
            )
            planner = PublicPlanner(depth=2, dynamic_legality=True)
            chosen, details = planner.choose(clean_frame(frame))
            self.assertEqual(chosen["verb"], "USE_POTION")
            self.assertIn(
                chosen["candidate_ref"],
                {c["candidate_ref"] for c in frame["legal"]["candidates"]},
            )
            self.assertEqual(
                len(details["candidate_scores"]), len(frame["legal"]["candidates"])
            )
            predicted = CombatModel(frame).predict(chosen)
            after = engine.send(execution_command(frame, chosen["candidate_ref"]))
            self.assertEqual(after.get("type"), "decision_frame", after)
            self.assertEqual(player_entity(after)["energy"], predicted.state.energy)
            bludgeon = card_candidate(after, "CARD.BLUDGEON")
            final = engine.send(execution_command(after, bludgeon["candidate_ref"]))
            self.assertIn(final.get("type"), {"decision_frame", "terminal_frame"})

    def test_bloodletting_energy_and_hp_match_native(self):
        with CliEngine() as engine:
            frame = start_combat(
                engine,
                [
                    "BLOODLETTING",
                    "BLUDGEON",
                    "DEFEND_IRONCLAD",
                    "STRIKE_IRONCLAD",
                    "STRIKE_IRONCLAD",
                ],
                "lookahead_bloodletting",
            )
            defend = card_candidate(frame, "CARD.DEFEND_IRONCLAD")
            frame = engine.send(execution_command(frame, defend["candidate_ref"]))
            self.assertEqual(frame.get("type"), "decision_frame", frame)
            self.assertFalse(
                any(
                    c["verb"] == "PLAY_CARD"
                    and any(
                        e.get("content_id") == "CARD.BLUDGEON"
                        for e in frame["public"]["entities"]
                        if e.get("ref") in c.get("source_refs", [])
                    )
                    for c in frame["legal"]["candidates"]
                )
            )
            bloodletting = card_candidate(frame, "CARD.BLOODLETTING")
            predicted = CombatModel(frame, dynamic_legality=True).predict(bloodletting)
            after = engine.send(execution_command(frame, bloodletting["candidate_ref"]))
            self.assertEqual(after.get("type"), "decision_frame", after)
            self.assertEqual(player_entity(after)["energy"], predicted.state.energy)
            self.assertEqual(player_entity(after)["hp"], predicted.state.hp)
            self.assertEqual(
                card_candidate(after, "CARD.BLUDGEON")["verb"], "PLAY_CARD"
            )


if __name__ == "__main__":
    unittest.main()
