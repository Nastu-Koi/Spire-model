"""Production effect predictions checked against observed native outcomes."""

import json
import unittest
from pathlib import Path

from combat_solver_cli.public_effects import CombatModel
from combat_solver_cli.public_search import PublicPlanner
from combat_solver_cli.tests.test_public_search import card, combat_frame

FIXTURES = Path(__file__).parent / "fixtures"


class PublicEffectTests(unittest.TestCase):
    def test_second_wind_matches_native_exhaust_and_block(self):
        fixture = json.loads((FIXTURES / "second_wind.json").read_text())
        frame, chosen = fixture["before"], fixture["candidate"]
        prediction = CombatModel(frame, effect_profile="second_wind").predict(chosen)
        observed_player = next(
            e
            for e in fixture["after"]["public"]["entities"]
            if e.get("ref") == "player"
        )
        self.assertEqual(prediction.state.block, observed_player["block"])
        self.assertEqual(prediction.state.hand, frozenset())
        before_hand = {
            e["ref"]
            for e in frame["public"]["entities"]
            if e.get("zone") == "hand" and e.get("entity_type") == "card"
        }
        self.assertEqual(
            before_hand - set(chosen["source_refs"]), prediction.state.exhaust
        )
        self.assertIn(chosen["source_refs"][0], prediction.state.discard)

    def test_spite_without_public_history_does_not_assume_conditional_hits(self):
        for name in ("spite", "spite_no_kill"):
            fixture = json.loads((FIXTURES / (name + ".json")).read_text())
            model = CombatModel(fixture["before"], effect_profile="spite")
            predicted = model.predict(fixture["candidate"])
            target = fixture["candidate"]["target_refs"][0]
            actual = next(
                e["hp"]
                for e in fixture["after"]["public"]["entities"]
                if e.get("ref") == target
            )
            self.assertEqual(predicted.state.enemies[target][0], actual)
            self.assertEqual(predicted.coverage, "approximate")

    def test_second_wind_consumption_changes_live_planner_choice(self):
        defend = card("defend", "CARD.DEFEND_IRONCLAD", block=5, card_type="Skill")
        wind = card("wind", "CARD.SECOND_WIND", block=5, card_type="Skill")
        status = card("status", "CARD.INFECTION", cost=-1, card_type="Status")
        frame = combat_frame(
            hp=8, energy=1, enemy_hp=60, incoming=10, hand=[defend, wind]
        )
        frame["public"]["entities"].append(status)
        chosen, _ = PublicPlanner(depth=1, effect_profile="second_wind").choose(frame)
        self.assertEqual(chosen["source_refs"], ["wind"])

    def test_spite_uses_public_loss_fact_and_preserves_it_in_branch(self):
        from model.representation import clean_public

        spite = card("spite", "CARD.SPITE", cost=0, damage=5)
        spite["stats"]["Repeat"] = 3
        frame = combat_frame(enemy_hp=30, hand=[spite])
        for lost, expected in [(False, 25), (True, 15)]:
            frame["public"]["entities"][0]["lost_hp_this_turn"] = lost
            public = clean_public(frame["public"])
            self.assertIs(public["entities"][0]["lost_hp_this_turn"], lost)
            model = CombatModel({"public": public}, effect_profile="spite")
            result = model.predict(frame["legal"]["candidates"][1])
            self.assertEqual(result.state.enemies["enemy:1"][0], expected)

    def test_weak_and_vulnerable_round_after_all_multipliers(self):
        attack = card("attack", "CARD.TWIN_STRIKE", damage=5)
        frame = combat_frame(
            enemy_hp=20,
            hand=[attack],
            extra=[
                {
                    "entity_type": "power",
                    "content_id": "POWER.WEAK_POWER",
                    "owner_ref": "player",
                    "stacks": 1,
                },
                {
                    "entity_type": "power",
                    "content_id": "POWER.VULNERABLE_POWER",
                    "owner_ref": "enemy:1",
                    "stacks": 1,
                },
            ],
        )
        result = CombatModel(frame, effect_profile="cards").predict(
            frame["legal"]["candidates"][1]
        )
        self.assertEqual(result.state.enemies["enemy:1"][0], 10)

    def test_whirlwind_uses_all_paid_energy(self):
        whirlwind = card("whirl", "CARD.WHIRLWIND", cost=0, damage=5)
        whirlwind.update(x_cost=True, target_type="AllEnemies")
        frame = combat_frame(energy=21, enemy_hp=200, hand=[whirlwind])
        result = CombatModel(frame, effect_profile="cards").predict(
            frame["legal"]["candidates"][1]
        )
        self.assertEqual(result.state.energy, 0)
        self.assertEqual(result.state.enemies["enemy:1"][0], 95)

    def test_body_slam_with_zero_block_still_receives_strength(self):
        slam = card("slam", "CARD.BODY_SLAM", damage=0)
        frame = combat_frame(
            enemy_hp=20,
            hand=[slam],
            extra=[
                {
                    "entity_type": "power",
                    "content_id": "POWER.STRENGTH_POWER",
                    "owner_ref": "player",
                    "stacks": 2,
                }
            ],
        )
        result = CombatModel(frame, effect_profile="cards").predict(
            frame["legal"]["candidates"][1]
        )
        self.assertEqual(result.state.enemies["enemy:1"][0], 18)

    def test_acquisition_and_post_combat_relics_do_not_hide_immediate_effects(self):
        for relic in ("FISHING_ROD", "LOST_COFFER", "KALEIDOSCOPE"):
            frame = combat_frame(
                enemy_hp=20,
                hand=[card("strike", "CARD.STRIKE_IRONCLAD", damage=6)],
                extra=[{"entity_type": "relic", "content_id": "RELIC." + relic}],
            )
            result = CombatModel(frame).predict(frame["legal"]["candidates"][1])
            self.assertEqual(result.coverage, "modeled")
            self.assertEqual(result.state.enemies["enemy:1"][0], 14)

    def test_unsupported_hook_does_not_claim_a_modeled_kill(self):
        attack = card("attack", "CARD.STRIKE_IRONCLAD", damage=20)
        frame = combat_frame(
            enemy_hp=10,
            hand=[attack],
            extra=[
                {
                    "entity_type": "power",
                    "content_id": "POWER.INTANGIBLE_POWER",
                    "owner_ref": "enemy:1",
                    "stacks": 1,
                },
            ],
        )
        result = CombatModel(frame, effect_profile="cards").predict(
            frame["legal"]["candidates"][1]
        )
        self.assertEqual(result.coverage, "unknown")
        self.assertIsNone(result.state)
        self.assertIn("unsupported", result.reason)

    def test_fire_and_block_potions_use_unpowered_public_values(self):
        from combat_solver_cli.tests.test_public_search import candidate

        for name, stats, expected_hp, expected_block in [
            ("FIRE_POTION", {"Damage": 20}, 10, 0),
            ("BLOCK_POTION", {"Block": 12}, 30, 12),
        ]:
            potion = {
                "ref": "potion:0",
                "entity_type": "potion",
                "content_id": "POTION." + name,
                "stats": stats,
                "target_type": "AnyEnemy" if name == "FIRE_POTION" else "Self",
            }
            frame = combat_frame(
                enemy_hp=30,
                extra=[
                    potion,
                    {
                        "entity_type": "power",
                        "content_id": "POWER.STRENGTH_POWER",
                        "owner_ref": "player",
                        "stacks": 10,
                    },
                    {
                        "entity_type": "power",
                        "content_id": "POWER.WEAK_POWER",
                        "owner_ref": "player",
                        "stacks": 1,
                    },
                    {
                        "entity_type": "power",
                        "content_id": "POWER.DEXTERITY_POWER",
                        "owner_ref": "player",
                        "stacks": 5,
                    },
                    {
                        "entity_type": "power",
                        "content_id": "POWER.FRAIL_POWER",
                        "owner_ref": "player",
                        "stacks": 1,
                    },
                    {
                        "entity_type": "power",
                        "content_id": "POWER.VULNERABLE_POWER",
                        "owner_ref": "enemy:1",
                        "stacks": 1,
                    },
                ],
            )
            action = candidate(
                "p",
                "USE_POTION",
                "potion:0",
                "enemy:1" if name == "FIRE_POTION" else "player",
            )
            result = CombatModel(frame, effect_profile="potions").predict(action)
            self.assertIsNotNone(result.state)
            self.assertEqual(result.state.enemies["enemy:1"][0], expected_hp)
            self.assertEqual(result.state.block, expected_block)
            self.assertEqual(result.coverage, "modeled")
            self.assertIsNone(result.reason)
            self.assertNotIn("potion:0", result.state.potions)

    def test_live_planner_uses_potion_to_survive_but_preserves_unneeded_block(self):
        from combat_solver_cli.tests.test_public_search import candidate

        for incoming, expected in [(12, "USE_POTION"), (0, "END_TURN")]:
            potion = {
                "ref": "potion:0",
                "entity_type": "potion",
                "content_id": "POTION.BLOCK_POTION",
                "stats": {"Block": 12},
            }
            frame = combat_frame(hp=5, incoming=incoming, extra=[potion])
            frame["legal"]["candidates"].append(
                candidate("p", "USE_POTION", "potion:0", "player")
            )
            action, _ = PublicPlanner(effect_profile="potions").choose(frame)
            self.assertEqual(action["verb"], expected)

    def test_weak_potion_updates_each_visible_hit_only_once(self):
        from combat_solver_cli.tests.test_public_search import candidate

        potion = {
            "ref": "potion:0",
            "entity_type": "potion",
            "content_id": "POTION.WEAK_POTION",
            "stats": {"WeakPower": 3},
        }
        frame = combat_frame(hp=13, incoming=18, extra=[potion])
        intent = next(
            e for e in frame["public"]["entities"] if e.get("entity_type") == "intent"
        )
        intent.update(damage=6, hits=3, total_damage=18)
        action = candidate("p", "USE_POTION", "potion:0", "enemy:1")
        frame["legal"]["candidates"].append(action)
        result = CombatModel(frame, effect_profile="potions").predict(action)
        self.assertIsNotNone(result.state)
        self.assertEqual(result.state.incoming["enemy:1"], 12)
        self.assertEqual(
            PublicPlanner(effect_profile="potions").choose(frame)[0]["verb"],
            "USE_POTION",
        )

    def test_energy_potion_stops_before_new_legality(self):
        from combat_solver_cli.tests.test_public_search import candidate

        frame = combat_frame(
            energy=0,
            extra=[
                {
                    "ref": "potion:0",
                    "entity_type": "potion",
                    "content_id": "POTION.ENERGY_POTION",
                    "stats": {"Energy": 2},
                }
            ],
        )
        result = CombatModel(frame, effect_profile="potions").predict(
            candidate("p", "USE_POTION", "potion:0", "player")
        )
        self.assertEqual(result.state.energy, 2)
        self.assertEqual(result.coverage, "approximate")
        self.assertEqual(result.reason, "new_legality_requires_native_observation")

    def test_fire_potion_kills_threat_and_preserves_input(self):
        from copy import deepcopy

        from combat_solver_cli.tests.test_public_search import candidate

        frame = combat_frame(
            hp=3,
            enemy_hp=20,
            incoming=15,
            extra=[
                {
                    "ref": "potion:0",
                    "entity_type": "potion",
                    "content_id": "POTION.FIRE_POTION",
                    "stats": {"Damage": 20},
                }
            ],
        )
        action = candidate("p", "USE_POTION", "potion:0", "enemy:1")
        frame["legal"]["candidates"].append(action)
        original = deepcopy(frame)
        self.assertEqual(PublicPlanner().choose(frame)[0], action)
        result = CombatModel(frame, effect_profile="potions").predict(action)
        self.assertEqual(result.state.enemies["enemy:1"][0], 0)
        self.assertEqual(frame, original)

    def test_self_damage_enables_spite_only_in_its_branch(self):
        hemo = card("hemo", "CARD.HEMOKINESIS", damage=15)
        hemo["stats"]["HpLoss"] = 2
        spite = card("spite", "CARD.SPITE", cost=0, damage=5)
        spite["stats"]["Repeat"] = 2
        frame = combat_frame(energy=3, enemy_hp=50, hand=[hemo, spite])
        frame["public"]["entities"][0]["lost_hp_this_turn"] = False
        model = CombatModel(frame, effect_profile="potions")
        first, second = frame["legal"]["candidates"][1:3]
        branch = model.predict(first).state
        result = model.transition(branch, second)
        self.assertEqual(result.state.enemies["enemy:1"][0], 25)
        self.assertTrue(result.state.lost_hp_this_turn)
        self.assertFalse(model.root.lost_hp_this_turn)
        self.assertEqual(model.predict(second).state.enemies["enemy:1"][0], 45)


if __name__ == "__main__":
    unittest.main()
