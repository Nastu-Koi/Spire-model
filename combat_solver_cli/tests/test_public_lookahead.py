"""Behavioral checks for opt-in hypothetical combat search."""

import copy
import unittest

from combat_solver_cli.public_effects import CombatModel
from combat_solver_cli.public_future import FutureSampler
from combat_solver_cli.public_search import PublicPlanner
from combat_solver_cli.tests.test_public_search import candidate, card, combat_frame


class DynamicLegalityTests(unittest.TestCase):
    def test_energy_potion_unlocks_expensive_attack_without_root_candidate(self):
        heavy = card("heavy", "CARD.BLUDGEON", cost=3, damage=25)
        heavy["zone"] = "hand"
        potion = {
            "ref": "potion:0",
            "entity_type": "potion",
            "content_id": "POTION.ENERGY_POTION",
            "stats": {"Energy": 2},
            "target_type": "Self",
        }
        frame = combat_frame(
            hp=8, energy=1, enemy_hp=25, incoming=20, hand=[heavy], extra=[potion]
        )
        frame["legal"]["candidates"] = [
            candidate("end", "END_TURN"),
            candidate("potion", "USE_POTION", "potion:0", "player"),
        ]
        action, detail = PublicPlanner(depth=2, dynamic_legality=True).choose(frame)
        self.assertEqual(action["candidate_ref"], "potion")
        self.assertEqual(
            {x["candidate_ref"] for x in detail["candidate_scores"]},
            {"end", "potion"},
        )
        self.assertGreater(detail["candidate_scores"][1]["score"], 0)

    def test_bloodletting_model_is_opt_in_and_unlocks_heavy_attack(self):
        blood = card("blood", "CARD.BLOODLETTING", cost=0, card_type="Skill")
        blood["stats"].update(HpLoss=3, Energy=2)
        heavy = card("heavy", "CARD.BLUDGEON", cost=3, damage=25)
        frame = combat_frame(
            hp=10, energy=1, enemy_hp=25, incoming=20, hand=[blood, heavy]
        )
        frame["legal"]["candidates"] = [
            candidate("end", "END_TURN"),
            candidate("blood", "PLAY_CARD", "blood", "player"),
        ]
        self.assertIsNone(
            CombatModel(frame).predict(frame["legal"]["candidates"][1]).state
        )
        predicted = CombatModel(frame, dynamic_legality=True).predict(
            frame["legal"]["candidates"][1]
        )
        self.assertEqual((predicted.state.energy, predicted.state.hp), (3, 7))
        action, _ = PublicPlanner(depth=2, dynamic_legality=True).choose(frame)
        self.assertEqual(action["candidate_ref"], "blood")

    def test_used_and_exhausted_cards_do_not_reappear(self):
        attack = card("attack", "CARD.STRIKE_IRONCLAD", cost=0, damage=4)
        attack["keywords"] = ["Exhaust"]
        frame = combat_frame(enemy_hp=20, hand=[attack])
        _, detail = PublicPlanner(depth=3, dynamic_legality=True).choose(frame)
        played = next(
            x["score"]
            for x in detail["candidate_scores"]
            if x["candidate_ref"].startswith("play:")
        )
        self.assertLess(played, 20)

    def test_x_cost_uses_energy_after_potion_in_branch(self):
        whirlwind = card("whirl", "CARD.WHIRLWIND", cost=0, damage=5)
        whirlwind.update(x_cost=True, target_type="AllEnemies")
        potion = {
            "ref": "potion:0",
            "entity_type": "potion",
            "content_id": "POTION.ENERGY_POTION",
            "stats": {"Energy": 2},
            "target_type": "Self",
        }
        frame = combat_frame(
            hp=8, energy=1, enemy_hp=15, incoming=20, hand=[whirlwind], extra=[potion]
        )
        frame["legal"]["candidates"].append(
            candidate("potion", "USE_POTION", "potion:0", "player")
        )
        action, _ = PublicPlanner(depth=2, dynamic_legality=True).choose(frame)
        self.assertEqual(action["candidate_ref"], "potion")
        zero = combat_frame(energy=0, enemy_hp=15, hand=[whirlwind])
        zero_hit = CombatModel(zero).predict(zero["legal"]["candidates"][1])
        self.assertEqual(zero_hit.state.enemies["enemy:1"][0], 15)

    def test_negative_unknown_cost_truncates_without_losing_root(self):
        unknown = card("unknown", "CARD.STRIKE_IRONCLAD", cost=-2, damage=99)
        frame = combat_frame(hand=[unknown])
        action, detail = PublicPlanner(dynamic_legality=True).choose(frame)
        self.assertIn(
            action["candidate_ref"],
            {c["candidate_ref"] for c in frame["legal"]["candidates"]},
        )
        self.assertEqual(len(detail["candidate_scores"]), 2)

    def test_unknown_cost_change_does_not_create_a_branch_action(self):
        blood = card("blood", "CARD.BLOODLETTING", cost=0, card_type="Skill")
        blood["stats"].update(HpLoss=3, Energy=2)
        discounted = card("discount", "CARD.BLOOD_FOR_BLOOD", cost=4, damage=99)
        frame = combat_frame(energy=1, enemy_hp=99, hand=[blood, discounted])
        frame["legal"]["candidates"] = [
            candidate("end", "END_TURN"),
            candidate("blood", "PLAY_CARD", "blood", "player"),
        ]
        model = CombatModel(frame, dynamic_legality=True)
        state = model.predict(frame["legal"]["candidates"][1]).state
        self.assertEqual(state.energy, 3)
        # A health-loss discount needs a verified card rule; 4-cost Blood for
        # Blood is not invented from the current 3-energy branch state.
        hypothetical = PublicPlanner._hypothetical_candidates(
            state, model, frame["legal"]["candidates"]
        )
        self.assertFalse(
            any(c.get("source_refs") == ["discount"] for c in hypothetical)
        )

    def test_opt_in_parameters_are_bounded_and_frame_unchanged(self):
        frame = combat_frame(hand=[card("strike", "CARD.STRIKE_IRONCLAD", damage=6)])
        before = copy.deepcopy(frame)
        _, detail = PublicPlanner(dynamic_legality=True).choose(frame)
        self.assertTrue(detail["dynamic_legality"])
        self.assertEqual(frame, before)
        for kwargs in (
            {"draw_samples": -1},
            {"draw_samples": 9},
            {"draw_samples": True},
            {"future_value": 1},
            {"dynamic_legality": 1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                PublicPlanner(**kwargs)  # pyright: ignore[reportArgumentType]


class FutureSearchTests(unittest.TestCase):
    def test_samples_use_public_multiset_and_ignore_transport_order(self):
        deck = [
            card("a", "CARD.STRIKE_IRONCLAD", damage=6, zone="draw_pile"),
            card("b", "CARD.STRIKE_IRONCLAD", damage=6, zone="draw_pile"),
            card(
                "c",
                "CARD.DEFEND_IRONCLAD",
                block=5,
                card_type="Skill",
                zone="draw_pile",
            ),
        ]
        frame = combat_frame(hand=[], extra=deck)
        first = FutureSampler(CombatModel(frame), seed=17, samples=8)
        draws = first.draws(first.model.root)
        self.assertEqual(len(draws), 8)
        self.assertTrue(all(len(draw) == 3 for draw in draws))
        self.assertTrue(all(set(draw) == {"a", "b", "c"} for draw in draws))
        changed = copy.deepcopy(frame)
        changed["hidden_seed"] = "different"
        changed["public"]["entities"][-3:] = reversed(
            changed["public"]["entities"][-3:]
        )
        second = FutureSampler(CombatModel(changed), seed=17, samples=8)
        self.assertEqual(
            [
                sorted(second.model.refs[r]["content_id"] for r in draw)
                for draw in second.draws(second.model.root)
            ],
            [sorted(first.model.refs[r]["content_id"] for r in draw) for draw in draws],
        )

    def test_empty_draw_pile_reshuffles_discard_and_used_hand(self):
        hand = card("hand", "CARD.STRIKE_IRONCLAD", damage=6)
        discard = card(
            "discard",
            "CARD.DEFEND_IRONCLAD",
            block=5,
            card_type="Skill",
            zone="discard_pile",
        )
        exhausted = card(
            "exhaust", "CARD.STRIKE_IRONCLAD", damage=6, zone="exhaust_pile"
        )
        frame = combat_frame(hand=[hand], extra=[discard, exhausted])
        sampler = FutureSampler(CombatModel(frame), seed=4, samples=2)
        self.assertEqual(
            [set(draw) for draw in sampler.draws(sampler.model.root)],
            [{"hand", "discard"}, {"hand", "discard"}],
        )

    def test_known_order_or_incomplete_pile_truncates_future_sample(self):
        growth = card("growth", "CARD.INFLAME", card_type="Power")
        growth["stats"]["StrengthPower"] = 2
        deck = [card("draw", "CARD.STRIKE_IRONCLAD", damage=6, zone="draw_pile")]
        frame = combat_frame(hand=[growth], extra=deck)
        for summary in (
            {
                "entity_type": "pile_summary",
                "zone": "draw_pile",
                "count": 1,
                "order_known": True,
            },
            {
                "entity_type": "pile_summary",
                "zone": "draw_pile",
                "count": 2,
                "order_known": False,
            },
        ):
            with self.subTest(summary=summary):
                current = copy.deepcopy(frame)
                current["public"]["entities"].append(summary)
                sampler = FutureSampler(CombatModel(current), seed=1, samples=4)
                self.assertTrue(sampler.pile_boundary)
                self.assertEqual(sampler.estimate(sampler.model.root), 0)

    def test_temporary_cost_and_setup_strength_do_not_cross_turn_boundary(self):
        discounted = card(
            "discounted", "CARD.STRIKE_IRONCLAD", cost=0, damage=6, zone="draw_pile"
        )
        frame = combat_frame(enemy_hp=30, extra=[discounted])
        sampler = FutureSampler(CombatModel(frame), seed=1, samples=2)
        self.assertEqual(sampler.estimate(sampler.model.root), 0)
        setup = card("setup", "CARD.SETUP_STRIKE", cost=1, damage=5)
        setup["stats"]["StrengthPower"] = 2
        frame = combat_frame(
            enemy_hp=30,
            hand=[setup],
            extra=[card("draw", "CARD.STRIKE_IRONCLAD", damage=6, zone="draw_pile")],
        )
        model = CombatModel(frame)
        state = model.predict(frame["legal"]["candidates"][1]).state
        self.assertEqual(FutureSampler(model, seed=1, samples=2).estimate(state), 0)

    def test_future_samples_change_growth_decision_without_mutating_frame(self):
        strike = card("strike", "CARD.STRIKE_IRONCLAD", damage=6)
        growth = card("growth", "CARD.INFLAME", card_type="Power")
        growth["stats"]["StrengthPower"] = 2
        future = [
            card(f"draw:{i}", "CARD.STRIKE_IRONCLAD", damage=6, zone="draw_pile")
            for i in range(5)
        ]
        frame = combat_frame(energy=1, enemy_hp=60, hand=[strike, growth], extra=future)
        before = copy.deepcopy(frame)
        old, _ = PublicPlanner(depth=1).choose(frame)
        new, detail = PublicPlanner(depth=1, draw_samples=4).choose(frame)
        self.assertEqual(old["source_refs"], ["strike"])
        self.assertEqual(new["source_refs"], ["growth"])
        self.assertEqual(detail["future_model"], "one_turn_repeated_visible_intent")
        self.assertEqual(frame, before)

    def test_planner_scores_ignore_hidden_seed_pile_order_and_ref_renumbering(self):
        growth = card("growth", "CARD.INFLAME", card_type="Power")
        growth["stats"]["StrengthPower"] = 2
        strike = card("strike", "CARD.STRIKE_IRONCLAD", damage=6)
        pile = [
            card(
                f"draw:{i}",
                "CARD.STRIKE_IRONCLAD" if i % 2 else "CARD.DEFEND_IRONCLAD",
                damage=6 if i % 2 else 0,
                block=5 if i % 2 == 0 else 0,
                card_type="Attack" if i % 2 else "Skill",
                zone="draw_pile",
            )
            for i in range(8)
        ]
        frame = combat_frame(enemy_hp=60, hand=[strike, growth], extra=pile)
        planner = PublicPlanner(depth=2, draw_samples=8, sampling_seed=42)
        action, detail = planner.choose(frame)
        changed = copy.deepcopy(frame)
        changed["hidden_seed"] = "unrelated"
        changed["rng_state"] = {"secret": 100}
        for i, entity in enumerate(changed["public"]["entities"][-8:]):
            entity["ref"] = f"other:{i}"
        changed["public"]["entities"][-8:] = list(
            reversed(changed["public"]["entities"][-8:])
        )
        other, other_detail = planner.choose(changed)
        self.assertEqual(action["candidate_ref"], other["candidate_ref"])
        self.assertEqual(detail["candidate_scores"], other_detail["candidate_scores"])

    def test_draw_samples_preserve_semantics_when_modified_card_refs_change(self):
        pile = [
            card(f"card:{i}", "CARD.STRIKE_IRONCLAD", damage=6, zone="draw_pile")
            for i in range(6)
        ]
        for item in pile[2:]:
            item["enchantment"] = "ENCHANTMENT.UNKNOWN"
        frame = combat_frame(enemy_hp=60, hand=[], extra=pile)
        changed = copy.deepcopy(frame)
        for item in changed["public"]["entities"]:
            if item.get("ref") == "card:1":
                item["ref"] = "card:4"
            elif item.get("ref") == "card:4":
                item["ref"] = "card:1"
        planner = PublicPlanner(draw_samples=1, sampling_seed=0)
        self.assertEqual(planner.choose(frame), planner.choose(changed))

    def test_future_draw_prior_requires_cards_available_to_draw(self):
        draw = card("draw", "CARD.BATTLE_TRANCE", cost=0, card_type="Skill")
        draw["stats"]["Cards"] = 3
        frame = combat_frame(energy=3, hand=[draw])
        action, _ = PublicPlanner(depth=1, future_value=True).choose(frame)
        self.assertEqual(action["verb"], "END_TURN")

    def test_future_valuation_prefers_growth_but_visible_lethal_survival_first(self):
        strike = card("strike", "CARD.STRIKE_IRONCLAD", damage=6)
        growth = card("growth", "CARD.INFLAME", card_type="Power")
        growth["stats"]["StrengthPower"] = 2
        frame = combat_frame(energy=1, enemy_hp=60, hand=[strike, growth])
        old, _ = PublicPlanner(depth=1).choose(frame)
        new, detail = PublicPlanner(depth=1, future_value=True).choose(frame)
        self.assertEqual(old["source_refs"], ["strike"])
        self.assertEqual(new["source_refs"], ["growth"])
        self.assertIn("strength", detail["future_terms"])
        defend = card("defend", "CARD.DEFEND_IRONCLAD", block=10, card_type="Skill")
        danger = combat_frame(
            hp=8, energy=1, enemy_hp=60, incoming=10, hand=[growth, defend]
        )
        action, _ = PublicPlanner(depth=1, future_value=True).choose(danger)
        self.assertEqual(action["source_refs"], ["defend"])


if __name__ == "__main__":
    unittest.main()
