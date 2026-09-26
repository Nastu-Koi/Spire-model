"""Deck marginal value through the production public planner."""

import unittest

from combat_solver_cli.public_search import PublicPlanner
from combat_solver_cli.tests.test_public_resources import frame
from combat_solver_cli.tests.test_public_search import candidate, card


def deck_card(ref, *, damage=0, block=0, draw=0, strength=0, name=None):
    result = card(
        ref,
        name or "CARD." + ref.upper(),
        damage=damage,
        block=block,
        card_type="Attack" if damage else "Power" if strength else "Skill",
        zone="deck",
    )
    result["stats"].update(Cards=draw, StrengthPower=strength)
    return result


def reward_frame(offered, deck):
    return frame(
        "card_reward",
        [*deck, {**offered, "zone": "reward"}],
        [
            candidate("take", "TAKE_CARD_REWARD", offered["ref"]),
            candidate("skip", "SKIP"),
        ],
    )


class DeckRewardTests(unittest.TestCase):
    def test_same_defense_card_fills_gap_but_is_skipped_in_saturated_deck(self):
        offered = deck_card("offered", block=8, name="CARD.SHRUG_IT_OFF")
        missing = [deck_card(str(i), damage=8) for i in range(10)]
        saturated = [
            deck_card(str(i), block=8, name="CARD.SHRUG_IT_OFF") for i in range(10)
        ]
        for deck, expected in ((missing, "take"), (saturated, "skip")):
            with self.subTest(expected=expected):
                action, diagnostic = PublicPlanner(deck_rewards=True).choose(
                    reward_frame(offered, deck)
                )
                self.assertEqual(action["candidate_ref"], expected)
                self.assertEqual(len(diagnostic["candidate_scores"]), 2)

    def test_damage_draw_and_growth_gaps_affect_choice(self):
        for role in ({"damage": 10}, {"draw": 3}, {"strength": 2}):
            with self.subTest(role=role):
                offered = deck_card("offered", **role)
                lacking = [deck_card(str(i), block=6) for i in range(10)]
                saturated = [
                    deck_card(str(i), **role, name=offered["content_id"])
                    for i in range(10)
                ]
                for deck, expected in ((lacking, "take"), (saturated, "skip")):
                    self.assertEqual(
                        PublicPlanner(deck_rewards=True).choose(
                            reward_frame(offered, deck)
                        )[0]["candidate_ref"],
                        expected,
                    )

    def test_upgrade_and_removal_follow_deck_needs(self):
        deck = [deck_card(str(i), damage=8) for i in range(10)]
        defense = deck_card("defense", block=8)
        curse = card("curse", "CARD.REGRET", card_type="Curse", zone="deck")
        deck.extend([defense, curse])
        candidates = [candidate(c["ref"], "SELECT_ONE", c["ref"]) for c in deck]
        state = frame("card_select", deck, candidates)
        for operation, expected in (("remove", "curse"), ("upgrade", "defense")):
            state["public"]["selection_context"] = {"operation": operation}
            action, scores = PublicPlanner(deck_rewards=True).choose(state)
            self.assertEqual(action["candidate_ref"], expected)
            self.assertEqual(len(scores["candidate_scores"]), len(candidates))

    def test_unknown_special_effect_remains_legal_and_public_order_is_irrelevant(self):
        import copy

        special = deck_card("special", name="CARD.UNKNOWN")
        state = reward_frame(special, [deck_card(str(i), damage=8) for i in range(10)])
        changed = copy.deepcopy(state)
        changed["public"]["entities"].reverse()
        changed["public"]["seed"] = "hidden"
        for entity in changed["public"]["entities"]:
            entity["hidden_reward"] = "perfect"
        planner = PublicPlanner(deck_rewards=True)
        self.assertEqual(planner.choose(state), planner.choose(changed))
        self.assertEqual(len(planner.choose(state)[1]["candidate_scores"]), 2)

    def test_strength_synergy_in_attack_deck_increases_its_marginal_value(self):
        offered = deck_card("growth", strength=2)
        attacks = [deck_card(str(i), damage=8) for i in range(10)]
        skills = [deck_card(str(i), block=6) for i in range(10)]
        planner = PublicPlanner(deck_rewards=True)
        attack_scores = planner.choose(reward_frame(offered, attacks))[1][
            "candidate_scores"
        ]
        skill_scores = planner.choose(reward_frame(offered, skills))[1][
            "candidate_scores"
        ]
        self.assertGreater(attack_scores[0]["score"], skill_scores[0]["score"])

    def test_reward_switch_leaves_combat_frozen(self):
        from combat_solver_cli.tests.test_public_search import combat_frame

        state = combat_frame(hand=[card("strike", "CARD.STRIKE_IRONCLAD", damage=6)])
        old_action, old = PublicPlanner().choose(state)
        new_action, new = PublicPlanner(deck_rewards=True).choose(state)
        self.assertEqual(old_action, new_action)
        self.assertEqual(old["candidate_scores"], new["candidate_scores"])
