"""Route and supply choices through the public planner boundary."""

import unittest

from combat_solver_cli.public_search import PublicPlanner
from combat_solver_cli.tests.test_public_search import candidate


def frame(phase, entities, candidates, *, hp=80, gold=150, relations=()):
    return {
        "public": {
            "phase": phase,
            "entities": [
                {
                    "ref": "player",
                    "entity_type": "player",
                    "hp": hp,
                    "max_hp": 80,
                    "gold": gold,
                },
                *entities,
            ],
            "relations": list(relations),
        },
        "legal": {"candidates": candidates},
    }


def node(ref, kind):
    return {"ref": ref, "entity_type": "map_node", "content_id": kind}


def edge(source, target):
    return {"source": source, "target": target, "role": "map_edge"}


class ResourcePlannerTests(unittest.TestCase):
    def test_low_health_takes_supply_while_healthy_player_takes_growth(self):
        entities = [
            node("rest", "RestSite"),
            node("elite", "Elite"),
            node("boss", "Boss"),
        ]
        moves = [
            candidate("rest", "MOVE_TO_NODE", "rest"),
            candidate("elite", "MOVE_TO_NODE", "elite"),
        ]
        for hp, expected in ((16, "rest"), (80, "elite")):
            with self.subTest(hp=hp):
                state = frame(
                    "map",
                    entities,
                    moves,
                    hp=hp,
                    relations=[edge("rest", "boss"), edge("elite", "boss")],
                )
                action, diagnostic = PublicPlanner(route_resources=True).choose(state)
                self.assertEqual(action["candidate_ref"], expected)
                self.assertEqual(len(diagnostic["candidate_scores"]), len(moves))

    def test_route_accounts_for_danger_before_later_healing(self):
        state = frame(
            "map",
            [
                node("safe", "RestSite"),
                node("risky", "Elite"),
                node("late", "RestSite"),
                node("boss", "Boss"),
            ],
            [
                candidate("safe", "MOVE_TO_NODE", "safe"),
                candidate("risky", "MOVE_TO_NODE", "risky"),
            ],
            hp=15,
            relations=[
                edge("safe", "boss"),
                edge("risky", "late"),
                edge("late", "boss"),
            ],
        )
        self.assertEqual(
            PublicPlanner(route_resources=True).choose(state)[0]["candidate_ref"],
            "safe",
        )

    def test_rest_treats_injury_or_invests_in_upgrade(self):
        entities = [
            {"ref": name, "entity_type": "rest_option", "content_id": name}
            for name in ("HEAL", "SMITH", "UNKNOWN")
        ]
        options = [
            candidate(name, "CHOOSE_REST_OPTION", name)
            for name in ("HEAL", "SMITH", "UNKNOWN")
        ]
        for hp, expected in ((16, "HEAL"), (80, "SMITH")):
            with self.subTest(hp=hp):
                action, diagnostic = PublicPlanner(route_resources=True).choose(
                    frame("rest_site", entities, options, hp=hp)
                )
                self.assertEqual(action["candidate_ref"], expected)
                self.assertEqual(len(diagnostic["candidate_scores"]), 3)

    def test_shop_price_changes_purchase_to_leaving(self):
        from combat_solver_cli.tests.test_public_search import card

        offered = card("card", "CARD.INFLAME", card_type="Power", zone="shop")
        offered["stats"] = {"StrengthPower": 2}
        for price, expected in ((30, "buy"), (149, "leave")):
            with self.subTest(price=price):
                state = frame(
                    "shop",
                    [
                        offered,
                        {
                            "ref": "item",
                            "entity_type": "shop_item",
                            "content_id": "CARD.INFLAME",
                            "price": price,
                        },
                    ],
                    [
                        candidate("buy", "BUY_ITEM", "item"),
                        candidate("leave", "LEAVE_ROOM"),
                    ],
                    relations=[{"source": "item", "target": "card", "role": "offers"}],
                )
                self.assertEqual(
                    PublicPlanner(route_resources=True).choose(state)[0][
                        "candidate_ref"
                    ],
                    expected,
                )

    def test_remove_curse_and_upgrade_useful_card_keep_native_candidates(self):
        from combat_solver_cli.tests.test_public_search import card

        curse = card("curse", "CARD.REGRET", card_type="Curse", zone="deck")
        strike = card("strike", "CARD.STRIKE_IRONCLAD", damage=6, zone="deck")
        power = card("power", "CARD.INFLAME", card_type="Power", zone="deck")
        power["stats"] = {"StrengthPower": 2}
        options = [
            candidate(c["ref"], "SELECT_ONE", c["ref"]) for c in (curse, strike, power)
        ]
        for operation, expected in (("remove", "curse"), ("upgrade", "power")):
            state = frame("card_select", [curse, strike, power], options)
            state["public"]["selection_context"] = {"operation": operation}
            action, diagnostic = PublicPlanner(route_resources=True).choose(state)
            self.assertEqual(action["candidate_ref"], expected)
            self.assertEqual(len(diagnostic["candidate_scores"]), 3)

    def test_hidden_fields_and_entity_order_do_not_change_route_scores(self):
        import copy

        state = frame(
            "map",
            [node("rest", "RestSite"), node("elite", "Elite"), node("boss", "Boss")],
            [
                candidate("rest", "MOVE_TO_NODE", "rest"),
                candidate("elite", "MOVE_TO_NODE", "elite"),
            ],
            relations=[edge("rest", "boss"), edge("elite", "boss")],
        )
        changed = copy.deepcopy(state)
        changed["public"]["entities"].reverse()
        changed["public"]["hidden_encounter"] = "DEATH"
        changed["public"]["seed"] = "secret"
        for entity in changed["public"]["entities"]:
            entity["hidden_stock"] = ["RELIC.WIN"]
        planner = PublicPlanner(route_resources=True)
        self.assertEqual(planner.choose(state), planner.choose(changed))

    def test_resource_switch_leaves_combat_and_rewards_frozen(self):
        from combat_solver_cli.tests.test_public_search import card, combat_frame

        combat = combat_frame(hand=[card("strike", "CARD.STRIKE_IRONCLAD", damage=6)])
        reward = frame(
            "card_reward",
            [card("reward", "CARD.INFLAME", card_type="Power")],
            [
                candidate("take", "TAKE_CARD_REWARD", "reward"),
                candidate("skip", "SKIP"),
            ],
        )
        for state in (combat, reward):
            before, old_scores = PublicPlanner().choose(state)
            after, new_scores = PublicPlanner(route_resources=True).choose(state)
            self.assertEqual(before, after)
            self.assertEqual(
                old_scores["candidate_scores"], new_scores["candidate_scores"]
            )
