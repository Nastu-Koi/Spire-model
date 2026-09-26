"""Behavioral checks for the bounded public-state planner."""

import copy
import unittest

from combat_solver_cli.public_search import PublicPlanner
from model.representation import clean_public


def card(ref, name, *, cost=1, damage=0, block=0, card_type="Attack", zone="hand"):
    stats = {}
    if damage:
        stats["Damage"] = damage
    if block:
        stats["Block"] = block
    return {
        "ref": ref,
        "entity_type": "card",
        "content_id": name,
        "zone": zone,
        "cost": cost,
        "card_type": card_type,
        "target_type": "AnyEnemy" if damage else "Self",
        "stats": stats,
        "effect_coverage": "opaque",
        "semantic_program": {"kind": "effect", "op": "OPAQUE_RULE"},
    }


def candidate(ref, verb, source=None, target=None):
    return {
        "candidate_ref": ref,
        "decoder_slot_ref": "slot:" + ref,
        "verb": verb,
        "source_refs": [source] if source else [],
        "target_refs": [target] if target else [],
        "effect_coverage": "opaque",
    }


def combat_frame(*, hp=30, energy=3, enemy_hp=20, incoming=0, hand=None, extra=None):
    entities = [
        {
            "ref": "player",
            "entity_type": "player",
            "hp": hp,
            "max_hp": 80,
            "block": 0,
            "energy": energy,
            "max_energy": 3,
        },
        {
            "ref": "enemy:1",
            "entity_type": "enemy",
            "hp": enemy_hp,
            "max_hp": enemy_hp,
            "block": 0,
        },
    ]
    if incoming:
        entities.append(
            {
                "entity_type": "intent",
                "owner_ref": "enemy:1",
                "intent": "Attack",
                "damage": incoming,
                "hits": 1,
                "total_damage": incoming,
            }
        )
    hand = hand or []
    entities.extend(hand)
    entities.extend(extra or [])
    candidates = [candidate("end", "END_TURN")]
    for item in hand:
        if item["content_id"] in {"CARD.STRIKE_IRONCLAD", "CARD.DEFEND_IRONCLAD"}:
            targets = (
                ["enemy:1"]
                if item["content_id"].endswith("STRIKE_IRONCLAD")
                else [None]
            )
            for target in targets:
                candidates.append(
                    candidate(
                        "play:" + item["ref"] + ":" + str(target),
                        "PLAY_CARD",
                        item["ref"],
                        target,
                    )
                )
        else:
            candidates.append(
                candidate("play:" + item["ref"], "PLAY_CARD", item["ref"], "enemy:1")
            )
    return {
        "type": "decision_frame",
        "routing": {
            "episode_id": "episode",
            "decision_id": "decision",
            "state_version": 1,
        },
        "public": {"phase": "combat", "entities": entities},
        "legal": {"candidates": candidates},
    }


class PublicPlannerTests(unittest.TestCase):
    def test_energy_gain_keeps_opportunity_value_when_attacks_are_not_yet_legal(self):
        energy = card("energy", "CARD.BLOODLETTING", cost=0, card_type="Skill")
        energy["stats"].update(HpLoss=3, Energy=2)
        frame = combat_frame(hp=50, energy=0, enemy_hp=12, incoming=20, hand=[energy])
        frame["public"]["entities"].extend(
            card(f"s{i}", "CARD.STRIKE_IRONCLAD", damage=6) for i in range(2)
        )
        action, _ = PublicPlanner().choose(frame)
        self.assertEqual(action["source_refs"], ["energy"])

    def test_bash_before_strike_can_prevent_lethal(self):
        bash = card("bash", "CARD.BASH", cost=2, damage=8)
        bash["stats"]["VulnerablePower"] = 2
        strike = card("strike", "CARD.STRIKE_IRONCLAD", damage=6)
        frame = combat_frame(hp=10, enemy_hp=17, incoming=30, hand=[strike, bash])
        action, _ = PublicPlanner(depth=2).choose(frame)
        self.assertEqual(action["source_refs"], ["bash"])

    def test_strength_buff_is_used_before_attack_combination(self):
        buff = card("buff", "CARD.INFLAME", card_type="Power")
        buff["stats"]["StrengthPower"] = 2
        hand = [card(f"s{i}", "CARD.STRIKE_IRONCLAD", damage=6) for i in range(2)] + [
            buff
        ]
        frame = combat_frame(hp=10, enemy_hp=16, incoming=30, hand=hand)
        action, _ = PublicPlanner(depth=3).choose(frame)
        self.assertEqual(action["source_refs"], ["buff"])

    def test_public_health_cost_is_preserved_and_avoids_self_lethal(self):
        attack = card("self-harm", "CARD.BREAKTHROUGH", cost=0, damage=1)
        attack["stats"].update(HpLoss=5, StrengthPower=2, hidden_rng=12345)
        frame = combat_frame(hp=3, hand=[attack])
        public = clean_public(frame["public"])
        stats = next(
            e["stats"] for e in public["entities"] if e.get("ref") == "self-harm"
        )
        self.assertEqual(stats["HpLoss"], 5)
        self.assertEqual(stats["StrengthPower"], 2)
        self.assertNotIn("hidden_rng", stats)
        action, _ = PublicPlanner().choose(frame)
        self.assertEqual(action["verb"], "END_TURN")

    def test_treasure_is_taken_instead_of_leaving_without_it(self):
        frame = {
            "public": {
                "phase": "treasure",
                "entities": [
                    {"ref": "player", "entity_type": "player", "hp": 40, "max_hp": 80},
                    {
                        "ref": "relic",
                        "entity_type": "relic",
                        "content_id": "RELIC.EXAMPLE",
                    },
                ],
            },
            "legal": {
                "candidates": [
                    candidate("leave", "LEAVE_ROOM"),
                    candidate("take", "TAKE_TREASURE_RELIC", "relic"),
                ]
            },
        }
        action, _ = PublicPlanner().choose(frame)
        self.assertEqual(action["verb"], "TAKE_TREASURE_RELIC")
        frame["legal"]["candidates"][1]["verb"] = "OPEN_CHEST"
        action, _ = PublicPlanner().choose(frame)
        self.assertEqual(action["verb"], "OPEN_CHEST")

    def test_hidden_transport_and_draw_order_do_not_change_scores_or_action(self):
        strike = card("strike", "CARD.STRIKE_IRONCLAD", damage=6)
        draw = [
            card("draw-a", "CARD.UNKNOWN_A", zone="draw_pile"),
            card("draw-b", "CARD.UNKNOWN_B", zone="draw_pile"),
        ]
        frame = combat_frame(hp=20, enemy_hp=8, incoming=6, hand=[strike], extra=draw)
        planner = PublicPlanner(depth=2, beam_width=4)
        action, details = planner.choose(frame)

        changed = copy.deepcopy(frame)
        changed["routing"] = {
            "episode_id": "other",
            "decision_id": "other",
            "state_version": 999,
        }
        changed["hidden_seed"] = "different-seed"
        changed["rng_state"] = {"state": [99, 1, 2]}
        changed["public"]["entities"][-2:] = reversed(
            changed["public"]["entities"][-2:]
        )
        other_action, other_details = planner.choose(changed)

        self.assertEqual(action["verb"], other_action["verb"])
        self.assertEqual(action["source_refs"], other_action["source_refs"])
        self.assertEqual(details["candidate_scores"], other_details["candidate_scores"])

    def test_choose_does_not_mutate_input_frame(self):
        frame = combat_frame(hand=[card("strike", "CARD.STRIKE_IRONCLAD", damage=6)])
        before = copy.deepcopy(frame)
        PublicPlanner().choose(frame)
        self.assertEqual(frame, before)

    def test_every_native_root_candidate_gets_a_score_including_opaque_card(self):
        strike = card("strike", "CARD.STRIKE_IRONCLAD", damage=6)
        opaque = card("opaque", "CARD.UNKNOWN_CARD", damage=2)
        frame = combat_frame(hand=[strike, opaque])
        _, details = PublicPlanner().choose(frame)
        scored = {item["candidate_ref"] for item in details["candidate_scores"]}
        self.assertEqual(
            scored, {c["candidate_ref"] for c in frame["legal"]["candidates"]}
        )
        opaque_score = next(
            x
            for x in details["candidate_scores"]
            if x["candidate_ref"] == "play:opaque"
        )
        self.assertTrue(isinstance(opaque_score["score"], (int, float)))

    def test_kill_is_preferred_when_it_prevents_lethal_attack(self):
        strike = card("strike", "CARD.STRIKE_IRONCLAD", damage=6)
        defend = card("defend", "CARD.DEFEND_IRONCLAD", block=5, card_type="Skill")
        frame = combat_frame(
            hp=10, energy=1, enemy_hp=6, incoming=10, hand=[strike, defend]
        )
        action, _ = PublicPlanner(depth=2, beam_width=8).choose(frame)
        self.assertEqual(action["source_refs"], ["strike"])

    def test_same_energy_card_combination_has_better_continuation_score(self):
        strike = card("strike", "CARD.STRIKE_IRONCLAD", cost=1, damage=4)
        defend = card(
            "defend", "CARD.DEFEND_IRONCLAD", cost=1, block=6, card_type="Skill"
        )
        frame = combat_frame(
            hp=20, energy=2, enemy_hp=30, incoming=8, hand=[strike, defend]
        )
        _, shallow = PublicPlanner(depth=1, beam_width=8).choose(frame)
        _, deep = PublicPlanner(depth=2, beam_width=8).choose(frame)
        shallow_scores = {
            x["candidate_ref"]: x["score"] for x in shallow["candidate_scores"]
        }
        deep_scores = {x["candidate_ref"]: x["score"] for x in deep["candidate_scores"]}
        self.assertGreater(
            deep_scores["play:strike:enemy:1"], shallow_scores["play:strike:enemy:1"]
        )

    def test_unknown_draw_pile_cards_are_not_read_as_hypothetical_hand(self):
        strike = card("strike", "CARD.STRIKE_IRONCLAD", damage=6)
        frame = combat_frame(
            hand=[strike],
            extra=[
                card("future", "CARD.UNKNOWN_WIN_BUTTON", damage=999, zone="draw_pile")
            ],
        )
        _, with_unknown = PublicPlanner(depth=4, beam_width=8).choose(frame)
        frame["public"]["entities"] = [
            e for e in frame["public"]["entities"] if e.get("ref") != "future"
        ]
        _, without_unknown = PublicPlanner(depth=4, beam_width=8).choose(frame)
        self.assertEqual(
            with_unknown["candidate_scores"], without_unknown["candidate_scores"]
        )

    def test_depth_and_width_are_bounded_integers(self):
        for kwargs in (
            {"depth": 0},
            {"depth": 13},
            {"depth": True},
            {"beam_width": 0},
            {"beam_width": 257},
            {"beam_width": 1.5},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                PublicPlanner(**kwargs)

    def test_abandon_only_when_low_hp_forced_into_combat_without_rescue(self):
        frame = {
            "public": {
                "phase": "map",
                "entities": [
                    {"ref": "player", "entity_type": "player", "hp": 4, "max_hp": 80},
                    {
                        "ref": "fight",
                        "entity_type": "map_node",
                        "content_id": "MONSTER",
                    },
                ],
            },
            "legal": {"candidates": [candidate("move", "MOVE_TO_NODE", "fight")]},
        }
        planner = PublicPlanner()
        self.assertEqual(
            planner.abandon_reason(frame), "low_hp_before_forced_combat_without_potion"
        )

        rest = copy.deepcopy(frame)
        rest["public"]["entities"][1]["content_id"] = "RESTSITE"
        self.assertIsNone(planner.abandon_reason(rest))

        potion = copy.deepcopy(frame)
        potion["public"]["entities"].append(
            {"ref": "potion:0", "entity_type": "potion", "content_id": "HEALING_POTION"}
        )
        self.assertIsNone(planner.abandon_reason(potion))


if __name__ == "__main__":
    unittest.main()
