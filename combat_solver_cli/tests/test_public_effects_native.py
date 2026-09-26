"""Compare public effect predictions with actions in the real native engine.

These debug runs are deliberately not training-ready and must never be exported
as demonstrations. Enable them with STS2_NATIVE_EFFECT_TESTS=1.
"""

import math
import os
import unittest
from collections import Counter

from combat_solver_cli.public_effects import CombatModel
from combat_solver_cli.public_search import PublicPlanner
from model.engine import CliEngine
from model.protocol import clean_frame, execution_command

ENABLED = os.environ.get("STS2_NATIVE_EFFECT_TESTS") == "1"


def start_combat(
    engine,
    deck,
    seed,
    potions=(),
    encounter="SHRINKER_BEETLE_WEAK",
    draw_order=(),
):
    initial = engine.send(
        {
            "cmd": "start_run",
            "character": "Ironclad",
            "seed": seed,
            "ascension": 0,
            "lang": "en",
            "decision_protocol": False,
        }
    )
    engine.send(
        {
            "cmd": "set_player",
            "hp": 70,
            "max_hp": 80,
            "gold": 999,
            "relics": [],
            "potions": list(potions),
            "deck": deck,
        }
    )
    engine.send(
        {
            "cmd": "enter_room",
            "type": "combat",
            "encounter": encounter,
        }
    )
    if initial.get("type") == "error":
        raise AssertionError(f"Native debug setup failed: {initial}")
    if draw_order:
        result = engine.send({"cmd": "set_draw_order", "cards": list(draw_order)})
        if result.get("type") == "error":
            raise AssertionError(f"Could not set native draw order: {result}")
    frame = engine.send({"cmd": "advance_to_boundary"})
    if frame.get("type") != "decision_frame":
        raise AssertionError(f"Expected native decision frame, got {frame}")
    if frame["contract"]["training_ready"] is not False:
        raise AssertionError("Debug native fixture must not be training-ready")
    return frame


def card_candidate(frame, content_id):
    refs = {
        entity.get("ref"): entity
        for entity in frame["public"]["entities"]
        if entity.get("entity_type") == "card"
    }
    for candidate in frame["legal"]["candidates"]:
        if candidate["verb"] != "PLAY_CARD":
            continue
        source_ref = (candidate.get("source_refs") or [None])[0]
        if refs.get(source_ref, {}).get("content_id") == content_id:
            return candidate
    raise AssertionError(f"No legal {content_id} action in the native frame")


def candidate_with_verb(frame, verb, source_content_id=None):
    entities = {
        entity.get("ref"): entity
        for entity in frame["public"]["entities"]
        if entity.get("ref")
    }
    for candidate in frame["legal"]["candidates"]:
        if candidate["verb"] != verb:
            continue
        if source_content_id is None or any(
            entities.get(ref, {}).get("content_id") == source_content_id
            for ref in candidate.get("source_refs", [])
        ):
            return candidate
    raise AssertionError(f"No legal {verb} action for {source_content_id}")


def player_entity(frame):
    return next(
        entity
        for entity in frame["public"]["entities"]
        if entity.get("entity_type") == "player"
    )


def enemy_hp_by_content(frame):
    return Counter(
        {
            entity["content_id"]: entity["hp"]
            for entity in frame["public"]["entities"]
            if entity.get("entity_type") == "enemy"
        }
    )


def enemy_state(prediction, model):
    return sorted(
        (
            model.refs[ref]["content_id"],
            values[0],
            values[1],
        )
        for ref, values in prediction.state.enemies.items()
    )


def observed_enemy_state(frame):
    return sorted(
        (entity["content_id"], entity["hp"], entity["block"])
        for entity in frame["public"]["entities"]
        if entity.get("entity_type") == "enemy"
    )


def potion_counts(frame):
    return Counter(
        entity["content_id"]
        for entity in frame["public"]["entities"]
        if entity.get("entity_type") == "potion"
    )


def predicted_potion_counts(prediction, model):
    return Counter(model.refs[ref]["content_id"] for ref in prediction.state.potions)


def power_stack(frame, content_id, owner_ref):
    return sum(
        entity["stacks"]
        for entity in frame["public"]["entities"]
        if entity.get("entity_type") == "power"
        and entity.get("content_id") == content_id
        and entity.get("owner_ref") == owner_ref
    )


def predicted_zone_counts(prediction, frame, attribute):
    refs = {
        entity.get("ref"): entity.get("content_id")
        for entity in frame["public"]["entities"]
        if entity.get("entity_type") == "card"
    }
    return Counter(refs[ref] for ref in getattr(prediction.state, attribute))


def observed_zone_counts(frame, zone):
    return Counter(
        entity["content_id"]
        for entity in frame["public"]["entities"]
        if entity.get("entity_type") == "card" and entity.get("zone") == zone
    )


@unittest.skipUnless(
    ENABLED, "set STS2_NATIVE_EFFECT_TESTS=1 to launch the native game"
)
class NativeSecondWindTests(unittest.TestCase):
    def assert_prediction_matches_native(self, other_cards):
        deck = ["SECOND_WIND", *other_cards]
        self.assertEqual(len(deck), 5)
        with CliEngine() as engine:
            frame = start_combat(engine, deck, f"second_wind_{len(other_cards)}")
            candidate = card_candidate(frame, "CARD.SECOND_WIND")
            prediction = CombatModel(frame, effect_profile="second_wind").predict(
                candidate
            )
            self.assertIsNotNone(prediction.state)
            self.assertEqual(prediction.coverage, "approximate")

            after = engine.send(execution_command(frame, candidate["candidate_ref"]))
            self.assertEqual(after.get("type"), "decision_frame", after)
            self.assertIs(after["contract"]["training_ready"], False)
            native_player = next(
                entity
                for entity in after["public"]["entities"]
                if entity.get("entity_type") == "player"
            )

            self.assertEqual(prediction.state.hp, native_player["hp"])
            self.assertEqual(prediction.state.block, native_player["block"])
            self.assertEqual(prediction.state.energy, native_player["energy"])
            self.assertEqual(
                predicted_zone_counts(prediction, frame, "hand"),
                observed_zone_counts(after, "hand"),
            )
            self.assertEqual(
                predicted_zone_counts(prediction, frame, "discard"),
                observed_zone_counts(after, "discard_pile"),
            )
            self.assertEqual(
                predicted_zone_counts(prediction, frame, "exhaust"),
                observed_zone_counts(after, "exhaust_pile"),
            )

    def test_zero_other_non_attack_cards(self):
        self.assert_prediction_matches_native(["STRIKE_IRONCLAD"] * 4)

    def test_one_other_non_attack_card(self):
        self.assert_prediction_matches_native(
            ["DEFEND_IRONCLAD", "STRIKE_IRONCLAD", "STRIKE_IRONCLAD", "ANGER"]
        )

    def test_multiple_other_non_attack_cards(self):
        self.assert_prediction_matches_native(
            ["DEFEND_IRONCLAD", "DEFEND_IRONCLAD", "DEFEND_IRONCLAD", "STRIKE_IRONCLAD"]
        )

    def test_mixed_attack_and_non_attack_cards(self):
        self.assert_prediction_matches_native(
            ["DEFEND_IRONCLAD", "STRIKE_IRONCLAD", "ANGER", "STRIKE_IRONCLAD"]
        )


@unittest.skipUnless(
    ENABLED, "set STS2_NATIVE_EFFECT_TESTS=1 to launch the native game"
)
class NativeSpiteTests(unittest.TestCase):
    def test_spite_repeats_after_hemo_and_healing_then_resets_next_turn(self):
        deck = [
            "HEMOKINESIS",
            "SPITE",
            "STRIKE_IRONCLAD",
            "DEFEND_IRONCLAD",
            "ANGER",
        ]
        with CliEngine() as engine:
            frame = start_combat(
                engine, deck, "spite_hemo_heal_turn", potions=("BLOOD_POTION",)
            )
            self.assertIs(player_entity(frame)["lost_hp_this_turn"], False)

            hemo = card_candidate(frame, "CARD.HEMOKINESIS")
            hemo_prediction = CombatModel(frame, effect_profile="spite").predict(hemo)
            after_hemo = engine.send(execution_command(frame, hemo["candidate_ref"]))
            self.assertIs(player_entity(after_hemo)["lost_hp_this_turn"], True)
            self.assertEqual(player_entity(after_hemo)["hp"], hemo_prediction.state.hp)
            self.assertIs(hemo_prediction.state.lost_hp_this_turn, True)

            potion = candidate_with_verb(after_hemo, "USE_POTION")
            after_heal = engine.send(
                execution_command(after_hemo, potion["candidate_ref"])
            )
            self.assertGreater(
                player_entity(after_heal)["hp"], player_entity(after_hemo)["hp"]
            )
            self.assertIs(player_entity(after_heal)["lost_hp_this_turn"], True)

            spite = card_candidate(after_heal, "CARD.SPITE")
            model = CombatModel(after_heal, effect_profile="spite")
            prediction = model.predict(spite)
            self.assertIs(prediction.state.lost_hp_this_turn, True)
            after_spite = engine.send(
                execution_command(after_heal, spite["candidate_ref"])
            )
            self.assertEqual(
                enemy_hp_by_content(after_spite),
                Counter(
                    {
                        entity["content_id"]: hp_block[0]
                        for ref, hp_block in prediction.state.enemies.items()
                        if (entity := model.refs[ref]).get("entity_type") == "enemy"
                    }
                ),
            )
            self.assertIs(player_entity(after_spite)["lost_hp_this_turn"], True)

            end_turn = candidate_with_verb(after_spite, "END_TURN")
            next_turn = engine.send(
                execution_command(after_spite, end_turn["candidate_ref"])
            )
            self.assertEqual(next_turn.get("type"), "decision_frame", next_turn)
            self.assertIs(player_entity(next_turn)["lost_hp_this_turn"], False)

            next_hemo = card_candidate(next_turn, "CARD.HEMOKINESIS")
            after_next_hemo = engine.send(
                execution_command(next_turn, next_hemo["candidate_ref"])
            )
            self.assertIs(player_entity(after_next_hemo)["lost_hp_this_turn"], True)

            # The headless RunManager does not allow start_run re-entry in one
            # process. Production creates a fresh CliEngine for each attempt,
            # so verify cross-run isolation through that actual lifecycle.
            engine.close()
            with CliEngine() as fresh_engine:
                fresh_run = start_combat(
                    fresh_engine, deck, "spite_fresh_engine_run_reset"
                )
                self.assertIs(player_entity(fresh_run)["lost_hp_this_turn"], False)
                fresh_spite = card_candidate(fresh_run, "CARD.SPITE")
                fresh_model = CombatModel(fresh_run, effect_profile="spite")
                fresh_prediction = fresh_model.predict(fresh_spite)
                self.assertIs(fresh_prediction.state.lost_hp_this_turn, False)
                source = next(
                    entity
                    for entity in fresh_run["public"]["entities"]
                    if entity.get("ref") == fresh_spite["source_refs"][0]
                )
                before_hp = enemy_hp_by_content(fresh_run)
                after_fresh_spite = fresh_engine.send(
                    execution_command(fresh_run, fresh_spite["candidate_ref"])
                )
                self.assertEqual(
                    sum(before_hp.values())
                    - sum(enemy_hp_by_content(after_fresh_spite).values()),
                    source["stats"]["Damage"],
                )
                self.assertEqual(
                    enemy_hp_by_content(after_fresh_spite),
                    Counter(
                        {
                            entity["content_id"]: hp_block[0]
                            for ref, hp_block in fresh_prediction.state.enemies.items()
                            if (entity := fresh_model.refs[ref]).get("entity_type")
                            == "enemy"
                        }
                    ),
                )

    def test_upgraded_spite_without_hp_loss_uses_only_one_hit(self):
        deck = [
            "SPITE",
            "STRIKE_IRONCLAD",
            "DEFEND_IRONCLAD",
            "ANGER",
            "HEMOKINESIS",
        ]
        with CliEngine() as engine:
            engine.send(
                {
                    "cmd": "start_run",
                    "character": "Ironclad",
                    "seed": "spite_upgraded_no_loss",
                    "ascension": 0,
                    "lang": "en",
                    "decision_protocol": False,
                }
            )
            engine.send(
                {
                    "cmd": "set_player",
                    "hp": 70,
                    "max_hp": 80,
                    "gold": 999,
                    "relics": [],
                    "potions": [],
                    "deck": deck,
                }
            )
            rest = engine.send({"cmd": "enter_room", "type": "rest_site"})
            smith = next(
                option for option in rest["options"] if option["option_id"] == "SMITH"
            )
            selection = engine.send(
                {
                    "cmd": "action",
                    "action": "choose_option",
                    "args": {"option_index": smith["index"]},
                }
            )
            spite_index = next(
                card["index"]
                for card in selection["cards"]
                if card["id"] == "CARD.SPITE"
            )
            engine.send(
                {
                    "cmd": "action",
                    "action": "select_cards",
                    "args": {"indices": str(spite_index)},
                }
            )
            engine.send(
                {
                    "cmd": "enter_room",
                    "type": "combat",
                    "encounter": "SHRINKER_BEETLE_WEAK",
                }
            )
            frame = engine.send({"cmd": "advance_to_boundary"})
            self.assertIs(frame["contract"]["training_ready"], False)
            self.assertIs(player_entity(frame)["lost_hp_this_turn"], False)

            spite = card_candidate(frame, "CARD.SPITE")
            source_ref = spite["source_refs"][0]
            source = next(
                entity
                for entity in frame["public"]["entities"]
                if entity.get("ref") == source_ref
            )
            self.assertTrue(source["upgraded"])
            self.assertEqual(source["stats"]["Repeat"], 3)
            model = CombatModel(frame, effect_profile="spite")
            prediction = model.predict(spite)
            self.assertEqual(prediction.state.lost_hp_this_turn, False)
            before_hp = enemy_hp_by_content(frame)
            after = engine.send(execution_command(frame, spite["candidate_ref"]))
            after_hp = enemy_hp_by_content(after)
            damage = source["stats"]["Damage"]
            self.assertEqual(sum(before_hp.values()) - sum(after_hp.values()), damage)
            self.assertEqual(
                after_hp,
                Counter(
                    {
                        entity["content_id"]: hp_block[0]
                        for ref, hp_block in prediction.state.enemies.items()
                        if (entity := model.refs[ref]).get("entity_type") == "enemy"
                    }
                ),
            )


@unittest.skipUnless(
    ENABLED, "set STS2_NATIVE_EFFECT_TESTS=1 to launch the native game"
)
class NativeRelicModelTests(unittest.TestCase):
    def test_matrix_relics_keep_strike_prediction_modeled(self):
        relic_seeds = {
            "FISHING_ROD": "5ADF5172BEF3142B",
            "LOST_COFFER": "BD1D31C2468C7C1F",
            "KALEIDOSCOPE": "D011001234ABCDEF",
        }
        planner = PublicPlanner()

        for relic_id, seed in relic_seeds.items():
            with self.subTest(relic=relic_id), CliEngine() as engine:
                frame = engine.reset("Ironclad", seed, 0)
                initial_gold = player_entity(frame)["gold"]
                engine.send({"cmd": "set_player", "gold": initial_gold})
                frame = engine.send({"cmd": "advance_to_boundary"})
                self.assertIs(frame["contract"]["training_ready"], False)

                matched = False
                for _ in range(200):
                    if frame.get("boundary") == "waiting":
                        frame = engine.send({"cmd": "advance_to_boundary"})
                        continue
                    if frame.get("boundary") == "terminal":
                        break

                    public = frame["public"]
                    relic_present = any(
                        entity.get("entity_type") == "relic"
                        and entity.get("content_id") == f"RELIC.{relic_id}"
                        for entity in public.get("entities", [])
                    )
                    if relic_present and public.get("phase") == "combat":
                        try:
                            strike = card_candidate(frame, "CARD.STRIKE_IRONCLAD")
                        except AssertionError:
                            strike = None
                        if strike is not None:
                            model = CombatModel(
                                clean_frame(frame), effect_profile="cards"
                            )
                            prediction = model.predict(strike)
                            if (
                                prediction.state is not None
                                and prediction.coverage == "modeled"
                            ):
                                after = engine.send(
                                    execution_command(frame, strike["candidate_ref"])
                                )
                                self.assertEqual(
                                    after.get("type"), "decision_frame", after
                                )
                                self.assertIs(
                                    after["contract"]["training_ready"], False
                                )
                                native_player = player_entity(after)
                                self.assertEqual(
                                    native_player["hp"], prediction.state.hp
                                )
                                self.assertEqual(
                                    native_player["block"], prediction.state.block
                                )
                                self.assertEqual(
                                    native_player["energy"], prediction.state.energy
                                )
                                self.assertEqual(
                                    observed_enemy_state(after),
                                    enemy_state(prediction, model),
                                )
                                matched = True
                                break

                    candidate, _ = planner.choose(clean_frame(frame))
                    frame = engine.send(
                        execution_command(frame, candidate["candidate_ref"])
                    )

                self.assertTrue(
                    matched,
                    f"No modeled Strike with owned {relic_id} within 200 decisions",
                )


@unittest.skipUnless(
    ENABLED, "set STS2_NATIVE_EFFECT_TESTS=1 to launch the native game"
)
class NativePotionAndCardTests(unittest.TestCase):
    def test_fire_block_and_weak_potions_match_native_targets_and_consumption(self):
        with CliEngine() as engine:
            frame = start_combat(
                engine,
                ["TWIN_STRIKE"],
                "native_potions_fire_block_weak",
                potions=("FIRE_POTION", "BLOCK_POTION", "WEAK_POTION"),
            )
            for potion_id in (
                "POTION.FIRE_POTION",
                "POTION.BLOCK_POTION",
                "POTION.WEAK_POTION",
            ):
                candidate = candidate_with_verb(frame, "USE_POTION", potion_id)
                model = CombatModel(frame, effect_profile="potions")
                prediction = model.predict(candidate)
                self.assertIsNotNone(prediction.state, potion_id)
                if potion_id == "POTION.BLOCK_POTION":
                    self.assertEqual(candidate["target_refs"], ["player"])
                else:
                    self.assertTrue(
                        set(candidate["target_refs"]) & set(model.root.enemies)
                    )
                after = engine.send(
                    execution_command(frame, candidate["candidate_ref"])
                )
                self.assertEqual(after.get("type"), "decision_frame", after)
                self.assertEqual(
                    player_entity(after)["hp"], prediction.state.hp, potion_id
                )
                self.assertEqual(
                    player_entity(after)["block"], prediction.state.block, potion_id
                )
                self.assertEqual(
                    player_entity(after)["energy"], prediction.state.energy, potion_id
                )
                self.assertEqual(
                    observed_enemy_state(after),
                    enemy_state(prediction, model),
                    potion_id,
                )
                expected_potions = potion_counts(frame) - Counter({potion_id: 1})
                self.assertEqual(potion_counts(after), expected_potions, potion_id)
                self.assertEqual(
                    predicted_potion_counts(prediction, model), expected_potions
                )
                if potion_id == "POTION.WEAK_POTION":
                    target = candidate["target_refs"][0]
                    self.assertEqual(
                        power_stack(after, "POWER.WEAK_POWER", target),
                        prediction.state.weak[target],
                    )
                frame = after

    def test_strength_hemo_and_twin_strike_use_sequential_native_state(self):
        with CliEngine() as engine:
            frame = start_combat(
                engine,
                ["INFLAME", "HEMOKINESIS", "TWIN_STRIKE"],
                "native_strength_hemo_twin",
            )
            for content_id in (
                "CARD.INFLAME",
                "CARD.HEMOKINESIS",
                "CARD.TWIN_STRIKE",
            ):
                candidate = card_candidate(frame, content_id)
                model = CombatModel(frame, effect_profile="cards")
                prediction = model.predict(candidate)
                self.assertIsNotNone(prediction.state, content_id)
                after = engine.send(
                    execution_command(frame, candidate["candidate_ref"])
                )
                self.assertEqual(after.get("type"), "decision_frame", after)
                self.assertEqual(
                    player_entity(after)["hp"], prediction.state.hp, content_id
                )
                self.assertEqual(
                    player_entity(after)["block"], prediction.state.block, content_id
                )
                self.assertEqual(
                    player_entity(after)["energy"], prediction.state.energy, content_id
                )
                self.assertEqual(
                    observed_enemy_state(after),
                    enemy_state(prediction, model),
                    content_id,
                )
                if content_id == "CARD.INFLAME":
                    self.assertEqual(
                        power_stack(after, "POWER.STRENGTH_POWER", "player"),
                        prediction.state.strength,
                    )
                frame = after

    def test_dexterity_potion_then_defend_matches_native_block(self):
        with CliEngine() as engine:
            frame = start_combat(
                engine,
                ["DEFEND_IRONCLAD"],
                "native_dexterity_block",
                potions=("DEXTERITY_POTION",),
            )
            dexterity = candidate_with_verb(frame, "USE_POTION")
            after_potion = engine.send(
                execution_command(frame, dexterity["candidate_ref"])
            )
            self.assertEqual(after_potion.get("type"), "decision_frame", after_potion)
            self.assertGreater(
                power_stack(after_potion, "POWER.DEXTERITY_POWER", "player"), 0
            )

            defend = card_candidate(after_potion, "CARD.DEFEND_IRONCLAD")
            model = CombatModel(after_potion, effect_profile="cards")
            prediction = model.predict(defend)
            self.assertIsNotNone(prediction.state)
            after_defend = engine.send(
                execution_command(after_potion, defend["candidate_ref"])
            )
            self.assertEqual(
                player_entity(after_defend)["block"], prediction.state.block
            )

    def test_strength_potion_then_zero_block_body_slam_matches_native(self):
        with CliEngine() as engine:
            frame = start_combat(
                engine,
                ["BODY_SLAM"],
                "native_strength_zero_block_body_slam",
                potions=("STRENGTH_POTION",),
            )
            strength = candidate_with_verb(frame, "USE_POTION")
            after_potion = engine.send(
                execution_command(frame, strength["candidate_ref"])
            )
            self.assertGreater(
                power_stack(after_potion, "POWER.STRENGTH_POWER", "player"), 0
            )
            self.assertEqual(player_entity(after_potion)["block"], 0)

            body_slam = card_candidate(after_potion, "CARD.BODY_SLAM")
            model = CombatModel(after_potion, effect_profile="cards")
            prediction = model.predict(body_slam)
            self.assertIsNotNone(prediction.state)
            after = engine.send(
                execution_command(after_potion, body_slam["candidate_ref"])
            )
            self.assertEqual(
                observed_enemy_state(after), enemy_state(prediction, model)
            )

    def test_frail_from_retained_shame_reduces_defend_like_native(self):
        with CliEngine() as engine:
            frame = start_combat(
                engine,
                [
                    "SHAME",
                    *["STRIKE_IRONCLAD"] * 4,
                    *["DEFEND_IRONCLAD"] * 5,
                ],
                "frail10e",
                encounter="FLYCONID_NORMAL",
                draw_order=["DEFEND_IRONCLAD"] * 5
                + ["STRIKE_IRONCLAD"] * 4
                + ["SHAME"],
            )
            end_turn = candidate_with_verb(frame, "END_TURN")
            after_turn = engine.send(
                execution_command(frame, end_turn["candidate_ref"])
            )
            self.assertGreater(
                power_stack(after_turn, "POWER.FRAIL_POWER", "player"), 0
            )
            self.assertFalse(
                any(
                    entity.get("entity_type") == "power"
                    and entity.get("content_id")
                    not in {
                        "POWER.FRAIL_POWER",
                    }
                    for entity in after_turn["public"]["entities"]
                ),
                "This fixture should isolate Frail from other power hooks",
            )

            defend = card_candidate(after_turn, "CARD.DEFEND_IRONCLAD")
            model = CombatModel(after_turn, effect_profile="cards")
            prediction = model.predict(defend)
            self.assertIsNotNone(prediction.state)
            self.assertEqual(prediction.state.block, 3)
            after_defend = engine.send(
                execution_command(after_turn, defend["candidate_ref"])
            )
            self.assertEqual(
                player_entity(after_defend)["block"], prediction.state.block
            )

    def test_weak_and_vulnerable_potions_match_twin_strike_rounding(self):
        with CliEngine() as engine:
            frame = start_combat(
                engine,
                ["TWIN_STRIKE"],
                "native_merc_weak_vulnerable_twin",
                potions=("VULNERABLE_POTION",),
                encounter="GREMLIN_MERC_NORMAL",
            )
            for _ in range(2):
                end_turn = candidate_with_verb(frame, "END_TURN")
                frame = engine.send(execution_command(frame, end_turn["candidate_ref"]))
                self.assertEqual(frame.get("type"), "decision_frame", frame)

            # Gremlin Merc's native GIMME -> DOUBLE_SMASH -> HEHE sequence
            # applies Weak to the player after the second end turn. It also has
            # unrelated Surprise/Thievery hooks, so the strict model must
            # report unknown for this complete observation.
            self.assertEqual(power_stack(frame, "POWER.WEAK_POWER", "player"), 2)

            vulnerable = candidate_with_verb(frame, "USE_POTION")
            target = vulnerable["target_refs"][0]
            before_vulnerable = engine.send(
                execution_command(frame, vulnerable["candidate_ref"])
            )
            self.assertEqual(before_vulnerable.get("type"), "decision_frame")
            self.assertGreater(
                power_stack(before_vulnerable, "POWER.VULNERABLE_POWER", target), 0
            )
            self.assertGreater(
                power_stack(before_vulnerable, "POWER.WEAK_POWER", "player"), 0
            )

            twin_strike = card_candidate(before_vulnerable, "CARD.TWIN_STRIKE")
            source = next(
                entity
                for entity in before_vulnerable["public"]["entities"]
                if entity.get("ref") == twin_strike["source_refs"][0]
            )
            model = CombatModel(before_vulnerable, effect_profile="cards")
            prediction = model.predict(twin_strike)
            self.assertIsNone(prediction.state)
            self.assertEqual(prediction.reason, "unsupported_hooks")

            # Verify native per-hit rounding under both visible modifiers.
            # This is a native outcome check; the complete Gremlin frame stays
            # opaque to the production effect model because of its other hooks.
            before_hp = enemy_hp_by_content(before_vulnerable)
            after = engine.send(
                execution_command(before_vulnerable, twin_strike["candidate_ref"])
            )
            after_hp = enemy_hp_by_content(after)
            base_damage = source["stats"]["Damage"]
            per_hit = math.floor(base_damage * 0.75 * 1.5)
            self.assertEqual((base_damage, per_hit), (5, 5))
            self.assertEqual(
                sum(before_hp.values()) - sum(after_hp.values()), 2 * per_hit
            )


if __name__ == "__main__":
    unittest.main()
