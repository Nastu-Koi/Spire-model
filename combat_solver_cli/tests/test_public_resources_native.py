"""Exercise experimental route and supply choices in the native game engine.

These debug runs are deliberately not training-ready and must never be exported
as demonstrations. Enable them with STS2_NATIVE_RESOURCE_TESTS=1.
"""

import os
import time
import unittest
from collections import deque

from combat_solver_cli.public_search import PublicPlanner
from model.engine import CliEngine
from model.protocol import execution_command

ENABLED = os.environ.get("STS2_NATIVE_RESOURCE_TESTS") == "1"
PLANNER = PublicPlanner(route_resources=True)


def debug_room(engine, room_type, *, hp=70, gold=999, deck=None):
    started = engine.send(
        {
            "cmd": "start_run",
            "character": "Ironclad",
            "seed": f"native_resource_{room_type}_{hp}_{gold}",
            "ascension": 0,
            "lang": "en",
            "decision_protocol": True,
        }
    )
    if started.get("type") == "error":
        raise AssertionError(f"Native debug run setup failed: {started}")
    changed = engine.send(
        {
            "cmd": "set_player",
            "hp": hp,
            "max_hp": 80,
            "gold": gold,
            "relics": [],
            "potions": [],
            "deck": deck or ["STRIKE_IRONCLAD", "DEFEND_IRONCLAD"],
        }
    )
    if changed.get("type") == "error":
        raise AssertionError(f"Could not set native player fixture: {changed}")
    frame = engine.send(
        {"cmd": "enter_room", "type": room_type, "decision_protocol": True}
    )
    if frame.get("type") != "decision_frame":
        raise AssertionError(f"Expected native {room_type} decision frame, got {frame}")
    if frame["contract"]["training_ready"] is not False:
        raise AssertionError("Debug native fixture must not be training-ready")
    return frame


def plan(frame):
    action, diagnostic = PLANNER.choose(frame)
    expected = {c["candidate_ref"] for c in frame["legal"]["candidates"]}
    observed = {s["candidate_ref"] for s in diagnostic["candidate_scores"]}
    if observed != expected:
        raise AssertionError(
            f"Planner did not score every native candidate: {observed} != {expected}"
        )
    if action["candidate_ref"] not in expected:
        raise AssertionError(
            "Planner returned an action outside native legal candidates"
        )
    return action, diagnostic


def source_entity(frame, candidate):
    refs = {e.get("ref"): e for e in frame["public"]["entities"]}
    source_refs = candidate.get("source_refs") or []
    return refs.get(source_refs[0], {}) if source_refs else {}


def player(frame):
    return next(e for e in frame["public"]["entities"] if e["entity_type"] == "player")


def settle(engine, frame, limit=40):
    """Poll native continuations until they publish the next legal decision."""
    for _ in range(limit):
        if (
            frame.get("type") == "decision_frame"
            and frame.get("boundary") == "decision"
            and frame.get("legal", {}).get("candidates")
        ):
            return frame
        if frame.get("boundary") in {"terminal", "error"}:
            raise AssertionError(
                f"Native run ended before the expected decision: {frame}"
            )
        time.sleep(0.05)
        frame = engine.send({"cmd": "advance_to_boundary"})
    raise AssertionError(f"Native engine did not settle at a decision: {frame}")


def nearest_rest_move(frame):
    """Choose a legal next node with the shortest public path to a rest site."""
    entities = {e.get("ref"): e for e in frame["public"]["entities"] if e.get("ref")}
    children = {}
    for relation in frame["public"].get("relations", []):
        if relation.get("role") == "map_edge":
            children.setdefault(relation["source"], []).append(relation["target"])
    targets = {
        ref
        for ref, entity in entities.items()
        if entity.get("entity_type") == "map_node"
        and entity.get("content_id", "").upper() in {"REST", "RESTSITE"}
    }

    def distance(start):
        pending = deque([(start, 0)])
        visited = set()
        while pending:
            ref, steps = pending.popleft()
            if ref in targets:
                return steps
            if ref in visited:
                continue
            visited.add(ref)
            pending.extend((child, steps + 1) for child in children.get(ref, []))
        return None

    moves = [c for c in frame["legal"]["candidates"] if c["verb"] == "MOVE_TO_NODE"]
    ranked = [
        (d, c)
        for c in moves
        if (d := distance((c.get("source_refs") or [None])[0])) is not None
    ]
    if not ranked:
        raise AssertionError("No legal map move can reach a visible rest site")
    return min(ranked, key=lambda item: item[0])[1]


@unittest.skipUnless(
    ENABLED, "set STS2_NATIVE_RESOURCE_TESTS=1 to launch the native game"
)
class NativeRouteResourceTests(unittest.TestCase):
    def test_public_map_scores_every_legal_move_before_native_travel(self):
        with CliEngine(timeout=60) as engine:
            frame = engine.send(
                {
                    "cmd": "start_run",
                    "character": "Ironclad",
                    "seed": "native_resource_map",
                    "ascension": 0,
                    "lang": "en",
                    "decision_protocol": True,
                }
            )
            changed = engine.send(
                {
                    "cmd": "set_player",
                    "hp": 70,
                    "max_hp": 80,
                    "gold": 999,
                    "relics": [],
                    "potions": [],
                    "deck": ["STRIKE_IRONCLAD", "DEFEND_IRONCLAD"],
                }
            )
            if changed.get("type") == "error":
                self.fail(f"Could not set native map fixture: {changed}")
            frame = engine.send({"cmd": "advance_to_boundary"})
            # The opening event is a genuine public decision. Resolve one
            # native option, then inspect and execute a legal map movement.
            self.assertEqual(frame.get("type"), "decision_frame", frame)
            self.assertIs(frame["contract"]["training_ready"], False)
            if frame["public"]["phase"] != "map":
                opening, _ = plan(frame)
                frame = engine.send(execution_command(frame, opening["candidate_ref"]))
            self.assertEqual(frame["public"]["phase"], "map")
            action, diagnostic = plan(frame)
            self.assertEqual(action["verb"], "MOVE_TO_NODE")
            self.assertEqual(
                len(diagnostic["candidate_scores"]), len(frame["legal"]["candidates"])
            )
            after = engine.send(execution_command(frame, action["candidate_ref"]))
            self.assertEqual(after.get("type"), "decision_frame", after)
            self.assertIs(after["contract"]["training_ready"], False)
            self.assertNotEqual(after["public"]["phase"], "map")

    def test_native_map_route_reaches_rest_site_and_heals(self):
        with CliEngine(timeout=60) as engine:
            frame = engine.send(
                {
                    "cmd": "start_run",
                    "character": "Ironclad",
                    "seed": "native_resource_map_to_rest",
                    "ascension": 0,
                    "lang": "en",
                    "decision_protocol": True,
                }
            )
            changed = engine.send(
                {
                    "cmd": "set_player",
                    "hp": 999,
                    "max_hp": 999,
                    "gold": 999,
                    "relics": [],
                    "potions": [],
                    "deck": ["BLUDGEON"] * 20,
                }
            )
            self.assertNotEqual(changed.get("type"), "error", changed)
            frame = settle(engine, engine.send({"cmd": "advance_to_boundary"}))
            if frame["public"]["phase"] == "event":
                opening, _ = plan(frame)
                frame = settle(
                    engine,
                    engine.send(execution_command(frame, opening["candidate_ref"])),
                )

            reached_rest = False
            for _ in range(120):
                phase = frame["public"]["phase"]
                if phase == "rest_site":
                    reached_rest = True
                    break
                if phase == "map":
                    planned, diagnostics = plan(frame)
                    self.assertEqual(planned["verb"], "MOVE_TO_NODE")
                    self.assertEqual(
                        len(diagnostics["candidate_scores"]),
                        len(frame["legal"]["candidates"]),
                    )
                    chosen = nearest_rest_move(frame)
                    target = source_entity(frame, chosen)
                    if target.get("content_id", "").upper() == "RESTSITE":
                        engine.send(
                            {
                                "cmd": "set_player",
                                "hp": 20,
                                "max_hp": 80,
                                "gold": 999,
                            }
                        )
                        frame = settle(
                            engine, engine.send({"cmd": "advance_to_boundary"})
                        )
                        chosen = nearest_rest_move(frame)
                        target = source_entity(frame, chosen)
                        self.assertEqual(
                            target.get("content_id", "").upper(), "RESTSITE"
                        )
                    frame = settle(
                        engine,
                        engine.send(execution_command(frame, chosen["candidate_ref"])),
                    )
                    continue
                if phase == "combat":
                    if player(frame)["hp"] < 100:
                        engine.send(
                            {
                                "cmd": "set_player",
                                "hp": 999,
                                "max_hp": 999,
                                "gold": player(frame)["gold"],
                            }
                        )
                        frame = settle(
                            engine, engine.send({"cmd": "advance_to_boundary"})
                        )
                    refs = {e.get("ref"): e for e in frame["public"]["entities"]}
                    plays = [
                        c
                        for c in frame["legal"]["candidates"]
                        if c["verb"] == "PLAY_CARD"
                    ]
                    if plays:
                        action = max(
                            plays,
                            key=lambda c: (
                                refs.get((c.get("source_refs") or [None])[0], {})
                                .get("stats", {})
                                .get("Damage", 0)
                            ),
                        )
                    else:
                        action = next(
                            c
                            for c in frame["legal"]["candidates"]
                            if c["verb"] == "END_TURN"
                        )
                    frame = settle(
                        engine,
                        engine.send(execution_command(frame, action["candidate_ref"])),
                    )
                    continue
                action, _ = plan(frame)
                frame = settle(
                    engine,
                    engine.send(execution_command(frame, action["candidate_ref"])),
                )

            self.assertTrue(
                reached_rest, "The bounded public route did not reach a rest site"
            )
            self.assertIs(frame["contract"]["training_ready"], False)
            before_hp = player(frame)["hp"]
            action, diagnostics = plan(frame)
            self.assertEqual(
                len(diagnostics["candidate_scores"]), len(frame["legal"]["candidates"])
            )
            self.assertEqual(source_entity(frame, action).get("content_id"), "HEAL")
            after = settle(
                engine,
                engine.send(execution_command(frame, action["candidate_ref"])),
            )
            self.assertEqual(after["public"]["phase"], "map")
            self.assertGreater(player(after)["hp"], before_hp)

    def test_low_hp_heals_at_rest_site_and_hp_increases(self):
        with CliEngine(timeout=60) as engine:
            frame = debug_room(engine, "rest_site", hp=20)
            action, diagnostic = plan(frame)
            self.assertEqual(
                len(diagnostic["candidate_scores"]), len(frame["legal"]["candidates"])
            )
            self.assertEqual(source_entity(frame, action).get("content_id"), "HEAL")
            after = engine.send(execution_command(frame, action["candidate_ref"]))
            self.assertEqual(after.get("type"), "decision_frame", after)
            self.assertEqual(after["public"]["phase"], "map")
            self.assertGreater(player(after)["hp"], player(frame)["hp"])
            self.assertIn(
                "MOVE_TO_NODE", {c["verb"] for c in after["legal"]["candidates"]}
            )

    def test_healthy_player_smiths_and_native_card_becomes_upgraded(self):
        with CliEngine(timeout=60) as engine:
            frame = debug_room(engine, "rest_site", hp=80)
            action, _ = plan(frame)
            self.assertEqual(source_entity(frame, action).get("content_id"), "SMITH")
            selection = engine.send(execution_command(frame, action["candidate_ref"]))
            self.assertEqual(selection.get("type"), "decision_frame", selection)
            self.assertEqual(selection["public"]["phase"], "card_select")
            choose_card, diagnostics = plan(selection)
            self.assertEqual(
                len(diagnostics["candidate_scores"]),
                len(selection["legal"]["candidates"]),
            )
            self.assertEqual(choose_card["verb"], "SELECT_ONE")
            chosen_content = source_entity(selection, choose_card).get("content_id")
            upgraded = engine.send(
                execution_command(selection, choose_card["candidate_ref"])
            )
            self.assertEqual(upgraded.get("type"), "decision_frame", upgraded)
            self.assertEqual(upgraded["public"]["phase"], "card_select")
            finish, _ = plan(upgraded)
            self.assertEqual(finish["verb"], "FINISH_SELECTION")
            upgraded = engine.send(execution_command(upgraded, finish["candidate_ref"]))
            self.assertEqual(upgraded.get("type"), "decision_frame", upgraded)
            selected = next(
                (
                    e
                    for e in upgraded["public"]["entities"]
                    if e.get("entity_type") == "card"
                    and e.get("content_id") == chosen_content
                    and e.get("zone") == "deck"
                ),
                None,
            )
            self.assertIsNotNone(
                selected, "Smithing should keep the chosen card observable"
            )
            self.assertIs(selected["upgraded"], True)

    def test_shop_removes_visible_curse_and_spends_gold(self):
        with CliEngine(timeout=60) as engine:
            frame = debug_room(
                engine,
                "shop",
                hp=70,
                gold=999,
                deck=["STRIKE_IRONCLAD", "SHAME"],
            )
            action, diagnostics = plan(frame)
            self.assertEqual(
                len(diagnostics["candidate_scores"]), len(frame["legal"]["candidates"])
            )
            self.assertEqual(action["verb"], "BUY_ITEM")
            self.assertEqual(
                source_entity(frame, action).get("content_id"),
                "MerchantCardRemovalEntry",
            )
            spent = player(frame)["gold"]
            selection = engine.send(execution_command(frame, action["candidate_ref"]))
            self.assertEqual(selection.get("type"), "decision_frame", selection)
            self.assertEqual(selection["public"]["phase"], "card_select")
            curse, selection_diagnostics = plan(selection)
            self.assertEqual(
                len(selection_diagnostics["candidate_scores"]),
                len(selection["legal"]["candidates"]),
            )
            self.assertEqual(curse["verb"], "SELECT_ONE")
            self.assertEqual(
                source_entity(selection, curse).get("content_id"), "CARD.SHAME"
            )
            after = engine.send(execution_command(selection, curse["candidate_ref"]))
            self.assertEqual(after.get("type"), "decision_frame", after)
            self.assertEqual(after["public"]["phase"], "card_select")
            finish, finish_diagnostics = plan(after)
            self.assertEqual(
                len(finish_diagnostics["candidate_scores"]),
                len(after["legal"]["candidates"]),
            )
            self.assertEqual(finish["verb"], "FINISH_SELECTION")
            after = engine.send(execution_command(after, finish["candidate_ref"]))
            self.assertEqual(after.get("type"), "decision_frame", after)
            deck = [
                e["content_id"]
                for e in after["public"]["entities"]
                if e.get("entity_type") == "card" and e.get("zone") == "deck"
            ]
            self.assertNotIn("CARD.SHAME", deck)
            self.assertLess(player(after)["gold"], spent)

    def test_shop_without_gold_leaves_and_native_room_closes(self):
        with CliEngine(timeout=60) as engine:
            frame = debug_room(engine, "shop", hp=70, gold=0)
            action, diagnostics = plan(frame)
            self.assertEqual(
                len(diagnostics["candidate_scores"]), len(frame["legal"]["candidates"])
            )
            self.assertEqual(action["verb"], "LEAVE_ROOM")
            after = engine.send(execution_command(frame, action["candidate_ref"]))
            self.assertEqual(after.get("type"), "decision_frame", after)
            self.assertEqual(after["public"]["phase"], "map")

    def test_shop_buys_card_when_public_value_exceeds_leave(self):
        with CliEngine(timeout=60) as engine:
            frame = debug_room(
                engine,
                "shop",
                hp=70,
                gold=999,
                deck=["STRIKE_IRONCLAD", "DEFEND_IRONCLAD"],
            )
            action, diagnostics = plan(frame)
            self.assertEqual(
                len(diagnostics["candidate_scores"]), len(frame["legal"]["candidates"])
            )
            self.assertEqual(action["verb"], "BUY_ITEM")
            offered = source_entity(frame, action)
            relation = next(
                r
                for r in frame["public"]["relations"]
                if r.get("role") == "offers" and r.get("source") == offered.get("ref")
            )
            item = next(
                e
                for e in frame["public"]["entities"]
                if e.get("ref") == relation["target"]
            )
            gold_before = player(frame)["gold"]
            after = engine.send(execution_command(frame, action["candidate_ref"]))
            self.assertEqual(after.get("type"), "decision_frame", after)
            self.assertEqual(after["public"]["phase"], "shop")
            self.assertLess(player(after)["gold"], gold_before)
            self.assertTrue(
                any(
                    e.get("entity_type") == "card"
                    and e.get("content_id") == item["content_id"]
                    and e.get("zone") == "deck"
                    for e in after["public"]["entities"]
                ),
                f"Purchased card {item['content_id']} should enter the native deck",
            )


if __name__ == "__main__":
    unittest.main()
