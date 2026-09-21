"""Native engine regressions; these runs exercise no synthetic transitions."""
import json
from pathlib import Path
import time

import pytest

from model.engine import CliEngine
from model.protocol import CHARACTERS, execution_command, validate_frame
from model.rewards import MilestoneLedger

pytestmark = pytest.mark.engine


@pytest.fixture
def engine():
    root = Path(__file__).resolve().parents[1] / "sts2-cli"
    if not (root / "src/Sts2Headless/bin/Debug/net9.0/Sts2Headless.dll").exists():
        pytest.skip("Build the local engine first")
    with CliEngine(root=root) as client:
        yield client


def settle(engine, frame):
    deadline = time.monotonic() + 20
    while frame["boundary"] == "waiting" and time.monotonic() < deadline:
        time.sleep(.01)
        frame = engine.send({"cmd": "advance_to_boundary"})
    validate_frame(frame, allow_prototype=True)
    assert frame["boundary"] != "waiting", frame
    return frame


def choose(engine, frame, verb=None, content=None):
    entities = {e.get("ref"): e for e in frame["public"]["entities"] if e.get("ref")}
    candidate = next(c for c in frame["legal"]["candidates"]
                     if (verb is None or c["verb"] == verb)
                     and (content is None or any(entities[r].get("content_id") == content for r in c["source_refs"])))
    return settle(engine, engine.send(execution_command(frame, candidate["candidate_ref"])))


def fixture_room(engine, room, **args):
    engine.reset("Ironclad", "protocol-native-fixture")
    engine.send({"cmd": "set_player", "hp": 70, "max_hp": 80, "gold": 999})
    frame = engine.send({"cmd": "enter_room", "type": room, "decision_protocol": True, **args})
    assert frame["contract"]["training_ready"] is False  # Debug trajectories cannot enter PPO.
    return settle(engine, frame)


@pytest.mark.parametrize("character", CHARACTERS)
@pytest.mark.parametrize("seed_index", range(5))
def test_five_characters_complete_native_a10_runs(engine, character, seed_index):
    frame = engine.reset(character, f"native-regression-{character}-{seed_index}")
    ledger = MilestoneLedger()
    for _ in range(2500):
        validate_frame(frame)
        ledger.apply(frame.get("events", []))
        if frame["boundary"] == "terminal":
            assert frame["public"]["outcome"]["victory"] == ledger.victory_paid
            return
        if frame["boundary"] == "waiting":
            time.sleep(.01)
            frame = engine.send({"cmd": "advance_to_boundary"})
            continue
        command = execution_command(frame, frame["legal"]["candidates"][0]["candidate_ref"])
        frame = engine.send(command)
    pytest.fail("Native run did not reach a terminal outcome")


def test_smith_cancel_preserves_deck_and_rest_choice(engine):
    frame = fixture_room(engine, "rest_site")
    before = [e for e in frame["public"]["entities"] if e.get("zone") == "deck" and e["entity_type"] == "card"]
    frame = choose(engine, frame, "CHOOSE_REST_OPTION", "SMITH")
    assert frame["public"]["selection_context"]["operation"] == "upgrade"
    frame = choose(engine, frame, "SELECT_ONE")
    assert {c["verb"] for c in frame["legal"]["candidates"]} == {"FINISH_SELECTION", "CANCEL"}
    frame = choose(engine, frame, "CANCEL")
    assert frame["public"]["phase"] == "rest_site"
    assert not any(c["verb"] == "LEAVE_ROOM" for c in frame["legal"]["candidates"])
    after = [e for e in frame["public"]["entities"] if e.get("zone") == "deck" and e["entity_type"] == "card"]
    assert before == after


def test_treasure_requires_explicit_open_and_relic_choice(engine):
    frame = fixture_room(engine, "treasure")
    assert frame["public"]["phase"] == "treasure"
    assert not any(e.get("zone") == "treasure" for e in frame["public"]["entities"])
    frame = choose(engine, frame, "OPEN_CHEST")
    offered = next(e["content_id"] for e in frame["public"]["entities"] if e.get("zone") == "treasure")
    frame = choose(engine, frame, "TAKE_TREASURE_RELIC")
    assert frame["public"]["phase"] == "map"
    assert any(e["entity_type"] == "relic" and e["content_id"] == offered and e.get("owner_ref") == "player"
               for e in frame["public"]["entities"])


def test_crystal_fog_native_reveal_and_cache_reset(engine):
    frame = fixture_room(engine, "event", event="CRYSTAL_SPHERE")
    frame = choose(engine, frame, "CHOOSE_EVENT_OPTION")
    assert frame["public"]["phase"] == "crystal_sphere"
    cells = [e for e in frame["public"]["entities"] if e["entity_type"] == "board_cell"]
    hidden = [e for e in cells if not e["revealed"]]
    assert len(cells) == 121
    assert all(set(e) == {"entity_type", "ref", "x", "y", "revealed", "known"} for e in hidden)
    before = len(hidden)
    old = frame
    candidate = next(c for c in frame["legal"]["candidates"] if c["source_refs"] == ["tool:Big"])
    frame = settle(engine, engine.send(execution_command(frame, candidate["candidate_ref"])))
    assert frame["routing"]["action_bank_version"] != old["routing"]["action_bank_version"]
    assert frame["routing"].get("selection_id") is None
    assert 1 <= before - sum(not e["revealed"] for e in frame["public"]["entities"] if e["entity_type"] == "board_cell") <= 9
    stale = engine.send(execution_command(old, candidate["candidate_ref"]))
    assert stale["error"]["code"] == "stale_decision"
    for _ in range(20):
        if frame["public"]["phase"] == "map":
            break
        frame = choose(engine, frame)
    assert frame["public"]["phase"] == "map"


@pytest.mark.parametrize("event,choice", [("JUNGLE_MAZE_ADVENTURE", 1), ("TRIAL", 0)])
def test_event_audio_and_portrait_do_not_replace_game_effects(engine, event, choice):
    frame = fixture_room(engine, "event", event=event)
    frame = settle(engine, engine.send(execution_command(frame, frame["legal"]["candidates"][choice]["candidate_ref"])))
    assert frame["public"]["phase"] == ("event" if event == "TRIAL" else "map")


def test_automatic_potion_not_manually_usable_and_fake_merchant_inventory(engine):
    frame = fixture_room(engine, "event", event="FAKE_MERCHANT")
    assert frame["public"]["phase"] == "shop"
    assert len([c for c in frame["legal"]["candidates"] if c["verb"] == "BUY_ITEM"]) == 6
    engine.send({"cmd": "set_player", "potions": ["FAIRY_IN_A_BOTTLE", "FOUL_POTION"]})
    frame = settle(engine, engine.send({"cmd": "advance_to_boundary"}))
    potions = {e["ref"]: e["content_id"] for e in frame["public"]["entities"] if e["entity_type"] == "potion"}
    uses = [c for c in frame["legal"]["candidates"] if c["verb"] == "USE_POTION"]
    assert uses and all(potions[c["source_refs"][0]] == "POTION.FOUL_POTION" for c in uses)
    frame = choose(engine, frame, "USE_POTION")
    assert frame["public"]["phase"] == "combat"


def test_native_three_act_victory_milestones_with_explicit_fixture(engine):
    frame = engine.reset("Ironclad", "protocol-three-act")
    engine.send({"cmd": "set_player", "hp": 9999, "max_hp": 9999, "gold": 9999})
    frame = engine.send({"cmd": "advance_to_boundary"})
    ledger = MilestoneLedger()
    for _ in range(3000):
        validate_frame(frame, allow_prototype=True)
        assert frame["contract"]["training_ready"] is False
        ledger.apply(frame.get("events", []))
        if frame["boundary"] == "terminal":
            assert frame["public"]["outcome"]["victory"] is True
            assert ledger.bosses == {1, 2, 3}
            assert ledger.components["boss"] == pytest.approx(.3)
            assert ledger.components["victory"] == 1
            assert ledger.apply(frame["events"]) == 0
            return
        if frame["boundary"] == "waiting":
            time.sleep(.01)
            frame = engine.send({"cmd": "advance_to_boundary"})
            continue
        candidates = frame["legal"]["candidates"]
        candidate = next((c for c in candidates if frame["public"]["phase"] == "shop" and c["verb"] == "LEAVE_ROOM"), candidates[0])
        frame = engine.send(execution_command(frame, candidate["candidate_ref"]))
    pytest.fail("Three-act fixture did not finish")
