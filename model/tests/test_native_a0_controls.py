"""Reward accounting and crystal control against native A0 replay fixtures.

The stored combat actions only reach the diagnostic boundary; no solver searches
or training updates run here. Controllers receive decision frames only.
"""

import json
import os
import random
import time
from collections import Counter
from pathlib import Path

import pytest

from combat_solver_cli.client import SolverEngine
from combat_solver_cli.search_support import resolve, state_key
from combat_solver_cli.trajectory import native_replay_error
from model.crystal_rule import collection_candidate, plan
from model.control import environment_action, is_policy_frame
from model.protocol import action_semantics, clean_frame, execution_command, validate_frame
from model.rewards import MilestoneLedger
from model.treasure_rule import collection_candidate as treasure_candidate

FIXTURES = Path(__file__).resolve().parents[2] / "combat_solver_cli/tests/native/fixtures"


def native_config():
    value = os.environ.get("COMBAT_SOLVER_CONFIG")
    if not value:
        pytest.skip("Pinned native configuration required")
    return value


def settle(engine, frame, ledger):
    deadline = time.monotonic() + 30
    while True:
        validate_frame(frame)
        ledger.apply(frame.get("events", []))
        if frame["boundary"] != "waiting":
            return frame
        assert time.monotonic() < deadline, "Native action did not reach a boundary"
        frame = engine.send({"cmd": "advance_to_boundary"})


def replay(engine, filename, ledger, stop=None):
    data = json.loads((FIXTURES / filename).read_text())
    assert data["ascension"] == 0
    frame = engine.reset(data["character"], data["seed"], 0)
    for index, record in enumerate(data["records"][:stop]):
        frame = settle(engine, frame, ledger)
        assert state_key(frame) == record["before_hash"], f"Replay diverged at {filename}:{index}"
        candidate = resolve(frame, record["action"])
        frame = engine.send(execution_command(frame, candidate["candidate_ref"]))
    return settle(engine, frame, ledger)


@pytest.mark.engine
def test_native_single_event_advance_reveals_choices_only_after_execution():
    from spire_codex_data import anchor
    from spire_codex_data.tests.test_events import node

    config = native_config()
    with SolverEngine(config) as engine:
        settle(engine, engine.reset("Ironclad", "single-event-control", 0), MilestoneLedger())
        base = engine.send(dict(cmd="anchor_state"))["save"]
    player = base["players"][0]
    state = dict(deck=player["deck"], relics=player["relics"], potions=["POTION.FIRE_POTION"],
                 hp=60, max_hp=player["max_hp"], gold=99)
    nodes = [node("ancient", [dict(room_type="event")]),
             node("unknown", [dict(room_type="event", model_id="EVENT.TINKER_TIME")])]
    run = dict(seed="single-event-control", ascension=0, map_point_history=[nodes],
               players=[dict(deck=player["deck"], relics=player["relics"])])
    save = anchor.build_save(base, state, run, 1, 2, nodes[1], "single-event-control")
    spot = dict(floor=2, map_point_type="unknown", second_boss=False, route=["ancient", "unknown"],
                kind="event", id="single-event-control", room=dict(type="event", event="TINKER_TIME"))
    with anchor.RunProcess(config) as shared:
        engine, frame = shared.enter(save, spot)
        assert sorted(c["verb"] for c in frame["legal"]["candidates"]) == ["CHOOSE_EVENT_OPTION", "DISCARD_POTION"]
        options = [e["content_id"] for e in frame["public"]["entities"] if e["entity_type"] == "event_option"]
        assert len(options) == 1 and "CHOOSE_CARD_TYPE" in options[0]
        action = environment_action(clean_frame(frame))
        assert action["actor"] == "event_controller" and not is_policy_frame(frame)
        frame = settle(engine, engine.send(execution_command(frame, action["candidate_ref"])), MilestoneLedger())
        assert frame["public"]["phase"] == "event"
        assert sum(c["verb"] == "CHOOSE_EVENT_OPTION" for c in frame["legal"]["candidates"]) == 2
        assert environment_action(clean_frame(frame)) is None and is_policy_frame(frame)
        assert shared.native_error() is None


@pytest.mark.engine
@pytest.mark.parametrize("chest_index,node_type", [(43, "Unknown"), (72, "Treasure")])
def test_native_fixed_chest_flow_from_question_mark_and_treasure_node(chest_index, node_type):
    fixture = "lords-parasol-prefix.json"
    records = json.loads((FIXTURES / fixture).read_text())["records"]
    ledger = MilestoneLedger()
    with SolverEngine(native_config()) as engine:
        frame = replay(engine, fixture, ledger, stop=chest_index - 1)
        target = records[chest_index - 1]["action"]["source_refs"][0]
        node = next(e for e in frame["public"]["entities"] if e.get("ref") == target)
        assert node["content_id"] == node_type
        move = resolve(frame, records[chest_index - 1]["action"])
        frame = settle(engine, engine.send(execution_command(frame, move["candidate_ref"])), ledger)
        assert frame["public"]["phase"] == "treasure"
        assert not any(e.get("zone") == "treasure" for e in frame["public"]["entities"])
        opened = treasure_candidate(frame)
        assert opened["verb"] == "OPEN_CHEST"
        frame = settle(engine, engine.send(execution_command(frame, opened["candidate_ref"])), ledger)
        relics = [e["content_id"] for e in frame["public"]["entities"] if e.get("zone") == "treasure"]
        assert len(relics) == 1
        claimed = treasure_candidate(frame)
        assert claimed["verb"] == "TAKE_TREASURE_RELIC"
        frame = settle(engine, engine.send(execution_command(frame, claimed["candidate_ref"])), ledger)
        # Native collection returns to the map directly; no extra leave action is needed.
        assert frame["public"]["phase"] == "map"
        assert state_key(frame) == records[chest_index + 2]["before_hash"]
        assert any(e.get("entity_type") == "relic" and e.get("content_id") == relics[0]
                   for e in frame["public"]["entities"])
        assert native_replay_error(engine) is None


@pytest.mark.engine
def test_native_a0_victory_pays_eight_once():
    ledger = MilestoneLedger()
    with SolverEngine(native_config()) as engine:
        frame = replay(engine, "final-victory-prefix.json", ledger)
        assert frame["boundary"] == "terminal" and frame["public"]["outcome"]["victory"]
        assert ledger.bosses == {1, 2, 3}
        assert ledger.components == {"combat": 0, "boss": 3, "victory": 5}
        assert ledger.apply(frame.get("events", [])) == 0
        assert native_replay_error(engine) is None


@pytest.mark.engine
def test_native_crystal_payment_grid_rewards(tmp_path):
    results = []
    config = native_config()
    initial_states = {}
    for payment, clicks in ((0, 3), (1, 6)):
        for controller in ("planner", "Small", "Big"):
            ledger = MilestoneLedger()
            with SolverEngine(config) as engine:
                frame = replay(engine, "battleworn-rewards-prefix.json", ledger, stop=206)
                assert frame["public"]["phase"] == "event"
                assert collection_candidate(frame) is None
                candidate = next(c for c in frame["legal"]["candidates"]
                                 if c["verb"] == "CHOOSE_EVENT_OPTION"
                                 and c["source_refs"] == [f"option:{payment}"])
                frame = settle(engine, engine.send(execution_command(frame, candidate["candidate_ref"])), ledger)
                initial = state_key(frame)
                assert initial == initial_states.setdefault(payment, initial)
                initial_curses = Counter(e.get("content_id") for e in frame["public"]["entities"]
                                        if e.get("zone") == "deck" and e.get("card_type") == "Curse")
                records, costs, fallbacks = [], [], 0
                rng = random.Random(101)
                for remaining in range(clicks, 0, -1):
                    assert frame["public"]["phase"] == "crystal_sphere"
                    assert next(e["remaining_steps"] for e in frame["public"]["entities"]
                                if e["entity_type"] == "event_state") == remaining
                    public = clean_frame(frame)
                    start = time.perf_counter()
                    if controller == "planner":
                        decision = plan(public)
                        candidate = decision.candidate
                        fallbacks += decision.fallback
                    else:
                        candidate = rng.choice([c for c in public["legal"]["candidates"]
                                                if c["source_refs"] == [f"tool:{controller}"]])
                    costs.append(time.perf_counter() - start)
                    records.append(action_semantics(candidate))
                    frame = settle(engine, engine.send(execution_command(frame, candidate["candidate_ref"])), ledger)
                assert frame["public"]["phase"] in {"rewards", "event", "map"}
                assert collection_candidate(frame) is None
                rewards = [e for e in frame["public"]["entities"] if e["entity_type"] == "reward"]
                final_curses = Counter(e.get("content_id") for e in frame["public"]["entities"]
                                      if e.get("zone") == "deck" and e.get("card_type") == "Curse")
                results.append(dict(payment=payment, clicks=clicks, controller=controller,
                    rewards=rewards, added_grid_curses=dict(final_curses - initial_curses),
                    seconds_per_click=costs, geometry_fallbacks=fallbacks, actions=records))
                # Every offered reward remains a policy choice, including leaving
                # everything behind. Exiting must resume the native event task.
                if frame["public"]["phase"] == "rewards":
                    leave = next(c for c in frame["legal"]["candidates"] if c["verb"] == "LEAVE_REWARDS")
                    frame = settle(engine, engine.send(execution_command(frame, leave["candidate_ref"])), ledger)
                if frame["public"]["phase"] == "event":
                    done = next(c for c in frame["legal"]["candidates"] if c["verb"] == "CHOOSE_EVENT_OPTION")
                    frame = settle(engine, engine.send(execution_command(frame, done["candidate_ref"])), ledger)
                assert frame["public"]["phase"] == "map"
                assert native_replay_error(engine) is None
    report = dict(domain="native-prefix-replay", boards=1, ascension=0,
                  fixture="battleworn-rewards-prefix.json", comparisons=results)
    path = tmp_path / "crystal-native.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    destination = os.environ.get("SPIRE_A0_NATIVE_REPORT")
    if destination:
        with Path(destination).open("x") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
