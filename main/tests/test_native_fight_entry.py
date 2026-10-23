"""Entering a fight again from its entry save, against the native engine.

The entry save is written at the fight's first decision (`saves.continue_save`) and
`resume_fight` enters the fight from it (`CliEngine.resume_fight`). The save holds the
run as the room was entered, so what the fight's start does to the run (a relic's heal
on the first turn) happens once, and the fight starts as it did.
"""

import os

import pytest

from combat_solver_cli.client import SolverEngine
from combat_solver_cli.search_support import state_key
from main.env.protocol import execution_command
from main.env.saves import continue_save


def native_config():
    value = os.environ.get("COMBAT_SOLVER_CONFIG")
    if not value:
        pytest.skip("Pinned native configuration required")
    return value


def settle(engine, frame):
    while frame["boundary"] == "waiting":
        frame = engine.send({"cmd": "advance_to_boundary"})
    assert frame["boundary"] == "decision", frame.get("error")
    return frame


def choose(engine, frame, candidate):
    return settle(engine, engine.send(execution_command(frame, candidate["candidate_ref"])))


def player(frame):
    return next(e for e in frame["public"]["entities"] if e["entity_type"] == "player")


def to_map(engine, frame):
    """The next map decision: the first candidate of each decision before it, rewards
    left behind."""
    while frame["public"]["phase"] != "map":
        candidates = frame["legal"]["candidates"]
        leave = [c for c in candidates if c["verb"] == "LEAVE_REWARDS"]
        frame = choose(engine, frame, (leave or candidates)[0])
    return frame


def into_monster_room(engine, frame):
    """Travel from a map decision to a monster node: the fight's first decision and
    its `encounter_started` event."""
    kinds = {e["ref"]: e["content_id"] for e in frame["public"]["entities"] if e["entity_type"] == "map_node"}
    move = next(c for c in frame["legal"]["candidates"]
                if c["verb"] == "MOVE_TO_NODE" and kinds[c["source_refs"][0]] == "Monster")
    frame = choose(engine, frame, move)
    started = next(e for e in frame["events"] if e["type"] == "encounter_started")
    assert frame["public"]["phase"] == "combat"
    return frame, started


def resume_row(engine, frame, started):
    return dict(save=continue_save(engine), encounter=started["encounter"], floor=player(frame)["floor"],
                map_point_type="monster", second_boss=False)


def map_save(engine, character, seed):
    """A continue save that `resume_save` takes: the one at the map decision after the
    first fight. Written there, it holds the run before that fight's room."""
    frame = to_map(engine, settle(engine, engine.reset(character, seed, 0)))
    to_map(engine, into_monster_room(engine, frame)[0])
    return continue_save(engine)


def with_relics(save, relics, missing):
    save["players"][0]["relics"] += [{"id": relic, "floor_added_to_deck": 1} for relic in relics]
    save["players"][0]["current_hp"] = save["players"][0]["max_hp"] - missing
    return save


def observed(frame):
    """The frame's public state and candidates, without the public history: a loaded
    run does not know whether the point before had a shop."""
    return state_key(dict(frame, public={k: v for k, v in frame["public"].items() if k != "memory"}))


def fought(engine, frame, refs=None):
    """The fight played to its end with the given candidates (the first candidate of
    each decision when none are given): each frame observed, and the candidates."""
    frames, chosen = [observed(frame)], []
    while frame["public"]["phase"] == "combat":
        ref = refs[len(chosen)] if refs else frame["legal"]["candidates"][0]["candidate_ref"]
        chosen.append(ref)
        frame = choose(engine, frame, next(c for c in frame["legal"]["candidates"] if c["candidate_ref"] == ref))
        frames.append(observed(frame))
    return frames, chosen


@pytest.mark.engine
def test_native_fight_resume_heals_blood_vial_once():
    with SolverEngine(native_config()) as engine:
        save = with_relics(map_save(engine, "Ironclad", "fight-entry-blood-vial"), ["RELIC.BLOOD_VIAL"], missing=20)
        maximum = save["players"][0]["max_hp"]
        entered, started = into_monster_room(engine, settle(engine, engine.resume(save)))
        assert player(entered)["hp"] == maximum - 18  # Blood Vial heals 2 on the first turn
        again = engine.resume_fight(resume_row(engine, entered, started))
        assert player(again)["hp"] == player(entered)["hp"]


# Relics that change the run, not only the fight, as a fight starts: HP (Blood Vial,
# Royal Poison), gold (Maw Bank on entering the room), a potion (Petrified Toad), a
# count of fights (Ember Tea) and of turns (Happy Flower).
RUN_RELICS = ["RELIC.BLOOD_VIAL", "RELIC.ROYAL_POISON", "RELIC.MAW_BANK", "RELIC.PETRIFIED_TOAD",
              "RELIC.EMBER_TEA", "RELIC.HAPPY_FLOWER"]


@pytest.mark.engine
def test_native_fight_resume_replays_the_fight():
    with SolverEngine(native_config()) as engine:
        save = with_relics(map_save(engine, "Ironclad", "fight-entry-blood-vial"), RUN_RELICS, missing=20)
        entered, started = into_monster_room(engine, settle(engine, engine.resume(save)))
        row = resume_row(engine, entered, started)
        frames, chosen = fought(engine, entered)
        again = engine.resume_fight(row)
        # A save written in the fight entered again holds the same run.
        assert observed(engine.resume_fight(resume_row(engine, again, started))) == frames[0]
        # The enemies' HP and the opening hand are drawn as they were: the same
        # actions play the same fight.
        assert fought(engine, engine.resume_fight(row), chosen)[0] == frames
