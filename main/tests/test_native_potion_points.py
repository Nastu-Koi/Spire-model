"""A potion point's continuations against the native engine (`main.play.potion_value`):
each resumes at the point's map decision, as the source run stood there, with the potion
taken out thrown away. The engine's continue save written at that decision would resume
before the room the point follows, and play the room again."""

import gzip
import json

import pytest

from combat_solver_cli.client import SolverEngine
from main.env.protocol import execution_command
from main.play.fights import state as fight_state
from main.play.potion_value import PointEngine, branch_saves, point_key, read_point
from main.tests.test_native_fight_entry import native_config, settle

# A run whose first candidates (rewards taken, potions kept) come to a map decision after
# a fight with a potion in the belt, on the act's fifth floor.
RUN = ("Ironclad", "pv-1", 0)


def first_choice(frame):
    candidates = [c for c in frame["legal"]["candidates"] if c["verb"] not in ("DISCARD_POTION", "USE_POTION")]
    taken = [c for c in candidates if c["verb"] == "CLAIM_REWARD"] or \
        [c for c in candidates if c["verb"] == "LEAVE_REWARDS"] or candidates
    return taken[0]


@pytest.mark.engine
def test_native_potion_point_resumes_at_the_point(tmp_path):
    points = tmp_path / "points"
    points.mkdir()
    with SolverEngine(native_config()) as native:
        engine = PointEngine(native, points, {RUN[:2]})
        frame = settle(native, engine.reset(*RUN))
        while not list(points.glob("*.json.gz")):
            assert frame["boundary"] == "decision" and fight_state(frame)["act"] == 1
            frame = settle(native, engine.send(execution_command(frame, first_choice(frame)["candidate_ref"])))
        at_point = fight_state(frame)
        assert frame["public"]["phase"] == "map" and at_point["floor"] > 1 and at_point["potions"]
        point = read_point(next(points.glob("*.json.gz")))
        assert point["state"] == at_point
        for branch in branch_saves([point], tmp_path / "branches"):
            with gzip.open(branch["path"], "rt", encoding="utf-8") as stream:
                resumed = settle(native, PointEngine(native, points, set()).resume(json.load(stream)["save"]))
            where = fight_state(resumed)
            assert resumed["public"]["phase"] == "map"
            assert (where["act"], where["floor"], where["hp"]) == (at_point["act"], at_point["floor"], at_point["hp"])
            if branch["dropped"] is None:
                assert point_key(resumed) == point_key(frame)
            else:
                held = list(at_point["potions"])
                assert held.pop(int(branch["branch"][len("drop"):])) == branch["dropped"]
                assert where["potions"] == held
