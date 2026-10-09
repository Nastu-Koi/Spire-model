"""A complete native A0 run through the rollout loop: its fights as outcome labels.

The run follows a stored action prefix; nothing is searched and no model decides.
"""

import gzip
import json
import os
from collections import Counter
from pathlib import Path

import pytest
import torch

from combat_solver_cli.client import SolverEngine
from model.cli import vocabulary_for
from model.config import ModelConfig
from model.model import PolicyValue
from model.policy import Choice
from model.rollout import RolloutRunner
from model.testing import demonstration

FIXTURE = Path(__file__).resolve().parents[2] / "combat_solver_cli/tests/native/fixtures/final-victory-prefix.json"


class Scripted:
    """Takes the recorded action at every decision, in order."""

    def __init__(self, records):
        self.records = iter(records)

    def reset(self):
        pass

    def choose(self, frame, *, sample=True):
        slot = next(self.records)["action"]["decoder_slot_ref"]
        (candidate,) = [c for c in frame["legal"]["candidates"] if c["decoder_slot_ref"] == slot]
        return Choice(candidate["candidate_ref"], log_prob=torch.zeros(()), value=torch.zeros(()))


@pytest.mark.engine
def test_native_run_labels_every_fight_the_engine_confirmed(tmp_path):
    config = os.environ.get("COMBAT_SOLVER_CONFIG")
    if not config:
        pytest.skip("Pinned native configuration required")
    data = json.loads(FIXTURE.read_text())
    tiny = ModelConfig.tiny()
    runner = RolloutRunner(PolicyValue(tiny), vocabulary_for([demonstration("Ironclad", "labels", 4, 2)], tiny))
    path = tmp_path / "run.combats.jsonl.gz"
    with SolverEngine(config) as engine:
        trace = runner.run(engine, data["character"], data["seed"], sample=False, combats=path,
                           _policy=Scripted(data["records"]))
    assert (trace["status"], trace.get("victory"), trace.get("error")) == ("complete", True, None)
    with gzip.open(path, "rt") as stream:
        fights = [json.loads(line) for line in stream]

    # One label for each fight the engine confirmed, and nothing left over.
    summary = trace["combats"]
    assert summary["wins"] == len(fights) == len(trace["ledger"]["encounters"])
    assert (summary["losses"], summary["unfinished"], summary["unobserved"]) == (0, 0, 0)
    assert summary["acts"] == {"1": "passed", "2": "passed", "3": "passed"}
    assert [f["encounter_id"] for f in fights] == sorted(trace["ledger"]["encounters"], key=lambda x: int(x.rsplit(":", 1)[1]))
    assert Counter(f["kind"] for f in fights)["boss"] == 3 and {f["act"] for f in fights} == {1, 2, 3}
    assert all(f["encounter"].startswith("ENCOUNTER.") and f["act_result"] == "passed" for f in fights)
    assert {f["policy"]["digest"] for f in fights} == {trace["policy_digest"]}

    actions = Counter()
    for fight in fights:
        frames = fight["frames"]
        hero = next(e for e in frames[0]["public"]["entities"] if e["entity_type"] == "player")
        # The entry is the first decision of the fight: enemies are in sight, HP is the HP entered with.
        assert any(e["entity_type"] == "enemy" for e in frames[0]["public"]["entities"])
        assert (fight["start_hp"], fight["max_hp"], fight["floor"]) == (hero["hp"], hero["max_hp"], hero["floor"])
        assert fight["hp_lost"] == fight["start_hp"] - fight["end_hp"] and fight["end_hp"] > 0
        assert len(frames) == fight["steps"] and [f["step"] for f in frames] == sorted(f["step"] for f in frames)
        assert all(f["public"]["phase"] in {"combat", "card_select"} for f in frames)
        actions.update(f["action"]["verb"] for f in frames)
        # Potions drunk in the fight are the potion actions of its frames.
        assert len(fight["potions_used"]) == sum(f["action"]["verb"] == "USE_POTION" for f in frames)
    # Every stored combat action lies in exactly one fight.
    recorded = Counter(r["action"]["verb"] for r in data["records"])
    assert actions["PLAY_CARD"] == recorded["PLAY_CARD"] and actions["END_TURN"] == recorded["END_TURN"]
    # No frame follows the last fight: its result comes from the run state.
    assert fights[-1]["end"] is None and all(f["end"]["hp"] == f["end_hp"] for f in fights[:-1])
