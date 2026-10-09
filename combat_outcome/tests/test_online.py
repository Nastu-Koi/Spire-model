"""Fights of a rollout as combat-outcome labels of the policy that played them."""

import gzip
import json

import pytest

from combat_outcome.data import SCHEMA, coverage, fights, load_data, split_runs
from combat_outcome.online import CombatRecorder, act_results
from model.cli import vocabulary_for
from model.config import ModelConfig
from model.model import PolicyValue
from model.protocol import ProtocolError
from model.rollout import RolloutRunner, collect_round
from model.protocol import CHARACTERS
from model.testing import SyntheticEngine, demonstration

POLICY = dict(version=3, digest="weights", vocabulary_hash="v", precision="no", sampling="argmax")
CONTRACT = dict(observation_schema="public-state-v6", public_history_version="public-history-v1")
END_TURN = dict(verb="END_TURN", candidate_ref="c0", source_refs=[], target_refs=[])
DRINK = dict(verb="USE_POTION", candidate_ref="c1", source_refs=["potion:0"], target_refs=[])


def frame(hp, *, act=1, turn=1, phase="combat", potions=("POTION.FIRE",), gold=99):
    entities = [dict(entity_type="player", ref="player", hp=hp, max_hp=80, gold=gold, act=act, floor=4,
                     round=turn, ascension=0),
                dict(entity_type="card", zone="deck", content_id="CARD.BASH"),
                dict(entity_type="card", zone="hand", content_id="CARD.BASH"),
                dict(entity_type="enemy", ref="creature:1", hp=30)]
    entities += [dict(entity_type="potion", ref=f"potion:{i}", content_id=p) for i, p in enumerate(potions)]
    return dict(public=dict(phase=phase, entities=entities), legal=dict(candidates=[END_TURN, DRINK]))


def started(index, act=1, kind="normal"):
    return dict(type="encounter_started", encounter_id=f"e{index}", encounter="ENCOUNTER.TOADPOLES_WEAK",
                act=act, kind=kind, from_event=False)


def completed(index, act=1):
    return dict(type="encounter_completed", encounter_id=f"e{index}", act=act, kind="normal", result="victory")


def recorder(tmp_path, name="run"):
    return CombatRecorder(tmp_path / f"{name}.combats.jsonl.gz", seed="seed-" + name, character="Ironclad",
                          run_id=name, policy=POLICY, contract=CONTRACT)


def rows(path):
    with gzip.open(path, "rt") as stream:
        return [json.loads(line) for line in stream]


def test_a_won_and_a_lost_fight_become_labels_with_their_frames(tmp_path):
    r = recorder(tmp_path)
    # The engine names the fight with its first frame; the first decision is the entry.
    r.events([started(1)])
    first = frame(60)
    r.decision(first)
    r.step(first, DRINK, 10)
    second = frame(52, turn=2, potions=())
    r.decision(second)
    r.step(second, END_TURN, 11)
    # The win arrives with the first frame after the fight: that frame holds the result.
    r.events([completed(1)])
    r.decision(frame(58, phase="rewards", potions=(), gold=120))
    assert r.pending is None
    r.events([started(2, kind="boss")])
    entry = frame(58, potions=())
    r.decision(entry)
    r.step(entry, END_TURN, 14)
    r.terminal(False, lambda: pytest.fail("A lost fight needs no final HP"))
    summary = r.close("complete", False, set())
    assert summary == dict(path="run.combats.jsonl.gz", schema=SCHEMA, wins=1, losses=1, unfinished=0,
                           unobserved=0, acts={"1": "died"})

    won, lost = rows(tmp_path / "run.combats.jsonl.gz")
    assert (won["source"], won["actor"], won["policy"], won["contract"]) == ("rollout", "policy", POLICY, CONTRACT)
    assert (won["seed"], won["run_id"], won["encounter"], won["kind"]) == (
        "seed-run", "run", "ENCOUNTER.TOADPOLES_WEAK", "regular")
    # Net loss includes what the end of the fight settled; a lost fight costs all it was entered with.
    assert (won["start_hp"], won["end_hp"], won["hp_lost"], won["result"], won["turns"]) == (60, 58, 2, "win", 2)
    assert (lost["start_hp"], lost["end_hp"], lost["hp_lost"], lost["result"], lost["kind"]) == (58, 0, 58, "loss", "boss")
    assert won["start"]["potions"] == ["POTION.FIRE"] and won["end"] == dict(hp=58, max_hp=80, gold=120, potions=[])
    assert [e["content_id"] for e in won["potions_used"]] == ["POTION.FIRE"] and lost["end"] is None
    assert won["act_result"] == lost["act_result"] == "died"
    # The entry model's input is the state the fight was entered with: no hand, no enemies.
    assert {(e["entity_type"], e.get("zone")) for e in won["entities"]} == {
        ("player", None), ("potion", None), ("card", "deck")}
    # Frames are the observations of the fight's own decisions, with the action taken there.
    assert [(f["step"], f["action"]["verb"]) for f in won["frames"]] == [(10, "USE_POTION"), (11, "END_TURN")]
    assert won["frames"][0]["public"] == first["public"] and won["frames"][1]["public"] == second["public"]
    assert "candidate_ref" not in won["frames"][0]["action"]


def test_a_fight_cut_off_is_counted_and_never_a_defeat(tmp_path):
    r = recorder(tmp_path)
    r.events([started(1)])
    entry = frame(60)
    r.decision(entry)
    r.step(entry, END_TURN, 0)
    summary = r.close("unresolved", None, set())
    assert (summary["wins"], summary["losses"], summary["unfinished"]) == (0, 0, 1)
    assert rows(tmp_path / "run.combats.jsonl.gz") == []
    # The act the run stopped in has no result.
    assert summary["acts"] == {"1": None}


def test_the_last_fight_of_a_won_run_reads_its_result_from_the_engine(tmp_path):
    r = recorder(tmp_path)
    r.events([started(1, act=3, kind="boss")])
    entry = frame(40, act=3)
    r.decision(entry)
    r.step(entry, END_TURN, 0)
    r.events([completed(1, act=3)])
    r.terminal(True, lambda: 33)
    summary = r.close("complete", True, {1, 2, 3})
    (won,) = rows(tmp_path / "run.combats.jsonl.gz")
    assert (won["end_hp"], won["hp_lost"], won["end"], won["act_result"]) == (33, 7, None, "passed")
    assert summary["acts"] == {"1": "passed", "2": "passed", "3": "passed"}

    # Without that reading the fight has no result to label.
    r = recorder(tmp_path, "blind")
    r.events([started(1, act=3, kind="boss")])
    r.decision(entry)
    r.step(entry, END_TURN, 0)
    r.events([completed(1, act=3)])
    r.terminal(True, lambda: None)
    assert r.close("complete", True, {1, 2, 3})["unobserved"] == 1


def test_fight_boundaries_come_from_the_engine_only(tmp_path):
    r = recorder(tmp_path)
    with pytest.raises(ProtocolError, match="never announced"):
        r.events([completed(1)])
    r.events([started(1)])
    with pytest.raises(ProtocolError, match="before the previous one ended"):
        r.events([started(2)])
    with pytest.raises(ProtocolError, match="did not complete"):
        r.terminal(True, lambda: 1)


def test_act_results_follow_the_run():
    assert act_results("complete", True, {1, 2, 3}, 3) == {1: "passed", 2: "passed", 3: "passed"}
    assert act_results("complete", False, {1}, 2) == {1: "passed", 2: "died"}
    # A run that stopped without an outcome did not die in its last act.
    assert act_results("unresolved", None, {1}, 2) == {1: "passed", 2: None}
    assert act_results("error", None, set(), None) == {}


def tiny_runner(**options):
    config = ModelConfig.tiny()
    vocabulary = vocabulary_for([demonstration("Ironclad", "combat-labels", 4, 2)], config)
    return RolloutRunner(PolicyValue(config), vocabulary, **options)


def test_a_rollout_records_the_fights_its_policy_played(tmp_path):
    runner = tiny_runner(version=5)
    seeds = {c: [f"{c}-a", f"{c}-b"] for c in CHARACTERS}
    paths = collect_round(SyntheticEngine, runner, seeds, tmp_path / "round", sample=False)
    traces = [json.loads(p.read_text()) for p in paths]
    for path, trace in zip(paths, traces):
        (fight,) = rows(path.with_suffix(".combats.jsonl.gz"))
        # As many labels as fights the engine confirmed, by the policy that sampled the run.
        assert trace["combats"]["wins"] == len(trace["ledger"]["encounters"]) == 1
        assert (fight["policy"]["version"], fight["policy"]["digest"]) == (5, trace["policy_digest"])
        assert (fight["seed"], fight["run_id"], fight["contract"]) == (trace["seed"], trace["run_id"], trace["contract"])
        # Each frame is exactly what the policy was shown at that decision.
        shown = [step["frame"]["public"] for macro in trace["macros"] for step in macro["steps"]]
        assert [f["public"] for f in fight["frames"]] == shown[:len(fight["frames"])]
    assert len(set(t["policy_digest"] for t in traces)) == 1

    data = load_data(tmp_path / "round")
    assert len(data) == 10 and {r["actor"] for r in data} == {"policy"}
    assert {r["policy"] for r in data} == {traces[0]["policy_digest"]}
    # Every frame of a fight stays on the side of the split its seed is on.
    train, test, _, test_runs = split_runs(data, seed=0, holdout=0.3)
    assert {r["run"] for r in train}.isdisjoint(test_runs) and {r["run"] for r in test} == test_runs
    for label, frames in fights(tmp_path / "round"):
        assert (label["seed"] in test_runs) != (label["seed"] in {r["run"] for r in train})
        assert len(frames) == label["steps"] == 6 and "frames" not in label
        # What followed a frame is in the label only.
        assert all(set(f) == {"step", "forced", "public", "action"} for f in frames)
    report = coverage(tmp_path / "round")
    assert sum(item["fights"] for item in report["fights"]) == 10
    assert {(item["actor"], item["policy_version"], item["result"]) for item in report["fights"]} == {("policy", 5, "win")}
    assert sum(item["runs"] for item in report["acts"]) == 10


def test_labels_declare_who_fought(tmp_path):
    path = tmp_path / "combats.jsonl"
    row = dict(schema=SCHEMA, source="rollout", actor="combat_solver", seed="s", character="Ironclad", act=1,
               kind="regular", encounter="X", entities=[], result="win", max_hp=70, start_hp=60, end_hp=50, hp_lost=10)
    path.write_text(json.dumps(row) + "\n")
    # A solver's fights cannot pass for a policy's.
    with pytest.raises(ValueError, match="source and actor"):
        load_data(path)
    path.write_text(json.dumps(dict(row, schema="combat-outcomes-v1", actor="policy")) + "\n")
    with pytest.raises(ValueError, match="Unsupported combat schema"):
        load_data(path)
