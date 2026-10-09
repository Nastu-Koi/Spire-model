"""Fixed chests agree across rollout, Steam, demonstration import and BC weights."""

import gzip
import json
from copy import deepcopy

import pytest

from model.config import ModelConfig, TrainConfig
from model.control import CONTROL_VERSION
from model.data import RunShards, load_runs, samples as training_samples, validate_run
from model.dataset_index import HEADER, index_row
from model.independent import convert_group
from model.loader import stream
from model.model import PolicyValue
from model.protocol import ProtocolError, action_semantics
from model.representation import Vocabulary
from model.rollout import RolloutRunner
from model.steam import choose_action
from model.testing import demonstration
from model.tests.test_independent import samples
from model.tests.test_reward_control import Policy, ScriptedEngine, policy_frame
from model.tests.test_steam import Policy as SteamPolicy, message
from model.treasure_rule import RULE_VERSION, collection_candidate


@pytest.mark.parametrize("verbs,expected", [
    (["LEAVE_ROOM", "OPEN_CHEST", "DISCARD_POTION", "USE_POTION"], "OPEN_CHEST"),
    (["LEAVE_ROOM", "TAKE_TREASURE_RELIC", "DISCARD_POTION"], "TAKE_TREASURE_RELIC"),
    (["DISCARD_POTION", "LEAVE_ROOM"], "LEAVE_ROOM"),
])
def test_treasure_uses_the_fixed_flow_even_with_other_legal_actions(verbs, expected):
    frame = policy_frame("treasure", verbs, 0)
    assert collection_candidate(frame)["verb"] == expected
    frame["public"]["phase"] = "event"
    assert collection_candidate(frame) is None


def test_ordinary_treasure_does_not_invent_a_choice_between_multiple_relics():
    frame = policy_frame("treasure", ["TAKE_TREASURE_RELIC", "TAKE_TREASURE_RELIC", "LEAVE_ROOM"], 0)
    with pytest.raises(ProtocolError, match="requires one TAKE_TREASURE_RELIC"):
        collection_candidate(frame)


def test_rollout_chest_steps_have_no_policy_probability_but_keep_downstream_returns():
    frames = [
        policy_frame("map", ["CHOOSE_MAP_NODE", "CHOOSE_MAP_NODE"], 0),
        policy_frame("treasure", ["OPEN_CHEST", "LEAVE_ROOM", "DISCARD_POTION"], 1),
        policy_frame("treasure", ["TAKE_TREASURE_RELIC", "LEAVE_ROOM", "DISCARD_POTION"], 2),
        # A relic's follow-up choice still belongs to the policy.
        policy_frame("card_select", ["SELECT_ONE", "SELECT_ONE"], 3),
        policy_frame("treasure", ["LEAVE_ROOM", "DISCARD_POTION"], 4),
    ]
    terminal = policy_frame("terminal", [], 5)
    terminal.update(boundary="terminal", events=[dict(type="run_completed", victory=True,
                    act=3, final_boss_defeated=True)])
    terminal["public"]["outcome"] = dict(victory=True)
    engine = ScriptedEngine([*frames, terminal])
    config = ModelConfig.tiny()
    vocabulary = Vocabulary.from_frames(frames, config)
    policy = Policy()
    trace = RolloutRunner(PolicyValue(config), vocabulary).run(
        engine, "Ironclad", "control", _policy=policy)
    assert trace["status"] == "complete", trace.get("error")
    assert policy.phases == ["map", "card_select"]
    assert [m["phase"] for m in trace["macros"]] == policy.phases
    assert [m["return"] for m in trace["macros"]] == [5, 5]
    assert [m["old_log_prob"] for m in trace["macros"]] == [-0.7, -0.7]
    assert trace["automatic_steps"] == 3
    actions = trace["environment_actions"]
    assert [a["candidate_ref"] for a in actions] == ["1:0", "2:0", "4:0"]
    assert all(a["actor"] == "treasure_controller" and a["rule"] == RULE_VERSION for a in actions)
    validate_run(trace, on_policy=True)
    del trace["control_version"]
    with pytest.raises(ProtocolError, match="current policy control version"):
        validate_run(trace, on_policy=True)
    trace["control_version"] = CONTROL_VERSION
    bad = deepcopy(trace["macros"][0])
    bad.update(phase="treasure", steps=[dict(frame=frames[1], candidate_ref="1:0", forced=False)])
    trace["macros"].insert(1, bad)
    with pytest.raises(ProtocolError, match="environment steps"):
        validate_run(trace, on_policy=True)


def test_steam_uses_the_same_chest_flow_without_calling_the_model():
    leave = dict(command="pick_relic", args=dict(index=None))
    open_chest = dict(command="open_chest", args={})
    take = dict(command="pick_relic", args=dict(index=0))
    policy = SteamPolicy("LEAVE_ROOM")
    for actions, context, expected in (
        ([open_chest, leave], [], open_chest),
        ([take, leave], [dict(index=0, relic=dict(id="ANCHOR"))], take),
        ([leave], [], leave),
    ):
        assert choose_action(policy, message("treasure", actions, context)) == expected
    assert policy.frames == []


def test_independent_import_keeps_treasure_as_evidence_and_relic_choices_as_labels():
    rows = samples()
    chest = deepcopy(rows[0])
    frame = policy_frame("treasure", ["OPEN_CHEST", "LEAVE_ROOM"], 0)
    chest.update(observation=frame["public"], options=[action_semantics(c) for c in frame["legal"]["candidates"]])
    chest["label"] = chest["options"][0]
    chest["metadata"]["routing"] = dict(episode_id="chest", decision_id="open", state_version=0)
    only = convert_group([chest])
    assert only["macros"] == [] and training_samples([only]) == []
    assert only["automatic_steps"] == len(only["environment_actions"]) == 1
    mixed = convert_group([chest, *rows])
    assert mixed["macros"] == convert_group(rows)["macros"]
    assert mixed["environment_actions"][0]["frame"]["public"]["phase"] == "treasure"


def test_sharded_and_in_memory_bc_exclude_chests_before_weighting_and_batching(tmp_path):
    mixed = demonstration("Ironclad", "mixed", 4, 2)
    frame = policy_frame("treasure", ["OPEN_CHEST", "LEAVE_ROOM", "DISCARD_POTION"], 0)
    frame["contract"] = deepcopy(mixed["contract"])
    chest = dict(phase="treasure", steps=[dict(frame=frame, candidate_ref="0:1", forced=False)])
    mixed["macros"].insert(0, chest)
    only = dict(mixed, run_id="only-chest", macros=[chest])
    plain = demonstration("Ironclad", "plain", 4, 2)
    runs = [only, mixed, plain]
    shard = "data.jsonl.gz"
    with gzip.open(tmp_path / shard, "wt") as out:
        out.writelines(json.dumps(run) + "\n" for run in runs)
    index = [index_row(r, shard, line) for line, r in enumerate(runs)]
    with gzip.open(tmp_path / "index.jsonl.gz", "wt") as out:
        out.write(json.dumps(HEADER) + "\n")
        out.writelines(json.dumps(row) + "\n" for row in index)
    dataset = RunShards(tmp_path)
    assert [row["line"] for row in dataset.rows] == [1, 2]
    assert len(dataset) == 2 and ("Ironclad", "treasure") not in dataset.counts
    window = dataset.read([shard])
    assert [r["run_id"] for r in window] == [mixed["run_id"], plain["run_id"]]
    expected = training_samples(load_runs(tmp_path / shard))
    for items in (training_samples(window, counts=dataset.counts), training_samples(runs)):
        assert len(items) == len(expected) == sum(dataset.counts.values())
        assert [i.weight for i in items] == [i.weight for i in expected]
        assert sum(i.weight for i in items) == pytest.approx(1.0)
        assert all(i.phase != "treasure" for i in items)
    assert mixed["macros"][0] is chest  # Sampling did not overwrite historical evidence.
    config = ModelConfig.tiny()
    vocabulary = Vocabulary.from_frames([s["frame"] for i in expected for s in i.macro["steps"]], config)
    training = TrainConfig(loader_workers=0, logical_batch_size=3, microbatch_size=3)
    logical = list(stream(dataset, vocabulary, training, window_shards=1, shuffle=False))
    assert [sum(len(items) for items, _ in batch) for batch in logical] == [3, 1]
