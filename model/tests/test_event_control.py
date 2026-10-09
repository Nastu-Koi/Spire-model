"""Only event advances without meaningful alternatives leave the policy."""

import gzip
import json
from copy import deepcopy

import pytest

from model.config import ModelConfig, TrainConfig
from model.control import CONTROL_VERSION, EVENT_RULE, environment_action, is_policy_frame
from model.data import RunShards, samples as training_samples, validate_run
from model.dataset_index import INDEX, rebuild_index
from model.independent import convert_group
from model.loader import stream
from model.model import PolicyValue
from model.policy import prepare_replay
from model.protocol import ProtocolError, action_semantics
from model.representation import Vocabulary
from model.rollout import RolloutRunner
from model.steam import choose_action
from model.testing import demonstration
from model.tests.test_independent import samples
from model.tests.test_reward_control import Policy, ScriptedEngine, policy_frame
from model.tests.test_steam import Policy as SteamPolicy, message


@pytest.mark.parametrize("phase,verbs,automatic", [
    ("event", ["CHOOSE_EVENT_OPTION"], True),
    ("event", ["DISCARD_POTION", "CHOOSE_EVENT_OPTION", "DISCARD_POTION"], True),
    ("event", ["CHOOSE_EVENT_OPTION", "USE_POTION", "DISCARD_POTION"], False),
    ("event", ["CHOOSE_EVENT_OPTION", "CHOOSE_EVENT_OPTION", "DISCARD_POTION"], False),
    ("event", ["ABANDON_RUN", "DISCARD_POTION"], False),
    ("event", ["CHOOSE_EVENT_OPTION", "LEAVE_ROOM"], False),
    ("shop", ["LEAVE_ROOM", "DISCARD_POTION"], False),
    ("combat", ["END_TURN", "DISCARD_POTION"], False),
    ("rewards", ["LEAVE_REWARDS", "DISCARD_POTION"], False),
    ("map", ["CHOOSE_MAP_NODE", "DISCARD_POTION"], False),
    ("card_select", ["SELECT_ONE", "SELECT_ONE"], False),
])
def test_only_the_agreed_event_case_bypasses_the_policy(phase, verbs, automatic):
    frame = policy_frame(phase, verbs, 0)
    before = deepcopy(frame)
    action = environment_action(frame)
    assert (action is not None) == automatic
    assert is_policy_frame(frame) == (not automatic)
    if action:
        assert action["actor"] == "event_controller" and action["rule"] == EVENT_RULE
        chosen = next(c for c in frame["legal"]["candidates"] if c["candidate_ref"] == action["candidate_ref"])
        assert chosen["verb"] == "CHOOSE_EVENT_OPTION"
    assert frame == before  # Classification never reveals or changes the public state.


def macro(phase, verbs, index):
    frame = policy_frame(phase, verbs, index)
    return dict(phase=phase, steps=[dict(frame=frame, candidate_ref=frame["legal"]["candidates"][-1]["candidate_ref"],
                                       forced=len(verbs) == 1)])


def test_event_advance_preserves_followup_choices_and_downstream_ppo_returns():
    frames = [
        policy_frame("map", ["CHOOSE_MAP_NODE", "CHOOSE_MAP_NODE"], 0),
        policy_frame("event", ["DISCARD_POTION", "CHOOSE_EVENT_OPTION"], 1),
        policy_frame("event", ["CHOOSE_EVENT_OPTION", "CHOOSE_EVENT_OPTION"], 2),
        policy_frame("card_select", ["SELECT_ONE", "SELECT_ONE"], 3),
    ]
    terminal = policy_frame("terminal", [], 4)
    terminal.update(boundary="terminal", events=[dict(type="run_completed", victory=True,
                    act=3, final_boss_defeated=True)])
    terminal["public"]["outcome"] = dict(victory=True)
    config, policy = ModelConfig.tiny(), Policy()
    engine = ScriptedEngine([*frames, terminal])
    vocabulary = Vocabulary.from_frames(frames, config)
    trace = RolloutRunner(PolicyValue(config), vocabulary).run(engine, "Ironclad", "control", _policy=policy)
    assert trace["status"] == "complete", trace.get("error")
    assert trace["control_version"] == CONTROL_VERSION
    assert policy.phases == ["map", "event", "card_select"]
    assert [m["phase"] for m in trace["macros"]] == policy.phases
    assert [m["return"] for m in trace["macros"]] == [5, 5, 5]
    assert [m["old_log_prob"] for m in trace["macros"]] == [-0.7] * 3
    assert trace["automatic_steps"] == 1
    assert trace["environment_actions"][0]["candidate_ref"] == "1:1"
    validate_run(trace, on_policy=True)
    bad = dict(trace["macros"][1], steps=[dict(frame=frames[1], candidate_ref="1:1", forced=False)])
    trace["macros"].insert(1, bad)
    with pytest.raises(ProtocolError, match="environment steps"):
        validate_run(trace, on_policy=True)
    with pytest.raises(ProtocolError, match="without policy decisions"):
        prepare_replay(vocabulary, [bad["steps"]])


def event_message():
    action = dict(command="choose_event_option", args=dict(option_instance_id="o0"))
    discard = dict(command="discard_potion", args=dict(potion_instance_id="p0"))
    shown = message("event_choice", [action, discard], dict(event_id="TINKER_TIME", options=[
        dict(instance_id="o0", text_key="TINKER_TIME.pages.INITIAL.options.CHOOSE_CARD_TYPE", locked=False),
        dict(instance_id="locked", text_key="UNAVAILABLE", locked=True)]), ascension=0)
    shown["state"]["players"][0]["potions"] = [dict(slot=0, potion=dict(id="FIRE_POTION", instance_id="p0"))]
    return shown, action


def test_steam_uses_the_public_legal_options_and_retains_the_next_real_choice():
    shown, action = event_message()
    policy = SteamPolicy("CHOOSE_EVENT_OPTION")
    assert choose_action(policy, shown) == action
    assert policy.frames == []
    next_action = dict(command="choose_event_option", args=dict(option_instance_id="o1"))
    shown["state"]["legal"]["actions"].append(next_action)
    shown["state"]["legal"]["context"]["options"].append(
        dict(instance_id="o1", text_key="TINKER_TIME.pages.TYPES.options.SKILL", locked=False))
    assert choose_action(policy, shown) == action
    assert len(policy.frames) == 1


def test_infer_emits_the_controller_action_without_model_scoring(tmp_path, monkeypatch, capsys):
    from model import cli

    config = ModelConfig.tiny()
    frame = policy_frame("event", ["DISCARD_POTION", "CHOOSE_EVENT_OPTION"], 0)
    path = tmp_path / "frame.json"
    path.write_text(json.dumps(frame))
    monkeypatch.setattr(cli, "load_model", lambda *_: (
        PolicyValue(config), Vocabulary.from_frames([frame], config), dict(training=dict(precision="no"))))

    def unexpected(*_, **__):
        raise AssertionError("An automatic event advance must not call the model")

    monkeypatch.setattr(cli.SessionPolicy, "choose", unexpected)
    assert cli.main(["infer", "--checkpoint", "unused", "--frame", str(path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["command"]["candidate_ref"] == "0:1"
    assert result["ranking"] is None and result["control"]["actor"] == "event_controller"


def test_new_independent_import_keeps_event_evidence_and_substantive_labels():
    baseline = samples()
    row = deepcopy(baseline[0])
    frame = policy_frame("event", ["CHOOSE_EVENT_OPTION", "DISCARD_POTION"], 99)
    row.update(observation=frame["public"], options=[action_semantics(c) for c in frame["legal"]["candidates"]])
    row["label"] = row["options"][-1]  # A historical discard must not become a policy label.
    row["metadata"]["routing"] = frame["routing"]
    only = convert_group([row])
    assert only["macros"] == [] and training_samples([only]) == []
    assert only["environment_actions"][0]["candidate_ref"] == "c1"
    assert only["automatic_steps"] == 1
    mixed = convert_group([row, *baseline])
    assert mixed["macros"] == convert_group(baseline)["macros"]


@pytest.mark.parametrize("workers", [1, 2])
def test_reindex_removes_only_event_advances_and_preserves_lines_weights_and_batches(tmp_path, workers):
    event = macro("event", ["CHOOSE_EVENT_OPTION", "DISCARD_POTION"], 0)
    kept = [macro("event", ["CHOOSE_EVENT_OPTION", "CHOOSE_EVENT_OPTION"], 1),
            macro("shop", ["LEAVE_ROOM", "DISCARD_POTION"], 2),
            macro("combat", ["END_TURN", "DISCARD_POTION"], 3)]
    base = demonstration("Ironclad", "raw", 4, 2)
    base.update(ascension=0, contract=event["steps"][0]["frame"]["contract"])
    only = dict(base, run_id="only", macros=[event])
    mixed = dict(base, run_id="mixed", macros=[event, *kept])
    (tmp_path / "accepted").mkdir()
    shard = tmp_path / "accepted/00000.jsonl.gz"
    with gzip.open(shard, "wt") as out:
        out.writelines(json.dumps(r) + "\n" for r in (only, mixed))
    raw_bytes = shard.read_bytes()
    stale = gzip.compress(b'{}\n')
    (tmp_path / INDEX).write_bytes(stale)
    with pytest.raises(ProtocolError, match="reindex"):
        RunShards(tmp_path)
    report = rebuild_index(tmp_path, workers=workers)
    assert report["groups"] == 2 and report["training_groups"] == 1
    assert report["raw_macros"] == 5 and report["policy_macros"] == 3
    assert report["excluded_macros"] == {"event": 2}
    assert shard.read_bytes() == raw_bytes
    assert (tmp_path / (INDEX + ".before-" + CONTROL_VERSION)).read_bytes() == stale
    dataset = RunShards(tmp_path)
    assert [r["line"] for r in dataset.rows] == [1]
    window = dataset.read(["accepted/00000.jsonl.gz"])
    assert window == [mixed]
    expected = training_samples([only, mixed])
    actual = training_samples(window, counts=dataset.counts)
    assert [s.phase for s in actual] == ["event", "shop", "combat"]
    assert [s.weight for s in actual] == [s.weight for s in expected] == [1 / 3] * 3
    config = ModelConfig.tiny()
    vocab = Vocabulary.from_frames([s["frame"] for m in kept for s in m["steps"]], config)
    training = TrainConfig(loader_workers=0, logical_batch_size=2, microbatch_size=2)
    batches = list(stream(dataset, vocab, training, window_shards=1, shuffle=False))
    assert [sum(len(items) for items, _ in batch) for batch in batches] == [2, 1]
    # Failure never publishes a partly recounted index.
    good_index = (tmp_path / INDEX).read_bytes()
    (tmp_path / "accepted/00001.jsonl.gz").write_bytes(gzip.compress(b'invalid json\n'))
    with pytest.raises(ValueError):
        rebuild_index(tmp_path, workers=workers)
    assert (tmp_path / INDEX).read_bytes() == good_index
