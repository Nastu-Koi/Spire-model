"""Controller actions remain replay evidence, separate from policy samples."""

import json
from copy import deepcopy

import pytest

from model.data import samples as training_samples, validate_run
from model.independent import convert_group
from model.protocol import ProtocolError, action_semantics
from model.recorder import RECORDER_SCHEMA, convert_journal
from model.steam import choose_action
from model.testing import demonstration
from model.tests.test_crystal_planner import board_frame, corners
from model.tests.test_independent import samples
from model.tests.test_steam import message, rewards, GOLD, Policy


def test_independent_import_preserves_grid_proof_without_policy_sample():
    rows = samples()
    grid = deepcopy(rows[0])
    frame = board_frame(revealed=corners())
    grid.update(observation=frame["public"], options=[action_semantics(c) for c in frame["legal"]["candidates"]],
                actor="crystal_sphere_planner")
    grid["label"] = grid["options"][0]
    grid["metadata"]["routing"] = dict(episode_id="grid", decision_id="click", state_version=0)
    baseline = convert_group(rows)
    run = convert_group([grid, *rows])
    assert run["macros"] == baseline["macros"]
    assert run["automatic_steps"] == baseline["automatic_steps"] + 1
    assert run["environment_actions"][0]["actor"] == "crystal_sphere_planner"
    assert run["environment_actions"][0]["candidate_ref"] == "c0"


@pytest.mark.parametrize("phase", ["crystal_sphere", "treasure"])
def test_cached_environment_macro_cannot_reenter_training(phase):
    run = demonstration("Ironclad", "fixture", 4, 2)
    macro = run["macros"][0]
    macro["phase"] = phase
    for step in macro["steps"]:
        step["frame"]["public"]["phase"] = phase
    assert validate_run(run) is run  # The historical evidence stays readable.
    assert len(training_samples([run])) == len(run["macros"]) - 1
    assert all(item.phase != phase for item in training_samples([run]))
    macro["phase"] = "combat"
    with pytest.raises(ProtocolError, match="phase disagrees"):
        validate_run(run)


@pytest.mark.parametrize("phase", ["crystal_sphere", "treasure", "event"])
def test_recorder_import_keeps_reward_labels_and_excludes_controller_steps(tmp_path, phase):
    cells = [dict(x=x, y=y, hidden=(x, y) not in corners()) for x in range(11) for y in range(11)]
    clicks = [dict(command="crystal_sphere_cell", args=dict(x=x, y=y, tool=t))
              for x in range(11) for y in range(11) if (x, y) not in corners() for t in ("Small", "Big")]
    grid = message("crystal_sphere", clicks, dict(remaining=3, cells=cells), ascension=0)
    actor = "crystal_sphere_planner"
    if phase == "treasure":
        clicks = [dict(command="open_chest", args={}), dict(command="pick_relic", args=dict(index=None))]
        grid = message("treasure", clicks, [], ascension=0)
        actor = "treasure_controller"
    elif phase == "event":
        from model.tests.test_event_control import event_message

        grid, choice = event_message()
        clicks = grid["state"]["legal"]["actions"]
        assert clicks[0] == choice
        actor = "event_controller"
    policy = Policy("TAKE_REWARD")
    assert choose_action(policy, grid) in clicks
    assert policy.frames == [], "Controller actions must not call the main policy"
    reward = rewards("FEED", "ANGER")
    reward["state"]["run"]["ascension"] = 0
    payloads = []
    for i, (shown, choice, source) in enumerate(((grid, clicks[0], actor), (reward, GOLD, "model"))):
        state = deepcopy(shown["state"])
        state["run"]["seed"] = "fixture"
        payloads.append(dict(state=state, action_id=str(i), order=i, actor=source,
            status="completed", state_boundary="execution_or_semantic_entry", choice_key=choice,
            legal_match=dict(status="matched")))
    entries = [("segment_start", dict(state=payloads[0]["state"], seed="fixture",
                game_version="0.111.0", recorder_version="0.5.8",
                loaded_assemblies=[dict(name="sts2", mvid="fixture")]))]
    entries += [("decision_committed", d) for d in payloads]
    path = tmp_path / "recording.jsonl"
    path.write_text("".join(json.dumps(dict(schema=RECORDER_SCHEMA, seq=i, run_id="run",
        segment_id="segment", kind=kind, data=data)) + "\n" for i, (kind, data) in enumerate(entries, 1)))
    run = convert_journal(path, bc_only=True)
    assert [m["phase"] for m in run["macros"]] == ["rewards"]
    assert run["automatic_steps"] == 1
    assert run["environment_actions"][0]["actor"] == actor
    assert run["provenance"]["rejected_decisions"] == []
