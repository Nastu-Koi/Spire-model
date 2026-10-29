"""Controller actions remain replay evidence, separate from policy samples."""

from copy import deepcopy

import pytest

from main.data.shards import samples as training_samples, validate_run
from main.demos.independent import convert_group
from main.env.protocol import ProtocolError, action_semantics
from main.play.testing import demonstration
from main.tests.test_crystal_planner import board_frame, corners
from main.tests.test_independent import samples


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
