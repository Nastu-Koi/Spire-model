import hashlib
import json
from unittest.mock import patch

import pytest

from combat_solver_cli import trajectory
from combat_solver_cli.search_support import ReplayMismatch, state_key
from combat_solver_cli.tests.replay_fixture import FakeEngine
from model.data import validate_run
from model.protocol import ProtocolError, action_semantics


def replay_inputs(tmp_path):
    with FakeEngine() as engine:
        frame = engine.reset("Ironclad", "fixture", 0)
    prefix = tmp_path / "prefix.json"
    prefix.write_text(
        json.dumps(
            {
                "character": "Ironclad",
                "seed": "fixture",
                "ascension": 0,
                "records": [
                    {
                        "before_hash": state_key(frame),
                        "action": action_semantics(frame["legal"]["candidates"][0]),
                        "actor": "solver",
                    }
                ],
            }
        )
    )
    for name in ("worker.dll", "Sts2Headless.dll", "GodotSharp.dll"):
        (tmp_path / name).write_bytes(b"test-only assembly hash fixture")
    pinned = {
        "worker_dll": str(tmp_path / "worker.dll"),
        "solver_dll_sha256": "solver",
        "game_dll_sha256": "game",
    }
    return prefix, pinned


def test_only_final_public_decisions_are_exported_as_supervised_unverified(tmp_path):
    prefix, pinned = replay_inputs(tmp_path)
    output = tmp_path / "export"
    with (
        patch.object(trajectory, "configuration", return_value=pinned),
        patch.object(trajectory, "SolverEngine", side_effect=lambda _: FakeEngine()),
    ):
        run = trajectory.verify_and_export("config", prefix, output)
    validate_run(run)
    assert run["source"] == "recorder_bc"
    assert run["teacher_visibility"] == "unverified"
    assert run["provenance"]["bc_only"] is True
    assert run["provenance"]["bosses"] == [1, 2, 3]
    assert (
        run["provenance"]["raw_sha256"]
        == hashlib.sha256((output / "verified_trace.jsonl").read_bytes()).hexdigest()
    )
    step = run["macros"][0]["steps"][0]
    assert len(step["frame"]["legal"]["candidates"]) == 2
    assert "outcome" not in step["frame"]["public"]
    with pytest.raises(ProtocolError, match="supervised-only"):
        validate_run(run, on_policy=True)


@pytest.mark.parametrize(
    "options",
    [
        {"victory": False},
        {"milestones": False},
        {"divergent": True},
        {"native_error": True},
    ],
)
def test_failed_verification_keeps_evidence_without_training_export(tmp_path, options):
    prefix, pinned = replay_inputs(tmp_path)
    output = tmp_path / "export"
    with (
        patch.object(trajectory, "configuration", return_value=pinned),
        patch.object(
            trajectory, "SolverEngine", side_effect=lambda _: FakeEngine(**options)
        ),
        pytest.raises(ReplayMismatch),
    ):
        trajectory.verify_and_export("config", prefix, output)
    assert not (output / "accepted.jsonl").exists()
    assert not (output / "accepted.jsonl.tmp").exists()
    assert (output / "failed_replay_trace.jsonl").exists()
    assert json.loads((output / "verification_error.json").read_text())["message"]


def test_verification_preserves_source_act_configuration(tmp_path):
    prefix,pinned=replay_inputs(tmp_path)
    data=json.loads(prefix.read_text());data['acts']=['ACT.OVERGROWTH','ACT.HIVE','ACT.GLORY']
    prefix.write_text(json.dumps(data))
    class ActsEngine(FakeEngine):
        def send(self,command):
            if command['cmd']=='start_run':
                assert command['acts']==data['acts']
                return self.reset(command['character'],command['seed'],command['ascension'])
            return super().send(command)
    with patch.object(trajectory,'configuration',return_value=pinned),patch.object(trajectory,'SolverEngine',side_effect=lambda _:ActsEngine()):
        result=trajectory.verify_and_export('config',prefix,tmp_path/'export')
    assert result['provenance']['initialization']['acts']==data['acts']


def test_fixed_rule_steps_remain_in_proof_but_not_policy_samples(tmp_path):
    prefix,pinned=replay_inputs(tmp_path)
    data=json.loads(prefix.read_text());data['records'][0].update(actor='crystal_sphere_rule',policy_excluded=True)
    prefix.write_text(json.dumps(data))
    with patch.object(trajectory,'configuration',return_value=pinned),patch.object(trajectory,'SolverEngine',side_effect=lambda _:FakeEngine()):
        result=trajectory.verify_and_export('config',prefix,tmp_path/'export')
    assert result['macros']==[] and result['automatic_steps']==1
    assert json.loads((tmp_path/'export'/'verified_trace.jsonl').read_text().splitlines()[0])['actor']=='crystal_sphere_rule'
