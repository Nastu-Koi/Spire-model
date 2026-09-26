"""Driver tests need no game DLL or model dependencies."""

import importlib.util
import sys
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "environment_runner", Path(__file__).parents[1] / "python" / "environment_runner.py"
)
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)


def frame(n=0, boundary="decision", revision=0, **extra):
    return {
        "type": "decision_frame",
        "boundary": boundary,
        "contract": {"fixed_ascension": 10, "training_ready": True},
        "routing": {
            "decision_id": str(revision),
            "state_version": revision,
            "selection_revision": revision,
        },
        "public": {"phase": "test"},
        "legal": {"candidates": [{"candidate_ref": f"c{i}"} for i in range(n)]},
        "events": [],
        **extra,
    }


class Engine:
    def __init__(self, frames):
        self.frames = iter(frames)
        self.commands = []

    def send(self, command):
        self.commands.append(command)
        return next(self.frames)


def no_policy(*args):
    raise AssertionError("Forced steps must bypass the entire policy")


def test_forced_chain_events_survive_to_terminal():
    engine = Engine(
        [
            frame(1),
            frame(1, revision=1, events=[{"reward": 1}]),
            frame(boundary="terminal", events=[{"reward": 3}]),
        ]
    )
    trace = runner.EnvironmentRunner(engine, no_policy).run()
    assert trace.policy_calls == 0
    assert [x.origin for x in trace.executions] == ["forced", "forced"]
    assert trace.events == [{"reward": 1}, {"reward": 3}]
    assert trace.terminal["boundary"] == "terminal"


def test_only_branch_calls_policy_and_sends_current_revision():
    engine = Engine(
        [
            frame(1),
            frame(2, revision=1),
            frame(1, revision=2),
            frame(boundary="terminal"),
        ]
    )

    def choose(public, candidates):
        assert public == {"phase": "test"}
        return candidates[-1]["candidate_ref"]

    trace = runner.EnvironmentRunner(engine, choose).run()
    assert trace.policy_calls == 1
    assert [x.origin for x in trace.executions] == ["forced", "policy", "forced"]
    assert engine.commands[2] == {
        "cmd": "execute_candidate",
        "decision_id": "1",
        "state_version": 1,
        "selection_revision": 1,
        "candidate_ref": "c1",
    }


def test_waiting_is_not_empty_legal_set():
    engine = Engine([frame(boundary="waiting"), frame(1), frame(boundary="terminal")])
    trace = runner.EnvironmentRunner(engine, no_policy, poll_interval=0).run()
    assert trace.policy_calls == 0
    assert engine.commands[:2] == [{"cmd": "advance_to_boundary"}] * 2


@pytest.mark.parametrize(
    "initial,code",
    [
        (frame(), "empty_legal_set"),
        (frame(boundary="unknown"), "unknown_boundary"),
        (
            frame(boundary="error", error={"code": "unsupported_interaction"}),
            "unsupported_interaction",
        ),
        (
            frame(1, contract={"fixed_ascension": 10, "training_ready": False}),
            "prototype_contract_requires_opt_in",
        ),
        (frame(1, contract={"fixed_ascension": 0}), "invalid_ascension_contract"),
        ({"type": "error"}, "invalid_protocol_response"),
    ],
)
def test_error_never_becomes_game_loss_or_default_action(initial, code):
    engine = Engine([initial])
    env = runner.EnvironmentRunner(engine, no_policy)
    with pytest.raises(runner.EnvironmentError, match=code):
        env.run()
    assert len(engine.commands) == 1
    assert env.trace.terminal is None


def test_invalid_policy_reference_is_not_sent():
    engine = Engine([frame(2)])
    with pytest.raises(
        runner.EnvironmentError, match="policy_selected_unknown_candidate"
    ):
        runner.EnvironmentRunner(engine, lambda *_: "other").run()
    assert len(engine.commands) == 1


def test_no_progress_has_a_bound():
    engine = Engine([frame(boundary="waiting")] * 4)
    with pytest.raises(runner.EnvironmentError, match="unresolved_step_limit"):
        runner.EnvironmentRunner(engine, no_policy, max_steps=3, poll_interval=0).run()


def test_terminal_after_last_allowed_action_is_completed():
    engine = Engine([frame(1), frame(boundary="terminal")])
    trace = runner.EnvironmentRunner(engine, no_policy, max_steps=1).run()
    assert trace.terminal is not None
