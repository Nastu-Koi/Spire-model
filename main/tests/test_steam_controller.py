"""The controller beside the game (`main.steam.Controller`): it plays across runs, waits
out the phases and states that are not its to play, keeps the panel's step interval,
reports every decision, and halts on an error until the panel's reset."""

import json
from collections import deque

import pytest

from main.env.protocol import ProtocolError, validate_frame
from main.steam import DECISIONS, EMPTY_SELECTIONS, ERRORS, RUNS, Controller, live_frame, parse_duration
from main.tests.test_steam import Policy, offer


class Bridge:
    """Answers scripted per command: each queue gives its answers in turn and repeats
    its last one; an answer that is an exception is raised. Every request is kept."""

    def __init__(self, **answers):
        self.queues = {command: deque(items) for command, items in answers.items()}
        self.calls = []

    def request(self, command, **payload):
        self.calls.append((command, payload))
        queue = self.queues.get(command)
        if not queue:
            raise AssertionError(f"Unscripted bridge command: {command}")
        answer = queue.popleft() if len(queue) > 1 else queue[0]
        if isinstance(answer, Exception):
            raise answer
        return dict(answer)

    def requests(self, command):
        return [payload for name, payload in self.calls if name == command]


class Clock:
    """A clock that only `sleep` moves."""

    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class Resetting(Policy):
    def __init__(self, verb):
        super().__init__(verb)
        self.resets = 0

    def reset(self):
        self.resets += 1


def decision(token="t1", episode="e1", settings=None, ascension=7, reset_count=0, **run):
    shown = offer("FEED", "ANGER")
    shown["state"]["run"].update(ascension=ascension, **run)
    return dict(shown, status="decision", token=token, episode=episode, reset_count=reset_count,
                settings=settings or dict(control_combat=True, control_outside=True, step_interval_seconds=0))


def controller(bridge, policy, tmp_path, **options):
    clock = Clock()
    return Controller(bridge, policy, directory=tmp_path, clock=clock, sleep=clock.sleep, say=lambda _: None,
                      **options), clock


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_the_controller_plays_across_runs_and_reports_each_decision(tmp_path):
    bridge = Bridge(
        observe=[decision("t1", floor=3), dict(status="terminal", victory=True, reset_count=0),
                 dict(status="terminal", victory=True, reset_count=0), decision("t2", episode="e2", floor=1)],
        execute=[dict(status="executed")], report=[dict(status="reported")])
    policy = Resetting("TAKE_CARD_REWARD")
    control, _ = controller(bridge, policy, tmp_path, max_decisions=2)
    assert control.run() == dict(runs=1, wins=1, decisions=2, errors=0, status="decision_limit")
    # Each decision was executed with its token and reported with what the panel shows.
    assert [p["token"] for p in bridge.requests("execute")] == ["t1", "t2"]
    reports = bridge.requests("report")
    assert [r["kind"] for r in reports] == ["decision", "decision"]
    assert reports[0]["summary"] == "TAKE_CARD_REWARD FEED" and reports[0]["floor"] == 3 and reports[0]["phase"] == "card_reward"
    assert reports[0]["controller"] is False and reports[0]["action"]["command"] == "select_reward_option"
    assert lines(tmp_path / DECISIONS)[1]["decision"] == 2 and lines(tmp_path / RUNS) == [
        dict(time=lines(tmp_path / RUNS)[0]["time"], victory=True, decisions=1)]
    # The run's end was counted once, however long the terminal screen stayed.
    assert sum(1 for name, _ in bridge.calls if name == "observe") == 4


def test_idle_statuses_are_waited_out_and_only_waiting_is_advanced(tmp_path):
    bridge = Bridge(observe=[dict(status="manual", reason="combat is yours", reset_count=0),
                             dict(status="paused", reset_count=0), dict(status="halted", reset_count=0),
                             dict(status="waiting", reason="menu", reset_count=0), decision()],
                    execute=[dict(status="executed")], advance=[dict(status="waiting")], report=[dict(status="reported")])
    control, clock = controller(bridge, Resetting("TAKE_CARD_REWARD"), tmp_path, max_decisions=1)
    assert control.run()["decisions"] == 1
    names = [name for name, _ in bridge.calls]
    assert names == ["observe"] * 4 + ["advance", "observe", "execute", "report"]
    assert clock.now - 1000 == pytest.approx(4 * control.poll)


def test_an_error_halts_the_controller_until_the_panel_resets(tmp_path):
    bridge = Bridge(
        observe=[decision("t1"), dict(status="halted", reset_count=0), dict(status="halted", reset_count=0),
                 dict(status="halted", reset_count=1), decision("t2", reset_count=1)],
        execute=[ProtocolError("Game UI is not ready for select_reward_option"), dict(status="executed")],
        report=[dict(status="reported")])
    policy = Resetting("TAKE_CARD_REWARD")
    control, _ = controller(bridge, policy, tmp_path, max_decisions=1)
    assert control.run() == dict(runs=0, wins=0, decisions=1, errors=1, status="decision_limit")
    reports = bridge.requests("report")
    assert reports[0]["kind"] == "error" and "not ready" in reports[0]["error"] and reports[0]["after"] is None
    assert reports[1]["kind"] == "decision"
    (error,) = lines(tmp_path / ERRORS)
    assert error["exception"] == "ProtocolError" and "not ready" in error["error"]
    # The session was reset when halting and again when the panel's count changed.
    assert policy.resets >= 2
    # Nothing was executed while halted; the first decision after the reset was.
    executes = [i for i, (name, _) in enumerate(bridge.calls) if name == "execute"]
    halted = [i for i, (name, payload) in enumerate(bridge.calls) if name == "observe"]
    assert executes[-1] > halted[-1]


def test_a_bridge_that_stays_silent_during_a_halt_is_retried_until_the_deadline(tmp_path):
    bridge = Bridge(observe=[TimeoutError("Steam bridge did not respond")], report=[TimeoutError("still silent")])
    control, clock = controller(bridge, Resetting("TAKE_CARD_REWARD"), tmp_path, run_for=5)
    assert control.run() == dict(runs=0, wins=0, decisions=0, errors=1, status="run_for")
    assert clock.now - 1000 >= 5 and lines(tmp_path / ERRORS)[0]["exception"] == "TimeoutError"


def test_the_step_interval_comes_from_the_panel(tmp_path):
    slow = dict(control_combat=True, control_outside=True, step_interval_seconds=2.5)
    bridge = Bridge(observe=[decision("t1", settings=slow), decision("t2", settings=slow), decision("t3", settings=slow)],
                    execute=[dict(status="executed")], report=[dict(status="reported")])
    control, clock = controller(bridge, Resetting("TAKE_CARD_REWARD"), tmp_path, max_decisions=2)
    executed = []
    original = bridge.request

    def timed(command, **payload):
        if command == "execute":
            executed.append(clock.now)
        return original(command, **payload)
    bridge.request = timed
    assert control.run() == dict(runs=0, wins=0, decisions=2, errors=0, status="decision_limit")
    assert executed[1] - executed[0] == pytest.approx(2.5)


def test_a_stale_execution_is_observed_again_and_a_reset_count_change_restarts_the_session(tmp_path):
    bridge = Bridge(observe=[decision("t1"), decision("t2", reset_count=3)],
                    execute=[dict(status="stale"), dict(status="executed")], report=[dict(status="reported")])
    policy = Resetting("TAKE_CARD_REWARD")
    control, _ = controller(bridge, policy, tmp_path, max_decisions=1)
    assert control.run()["decisions"] == 1
    assert [p["token"] for p in bridge.requests("execute")] == ["t1", "t2"] and policy.resets == 2


def test_any_ascension_is_played_without_a_warning(tmp_path):
    bridge = Bridge(observe=[decision("t1", ascension=15), decision("t2", ascension=15)],
                    execute=[dict(status="executed")], report=[dict(status="reported")])
    control, _ = controller(bridge, Resetting("TAKE_CARD_REWARD"), tmp_path, max_decisions=2)
    assert control.run()["decisions"] == 2
    assert [r["kind"] for r in bridge.requests("report")] == ["decision", "decision"]
    frame = live_frame(decision(ascension=15))[0]
    assert frame["contract"]["fixed_ascension"] == 15 and validate_frame(frame)
    with pytest.raises(ProtocolError, match="single-player"):
        live_frame(decision(ascension=-1))


def test_durations_are_read_in_hours_minutes_and_seconds():
    assert [parse_duration(x) for x in ("8h", "45m", "1h30m", "90s", "90", "2.5h")] == [
        28800, 2700, 5400, 90, 90, 9000]
    with pytest.raises(ValueError, match="duration"):
        parse_duration("soon")
    with pytest.raises(ValueError):
        Controller(None, None, directory="x", run_for=0)


def selecting(cards):
    """A combat card selection of one of `cards` that may be skipped, as the bridge sends it."""
    shown = decision()
    legal = shown["state"]["legal"]
    legal.update(scope="card_select", context=[dict(id=f"CARD.{name}", instance_id=f"s{i}", cost=1)
                                               for i, name in enumerate(cards)],
                 actions=[dict(command="select_card", args=dict(card_instance_id=f"s{i}")) for i in range(len(cards))],
                 selection=dict(min=0, max=1, cancelable=True, ordered=True))
    shown["state"]["combat"] = dict(in_progress=True)
    return shown


def test_a_selection_that_offers_no_card_is_written_down_whole(tmp_path):
    # No card sent, or cards with bounds that let none be taken: both reach the policy cardless.
    empty, closed, full = selecting([]), selecting(["FEED"]), selecting(["FEED"])
    closed["state"]["legal"]["selection"]["max"] = 0
    bridge = Bridge(observe=[empty, closed, full], execute=[dict(status="executed")],
                    report=[dict(status="reported")])
    control, _ = controller(bridge, Policy("CANCEL"), tmp_path, max_decisions=3)
    assert control.run()["decisions"] == 3
    assert [record["message"] for record in lines(tmp_path / EMPTY_SELECTIONS)] == [empty, closed]


def test_a_skippable_choose_a_card_screen_reads_as_the_engine_offers_it():
    # The mod asks as the engine's selector is asked: at most one, skipping is finishing
    # with none, no cancel; so a chosen card cannot be thrown away by a cancel.
    shown = selecting(["FEED", "ANGER", "HAVOC"])
    shown["state"]["legal"]["selection"].update(min=0, max=1, cancelable=False)
    frame, _ = live_frame(shown)
    assert [c["verb"] for c in frame["legal"]["candidates"]] == ["SELECT_ONE"] * 3 + ["FINISH_SELECTION"]
    frame, _ = live_frame(shown, selected=["s1"])
    assert [c["verb"] for c in frame["legal"]["candidates"]] == ["FINISH_SELECTION"]
