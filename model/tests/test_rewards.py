"""Act scoring, duplicate engine notifications, and old training artifact boundaries."""

import json
from copy import deepcopy

import pytest

from model.checkpoint import load_model, save_checkpoint
from model.cli import _learner, parser, vocabulary_for
from model.config import ModelConfig, TrainConfig
from model.data import validate_run
from model.model import PolicyValue
from model.protocol import ProtocolError
from model.rewards import MilestoneLedger, REWARD_VERSION
from model.rollout import RolloutRunner
from model.testing import SyntheticEngine, demonstration


def encounter(act, kind="boss", *, final=True, identity=None):
    return dict(type="encounter_completed", encounter_id=identity or f"{act}:{kind}",
                act=act, kind=kind, result="victory", final_in_act=final)


WIN = dict(type="run_completed", victory=True, act=3, final_boss_defeated=True)


@pytest.mark.parametrize("acts,total", [(0, 0), (1, 1), (2, 3), (3, 8)])
def test_death_keeps_only_earned_act_points(acts, total):
    ledger = MilestoneLedger()
    paid = 0
    for act in range(1, acts + 1):
        paid += ledger.apply([encounter(act)])
    if acts < 3:
        paid += ledger.apply([dict(type="run_completed", victory=False)])
    else:
        paid += ledger.apply([WIN])
    assert paid == total == sum(ledger.components.values())
    assert ledger.components["combat"] == 0


@pytest.mark.parametrize("notifications", [[encounter(3), WIN], [WIN, encounter(3)]])
def test_final_boss_and_run_completion_pay_once_in_either_order(notifications):
    ledger = MilestoneLedger()
    assert ledger.apply(notifications * 3) == 5
    assert ledger.apply([encounter(3, identity="another-notification")]) == 0
    assert ledger.victory_paid and ledger.components["victory"] == 5


def test_only_final_boss_in_an_act_pays_and_repeat_events_do_not():
    ledger = MilestoneLedger()
    assert ledger.apply([encounter(1, final=False, identity="first-boss")]) == 0
    assert ledger.apply([encounter(1, identity="final-boss")]) == 1
    assert ledger.apply([encounter(1, identity="final-boss"), encounter(1, identity="repeat")]) == 0
    for act in (1, 2, 3):
        assert ledger.apply([encounter(act, kind) for kind in ("normal", "elite", "event")]) == 0


def test_reward_ledger_round_trip_and_old_reward_rejection():
    ledger = MilestoneLedger()
    ledger.apply([encounter(1), encounter(2)])
    state = json.loads(json.dumps(ledger.state_dict()))
    restored = MilestoneLedger.restore(state)
    assert restored.apply([encounter(1), encounter(2), encounter(3), WIN]) == 5
    assert sum(restored.components.values()) == 8
    with pytest.raises(ProtocolError, match="Reward version mismatch"):
        MilestoneLedger.restore(dict(state, version="milestones-v1"))
    with pytest.raises(ProtocolError, match="third-act boss"):
        MilestoneLedger().apply([dict(WIN, final_boss_defeated=False)])


def tiny_policy():
    config = ModelConfig.tiny()
    vocabulary = vocabulary_for([demonstration("Ironclad", "reward-test", 4, 2)], config)
    return PolicyValue(config), vocabulary


def test_default_rollout_is_a0_and_old_rewards_cannot_enter_ppo():
    model, vocabulary = tiny_policy()
    run = RolloutRunner(model, vocabulary).run(SyntheticEngine(4, 2), "Ironclad", "fresh")
    assert run["status"] == "complete"
    assert run["ascension"] == run["contract"]["fixed_ascension"] == 0
    assert run["reward_version"] == run["ledger"]["version"] == REWARD_VERSION
    validate_run(run, on_policy=True)
    for location in ("trace", "ledger"):
        old = deepcopy(run)
        if location == "trace":
            old["reward_version"] = "milestones-v1"
        else:
            old["ledger"]["version"] = "milestones-v1"
        with pytest.raises(ProtocolError, match="Reward version mismatch"):
            validate_run(old, on_policy=True)
    unfinished = dict(run, status="unresolved")
    with pytest.raises(ProtocolError, match="Incomplete"):
        validate_run(unfinished, on_policy=True)


def test_checkpoint_keeps_explicit_difficulty_and_rejects_implicit_migration(tmp_path):
    assert TrainConfig().ascension == 0
    assert parser().parse_args(["inspect"]).ascension == 0
    model, vocabulary = tiny_policy()
    path = tmp_path / "old"
    save_checkpoint(path, model, vocabulary,
                    training=TrainConfig(ascension=10, loader_workers=0, precision="no"),
                    progress={"reward_version": "milestones-v1"})
    _, _, manifest = load_model(path)
    assert manifest["training"]["ascension"] == 10
    assert _learner(path, "cpu", weights_only=True)[0].config.ascension == 10
    with pytest.raises(ProtocolError, match="Reward version mismatch"):
        _learner(path, "cpu", require_current_rewards=True)
    assert json.loads((path / "manifest.json").read_text())["progress"]["reward_version"] == "milestones-v1"
    del manifest["training"]["ascension"]
    (path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="declared ascension"):
        load_model(path)
