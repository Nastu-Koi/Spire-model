"""The outcome reward, duplicate engine notifications, and old training artifact boundaries."""

import json
from copy import deepcopy

import pytest

from model.checkpoint import load_model, save_checkpoint
from model.cli import _learner, parser, vocabulary_for
from model.config import ModelConfig, TrainConfig
from model.data import validate_run
from model.model import PolicyNetwork
from model.protocol import ProtocolError
from model.rewards import MilestoneLedger, REWARD_VERSION
from model.rollout import RolloutRunner
from model.trainer import Learner
from model.testing import SyntheticEngine, demonstration


def encounter(act, kind="boss", *, final=True, identity=None):
    return dict(type="encounter_completed", encounter_id=identity or f"{act}:{kind}",
                act=act, kind=kind, result="victory", final_in_act=final)


WIN = dict(type="run_completed", victory=True, act=3, final_boss_defeated=True)


LOSS = dict(type="run_completed", victory=False, act=2, final_boss_defeated=False)


@pytest.mark.parametrize("acts", [0, 1, 2, 3])
def test_a_run_pays_only_when_it_ends_and_only_for_its_outcome(acts):
    ledger = MilestoneLedger()
    for act in range(1, min(acts, 2) + 1):
        assert ledger.apply([encounter(act)]) == 0 and ledger.outcome is None
    won = acts == 3
    assert ledger.apply([WIN if won else LOSS]) == (1 if won else -1)
    assert ledger.outcome is won
    assert sum(ledger.components.values()) == (1 if won else -1)
    # The acts whose boss fell are kept for the reports, though they pay nothing.
    assert ledger.bosses == set(range(1, min(acts, 2) + 1))


@pytest.mark.parametrize("notifications", [[encounter(3), WIN], [WIN, encounter(3)]])
def test_final_boss_and_run_completion_pay_once_in_either_order(notifications):
    ledger = MilestoneLedger()
    assert ledger.apply(notifications * 3) == 1
    assert ledger.apply([encounter(3, identity="another-notification")]) == 0
    assert ledger.victory_paid and ledger.components == {"victory": 1, "defeat": 0}


def test_only_the_end_pays_and_repeat_events_do_not():
    ledger = MilestoneLedger()
    assert ledger.apply([encounter(1, final=False, identity="first-boss")]) == 0 and not ledger.bosses
    assert ledger.apply([encounter(1, identity="final-boss")]) == 0 and ledger.bosses == {1}
    for act in (1, 2, 3):
        assert ledger.apply([encounter(act, kind) for kind in ("normal", "elite", "event")]) == 0
    assert ledger.apply([LOSS] * 3) == -1
    assert ledger.apply([LOSS]) == 0
    assert ledger.defeat_paid and ledger.components == {"victory": 0, "defeat": -1}


def test_an_ended_run_cannot_end_the_other_way_and_an_end_must_say_which():
    lost = MilestoneLedger()
    lost.apply([LOSS])
    with pytest.raises(ProtocolError, match="lost run cannot be won"):
        lost.apply([WIN])
    won = MilestoneLedger()
    won.apply([WIN])
    with pytest.raises(ProtocolError, match="won run cannot be lost"):
        won.apply([LOSS])
    with pytest.raises(ProtocolError, match="whether the run was won"):
        MilestoneLedger().apply([dict(type="run_completed")])


def test_the_terminal_state_pays_a_defeat_no_event_announced_and_no_unconfirmed_victory():
    silent = MilestoneLedger()
    silent.apply([encounter(1)])
    assert silent.finish(False) == -1 and silent.outcome is False
    assert silent.finish(False) == 0
    announced = MilestoneLedger()
    assert announced.apply([LOSS]) == -1 and announced.finish(False) == 0
    won = MilestoneLedger()
    assert won.apply([WIN]) == 1 and won.finish(True) == 0
    with pytest.raises(ProtocolError, match="won run cannot be lost"):
        won.finish(False)
    with pytest.raises(ProtocolError, match="not confirmed"):
        MilestoneLedger().finish(True)


def test_reward_ledger_round_trip_and_old_reward_rejection():
    ledger = MilestoneLedger()
    ledger.apply([encounter(1), encounter(2)])
    state = json.loads(json.dumps(ledger.state_dict()))
    restored = MilestoneLedger.restore(state)
    assert restored.bosses == {1, 2} and restored.outcome is None
    assert restored.apply([encounter(1), encounter(2), encounter(3), WIN]) == 1
    assert sum(restored.components.values()) == 1
    for old in ("milestones-v1", "act-progress-v2"):
        with pytest.raises(ProtocolError, match="Reward version mismatch"):
            MilestoneLedger.restore(dict(state, version=old))
    with pytest.raises(ProtocolError, match="third-act boss"):
        MilestoneLedger().apply([dict(WIN, final_boss_defeated=False)])


def tiny_policy():
    config = ModelConfig.tiny()
    vocabulary = vocabulary_for([demonstration("Ironclad", "reward-test", 4, 2)], config)
    return PolicyNetwork(config), vocabulary


def test_default_rollout_is_a0_and_old_rewards_are_not_sampled_data():
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
                    progress={"reward_version": "milestones-v1", "sampling_round": 0})
    _, _, manifest = load_model(path)
    assert manifest["training"]["ascension"] == 10
    assert _learner(path, "cpu", weights_only=True)[0].config.ascension == 10
    # It sampled rounds: a policy of the removed PPO, which only plays or starts over.
    with pytest.raises(ValueError, match="PPO"):
        _learner(path, "cpu")
    assert json.loads((path / "manifest.json").read_text())["progress"]["reward_version"] == "milestones-v1"
    del manifest["training"]["ascension"]
    (path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="declared ascension"):
        load_model(path)


def test_a_bc_checkpoint_resumes_whatever_reward_it_names_and_a_ppo_one_does_not(tmp_path):
    model, vocabulary = tiny_policy()
    training = TrainConfig(loader_workers=0, precision="no")
    learner = Learner(model, vocabulary, training)

    def saved(name, **progress):
        save_checkpoint(tmp_path / name, model, vocabulary, learner.optimizer, learner.scheduler,
                        training=training, progress={"reward_version": "act-progress-v2", **progress})
        return tmp_path / name

    # Demonstrations taught it all it knows: no reward has shaped its policy.
    cloned = saved("cloned", completed_bootstrap_epochs=17)
    assert _learner(cloned, "cpu")[0].config == training
    # Rounds of the removed PPO shaped these: they play, or start over from their weights.
    for name, progress in (("warmed", {"sampling_round": 0}), ("updated", {"completed_ppo_rounds": 1})):
        with pytest.raises(ValueError, match="PPO"):
            _learner(saved(name, **progress), "cpu")
        assert _learner(tmp_path / name, "cpu", inference=True)[0].updates == 0
        assert _learner(tmp_path / name, "cpu", weights_only=True)[0].policy_version == 0


def test_a_rollout_pays_its_outcome_once_at_its_end():
    model, vocabulary = tiny_policy()
    runner = RolloutRunner(model, vocabulary)
    outcomes = set()
    for index in range(40):
        run = runner.run(SyntheticEngine(4, 2), "Ironclad", f"either-{index}")
        assert run["status"] == "complete"
        outcome = 1.0 if run["victory"] else -1.0
        rewards = [macro["reward"] for macro in run["macros"]]
        assert rewards[-1] == outcome and sum(rewards) == outcome
        assert sum(run["ledger"]["components"].values()) == outcome
        validate_run(run, on_policy=True)
        outcomes.add(outcome)
    assert outcomes == {1.0, -1.0}

