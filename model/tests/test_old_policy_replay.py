"""Before PPO learns from a round, the learner scores it again as the policy that sampled it."""

import math

import pytest
import torch

from model import trainer
from model.cli import vocabulary_for
from model.config import ModelConfig, TrainConfig
from model.data import samples
from model.model import PolicyValue
from model.protocol import CHARACTERS, ProtocolError
from model.rollout import RolloutRunner
from model.testing import SyntheticEngine, demonstration
from model.trainer import Learner


def sampled_round():
    demonstrations = [demonstration(character, "replay", 4, 2) for character in CHARACTERS]
    config = ModelConfig.tiny()
    vocabulary = vocabulary_for(demonstrations, config)
    training = TrainConfig(optimizer="adamw", precision="no", loader_workers=0, logical_batch_size=8,
                           microbatch_size=4, bootstrap_warmup_updates=0, ascension=0)
    learner = Learner(PolicyValue(config), vocabulary, training)
    runner = RolloutRunner(learner.model, vocabulary, version=learner.policy_version,
                           precision="no", ascension=0)
    runs = [runner.run(SyntheticEngine(4, 2), character, f"replay-{index}")
            for character in CHARACTERS for index in range(4)]
    items = samples(runs, ppo=True, horizon_scale=training.horizon_scale)
    return learner, runs, items


def shift(items, key, amount):
    for item in items:
        item.macro[key] += amount * (item.branches if key == "old_log_prob" else 1)


@pytest.fixture
def bf16_limits(monkeypatch):
    """The limits of a BF16 round, on a round whose arithmetic is exact."""
    monkeypatch.setitem(trainer.REPLAY_LIMITS, "no", trainer.REPLAY_LIMITS["bf16"])


def test_a_sampled_round_replays_as_it_was_stored():
    learner, runs, items = sampled_round()
    found = learner.check_old_policy(runs, items)
    assert set(found) == {"old_replay_median_error", "old_replay_p99_error", "old_replay_max_error",
                          "old_value_replay_p99_error", "old_value_replay_max_error"}
    assert all(0 <= error < 1e-4 for error in found.values())


def test_a_round_scored_by_other_weights_is_rejected():
    learner, runs, items = sampled_round()
    torch.manual_seed(0)
    with torch.no_grad():
        for parameter in learner.model.parameters():
            parameter.add_(0.01 * torch.randn_like(parameter))
    with pytest.raises(ProtocolError, match="do not replay"):
        learner.check_old_policy(runs, items)


def test_exact_arithmetic_allows_no_decision_to_differ():
    learner, runs, items = sampled_round()
    shift(items[:1], "old_log_prob", 1e-3)
    with pytest.raises(ProtocolError, match="do not replay"):
        learner.check_old_policy(runs, items)


def test_bf16_noise_is_judged_over_the_round_not_decision_by_decision(bf16_limits):
    learner, runs, items = sampled_round()
    # A round of thousands always holds a few decisions this far out.
    shift(items[:1], "old_log_prob", 0.3)
    shift(items[1:2], "old_value", 0.3)
    found = learner.check_old_policy(runs, items)
    assert found["old_replay_max_error"] == pytest.approx(0.3, abs=1e-4)
    assert found["old_value_replay_max_error"] == pytest.approx(0.3, abs=1e-4)
    assert found["old_replay_median_error"] < 1e-4 and found["old_replay_p99_error"] < 1e-4


@pytest.mark.parametrize("key", ["old_log_prob", "old_value"])
@pytest.mark.parametrize("share, amount", [
    (1.0, 0.01),  # every decision a little: the median
    (0.1, 0.1),  # one in ten clearly: the 99th percentile
    (0.0, 2.0),  # a single one scored as something else
])
def test_bf16_limits_reject_what_noise_does_not_do(bf16_limits, key, share, amount):
    learner, runs, items = sampled_round()
    count = max(1, math.ceil(share * len(items)))
    assert len(items) >= 20 and (share == 1.0 or count < len(items) // 2)
    shift(items[:count], key, amount)
    with pytest.raises(ProtocolError, match="do not replay"):
        learner.check_old_policy(runs, items)


def test_a_large_round_is_checked_on_a_sample_of_it(monkeypatch):
    learner, runs, items = sampled_round()
    monkeypatch.setattr(trainer, "REPLAY_CHECKED", 5)
    replayed = []
    replay = learner._replay
    monkeypatch.setattr(learner, "_replay", lambda prepared, **how: replayed.append(len(prepared.first)) or replay(prepared, **how))
    found = learner.check_old_policy(runs, items)
    assert sum(replayed) == 5 < len(items) and max(found.values()) < 1e-4
    # What is wrong with the round at large shows in any sample of it.
    shift(items, "old_log_prob", 1e-3)
    with pytest.raises(ProtocolError, match="do not replay"):
        learner.check_old_policy(runs, items)


def test_a_value_that_is_not_a_number_is_rejected(bf16_limits):
    learner, runs, items = sampled_round()
    items[0].macro["old_log_prob"] = float("nan")
    with pytest.raises(ProtocolError, match="do not replay"):
        learner.check_old_policy(runs, items)
