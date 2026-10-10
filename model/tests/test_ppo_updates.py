"""PPO stops updating on a round when its recent batches, not one of them, pass the KL target,
paces its step by how far the round got, and learns from demonstrations beside the round."""

from collections import Counter

import pytest
import torch

from model import trainer
from model.cli import vocabulary_for
from model.config import ModelConfig, TrainConfig
from model.model import PolicyValue
from model.optim import PPO_STEP_FLOOR, PPO_STEP_GROWTH, PPO_STEP_SHRINK
from model.protocol import CHARACTERS
from model.rollout import RolloutRunner
from model.testing import SyntheticEngine, demonstration
from model.trainer import KL_WINDOW, Learner

TARGET = 0.015


def round_with_scripted_kl(monkeypatch, script, **settings):
    """A learner and a sampled round; update n of the round reports a KL of script(n)
    for each of its macro-actions (the first update is n = 1)."""
    demonstrations = [demonstration(character, "ppo", 4, 2) for character in CHARACTERS]
    config = ModelConfig.tiny()
    vocabulary = vocabulary_for(demonstrations, config)
    training = TrainConfig(**{**dict(
        optimizer="adamw", precision="no", loader_workers=0, logical_batch_size=2,
        microbatch_size=2, bootstrap_warmup_updates=0, ascension=0,
        ppo_epochs=2, target_kl=TARGET), **settings})
    torch.manual_seed(0)
    learner = Learner(PolicyValue(config), vocabulary, training)
    runner = RolloutRunner(learner.model, vocabulary, version=learner.policy_version, precision="no", ascension=0)
    runs = [runner.run(SyntheticEngine(4, 2), character, f"ppo-{index}")
            for character in CHARACTERS for index in range(4)]
    learner.check_old_policy = lambda runs, items: {}
    objective = trainer.ppo_objective

    first = learner.updates

    def scripted(new_log_prob, old_log_prob, advantage, clip_ratio):
        actor, kl, clipped = objective(new_log_prob, old_log_prob, advantage, clip_ratio)
        return actor, torch.full_like(kl, script(learner.updates - first + 1)), clipped

    monkeypatch.setattr(trainer, "ppo_objective", scripted)
    return learner, runs


def updates_of(learner, runs, demonstrations=None):
    records = []
    metrics = learner.ppo(runs, demonstrations,
                          on_update=lambda update, *, epoch, stage: records.append((epoch, update)))
    return records, metrics


def test_one_batch_over_the_target_does_not_stop_the_round(monkeypatch):
    learner, runs = round_with_scripted_kl(monkeypatch, lambda n: 0.001)
    records, _ = updates_of(learner, runs)
    per_epoch = Counter(epoch for epoch, _ in records)
    assert set(per_epoch) == {1, 2} and per_epoch[1] == per_epoch[2] >= 2 * KL_WINDOW
    # The same round again, with the first batch of the second pass far over the target:
    # the batches before it are remembered across the passes.
    spike = per_epoch[1] + 1
    learner, runs = round_with_scripted_kl(monkeypatch, lambda n: 3 * TARGET if n == spike else 0.001)
    records, metrics = updates_of(learner, runs)
    assert len(records) == 2 * per_epoch[1] and not any(update["early_stop"] for _, update in records)
    assert records[spike - 1][1]["kl"] == pytest.approx(3 * TARGET)
    assert records[spike - 1][1]["kl_recent"] == pytest.approx((3 * TARGET + (KL_WINDOW - 1) * 0.001) / KL_WINDOW)
    assert not any(result.get("early_stop") for result in metrics)


def test_the_round_stops_once_its_recent_batches_average_over_the_target(monkeypatch):
    # Quiet for a window, then every batch at twice the target: half a window of those
    # brings the average over it.
    learner, runs = round_with_scripted_kl(monkeypatch, lambda n: 0.001 if n <= KL_WINDOW else 2 * TARGET)
    records, metrics = updates_of(learner, runs)
    means = [update["kl_recent"] for _, update in records]
    assert all(mean <= TARGET for mean in means[:-1]) and means[-1] > TARGET
    assert len(records) == KL_WINDOW + KL_WINDOW // 2
    assert [update["early_stop"] for _, update in records] == [0] * (len(records) - 1) + [1]
    # The stop ends the round's updates, not only the pass it came in.
    assert len(metrics) == 1 and metrics[0]["early_stop"] == 1


def test_a_policy_that_leaps_is_stopped_before_a_window_has_passed(monkeypatch):
    learner, runs = round_with_scripted_kl(monkeypatch, lambda n: 0.0 if n == 1 else 10 * TARGET)
    records, _ = updates_of(learner, runs)
    # The first batch is scored before any update; the second already shows the leap.
    assert len(records) == 2 and records[-1][1]["early_stop"] == 1


def test_a_round_the_budget_stopped_shrinks_the_step_of_the_rounds_after_it(monkeypatch):
    learner, runs = round_with_scripted_kl(monkeypatch, lambda n: 0.001)
    planned = len(updates_of(learner, runs)[0])
    # Under half the budget spent by the end: the step would grow, but never past the
    # configured rates.
    assert learner.scheduler.ppo_step == 1.0
    # Quiet for most of the round, then every batch at twice the target.
    quiet = [int(0.7 * planned)]
    learner, runs = round_with_scripted_kl(monkeypatch, lambda n: 0.001 if n <= quiet[0] else 2 * TARGET)
    records, metrics = updates_of(learner, runs)
    share = len(records) / planned
    assert PPO_STEP_SHRINK < share < 1 and metrics[-1]["early_stop"] == 1
    quiet[0] = 10 ** 9
    assert metrics[0]["ppo_step"] == 1.0 and learner.scheduler.ppo_step == pytest.approx(share)
    before = records[0][1]["learning_rates"]
    # The next round moves the policy at the smaller step; the value head keeps its rate.
    records, metrics = updates_of(learner, runs)
    after = records[0][1]["learning_rates"]
    assert metrics[0]["ppo_step"] == pytest.approx(share)
    assert {name for name in after if name.startswith("value")}
    for name, rate in after.items():
        assert rate == pytest.approx(before[name] * (1.0 if name.startswith("value") else share))
    # That round ran to its end with little of the budget spent: the step grows back.
    assert learner.scheduler.ppo_step == pytest.approx(min(1.0, share * PPO_STEP_GROWTH))


def test_the_step_is_at_most_halved_by_a_round_and_holds_where_the_budget_is_used(monkeypatch):
    learner, runs = round_with_scripted_kl(monkeypatch, lambda n: 0.0 if n == 1 else 10 * TARGET)
    updates_of(learner, runs)
    assert learner.scheduler.ppo_step == PPO_STEP_SHRINK
    # A round that ends with most of its budget spent is paced as it should be.
    learner, runs = round_with_scripted_kl(monkeypatch, lambda n: 0.8 * TARGET)
    learner.scheduler.ppo_step = 0.25
    records, metrics = updates_of(learner, runs)
    assert not metrics[-1].get("early_stop") and learner.scheduler.ppo_step == 0.25
    for _ in range(20):
        learner.scheduler.pace(0.0, 1.0)
    assert learner.scheduler.ppo_step == PPO_STEP_FLOOR


def test_the_step_is_kept_with_the_schedule(monkeypatch):
    learner, runs = round_with_scripted_kl(monkeypatch, lambda n: 0.0 if n == 1 else 10 * TARGET)
    updates_of(learner, runs)
    state = learner.scheduler.state_dict()
    assert state["ppo_step"] == PPO_STEP_SHRINK
    other = Learner(PolicyValue(ModelConfig.tiny()), learner.vocabulary, learner.config)
    other.scheduler.load_state_dict(state)
    assert other.scheduler.ppo_step == PPO_STEP_SHRINK
    # A schedule saved before rounds were paced starts at the configured rates.
    del state["ppo_step"]
    other.scheduler.load_state_dict(state)
    assert other.scheduler.ppo_step == 1.0


def demonstrations_for(learner):
    return [demonstration(character, "ppo", 4, 2) for character in CHARACTERS]


def test_ppo_updates_learn_from_a_batch_of_demonstrations_each(monkeypatch):
    learner, runs = round_with_scripted_kl(monkeypatch, lambda n: 0.001)
    records, metrics = updates_of(learner, runs, demonstrations_for(learner))
    # No coefficient, no demonstrations in the update.
    assert all("demonstration_nll" not in update for _, update in records)
    assert "demonstration_nll" not in metrics[0]

    # Every demonstration in every update, weighed far above the round's own loss, at
    # rates that show in one round: the policy moves towards them.
    rates = dict(head_lr=3e-3, backbone_lr=3e-3, embedding_lr=3e-3)
    learner, runs = round_with_scripted_kl(monkeypatch, lambda n: 0.001, bootstrap_coef=1000.0,
                                           bootstrap_batch_size=64, **rates)
    demonstrations = demonstrations_for(learner)
    macros = sum(len(run["macros"]) for run in demonstrations)
    assert macros <= 64
    records, metrics = updates_of(learner, runs, demonstrations)
    nll = [update["demonstration_nll"] for _, update in records]
    assert len(nll) == len(records) and nll[-1] < 0.5 * nll[0]
    assert metrics[0]["demonstration_nll"] == pytest.approx(
        sum(update["demonstration_nll"] for epoch, update in records if epoch == 1)
        / sum(epoch == 1 for epoch, _ in records))

    # The same round without them leaves their NLL where the round's own loss takes it.
    alone, runs = round_with_scripted_kl(monkeypatch, lambda n: 0.001, **rates)
    updates_of(alone, runs)
    scored = lambda who: sum(
        group["nll"] for group in who.evaluate_bootstrap(demonstrations).values()) / macros
    assert scored(learner) < 0.5 * scored(alone)


def test_demonstrations_come_in_batches_of_their_own_size_without_end(monkeypatch):
    learner, runs = round_with_scripted_kl(monkeypatch, lambda n: 0.001, bootstrap_coef=0.1,
                                           bootstrap_batch_size=3)
    demonstrations = demonstrations_for(learner)
    macros = sum(len(run["macros"]) for run in demonstrations)
    seen = []
    anchors = learner._anchors

    def counted(items):
        for parts in anchors(items):
            seen.append(sum(len(micro) for micro, _ in parts))
            yield parts

    monkeypatch.setattr(learner, "_anchors", counted)
    records, _ = updates_of(learner, runs, demonstrations)
    # More updates than the demonstrations fill once: they are drawn again in a new order.
    assert len(records) * 3 > macros
    assert len(seen) >= len(records) and set(seen) <= {3, macros % 3 or 3}

