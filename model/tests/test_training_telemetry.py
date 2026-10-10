"""Live monitoring observes real optimizer steps, without changing training."""

import copy
from dataclasses import asdict
import random
from types import SimpleNamespace

import pytest
import torch

from model.cli import vocabulary_for
from model.config import ModelConfig, TrainConfig
from model.control import is_policy_frame
from model.history import TrainingHistory
from model.loader import batches
from model.data import samples
from model.model import PolicyNetwork
from model.monitor import MonitorStore
from model.protocol import CHARACTERS
from model.testing import demonstration
from model.trainer import Learner


def setup_learner():
    runs = [demonstration(character, "telemetry", 4, 2) for character in CHARACTERS]
    config = ModelConfig.tiny()
    vocabulary = vocabulary_for(runs, config)
    training = TrainConfig(optimizer="adamw", precision="no", loader_workers=0,
                           logical_batch_size=3, microbatch_size=1,
                           bootstrap_warmup_updates=0, update_stats_every=2, ascension=0)
    return Learner(PolicyNetwork(config), vocabulary, training), runs


def test_accuracy_counts_legal_top1_decisions_without_changing_replay_or_gradients(monkeypatch):
    model = PolicyNetwork(ModelConfig.tiny())
    width = model.config.hidden_size
    hidden = torch.zeros(4, 5, width)
    hidden[:, 0, 0] = 1
    hidden[:, 1:, 0] = torch.tensor([[10, 3, 1, 0], [8, 4, 1, 2],
                                    [9, 4, 1, 0], [5, 1, 3, 0]])
    hidden.requires_grad_()
    with torch.no_grad():
        for head in (model.query, model.key):
            head.weight.copy_(torch.eye(width))
            head.bias.zero_()
    monkeypatch.setattr(model, "hidden", lambda batch: hidden)
    prepared = SimpleNamespace(
        batch=SimpleNamespace(lengths=[5] * 4, actions=torch.tensor([[1, 2, 3, 4]] * 4)),
        masks=torch.tensor([[False, True, True, False], [True, True, False, True],
                            [False, True, False, False], [True, True, True, False]]),
        labels=torch.tensor([1, 1, 1, 2]), owners=torch.tensor([0, 0, 0, 1]), first=torch.tensor([0, 3]),
    )
    original = model.replay(prepared)
    measured = model.replay(prepared, with_accuracy=True)
    # The masked highest score in row 0 cannot win; row 2 is a forced choice.
    assert measured[2].tolist() == [[1, 2], [0, 1]]
    assert measured[2].dtype == torch.long and not measured[2].requires_grad
    assert len(original) == 2 and len(measured) == 3
    for actual, expected in zip(measured[:2], original):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    parameters = (hidden, model.query.weight, model.key.weight)
    before = torch.autograd.grad(-original[0].sum(), parameters)
    after = torch.autograd.grad(-measured[0].sum(), parameters)
    for actual, expected in zip(after, before):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_per_update_metrics_match_epoch_aggregate_without_changing_weights():
    learner, runs = setup_learner()
    baseline = Learner(copy.deepcopy(learner.model), learner.vocabulary, learner.config)
    # Exercise multiple microbatches and a short final logical batch.
    items = samples(runs)[:7]
    records = []

    def completed(metrics):
        assert metrics["updates"] == learner.updates
        assert len(records) + 1 == learner.updates
        assert metrics["learning_rates"]
        assert metrics["duration_seconds"] >= 0
        records.append(metrics)

    actual = learner._optimize(batches(items, learner.vocabulary, learner.config),
                               total=len(items), on_update=completed)
    expected = baseline._optimize(batches(items, baseline.vocabulary, baseline.config),
                                  total=len(items))
    assert [row["updates"] for row in records] == [1, 2, 3]
    assert [row["samples"] for row in records] == [3, 3, 1]
    assert [row["accuracy_decisions"] for row in records] == [
        sum(is_policy_frame(step["frame"]) for item in items[start:start + 3] for step in item.macro["steps"])
        for start in range(0, len(items), 3)
    ]
    for row in records:
        assert 0 <= row["accuracy_correct"] <= row["accuracy_decisions"]
        assert row["accuracy"] == row["accuracy_correct"] / row["accuracy_decisions"]
    assert actual["accuracy_correct"] == sum(row["accuracy_correct"] for row in records)
    assert actual["accuracy_decisions"] == sum(row["accuracy_decisions"] for row in records)
    assert actual["accuracy"] == actual["accuracy_correct"] / actual["accuracy_decisions"]
    assert actual["accuracy"] == expected["accuracy"]
    for name in ("loss", "entropy"):
        weighted = sum(row[name] * row["weight"] for row in records) / sum(row["weight"] for row in records)
        assert actual[name] == pytest.approx(weighted)
        assert actual[name] == pytest.approx(expected[name])
    for a, b in zip(learner.model.parameters(), baseline.model.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_bootstrap_writes_visible_updates_before_epoch_completion(tmp_path):
    learner, runs = setup_learner()
    writer = TrainingHistory(tmp_path, "bootstrap", asdict(learner.config))
    epochs = []

    def stop_after_update(metrics, *, epoch, stage):
        writer.append_update(metrics, round=epoch, stage=stage)
        assert stage == "bootstrap" and epoch == 1 and not epochs
        raise KeyboardInterrupt

    try:
        with pytest.raises(KeyboardInterrupt):
            learner.bootstrap(runs, on_update=stop_after_update, on_epoch=lambda *args: epochs.append(args))
        store = MonitorStore(tmp_path)
        assert store.list_runs()["runs"][0]["status"]["updates"] == 1
        row = store.detail(".")
        assert row["latest"]["update"] == learner.updates == 1
        assert row["summary"] is None
        assert row["status"]["updates"] == 1
        assert row["latest"]["metrics"]["accuracy_decisions"] > 0
        assert 0 <= row["latest"]["metrics"]["accuracy"] <= 1
    finally:
        writer.close()


def test_update_counter_survives_epoch_boundaries():
    learner, runs = setup_learner()
    # Simulate the counter restored from a checkpoint.
    learner.updates = 30
    events = []
    random.seed(2)
    learner.bootstrap(runs, epochs=2, on_update=lambda metrics, **context: events.append((metrics["updates"], context["epoch"])))
    assert events[0] == (31, 1)
    assert {epoch for _, epoch in events} == {1, 2}
    assert [step for step, _ in events] == list(range(31, learner.updates + 1))


def test_an_update_limit_stops_in_the_middle_of_an_epoch_and_reports_it():
    learner, runs = setup_learner()
    random.seed(2)
    whole = learner.bootstrap(runs, epochs=1)[-1]["updates"]
    assert whole > 2 and "update_limit" not in learner.bootstrap(runs, epochs=1)[-1]

    learner, runs = setup_learner()
    events, epochs = [], []
    random.seed(2)
    # Without an epoch budget the limit is what ends the run, one update into the second epoch.
    metrics = learner.bootstrap(
        runs, epochs=None, max_updates=whole + 1,
        on_update=lambda metrics, **context: events.append((metrics["updates"], context["epoch"])),
        on_epoch=lambda epoch, metrics: epochs.append((epoch, metrics["updates"], metrics.get("update_limit"))))
    assert learner.updates == whole + 1
    assert [step for step, _ in events] == list(range(1, whole + 2))
    assert epochs == [(1, whole, None), (2, whole + 1, whole + 1)]
    assert metrics[-1]["update_limit"] == whole + 1 and learner.policy_version == 2

    # A limit an epoch reaches exactly ends the run there, whatever epochs were asked for.
    learner, runs = setup_learner()
    random.seed(2)
    assert len(learner.bootstrap(runs, epochs=3, max_updates=whole)) == 1 and learner.updates == whole
    with pytest.raises(ValueError, match="at least one update"):
        learner.bootstrap(runs, max_updates=0)
