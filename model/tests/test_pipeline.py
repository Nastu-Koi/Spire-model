"""Frozen name tables and replay batches prepared in worker processes."""

import pytest
import torch

from model.cli import vocabulary_for
from model.config import ModelConfig, TrainConfig
from model.data import microbatches, samples
from model.loader import measure, replays
from model.model import PolicyValue
from model.policy import SessionPolicy, replay_batch
from model.protocol import CHARACTERS
from model.representation import Table, Vocabulary
from model.testing import demonstration


def test_registered_names_never_share_a_row():
    names = [f"field.{i}" for i in range(300)]
    table = Table(names, 512)
    rows = [table.encode(name) for name in names]
    assert len(set(rows)) == 300 and min(rows) == 1 and not table.unregistered
    late = table.encode("field.late")
    assert late not in rows and late < 512 and table.encode("field.late") == late
    assert list(table.unregistered) == ["field.late"]
    with pytest.raises(ValueError):
        Table(names, 301)


def test_vocabulary_round_trips_through_its_state():
    config = ModelConfig.tiny()
    runs = [demonstration("Ironclad", "state", 4, 2)]
    vocabulary = vocabulary_for(runs, config)
    restored = Vocabulary.from_state(vocabulary.state(), config)
    assert restored.digest == vocabulary.digest
    assert restored.relations.encode("owner") == vocabulary.relations.encode("owner")


def test_workers_prepare_the_batches_the_trainer_would():
    config = ModelConfig.tiny()
    runs = [demonstration(c, "loader", 6, 3) for c in CHARACTERS]
    vocabulary = vocabulary_for(runs, config)
    torch.manual_seed(0)
    model = PolicyValue(config).eval()
    results = []
    for workers in (0, 2):
        training = TrainConfig(precision="no", microbatch_size=2, token_buckets=[32, 64],
                               action_buckets=[8, 16], loader_workers=workers)
        items = samples(runs)
        measure(items, training)
        assert all(item._lengths for item in items)
        micros = microbatches(items, training)
        with torch.no_grad():
            results.append(torch.cat([model.replay(prepared)[0] for prepared in replays(micros, vocabulary, training)]))
    assert torch.equal(results[0], results[1])
    with torch.no_grad():
        direct = torch.stack([replay_batch(model, vocabulary, [item.macro["steps"]])[0][0] for item in items])
    assert torch.allclose(results[0], direct, atol=1e-5)


def test_batch_budgets_count_every_decision_of_a_multi_step_sample():
    runs = [demonstration("Ironclad", f"budget-{i}", 10, 5) for i in range(4)]
    # Six 32-token rows fit: one five-step sample with one single-step sample, not two of the former.
    training = TrainConfig(precision="no", token_buckets=[32, 64], action_buckets=[8, 16],
                           token_budget=6 * 32, loader_workers=0)
    items = samples(runs)
    micros = microbatches(items, training)
    assert sum(map(len, micros)) == len(items)
    for micro in micros:
        rows = sum(len(item._lengths) for item in micro)
        assert rows * 32 <= training.token_budget
    # A sample over the budget on its own is still trained, alone.
    tight = TrainConfig(precision="no", token_buckets=[32, 64], action_buckets=[8, 16],
                        token_budget=64, loader_workers=0)
    assert [len(m) for m in microbatches(items, tight)] == [1] * len(items)


def test_replay_matches_the_sampling_path():
    config = ModelConfig.tiny()
    run = demonstration("Silent", "paths", 6, 3)
    vocabulary = vocabulary_for([run], config)
    torch.manual_seed(0)
    model = PolicyValue(config).eval()
    torch.nn.init.normal_(model.value.weight, std=0.1)
    for macro in run["macros"]:
        policy, sampled, value = SessionPolicy(model, vocabulary), 0.0, None
        with torch.no_grad():
            for step in macro["steps"]:
                choice = policy.choose(step["frame"], teacher=step["candidate_ref"])
                if choice.log_prob is not None:
                    sampled += float(choice.log_prob)
                    value = float(choice.value) if value is None else value
            log_prob, replayed, _ = replay_batch(model, vocabulary, [macro["steps"]])[0]
        assert abs(float(log_prob) - sampled) < 1e-5 and abs(float(replayed) - value) < 1e-5
