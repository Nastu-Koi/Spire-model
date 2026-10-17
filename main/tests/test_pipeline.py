"""Frozen name tables and replay batches prepared in worker processes."""

import pytest
import torch

from main.commands import vocabulary_for
from main.network.config import ModelConfig, TrainConfig
from main.data.shards import microbatches, samples
from main.learn.loader import batches
from main.network.model import PolicyNetwork
from main.network.policy import SessionPolicy, prepare_replay, replay_batch
from main.env.protocol import CHARACTERS
from main.encoding.vocabulary import Table, Vocabulary
from main.play.testing import demonstration


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
    model = PolicyNetwork(config).eval()
    results = []
    for workers in (0, 2):
        training = TrainConfig(precision="no", logical_batch_size=5, microbatch_size=2, token_buckets=[32, 64],
                               action_buckets=[8, 16], loader_workers=workers)
        items = samples(runs)
        logicals = list(batches(items, vocabulary, training))
        # A logical batch is the next five samples, split in parts of two at most.
        assert [sum(len(micro) for micro, _ in logical) for logical in logicals] == [5] * (len(items) // 5) + (
            [len(items) % 5] if len(items) % 5 else [])
        assert all(len(micro) <= 2 for logical in logicals for micro, _ in logical)
        for start, logical in zip(range(0, len(items), 5), logicals):
            assert sorted(id(item) for micro, _ in logical for item in micro) == sorted(map(id, items[start:start + 5]))
        with torch.no_grad():
            rows = {id(item): row for logical in logicals for micro, prepared in logical
                    for item, row in zip(micro, model.replay(prepared)[0])}
        results.append(torch.stack([rows[id(item)] for item in items]))
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
    # A sample over the budget on its own is still trained, alone; the short ones go together.
    tight = TrainConfig(precision="no", token_buckets=[32, 64], action_buckets=[8, 16],
                        token_budget=64, loader_workers=0)
    alone = microbatches(items, tight)
    over = [m for m in alone if sum(len(item._lengths) for item in m) * 32 > tight.token_budget]
    assert sum(map(len, alone)) == len(items) and over and all(len(m) == 1 for m in over)
    assert [len(m) for m in alone if m not in over] == [2, 2]


def test_replay_matches_the_sampling_path():
    config = ModelConfig.tiny()
    run = demonstration("Silent", "paths", 6, 3)
    vocabulary = vocabulary_for([run], config)
    torch.manual_seed(0)
    model = PolicyNetwork(config).eval()
    for macro in run["macros"]:
        policy, sampled = SessionPolicy(model, vocabulary), 0.0
        with torch.no_grad():
            for step in macro["steps"]:
                choice = policy.choose(step["frame"], teacher=step["candidate_ref"])
                if choice.log_prob is not None:
                    sampled += float(choice.log_prob)
            log_prob, _ = replay_batch(model, vocabulary, [macro["steps"]])[0]
        assert abs(float(log_prob) - sampled) < 1e-5


def test_replay_at_temperatures_with_a_reference_reports_the_kl_to_it():
    config = ModelConfig.tiny()
    run = demonstration("Silent", "paths", 6, 3)
    vocabulary = vocabulary_for([run], config)
    torch.manual_seed(0)
    model, other = PolicyNetwork(config).eval(), PolicyNetwork(config).eval()
    prepared = prepare_replay(vocabulary, [macro["steps"] for macro in run["macros"]])
    count = len(run["macros"])
    temperatures = torch.linspace(0.3, 1.0, count)
    with torch.no_grad():
        plain = model.replay(prepared)
        # At temperature 1 nothing changes; at another, only the log-probability does.
        assert all(torch.allclose(a, b) for a, b in zip(plain, model.replay(prepared, temperatures=[1.0] * count)))
        cold, entropy = model.replay(prepared, temperatures=temperatures)
        assert torch.allclose(entropy, plain[1]) and not torch.allclose(cold, plain[0])
        # Against itself the KL is zero; against another policy it is the sum over the
        # macro-action's decisions of Σ p (log p − log q), both at its temperature.
        own = model.decision_log_probs(prepared, temperatures)
        assert torch.allclose(model.replay(prepared, temperatures=temperatures, reference=own)[2],
                              torch.zeros(count), atol=1e-6)
        theirs = other.decision_log_probs(prepared, temperatures)
        kl = model.replay(prepared, temperatures=temperatures, reference=theirs)[2]
        terms = (own.exp() * (own - theirs).masked_fill(~prepared.masks, 0)).sum(-1)
        expected = torch.zeros(count).index_add(0, prepared.owners, terms)
        assert torch.allclose(kl, expected) and bool((kl >= -1e-6).all()) and float(kl.sum()) > 0
