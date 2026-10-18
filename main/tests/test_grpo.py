import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from main.commands import vocabulary_for
from main.data.shards import samples
from main.env.protocol import ProtocolError
from main.learn.grpo import Clipped
from main.network.config import ModelConfig
from main.network.model import PolicyNetwork
from main.network.policy import prepare_replay
from main.play.testing import demonstration


def setup(temperature=0.5):
    """A policy, a frozen copy of it, and a run's macro-actions as sampled decisions drawn
    by it at `temperature`: their log-probabilities are the policy's own."""
    config = ModelConfig.tiny()
    run = demonstration("Silent", "paths", 6, 3)
    vocabulary = vocabulary_for([run], config)
    torch.manual_seed(0)
    model = PolicyNetwork(config)
    reference = copy.deepcopy(model)
    items = samples([run])
    prepared = prepare_replay(vocabulary, [item.macro["steps"] for item in items])
    with torch.no_grad():
        drawn = model.replay(prepared, temperatures=[temperature] * len(items))[0]
    items = [replace(item, temperature=temperature, old_log_prob=float(log_prob), advantage=0.0)
             for item, log_prob in zip(items, drawn)]
    learner = SimpleNamespace(model=model, config=SimpleNamespace(precision="fp32"))
    return learner, reference, prepared, items


def gradient(learner, objective, prepared, items):
    learner.model.zero_grad(set_to_none=True)
    loss, entropy, accuracy = objective(learner, prepared, items)
    loss.sum().backward()
    norms = [p.grad.norm() for p in learner.model.parameters() if p.grad is not None]
    return float(torch.stack(norms).norm()) if norms else 0.0


def test_no_advantage_and_no_drift_from_the_reference_is_no_step():
    learner, reference, prepared, items = setup()
    objective = Clipped(reference, clip=0.2, kl=0.01)
    assert gradient(learner, objective, prepared, items) < 1e-6
    seen = objective.take()
    assert seen["ratio_mean"] == pytest.approx(1.0, abs=1e-5) and seen["kl"] == pytest.approx(0.0, abs=1e-6)
    assert seen["zero_share"] == 1.0 and seen["clip_fraction"] == 0.0
    # An advantage moves it.
    moved = [replace(item, advantage=1.0 if index % 2 else -1.0) for index, item in enumerate(items)]
    assert gradient(learner, objective, prepared, moved) > 1e-4
    assert objective.take()["zero_share"] == 0.0


def test_a_ratio_beyond_the_clip_in_the_advantages_direction_stops_pushing():
    learner, reference, prepared, items = setup()
    objective = Clipped(reference, clip=0.2, kl=0.0)
    # Drawn with half the probability the policy now gives it: r = 2.
    doubled = [replace(item, old_log_prob=item.old_log_prob - 0.6931471805599453) for item in items]
    up = [replace(item, advantage=1.0) for item in doubled]
    assert gradient(learner, objective, prepared, up) < 1e-6
    seen = objective.take()
    assert seen["ratio_mean"] == pytest.approx(2.0, rel=1e-4) and seen["clip_fraction"] == 1.0
    # A negative advantage at the same ratio is not clipped: the policy is pulled back.
    down = [replace(item, advantage=-1.0) for item in doubled]
    assert gradient(learner, objective, prepared, down) > 1e-4
    assert objective.take()["clip_fraction"] == 0.0
    # Inside the clip both directions push.
    assert gradient(learner, objective, prepared, [replace(item, advantage=1.0) for item in items]) > 1e-4


def test_an_explored_decision_pushes_only_while_its_ratio_is_within_the_clip_on_either_side():
    learner, reference, prepared, items = setup()
    objective = Clipped(reference, clip=0.2, kl=0.0)
    explored = [replace(item, explored=True) for item in items]
    # Inside the region an explored decision pushes like any other, both ways.
    for sign in (1.0, -1.0):
        assert gradient(learner, objective, prepared, [replace(item, advantage=sign) for item in explored]) > 1e-4
        assert objective.take()["clip_fraction"] == 0.0
    # At r = 2 a plain decision with a negative advantage is pulled back with the whole ratio;
    # an explored one is held: outside the region nothing moves it, whatever the sign.
    doubled = [replace(item, old_log_prob=item.old_log_prob - 0.6931471805599453, advantage=-1.0) for item in items]
    assert gradient(learner, objective, prepared, doubled) > 1e-4
    assert objective.take()["clip_fraction"] == 0.0
    held = [replace(item, explored=True) for item in doubled]
    assert gradient(learner, objective, prepared, held) < 1e-6
    assert objective.take()["clip_fraction"] == 1.0
    assert gradient(learner, objective, prepared, [replace(item, advantage=1.0) for item in held]) < 1e-6
    assert objective.take()["clip_fraction"] == 1.0
    # At r = 1/2 likewise: a plain positive decision pushes on (weakly), an explored one waits.
    halved = [replace(item, old_log_prob=item.old_log_prob + 0.6931471805599453, advantage=1.0) for item in items]
    assert gradient(learner, objective, prepared, halved) > 1e-4
    assert objective.take()["clip_fraction"] == 0.0
    assert gradient(learner, objective, prepared, [replace(item, explored=True) for item in halved]) < 1e-6
    assert objective.take()["clip_fraction"] == 1.0
    # The loss of a held decision is still the surrogate's value: it is detached, not dropped.
    loss, _, _ = objective(learner, prepared, held)
    assert torch.allclose(loss.detach(), torch.full_like(loss, 2.0), atol=1e-3)


def test_the_kl_pulls_back_toward_the_reference_and_is_what_is_reported():
    learner, reference, prepared, items = setup()
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.add_(0.05 * torch.randn_like(parameter))
    objective = Clipped(reference, clip=0.2, kl=0.01)
    assert gradient(learner, objective, prepared, items) > 1e-6
    assert objective.take()["kl"] > 0
    assert all(not parameter.requires_grad for parameter in reference.parameters())
    assert objective.summary()["samples"] == len(items)


def test_a_decision_without_what_the_ratio_needs_is_refused():
    learner, reference, prepared, items = setup()
    with pytest.raises(ProtocolError, match="temperature"):
        Clipped(reference)(learner, prepared, [replace(items[0], temperature=None)] + items[1:])
    with pytest.raises(ValueError):
        Clipped(reference, clip=1.5)
