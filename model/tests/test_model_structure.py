"""Structural options of the shared policy: QK-norm, depth-scaled residual init,
and the 200M configuration's parameter budget."""

import copy
import json
import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from model.cli import vocabulary_for
from model.config import ModelConfig, TrainConfig
from model.data import samples
from model.loader import batches
from model.model import PolicyNetwork
from model.optim import _adamw_groups, _muon_parameter_ids
from model.testing import demonstration
from model.trainer import Learner

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


OPTIONS = dict(qk_norm=True, scaled_residual_init=True, head_qk_norm=True,
               relation_bias_per_layer=True, field_attention=True,
               linear_bias=False, rms_norm=True)
NEW_NAMES = ("q_norm", "k_norm", "query_norm", "key_norm", "field_block", "relation.0")


def test_default_options_leave_the_existing_structure_unchanged():
    model = PolicyNetwork(ModelConfig.tiny())
    assert not any(tag in name for name in model.state_dict() for tag in NEW_NAMES)
    assert all(isinstance(m, torch.nn.LayerNorm) for m in model.modules() if "Norm" in type(m).__name__)
    assert all(m.bias is not None for m in model.modules() if isinstance(m, torch.nn.Linear))
    assert ModelConfig(**json.loads((CONFIGS / "rtxpro6000.json").read_text())["model"]) == ModelConfig(
        hidden_size=256, num_heads=4, layers=24, ffn_size=704, local_size=128, local_heads=4,
        local_layers=2, vocabulary_size=8192, compile_blocks=True)


def test_scaled_residual_init_shrinks_only_the_branch_outputs():
    torch.manual_seed(0)
    plain = PolicyNetwork(ModelConfig.tiny())
    torch.manual_seed(0)
    scaled = PolicyNetwork(replace(ModelConfig.tiny(), scaled_residual_init=True))
    for blocks, count in ((["blocks"], 2), (["encoder", "local_blocks"], 1)):
        a = plain
        b = scaled
        for part in blocks:
            a, b = getattr(a, part), getattr(b, part)
        assert len(a) == count
        factor = 1 / math.sqrt(2 * count)
        for x, y in zip(a, b):
            torch.testing.assert_close(y.attention.output.weight, x.attention.output.weight * factor)
            torch.testing.assert_close(y.ffn.output.weight, x.ffn.output.weight * factor)
            torch.testing.assert_close(y.attention.qkv.weight, x.attention.qkv.weight)
            torch.testing.assert_close(y.ffn.input.weight, x.ffn.input.weight)


@pytest.mark.parametrize("backend", ["reference", "sdpa"])
def test_qk_norm_trains_with_either_attention_backend(backend):
    runs = [demonstration("Ironclad", f"qk-{i}", 4 + i, 2) for i in range(2)]
    config = replace(ModelConfig.tiny(), backend=backend, qk_norm=True, scaled_residual_init=True)
    vocabulary = vocabulary_for(runs, config)
    torch.manual_seed(3)
    model = PolicyNetwork(config)
    assert sum("q_norm" in n or "k_norm" in n for n, _ in model.named_parameters()) == 4 * (2 + 1)
    training = TrainConfig(optimizer="adamw", precision="no", loader_workers=0,
                           logical_batch_size=3, microbatch_size=3, bootstrap_warmup_updates=0,
                           update_stats_every=1)
    learner = Learner(model, vocabulary, training)
    items = samples(runs)[:3]
    result = learner._optimize(batches(items, vocabulary, training), total=3)
    assert result["updates"] == 1 and math.isfinite(result["loss"])
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for n, p in model.named_parameters() if "q_norm" in n or "k_norm" in n)


def test_qk_norm_gains_train_under_adamw_without_decay():
    model = PolicyNetwork(replace(ModelConfig.tiny(), qk_norm=True))
    muon = _muon_parameter_ids(model)
    named = [(n, p) for n, p in model.named_parameters() if id(p) not in muon]
    norms = {id(p) for n, p in named if "q_norm" in n or "k_norm" in n}
    assert norms and not norms & muon
    groups = _adamw_groups(named, TrainConfig())
    holding = [g for g in groups if norms & {id(p) for p in g["params"]}]
    assert [g["name"] for g in holding] == ["backbone_no_decay"]
    assert holding[0]["weight_decay"] == 0.0


def step(config, seed=3):
    runs = [demonstration("Ironclad", f"opt-{i}", 4 + i, 2) for i in range(2)]
    vocabulary = vocabulary_for(runs, config)
    torch.manual_seed(seed)
    model = PolicyNetwork(config)
    training = TrainConfig(optimizer="adamw", precision="no", loader_workers=0,
                           logical_batch_size=3, microbatch_size=3, bootstrap_warmup_updates=0,
                           update_stats_every=1)
    learner = Learner(model, vocabulary, training)
    items = samples(runs)[:3]
    result = learner._optimize(batches(items, vocabulary, training), total=3)
    return model, result


@pytest.mark.parametrize("backend", ["reference", "sdpa"])
def test_200m_options_train_with_either_attention_backend(backend):
    config = replace(ModelConfig.tiny(), backend=backend, **OPTIONS)
    model, result = step(config)
    names = dict(model.named_parameters())
    assert result["updates"] == 1 and math.isfinite(result["loss"])
    assert all(any(tag in name for name in names) for tag in NEW_NAMES)
    assert {n for n in names if n.startswith("relation.")} == {f"relation.{i}.weight" for i in range(config.layers)}
    # No Linear layer keeps a bias; every norm is RMSNorm.
    assert [n for n in names if n.endswith(".bias")] == []
    assert all(isinstance(m, torch.nn.RMSNorm) for m in model.modules() if "Norm" in type(m).__name__)
    # Every parameter trains.
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in names.values())


def test_field_attention_keeps_rows_without_fields_finite():
    from model.batch import Fields
    config = replace(ModelConfig.tiny(), field_attention=True)
    torch.manual_seed(0)
    encoder = PolicyNetwork(config).encoder
    rows, width = 3, 4
    ids = torch.randint(1, config.vocabulary_size, (rows, width))
    kinds = torch.randint(1, config.field_size, (rows, width))
    numbers = torch.zeros((rows, width, 4))
    mask = torch.tensor([[1, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]], dtype=torch.float32)
    out = encoder.fields(Fields(ids, kinds, numbers, mask))
    assert out.shape == (rows, config.local_size) and torch.isfinite(out).all()
    # A row without fields carries nothing into the sum: the norm of zero.
    torch.testing.assert_close(out[2], encoder.field_norm(torch.zeros(config.local_size)))


def test_optimizer_groups_cover_the_200m_structure():
    model = PolicyNetwork(replace(ModelConfig.tiny(), **OPTIONS))
    muon = _muon_parameter_ids(model)
    names = {id(p): n for n, p in model.named_parameters()}
    in_muon = {names[i] for i in muon}
    assert {n for n in in_muon if n.startswith("encoder.field_block")} == {
        "encoder.field_block.attention.qkv.weight", "encoder.field_block.attention.output.weight",
        "encoder.field_block.ffn.input.weight", "encoder.field_block.ffn.output.weight"}
    assert not any(n.startswith(("relation.", "query", "key")) for n in in_muon)
    rest = [(n, p) for n, p in model.named_parameters() if id(p) not in muon]
    groups = {g["name"]: {names[id(p)] for p in g["params"]} for g in _adamw_groups(rest, TrainConfig())}
    assert {n for n in groups["embedding_no_decay"] if n.startswith("relation.")} == {
        f"relation.{i}.weight" for i in range(model.config.layers)}
    assert groups["head_no_decay"] == {"query_norm.weight", "key_norm.weight"}
    assert groups["head_decay"] == {"query.weight", "key.weight"}
    assert set(groups) == {"embedding_no_decay", "backbone_decay", "backbone_no_decay", "head_decay", "head_no_decay"}


def test_200m_configuration_meets_its_budget():
    spec = json.loads((CONFIGS / "rtxpro6000-200m.json").read_text())
    config = ModelConfig(**spec["model"])
    training = TrainConfig.from_dict(spec["training"])
    assert config.qk_norm and config.scaled_residual_init and config.compile_blocks
    assert all(getattr(config, key) == value for key, value in OPTIONS.items())
    assert config.hidden_size // config.num_heads == 64 == config.local_size // config.local_heads
    # Same vocabulary layout as the 20M model, so the data pipeline is shared.
    assert (config.vocabulary_size, config.field_size, config.relation_size) == (8192, 1024, 128)
    with torch.device("meta"):
        model = PolicyNetwork(config)
    total = sum(p.numel() for p in model.parameters())
    assert 185e6 <= total <= 205e6
    # The relation bias is [sequences, heads, tokens, tokens]: pairs x heads stays
    # near the 20M budget (41943040 pairs x 4 heads).
    assert training.pair_budget * config.num_heads <= 41943040 * 4 * 1.25
    assert training.token_budget * config.hidden_size <= 163840 * 256 * 2
