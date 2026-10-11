"""The structure of the shared policy: normalized queries and keys, depth-scaled
residual init, field attention, a relation table per layer, RMSNorm and no Linear
biases; and the parameter budget of the 200M configuration."""

import json
import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch

import model.model as network
from model.cli import vocabulary_for
from model.config import ModelConfig, TrainConfig
from model.data import samples
from model.loader import batches
from model.model import PolicyNetwork
from model.optim import _adamw_groups, _muon_parameter_ids
from model.testing import demonstration
from model.trainer import Learner

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


NAMES = ("q_norm", "k_norm", "query_norm", "key_norm", "field_block", "relation.0")
SMALL = dict(hidden_size=256, num_heads=4, layers=24, ffn_size=704, local_size=128, local_heads=4,
             local_layers=2, vocabulary_size=8192, compile_blocks=True)
LARGE = dict(hidden_size=768, num_heads=12, layers=24, ffn_size=2304, local_size=384, local_heads=6,
             local_layers=2, vocabulary_size=8192, compile_blocks=True, checkpoint_layers=True)


def test_the_profiles_are_one_structure_at_two_sizes():
    for name, sizes in (("rtxpro6000.json", SMALL), ("rtxpro6000-200m.json", LARGE)):
        assert ModelConfig(**json.loads((CONFIGS / name).read_text())["model"]) == ModelConfig(**sizes)
    # A profile of the structure before this one names its version and is refused.
    with pytest.raises(ValueError, match="architecture version"):
        ModelConfig(**dict(SMALL, architecture_version=2))


def test_residual_branch_outputs_start_scaled_by_depth(monkeypatch):
    torch.manual_seed(0)
    scaled = PolicyNetwork(ModelConfig.tiny())
    # The same draws without the scaling, which draws nothing itself.
    monkeypatch.setattr(network, "scale_residual_outputs", lambda blocks: None)
    torch.manual_seed(0)
    plain = PolicyNetwork(ModelConfig.tiny())
    pairs = [(plain.blocks, scaled.blocks, 2), (plain.encoder.local_blocks, scaled.encoder.local_blocks, 1),
             ([plain.encoder.field_block], [scaled.encoder.field_block], 1)]
    for a, b, count in pairs:
        assert len(a) == count
        factor = 1 / math.sqrt(2 * count)
        for x, y in zip(a, b):
            torch.testing.assert_close(y.attention.output.weight, x.attention.output.weight * factor)
            torch.testing.assert_close(y.ffn.output.weight, x.ffn.output.weight * factor)
            torch.testing.assert_close(y.attention.qkv.weight, x.attention.qkv.weight)
            torch.testing.assert_close(y.ffn.input.weight, x.ffn.input.weight)


@pytest.mark.parametrize("backend", ["reference", "sdpa"])
def test_every_parameter_trains_with_either_attention_backend(backend):
    runs = [demonstration("Ironclad", f"opt-{i}", 4 + i, 2) for i in range(2)]
    config = replace(ModelConfig.tiny(), backend=backend)
    vocabulary = vocabulary_for(runs, config)
    torch.manual_seed(3)
    model = PolicyNetwork(config)
    training = TrainConfig(optimizer="adamw", precision="no", loader_workers=0,
                           logical_batch_size=3, microbatch_size=3, bootstrap_warmup_updates=0,
                           update_stats_every=1)
    learner = Learner(model, vocabulary, training)
    items = samples(runs)[:3]
    result = learner._optimize(batches(items, vocabulary, training), total=3)
    names = dict(model.named_parameters())
    assert result["updates"] == 1 and math.isfinite(result["loss"])
    assert all(any(tag in name for name in names) for tag in NAMES)
    # Queries and keys are normalized in every block: the global layers, the local one
    # and the field block.
    assert sum("q_norm" in n or "k_norm" in n for n in names) == 2 * (config.layers + config.local_layers + 1)
    assert {n for n in names if n.startswith("relation.")} == {f"relation.{i}.weight" for i in range(config.layers)}
    # No Linear layer keeps a bias; every norm is RMSNorm.
    assert [n for n in names if n.endswith(".bias")] == []
    assert all(isinstance(m, torch.nn.RMSNorm) for m in model.modules() if "Norm" in type(m).__name__)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in names.values())


def test_qk_norm_gains_train_under_adamw_without_decay():
    model = PolicyNetwork(ModelConfig.tiny())
    muon = _muon_parameter_ids(model)
    named = [(n, p) for n, p in model.named_parameters() if id(p) not in muon]
    norms = {id(p) for n, p in named if "q_norm" in n or "k_norm" in n}
    assert norms and not norms & muon
    groups = _adamw_groups(named, TrainConfig())
    holding = [g for g in groups if norms & {id(p) for p in g["params"]}]
    assert [g["name"] for g in holding] == ["backbone_no_decay"]
    assert holding[0]["weight_decay"] == 0.0


def test_field_attention_keeps_rows_without_fields_finite():
    from model.batch import Fields
    config = ModelConfig.tiny()
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


def test_optimizer_groups_cover_the_structure():
    model = PolicyNetwork(ModelConfig.tiny())
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
    training = TrainConfig(**spec["training"])
    assert config.hidden_size // config.num_heads == 64 == config.local_size // config.local_heads
    # Same vocabulary layout as the 20M profile, so the data pipeline is shared.
    assert (config.vocabulary_size, config.field_size, config.relation_size) == (8192, 1024, 128)
    with torch.device("meta"):
        model = PolicyNetwork(config)
    total = sum(p.numel() for p in model.parameters())
    assert 185e6 <= total <= 205e6
    # The relation bias is [sequences, heads, tokens, tokens]: pairs x heads stays
    # near the 20M profile's budget (41943040 pairs x 4 heads).
    assert training.pair_budget * config.num_heads <= 41943040 * 4 * 1.25
    assert training.token_budget * config.hidden_size <= 163840 * 256 * 2
