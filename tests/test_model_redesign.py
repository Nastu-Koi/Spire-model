from copy import deepcopy

import pytest
import torch

from model.config import ModelConfig
from model.attention import MapAttentionBias
from model.model import PolicyValue, SwiGLU
from model.representation import (Observation, Vocabulary, effect_tree, fields_of,
                                  map_relations)


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(2)


def make_observation(*, extra=False):
    entities = [
        {"entity_type": "global", "phase": "map"},
        {"ref": "node:a", "entity_type": "map_node", "floor": 0, "x": 0},
        {"ref": "node:b", "entity_type": "map_node", "floor": 2, "x": 1},
        {"ref": "player", "entity_type": "player", "hp": 70},
        {"ref": "action:a", "entity_type": "action", "verb": "SELECT_ONE"},
        {"ref": "action:b", "entity_type": "action", "verb": "SELECT_ONE"},
    ]
    if extra:
        entities.insert(4, {"ref": "card", "entity_type": "card", "cost": 2})
    refs = {entity["ref"]: index for index, entity in enumerate(entities) if "ref" in entity}
    direct = [(refs["node:a"], refs["node:b"], "map_edge")]
    floors = {refs["node:a"]: 0, refs["node:b"]: 2}
    edges = direct + map_relations(floors, direct)
    edges.append((refs["action:a"], refs["player"], "source"))
    programs = []
    for entity in entities:
        if entity.get("ref") == "action:a":
            program = {"kind": "sequence", "steps": [
                {"kind": "entity", "ref": "player"},
                {"kind": "literal", "value": 3},
            ]}
        elif entity.get("ref") == "action:b":
            program = {"kind": "sequence", "steps": [
                {"kind": "literal", "value": 3},
                {"kind": "entity", "ref": "player"},
            ]}
        else:
            program = None
        programs.append(effect_tree(program))
    actions = [refs["action:a"], refs["action:b"]]
    return Observation([fields_of(entity) for entity in entities], programs, refs, actions,
                       ["action:a", "action:b"], edges, floors,
                       {"order_matters": True, "selected_refs": []}, "test")


def vocabulary_for(*observations):
    symbols = []
    for obs in observations:
        for row in obs.tokens:
            symbols.extend(field.symbol for field in row)
        for effect in obs.effects:
            for row in effect.nodes:
                symbols.extend(field.symbol for field in row)
    return Vocabulary(symbols, ModelConfig.tiny().vocabulary_size)


def model(backend="reference"):
    config = ModelConfig.tiny()
    config.backend = backend
    return PolicyValue(config).eval()


def test_swiglu_is_used_by_global_and_local_transformers():
    network = model()
    assert all(isinstance(block.ffn, SwiGLU) for block in network.blocks)
    assert all(isinstance(block.ffn, SwiGLU) for block in network.encoder.local_blocks)
    assert isinstance(network.encoder.numeric[1], torch.nn.GELU)
    assert network.parameter_report()["full_layers"] == network.config.layers
    assert network.parameter_report()["linear_layers"] == 0


def test_compact_map_bias_has_map_shape_and_bias_gradients():
    obs = make_observation(extra=True)
    vocab = vocabulary_for(obs)
    network = model()
    metadata = network._map_bias([obs], len(obs.tokens), network.device)
    assert metadata.categories.shape == (1, 2, 2)
    assert metadata.node_slots.shape == (1, len(obs.tokens))
    assert (metadata.node_slots < 0).sum() == len(obs.tokens) - 2
    encoded = network.encode([obs], vocab)[0]
    encoded.hidden.square().sum().backward()
    attention = network.blocks[0].attention
    used_categories = metadata.categories.unique()
    assert attention.map_relation.weight.grad[used_categories].abs().sum() > 0
    assert attention.floor_delta.weight.grad.abs().sum() > 0

    # Validate the exact score-bias method against FlexAttention's backward
    # restrictions. EmbeddingBackward would fail this validator.
    from torch._dynamo._trace_wrapped_higher_order_op import TransformGetItemToIndex
    from torch._higher_order_ops.flex_attention import create_fw_bw_graph
    from torch._inductor.kernel.flex.flex_attention import validate_joint_graph

    def score_mod(score, batch, head, query, key, node_slots, categories, floors,
                  map_weight, floor_weight):
        compact = MapAttentionBias(node_slots, categories, floors)
        return score + attention._score_bias(compact, batch, head, query, key,
                                             map_weight, floor_weight)

    score = torch.randn((), requires_grad=True)
    indices = (score, *(torch.tensor(0, dtype=torch.int64) for _ in range(4)))
    buffers = (metadata.node_slots, metadata.categories, metadata.floors,
               attention.map_relation.weight, attention.floor_delta.weight)
    with TransformGetItemToIndex():
        _, backward = create_fw_bw_graph(score_mod, indices, buffers)
    validate_joint_graph(backward.graph)


def test_reference_and_sdpa_match_for_map_bias():
    obs = make_observation(extra=True)
    vocab = vocabulary_for(obs)
    reference = model("reference")
    sdpa = model("sdpa")
    sdpa.load_state_dict(reference.state_dict())
    with torch.no_grad():
        expected = reference.encode([obs], vocab)[0].hidden
        actual = sdpa.encode([obs], vocab)[0].hidden
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


def test_flash_cpu_fallback_preserves_map_bias_and_gradients():
    ModelConfig(backend="flash")
    obs = make_observation(extra=True)
    vocab = vocabulary_for(obs)
    network = model("flash")
    reference = model()
    reference.load_state_dict(network.state_dict())
    with pytest.warns(UserWarning, match="requires CUDA"):
        actual = network.encode([obs], vocab)[0].hidden
    expected = reference.encode([obs], vocab)[0].hidden
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    actual.square().sum().backward()
    for block in network.blocks:
        assert block.attention.actual_backend == "reference"
        assert block.attention.map_relation.weight.grad.abs().sum() > 0
        assert block.attention.floor_delta.weight.grad.abs().sum() > 0


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA FlashAttention-4")
def test_flash_cuda_inference_matches_reference_with_padding_and_map_bias():
    pytest.importorskip("flash_attn.cute")
    from model.attention import MapAttention

    attention = MapAttention(256, 4, backend="flash").cuda().eval()
    reference = MapAttention(256, 4, backend="reference").cuda().eval()
    reference.load_state_dict(attention.state_dict())
    x = torch.randn(2, 128, 256, device="cuda")
    valid = torch.arange(128, device="cuda")[None, :] < torch.tensor([117, 83], device="cuda")[:, None]
    slots = torch.full((2, 128), -1, device="cuda", dtype=torch.long)
    slots[:, 1:4] = torch.arange(3, device="cuda")
    metadata = MapAttentionBias(slots, torch.tensor([
        [[0, 1, 2], [3, 0, 4], [5, 6, 0]],
        [[0, 2, 3], [4, 0, 5], [6, 1, 0]],
    ], device="cuda"), torch.tensor([[0, 2, 4], [1, 3, 5]], device="cuda"))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        actual = attention(x, valid, metadata)
        expected = reference(x, valid, metadata)
        torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)
        assert attention.actual_backend == "flash"
        assert actual[~valid].count_nonzero() == 0
        # A cached compiled callable must read updated parameter values.
        attention.floor_delta.weight.add_(1.0)
        reference.floor_delta.weight.add_(1.0)
        updated = attention(x, valid, metadata)
        torch.testing.assert_close(updated, reference(x, valid, metadata), rtol=3e-2, atol=3e-2)
        assert not torch.equal(actual, updated)
    with pytest.raises(ValueError, match="FP16/BF16"), torch.no_grad():
        attention(x, valid, metadata)


def test_bf16_autocast_handles_vectorized_reference_and_binding_updates():
    obs = make_observation(extra=True)
    vocab = vocabulary_for(obs)
    network = model()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        encoded = network.encode([obs, obs], vocab)
        loss = sum(item.hidden.float().square().mean() for item in encoded)
    loss.backward()
    assert torch.isfinite(network.encoder.reference.weight.grad).all()
    assert torch.isfinite(network.encoder.program_binding.weight.grad).all()


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA FlexAttention")
@pytest.mark.parametrize("backend", ["flex", "flash"])
def test_flex_cuda_forward_backward_updates_compact_map_bias(backend):
    obs = make_observation(extra=True)
    vocab = vocabulary_for(obs)
    flex_config = ModelConfig(backend=backend)
    reference_config = ModelConfig(backend="reference")
    network = PolicyValue(flex_config).cuda().train()
    reference = PolicyValue(reference_config).cuda().train()
    reference.load_state_dict(network.state_dict())
    probe = torch.randn((len(obs.tokens), flex_config.hidden_size), device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        flex_encoded = network.encode([obs], vocab)[0]
        reference_encoded = reference.encode([obs], vocab)[0]
        flex_loss = (flex_encoded.hidden.float() * probe).sum()
        reference_loss = (reference_encoded.hidden.float() * probe).sum()
    torch.testing.assert_close(flex_encoded.hidden, reference_encoded.hidden, rtol=3e-2, atol=3e-2)
    flex_loss.backward()
    reference_loss.backward()
    for flex_block, reference_block in zip(network.blocks, reference.blocks):
        attention = flex_block.attention
        expected = reference_block.attention
        assert attention.actual_backend == "flex"
        assert torch.isfinite(attention.map_relation.weight.grad).all()
        assert torch.isfinite(attention.floor_delta.weight.grad).all()
        assert attention.map_relation.weight.grad.abs().sum() > 0
        assert attention.floor_delta.weight.grad.abs().sum() > 0
        torch.testing.assert_close(attention.map_relation.weight.grad,
                                   expected.map_relation.weight.grad, rtol=0.15, atol=2e-2)
        torch.testing.assert_close(attention.floor_delta.weight.grad,
                                   expected.floor_delta.weight.grad, rtol=0.15, atol=2e-2)


def test_batched_encode_decode_matches_independent_output_and_gradients():
    first, second = make_observation(), make_observation(extra=True)
    vocab = vocabulary_for(first, second)
    batched = model()
    independent = deepcopy(batched)

    batched_encoded = batched.encode([first, second], vocab)
    separate_encoded = [independent.encode([obs], vocab)[0] for obs in (first, second)]
    for together, alone in zip(batched_encoded, separate_encoded):
        torch.testing.assert_close(together.hidden, alone.hidden, rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(together.local, alone.local, rtol=2e-5, atol=2e-5)

    masks = [torch.tensor([True, True]), torch.tensor([True, True])]
    contexts = [first.context, second.context]
    batch_output = batched.decode_batch(batched_encoded, masks, contexts, vocab,
                                        hidden=[None, None], previous=[None, 0], value=[True, False])
    single_output = [independent.decode(encoded, mask, context, vocab,
                                        previous=previous, value=want_value)
                     for encoded, mask, context, previous, want_value in
                     zip(separate_encoded, masks, contexts, [None, 0], [True, False])]
    for together, alone in zip(batch_output, single_output):
        torch.testing.assert_close(together.logits, alone.logits, rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(together.hidden, alone.hidden, rtol=2e-5, atol=2e-5)
        assert (together.value is None) == (alone.value is None)

    sum(output.logits.logsumexp(0) + output.hidden.square().mean()
        + (output.value if output.value is not None else 0) for output in batch_output).backward()
    sum(output.logits.logsumexp(0) + output.hidden.square().mean()
        + (output.value if output.value is not None else 0) for output in single_output).backward()
    for name in ("query.weight", "gru.weight_ih", "encoder.reference.weight",
                 "encoder.program_binding.weight", "blocks.0.ffn.input.weight"):
        torch.testing.assert_close(dict(batched.named_parameters())[name].grad,
                                   dict(independent.named_parameters())[name].grad,
                                   rtol=4e-4, atol=4e-5)


def test_entity_permutation_preserves_aligned_outputs_and_local_program_order():
    obs = make_observation(extra=True)
    order = [0, 3, 2, 5, 1, 6, 4]
    inverse = {old: new for new, old in enumerate(order)}
    permuted = Observation(
        [obs.tokens[index] for index in order],
        [obs.effects[index] for index in order],
        {ref: inverse[index] for ref, index in obs.refs.items()},
        [inverse[index] for index in obs.action_indices],
        obs.slot_refs,
        [(inverse[source], inverse[target], role) for source, target, role in obs.edges],
        {inverse[index]: floor for index, floor in obs.map_floors.items()},
        obs.context,
        "permuted",
    )
    vocab = vocabulary_for(obs)
    network = model()
    with torch.no_grad():
        original = network.encode([obs], vocab)[0]
        changed = network.encode([permuted], vocab)[0]
    restored = torch.stack([changed.hidden[inverse[index]] for index in range(len(order))])
    torch.testing.assert_close(restored, original.hidden, rtol=2e-5, atol=2e-5)
    # The two actions have the same scalar literal but a different program order.
    assert not torch.allclose(original.local[obs.action_indices[0]],
                              original.local[obs.action_indices[1]])


def test_vocabulary_and_effect_cache_freeze_only_static_inputs():
    vocab = Vocabulary(["b", "a"])
    assert isinstance(vocab.symbols, tuple)
    assert vocab.digest == vocab.digest
    program = {"kind": "literal", "value": 4}
    assert effect_tree(program) is effect_tree({"value": 4, "kind": "literal"})
