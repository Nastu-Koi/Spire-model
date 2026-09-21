from copy import deepcopy
import math

import pytest
import torch
from torch.nn import functional as F

from model.config import ModelConfig
from model.model import PolicyValue, linear_attention
from model.policy import SessionPolicy, replay
from model.protocol import execution_command
from model.representation import Vocabulary, observation, symbols_from_frames
from model.testing import SyntheticEngine, demonstration


def test_one_billion_schedule_and_gru_budget():
    with torch.device("meta"):
        model = PolicyValue(ModelConfig())
    report = model.parameter_report()
    assert 1_000_000_000 <= report["total"] <= 1_010_000_000
    assert report["gru"] == 19_278_336
    assert [b.attention.linear for b in model.blocks] == [False, True] * 12 + [False]


def test_linear_kernel_matches_explicit_kernel_with_padding():
    q, k, v = [torch.randn(2, 3, 7, 4, requires_grad=True) for _ in range(3)]
    valid = torch.tensor([[True] * 5 + [False] * 2, [True] * 7])
    actual = linear_attention(q, k, v, valid)
    scores = (F.elu(q) + 1) @ (F.elu(k) + 1).transpose(-1, -2)
    scores = scores * valid[:, None, None, :]
    expected = scores @ v / scores.sum(-1, keepdim=True).clamp_min(1e-6)
    expected *= valid[:, None, :, None]
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    actual.square().sum().backward()
    assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in (q, k, v))
    assert k.grad[0, :, 5:].count_nonzero() == 0


def test_static_ten_choose_five_cache_and_replay():
    demo = demonstration()
    frames = [s["frame"] for m in demo["macros"] for s in m["steps"]]
    cfg = ModelConfig.tiny()
    vocab = Vocabulary(symbols_from_frames(frames), cfg.vocabulary_size)
    model = PolicyValue(cfg)
    policy = SessionPolicy(model, vocab)
    old = []
    with torch.no_grad():
        for step in demo["macros"][0]["steps"]:
            out = policy.choose(step["frame"], teacher=step["candidate_ref"])
            if out.log_prob is not None:
                old.append(out.log_prob)
    assert model.encoder_calls == 1
    assert model.decoder_calls == 5
    new, value, _ = replay(model, vocab, demo["macros"][0]["steps"])
    torch.testing.assert_close(new, torch.stack(old).sum())
    assert abs(float((new - torch.stack(old).sum()).exp().detach()) - 1) < 1e-6
    (-new + value.square()).backward()
    for parameter in (model.gru.weight_hh, model.blocks[0].attention.qkv.weight, model.encoder.symbol.weight,
                      model.encoder.local_blocks[0].attention.qkv.weight, model.blocks[0].relation.weight):
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0


def test_forced_step_bypasses_all_model_heads(setup):
    model, vocab = setup
    engine = SyntheticEngine(5, 3, forced_first=True)
    frame = engine.reset()
    policy = SessionPolicy(model, vocab)
    choice = policy.choose(frame)
    assert choice.log_prob is choice.value is None
    assert model.encoder_calls == model.decoder_calls == 0
    next_frame = engine.send(execution_command(frame, choice.candidate_ref))
    policy.choose(next_frame)
    assert model.encoder_calls == model.decoder_calls == 1
    assert policy.previous_slot is not None


def test_padded_batched_encoding_and_permutation(setup, demo):
    model, vocab = setup
    model.eval()
    frame = demo["macros"][0]["steps"][0]["frame"]
    obs = observation(frame)
    other = observation(demo["macros"][1]["steps"][0]["frame"])
    with torch.no_grad():
        alone = model.encode([obs], vocab)[0]
        padded = model.encode([obs, other], vocab, pad_to=32)[0]
    torch.testing.assert_close(alone.hidden, padded.hidden, rtol=1e-5, atol=2e-6)
    changed = deepcopy(frame)
    changed["public"]["entities"].reverse()
    changed["public"]["decoder_bank"].reverse()
    changed["legal"]["candidates"].reverse()
    with torch.no_grad():
        a = SessionPolicy(model, vocab).choose(frame, sample=False, top_k=100)
        b = SessionPolicy(model, vocab).choose(changed, sample=False, top_k=100)
    probs_a = {x["candidate_ref"]: x["probability"] for x in a.ranking}
    probs_b = {x["candidate_ref"]: x["probability"] for x in b.ranking}
    assert probs_a.keys() == probs_b.keys()
    for key in probs_a:
        assert probs_a[key] == pytest.approx(probs_b[key], abs=1e-6)
    torch.testing.assert_close(a.value, b.value, atol=2e-6, rtol=1e-5)


def test_mask_stop_padding_and_topk(setup, demo):
    model, vocab = setup
    obs = observation(demo["macros"][0]["steps"][0]["frame"])
    encoded = model.encode([obs], vocab)[0]
    mask = torch.tensor([True, False, True, False, False, False])
    out = model.decode(encoded, mask, obs.context, vocab)
    assert out.distribution().probs[~mask].count_nonzero() == 0
    assert set(out.topk(20)[0].tolist()) == {0, 2}
    with pytest.raises(ValueError, match="Forced"):
        model.decode(encoded, torch.tensor([True] + [False]*5), obs.context, vocab)


def test_feedback_changes_next_step_and_cache_invalidation(setup, demo):
    model, vocab = setup
    steps = demo["macros"][0]["steps"]
    encoded = model.encode([observation(steps[0]["frame"])], vocab)[0]
    mask = torch.tensor([True]*5 + [False])
    a = model.decode(encoded, mask, {}, vocab, previous=0)
    b = model.decode(encoded, mask, {}, vocab, previous=1)
    assert not torch.allclose(a.logits[mask], b.logits[mask])
    policy = SessionPolicy(model, vocab)
    policy.choose(steps[0]["frame"], teacher=steps[0]["candidate_ref"])
    old_hidden = policy.hidden
    changed = deepcopy(steps[1]["frame"])
    changed["public"]["entities"][1]["cost"] = 9
    before = model.encoder_calls
    policy.choose(changed, teacher=steps[1]["candidate_ref"])
    assert model.encoder_calls == before + 1
    assert policy.hidden.grad_fn is not None and old_hidden.grad_fn is not None
    policy.choose(demo["macros"][1]["steps"][0]["frame"], sample=False)
    assert policy.segment != ("Ironclad:demo", "Ironclad:demo:selection")


def test_known_unknown_and_inapplicable_have_distinct_features():
    from model.representation import fields_of
    zero = fields_of({"hp": 0})[0]
    unknown = fields_of({"hp": {"value": 100, "known": False}})[0]
    absent = fields_of({"hp": {"value": 100, "applicable": False}})[0]
    assert len({zero.number, unknown.number, absent.number}) == 3


def test_sdpa_matches_reference_with_relation_gradients(setup, demo):
    model, vocab = setup
    other = deepcopy(model)
    for block in other.blocks:
        if not block.attention.linear:
            block.attention.backend = "sdpa"
    steps = demo["macros"][0]["steps"]
    a, _, _ = replay(model, vocab, steps)
    b, _, _ = replay(other, vocab, steps)
    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
    a.backward()
    b.backward()
    torch.testing.assert_close(model.blocks[0].relation.weight.grad, other.blocks[0].relation.weight.grad, atol=2e-6, rtol=1e-4)


def test_bf16_forward_backward(setup, demo):
    from model.rollout import precision_context
    model, vocab = setup
    with precision_context(model, "bf16"):
        lp, value, entropy = replay(model, vocab, demo["macros"][0]["steps"])
    (-lp + value.square()).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_activation_checkpointing_preserves_macro_gradients(setup, demo):
    model, vocab = setup
    other = deepcopy(model)
    other.config.checkpoint_layers = True
    steps = demo["macros"][0]["steps"]
    a, _, _ = replay(model, vocab, steps)
    b, _, _ = replay(other, vocab, steps)
    a.backward()
    b.backward()
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    torch.testing.assert_close(model.gru.weight_hh.grad, other.gru.weight_hh.grad, atol=0, rtol=0)
    torch.testing.assert_close(model.blocks[0].attention.qkv.weight.grad, other.blocks[0].attention.qkv.weight.grad, atol=0, rtol=0)
def test_schema_categories_are_available_before_first_combat():
    from model.representation import Vocabulary, symbols_from_frames
    vocab = Vocabulary(symbols_from_frames([]))
    for prefix, values in {"phase": ["combat", "shop", "map"],
                           "entity_type": ["enemy", "orb", "board_cell"],
                           "zone": ["hand", "draw_pile", "exhaust_pile"]}.items():
        ids = [vocab.encode(f"{prefix}={value}") for value in values]
        assert 1 not in ids and len(set(ids)) == len(values)

