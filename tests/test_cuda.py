from copy import deepcopy
import os

import pytest
import torch

from model.config import ModelConfig, TrainConfig
from model.model import PolicyValue
from model.policy import replay
from model.representation import Vocabulary, symbols_from_frames
from model.rollout import precision_context
from model.runtime import configure_runtime
from model.testing import demonstration
from model.trainer import build_optimizer


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_bf16_fused_optimizer_and_flex_gradient():
    configure_runtime("cuda", threads=2)
    demo = demonstration(count=4, select=2)
    cfg = ModelConfig.tiny()
    vocab = Vocabulary(symbols_from_frames([s["frame"] for m in demo["macros"] for s in m["steps"]]), cfg.vocabulary_size)
    model = PolicyValue(cfg).cuda()
    fused = deepcopy(model)
    for block in fused.blocks:
        if not block.attention.linear:
            block.attention.backend = "flex"
    steps = demo["macros"][0]["steps"]
    with precision_context(model, "bf16"):
        reference, value, _ = replay(model, vocab, steps)
        actual, flex_value, _ = replay(fused, vocab, steps)
    torch.testing.assert_close(reference, actual, atol=.02, rtol=.02)
    (reference + value.square()).backward()
    (actual + flex_value.square()).backward()
    assert all(b.attention.actual_backend == "flex" for b in fused.blocks if not b.attention.linear)
    torch.testing.assert_close(model.blocks[0].relation.weight.grad, fused.blocks[0].relation.weight.grad,
                               atol=.01, rtol=.1)
    optimizer = build_optimizer(fused, TrainConfig())
    optimizer.step()
    assert optimizer.defaults["fused"]
    assert all(p.grad is None or p.grad.dtype == torch.float32 for p in fused.parameters())


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available() or os.environ.get("SPIRE_TEST_1B") != "1", reason="Explicit 1B hardware test required")
def test_one_billion_cuda_training_step():
    configure_runtime("cuda", threads=2)
    demo = demonstration(count=4, select=2)
    config = ModelConfig()
    vocab = Vocabulary(symbols_from_frames([s["frame"] for m in demo["macros"] for s in m["steps"]]), config.vocabulary_size)
    with torch.device("cuda"):
        model = PolicyValue(config)
    optimizer = build_optimizer(model, TrainConfig())
    with precision_context(model, "bf16"):
        lp, value, entropy = replay(model, vocab, demo["macros"][0]["steps"])
        loss = -lp + value.square() - .001 * entropy
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    torch.cuda.synchronize()
    print({"parameters": model.parameter_report(), "loss": float(loss.detach()),
           "backend": model.blocks[0].attention.actual_backend,
           "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved()})
    assert torch.isfinite(loss)
