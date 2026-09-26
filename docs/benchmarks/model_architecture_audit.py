"""Disposable CPU architecture audit; no checkpoint reads, updates, or writes.

Run from the repository root. Uses a fixed sample of the first accepted run.
Timings are diagnostic CPU measurements, not GPU throughput predictions.
"""

import cProfile
import io
import json
import pstats
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch

from model.config import ModelConfig, TrainConfig
from model.data import Sample, batch_pad_size, microbatches, sample_pad_size
from model.model import PolicyValue
from model.policy import replay_batch
from model.representation import Vocabulary, observation, symbols_from_frames
from model.rollout import precision_context

random.seed(0)
torch.manual_seed(0)
torch.set_num_threads(4)
cfg = json.load(open("configs/rtxpro6000.json"))
tc = TrainConfig.from_dict(cfg["training"])
run = json.loads(next(open("data/steam-001/accepted.jsonl")))
macros = [m for m in run["macros"] if any(not s["forced"] for s in m["steps"])]
items = [Sample(m, run["character"], "audit", 1.0) for m in macros]
random.shuffle(items)
items = items[: min(128, len(items))]
obs = [
    observation(next(s["frame"] for s in x.macro["steps"] if not s["forced"]))
    for x in items
]
lengths = [len(o.tokens) for o in obs]
print(
    "sample",
    json.dumps(
        {
            "macros": len(items),
            "tokens_min": min(lengths),
            "tokens_median": statistics.median(lengths),
            "tokens_max": max(lengths),
        }
    ),
    flush=True,
)


def costs(ordered):
    total = 0
    actual = 0
    groups = 0
    for start in range(0, len(ordered), 32):
        for mb in microbatches(ordered[start : start + 32], tc):
            groups += 1
            total += len(mb) * batch_pad_size(mb, tc) ** 2
            actual += sum(max(n for n, a in x._lengths) ** 2 for x in mb)
    return {"padded_pairs": total, "individual_actual_pairs": actual, "groups": groups}


base = costs(items)
ordered = []
for start in range(0, len(items), 32):
    ordered += sorted(items[start : start + 32], key=lambda x: sample_pad_size(x, tc))
print(
    "padding",
    json.dumps({"random": base, "within_logical_sorted": costs(ordered)}),
    flush=True,
)
# Fixed replay batch spans early sampled lengths; no trained weights or updates.
batch = items[:4]
seq = [x.macro["steps"] for x in batch]
frames = [s["frame"] for m in seq for s in m]
v = Vocabulary(symbols_from_frames(frames), cfg["model"]["vocabulary_size"])
model = PolicyValue(ModelConfig(**cfg["model"]))
batchobs = [
    observation(next(s["frame"] for s in x.macro["steps"] if not s["forced"]))
    for x in batch
]
effects = [e for o in batchobs for e in o.effects]
lens = [len(e.nodes) for e in effects]
print(
    "batch",
    json.dumps(
        {
            "tokens": [len(o.tokens) for o in batchobs],
            "macro_steps": list(map(len, seq)),
            "bucket": batch_pad_size(batch, tc),
            "effects": len(lens),
            "effect_nodes": sum(lens),
            "max_program": max(lens),
            "padded_local_pairs": len(lens) * max(lens) ** 2,
            "actual_local_pairs": sum(n * n for n in lens),
            "params": model.parameter_report(),
        }
    ),
    flush=True,
)


def step(pad, capture=True):
    model.zero_grad(set_to_none=True)
    with precision_context(model, tc.precision):
        outs = replay_batch(model, v, seq, pad_to=pad)
        loss = torch.stack(
            [-lp + 0.5 * value.square() - 0.001 * ent for lp, value, ent in outs]
        ).mean()
    loss.backward()
    assert bool(torch.isfinite(loss))
    if not capture:
        return
    return torch.stack([torch.stack(x) for x in outs]).detach(), {
        n: p.grad.detach().clone()
        for n, p in model.named_parameters()
        if p.grad is not None
    }


results = {}
saved = {}
for name, pad in [("bucket", batch_pad_size(batch, tc)), ("exact", None)]:
    step(pad, capture=False)
    times = []
    for _ in range(5):
        t = time.perf_counter()
        step(pad, capture=False)
        times.append(time.perf_counter() - t)
    results[name] = {"median_s": statistics.median(times), "times": times}
    saved[name] = step(pad)
print("timings", json.dumps(results), flush=True)
print(
    "difference",
    json.dumps(
        {
            "output_max_abs": (saved["bucket"][0] - saved["exact"][0])
            .abs()
            .max()
            .item(),
            "gradient_max_abs": max(
                (g - saved["exact"][1][n]).abs().max().item()
                for n, g in saved["bucket"][1].items()
            ),
        }
    ),
    flush=True,
)
p = cProfile.Profile()
p.enable()
step(batch_pad_size(batch, tc), capture=False)
p.disable()
s = io.StringIO()
pstats.Stats(p, stream=s).strip_dirs().sort_stats("cumtime").print_stats(25)
print(s.getvalue(), flush=True)
from dataclasses import replace

fine = replace(
    tc, token_buckets=[64, 96, 128, 160, 192, 256, 384, 512, 1024, 2048, 4096]
)


def capacity(ordered, conf):
    return sum(
        len(mb) * batch_pad_size(mb, conf) ** 2
        for start in range(0, len(ordered), 32)
        for mb in microbatches(ordered[start : start + 32], conf)
    )


ordered = []
for start in range(0, len(items), 32):
    ordered += sorted(items[start : start + 32], key=lambda x: sample_pad_size(x, fine))
print(
    "fine_buckets",
    json.dumps(
        {
            "current": capacity(items, tc),
            "fine_random": capacity(items, fine),
            "fine_sorted_within_logical": capacity(ordered, fine),
        }
    ),
    flush=True,
)
# FP32 output/gradient differential, independent of BF16 rounding.
tc.precision = "no"
a, ga = step(batch_pad_size(batch, tc))
b, gb = step(None)
err = (a - b).abs().max().item()
ge = max((g - gb[n]).abs().max().item() for n, g in ga.items())
assert torch.allclose(a, b, atol=1e-5, rtol=1e-5), (a, b)
assert all(torch.allclose(g, gb[n], atol=1e-5, rtol=1e-4) for n, g in ga.items()), ge
print(
    "fp32_padding_equivalence",
    json.dumps({"output_max_abs": err, "gradient_max_abs": ge, "verdict": "PASS"}),
    flush=True,
)
