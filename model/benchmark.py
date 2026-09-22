"""Capacity inspection and reproducible replay benchmarks, without policy updates."""
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from .config import ModelConfig, TrainConfig
from .data import Sample, batch_pad_size
from .model import PolicyValue
from .policy import replay_batch
from .protocol import CHARACTERS
from .representation import Vocabulary, observation, symbols_from_frames
from .rollout import precision_context
from .testing import demonstration


def iter_runs(path, max_runs=32):
    """Inspect trajectory files, excluding journals and seed metadata in directories."""
    if max_runs < 1:
        raise ValueError("max_runs must be positive")
    path = Path(path)
    files = sorted(path.rglob('*.json')) if path.is_dir() else [path]
    count = 0
    for file in files:
        if path.is_dir() and 'metadata' in file.relative_to(path).parts:
            continue
        if file.suffix == '.jsonl':
            with file.open() as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    run = json.loads(line)
                    if isinstance(run, dict) and 'macros' in run:
                        yield run
                        count += 1
                        if count >= max_runs:
                            return
        else:
            raw = json.loads(file.read_text())
            for run in raw if isinstance(raw, list) else [raw]:
                if not isinstance(run, dict) or 'macros' not in run:
                    continue
                yield run
                count += 1
                if count >= max_runs:
                    return


def capacity_report(path, *, max_runs=32):
    groups = defaultdict(lambda: defaultdict(list))
    run_count = 0
    for run in iter_runs(path, max_runs):
        run_count += 1
        for macro in run['macros']:
            for step in macro['steps']:
                if step.get('forced'):
                    continue
                obs = observation(step['frame'])
                values = {
                    'tokens': len(obs.tokens), 'actions': len(obs.slot_refs),
                    'effect_nodes': sum(len(e.nodes) for e in obs.effects),
                    'max_program_length': max((len(e.nodes) for e in obs.effects), default=0),
                    'effect_edges': sum(len(e.edges) for e in obs.effects),
                    'map_nodes': len(obs.map_floors),
                }
                for group in ('all', f"{run['character']}/{macro['phase']}"):
                    for key, value in values.items():
                        groups[group][key].append(value)
    if not run_count:
        raise ValueError('No trajectory runs found')
    return {'runs': run_count, 'groups': {
        group: {key: {'count': len(values), 'max': max(values),
                      **dict(zip(('p50', 'p90', 'p95', 'p99'), np.percentile(values, [50, 90, 95, 99]).tolist()))}
                for key, values in fields.items()} for group, fields in groups.items()}}


def benchmark(config_path, *, device='cpu', data=None, batch_size=4, steps=10, warmup=2, trace=None,
              optimizer_steps=False):
    if batch_size < 1 or steps < 1 or warmup < 0:
        raise ValueError('Positive batch/steps and nonnegative warmup required')
    profile = json.loads(Path(config_path).read_text())
    config = ModelConfig(**profile['model'])
    training = TrainConfig.from_dict(profile['training'])
    runs = list(iter_runs(data)) if data else [demonstration(c, f'benchmark-{c}', 8, 3) for c in CHARACTERS]
    macros = [m['steps'] for r in runs for m in r['macros'] if any(not s['forced'] for s in m['steps'])]
    if not macros:
        raise ValueError('No non-forced macros to benchmark')
    batch = [macros[i % len(macros)] for i in range(batch_size)]
    pad_to = batch_pad_size([Sample({'steps': sequence}, '', '', 1.) for sequence in batch], training)
    vocabulary = Vocabulary(symbols_from_frames(s['frame'] for m in batch for s in m), config.vocabulary_size)
    with torch.device(device):
        model = PolicyValue(config)
    model.train()
    if optimizer_steps:
        from .optim import build_optimizer
        optimizer = build_optimizer(model, training)

    def sync():
        if model.device.type == 'cuda':
            torch.cuda.synchronize(model.device)

    def iteration():
        model.zero_grad(set_to_none=True)
        with precision_context(model, training.precision):
            outputs = replay_batch(model, vocabulary, batch, pad_to=pad_to)
            # Synthetic benchmark objective exercises both policy and value backward.
            loss = torch.stack([-lp + .5 * value.square() - .001 * entropy for lp, value, entropy in outputs]).mean()
        loss.backward()
        if optimizer_steps:
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), training.max_grad_norm)
            if not bool(torch.isfinite(norm)):
                raise FloatingPointError('Non-finite benchmark gradients')
            optimizer.step()
        return loss

    sync()
    started = time.perf_counter()
    loss = iteration()
    sync()
    first = time.perf_counter() - started
    for _ in range(warmup):
        loss = iteration()
    sync()
    if model.device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(model.device)
    durations = []
    for _ in range(steps):
        started = time.perf_counter()
        loss = iteration()
        sync()
        durations.append(time.perf_counter() - started)
    if not bool(torch.isfinite(loss)) or any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
        raise FloatingPointError('Non-finite benchmark loss or gradients')
    if trace:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if model.device.type == 'cuda':
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True) as profiler:
            iteration()
            sync()
        Path(trace).parent.mkdir(parents=True, exist_ok=True)
        profiler.export_chrome_trace(str(trace))
    return {
        'domain': 'trajectory_replay_benchmark' if data else 'synthetic_test_only',
        'objective': 'synthetic policy+value benchmark; no on-policy claims or checkpoint writes',
        'optimizer_steps': optimizer_steps, 'optimizer': training.optimizer if optimizer_steps else None,
        'model': asdict(config), 'precision': training.precision, 'parameters': model.parameter_report(),
        'batch_size': batch_size, 'steps': steps, 'warmup': warmup,
        'pad_to': pad_to,
        'first_step_seconds': first, 'mean_step_seconds': float(np.mean(durations)),
        'p95_step_seconds': float(np.percentile(durations, 95)),
        'macros_per_second': batch_size / float(np.mean(durations)),
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(model.device) if model.device.type == 'cuda' else None,
        'attention_backends': sorted({m.actual_backend for m in model.modules() if hasattr(m, 'actual_backend')}),
    }
