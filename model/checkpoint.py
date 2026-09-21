"""Atomic directory checkpoints with bounded tensor staging and optimizer/RNG state."""
from dataclasses import asdict
import ctypes
import json
import os
from pathlib import Path
import random
import shutil
import uuid

import torch

from .config import ModelConfig, TrainConfig
from .model import PolicyValue
from .representation import Vocabulary


def _bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_bytes(x) for x in value.values())
    return 0


def _shards(items, directory, prefix, limit=64 * 1024**2):
    names, shard, size = [], {}, 0
    for key, value in items:
        needed = _bytes(value)
        if shard and size + needed > limit:
            name = f"{prefix}-{len(names):05}.pt"
            torch.save(shard, directory / name)
            names.append(name)
            shard, size = {}, 0
        shard[key], size = value, size + needed
    if shard:
        name = f"{prefix}-{len(names):05}.pt"
        torch.save(shard, directory / name)
        names.append(name)
    return names


def save_checkpoint(path, model, vocabulary, optimizer=None, scheduler=None, *, training=None, progress=None, overwrite=False):
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError("Use a new checkpoint directory; existing checkpoints are immutable")
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = path.with_name(path.name + ".staging-" + uuid.uuid4().hex)
    stage.mkdir()
    try:
        manifest = {"format": 1, "model": asdict(model.config), "vocabulary": vocabulary.symbols,
                    "training": asdict(training or TrainConfig()), "progress": progress or {},
                    "torch_version": str(torch.__version__)}
        manifest["weights"] = _shards(model.state_dict().items(), stage, "weights")
        if optimizer:
            state = optimizer.state_dict()
            manifest["optimizer_groups"] = state["param_groups"]
            manifest["optimizer"] = _shards(state["state"].items(), stage, "optimizer")
        torch.save({"torch": torch.get_rng_state(), "python": random.getstate(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                    "scheduler": scheduler.state_dict() if scheduler else None}, stage / "runtime.pt")
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2))
        if path.exists():
            # Linux renameat2 exchanges two directories atomically: readers always
            # see a complete current checkpoint, even if the process is killed.
            libc = ctypes.CDLL(None, use_errno=True)
            exchange = libc.renameat2
            exchange.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
            if exchange(-100, os.fsencode(stage), -100, os.fsencode(path), 2) != 0:
                code = ctypes.get_errno()
                raise OSError(code, os.strerror(code), str(path))
            shutil.rmtree(stage)
        else:
            stage.rename(path)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def load_model(path, device="cpu"):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("format") != 1:
        raise ValueError("Unsupported checkpoint format")
    config = ModelConfig(**manifest["model"])
    # Allocate once on the destination without a second full CPU model.
    with torch.device("meta"):
        model = PolicyValue(config)
    model.to_empty(device=device)
    expected = set(model.state_dict())
    loaded = set()
    for name in manifest["weights"]:
        shard = torch.load(path / name, map_location=device, weights_only=True)
        if loaded & shard.keys() or not shard.keys() <= expected:
            raise ValueError("Duplicate or unexpected checkpoint tensors")
        model.load_state_dict(shard, strict=False)
        loaded.update(shard)
        del shard
    if loaded != expected:
        raise ValueError("Checkpoint is missing model tensors")
    vocabulary = Vocabulary(manifest["vocabulary"], config.vocabulary_size)
    return model, vocabulary, manifest


def restore_training(path, optimizer, scheduler=None):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if "optimizer" not in manifest:
        raise ValueError("Checkpoint has no optimizer state")
    # Load each parameter's Adam state without accumulating a duplicate full state_dict.
    groups = manifest["optimizer_groups"]
    current = optimizer.param_groups
    if len(groups) != len(current) or any(len(g["params"]) != len(c["params"]) for g, c in zip(groups, current)):
        raise ValueError("Optimizer parameter groups differ")
    mapping = {index: p for group, now in zip(groups, current) for index, p in zip(group["params"], now["params"])}
    for old, now in zip(groups, current):
        now.update({k: v for k, v in old.items() if k != "params"})
    optimizer.state.clear()
    for name in manifest["optimizer"]:
        shard = torch.load(path / name, map_location="cpu", weights_only=True)
        for index, state in shard.items():
            parameter = mapping[index]
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    use_device = key != "step" or optimizer.defaults.get("fused") or optimizer.defaults.get("capturable")
                    state[key] = value.to(parameter.device if use_device else "cpu")
            optimizer.state[parameter] = state
        del shard
    runtime = torch.load(path / "runtime.pt", map_location="cpu", weights_only=True)
    torch.set_rng_state(runtime["torch"])
    random.setstate(runtime["python"])
    if runtime["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(runtime["cuda"])
    if scheduler and runtime["scheduler"] is not None:
        scheduler.load_state_dict(runtime["scheduler"])
    return manifest["progress"]
