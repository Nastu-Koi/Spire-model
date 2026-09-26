"""Atomic directory checkpoints with model, optimizer, and RNG state."""
from dataclasses import asdict
import ctypes
import json
import os
from pathlib import Path
import random
import shutil
import sys
import uuid

import torch

from .config import ModelConfig, TrainConfig
from .model import PolicyValue
from .representation import Vocabulary



def save_checkpoint(path, model, vocabulary, optimizer=None, scheduler=None, *, training=None, progress=None, overwrite=False):
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError("Use a new checkpoint directory; existing checkpoints are immutable")
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = path.with_name(path.name + ".staging-" + uuid.uuid4().hex)
    stage.mkdir()
    try:
        manifest = {"format": 2, "model": asdict(model.config), "vocabulary": vocabulary.symbols,
                    "training": asdict(training or TrainConfig()), "progress": progress or {},
                    "torch_version": str(torch.__version__)}
        torch.save(model.state_dict(), stage / "weights.pt")
        manifest["weights"] = ["weights.pt"]
        if optimizer:
            torch.save(optimizer.state_dict(), stage / "optimizer.pt")
            manifest["optimizer"] = "optimizer.pt"
        torch.save({"torch": torch.get_rng_state(), "python": random.getstate(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                    "scheduler": scheduler.state_dict() if scheduler else None}, stage / "runtime.pt")
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2))
        if path.exists():
            # Linux renameat2 exchanges two directories atomically: readers always
            # see a complete current checkpoint, even if the process is killed.
            libc = ctypes.CDLL(None, use_errno=True)
            if sys.platform == "darwin":
                exchange = libc.renamex_np
                exchange.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
                result = exchange(os.fsencode(stage), os.fsencode(path), 2)  # RENAME_SWAP
            else:
                exchange = libc.renameat2
                exchange.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
                result = exchange(-100, os.fsencode(stage), -100, os.fsencode(path), 2)
            if result != 0:
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
    if manifest.get("format") != 2 or manifest.get("model", {}).get("architecture_version") != 2:
        raise ValueError("Checkpoint requires architecture v2 (Full Attention/SwiGLU); retrain old hybrid checkpoints")
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
    if manifest.get("format") != 2 or "optimizer" not in manifest:
        raise ValueError("Checkpoint has no optimizer state")
    if isinstance(manifest["optimizer"], str):
        restored = torch.load(path / manifest["optimizer"], map_location="cpu", weights_only=True)
    else:
        # Read checkpoints written before optimizer state was saved in one file.
        restored = {}
        for kind, part in manifest["optimizer"].items():
            state = {}
            for name in part["shards"]:
                shard = torch.load(path / name, map_location="cpu", weights_only=True)
                if state.keys() & shard.keys():
                    raise ValueError("Duplicate optimizer checkpoint tensors")
                state.update(shard)
            restored[kind] = {"state": state, "param_groups": part["groups"]}
        if set(restored) == {"single"}:
            restored = restored["single"]
    # Delegate dtype/device restoration to the actual optimizer, including Muon.
    optimizer.load_state_dict(restored)
    runtime = torch.load(path / "runtime.pt", map_location="cpu", weights_only=True)
    torch.set_rng_state(runtime["torch"])
    random.setstate(runtime["python"])
    if runtime["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(runtime["cuda"])
    if scheduler and runtime["scheduler"] is not None:
        scheduler.load_state_dict(runtime["scheduler"])
    return manifest["progress"]
