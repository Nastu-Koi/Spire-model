"""Atomic directory checkpoints with model, optimizer, and RNG state."""

import ctypes
import json
import os
import random
import shutil
import sys
import uuid
from dataclasses import asdict
from pathlib import Path

import torch

from . import rules
from .config import ModelConfig, TrainConfig
from .model import PolicyValue
from .representation import ENCODING_VERSION, Vocabulary
from .optim import TRAINING_VERSION
from .rewards import REWARD_VERSION
from .public_history import HISTORY_VERSION, require_history_version


def archive_checkpoint(source, destination):
    """Publish an immutable epoch snapshot without duplicating its tensor files."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError("Epoch checkpoint already exists; use a new output directory")
    stage = destination.with_name(destination.name + ".staging-" + uuid.uuid4().hex)
    try:
        shutil.copytree(source, stage, copy_function=os.link)
        stage.rename(destination)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def save_checkpoint(
    path,
    model,
    vocabulary,
    optimizer=None,
    scheduler=None,
    *,
    training=None,
    progress=None,
    overwrite=False,
):
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(
            "Use a new checkpoint directory; existing checkpoints are immutable"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = path.with_name(path.name + ".staging-" + uuid.uuid4().hex)
    stage.mkdir()
    try:
        manifest = {
            "format": 1,
            "encoding_version": ENCODING_VERSION,
            "training_version": TRAINING_VERSION,
            "public_history_version": HISTORY_VERSION,
            "model": asdict(model.config),
            "vocabulary": vocabulary.state(),
            "rules": rules.digest(),
            "training": asdict(training or TrainConfig()),
            "progress": {"reward_version": REWARD_VERSION, **(progress or {})},
            "torch_version": str(torch.__version__),
        }
        torch.save(model.state_dict(), stage / "weights.pt")
        manifest["weights"] = ["weights.pt"]
        if optimizer:
            torch.save(optimizer.state_dict(), stage / "optimizer.pt")
            manifest["optimizer"] = "optimizer.pt"
        torch.save(
            {
                "torch": torch.get_rng_state(),
                "python": random.getstate(),
                "cuda": torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else [],
                "scheduler": scheduler.state_dict() if scheduler else None,
                "training": asdict(training or TrainConfig()),
            },
            stage / "runtime.pt",
        )
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2))
        if path.exists():
            # Linux renameat2 exchanges two directories atomically: readers always
            # see a complete current checkpoint, even if the process is killed.
            libc = ctypes.CDLL(None, use_errno=True)
            if sys.platform == "darwin":
                exchange = libc.renamex_np
                exchange.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
                result = exchange(
                    os.fsencode(stage), os.fsencode(path), 2
                )  # RENAME_SWAP
            else:
                exchange = libc.renameat2
                exchange.argtypes = [
                    ctypes.c_int,
                    ctypes.c_char_p,
                    ctypes.c_int,
                    ctypes.c_char_p,
                    ctypes.c_uint,
                ]
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


def load_model(path, device="cpu", *, allow_legacy_encoding=False):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("format") != 1:
        raise ValueError("Unsupported checkpoint format")
    require_history_version(manifest.get("public_history_version"))
    if manifest.get("encoding_version") != ENCODING_VERSION and not allow_legacy_encoding:
        raise ValueError("Checkpoint input encoding changed; use bootstrap --weights-only to start a new run")
    if manifest.get("model", {}).get("architecture_version") != 2:
        raise ValueError("Unsupported checkpoint architecture version")
    if manifest.get("rules") != rules.digest():
        raise ValueError("Checkpoint was trained with different rule text")
    if "ascension" not in manifest.get("training", {}):
        raise ValueError("Checkpoint has no declared ascension; do not infer a new default")
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
    vocabulary = Vocabulary.from_state(manifest["vocabulary"], config)
    return model, vocabulary, manifest


def restore_training(path, optimizer, scheduler=None):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("format") != 1 or "optimizer" not in manifest:
        raise ValueError("Checkpoint has no optimizer state")
    if manifest.get("training_version") != TRAINING_VERSION:
        raise ValueError("Checkpoint optimizer/loss version changed; start with --weights-only")
    if manifest.get("encoding_version") != ENCODING_VERSION:
        raise ValueError("Checkpoint input encoding changed; start with --weights-only")
    require_history_version(manifest.get("public_history_version"))
    runtime = torch.load(path / "runtime.pt", map_location="cpu", weights_only=True)
    if runtime.get("training") != manifest["training"]:
        raise ValueError("Checkpoint training configuration conflicts with saved runtime; use --weights-only")
    restored = torch.load(path / manifest["optimizer"], map_location="cpu", weights_only=True)
    # A strict resume restores its original groups. It must never swallow requested settings.
    groups = (restored["adamw"]["param_groups"] + restored["muon"]["param_groups"]
              if set(restored) == {"adamw", "muon"} else restored["param_groups"])
    current = optimizer.param_groups
    keys = ("name", "initial_lr", "weight_decay", "betas", "eps", "momentum", "ns_steps")
    if len(groups) != len(current) or any(
        len(old["params"]) != len(new["params"]) or any(old.get(k) != new.get(k) for k in keys)
        for old, new in zip(groups, current)
    ):
        raise ValueError("Optimizer configuration differs from checkpoint; use --weights-only")
    if scheduler:
        # Validate the schedule before changing any optimizer or RNG state.
        if runtime["scheduler"] is None:
            raise ValueError("Checkpoint has no schedule state; use --weights-only")
        scheduler.check(runtime["scheduler"])
    # Delegate dtype/device restoration to the actual optimizer, including Muon.
    optimizer.load_state_dict(restored)
    torch.set_rng_state(runtime["torch"])
    random.setstate(runtime["python"])
    if runtime["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(runtime["cuda"])
    if scheduler:
        scheduler.load_state_dict(runtime["scheduler"])
    return manifest["progress"]
