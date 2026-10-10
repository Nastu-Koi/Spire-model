"""Critic profile: the policy's encoder architecture and the critic's training settings.

The `model` section of a policy profile (default `configs/rtxpro6000.json`) is the
network; the critic has the policy's ModelConfig fields. Its own settings are the
profile's optional `critic` section, the dataclass below, which may be partial.
Command-line flags override single values.
"""

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

from model.config import ModelConfig

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs/rtxpro6000.json"
INITIALIZATIONS = ("checkpoint", "random")


@dataclass
class Budgets:
    """How decision frames are packed: the policy's buckets and microbatch budgets."""

    microbatch_size: int = 256
    token_buckets: list[int] = field(
        default_factory=lambda: [64, 96, 128, 192, 256, 384, 512, 1024, 2048, 4096]
    )
    action_buckets: list[int] = field(default_factory=lambda: [16, 32, 64, 128, 256, 512, 1024])
    token_budget: int = 65536
    pair_budget: int = 16777216

    def __post_init__(self):
        if min(self.microbatch_size, self.token_budget, self.pair_budget) < 1:
            raise ValueError("Microbatch size and budgets must be positive")
        if any(not bins or bins != sorted(set(bins)) or min(bins) < 1
               for bins in (self.token_buckets, self.action_buckets)):
            raise ValueError("Capacity buckets must be sorted unique positive sizes")


@dataclass
class TrainingConfig(Budgets):
    epochs: int = 4
    # "checkpoint": the encoder and global layers start from the policy checkpoint
    # that also supplies the vocabulary; "random": only the vocabulary is taken.
    init: str = "checkpoint"
    # Muon for the Transformer matrices; AdamW for the lookup tables (`embedding_lr`),
    # the two-logit head (`head_lr`) and the rest (`backbone_lr`), as for the policy.
    muon_lr: float = 2e-4
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5
    backbone_lr: float = 1e-5
    embedding_lr: float = 1e-3
    head_lr: float = 1e-3
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_updates: int = 20  # linear warmup, then a linear decay to zero at the last update
    # What trains beside the head: the whole body, only its top `train_blocks` global
    # layers with the final norm, or nothing of it (`train_body` false: a linear probe).
    train_body: bool = True
    train_blocks: int | None = None
    act_weight: float = 1.0  # weight of the act-passed loss against the run-won loss
    logical_batch_size: int = 256  # decisions per update
    precision: str = "bf16"
    holdout: float = 0.2  # share of the seeds held out, by the policy's split rule
    seed: int = 0
    loader_workers: int = 3  # processes that parse frames and pack batches

    def __post_init__(self):
        super().__post_init__()
        if self.init not in INITIALIZATIONS:
            raise ValueError("Initialize from the checkpoint or at random")
        if min(self.epochs, self.logical_batch_size) < 1 or self.warmup_updates < 0:
            raise ValueError("Epochs and batch size must be positive, warmup nonnegative")
        if (min(self.muon_lr, self.backbone_lr, self.embedding_lr, self.head_lr, self.max_grad_norm) <= 0
                or not 0 <= self.muon_momentum < 1 or type(self.muon_ns_steps) is not int
                or self.muon_ns_steps < 1):
            raise ValueError("Learning rates, gradient limit and Muon settings must be positive")
        if self.train_blocks is not None and (type(self.train_blocks) is not int or self.train_blocks < 1):
            raise ValueError("train_blocks counts global layers from the top and must be positive")
        if min(self.weight_decay, self.act_weight) < 0:
            raise ValueError("Weight decay and loss weights must be nonnegative")
        if self.precision not in {"no", "bf16"}:
            raise ValueError("Use FP32 or BF16")
        if not 0 < self.holdout < 1:
            raise ValueError("Holdout must be in (0, 1)")
        if type(self.loader_workers) is not int or self.loader_workers < 0:
            raise ValueError("Loader workers must be a non-negative integer")


@dataclass
class CriticConfig:
    model: ModelConfig
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def load(cls, path=DEFAULT_CONFIG):
        """The `model` section of a policy profile and its `critic` section, if any.
        The profile's `training` section belongs to the policy and is not read."""
        profile = json.loads(Path(path).read_text())
        if "model" not in profile:
            raise ValueError(f"Profile {path} has no model section")
        return cls(ModelConfig(**profile["model"]), TrainingConfig(**profile.get("critic", {})))


def override(section, args, names):
    """Apply the command-line values that were given; the result is validated again."""
    return replace(section, **{n: v for n in names if (v := getattr(args, n, None)) is not None})
