from dataclasses import asdict, dataclass, field
import json
from pathlib import Path


@dataclass
class ModelConfig:
    hidden_size: int = 1792
    num_heads: int = 28
    layers: int = 25
    ffn_size: int = 7168
    local_size: int = 256
    local_heads: int = 8
    local_layers: int = 2
    vocabulary_size: int = 16384
    field_buckets: int = 512
    relation_buckets: int = 128
    backend: str = "auto"
    checkpoint_layers: bool = False
    compile_blocks: bool = False

    def __post_init__(self):
        dimensions = (self.hidden_size, self.num_heads, self.layers, self.ffn_size, self.local_size,
                      self.local_heads, self.local_layers, self.vocabulary_size, self.field_buckets, self.relation_buckets)
        if any(type(x) is not int or x < 1 for x in dimensions) or min(self.vocabulary_size, self.field_buckets, self.relation_buckets) < 2:
            raise ValueError("Dimensions must be positive integers and vocabularies need reserved entries")
        if self.hidden_size % self.num_heads or self.local_size % self.local_heads:
            raise ValueError("Attention dimensions must divide evenly")
        if self.layers < 1 or self.layers % 2 != 1:
            raise ValueError("Hybrid schedule must start and end with Full")
        if self.backend not in {"reference", "sdpa", "flex", "auto"}:
            raise ValueError("Unknown attention backend")

    @classmethod
    def tiny(cls):
        return cls(hidden_size=48, num_heads=3, layers=3, ffn_size=96,
                   local_size=24, local_heads=3, vocabulary_size=1024, backend="reference")

    @classmethod
    def from_architecture(cls, path):
        spec = json.loads(Path(path).read_text())["model"]
        return cls(hidden_size=spec["hidden_size"], num_heads=spec["num_heads"],
                   layers=spec["layers"], ffn_size=spec["ffn_size"],
                   local_size=spec["local_effect_encoder"]["hidden_size"],
                   local_layers=spec["local_effect_encoder"]["layers"])


@dataclass
class TrainConfig:
    backbone_lr: float = 1e-5
    head_lr: float = 3e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    logical_batch_size: int = 32
    microbatch_size: int = 4
    clip_ratio: float = 0.2
    target_kl: float = 0.015
    ppo_epochs: int = 2
    value_coef: float = 0.5
    entropy_coef: float = 0.001
    bootstrap_coef: float = 0.0
    horizon_scale: float = 100.0
    precision: str = "bf16"
    token_buckets: list[int] = field(default_factory=lambda: [256, 512, 1024, 2048, 4096])
    action_buckets: list[int] = field(default_factory=lambda: [16, 32, 64, 128, 256, 512, 1024])
    token_budget: int = 8192
    pair_budget: int = 16777216

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        legacy = values.pop("bc_coef", None)
        if legacy is not None:
            values.setdefault("bootstrap_coef", legacy)
        return cls(**values)

    def __post_init__(self):
        if min(self.logical_batch_size, self.microbatch_size, self.horizon_scale) <= 0:
            raise ValueError("Batch sizes and horizon scale must be positive")
        if self.precision not in {"no", "bf16"}:
            raise ValueError("Use FP32 or BF16 consistently for rollout and learning")
        if not 0 < self.clip_ratio < 1 or self.target_kl <= 0 or self.ppo_epochs < 1:
            raise ValueError("Invalid PPO clip/KL/epoch settings")
        if min(self.backbone_lr, self.head_lr, self.max_grad_norm, self.token_budget, self.pair_budget) <= 0:
            raise ValueError("Learning rates, gradient limit and capacity budgets must be positive")
        if min(self.value_coef, self.entropy_coef, self.bootstrap_coef, self.weight_decay) < 0:
            raise ValueError("Loss coefficients and weight decay must be nonnegative")
        if any(not bins or bins != sorted(set(bins)) or min(bins) < 1 for bins in (self.token_buckets, self.action_buckets)):
            raise ValueError("Capacity buckets must be sorted unique positive sizes")


def save_config(path, model, training):
    Path(path).write_text(json.dumps({"model": asdict(model), "training": asdict(training)}, indent=2))
