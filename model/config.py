import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    architecture_version: int = 2
    hidden_size: int = 256
    num_heads: int = 4
    layers: int = 4
    ffn_size: int = 704
    local_size: int = 64
    local_heads: int = 4
    local_layers: int = 1
    vocabulary_size: int = 16384
    field_size: int = 1024
    relation_size: int = 128
    backend: str = "sdpa"
    encoder_precision: str = "inherit"
    checkpoint_layers: bool = False
    compile_blocks: bool = False

    def __post_init__(self):
        if self.architecture_version != 2:
            raise ValueError("Unsupported architecture version")
        if self.encoder_precision not in {"inherit", "fp32"}:
            raise ValueError("Unknown encoder precision")
        dimensions = (
            self.hidden_size,
            self.num_heads,
            self.layers,
            self.ffn_size,
            self.local_size,
            self.local_heads,
            self.local_layers,
            self.vocabulary_size,
            self.field_size,
            self.relation_size,
        )
        if (
            any(type(x) is not int or x < 1 for x in dimensions)
            or min(self.vocabulary_size, self.field_size, self.relation_size) < 2
        ):
            raise ValueError(
                "Dimensions must be positive integers and vocabularies need reserved entries"
            )
        if self.hidden_size % self.num_heads or self.local_size % self.local_heads:
            raise ValueError("Attention dimensions must divide evenly")
        if self.backend not in {"reference", "sdpa"}:
            raise ValueError("Unknown attention backend")

    @classmethod
    def tiny(cls):
        return cls(
            hidden_size=64,
            num_heads=4,
            layers=2,
            ffn_size=176,
            local_size=32,
            local_heads=4,
            local_layers=1,
            vocabulary_size=4096,
            backend="reference",
        )

    @classmethod
    def from_architecture(cls, path):
        spec = json.loads(Path(path).read_text())["model"]
        return cls(
            hidden_size=spec["hidden_size"],
            num_heads=spec["num_heads"],
            layers=spec["layers"],
            ffn_size=spec["ffn_size"],
            local_size=spec["local_effect_encoder"]["hidden_size"],
            local_layers=spec["local_effect_encoder"]["layers"],
        )


@dataclass
class TrainConfig:
    optimizer: str = "muon_adamw"
    muon_lr: float = 0.001
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5
    backbone_lr: float = 1e-5
    head_lr: float = 3e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    logical_batch_size: int = 32
    microbatch_size: int = 32
    clip_ratio: float = 0.2
    target_kl: float = 0.015
    ppo_epochs: int = 2
    value_coef: float = 0.5
    entropy_coef: float = 0.001
    bootstrap_coef: float = 0.0
    horizon_scale: float = 100.0
    precision: str = "bf16"
    token_buckets: list[int] = field(
        default_factory=lambda: [64, 96, 128, 192, 256, 384, 512, 1024, 2048, 4096]
    )
    action_buckets: list[int] = field(
        default_factory=lambda: [16, 32, 64, 128, 256, 512, 1024]
    )
    token_budget: int = 32768
    pair_budget: int = 16777216
    loader_workers: int = 8
    # Difficulty of the runs the policy plays itself: rollouts and evaluation.
    # Demonstrations keep their own; the player entity carries it into the model.
    ascension: int = 10

    @classmethod
    def from_dict(cls, values):
        return cls(**values)

    def __post_init__(self):
        if self.optimizer not in {"muon_adamw", "adamw"}:
            raise ValueError("Use muon_adamw or adamw")
        if (
            self.muon_lr <= 0
            or not 0 <= self.muon_momentum < 1
            or type(self.muon_ns_steps) is not int
            or self.muon_ns_steps < 1
        ):
            raise ValueError("Invalid Muon learning rate, momentum or iteration count")
        if min(self.logical_batch_size, self.microbatch_size, self.horizon_scale) <= 0:
            raise ValueError("Batch sizes and horizon scale must be positive")
        if type(self.loader_workers) is not int or self.loader_workers < 0:
            raise ValueError("Loader workers must be a non-negative integer")
        if type(self.ascension) is not int or not 0 <= self.ascension <= 10:
            raise ValueError("Ascension must be an integer from 0 to 10")
        if self.precision not in {"no", "bf16"}:
            raise ValueError("Use FP32 or BF16 consistently for rollout and learning")
        if not 0 < self.clip_ratio < 1 or self.target_kl <= 0 or self.ppo_epochs < 1:
            raise ValueError("Invalid PPO clip/KL/epoch settings")
        if (
            min(
                self.backbone_lr,
                self.head_lr,
                self.max_grad_norm,
                self.token_budget,
                self.pair_budget,
            )
            <= 0
        ):
            raise ValueError(
                "Learning rates, gradient limit and capacity budgets must be positive"
            )
        if (
            min(
                self.value_coef,
                self.entropy_coef,
                self.bootstrap_coef,
                self.weight_decay,
            )
            < 0
        ):
            raise ValueError("Loss coefficients and weight decay must be nonnegative")
        if any(
            not bins or bins != sorted(set(bins)) or min(bins) < 1
            for bins in (self.token_buckets, self.action_buckets)
        ):
            raise ValueError("Capacity buckets must be sorted unique positive sizes")


def save_config(path, model, training):
    Path(path).write_text(
        json.dumps({"model": asdict(model), "training": asdict(training)}, indent=2)
    )
