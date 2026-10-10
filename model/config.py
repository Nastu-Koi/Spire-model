import json
from dataclasses import asdict, dataclass, field, replace
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
    # LayerNorm on each head's queries and keys inside every attention layer.
    qk_norm: bool = False
    # Residual-branch output projections start scaled by 1/sqrt(2 * layers).
    scaled_residual_init: bool = False
    # Options of the 200M structure. Off, the module names and tensors are those
    # of the 20M model.
    head_qk_norm: bool = False  # normalize the decision query and the action keys
    relation_bias_per_layer: bool = False  # a relation-bias table per global layer
    field_attention: bool = False  # one attention layer over an entity's fields
    value_mlp: bool = False  # two-layer value head instead of a linear one
    linear_bias: bool = True  # biases in Linear layers; the value output keeps its own
    rms_norm: bool = False  # RMSNorm instead of LayerNorm everywhere

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


# The settings of PPO rounds that no optimizer state depends on: where training goes on
# from a checkpoint, a profile may set them anew (`TrainConfig.with_ppo`).
PPO_SETTINGS = ("clip_ratio", "target_kl", "ppo_epochs", "ppo_muon_lr", "value_coef",
                "entropy_coef", "bootstrap_coef", "bootstrap_batch_size", "sampling_temperature")


@dataclass
class TrainConfig:
    optimizer: str = "muon_adamw"
    muon_lr: float = 0.001
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5
    backbone_lr: float = 1e-5
    embedding_lr: float = 1e-5
    # The lookup tables fall geometrically to `embedding_lr_final` over the first
    # `embedding_decay_updates` BC updates (none: constant), then hold.
    embedding_lr_final: float | None = None
    embedding_decay_updates: int = 0
    head_lr: float = 3e-5
    # The value head starts from zero when PPO begins, and BC never moves it: it
    # learns at a rate of its own, not at the one that keeps the policy heads steady.
    value_lr: float = 1e-3
    bootstrap_warmup_updates: int = 200
    update_stats_every: int = 100
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    logical_batch_size: int = 512
    microbatch_size: int = 32
    clip_ratio: float = 0.2
    target_kl: float = 0.015
    ppo_epochs: int = 2
    # The Muon rate of PPO updates: `ppo_epochs` passes over a round at this rate
    # move the policy by about `target_kl`. The other groups keep their BC rates,
    # the lookup tables the one BC's decay left them at.
    ppo_muon_lr: float = 2e-4
    value_coef: float = 0.5
    entropy_coef: float = 0.001
    # PPO updates also lower the NLL of `bootstrap_batch_size` demonstrations each,
    # their mean weighed at `bootstrap_coef` against the round's own loss, sample for
    # sample. 0 leaves the demonstrations out.
    bootstrap_coef: float = 0.0
    bootstrap_batch_size: int = 256
    # PPO rounds are sampled from the scores divided by this, and their updates score
    # the same distribution: under 1 the policy that is sampled, and improved, plays
    # closer to its most likely actions. Demonstrations are learned at 1 throughout.
    sampling_temperature: float = 1.0
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
    # Feeder processes that decode shards in turn, each with `loader_workers` packers;
    # together they hold one window.
    loader_feeders: int = 4
    # Difficulty of the runs the policy plays itself: rollouts and evaluation.
    # Demonstrations keep their own; the player entity carries it into the model.
    ascension: int = 0

    @classmethod
    def from_dict(cls, values):
        return cls(**values)

    def with_ppo(self, profile):
        """This configuration with the PPO settings of `profile`, which must agree with
        it on everything else: the rest is what the optimizer state was built under."""
        differing = sorted(
            name for name, value in asdict(self).items()
            if name not in PPO_SETTINGS and getattr(profile, name) != value)
        if differing:
            raise ValueError(
                "Strict resume can change PPO settings only; the profile also differs from the "
                "checkpoint in " + ", ".join(differing) + "; use --weights-only")
        return replace(self, **{name: getattr(profile, name) for name in PPO_SETTINGS})

    def __post_init__(self):
        if self.optimizer not in {"muon_adamw", "adamw"}:
            raise ValueError("Use muon_adamw or adamw")
        if (
            min(self.muon_lr, self.ppo_muon_lr) <= 0
            or not 0 <= self.muon_momentum < 1
            or type(self.muon_ns_steps) is not int
            or self.muon_ns_steps < 1
        ):
            raise ValueError("Invalid Muon learning rate, momentum or iteration count")
        if min(self.logical_batch_size, self.microbatch_size, self.bootstrap_batch_size,
               self.horizon_scale) <= 0:
            raise ValueError("Batch sizes and horizon scale must be positive")
        if not 0 < self.sampling_temperature <= 1:
            raise ValueError("The sampling temperature must be in (0, 1]")
        if type(self.loader_workers) is not int or self.loader_workers < 0:
            raise ValueError("Loader workers must be a non-negative integer")
        if type(self.loader_feeders) is not int or self.loader_feeders < 1:
            raise ValueError("Loader feeders must be a positive integer")
        if (type(self.bootstrap_warmup_updates) is not int or self.bootstrap_warmup_updates < 0
                or type(self.update_stats_every) is not int or self.update_stats_every < 1):
            raise ValueError("Warmup updates must be nonnegative and update stats interval positive")
        if type(self.embedding_decay_updates) is not int or self.embedding_decay_updates < 0:
            raise ValueError("Embedding decay updates must be a non-negative integer")
        if (self.embedding_lr_final is None) != (self.embedding_decay_updates == 0):
            raise ValueError("Set embedding_lr_final and embedding_decay_updates together")
        if self.embedding_lr_final is not None and not 0 < self.embedding_lr_final <= self.embedding_lr:
            raise ValueError("The final embedding rate must be positive and at most the peak rate")
        if type(self.ascension) is not int or not 0 <= self.ascension <= 10:
            raise ValueError("Ascension must be an integer from 0 to 10")
        if self.precision not in {"no", "bf16"}:
            raise ValueError("Use FP32 or BF16 consistently for rollout and learning")
        if not 0 < self.clip_ratio < 1 or self.target_kl <= 0 or self.ppo_epochs < 1:
            raise ValueError("Invalid PPO clip/KL/epoch settings")
        if (
            min(
                self.backbone_lr,
                self.embedding_lr,
                self.head_lr,
                self.value_lr,
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
