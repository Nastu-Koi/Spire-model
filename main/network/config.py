import json
import math
from dataclasses import asdict, astuple, dataclass, field, fields, replace
from pathlib import Path


# The network's structure: what the tensors of a checkpoint are. Version 3 normalizes
# queries and keys in every attention layer and in the decision head, starts the
# residual branches scaled by depth, keeps a relation-bias table per global layer,
# lets the fields of an entity attend to each other, and uses RMSNorm and Linear
# layers without biases throughout.
ARCHITECTURE_VERSION = 3


@dataclass
class ModelConfig:
    architecture_version: int = ARCHITECTURE_VERSION
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
        if self.architecture_version != ARCHITECTURE_VERSION:
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
    def from_dict(cls, values):
        """A configuration as a checkpoint saved it. One of another architecture version
        is refused by name: its tensors are not this network's."""
        if values.get("architecture_version") != ARCHITECTURE_VERSION:
            raise ValueError(
                f"Checkpoint has architecture version {values.get('architecture_version')!r}; this "
                f"network is version {ARCHITECTURE_VERSION}. Start a new one with init")
        return cls(**values)

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


# The settings no optimizer state depends on: where training goes on from a checkpoint, a
# profile may set them anew (`TrainConfig.with_sampling`).
SAMPLING_SETTINGS = ("combat_temperature", "other_temperature")
# The settings a learning-rate file sets (`TrainConfig.with_rates`): a run given one does
# not take them from its profile.
RATE_SETTINGS = ("muon_lr", "backbone_lr", "embedding_lr", "embedding_lr_final", "head_lr")


@dataclass(frozen=True)
class LearningRates:
    """The peak rate of each optimizer group a run trains at: the Muon matrices, the
    rest of the backbone, the lookup tables and the head. `embedding` is the tables'
    rate once their decay is over (`TrainConfig.embedding_lr_final`), the rate a run
    that goes on from a checkpoint trains them at. A run takes them from a file of its
    own (`load`), apart from the checkpoint's training settings; without one it keeps
    the checkpoint's (`of`)."""

    muon: float
    backbone: float
    embedding: float
    head: float

    def __post_init__(self):
        if not all(isinstance(rate, (int, float)) and math.isfinite(rate) and rate > 0 for rate in astuple(self)):
            raise ValueError("Every learning rate is a positive number")

    @classmethod
    def load(cls, path):
        """The rates a JSON file names, all four of them and nothing else."""
        values = json.loads(Path(path).read_text())
        names = {item.name for item in fields(cls)}
        if not isinstance(values, dict) or set(values) != names:
            raise ValueError(f"A learning-rate file names exactly {', '.join(sorted(names))}")
        return cls(**values)

    @classmethod
    def of(cls, training):
        """The rates a training configuration runs at."""
        return cls(muon=training.muon_lr, backbone=training.backbone_lr,
                   embedding=training.embedding_lr if training.embedding_lr_final is None
                   else training.embedding_lr_final, head=training.head_lr)


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
    bootstrap_warmup_updates: int = 200
    update_stats_every: int = 100
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    logical_batch_size: int = 512
    microbatch_size: int = 32
    # A sampled run draws every decision from the scores divided by a temperature:
    # inside a fight (what the engine's encounter events bound, a selection made in one
    # included) at `combat_temperature`, everything else at `other_temperature`.
    # The temperatures are part of what the policy is, and travel with its runs and its
    # checkpoint. The most likely action, which evaluation takes, is the same at every
    # temperature; demonstrations are learned at 1.
    combat_temperature: float = 0.3
    other_temperature: float = 0.5
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

    @property
    def temperatures(self):
        """The sampling temperature of each phase, as runs record it."""
        return {"combat": self.combat_temperature, "other": self.other_temperature}

    def with_rates(self, rates):
        """This configuration at `rates`. A table decay keeps its length and its
        ratio: it now falls to `rates.embedding`."""
        if self.embedding_lr_final is None:
            tables = dict(embedding_lr=rates.embedding)
        else:
            tables = dict(embedding_lr=rates.embedding * self.embedding_lr / self.embedding_lr_final,
                          embedding_lr_final=rates.embedding)
        return replace(self, muon_lr=rates.muon, backbone_lr=rates.backbone, head_lr=rates.head, **tables)

    def with_sampling(self, profile, rates=False):
        """This configuration with the sampling settings of `profile`, which must agree
        with it on everything else: the rest is what the optimizer state was built under.
        With `rates` (a learning-rate file gives the run its rates) the profile's rates
        are not compared; this configuration's stay, for `with_rates` to replace."""
        free = SAMPLING_SETTINGS + (RATE_SETTINGS if rates else ())
        differing = sorted(
            name for name, value in asdict(self).items()
            if name not in free and getattr(profile, name) != value)
        if differing:
            raise ValueError(
                "Strict resume can change " + ("sampling and learning-rate" if rates else "sampling")
                + " settings only; the profile also differs from the checkpoint in "
                + ", ".join(differing) + "; use --weights-only")
        return replace(self, **{name: getattr(profile, name) for name in SAMPLING_SETTINGS})

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
        if min(self.logical_batch_size, self.microbatch_size) <= 0:
            raise ValueError("Batch sizes must be positive")
        if not all(0 < temperature <= 1 for temperature in self.temperatures.values()):
            raise ValueError("The sampling temperatures must be in (0, 1]")
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
        if (
            min(
                self.backbone_lr,
                self.embedding_lr,
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
        if self.weight_decay < 0:
            raise ValueError("Weight decay must be nonnegative")
        if any(
            not bins or bins != sorted(set(bins)) or min(bins) < 1
            for bins in (self.token_buckets, self.action_buckets)
        ):
            raise ValueError("Capacity buckets must be sorted unique positive sizes")


def save_config(path, model, training):
    Path(path).write_text(
        json.dumps({"model": asdict(model), "training": asdict(training)}, indent=2)
    )
