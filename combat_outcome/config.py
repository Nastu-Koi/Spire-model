"""Combat-outcome profile: encoder architecture and training.

`configs/combat-outcome.json` is the default. Its `model` section uses the policy's
ModelConfig fields; `training` is the dataclass below and may be partial.
Command-line flags override single values of a loaded profile.
"""

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

from model.config import ModelConfig

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs/combat-outcome.json"


@dataclass
class TrainingConfig:
    epochs: int = 30
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_fraction: float = 0.1  # OneCycle share spent raising the learning rate
    defeat_weight: float = 0.1  # weight of the lost-fight classification loss
    summary_entity: bool = False  # add a deck-aggregate entity to the input
    ridge_base: bool = False  # predict the residual of the deck-summary ridge baseline
    holdout: float = 0.2
    seed: int = 0
    loader_workers: int = 8  # processes that turn rows into model inputs

    def __post_init__(self):
        if min(self.epochs, self.batch_size) < 1 or min(self.lr, self.max_grad_norm) <= 0:
            raise ValueError("Epochs, batch size, learning rate and gradient limit must be positive")
        if min(self.weight_decay, self.defeat_weight, self.loader_workers) < 0:
            raise ValueError("Weight decay and loss weights must be nonnegative")
        if not 0 < self.warmup_fraction < 1 or not 0 < self.holdout < 1:
            raise ValueError("Warmup fraction and holdout must be in (0, 1)")


@dataclass
class CombatOutcomeConfig:
    model: ModelConfig
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def load(cls, path=DEFAULT_CONFIG):
        profile = json.loads(Path(path).read_text())
        if unknown := set(profile) - {"model", "training"}:
            raise ValueError(f"Unknown combat-outcome sections in {path}: {sorted(unknown)}")
        return cls(ModelConfig(**profile["model"]), TrainingConfig(**profile.get("training", {})))


def override(section, args, names):
    """Apply the command-line values that were given; the result is validated again."""
    return replace(section, **{n: v for n in names if (v := getattr(args, n)) is not None})
