"""The combat-outcome profile defines the model; flags override single values."""

import argparse
import json
from dataclasses import asdict

import pytest

from combat_outcome.config import DEFAULT_CONFIG, CombatOutcomeConfig, TrainingConfig, override
from combat_outcome.model import CombatOutcomeNet


def test_default_profile_builds_and_spells_out_every_setting():
    config = CombatOutcomeConfig.load()
    assert CombatOutcomeNet(config.model).body.config == config.model
    # Every knob appears in the file, so a run's settings can be read from it.
    profile = json.loads(DEFAULT_CONFIG.read_text())
    for section in ("model", "training"):
        assert set(profile[section]) == set(asdict(getattr(config, section)))


def test_sections_may_be_partial_and_unknown_names_are_rejected(tmp_path):
    model = asdict(CombatOutcomeConfig.load().model)
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(dict(model=dict(model, layers=4), training=dict(epochs=3))))
    config = CombatOutcomeConfig.load(path)
    assert (config.model.layers, config.training.epochs, config.training.batch_size) == (4, 3, 64)
    path.write_text(json.dumps(dict(model=model, trainin={})))
    with pytest.raises(ValueError, match="trainin"):
        CombatOutcomeConfig.load(path)
    path.write_text(json.dumps(dict(model=model, training=dict(epoch=3))))
    with pytest.raises(TypeError, match="epoch"):
        CombatOutcomeConfig.load(path)


def test_flags_override_only_given_values_and_are_validated():
    args = argparse.Namespace(epochs=5, lr=None, ridge_base=False)
    training = override(TrainingConfig(ridge_base=True), args, ("epochs", "lr", "ridge_base"))
    assert (training.epochs, training.lr, training.ridge_base) == (5, 1e-3, False)
    with pytest.raises(ValueError):
        override(TrainingConfig(), argparse.Namespace(epochs=0), ("epochs",))
