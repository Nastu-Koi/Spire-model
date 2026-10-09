"""Entity combat-outcome network: the policy's encoder and global blocks with an outcome head.

The architecture is a ModelConfig (the `model` section of a combat-outcome profile). A
saved model is `weights.pt` plus `manifest.json`, which holds that section, the
vocabulary and the training settings, so it can be rebuilt without the profile.
"""

import json
from dataclasses import asdict
from pathlib import Path

import torch

from model import rules
from model.config import ModelConfig
from model.model import PolicyValue
from model.representation import ENCODING_VERSION, Vocabulary
from model.public_history import HISTORY_VERSION, require_history_version

TARGET = "signed hp_lost/max_hp, defeat"


def check_encounters(encounters, symbols):
    """Distinct encounters must never silently share the unknown embedding."""
    unknown = sorted({name for name in encounters if "content_id=" + name not in symbols})
    if unknown:
        raise ValueError("Unknown encounter(s): " + ", ".join(unknown)
                         + "; initialize with a static public catalog containing these IDs")


class CombatOutcomeNet(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.body = PolicyValue(config)
        d = config.hidden_size
        self.head = torch.nn.Sequential(torch.nn.Linear(d, d), torch.nn.SiLU(), torch.nn.Linear(d, 2))

    def forward(self, observations, vocabulary, base=None):
        encounters = []
        for obs in observations:
            ids = [f.symbol.removeprefix("content_id=") for row in obs.tokens
                   if any(f.symbol == "entity_type=encounter" for f in row)
                   for f in row if f.name == "content_id"]
            if len(ids) != 1:
                raise ValueError("A combat-outcome input needs exactly one encounter ID")
            encounters.extend(ids)
        check_encounters(encounters, vocabulary.lookup)
        encoded = self.body.encode(observations, vocabulary)
        x = torch.stack([e.actions[0] for e in encoded])
        out = self.head(x)
        if base is not None:  # residual on a baseline prediction (loss fraction)
            return base + out[:, 0], out[:, 1]
        # HP lost as a signed fraction of max HP: a fight that heals more than it costs
        # ends below zero. The second output is the logit of losing the fight.
        return torch.tanh(out[:, 0]), out[:, 1]


def save_model(directory, net, vocabulary, config, **provenance):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    torch.save(net.state_dict(), directory / "weights.pt")
    (directory / "manifest.json").write_text(json.dumps(dict(
        model=asdict(config.model), training=asdict(config.training), vocabulary=vocabulary.state(), rules=rules.digest(),
        target=TARGET, encoding_version=ENCODING_VERSION,
        public_history_version=HISTORY_VERSION, **provenance), indent=1))


def load_model(directory, device="cpu"):
    """Return (net in eval mode, vocabulary, manifest)."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("ridge_base") or manifest.get("training", {}).get("ridge_base"):
        raise ValueError("A ridge-residual model needs its ridge baseline, which is not saved")
    if manifest.get("target") != TARGET:
        raise ValueError("Model predicts a different target")
    require_history_version(manifest.get("public_history_version"))
    if manifest.get("encoding_version") != ENCODING_VERSION:
        raise ValueError("Model input encoding changed; retrain with the current public encoding")
    if manifest.get("rules") != rules.digest():
        raise ValueError("Model was trained with different rule text")
    config = ModelConfig(**manifest["model"])
    net = CombatOutcomeNet(config)
    net.load_state_dict(torch.load(directory / "weights.pt", map_location="cpu", weights_only=True))
    return net.to(device).eval(), Vocabulary.from_state(manifest["vocabulary"], config), manifest
