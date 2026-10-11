"""The critic network, its saved form and the scorer that serves it.

The network is the policy's entity encoder and global layers (`PolicyNetwork`, whose
decision heads it leaves untrained and unsaved) read at the global token, with a
two-logit head: the run is won, the act is passed. It is a ModelConfig (the `model`
section of a profile). A saved critic is `weights.pt` plus `manifest.json`, which
holds that section, the frozen vocabulary and the training settings, so it is
rebuilt without the profile.

    scorer = critic.model.load("runs/critic/v1", device="cuda")
    p_win, p_act = scorer.predict(frames)      # any number of decision frames
    scorer.score(observations)                 # [frames, 2] probabilities, batched under the budgets

The value of a ±1 return is 2 * p_win - 1.
"""

import json
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from model import rules
from model.config import ModelConfig
from model.model import PolicyNetwork
from model.public_history import HISTORY_VERSION, require_history_version
from model.representation import ENCODING_VERSION, Vocabulary

from .batches import pack, parse
from .config import Budgets

FORMAT = "critic-v1"
TARGET = "p(run won), p(act passed)"
# The policy's encoder and global layers: what the critic shares with the policy's
# weights and saves. The policy's decision heads are neither.
BODY = ("encoder.", "blocks.", "relation.", "final_norm.")


def autocast(device, precision):
    return torch.autocast(device.type, dtype=torch.bfloat16) if precision == "bf16" else nullcontext()


def shared(name):
    """Whether a tensor of a policy's state dict belongs to the shared body."""
    return name.startswith(BODY)


class Critic(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.body = PolicyNetwork(config)
        self.head = nn.Linear(config.hidden_size, 2)
        # An untrained head predicts its bias; the trainer sets that to the base rates.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    @property
    def device(self):
        return self.head.weight.device

    def owns(self, name):
        """Whether a tensor of this module's state dict is the critic's own: the shared
        body or the head, not the body's decision heads."""
        return name.startswith("head.") or (name.startswith("body.") and shared(name[len("body."):]))

    def trainable(self):
        """(name, parameter) of everything the critic trains."""
        return [(name, parameter) for name, parameter in self.named_parameters() if self.owns(name)]

    def forward(self, batch):
        """[observations, 2] logits of a packed batch: run won, act passed. The global
        token's final state is read in FP32, whatever the body ran in."""
        hidden = self.body.hidden(batch)
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            return self.head(hidden[:, 0].float())

    def state(self):
        return {name: tensor for name, tensor in self.state_dict().items() if self.owns(name)}

    def load_state(self, state):
        unexpected = [name for name in state if not self.owns(name)]
        if unexpected:
            raise ValueError(f"Critic weights hold tensors that are not the critic's: unexpected {unexpected}")
        missing, unexpected = self.load_state_dict(state, strict=False)
        missing = [name for name in missing if self.owns(name)]
        if missing or unexpected:
            raise ValueError(f"Critic weights do not fit the network: missing {missing}, unexpected {list(unexpected)}")

    def load_body(self, policy_state):
        """Start the shared body from a policy's weights; its decision heads are left out."""
        state = {"body." + name: tensor for name, tensor in policy_state.items() if shared(name)}
        missing, unexpected = self.load_state_dict(state, strict=False)
        missing = [name for name in missing if name.startswith("body.") and self.owns(name)]
        if missing or unexpected:
            raise ValueError(f"Policy weights do not fit the critic's body: missing {missing}, unexpected {list(unexpected)}")


def save(directory, net, vocabulary, config, **provenance):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    torch.save(net.state(), directory / "weights.pt")
    (directory / "manifest.json").write_text(json.dumps(dict(
        format=FORMAT, target=TARGET, model=asdict(config.model), training=asdict(config.training),
        vocabulary=vocabulary.state(), rules=rules.digest(), encoding_version=ENCODING_VERSION,
        public_history_version=HISTORY_VERSION, **provenance), indent=1))


def load(directory, device="cpu", *, budgets=None):
    """The saved critic as a Scorer, rebuilt from its manifest alone."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("format") != FORMAT or manifest.get("target") != TARGET:
        raise ValueError("Not a critic of this format")
    require_history_version(manifest.get("public_history_version"))
    if manifest.get("encoding_version") != ENCODING_VERSION:
        raise ValueError("Critic input encoding changed; retrain with the current public encoding")
    if manifest.get("rules") != rules.digest():
        raise ValueError("Critic was trained with different rule text")
    config = ModelConfig.from_dict(manifest["model"])
    net = Critic(config)
    net.load_state(torch.load(directory / "weights.pt", map_location="cpu", weights_only=True))
    vocabulary = Vocabulary.from_state(manifest["vocabulary"], config)
    return Scorer(net.to(device).eval(), vocabulary, manifest,
                  precision=manifest["training"]["precision"], budgets=budgets)


class Scorer:
    """A critic that scores decision frames; frames are packed under `budgets`, the
    policy's buckets and microbatch budgets, into as many forward passes as needed."""

    def __init__(self, net, vocabulary, manifest=None, *, precision="bf16", budgets=None):
        self.net, self.vocabulary, self.manifest = net, vocabulary, manifest
        self.precision = precision
        self.budgets = budgets or Budgets()

    @property
    def version(self):
        return (self.manifest or {}).get("version")

    @torch.no_grad()
    def probabilities(self, batch):
        """[observations, 2] probabilities of one packed batch, in one forward pass."""
        self.net.eval()
        with autocast(self.net.device, self.precision):
            logits = self.net(batch.to(self.net.device))
        return torch.sigmoid(logits.float())

    @torch.no_grad()
    def score(self, observations):
        """[observations, 2] probabilities (run won, act passed) on the CPU, in input order."""
        out = torch.zeros(len(observations), 2)
        if observations:
            for positions, batch in pack(observations, self.vocabulary, self.budgets):
                out[positions] = self.probabilities(batch).cpu()
        return out

    def predict(self, frames):
        """(p_win, p_act) of decision frames, as two lists of floats."""
        scored = self.score(parse(frames))
        return scored[:, 0].tolist(), scored[:, 1].tolist()
