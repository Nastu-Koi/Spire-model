"""Optimizer construction for Transformer matrices and the remaining model."""

from collections import ChainMap, defaultdict

import torch
from torch import nn


TRAINING_VERSION = "bc-decay-v5"
# Lookup tables start at unit scale, about twenty times a Linear weight, and AdamW
# moves every element by about its rate: they share a rate of their own.
TABLES = frozenset({"encoder.symbol.weight", "encoder.field.weight",
                    "relation.weight", "encoder.program_relation.weight"})


class BootstrapSchedule:
    """BC rates: a linear warmup, then constant rates, then an optional linear decay
    to zero over `decay_updates` updates, after which BC stops. The lookup tables
    (the `embedding` group) also fall geometrically from their peak to
    `table_final_ratio` of it over the first `table_decay_updates` updates, then
    hold. Other training modes use the peak rates."""

    def __init__(self, optimizer, warmup_updates, decay_updates=0, *,
                 table_final_ratio=1.0, table_decay_updates=0):
        if type(decay_updates) is not int or decay_updates < 0:
            raise ValueError("Decay updates must be a non-negative integer")
        if (type(table_decay_updates) is not int or table_decay_updates < 0
                or not 0 < table_final_ratio <= 1):
            raise ValueError("Table decay needs a non-negative length and a final ratio in (0, 1]")
        self.optimizer = optimizer
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.warmup_updates = warmup_updates
        self.decay_updates = decay_updates
        self.table_final_ratio = table_final_ratio
        self.table_decay_updates = table_decay_updates
        self.completed_updates = 0
        self.decayed_updates = 0
        self.mode = None
        for group, lr in zip(optimizer.param_groups, self.base_lrs):
            group["initial_lr"] = lr

    @property
    def exhausted(self):
        """The decay has spent its updates: the next BC update would run at rate zero."""
        return bool(self.decay_updates) and self.decayed_updates >= self.decay_updates

    def factor(self, mode):
        if mode != "bootstrap":
            return 1.0
        factor = (min(1.0, (self.completed_updates + 1) / self.warmup_updates)
                  if self.warmup_updates else 1.0)
        if self.decay_updates:
            factor *= max(0.0, 1 - self.decayed_updates / self.decay_updates)
        return factor

    def table_factor(self, mode):
        if mode != "bootstrap" or not self.table_decay_updates:
            return 1.0
        progress = min(1.0, self.completed_updates / self.table_decay_updates)
        return self.table_final_ratio ** progress

    def prepare(self, mode):
        self.mode = mode
        factor, tables = self.factor(mode), self.table_factor(mode)
        for group, peak in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = peak * factor * (tables if group.get("name", "").startswith("embedding") else 1.0)

    def step(self):
        if self.mode == "bootstrap":
            self.completed_updates += 1
            if self.decay_updates:
                self.decayed_updates += 1
        self.prepare(self.mode)

    def state_dict(self):
        return {"kind": TRAINING_VERSION, "base_lrs": self.base_lrs,
                "warmup_updates": self.warmup_updates,
                "table_final_ratio": self.table_final_ratio,
                "table_decay_updates": self.table_decay_updates,
                "completed_updates": self.completed_updates,
                "decay_updates": self.decay_updates,
                "decayed_updates": self.decayed_updates, "mode": self.mode}

    def check(self, state):
        """Raise unless this schedule continues the saved one: same warmup, and a decay
        either started now from a constant-rate checkpoint or resumed at its length."""
        if (state.get("kind") != TRAINING_VERSION or state["base_lrs"] != self.base_lrs
                or state["warmup_updates"] != self.warmup_updates
                or state.get("table_final_ratio", 1.0) != self.table_final_ratio
                or state.get("table_decay_updates", 0) != self.table_decay_updates):
            raise ValueError("Warmup or table decay configuration differs; start with weights only instead of resuming")
        saved = state.get("decay_updates", 0)
        if saved and saved != self.decay_updates:
            raise ValueError(f"Checkpoint is inside a decay of {saved} updates; "
                             "resume it with the same --decay-updates")

    def load_state_dict(self, state):
        self.check(state)
        self.completed_updates = state["completed_updates"]
        self.decayed_updates = (state.get("decayed_updates", 0)
                                if state.get("decay_updates", 0) == self.decay_updates else 0)
        self.prepare(state["mode"])


class HybridOptimizer(torch.optim.Optimizer):
    """Expose Muon and AdamW as one optimizer to schedulers and Accelerate."""

    def __init__(self, adamw, muon):
        self.adamw = adamw
        self.muon = muon
        parameters = [
            parameter
            for optimizer in (adamw, muon)
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        super().__init__(parameters, {})
        # Keep the exact dictionaries owned by the child optimizers: schedulers
        # update these groups and the child step methods observe the new LR.
        self.param_groups = adamw.param_groups + muon.param_groups
        self.state = ChainMap(adamw.state, muon.state)
        self.defaults = {"hybrid_optimizer": True}

    def zero_grad(self, set_to_none=True):
        self.adamw.zero_grad(set_to_none=set_to_none)
        self.muon.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self, closure=None):
        loss = self.adamw.step(closure)
        self.muon.step()
        return loss

    def state_dict(self):
        return {"adamw": self.adamw.state_dict(), "muon": self.muon.state_dict()}

    def load_state_dict(self, state_dict):
        if set(state_dict) != {"adamw", "muon"}:
            raise ValueError("Unsupported hybrid optimizer state")
        self.adamw.load_state_dict(state_dict["adamw"])
        self.muon.load_state_dict(state_dict["muon"])
        self.param_groups = self.adamw.param_groups + self.muon.param_groups
        self.state = ChainMap(self.adamw.state, self.muon.state)


def _muon_parameter_ids(model):
    result = set()
    for name, module in model.named_modules():
        in_global_block = name.startswith("blocks.")
        in_local_block = name.startswith("encoder.local_blocks.")
        in_transform = ".attention." in name or ".ffn." in name
        if (
            isinstance(module, nn.Linear)
            and (in_global_block or in_local_block)
            and in_transform
        ):
            result.add(id(module.weight))
    return result


def _adamw_groups(named_parameters, config):
    grouped = defaultdict(list)
    for name, parameter in named_parameters:
        head = name.startswith(("query.", "key.", "value."))
        family = "embedding" if name in TABLES else "head" if head else "backbone"
        decay = parameter.ndim >= 2 and not any(
            label in name for label in ("norm", "symbol", "field", "relation", "floor")
        )
        grouped[family, decay].append(parameter)
    return [
        {
            "params": parameters,
            "name": family + ("_decay" if decay else "_no_decay"),
            "lr": {"embedding": config.embedding_lr, "head": config.head_lr,
                   "backbone": config.backbone_lr}[family],
            "weight_decay": config.weight_decay if decay else 0.0,
        }
        for (family, decay), parameters in grouped.items()
        if parameters
    ]


def build_optimizer(model, config):
    """Build AdamW, or Muon for Transformer matrices plus AdamW elsewhere."""
    named = list(model.named_parameters())
    fused = model.device.type == "cuda"
    if config.optimizer == "adamw":
        return torch.optim.AdamW(
            _adamw_groups(named, config), betas=(0.9, 0.999), eps=1e-8, fused=fused
        )
    if not hasattr(torch.optim, "Muon"):
        raise RuntimeError(
            "muon_adamw requires a PyTorch build that provides torch.optim.Muon"
        )
    muon_ids = _muon_parameter_ids(model)
    muon_parameters = [
        (name, parameter) for name, parameter in named if id(parameter) in muon_ids
    ]
    adamw_parameters = [
        (name, parameter) for name, parameter in named if id(parameter) not in muon_ids
    ]
    if not muon_parameters:
        raise ValueError("No Transformer attention or FFN matrices were found for Muon")
    adamw = torch.optim.AdamW(
        _adamw_groups(adamw_parameters, config),
        betas=(0.9, 0.999),
        eps=1e-8,
        fused=fused,
    )
    muon = torch.optim.Muon(
        [
            {
                "params": [parameter for _, parameter in muon_parameters],
                "name": "muon",
                "lr": config.muon_lr,
                "weight_decay": config.weight_decay,
            }
        ],
        momentum=config.muon_momentum,
        ns_steps=config.muon_ns_steps,
    )
    return HybridOptimizer(adamw, muon)
