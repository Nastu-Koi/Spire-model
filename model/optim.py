"""Optimizer construction for Transformer matrices and the remaining model."""

from collections import ChainMap, defaultdict

import torch
from torch import nn


TRAINING_VERSION = "bc-decay-v5"
# Lookup tables start at unit scale, about twenty times a Linear weight, and AdamW
# moves every element by about its rate: they share a rate of their own.
TABLES = frozenset({"encoder.symbol.weight", "encoder.field.weight",
                    "relation.weight", "encoder.program_relation.weight"})


# A round's KL budget is meant to last its planned updates. One that the budget stopped
# shrinks the step of the rounds after it to the share of its updates that ran, by no
# more than half at a time; one that ended with under half the budget spent lets the
# step grow back, never past the configured rates.
PPO_STEP_SHRINK, PPO_STEP_GROWTH, PPO_STEP_SLACK, PPO_STEP_FLOOR = 0.5, 1.25, 0.5, 1 / 64


def is_table(name):
    """The lookup tables, counting a relation-bias table of each global layer."""
    return name in TABLES or (name.startswith("relation.") and name.endswith(".weight"))


class BootstrapSchedule:
    """BC rates: a linear warmup, then constant rates, then an optional linear decay
    to zero over `decay_updates` updates, after which BC stops. The lookup tables
    (the `embedding` group) also fall geometrically from their peak to
    `table_final_ratio` of it over the first `table_decay_updates` updates, then
    hold.

    PPO and the value warmup run at constant rates: BC's peak rates, except that the
    tables stay where BC's decay left them and the Muon matrices move at
    `ppo_muon_ratio` of their BC rate. At BC's rates a handful of updates moves a
    trained policy as far as PPO lets a whole round move it.

    PPO further scales the policy's rates by `ppo_step`, which `pace` sets round by
    round (see the constants above); the value head keeps its own rate."""

    def __init__(self, optimizer, warmup_updates, decay_updates=0, *,
                 table_final_ratio=1.0, table_decay_updates=0, ppo_muon_ratio=1.0):
        if type(decay_updates) is not int or decay_updates < 0:
            raise ValueError("Decay updates must be a non-negative integer")
        if (type(table_decay_updates) is not int or table_decay_updates < 0
                or not 0 < table_final_ratio <= 1):
            raise ValueError("Table decay needs a non-negative length and a final ratio in (0, 1]")
        if ppo_muon_ratio <= 0:
            raise ValueError("The PPO Muon rate must be positive")
        self.optimizer = optimizer
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.warmup_updates = warmup_updates
        self.decay_updates = decay_updates
        self.table_final_ratio = table_final_ratio
        self.table_decay_updates = table_decay_updates
        self.ppo_muon_ratio = ppo_muon_ratio
        self.ppo_step = 1.0
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

    def table_factor(self):
        """How far BC's updates have brought the tables down, in every mode."""
        if not self.table_decay_updates:
            return 1.0
        progress = min(1.0, self.completed_updates / self.table_decay_updates)
        return self.table_final_ratio ** progress

    def prepare(self, mode):
        self.mode = mode
        factor, tables = self.factor(mode), self.table_factor()
        muon = 1.0 if mode == "bootstrap" else self.ppo_muon_ratio
        policy = self.ppo_step if mode == "ppo" else 1.0
        for group, peak in zip(self.optimizer.param_groups, self.base_lrs):
            name = group.get("name", "")
            group["lr"] = peak * factor * (1.0 if name.startswith("value") else policy) * (
                tables if name.startswith("embedding") else muon if name == "muon" else 1.0)

    def pace(self, share, spent):
        """Set the PPO step of the next rounds from the round that ended: `share` of its
        planned updates ran (under 1 where the KL budget stopped it), and it ended with
        `spent` of the budget used."""
        if share < 1:
            self.ppo_step *= max(PPO_STEP_SHRINK, share)
        elif spent < PPO_STEP_SLACK:
            self.ppo_step *= PPO_STEP_GROWTH
        self.ppo_step = min(1.0, max(PPO_STEP_FLOOR, self.ppo_step))

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
                "decayed_updates": self.decayed_updates, "mode": self.mode,
                "ppo_step": self.ppo_step}

    def _saved_rates(self, state):
        """The saved peak rates, group by group. A schedule saved before the value head
        had groups of its own has none for them: they take today's."""
        saved, groups = list(state["base_lrs"]), self.optimizer.param_groups
        values = [group.get("name", "").startswith("value") for group in groups]
        if len(saved) != len(groups) - sum(values):
            return saved
        rest = iter(saved)
        return [ours if value else next(rest) for ours, value in zip(self.base_lrs, values)]

    def check(self, state):
        """Raise unless this schedule continues the saved one: same warmup, and a decay
        either started now from a constant-rate checkpoint or resumed at its length."""
        if (state.get("kind") != TRAINING_VERSION or self._saved_rates(state) != self.base_lrs
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
        self.ppo_step = state.get("ppo_step", 1.0)
        self.prepare(state["mode"])


def with_value_groups(saved, groups):
    """A saved AdamW state in today's `groups`, in place.

    Before the value head had groups of its own its parameters were the last of the
    head groups. They move to groups named after today's, with their state if they
    have any; nothing else of the state changes, so the policy continues on the
    moments it was trained with.
    """
    names = [group.get("name", "") for group in groups]
    old = {group.get("name"): group for group in saved["param_groups"]}
    if not any(name.startswith("value") for name in names) or any(
            str(name).startswith("value") for name in old):
        return
    for group in groups:
        name = group.get("name", "")
        head = old.get("head" + name[len("value"):]) if name.startswith("value") else None
        if head is None or len(head["params"]) <= len(group["params"]):
            continue
        keep = len(head["params"]) - len(group["params"])
        old[name] = dict(head, name=name, params=head["params"][keep:],
                         lr=group["lr"], initial_lr=group.get("initial_lr", group["lr"]))
        head["params"] = head["params"][:keep]
    if all(name in old for name in names):
        saved["param_groups"] = [old[name] for name in names]


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
        in_local_block = name.startswith(("encoder.local_blocks.", "encoder.field_block."))
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
        head = name.startswith(("query", "key"))
        family = ("embedding" if is_table(name) else "value" if name.startswith("value")
                  else "head" if head else "backbone")
        decay = parameter.ndim >= 2 and not any(
            label in name for label in ("norm", "symbol", "field", "relation", "floor")
        )
        grouped[family, decay].append(parameter)
    return [
        {
            "params": parameters,
            "name": family + ("_decay" if decay else "_no_decay"),
            "lr": {"embedding": config.embedding_lr, "head": config.head_lr,
                   "value": config.value_lr, "backbone": config.backbone_lr}[family],
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
