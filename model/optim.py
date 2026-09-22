"""Optimizer construction for Transformer matrices and the remaining model."""
from collections import ChainMap, defaultdict

import torch
from torch import nn


class HybridOptimizer(torch.optim.Optimizer):
    """Expose Muon and AdamW as one optimizer to schedulers and Accelerate."""

    def __init__(self, adamw, muon):
        self.adamw = adamw
        self.muon = muon
        parameters = [parameter for optimizer in (adamw, muon)
                      for group in optimizer.param_groups for parameter in group["params"]]
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
        return {"adamw": self.adamw.state_dict(),
                "muon": self.muon.state_dict()}

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
        if isinstance(module, nn.Linear) and (in_global_block or in_local_block) and in_transform:
            result.add(id(module.weight))
    return result


def _adamw_groups(named_parameters, config):
    grouped = defaultdict(list)
    for name, parameter in named_parameters:
        head = not name.startswith(("encoder.", "blocks.", "final_norm."))
        decay = parameter.ndim >= 2 and not any(
            label in name for label in ("norm", "symbol", "field", "relation", "floor")
        )
        grouped[head, decay].append(parameter)
    return [{"params": parameters,
             "lr": config.head_lr if head else config.backbone_lr,
             "weight_decay": config.weight_decay if decay else 0.}
            for (head, decay), parameters in grouped.items() if parameters]


def build_optimizer(model, config):
    """Build AdamW, or Muon for Transformer matrices plus AdamW elsewhere."""
    named = list(model.named_parameters())
    fused = model.device.type == "cuda"
    if config.optimizer == "adamw":
        return torch.optim.AdamW(_adamw_groups(named, config), betas=(.9, .999), eps=1e-8, fused=fused)
    if not hasattr(torch.optim, "Muon"):
        raise RuntimeError("muon_adamw requires a PyTorch build that provides torch.optim.Muon")
    muon_ids = _muon_parameter_ids(model)
    muon_parameters = [(name, parameter) for name, parameter in named if id(parameter) in muon_ids]
    adamw_parameters = [(name, parameter) for name, parameter in named if id(parameter) not in muon_ids]
    if not muon_parameters:
        raise ValueError("No Transformer attention or FFN matrices were found for Muon")
    adamw = torch.optim.AdamW(_adamw_groups(adamw_parameters, config), betas=(.9, .999), eps=1e-8,
                              fused=fused)
    muon = torch.optim.Muon([{"params": [parameter for _, parameter in muon_parameters],
                              "lr": config.muon_lr,
                              "weight_decay": config.weight_decay}],
                            momentum=config.muon_momentum, ns_steps=config.muon_ns_steps)
    return HybridOptimizer(adamw, muon)
