"""Single-device Accelerate Bootstrap, value warmup, and complete-run macro-action PPO."""
from collections import defaultdict
import math
import random

import torch

from .config import TrainConfig
from .data import microbatches, samples
from .policy import replay
from .protocol import ProtocolError
from .rollout import numeric_backend, precision_context


def build_optimizer(model, config):
    grouped = defaultdict(list)
    for name, parameter in model.named_parameters():
        head = not name.startswith(("encoder.", "blocks.", "final_norm."))
        decay = parameter.ndim >= 2 and not any(x in name for x in ("norm", "symbol", "field", "relation", "floor"))
        grouped[head, decay].append(parameter)
    groups = [{"params": parameters, "lr": config.head_lr if head else config.backbone_lr,
               "weight_decay": config.weight_decay if decay else 0.}
              for (head, decay), parameters in grouped.items()]
    return torch.optim.AdamW(groups, betas=(.9, .999), eps=1e-8, fused=model.device.type == "cuda")


def ppo_objective(new_log_prob, old_log_prob, advantage, clip_ratio):
    log_ratio = new_log_prob.float() - old_log_prob
    ratio = torch.exp(log_ratio)
    actor = -torch.minimum(ratio * advantage, ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * advantage)
    kl = (ratio - 1) - log_ratio
    clipped = ((ratio - 1).abs() > clip_ratio).float()
    return actor, kl, clipped


class Learner:
    def __init__(self, model, vocabulary, config=None, *, policy_version=0):
        from accelerate import Accelerator
        self.config = config or TrainConfig()
        self.accelerator = Accelerator(cpu=model.device.type == "cpu", mixed_precision=self.config.precision,
                                       gradient_accumulation_steps=1)
        if self.accelerator.num_processes != 1:
            raise ValueError("v1 uses one GPU learner; distributed reductions have not been enabled")
        self.model, self.vocabulary = model, vocabulary
        self.optimizer = build_optimizer(model, self.config)
        # Explicit logical-batch boundaries prevent a short/variable microbatch from
        # receiving a different statistical weight. No automatic loss division.
        self.optimizer = self.accelerator.prepare(self.optimizer)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer.optimizer, lambda _: 1.0)
        self.policy_version, self.updates = policy_version, 0

    def _optimize(self, items, mode, demonstrations=None):
        cfg = self.config
        self.model.train()
        total_metrics = defaultdict(float)
        weights = 0.
        groups = defaultdict(lambda: [0., 0.])
        value_sums = defaultdict(float)
        for start in range(0, len(items), cfg.logical_batch_size):
            logical = items[start:start + cfg.logical_batch_size]
            self.optimizer.zero_grad(set_to_none=True)
            batch_kl, batch_weight = 0., 0.
            for micro in microbatches(logical, cfg):
                # Sequential full-session replay bounds activation memory. Each
                # session backpropagates once; the shared logical batch steps once.
                for item in micro:
                    with precision_context(self.model, cfg.precision):
                        log_prob, value, entropy = replay(self.model, self.vocabulary, item.macro["steps"])
                        if mode == "bootstrap":
                            loss = -log_prob
                            kl = clipped = log_prob.new_zeros(())
                            value_loss = log_prob.new_zeros(())
                        else:
                            macro = item.macro
                            actor, kl, clipped = ppo_objective(log_prob, macro["old_log_prob"], macro["advantage"], cfg.clip_ratio)
                            value_loss = (value.float() - macro["return"]) ** 2
                            loss = value_loss if mode == "value" else actor + cfg.value_coef * value_loss - cfg.entropy_coef * entropy
                        weighted_loss = item.weight * loss
                    if not torch.isfinite(weighted_loss):
                        self.optimizer.zero_grad(set_to_none=True)
                        raise FloatingPointError("Non-finite loss; optimizer update cancelled")
                    self.accelerator.backward(weighted_loss)
                    w = item.weight
                    batch_kl += float(kl.detach()) * w
                    batch_weight += w
                    group = (item.character, item.macro["phase"])
                    groups[group][0] += float(kl.detach()) * w
                    groups[group][1] += w
                    total_metrics["loss"] += float(loss.detach()) * w
                    total_metrics["value_mse"] += float(value_loss.detach()) * w
                    total_metrics["entropy"] += float(entropy.detach()) * w
                    total_metrics["clip_fraction"] += float(clipped.detach()) * w
                    total_metrics["kl"] += float(kl.detach()) * w
                    weights += w
                    if mode != "bootstrap":
                        predicted, target = float(value.detach()), item.macro["return"]
                        value_sums["weight"] += w
                        value_sums["target"] += w * target
                        value_sums["target2"] += w * target * target
                        value_sums["predicted"] += w * predicted
                        value_sums["predicted2"] += w * predicted * predicted
                        value_sums["error"] += w * (target-predicted)
                        value_sums["error2"] += w * (target-predicted)**2
            if demonstrations and cfg.bootstrap_coef and mode == "ppo":
                auxiliary = random.choices(demonstrations, weights=[x.weight for x in demonstrations], k=1)[0]
                with precision_context(self.model, cfg.precision):
                    lp, _, _ = replay(self.model, self.vocabulary, auxiliary.macro["steps"])
                    auxiliary_loss = -cfg.bootstrap_coef * lp
                self.accelerator.backward(auxiliary_loss)
            norm = self.accelerator.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
            if not torch.isfinite(norm):
                self.optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError("Non-finite gradients; optimizer update cancelled")
            self.optimizer.step()
            self.scheduler.step()
            self.updates += 1
            total_metrics["last_grad_norm"] = float(norm)
            if mode == "ppo" and batch_kl / max(batch_weight, 1e-12) > cfg.target_kl:
                total_metrics["early_stop"] = 1
                break
        for key in ("loss", "value_mse", "entropy", "clip_fraction", "kl"):
            total_metrics[key] /= max(weights, 1e-12)
        total_metrics["kl_by_character_phase"] = {f"{c}/{p}": total / max(w, 1e-12) for (c, p), (total, w) in groups.items()}
        total_metrics["updates"] = self.updates
        if value_sums["weight"]:
            w = value_sums["weight"]
            variance = max(0., value_sums["target2"]/w - (value_sums["target"]/w)**2)
            error_variance = max(0., value_sums["error2"]/w - (value_sums["error"]/w)**2)
            total_metrics["value_rmse"] = math.sqrt(value_sums["error2"]/w)
            total_metrics["value_explained_variance"] = 1 - error_variance/variance if variance > 1e-12 else None
            total_metrics["value_mse_over_constant_baseline"] = value_sums["error2"]/w/variance if variance > 1e-12 else None
            total_metrics["return_std"] = math.sqrt(variance)
            total_metrics["value_std"] = math.sqrt(max(0., value_sums["predicted2"]/w-(value_sums["predicted"]/w)**2))
        return dict(total_metrics)

    def bootstrap(self, runs, epochs=1, on_epoch=None):
        if epochs < 1:
            raise ValueError("Bootstrap needs at least one epoch")
        items = samples(runs)
        if not items:
            raise ValueError("No accepted behavior-cloning samples")
        metrics = []
        for _ in range(epochs):
            random.shuffle(items)
            metrics.append(self._optimize(items, "bootstrap"))
            self.policy_version += 1
            if on_epoch:
                on_epoch(len(metrics), metrics[-1])
        return metrics

    @torch.no_grad()
    def check_old_policy(self, runs, items):
        self.model.eval()
        for run in runs:
            if run["policy_version"] != self.policy_version or run["precision"] != self.config.precision or run["vocabulary_hash"] != self.vocabulary.digest:
                raise ProtocolError("Rollout policy/precision/vocabulary differs from learner")
            if run.get("numeric_backend") != numeric_backend(self.model):
                raise ProtocolError("Rollout numeric backend differs from learner; collect fresh on-policy data")
        largest = 0.
        with precision_context(self.model, self.config.precision):
            for item in items:
                log_prob, value, _ = replay(self.model, self.vocabulary, item.macro["steps"])
                error = abs(float(log_prob) - item.macro["old_log_prob"])
                value_error = abs(float(value) - item.macro["old_value"])
                largest = max(largest, error, value_error)
        if largest > (0.02 if self.config.precision == "bf16" else 1e-4):
            raise ProtocolError(f"Stored old probabilities/values do not replay: max error={largest}")
        return largest

    def ppo(self, runs, demonstrations=None):
        items = samples(runs, ppo=True, horizon_scale=self.config.horizon_scale)
        if not items:
            raise ValueError("Round contains no stochastic decisions")
        replay_error = self.check_old_policy(runs, items)
        auxiliary = samples(demonstrations) if demonstrations else None
        metrics = []
        for _ in range(self.config.ppo_epochs):
            random.shuffle(items)
            result = self._optimize(items, "ppo", auxiliary)
            result["old_replay_max_error"] = replay_error
            metrics.append(result)
            if result.get("early_stop"):
                break
        self.policy_version += 1
        return metrics

    def value_warmup(self, runs, epochs=1):
        if epochs < 1:
            raise ValueError("Value warmup needs at least one epoch")
        items = samples(runs, ppo=True, horizon_scale=self.config.horizon_scale)
        self.check_old_policy(runs, items)
        trainable = {name: p.requires_grad for name, p in self.model.named_parameters()}
        try:
            for name, p in self.model.named_parameters():
                p.requires_grad_(name.startswith("value."))
            metrics = [self._optimize(items, "value") for _ in range(epochs)]
        finally:
            for name, p in self.model.named_parameters():
                p.requires_grad_(trainable[name])
        # The next round must capture fresh old values, despite unchanged actor weights.
        self.policy_version += 1
        return metrics

    @torch.no_grad()
    def evaluate_bootstrap(self, runs):
        self.model.eval()
        metrics = defaultdict(lambda: {"nll": 0., "branches": 0, "macros": 0})
        with precision_context(self.model, self.config.precision):
            for run in runs:
                for macro in run["macros"]:
                    lp, _, _ = replay(self.model, self.vocabulary, macro["steps"])
                    group = metrics[run["character"] + "/" + macro["phase"]]
                    group["nll"] -= float(lp)
                    group["branches"] += sum(not s["forced"] for s in macro["steps"])
                    group["macros"] += 1
        return {key: dict(value, nll_per_branch=value["nll"] / value["branches"]) for key, value in metrics.items()}


def evaluate_runs(runs):
    report = {}
    for character in ("Ironclad", "Silent", "Defect", "Regent", "Necrobinder"):
        group = [r for r in runs if r["character"] == character]
        completed = [r for r in group if r["status"] == "complete"]
        n, wins = len(completed), sum(r.get("victory", False) for r in completed)
        rate = wins / n if n else None
        z = 1.96
        if n:
            center = (rate + z*z/(2*n)) / (1 + z*z/n)
            radius = z * math.sqrt(rate*(1-rate)/n + z*z/(4*n*n)) / (1 + z*z/n)
            interval = [max(0., center-radius), min(1., center+radius)]
        else:
            interval = None
        report[character] = {"attempts": len(group), "complete": n, "wins": wins, "win_rate": rate,
                             "wilson_95": interval, "errors_or_unresolved": len(group)-n}
    rates = [x["win_rate"] for x in report.values()]
    return {"characters": report, "equal_character_win_rate": sum(rates)/5 if all(r is not None for r in rates) else None,
            "worst_character_win_rate": min(rates) if all(r is not None for r in rates) else None}
