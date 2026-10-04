"""Single-device Accelerate Bootstrap, value warmup, and complete-run macro-action PPO."""

import math
import random
from collections import defaultdict

import torch

from .config import TrainConfig
from .data import RunShards, batch_pad_size, microbatches, samples
from .loader import measure, replays
from .optim import build_optimizer
from .policy import replay_batch
from .protocol import ProtocolError
from .rollout import numeric_backend, precision_context


def ppo_objective(new_log_prob, old_log_prob, advantage, clip_ratio):
    log_ratio = new_log_prob.float() - old_log_prob
    ratio = torch.exp(log_ratio)
    actor = -torch.minimum(
        ratio * advantage, ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * advantage
    )
    kl = (ratio - 1) - log_ratio
    clipped = ((ratio - 1).abs() > clip_ratio).float()
    return actor, kl, clipped


def _adjacent_bf16_values(expected, actual):
    """Allow one quantization step, only for exactly representable BF16 values."""
    values = torch.tensor([expected, actual], dtype=torch.bfloat16)
    if values.float().tolist() != [expected, actual]:
        return False
    return float(torch.nextafter(values[0], values[1])) == actual


class Learner:
    def __init__(self, model, vocabulary, config=None, *, policy_version=0):
        from accelerate import Accelerator

        self.config = config or TrainConfig()
        self.accelerator = Accelerator(
            cpu=model.device.type == "cpu",
            mixed_precision=self.config.precision,
            gradient_accumulation_steps=1,
        )
        if self.accelerator.num_processes != 1:
            raise ValueError(
                "v1 uses one GPU learner; distributed reductions have not been enabled"
            )
        self.model, self.vocabulary = model, vocabulary
        self.optimizer = build_optimizer(model, self.config)
        # Explicit logical-batch boundaries prevent a short/variable microbatch from
        # receiving a different statistical weight. No automatic loss division.
        self.optimizer = self.accelerator.prepare(self.optimizer)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer.optimizer, lambda _: 1.0
        )
        self.policy_version, self.updates = policy_version, 0

    def _replays(self, micros):
        """Macro-action (log-prob, value, entropy) rows for each microbatch, in order."""
        for prepared in replays(micros, self.vocabulary, self.config):
            # Leave autocast before yielding: the consumer runs outside it.
            with precision_context(self.model, self.config.precision):
                outputs = self.model.replay(prepared.to(self.model.device))
            yield outputs

    def _optimize(self, items, mode, demonstrations=None):
        cfg = self.config
        self.model.train()
        total_metrics = defaultdict(float)
        weights = 0.0
        groups = defaultdict(lambda: [0.0, 0.0])
        value_sums = defaultdict(float)
        measure(items, cfg)
        logicals = [
            microbatches(items[start : start + cfg.logical_batch_size], cfg)
            for start in range(0, len(items), cfg.logical_batch_size)
        ]
        stream = self._replays([micro for logical in logicals for micro in logical])
        for logical in logicals:
            self.optimizer.zero_grad(set_to_none=True)
            logical_rows = []
            logical_items = []
            for micro in logical:
                outputs = next(stream)
                with precision_context(self.model, cfg.precision):
                    losses, rows = [], []
                    for item, log_prob, value, entropy in zip(micro, *outputs):
                        if mode == "bootstrap":
                            loss = -log_prob
                            kl = clipped = log_prob.new_zeros(())
                            value_loss = log_prob.new_zeros(())
                            predicted = log_prob.new_zeros(())
                        else:
                            macro = item.macro
                            actor, kl, clipped = ppo_objective(
                                log_prob,
                                macro["old_log_prob"],
                                macro["advantage"],
                                cfg.clip_ratio,
                            )
                            value_loss = (value.float() - macro["return"]) ** 2
                            loss = (
                                value_loss
                                if mode == "value"
                                else actor
                                + cfg.value_coef * value_loss
                                - cfg.entropy_coef * entropy
                            )
                            predicted = value.float()
                        losses.append(item.weight * loss)
                        rows.append(
                            torch.stack(
                                (
                                    loss.float(),
                                    value_loss.float(),
                                    entropy.float(),
                                    clipped.float(),
                                    kl.float(),
                                    predicted,
                                )
                            )
                        )
                    weighted_loss = torch.stack(losses).sum()
                if not bool(torch.isfinite(weighted_loss)):
                    self.optimizer.zero_grad(set_to_none=True)
                    raise FloatingPointError(
                        "Non-finite loss; optimizer update cancelled"
                    )
                self.accelerator.backward(weighted_loss)
                logical_rows.append(torch.stack(rows).detach())
                logical_items.extend(micro)
            if demonstrations and cfg.bootstrap_coef and mode == "ppo":
                auxiliary = random.choices(
                    demonstrations, weights=[x.weight for x in demonstrations], k=1
                )[0]
                with precision_context(self.model, cfg.precision):
                    lp, _, _ = replay_batch(
                        self.model,
                        self.vocabulary,
                        [auxiliary.macro["steps"]],
                        pad_to=batch_pad_size([auxiliary], cfg),
                    )[0]
                    auxiliary_loss = -cfg.bootstrap_coef * lp
                self.accelerator.backward(auxiliary_loss)
            norm = self.accelerator.clip_grad_norm_(
                self.model.parameters(), cfg.max_grad_norm
            )
            if not bool(torch.isfinite(norm)):
                self.optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError(
                    "Non-finite gradients; optimizer update cancelled"
                )
            self.optimizer.step()
            self.scheduler.step()
            self.updates += 1
            total_metrics["last_grad_norm"] = float(norm)
            # Transfer all per-sample diagnostics for this logical batch once.
            host_rows = torch.cat(logical_rows).cpu().tolist()
            batch_kl, batch_weight = 0.0, 0.0
            for item, (loss, value_loss, entropy, clipped, kl, predicted) in zip(
                logical_items, host_rows
            ):
                w = item.weight
                batch_kl += kl * w
                batch_weight += w
                group = (item.character, item.macro["phase"])
                groups[group][0] += kl * w
                groups[group][1] += w
                total_metrics["loss"] += loss * w
                total_metrics["value_mse"] += value_loss * w
                total_metrics["entropy"] += entropy * w
                total_metrics["clip_fraction"] += clipped * w
                total_metrics["kl"] += kl * w
                weights += w
                if mode != "bootstrap":
                    target = item.macro["return"]
                    value_sums["weight"] += w
                    value_sums["target"] += w * target
                    value_sums["target2"] += w * target * target
                    value_sums["predicted"] += w * predicted
                    value_sums["predicted2"] += w * predicted * predicted
                    value_sums["error"] += w * (target - predicted)
                    value_sums["error2"] += w * (target - predicted) ** 2
            if mode == "ppo" and batch_kl / max(batch_weight, 1e-12) > cfg.target_kl:
                total_metrics["early_stop"] = 1
                break
        stream.close()
        if self.vocabulary.unregistered():
            total_metrics["unregistered_names"] = self.vocabulary.unregistered()
        for key in ("loss", "value_mse", "entropy", "clip_fraction", "kl"):
            total_metrics[key] /= max(weights, 1e-12)
        total_metrics["kl_by_character_phase"] = {
            f"{c}/{p}": total / max(w, 1e-12) for (c, p), (total, w) in groups.items()
        }
        total_metrics["updates"] = self.updates
        total_metrics["weight"] = weights
        if value_sums["weight"]:
            w = value_sums["weight"]
            variance = max(
                0.0, value_sums["target2"] / w - (value_sums["target"] / w) ** 2
            )
            error_variance = max(
                0.0, value_sums["error2"] / w - (value_sums["error"] / w) ** 2
            )
            total_metrics["value_rmse"] = math.sqrt(value_sums["error2"] / w)
            total_metrics["value_explained_variance"] = (
                1 - error_variance / variance if variance > 1e-12 else None
            )
            total_metrics["value_mse_over_constant_baseline"] = (
                value_sums["error2"] / w / variance if variance > 1e-12 else None
            )
            total_metrics["return_std"] = math.sqrt(variance)
            total_metrics["value_std"] = math.sqrt(
                max(
                    0.0,
                    value_sums["predicted2"] / w - (value_sums["predicted"] / w) ** 2,
                )
            )
        return dict(total_metrics)

    def bootstrap(self, runs, epochs=1, on_epoch=None, window_shards=8):
        """Behavior cloning. `runs` is a list held in memory, or RunShards, which are
        read `window_shards` at a time: the order of shards and the samples of each
        window are shuffled, and weights stay normalized over the whole data set."""
        if epochs < 1:
            raise ValueError("Bootstrap needs at least one epoch")
        if isinstance(runs, RunShards):
            counts = runs.counts
            if not counts:
                raise ValueError("No accepted behavior-cloning samples")
            metrics = []
            for _ in range(epochs):
                parts = []
                for window in runs.windows(window_shards, shuffle=True):
                    items = samples(window, counts=counts)
                    random.shuffle(items)
                    parts.append(self._optimize(items, "bootstrap"))
                    del items, window
                metrics.append(merge_metrics(parts))
                self.policy_version += 1
                if on_epoch:
                    on_epoch(len(metrics), metrics[-1])
            return metrics
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
            if (
                run["policy_version"] != self.policy_version
                or run["precision"] != self.config.precision
                or run["vocabulary_hash"] != self.vocabulary.digest
            ):
                raise ProtocolError(
                    "Rollout policy/precision/vocabulary differs from learner"
                )
            if run.get("numeric_backend") != numeric_backend(self.model):
                raise ProtocolError(
                    "Rollout numeric backend differs from learner; collect fresh on-policy data"
                )
        largest = 0.0
        tolerance = 0.02 if self.config.precision == "bf16" else 1e-4
        measure(items, self.config)
        micros = microbatches(items, self.config)
        for micro, (log_probs, values, _) in zip(micros, self._replays(micros)):
            current = torch.stack((log_probs, values), dim=1).cpu().tolist()
            for item, (log_prob, value) in zip(micro, current):
                error = abs(log_prob - item.macro["old_log_prob"])
                value_error = abs(value - item.macro["old_value"])
                # BF16 values are cast back to float by the value head,
                # but still lie on its quantization grid. Different batch
                # shapes can round to adjacent values (e.g. 5.71875/5.75).
                # Never apply this exception to policy log probabilities.
                finite = all(math.isfinite(x) for x in (
                    log_prob, value, item.macro["old_log_prob"], item.macro["old_value"]))
                value_matches = finite and (
                    value_error <= tolerance
                    or self.config.precision == "bf16"
                    and _adjacent_bf16_values(item.macro["old_value"], value)
                )
                if not finite or error > tolerance or not value_matches:
                    raise ProtocolError(
                        "Stored old probabilities/values do not replay: "
                        f"run={item.run_id} log_prob_error={error} value_error={value_error} "
                        f"old_log_prob={item.macro['old_log_prob']} log_prob={log_prob} "
                        f"old_value={item.macro['old_value']} value={value}"
                    )
                largest = max(largest, error, value_error)
        return largest

    def ppo(self, runs, demonstrations=None, window_shards=8):
        items = samples(runs, ppo=True, horizon_scale=self.config.horizon_scale)
        if not items:
            raise ValueError("Round contains no stochastic decisions")
        replay_error = self.check_old_policy(runs, items)
        if isinstance(demonstrations, RunShards):
            # Demonstrations are drawn from, not swept: one random window of shards
            # serves the round, weighted as in the whole set.
            window = next(demonstrations.windows(window_shards, shuffle=True), [])
            auxiliary = samples(window, counts=demonstrations.counts)
        else:
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
    def evaluate_bootstrap(self, runs, window_shards=8):
        self.model.eval()
        metrics = defaultdict(lambda: {"nll": 0.0, "branches": 0, "macros": 0})
        windows = runs.windows(window_shards) if isinstance(runs, RunShards) else [runs]
        for window in windows:
            items = samples(window)
            measure(items, self.config)
            micros = microbatches(items, self.config)
            for batch, (log_probs, _, _) in zip(micros, self._replays(micros)):
                for item, log_prob in zip(batch, log_probs.cpu().tolist()):
                    macro = item.macro
                    group = metrics[item.character + "/" + macro["phase"]]
                    group["nll"] -= log_prob
                    group["branches"] += sum(not s["forced"] for s in macro["steps"])
                    group["macros"] += 1
        return {
            key: dict(value, nll_per_branch=value["nll"] / value["branches"])
            for key, value in metrics.items()
        }


def merge_metrics(parts):
    """One epoch's metrics from the windows it was optimized in, weighted by sample weight."""
    total = sum(p["weight"] for p in parts)
    merged = dict(parts[-1], weight=total)
    for key in ("loss", "value_mse", "entropy", "clip_fraction", "kl"):
        merged[key] = sum(p[key] * p["weight"] for p in parts) / max(total, 1e-12)
    groups = defaultdict(list)
    for part in parts:
        for name, value in part["kl_by_character_phase"].items():
            groups[name].append(value)
    merged["kl_by_character_phase"] = {name: sum(v) / len(v) for name, v in groups.items()}
    names = sorted({n for p in parts for n in p.get("unregistered_names", [])})
    if names:
        merged["unregistered_names"] = names
    return merged


def _wilson(wins, n, z=1.96):
    if not n:
        return None
    rate = wins / n
    center = (rate + z * z / (2 * n)) / (1 + z * z / n)
    radius = z * math.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [max(0.0, center - radius), min(1.0, center + radius)]


def evaluate_runs(runs):
    """Win rates over every attempt: a run that did not finish is not a win.

    The rate among finished runs is reported beside it, and `comparable` says
    whether every attempt finished; only then do the two agree and can reports
    be set against each other.
    """
    report = {}
    for character in ("Ironclad", "Silent", "Defect", "Regent", "Necrobinder"):
        group = [r for r in runs if r["character"] == character]
        completed = [r for r in group if r["status"] == "complete"]
        attempts, n = len(group), len(completed)
        wins = sum(r.get("victory", False) for r in completed)
        report[character] = {
            "attempts": attempts,
            "complete": n,
            "wins": wins,
            "win_rate": wins / attempts if attempts else None,
            "wilson_95": _wilson(wins, attempts),
            "completed_win_rate": wins / n if n else None,
            "completion_rate": n / attempts if attempts else None,
            "errors_or_unresolved": attempts - n,
        }

    def mean(key):
        values = [x[key] for x in report.values()]
        return sum(values) / 5 if all(v is not None for v in values) else None

    rates = [x["win_rate"] for x in report.values()]
    return {
        "characters": report,
        "ascensions": sorted({r["ascension"] for r in runs if "ascension" in r}),
        "comparable": bool(runs) and all(r["status"] == "complete" for r in runs),
        "equal_character_win_rate": mean("win_rate"),
        "worst_character_win_rate": min(rates) if all(r is not None for r in rates) else None,
        "equal_character_completed_win_rate": mean("completed_win_rate"),
        "equal_character_completion_rate": mean("completion_rate"),
    }
