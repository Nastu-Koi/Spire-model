"""Single-device Accelerate Bootstrap, value warmup, and complete-run macro-action PPO."""

import math
import random
import time
from collections import defaultdict
from itertools import count

import torch

from .config import TrainConfig
from .data import RunShards, batch_pad_size, samples
from .loader import batches, stream
from .optim import BootstrapSchedule, build_optimizer
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


def _value_metrics(sums):
    if not sums["weight"]:
        return {}
    w = sums["weight"]
    variance = max(0.0, sums["target2"] / w - (sums["target"] / w) ** 2)
    error_variance = max(0.0, sums["error2"] / w - (sums["error"] / w) ** 2)
    return {
        "value_rmse": math.sqrt(sums["error2"] / w),
        "value_explained_variance": 1 - error_variance / variance if variance > 1e-12 else None,
        "value_mse_over_constant_baseline": sums["error2"] / w / variance if variance > 1e-12 else None,
        "return_std": math.sqrt(variance),
        "value_std": math.sqrt(max(0.0, sums["predicted2"] / w - (sums["predicted"] / w) ** 2)),
    }


class Learner:
    def __init__(self, model, vocabulary, config=None, *, policy_version=0, decay_updates=0):
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
        cfg = self.config
        self.scheduler = BootstrapSchedule(
            self.optimizer.optimizer, cfg.bootstrap_warmup_updates, decay_updates,
            table_final_ratio=(cfg.embedding_lr_final / cfg.embedding_lr
                               if cfg.embedding_lr_final is not None else 1.0),
            table_decay_updates=cfg.embedding_decay_updates,
        )
        self.policy_version, self.updates = policy_version, 0

    def _replay(self, prepared, *, with_accuracy=False):
        """Macro-action (log-prob, value, entropy) rows of a prepared batch."""
        # Autocast ends here: the caller runs outside it.
        with precision_context(self.model, self.config.precision):
            return self.model.replay(prepared.to(self.model.device), with_accuracy=with_accuracy)

    def _optimize(self, logicals, mode, demonstrations=None, *, bootstrap_samples=None, on_update=None):
        try:
            return self._optimize_batches(logicals, mode, demonstrations,
                                          bootstrap_samples=bootstrap_samples, on_update=on_update)
        finally:
            if hasattr(logicals, "close"):
                logicals.close()

    def _optimize_batches(self, logicals, mode, demonstrations=None, *, bootstrap_samples=None, on_update=None):
        """One pass over logical batches as the loader yields them: an update for each.
        A BC pass ends early once the schedule's decay is spent."""
        cfg = self.config
        if mode == "bootstrap" and (bootstrap_samples is None or bootstrap_samples < 1):
            raise ValueError("BC updates require the full training split's macro count")
        # The same scale applies to every microbatch, window and the final short batch.
        loss_scale = bootstrap_samples / cfg.logical_batch_size if mode == "bootstrap" else 1.0
        self.model.train()
        total_metrics = defaultdict(float)
        weights = 0.0
        groups = defaultdict(lambda: [0.0, 0.0])
        value_sums = defaultdict(float)
        accuracy_correct = accuracy_decisions = 0
        grad_norms = []
        gradient_clipped = 0
        update_stats = defaultdict(lambda: {"samples": 0, "delta_l2_sum": 0.0,
                                            "relative_sum": 0.0, "relative_max": 0.0})
        last_rates = {}
        update_started = time.monotonic()
        for logical in logicals:
            if mode == "bootstrap" and self.scheduler.exhausted:
                total_metrics["decay_exhausted"] = 1
                break
            self.optimizer.zero_grad(set_to_none=True)
            logical_rows = []
            logical_items = []
            logical_accuracy = []
            for micro, prepared in logical:
                if mode == "bootstrap":
                    log_prob, value, entropy, accuracy = self._replay(prepared, with_accuracy=True)
                    logical_accuracy.append(accuracy.sum(0))
                else:
                    log_prob, value, entropy = self._replay(prepared)
                with precision_context(self.model, cfg.precision):
                    column = lambda values: log_prob.new_tensor(values, dtype=torch.float32)
                    if mode == "bootstrap":
                        loss = -log_prob
                        kl = clipped = value_loss = predicted = torch.zeros_like(loss)
                    else:
                        actor, kl, clipped = ppo_objective(
                            log_prob,
                            column([item.macro["old_log_prob"] for item in micro]),
                            column([item.macro["advantage"] for item in micro]),
                            cfg.clip_ratio,
                        )
                        value_loss = (
                            value.float() - column([item.macro["return"] for item in micro])
                        ) ** 2
                        loss = (
                            value_loss
                            if mode == "value"
                            else actor
                            + cfg.value_coef * value_loss
                            - cfg.entropy_coef * entropy
                        )
                        predicted = value.float()
                    weighted_loss = (column([item.weight for item in micro]) * loss).sum() * loss_scale
                    rows = torch.stack(
                        [
                            x.float()
                            for x in (loss, value_loss, entropy, clipped, kl, predicted)
                        ],
                        dim=1,
                    )
                # A non-finite loss gives non-finite gradients, which the norm check
                # below cancels; checking the loss here would stall the GPU queue.
                self.accelerator.backward(weighted_loss)
                logical_rows.append(rows.detach())
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
            grad_norms.append(float(norm))
            gradient_clipped += float(norm) > cfg.max_grad_norm
            self.scheduler.prepare(mode)
            last_rates = {g["name"]: g["lr"] for g in self.optimizer.param_groups}
            snapshots = []
            if self.updates % cfg.update_stats_every == 0:
                snapshots = [(group["name"], [(p, p.detach().clone()) for p in group["params"]
                                              if p.grad is not None])
                             for group in self.optimizer.param_groups]
            self.optimizer.step()
            parameter_changes = {}
            for name, pairs in snapshots:
                if not pairs:
                    continue
                delta = torch.stack([(p.detach() - before).float().square().sum() for p, before in pairs]).sum().sqrt()
                reference = torch.stack([before.float().square().sum() for _, before in pairs]).sum().sqrt()
                relative = float(delta / reference.clamp_min(1e-12))
                entry = update_stats[name]
                entry["samples"] += 1
                entry["delta_l2_sum"] += float(delta)
                entry["relative_sum"] += relative
                entry["relative_max"] = max(entry["relative_max"], relative)
                parameter_changes[name] = {"delta_l2": float(delta), "relative_l2": relative}
            self.scheduler.step()
            self.updates += 1
            total_metrics["last_grad_norm"] = float(norm)
            # Transfer all per-sample diagnostics for this logical batch once.
            host_rows = torch.cat(logical_rows).cpu().tolist()
            batch_kl, batch_weight = 0.0, 0.0
            batch_metrics = defaultdict(float)
            batch_values = defaultdict(float)
            for item, (loss, value_loss, entropy, clipped, kl, predicted) in zip(
                logical_items, host_rows
            ):
                w = item.weight
                batch_kl += kl * w
                batch_weight += w
                group = (item.character, item.phase)
                groups[group][0] += kl * w
                groups[group][1] += w
                total_metrics["loss"] += loss * w
                total_metrics["value_mse"] += value_loss * w
                total_metrics["entropy"] += entropy * w
                total_metrics["clip_fraction"] += clipped * w
                total_metrics["kl"] += kl * w
                for key, value in (("loss", loss), ("value_mse", value_loss),
                                   ("entropy", entropy), ("clip_fraction", clipped), ("kl", kl)):
                    batch_metrics[key] += value * w
                weights += w
                if mode != "bootstrap":
                    target = item.macro["return"]
                    for sums in (value_sums, batch_values):
                        sums["weight"] += w
                        sums["target"] += w * target
                        sums["target2"] += w * target * target
                        sums["predicted"] += w * predicted
                        sums["predicted2"] += w * predicted * predicted
                        sums["error"] += w * (target - predicted)
                        sums["error2"] += w * (target - predicted) ** 2
            early_stop = mode == "ppo" and batch_kl / max(batch_weight, 1e-12) > cfg.target_kl
            accuracy_metrics = {}
            if mode == "bootstrap":
                correct, decisions = torch.stack(logical_accuracy).sum(0).cpu().tolist()
                accuracy_correct += correct
                accuracy_decisions += decisions
                accuracy_metrics = {
                    "accuracy": correct / decisions if decisions else None,
                    "accuracy_correct": correct, "accuracy_decisions": decisions,
                }
            if on_update:
                on_update({
                    **{key: value / max(batch_weight, 1e-12) for key, value in batch_metrics.items()},
                    **_value_metrics(batch_values),
                    **accuracy_metrics,
                    "updates": self.updates, "samples": len(logical_items), "weight": batch_weight,
                    "last_grad_norm": float(norm), "gradient_clip_fraction": float(float(norm) > cfg.max_grad_norm),
                    "learning_rates": last_rates, "parameter_updates": parameter_changes,
                    "loss_scale": loss_scale, "early_stop": int(early_stop),
                    "duration_seconds": time.monotonic() - update_started,
                })
            update_started = time.monotonic()
            if early_stop:
                total_metrics["early_stop"] = 1
                break
        if self.vocabulary.unregistered():
            total_metrics["unregistered_names"] = self.vocabulary.unregistered()
        for key in ("loss", "value_mse", "entropy", "clip_fraction", "kl"):
            total_metrics[key] /= max(weights, 1e-12)
        total_metrics["kl_by_character_phase"] = {
            f"{c}/{p}": total / max(w, 1e-12) for (c, p), (total, w) in groups.items()
        }
        total_metrics["updates"] = self.updates
        total_metrics["weight"] = weights
        total_metrics["loss_scale"] = loss_scale
        total_metrics["learning_rates"] = last_rates
        total_metrics["gradient_clip_fraction"] = gradient_clipped / max(len(grad_norms), 1)
        if grad_norms:
            norms = torch.tensor(grad_norms)
            total_metrics["grad_norm"] = {"mean": float(norms.mean()), "max": max(grad_norms),
                                         "p50": float(norms.quantile(.5)), "p95": float(norms.quantile(.95))}
        total_metrics["parameter_updates"] = {
            name: {"samples": s["samples"], "mean_delta_l2": s["delta_l2_sum"] / s["samples"],
                   "mean_relative_l2": s["relative_sum"] / s["samples"],
                   "max_relative_l2": s["relative_max"]}
            for name, s in update_stats.items()
        }
        total_metrics.update(_value_metrics(value_sums))
        if mode == "bootstrap":
            total_metrics.update(
                accuracy=accuracy_correct / accuracy_decisions if accuracy_decisions else None,
                accuracy_correct=accuracy_correct, accuracy_decisions=accuracy_decisions,
            )
        if mode == "bootstrap" and self.scheduler.decay_updates:
            total_metrics["decay"] = {"updates": self.scheduler.decay_updates,
                                      "completed": self.scheduler.decayed_updates}
        return dict(total_metrics)

    def bootstrap(self, runs, epochs=1, on_epoch=None, window_shards=8, *, on_update=None):
        """Behavior cloning. `runs` is a list held in memory, or RunShards, which are
        read `window_shards` at a time: the order of shards and the samples of each
        window are shuffled, and weights stay normalized over the whole data set.
        A schedule with a decay stops once it is spent."""
        if epochs is not None and epochs < 1:
            raise ValueError("Bootstrap needs at least one epoch")
        if self.scheduler.exhausted:
            raise ValueError("The decay has finished; this checkpoint is final")
        if isinstance(runs, RunShards):
            counts = runs.counts
            if not counts:
                raise ValueError("No accepted behavior-cloning samples")
            total = sum(counts.values())

            def logicals():
                return stream(runs, self.vocabulary, self.config, window_shards=window_shards,
                              shuffle=True, counts=counts)
        else:
            items = samples(runs)
            if not items:
                raise ValueError("No accepted behavior-cloning samples")
            total = len(items)

            def logicals():
                random.shuffle(items)
                return batches(items, self.vocabulary, self.config)

        metrics = []
        for epoch in count(1):
            metrics.append(self._optimize(
                logicals(), "bootstrap", bootstrap_samples=total,
                on_update=(lambda metrics: on_update(metrics, epoch=epoch, stage="bootstrap"))
                if on_update else None))
            self.policy_version += 1
            if on_epoch:
                on_epoch(epoch, metrics[-1])
            if self.scheduler.exhausted or (epochs is not None and epoch >= epochs):
                break
            if epochs is None:
                metrics.clear()
        return metrics

    @torch.no_grad()
    def check_old_policy(self, runs, items):
        self.model.eval()
        for run in runs:
            if run["ascension"] != self.config.ascension:
                raise ProtocolError("Rollout ascension differs from learner")
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
        # The FP32 entity encoder replays a decision to about 1e-3 of its sampled
        # log-probability; a macro-action sums as many decisions as it branches.
        tolerance = 0.02 if self.config.precision == "bf16" else 1e-4
        for micro, prepared in (
            part for logical in batches(items, self.vocabulary, self.config) for part in logical
        ):
            log_probs, values, _ = self._replay(prepared)
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
                if not finite or error > tolerance * max(1, item.branches) or not value_matches:
                    raise ProtocolError(
                        "Stored old probabilities/values do not replay: "
                        f"run={item.run_id} log_prob_error={error} value_error={value_error} "
                        f"old_log_prob={item.macro['old_log_prob']} log_prob={log_prob} "
                        f"old_value={item.macro['old_value']} value={value}"
                    )
                largest = max(largest, error, value_error)
        return largest

    def ppo(self, runs, demonstrations=None, window_shards=8, *, on_update=None):
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
        for epoch in range(1, self.config.ppo_epochs + 1):
            random.shuffle(items)
            result = self._optimize(batches(items, self.vocabulary, self.config), "ppo", auxiliary,
                                    on_update=(lambda metrics: on_update(
                                        dict(metrics, old_replay_max_error=replay_error), epoch=epoch, stage="ppo"))
                                    if on_update else None)
            result["old_replay_max_error"] = replay_error
            metrics.append(result)
            if result.get("early_stop"):
                break
        self.policy_version += 1
        return metrics

    def value_warmup(self, runs, epochs=1, *, on_update=None):
        if epochs < 1:
            raise ValueError("Value warmup needs at least one epoch")
        items = samples(runs, ppo=True, horizon_scale=self.config.horizon_scale)
        replay_error = self.check_old_policy(runs, items)
        trainable = {name: p.requires_grad for name, p in self.model.named_parameters()}
        try:
            for name, p in self.model.named_parameters():
                p.requires_grad_(name.startswith("value."))
            metrics = [
                self._optimize(batches(items, self.vocabulary, self.config), "value",
                               on_update=(lambda metrics: on_update(
                                   dict(metrics, old_replay_max_error=replay_error), epoch=epoch, stage="value"))
                               if on_update else None)
                for epoch in range(1, epochs + 1)
            ]
        finally:
            for name, p in self.model.named_parameters():
                p.requires_grad_(trainable[name])
        # The next round must capture fresh old values, despite unchanged actor weights.
        self.policy_version += 1
        return metrics

    @torch.no_grad()
    def evaluate_bootstrap(self, runs, window_shards=8, *, include_a0=False):
        self.model.eval()
        metrics = defaultdict(lambda: {"nll": 0.0, "branches": 0, "macros": 0})
        if isinstance(runs, RunShards):
            logicals = stream(runs, self.vocabulary, self.config,
                              window_shards=window_shards, shuffle=False)
        else:
            logicals = batches(samples(runs), self.vocabulary, self.config)
        for logical in logicals:
            for micro, prepared in logical:
                log_probs, _, _ = self._replay(prepared)
                for item, log_prob in zip(micro, log_probs.cpu().tolist()):
                    names = [item.character + "/" + item.phase]
                    if include_a0 and item.ascension == 0:
                        names.append('A0/' + names[0])
                    for name in names:
                        group = metrics[name]
                        group["nll"] -= log_prob
                        group["branches"] += item.branches
                        group["macros"] += 1
        return {
            key: dict(value, nll_per_branch=value["nll"] / value["branches"])
            for key, value in metrics.items()
        }


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
