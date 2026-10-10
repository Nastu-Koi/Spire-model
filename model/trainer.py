"""Single-device Accelerate Bootstrap, value warmup, and complete-run macro-action PPO."""

import math
import random
import time
from collections import defaultdict, deque
from dataclasses import replace
from itertools import count

import torch

from .config import TrainConfig
from .data import RunShards, batch_pad_size, microbatches, samples
from .loader import batches, stream
from .optim import BootstrapSchedule, build_optimizer
from .protocol import ProtocolError
from .rollout import climbed, numeric_backend, precision_context, stand
from .wire import replay


def ppo_objective(new_log_prob, old_log_prob, advantage, clip_ratio):
    log_ratio = new_log_prob.float() - old_log_prob
    ratio = torch.exp(log_ratio)
    actor = -torch.minimum(
        ratio * advantage, ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * advantage
    )
    kl = (ratio - 1) - log_ratio
    clipped = ((ratio - 1).abs() > clip_ratio).float()
    return actor, kl, clipped


# What the replay of a round may differ by from what its sampling stored, in a decision's
# log-probability and in a macro-action's value: (median, 99th percentile, any single one).
#
# In BF16 one decision scores differently from batch to batch: the FP32 entity encoder
# differs in its last bits with the rows it is computed beside, and the rounding of the
# layers above amplifies that. The noise has a median near 1e-3 and a heavy tail. About
# three decisions in a thousand are over 0.02 and the largest of a round grows with the
# round, so no bound holds for every decision; the bound on a single one only stops a
# decision that is scored as another one. Weights a single optimizer step away already
# move the median and the 99th percentile about ten times further than the noise does.
REPLAY_LIMITS = {"bf16": (0.003, 0.05, 1.0), "no": (1e-4, 1e-4, 1e-4)}
# The macro-actions of a round that are replayed for it. The check judges a
# distribution, and what it is there to catch moves most decisions at once: a sample
# shows that as well as the whole round, in a fraction of the time. The bound on a
# single decision then holds for the sampled ones only.
REPLAY_CHECKED = 2048
# The demonstrations of a PPO update are a fraction of a logical batch: a few workers
# pack them as fast as the updates use them. A round draws them from a window of this
# many shards, again and again: every worker ends up with a copy of the window it
# reads, on top of the round the trainer holds, so the window is kept small.
ANCHOR_WORKERS, ANCHOR_SHARDS = 2, 4


# PPO stops updating on a round once the batches it has lately seen average over the
# KL target. One batch says little: under one policy, batches of 512 macro-actions
# scatter from half their median to more than twice it, and stopping at the first
# one over the target ends the round at about half the movement it allows.
KL_WINDOW = 8


def _spread(errors):
    """(median, 99th percentile, largest) of a round's replay errors."""
    ranked = sorted(errors)
    return ranked[len(ranked) // 2], ranked[int(0.99 * (len(ranked) - 1))], ranked[-1]


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
            ppo_muon_ratio=cfg.ppo_muon_lr / cfg.muon_lr,
        )
        self.policy_version, self.updates = policy_version, 0

    def _replay(self, prepared, *, with_accuracy=False, sampled=False):
        """Macro-action (log-prob, value, entropy) rows of a prepared batch: of the
        distribution rounds are sampled from when the batch is `sampled`, otherwise of
        the one demonstrations are learned with (temperature 1)."""
        # Autocast ends here: the caller runs outside it.
        with precision_context(self.model, self.config.precision):
            return self.model.replay(
                prepared.to(self.model.device), with_accuracy=with_accuracy,
                temperature=self.config.sampling_temperature if sampled else 1.0)

    def _batches(self, items):
        """The logical batches of `items`, each a list of (samples, Replay) parts.

        Where sampling kept the packed decisions of its macro-actions, the parts are
        joined from them here, on the device: no frame is parsed a second time and no
        process is started. Otherwise the loader's workers pack them from their frames.
        """
        if not items or "packed" not in items[0].macro:
            return batches(items, self.vocabulary, self.config)
        size = self.config.logical_batch_size
        return (
            [
                (micro, replay([(item.macro["packed"], item.macro["slots"]) for item in micro],
                               self.model.device, pad_to=batch_pad_size(micro, self.config)))
                for micro in microbatches(items[start:start + size], self.config)
            ]
            for start in range(0, len(items), size)
        )

    def _optimize(self, logicals, mode, demonstrations=None, *, bootstrap_samples=None, on_update=None,
                  recent_kl=None):
        try:
            return self._optimize_batches(logicals, mode, demonstrations, bootstrap_samples=bootstrap_samples,
                                          on_update=on_update, recent_kl=recent_kl)
        finally:
            if hasattr(logicals, "close"):
                logicals.close()

    def _optimize_batches(self, logicals, mode, demonstrations=None, *, bootstrap_samples=None, on_update=None,
                          recent_kl=None):
        """One pass over logical batches as the loader yields them: an update for each.
        A BC pass ends early once the schedule's decay is spent, a PPO pass once the KL
        of the batches in `recent_kl` (the last ones of the round, this pass's added as
        they come) averages over the target."""
        cfg = self.config
        recent_kl = deque(maxlen=KL_WINDOW) if recent_kl is None else recent_kl
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
        demonstration_nlls = []
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
                    log_prob, value, entropy = self._replay(prepared, sampled=True)
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
            demonstration_nll = None
            if demonstrations is not None and cfg.bootstrap_coef and mode == "ppo":
                # The demonstrations' mean NLL, weighed against the round's own loss
                # sample for sample: this update's samples weigh `scale` together.
                parts = next(demonstrations)
                scale = sum(item.weight for item in logical_items)
                total = sum(item.weight for micro, _ in parts for item in micro)
                demonstration_nll = 0.0
                for micro, prepared in parts:
                    log_prob, _, _ = self._replay(prepared)
                    with precision_context(self.model, cfg.precision):
                        share = log_prob.new_tensor([item.weight / total for item in micro],
                                                    dtype=torch.float32)
                        nll = -(share * log_prob.float()).sum()
                    self.accelerator.backward(cfg.bootstrap_coef * scale * nll)
                    demonstration_nll += float(nll.detach())
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
            policy_metrics, early_stop = {}, False
            if mode == "ppo":
                recent_kl.append(batch_kl / max(batch_weight, 1e-12))
                policy_metrics["kl_recent"] = sum(recent_kl) / len(recent_kl)
                early_stop = policy_metrics["kl_recent"] > cfg.target_kl
            if demonstration_nll is not None:
                policy_metrics["demonstration_nll"] = demonstration_nll
                demonstration_nlls.append(demonstration_nll)
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
                    **policy_metrics,
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
        if demonstration_nlls:
            total_metrics["demonstration_nll"] = sum(demonstration_nlls) / len(demonstration_nlls)
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
            if run.get("temperature", 1.0) != self.config.sampling_temperature:
                raise ProtocolError("Rollout sampling temperature differs from learner")
        policy_errors, value_errors, worst = [], [], (-1.0, None)
        checked = random.sample(items, min(len(items), REPLAY_CHECKED))
        for micro, prepared in (part for logical in self._batches(checked) for part in logical):
            log_probs, values, _ = self._replay(prepared, sampled=True)
            current = torch.stack((log_probs, values), dim=1).cpu().tolist()
            for item, (log_prob, value) in zip(micro, current):
                old_log_prob, old_value = item.macro["old_log_prob"], item.macro["old_value"]
                if not all(math.isfinite(x) for x in (log_prob, value, old_log_prob, old_value)):
                    raise ProtocolError(
                        "Stored old probabilities/values do not replay: "
                        f"run={item.run_id} old_log_prob={old_log_prob} log_prob={log_prob} "
                        f"old_value={old_value} value={value}"
                    )
                # A macro-action sums the log-probabilities of its decisions. Dividing
                # the scores by the temperature divides their noise by it too: the
                # bounds are on the noise of the scores.
                policy_errors.append(abs(log_prob - old_log_prob) * self.config.sampling_temperature
                                     / max(1, item.branches))
                value_errors.append(abs(value - old_value))
                worst = max(worst, (max(policy_errors[-1], value_errors[-1]), item.run_id))
        spread = {"log_prob": _spread(policy_errors), "value": _spread(value_errors)}
        limits = REPLAY_LIMITS[self.config.precision]
        if any(found > limit for stats in spread.values() for found, limit in zip(stats, limits)):
            raise ProtocolError(
                "Stored old probabilities/values do not replay: "
                + " ".join(
                    f"{name}_error median={stats[0]:.6f} p99={stats[1]:.6f} max={stats[2]:.6f}"
                    for name, stats in spread.items())
                + f" limits={limits} worst_run={worst[1]}"
            )
        return {
            "old_replay_median_error": spread["log_prob"][0],
            "old_replay_p99_error": spread["log_prob"][1],
            "old_replay_max_error": spread["log_prob"][2],
            "old_value_replay_p99_error": spread["value"][1],
            "old_value_replay_max_error": spread["value"][2],
        }

    def _anchors(self, items):
        """Batches of demonstrations for the updates of a PPO round, without end: `items`
        in a new order each time they have all been used."""
        config = replace(self.config, logical_batch_size=self.config.bootstrap_batch_size,
                         loader_workers=min(self.config.loader_workers, ANCHOR_WORKERS))
        while True:
            random.shuffle(items)
            yield from batches(items, self.vocabulary, config)

    def ppo(self, runs, demonstrations=None, window_shards=ANCHOR_SHARDS, *, on_update=None):
        cfg = self.config
        items = samples(runs, ppo=True, horizon_scale=cfg.horizon_scale)
        if not items:
            raise ValueError("Round contains no stochastic decisions")
        replay_error = self.check_old_policy(runs, items)
        auxiliary = None
        if demonstrations and cfg.bootstrap_coef:
            if isinstance(demonstrations, RunShards):
                # Demonstrations are drawn from, not swept: one random window of shards
                # serves the round, weighted as in the whole set.
                window = next(demonstrations.windows(window_shards, shuffle=True), [])
                auxiliary = samples(window, counts=demonstrations.counts)
            else:
                auxiliary = samples(demonstrations)
        anchors = self._anchors(auxiliary) if auxiliary else None
        metrics = []
        # The policy keeps moving from one pass to the next: so does what is known of it.
        recent_kl = deque(maxlen=KL_WINDOW)
        planned = cfg.ppo_epochs * -(-len(items) // cfg.logical_batch_size)
        started, step = self.updates, self.scheduler.ppo_step
        try:
            for epoch in range(1, cfg.ppo_epochs + 1):
                random.shuffle(items)
                result = self._optimize(self._batches(items), "ppo", anchors,
                                        recent_kl=recent_kl,
                                        on_update=(lambda metrics: on_update(
                                            dict(metrics, **replay_error), epoch=epoch, stage="ppo"))
                                        if on_update else None)
                result.update(replay_error, ppo_step=step)
                metrics.append(result)
                if result.get("early_stop"):
                    break
        finally:
            if anchors:
                anchors.close()
        # The next round's step: this one's budget should have lasted its updates.
        self.scheduler.pace(
            (self.updates - started) / planned if metrics[-1].get("early_stop") else 1.0,
            sum(recent_kl) / len(recent_kl) / cfg.target_kl)
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
                self._optimize(self._batches(items), "value",
                               on_update=(lambda metrics: on_update(
                                   dict(metrics, **replay_error), epoch=epoch, stage="value"))
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


def climb(run):
    """Where a run stood, read from the decisions it recorded: the furthest floor of
    each act it entered, and how many floors those acts have. Both are empty when the
    frames do not say. A run that kept no frames brings the answer with it."""
    if "climb" in run:
        return run["climb"]
    frames = [step["frame"] for macro in run.get("macros", ()) for step in macro["steps"]]
    frames += [action["frame"] for action in run.get("environment_actions", ())]
    return climbed(map(stand, frames))


def evaluate_runs(runs):
    """Win rates over every attempt: a run that did not finish is not a win.

    The rate among finished runs is reported beside it, and `comparable` says
    whether every attempt finished; only then do the two agree and can reports
    be set against each other.

    Beside the wins stands how far the finished runs got: the acts they passed,
    the reward they were paid, and the floor each defeat stopped at, counted across
    acts. A win is far rarer than a floor climbed, so these move before it does.

    `runs` is read once and nothing of a run is kept but what it adds to the report:
    an evaluation is more runs than memory holds, so they are passed one at a time
    from where they were written.
    """
    tallies = {
        character: {"attempts": 0, "complete": 0, "wins": 0, "passed": [0, 0, 0],
                    "scores": [], "floors": [], "deaths": {}}
        for character in ("Ironclad", "Silent", "Defect", "Regent", "Necrobinder")
    }
    act_floors, ascensions, attempts, unfinished = {}, set(), 0, 0
    for run in runs:
        attempts += 1
        unfinished += run["status"] != "complete"
        if "ascension" in run:
            ascensions.add(run["ascension"])
        tally = tallies.get(run["character"])
        if tally is None:
            continue
        tally["attempts"] += 1
        if run["status"] != "complete":
            continue
        tally["complete"] += 1
        tally["wins"] += run.get("victory", False)
        ledger = run.get("ledger", {})
        # The last act is passed by winning the run, whichever event confirmed it.
        for act in set(ledger.get("bosses", ())) | ({3} if run.get("victory") else set()):
            tally["passed"][act - 1] += 1
        if "components" in ledger:
            tally["scores"].append(sum(ledger["components"].values()))
        reached, lengths = climb(run)
        for act, length in lengths.items():
            act_floors[act] = max(act_floors.get(act, 0), length)
        if reached:
            floor = sum(reached.values())
            tally["floors"].append(floor)
            if not run.get("victory"):
                tally["deaths"][floor] = tally["deaths"].get(floor, 0) + 1
    report = {}
    for character, tally in tallies.items():
        total, n, wins = tally["attempts"], tally["complete"], tally["wins"]
        scores, floors, deaths = tally["scores"], tally["floors"], tally["deaths"]
        report[character] = {
            "attempts": total,
            "complete": n,
            "wins": wins,
            "win_rate": wins / total if total else None,
            "wilson_95": _wilson(wins, total),
            "completed_win_rate": wins / n if n else None,
            "completion_rate": n / total if total else None,
            "errors_or_unresolved": total - n,
            "acts_passed": tally["passed"],
            "act_pass_rates": [count / total for count in tally["passed"]] if total else None,
            "mean_score": sum(scores) / len(scores) if scores else None,
            "mean_floor": sum(floors) / len(floors) if floors else None,
            # [floor, runs] for the finished runs that were lost, floors counted across acts.
            "death_floors": [[floor, deaths[floor]] for floor in sorted(deaths)],
        }

    def mean(key):
        values = [x[key] for x in report.values()]
        return sum(values) / 5 if all(v is not None for v in values) else None

    rates = [x["win_rate"] for x in report.values()]
    passes = [x["act_pass_rates"] for x in report.values()]
    return {
        "characters": report,
        "ascensions": sorted(ascensions),
        "comparable": bool(attempts) and not unfinished,
        "equal_character_win_rate": mean("win_rate"),
        "worst_character_win_rate": min(rates) if all(r is not None for r in rates) else None,
        "equal_character_completed_win_rate": mean("completed_win_rate"),
        "equal_character_completion_rate": mean("completion_rate"),
        "equal_character_act_pass_rates": (
            [sum(act) / 5 for act in zip(*passes)] if all(p is not None for p in passes) else None
        ),
        "equal_character_mean_score": mean("mean_score"),
        "equal_character_mean_floor": mean("mean_floor"),
        # Floors of acts 1, 2, ... as far as any finished run entered them.
        "act_floors": [act_floors.get(act) for act in range(1, max(act_floors, default=0) + 1)],
    }
