"""Single-device Accelerate behavior cloning: a weighted NLL over macro-actions."""

import random
import time
from collections import defaultdict
from itertools import count

import torch

from .config import TrainConfig
from .data import HELDOUT_FRACTION, RunShards, extent, samples, split_held_out
from .loader import batches, stream
from .optim import BootstrapSchedule, build_optimizer
from .rollout import precision_context


class Learner:
    """One policy with its optimizer and schedule.

    Every update lowers the weighted negative log-likelihood of a logical batch of
    macro-actions. Where a macro-action comes from (a demonstration, a solver label, a
    sampled run) is the sample's `source`; its weight is what the loader gave it from
    the data set's index (`data.samples`). The batch's loss is scaled by the data
    set's macro count over the batch size, so that a short final batch and a split
    into microbatches change nothing of the update.
    """

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
        """Macro-action (log-prob, entropy) rows of a prepared batch, at temperature 1:
        the distribution demonstrations are learned with."""
        # Autocast ends here: the caller runs outside it.
        with precision_context(self.model, self.config.precision):
            return self.model.replay(prepared.to(self.model.device), with_accuracy=with_accuracy)

    def _optimize(self, logicals, *, total, on_update=None):
        try:
            return self._optimize_batches(logicals, total, on_update)
        finally:
            if hasattr(logicals, "close"):
                logicals.close()

    def _optimize_batches(self, logicals, total, on_update):
        """One pass over logical batches as the loader yields them: an update for each,
        ending early once the schedule's decay is spent. `total` is the macro count of
        the whole data set: the scale of every batch's loss."""
        cfg = self.config
        if total is None or total < 1:
            raise ValueError("BC updates require the full training split's macro count")
        # The same scale applies to every microbatch, window and the final short batch.
        loss_scale = total / cfg.logical_batch_size
        self.model.train()
        total_metrics = defaultdict(float)
        weights = 0.0
        accuracy_correct = accuracy_decisions = 0
        samples_seen = 0
        grad_norms = []
        gradient_clipped = 0
        update_stats = defaultdict(lambda: {"samples": 0, "delta_l2_sum": 0.0,
                                            "relative_sum": 0.0, "relative_max": 0.0})
        last_rates = {}
        update_started = time.monotonic()
        for logical in logicals:
            if self.scheduler.exhausted:
                total_metrics["decay_exhausted"] = 1
                break
            self.optimizer.zero_grad(set_to_none=True)
            logical_rows = []
            logical_items = []
            logical_accuracy = []
            for micro, prepared in logical:
                log_prob, entropy, accuracy = self._replay(prepared, with_accuracy=True)
                logical_accuracy.append(accuracy.sum(0))
                with precision_context(self.model, cfg.precision):
                    loss = -log_prob
                    weight = log_prob.new_tensor([item.weight for item in micro], dtype=torch.float32)
                    weighted_loss = (weight * loss).sum() * loss_scale
                    rows = torch.stack([loss.float(), entropy.float()], dim=1)
                # A non-finite loss gives non-finite gradients, which the norm check
                # below cancels; checking the loss here would stall the GPU queue.
                self.accelerator.backward(weighted_loss)
                logical_rows.append(rows.detach())
                logical_items.extend(micro)
            samples_seen += len(logical_items)
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
            self.scheduler.prepare()
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
            batch_weight = 0.0
            batch_metrics = defaultdict(float)
            for item, (loss, entropy) in zip(logical_items, host_rows):
                w = item.weight
                batch_weight += w
                for key, value in (("loss", loss), ("entropy", entropy)):
                    total_metrics[key] += value * w
                    batch_metrics[key] += value * w
                weights += w
            correct, decisions = torch.stack(logical_accuracy).sum(0).cpu().tolist()
            accuracy_correct += correct
            accuracy_decisions += decisions
            if on_update:
                on_update({
                    **{key: value / max(batch_weight, 1e-12) for key, value in batch_metrics.items()},
                    "accuracy": correct / decisions if decisions else None,
                    "accuracy_correct": correct, "accuracy_decisions": decisions,
                    "updates": self.updates, "samples": len(logical_items), "weight": batch_weight,
                    "last_grad_norm": float(norm), "gradient_clip_fraction": float(float(norm) > cfg.max_grad_norm),
                    "learning_rates": last_rates, "parameter_updates": parameter_changes,
                    "loss_scale": loss_scale,
                    "duration_seconds": time.monotonic() - update_started,
                })
            update_started = time.monotonic()
        if self.vocabulary.unregistered():
            total_metrics["unregistered_names"] = self.vocabulary.unregistered()
        for key in ("loss", "entropy"):
            total_metrics[key] /= max(weights, 1e-12)
        total_metrics["updates"] = self.updates
        # The macro-actions the loader delivered to this pass.
        total_metrics["samples"] = samples_seen
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
        total_metrics.update(
            accuracy=accuracy_correct / accuracy_decisions if accuracy_decisions else None,
            accuracy_correct=accuracy_correct, accuracy_decisions=accuracy_decisions,
        )
        if self.scheduler.decay_updates:
            total_metrics["decay"] = {"updates": self.scheduler.decay_updates,
                                      "completed": self.scheduler.decayed_updates}
        return dict(total_metrics)

    def bootstrap(self, runs, epochs=1, on_epoch=None, window_shards=8, *, on_update=None,
                  include_held_out=False):
        """Behavior cloning. `runs` is a list held in memory, or RunShards, which are
        read `window_shards` at a time: the order of shards and the samples of each
        window are shuffled, and weights stay normalized over the whole training part.

        The held-out seeds (`data.held_out`) are left out unless `include_held_out`:
        the loader never reads their runs. Every epoch's metrics count what was left
        out (`held_out`), what was trained on (`training`) and the macro-actions the
        loader delivered (`samples`), which is the training part's count. A schedule
        with a decay stops once it is spent."""
        if epochs is not None and epochs < 1:
            raise ValueError("Bootstrap needs at least one epoch")
        if self.scheduler.exhausted:
            raise ValueError("The decay has finished; this checkpoint is final")
        if include_held_out:
            training = runs
            coverage = {"held_out": {"fraction": 0.0, "groups": 0, "macros": 0, "seeds": 0}}
        else:
            training, held = split_held_out(runs)
            coverage = {"held_out": {"fraction": HELDOUT_FRACTION, **extent(held)}}
        coverage["training"] = extent(training)
        if isinstance(training, RunShards):
            counts = training.counts
            if not counts:
                raise ValueError("No accepted behavior-cloning samples")
            total = sum(counts.values())
            # The weights of the whole training part, by (character, phase, source).
            weights = training.weights

            def logicals():
                return stream(training, self.vocabulary, self.config, window_shards=window_shards,
                              shuffle=True, weights=weights)
        else:
            items = samples(training)
            if not items:
                raise ValueError("No accepted behavior-cloning samples")
            total = len(items)

            def logicals():
                random.shuffle(items)
                return batches(items, self.vocabulary, self.config)

        metrics = []
        for epoch in count(1):
            metrics.append(dict(self._optimize(
                logicals(), total=total,
                on_update=(lambda metrics: on_update(metrics, epoch=epoch, stage="bootstrap"))
                if on_update else None), **coverage))
            self.policy_version += 1
            if on_epoch:
                on_epoch(epoch, metrics[-1])
            if self.scheduler.exhausted or (epochs is not None and epoch >= epochs):
                break
            if epochs is None:
                metrics.clear()
        return metrics

    @torch.no_grad()
    def evaluate_bootstrap(self, runs, window_shards=8, *, include_a0=False):
        """The NLL of held-out macro-actions by character and phase, at temperature 1."""
        self.model.eval()
        metrics = defaultdict(lambda: {"nll": 0.0, "branches": 0, "macros": 0})
        if isinstance(runs, RunShards):
            logicals = stream(runs, self.vocabulary, self.config,
                              window_shards=window_shards, shuffle=False)
        else:
            logicals = batches(samples(runs), self.vocabulary, self.config)
        for logical in logicals:
            for micro, prepared in logical:
                log_probs, _ = self._replay(prepared)
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
