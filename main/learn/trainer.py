"""Single-device Accelerate behavior cloning: a weighted NLL over macro-actions."""

import math
import random
import time
from collections import defaultdict
from itertools import count
from pathlib import Path

import torch

from main.env.rewards import MilestoneLedger
from main.network.checkpoint import load_model, restore_training, save_checkpoint
from main.network.config import TrainConfig
from main.data.shards import HELDOUT_FRACTION, RunShards, extent, samples, split_held_out
from main.encoding.batch import given
from main.learn.loader import batches, stream
from main.network.optim import BootstrapSchedule, build_optimizer
from main.play.rollout import precision_context


def _by_source(sources):
    """{source: weighted mean loss, total weight, samples} of what was accumulated."""
    return {name: {"loss": loss / max(weight, 1e-12), "weight": weight, "samples": count}
            for name, (loss, weight, count) in sorted(sources.items())}


class _Readback:
    """A step's numbers on their way to the host: per-sample (loss, entropy) rows, the
    accuracy counts, the gradient norm and the objective's tally, copied into pinned
    memory behind the step's kernels and read (`wait`) once the next step is launched.
    On the CPU they are simply kept."""

    def __init__(self, rows, accuracy, norm, tally, **info):
        parts = [rows.flatten(), accuracy.flatten(), norm.reshape(1)] + ([tally.flatten()] if tally is not None else [])
        packed = torch.cat([part.detach().double() for part in parts])
        self.sizes = [part.numel() for part in parts]
        self.width = rows.shape[1] if rows.dim() == 2 else 2
        if packed.device.type == "cuda":
            self.host = torch.empty(packed.shape, dtype=packed.dtype, pin_memory=True)
            self.host.copy_(packed, non_blocking=True)
            self.event = torch.cuda.Event()
            self.event.record()
        else:
            self.host, self.event = packed, None
        self.__dict__.update(info)

    def wait(self):
        """(rows, accuracy counts, norm, tally) once they have arrived."""
        if self.event is not None:
            self.event.synchronize()
        parts = torch.split(self.host, self.sizes)
        rows = parts[0].view(-1, self.width).tolist()
        correct, decisions = (int(round(value)) for value in parts[1].tolist())
        norm = float(parts[2][0])
        tally = parts[3] if len(parts) > 3 else None
        return rows, (correct, decisions), norm, tally


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
        self.checkpoint_metadata = {}  # what the checkpoint it came from recorded

    @classmethod
    def load(cls, checkpoint, device, *, weights_only=False, training=None, decay_updates=0, inference=False,
             rates=None):
        """A checkpoint's policy with its training state, as (learner, manifest).
        `inference` loads it to play only: nothing is resumed, so a checkpoint whose
        training has ended, a finished decay among them, is accepted. `weights_only`
        starts a new run from its weights. `rates` (`LearningRates`) are the rates the
        run trains at instead of the checkpoint's; the optimizer's state and the
        schedule's progress go on."""
        model, vocab, manifest = load_model(checkpoint, device, allow_legacy_encoding=weights_only)
        progress = manifest.get("progress", {})
        saved_training = TrainConfig(**manifest["training"])
        if not weights_only and training is not None:
            # The sampling settings are the profile's, the rates a file's if given; the profile
            # may differ from the checkpoint in nothing else.
            training = saved_training.with_sampling(training, rates=rates is not None)
        if rates is not None:
            training = (training or saved_training).with_rates(rates)
        if weights_only:
            vocab = vocab.expanded()
        learner = cls(model, vocab, training or saved_training,
                      policy_version=0 if weights_only else progress.get("policy_version", 0),
                      decay_updates=decay_updates)
        if weights_only:
            learner.checkpoint_metadata = {
                "training_seeds": progress.get("training_seeds", []),
                "initialization": "weights-only",
                "source_checkpoint": str(Path(checkpoint).resolve()),
                "source_updates": progress.get("updates", 0),
            }
            return learner, manifest
        if inference:
            learner.updates = progress.get("updates", 0)
        else:
            if "optimizer" not in manifest:
                raise ValueError("Checkpoint has no optimizer state; use --weights-only")
            progress = restore_training(checkpoint, learner.optimizer, learner.scheduler)
            learner.updates = progress.get("updates", 0)
        learner.checkpoint_metadata = {
            k: v for k, v in manifest["progress"].items() if k not in {"policy_version", "updates"}
        }
        return learner, manifest

    def save(self, path, extra=None):
        """Write the policy with its optimizer and schedule to `path`, under the metadata
        it was loaded with and `extra` on top; the training seeds of both accumulate.
        A `current` is replaced in place."""
        metadata = {**self.checkpoint_metadata, **(extra or {})}
        metadata["training_seeds"] = sorted(
            set(self.checkpoint_metadata.get("training_seeds", []))
            | set(metadata.get("training_seeds", []))
        )
        save_checkpoint(
            path,
            self.model,
            self.vocabulary,
            self.optimizer,
            self.scheduler,
            overwrite=Path(path).name == "current",
            training=self.config,
            progress={
                **metadata,
                "reward_version": MilestoneLedger().version,
                "policy_version": self.policy_version,
                "updates": self.updates,
            },
        )
        self.checkpoint_metadata = metadata

    def _replay(self, prepared, *, with_accuracy=False):
        """Macro-action (log-prob, entropy) rows of a prepared batch, at temperature 1:
        the distribution demonstrations are learned with."""
        # Autocast ends here: the caller runs outside it.
        with precision_context(self.model, self.config.precision):
            return self.model.replay(prepared.to(self.model.device), with_accuracy=with_accuracy)

    def _optimize(self, logicals, *, total, on_update=None, objective=None):
        try:
            return self._optimize_batches(logicals, total, on_update, objective)
        finally:
            if hasattr(logicals, "close"):
                logicals.close()

    def _optimize_batches(self, logicals, total, on_update, objective=None):
        """One pass over logical batches as the loader yields them: an update for each,
        ending early once the schedule's decay is spent. `total` is the macro count of
        the whole data set: the scale of every batch's loss. Each macro-action's loss is
        its negative log-likelihood, or what `objective` makes of the microbatch
        (`fit`).

        The host never waits for a step: its numbers (per-sample rows, accuracy counts,
        gradient norm, the objective's tally) are queued to pinned memory behind its
        kernels (`_Readback`) and read once the next step is launched, so that the CPU
        launching kernels and the GPU running them overlap. Non-finite gradients are
        therefore seen one step late: the pass fails there, and the policy on disk is
        the last archive's."""
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
        # Weighted loss, weight and samples of every source: what a report of mixed
        # sources tells apart. The update itself sums by weight alone.
        sources = defaultdict(lambda: [0.0, 0.0, 0])
        last_rates = {}
        waited_seconds = 0.0  # blocked on the loader for the next logical batch
        duration_seconds = 0.0
        tallies = objective is not None and hasattr(objective, "flush")
        pending = []  # steps whose numbers are on their way to the host

        def finish(step):
            """Account a step whose numbers have arrived: totals, then `on_update`."""
            nonlocal weights, accuracy_correct, accuracy_decisions, gradient_clipped
            host_rows, (correct, decisions), norm, tally = step.wait()
            if not math.isfinite(norm):
                raise FloatingPointError(
                    f"Non-finite gradients in update {step.updates}; the pass cannot go on"
                )
            grad_norms.append(norm)
            gradient_clipped += norm > cfg.max_grad_norm
            total_metrics["last_grad_norm"] = norm
            batch_weight = 0.0
            batch_metrics = defaultdict(float)
            batch_sources = defaultdict(lambda: [0.0, 0.0, 0])
            for item, (loss, entropy) in zip(step.items, host_rows):
                w = item.weight
                batch_weight += w
                for key, value in (("loss", loss), ("entropy", entropy)):
                    total_metrics[key] += value * w
                    batch_metrics[key] += value * w
                weights += w
                for entry in (batch_sources[item.source or "unknown"], sources[item.source or "unknown"]):
                    entry[0] += loss * w
                    entry[1] += w
                    entry[2] += 1
            accuracy_correct += correct
            accuracy_decisions += decisions
            if on_update:
                on_update({
                    **{key: value / max(batch_weight, 1e-12) for key, value in batch_metrics.items()},
                    **(objective.means(tally, step.samples) if tallies else {}),
                    "accuracy": correct / decisions if decisions else None,
                    "accuracy_correct": correct, "accuracy_decisions": decisions,
                    "updates": step.updates, "samples": len(step.items), "weight": batch_weight,
                    "last_grad_norm": norm, "gradient_clip_fraction": float(norm > cfg.max_grad_norm),
                    "learning_rates": step.rates, "parameter_updates": step.changes,
                    "loss_scale": loss_scale, "sources": _by_source(batch_sources),
                    "duration_seconds": step.duration, "wait_seconds": step.waited,
                })

        logicals = iter(logicals)
        update_started = time.monotonic()
        while True:
            if self.scheduler.exhausted:
                total_metrics["decay_exhausted"] = 1
                break
            logical = next(logicals, None)
            waited = time.monotonic() - update_started
            if logical is None:
                break
            self.optimizer.zero_grad(set_to_none=True)
            logical_rows = []
            logical_items = []
            logical_accuracy = []
            for micro, prepared in logical:
                if objective is None:
                    log_prob, entropy, accuracy = self._replay(prepared, with_accuracy=True)
                    loss = -log_prob
                else:
                    loss, entropy, accuracy = objective(self, prepared, micro)
                logical_accuracy.append(accuracy.sum(0))
                with precision_context(self.model, cfg.precision):
                    weight = given([item.weight for item in micro], loss.device)
                    weighted_loss = (weight * loss).sum() * loss_scale
                    rows = torch.stack([loss.float(), entropy.float()], dim=1)
                # A non-finite loss gives non-finite gradients, which the norm read back
                # below reports; checking here would stall the GPU queue.
                self.accelerator.backward(weighted_loss)
                logical_rows.append(rows.detach())
                logical_items.extend(micro)
            samples_seen += len(logical_items)
            # The step before has finished on the GPU by now: account it while this
            # one's backward runs, before the update counter moves on.
            while pending:
                finish(pending.pop(0))
            norm = self.accelerator.clip_grad_norm_(
                self.model.parameters(), cfg.max_grad_norm
            )
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
            tally, tallied = objective.flush() if tallies else (None, 0)
            duration = time.monotonic() - update_started  # waiting for the batch included
            waited_seconds += waited
            duration_seconds += duration
            step = _Readback(torch.cat(logical_rows), torch.stack(logical_accuracy).sum(0), norm, tally,
                             items=logical_items, samples=tallied, updates=self.updates, rates=last_rates,
                             changes=parameter_changes, duration=duration, waited=waited)
            pending.append(step)
            update_started = time.monotonic()
        while pending:
            finish(pending.pop(0))
        if self.vocabulary.unregistered():
            total_metrics["unregistered_names"] = self.vocabulary.unregistered()
        for key in ("loss", "entropy"):
            total_metrics[key] /= max(weights, 1e-12)
        total_metrics["updates"] = self.updates
        # The macro-actions the loader delivered to this pass.
        total_metrics["samples"] = samples_seen
        total_metrics["weight"] = weights
        total_metrics["wait_seconds"] = waited_seconds
        total_metrics["duration_seconds"] = duration_seconds
        total_metrics["loss_scale"] = loss_scale
        total_metrics["sources"] = _by_source(sources)
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

    def fit(self, logicals, *, total, on_update=None, objective=None):
        """One pass over logical batches the caller supplies: an update for each, as
        `bootstrap` makes them. `total` is the macro count the samples' weights are
        normalized over, the scale of every batch's loss. A caller that mixes sources
        in a batch has put their weights on that one scale (`main/learn/awr.py`). The
        policy is a new version afterwards.

        `objective(learner, prepared, micro)` replaces the negative log-likelihood: it
        returns the loss, the entropy at temperature 1 and the accuracy counts of every
        macro-action of a microbatch (`main/learn/grpo.py`)."""
        if self.scheduler.exhausted:
            raise ValueError("The decay has finished; this checkpoint is final")
        metrics = self._optimize(logicals, total=total, on_update=on_update, objective=objective)
        self.policy_version += 1
        return metrics

    def _until(self, logicals, limit):
        """The logical batches of `logicals` while the learner has made fewer than
        `limit` updates in all; the stream is closed where it stops."""
        iterator = iter(logicals)
        try:
            while self.updates < limit:
                try:
                    logical = next(iterator)
                except StopIteration:
                    return
                yield logical
        finally:
            if hasattr(logicals, "close"):
                logicals.close()

    def bootstrap(self, runs, epochs=1, on_epoch=None, window_shards=8, *, on_update=None,
                  include_held_out=False, max_updates=None):
        """Behavior cloning. `runs` is a list held in memory, or RunShards, which are
        read `window_shards` at a time: the order of shards and the samples of each
        window are shuffled, and weights stay normalized over the whole training part.

        The held-out seeds (`data.held_out`) are left out unless `include_held_out`:
        the loader never reads their runs. Every epoch's metrics count what was left
        out (`held_out`), what was trained on (`training`) and the macro-actions the
        loader delivered (`samples`), which is the training part's count. A schedule
        with a decay stops once it is spent.

        `max_updates` stops after that many updates of this call, in the middle of an
        epoch if it comes to that: a probe of a setting over a fixed number of updates.
        What was done of the last epoch is reported, and handed to `on_epoch`, as an
        epoch, marked with `update_limit`."""
        if epochs is not None and epochs < 1:
            raise ValueError("Bootstrap needs at least one epoch")
        if max_updates is not None and max_updates < 1:
            raise ValueError("An update limit is at least one update")
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

        limit = None if max_updates is None else self.updates + max_updates
        metrics = []
        for epoch in count(1):
            metrics.append(dict(self._optimize(
                logicals() if limit is None else self._until(logicals(), limit), total=total,
                on_update=(lambda metrics: on_update(metrics, epoch=epoch, stage="bootstrap"))
                if on_update else None), **coverage))
            limited = limit is not None and self.updates >= limit
            if limited:
                metrics[-1]["update_limit"] = max_updates
            self.policy_version += 1
            if on_epoch:
                on_epoch(epoch, metrics[-1])
            if self.scheduler.exhausted or limited or (epochs is not None and epoch >= epochs):
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
