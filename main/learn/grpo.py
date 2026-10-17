"""Group-relative policy optimization of act groups: a clipped ratio and a KL to a reference.

What a round of act groups (`main/play/groups.py`) teaches, with the advantages of its
groups (`main/learn/groups.py`) and the batches of its waves (`main/learn/batches.py`).
A sampled macro-action i is learned with

    L_i = −min(r_i A_i, clip(r_i, 1 − ε, 1 + ε) A_i) + κ · KL_i

- **The policy is the tempered one.** Every macro-action was drawn at its temperature
  T_i from π_T ∝ π^(1/T) (`Sample.temperature`). The most likely action, which greedy
  evaluation takes, is the same at every temperature, so π_T is what is optimized:
  r_i = π_θ,T(a_i) / π_old,T(a_i) from the log-probability the macro-action was drawn
  with (`Sample.old_log_prob`). The update is then on-policy, with no correction for
  the temperature.
- **No signal, no step.** A macro-action with A_i = 0 has no policy gradient, whatever
  it was: a group whose segments all ended alike does not pull the policy toward what
  it already does, as weighted behavior cloning would.
- **A trust region.** Once a macro-action's ratio is outside [1 − ε, 1 + ε] in the
  direction its advantage pushes, it stops pushing.
- **An anchor.** KL_i is the sum over the macro-action's decisions of
  KL(π_θ,T ‖ π_ref,T) over the legal actions, exact, with π_ref a fixed reference
  (the iteration's starting checkpoint) run on the same batch without gradients. It
  keeps the drift of a policy learning from its own samples in check.
- **No division by the group's spread.** Returns are ±1 and groups are small: a group
  of one win in four would be magnified.

The weights are the waves' share weights times each wave's share of the round's
decisions (`batches`); the trainer sums Σ w_i L_i on the scale of behavior cloning
(`trainer.Learner.fit`).
"""

from dataclasses import replace

import torch

from main.data.shards import extent
from main.env.protocol import ProtocolError
from main.learn import batches
from main.play.rollout import precision_context

CLIP = 0.2
KL = 0.01
# A log-ratio beyond this is numerically the same as the bound: exp(20) is far outside any clip.
LOG_RATIO_BOUND = 20.0


class Clipped:
    """The clipped-ratio objective with a KL to `reference`, a frozen policy network
    of the same vocabulary. It is what `trainer.Learner.fit(objective=)` calls for every
    microbatch, and it keeps what it saw of each batch until `take`."""

    def __init__(self, reference, *, clip=CLIP, kl=KL):
        if not 0 < clip < 1:
            raise ValueError("The clip ε is in (0, 1)")
        if kl < 0:
            raise ValueError("The KL coefficient is not negative")
        self.reference, self.clip, self.kl = reference, clip, kl
        reference.eval().requires_grad_(False)
        self._seen = self._empty()
        self._all = self._empty()

    @staticmethod
    def _empty():
        return dict(samples=0, ratio=0.0, clipped=0, kl=0.0, zero=0)

    def __call__(self, learner, prepared, micro):
        """(loss, entropy at temperature 1, accuracy counts) of every macro-action of a
        microbatch, the loss differentiable."""
        missing = [item for item in micro if None in (item.temperature, item.old_log_prob, item.advantage)]
        if missing:
            item = missing[0]
            raise ProtocolError(f"Sampled decision {(item.run_id, item.index)} has no temperature, log-probability "
                                "or advantage; a clipped update needs all three")
        model = learner.model
        prepared = prepared.to(model.device)
        temperatures = torch.tensor([item.temperature for item in micro], dtype=torch.float32, device=model.device)
        with precision_context(model, learner.config.precision):
            with torch.no_grad():
                reference = self.reference.decision_log_probs(prepared, temperatures)
            log_prob, entropy, accuracy, kl = model.replay(
                prepared, with_accuracy=True, temperatures=temperatures, reference=reference)
        old = log_prob.new_tensor([item.old_log_prob for item in micro], dtype=torch.float32)
        advantage = log_prob.new_tensor([item.advantage for item in micro], dtype=torch.float32)
        ratio = (log_prob.float() - old).clamp(-LOG_RATIO_BOUND, LOG_RATIO_BOUND).exp()
        bounded = ratio.clamp(1 - self.clip, 1 + self.clip)
        surrogate = torch.minimum(ratio * advantage, bounded * advantage)
        loss = -surrogate + self.kl * kl.float()
        with torch.no_grad():
            # Clipped: the bound is the smaller term, so the ratio no longer moves it.
            clipped = int(((bounded * advantage < ratio * advantage) & (advantage != 0)).sum())
            values = dict(samples=len(micro), ratio=float(ratio.sum()), clipped=clipped,
                          kl=float(kl.float().sum()), zero=int((advantage == 0).sum()))
        for seen in (self._seen, self._all):
            for key, value in values.items():
                seen[key] += value
        return loss, entropy, accuracy

    @staticmethod
    def _means(seen):
        count = max(seen["samples"], 1)
        return dict(ratio_mean=seen["ratio"] / count, clip_fraction=seen["clipped"] / count,
                    kl=seen["kl"] / count, zero_share=seen["zero"] / count)

    def take(self):
        """What was seen since the last `take`, as means over its macro-actions."""
        seen, self._seen = self._seen, self._empty()
        return self._means(seen)

    def summary(self):
        """The same over everything seen."""
        return dict(self._means(self._all), samples=self._all["samples"])


def _priced(table, source, scale):
    """A stream's batches with every sample's advantage, reported as `source`, weights times `scale`."""
    def weigh(batch):
        try:
            priced = [([replace(item, source=source, weight=item.weight * scale,
                                advantage=table[item.run_id, item.index]) for item in micro], prepared)
                      for micro, prepared in batch]
        except KeyError as missing:
            raise ProtocolError(f"Sampled decision {missing.args[0]} has no advantage; compute the "
                                "advantages of the round first") from None
        advantages = [item.advantage for micro, _ in priced for item in micro]
        return priced, dict(samples=len(advantages), zero=sum(a == 0 for a in advantages),
                            advantage_abs=sum(map(abs, advantages)))
    return weigh


def group_update(learner, waves, reference, *, clip=CLIP, kl=KL, updates=None, window_shards=8, on_update=None,
                 weight_cap=batches.WEIGHT_CAP, shares=None, phase_weights="group"):
    """One pass over a round of act groups with the clipped objective (`Clipped`): every
    logical batch draws from each wave's segments in proportion to `shares` (each
    wave's share of the round's decisions by default, `batches.wave_shares`) at the
    wave's capped share weights (`batches.buffer_weights`, by phase group or even at
    `phase_weights`); `reference` is the frozen policy network the KL is to.
    `waves` maps a wave's name to its directory. `updates` sets the pass
    (`batches.pass_updates` sees every wave about once at the default shares); without
    it the largest wave does. Returns the pass's metrics, with `groups`: what each
    stream drew from and what the objective saw."""
    if not waves:
        raise ValueError("An update of act groups needs at least one wave")
    if reference.vocabulary_digest is not None and reference.vocabulary_digest != learner.vocabulary.digest:
        raise ProtocolError("The reference policy has another vocabulary than the policy it anchors")
    config = learner.config
    data = batches.priced_waves(waves)
    shares, counts, total = batches.streams_of(data, shares, config)
    primary = max(waves, key=lambda name: data[name][0].macros)
    window = max(1, window_shards // len(waves))
    feeders = batches.stream_feeders(config, len(waves))
    streams = {name: (batches.batches(learner, shards,
                                      batches.buffer_weights(shards.source_counts, weight_cap, phase_weights),
                                      counts[name], window, feeders),
                      _priced(table, name, shards.macros / total))
               for name, (shards, table) in data.items()}
    joint = batches.Joint(streams, primary, updates)
    objective = Clipped(reference.model, clip=clip, kl=kl)

    def told(metrics):
        losses = {name: metrics["sources"].get(name, {}).get("loss") for name in streams}
        done = joint.last
        on_update(dict(
            metrics, **objective.take(),
            **{f"{name}_loss": losses[name] for name in waves},
            **{f"{name}_zero_share": done[name]["zero"] / max(done[name]["samples"], 1) for name in waves}))

    metrics = learner.fit(joint, total=total, on_update=told if on_update else None, objective=objective)

    def wave(name, shards):
        seen = joint.seen[name]
        samples = sum(entry["samples"] for entry in seen)
        return dict(batches=len(seen), samples=samples,
                    zero_share=sum(entry["zero"] for entry in seen) / max(samples, 1),
                    advantage_abs=sum(entry["advantage_abs"] for entry in seen) / max(samples, 1),
                    data=extent(shards))

    metrics["groups"] = dict(
        clip=clip, kl=kl, shares=shares, sizes=counts, decisions=total,
        waves={name: wave(name, shards) for name, (shards, _) in data.items()},
        losses={name: metrics["sources"].get(name, {}).get("loss") for name in streams},
        weight_cap=weight_cap or None, phase_weights=phase_weights, seen=objective.summary())
    return metrics


class Reference:
    """The frozen policy a clipped update's KL is to, with the digest of its vocabulary."""

    def __init__(self, model, vocabulary_digest=None):
        self.model, self.vocabulary_digest = model, vocabulary_digest

    @classmethod
    def load(cls, checkpoint, device):
        from main.network.checkpoint import load_model

        model, vocabulary, _ = load_model(checkpoint, device)
        return cls(model, vocabulary.digest)
