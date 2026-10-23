"""The clipped objective of sampled decisions: a clipped ratio and a KL to a reference.

What a sampled decision teaches, given its advantage, whatever priced it: act groups
(`main/learn/groups.py`) or any other source of advantages. A sampled macro-action i
is learned with

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
- **Explored steps are learned by their score.** A decision drawn by exploration
  (`Sample.explored`) was recorded with the policy's own log-probability, so its ratio
  starts at 1 like any other; but the action is rare, and at a low temperature its
  log-probability moves 1/T times as fast as its logit: within the one pass every
  sample is seen once, so by the time one is seen its ratio is the drift the updates
  before it left, and a window of 1 ± ε would drop most of them on either side (in a
  round of v29, 86% of the explored potion uses ended the pass outside it, drifting
  alike whatever their advantage's sign). Such a macro-action is learned by
  A · (1 + log r), the tangent of r·A at r = 1: the same value and gradient there,
  and after that a gradient of A · ∇log π_θ,T(a), neither silenced by the drift nor
  scaled up by it. Only a log-ratio beyond `EXPLORED_LOG_RATIO` stops it.
- **Forked steps are held on both sides.** A decision taken by its rank
  (`Sample.forked`) pushes only while its ratio is within [1 − ε, 1 + ε], whatever the
  sign: the one-sided clip would let the samples that disagree with its motion pull
  with the full ratio while those that agree are silenced.
- **A trust region.** Once a macro-action's ratio is outside [1 − ε, 1 + ε] in the
  direction its advantage pushes, it stops pushing.
- **An anchor.** KL_i is the sum over the macro-action's decisions of
  KL(π_θ,T ‖ π_ref,T) over the legal actions, exact, with π_ref the reference the
  driver chooses (the run's starting checkpoint, or the policy as it stood when the
  current stretch of `reference_every` iterations began) run on the same batch
  without gradients. Where the reference is the policy that played the round, its
  distributions are those the decisions were drawn from, kept as they were sampled
  (`Choice.log_probs`, `Replay.reference`), and it is not run at all
  (`Reference.recorded`). It keeps the drift of a policy learning from its own
  samples in check.
- **No division by the group's spread.** Returns are ±1 and groups are small: a group
  of one win in four would be magnified.

The weights are the sampled data's share weights (`main/learn/streams.py`); the
trainer sums Σ w_i L_i on the scale of behavior cloning (`trainer.Learner.fit`).
"""

from dataclasses import replace

import torch

from main.encoding.batch import given
from main.env.protocol import ProtocolError
from main.play.rollout import precision_context

CLIP = 0.2
KL = 0.01
# A log-ratio beyond this is numerically the same as the bound: exp(20) is far outside any clip.
LOG_RATIO_BOUND = 20.0
# An explored decision whose log-ratio is beyond this (a factor of 20) no longer pushes:
# a guard against the pathological, far outside the drift a pass leaves (p90 1.85 in v29's round).
EXPLORED_LOG_RATIO = 3.0
# The tally's entries, in order (`Clipped.__call__`, `means`).
TALLY = ("ratio", "clipped", "kl", "zero", "explored", "explored_log_ratio", "explored_up", "explored_up_log_ratio",
         "explored_down", "explored_down_log_ratio", "explored_outside_clip", "explored_beyond")


class Clipped:
    """The clipped-ratio objective with a KL to `reference`, a frozen policy network
    of the same vocabulary, or with None to the distributions the decisions were drawn
    from (`Replay.reference`). It is what `trainer.Learner.fit(objective=)` calls for
    every microbatch, and it keeps what it saw of each batch until `take`."""

    def __init__(self, reference, *, clip=CLIP, kl=KL):
        if not 0 < clip < 1:
            raise ValueError("The clip ε is in (0, 1)")
        if kl < 0:
            raise ValueError("The KL coefficient is not negative")
        self.reference, self.clip, self.kl = reference, clip, kl
        if reference is not None:
            reference.eval().requires_grad_(False)
        # Tallies (`TALLY`) since the last flush and over everything, kept on the
        # device: reading them would make the host wait for the step (`flush`, `means`).
        self._seen = self._all = None
        self._samples = self._samples_all = 0

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
        temperatures = given([item.temperature for item in micro], model.device)
        if self.reference is None and prepared.reference is None:
            raise ProtocolError("A decision kept no distribution it was drawn from; the KL needs a reference policy")
        with precision_context(model, learner.config.precision):
            if self.reference is None:
                reference = prepared.reference
            else:
                with torch.no_grad():
                    reference = self.reference.decision_log_probs(prepared, temperatures)
            log_prob, entropy, accuracy, kl = model.replay(
                prepared, with_accuracy=True, temperatures=temperatures, reference=reference)
        old = given([item.old_log_prob for item in micro], model.device)
        advantage = given([item.advantage for item in micro], model.device)
        explored = given([float(item.explored) for item in micro], model.device) > 0
        forked = given([float(item.forked) for item in micro], model.device) > 0
        log_ratio = (log_prob.float() - old).clamp(-LOG_RATIO_BOUND, LOG_RATIO_BOUND)
        ratio = log_ratio.exp()
        bounded = ratio.clamp(1 - self.clip, 1 + self.clip)
        pushed = ratio * advantage
        inside = (ratio >= 1 - self.clip) & (ratio <= 1 + self.clip)
        # Explored: the score, whatever the drift, up to the guard.
        scored = advantage * (1 + log_ratio)
        guarded = log_ratio.abs() <= EXPLORED_LOG_RATIO
        # Forked: no push from outside the region on either side.
        surrogate = torch.where(explored, torch.where(guarded, scored, scored.detach()),
                                torch.where(forked, torch.where(inside, pushed, pushed.detach()),
                                            torch.minimum(pushed, bounded * advantage)))
        loss = -surrogate + self.kl * kl.float()
        with torch.no_grad():
            # Clipped: the bound is the smaller term, so the ratio no longer moves it; a
            # forked one, outside the region; an explored one, beyond the guard.
            stopped = torch.where(explored, ~guarded, torch.where(forked, ~inside, bounded * advantage < pushed))
            clipped = (stopped & (advantage != 0)).sum()
            # What the drift of a pass does to the explored decisions, by the sign of their advantage.
            up, down = explored & (advantage > 0), explored & (advantage < 0)
            tally = torch.stack([
                ratio.sum(), clipped.to(ratio.dtype), kl.float().sum(), (advantage == 0).sum().to(ratio.dtype),
                explored.sum().to(ratio.dtype), (log_ratio * explored).sum(),
                up.sum().to(ratio.dtype), (log_ratio * up).sum(), down.sum().to(ratio.dtype), (log_ratio * down).sum(),
                (explored & ~inside).sum().to(ratio.dtype), (explored & ~guarded).sum().to(ratio.dtype)]).double()
        self._seen = tally if self._seen is None else self._seen + tally
        self._all = tally if self._all is None else self._all + tally
        self._samples += len(micro)
        self._samples_all += len(micro)
        return loss, entropy, accuracy

    def flush(self):
        """The tally since the last flush and its macro-action count, the tally still on
        the device: what the trainer reads back with the step's own numbers (`means`)."""
        seen, samples = self._seen, self._samples
        self._seen, self._samples = None, 0
        return seen, samples

    @staticmethod
    def means(tally, samples):
        """The report of a tally (on any device; waits for it) over `samples` macro-actions."""
        count = max(samples, 1)
        seen = dict(zip(TALLY, tally.tolist() if tally is not None else [0.0] * len(TALLY)))

        def mean(total, n):  # None where there is nothing to average: the records are strict JSON
            return seen[total] / seen[n] if seen[n] else None

        return dict(ratio_mean=seen["ratio"] / count, clip_fraction=seen["clipped"] / count, kl=seen["kl"] / count,
                    zero_share=seen["zero"] / count, explored=int(seen["explored"]),
                    explored_log_ratio=mean("explored_log_ratio", "explored"),
                    explored_up_log_ratio=mean("explored_up_log_ratio", "explored_up"),
                    explored_down_log_ratio=mean("explored_down_log_ratio", "explored_down"),
                    explored_outside_clip=mean("explored_outside_clip", "explored"),
                    explored_beyond=mean("explored_beyond", "explored"))

    def take(self):
        """What was seen since the last `take` or `flush`, as means over its macro-actions."""
        return self.means(*self.flush())

    def summary(self):
        """The same over everything seen."""
        return dict(self.means(self._all, self._samples_all), samples=self._samples_all)


def with_advantages(table, source, scale):
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


class Reference:
    """The frozen policy a clipped update's KL is to, with the digest of its vocabulary.
    Without a model (`recorded`), the KL is to the distributions the decisions were
    drawn from: the reference is the policy that played the round."""

    def __init__(self, model, vocabulary_digest=None):
        self.model, self.vocabulary_digest = model, vocabulary_digest

    @classmethod
    def recorded(cls):
        return cls(None)

    @classmethod
    def load(cls, checkpoint, device):
        from main.network.checkpoint import load_model

        model, vocabulary, _ = load_model(checkpoint, device)
        return cls(model, vocabulary.digest)
