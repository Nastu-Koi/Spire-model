"""Advantage-weighted regression: what the policy learns from its own decisions.

A decision the policy made outside a fight is a supervised sample like any other: its
frame, its full legal candidates and the action taken. What it is worth is its
advantage (`model/advantage.py`): the sample's weight is multiplied by exp(A/β), so
that the policy moves toward what turned out better than expected and away from
nothing; there is no ratio and no need for the sampling policy's probabilities.

The trainer is the weighted NLL trainer of behavior cloning (`trainer.Learner.fit`),
and everything here is on the side that supplies its batches. An update's batch is two
halves: sampled decisions from the run buffer, weighted by advantage, and supervised
samples (demonstrations with the solver's labels) with the weights their index gives
them. Both halves carry half of the update's loss whatever their sizes are.

- **Weights.** In a half-batch of n sampled decisions with share weights g_i (every
  character and phase group of the buffer weighs the same, as the groups of
  demonstrations do, up to a cap on what one decision of a small group may weigh:
  `buffer_weights`), the weight of decision i is g_i · exp(A_i/β) · Σg / Σ(g·exp(A/β)):
  the multipliers are normalized within the batch, so the half's total weight does
  not depend on β.
- **β.** Chosen for each batch as the smallest β in [`low`, `high`] at which the
  effective sample size of the multipliers, (Σw)²/Σw² with w = exp(A/β), is at least
  `share` of n. The effective size grows with β, so a bisection on log β finds it. The
  lower bound keeps a critic's noise from being amplified where advantages are small;
  an infinite β makes every multiplier 1, which is behavior cloning of the policy's
  own decisions.
- **Scale.** The trainer scales every batch's loss by the supervised data's macro
  count over the logical batch size. A supervised half of B/2 samples then weighs
  half of what a batch of B would; the sampled half is put on the same scale by
  multiplying its weights by the buffer's macro count over the supervised data's.
"""

import math
from dataclasses import dataclass, replace

from . import advantage
from .data import RunShards, extent, group_weights, split_held_out
from .loader import stream
from .protocol import ProtocolError

SAMPLED = "rollout"
# The effective sample size a batch's multipliers must keep, as a share of the batch.
ESS_SHARE = 0.3
# The range β is chosen from. An advantage is in units of the ±1 return: at 0.1 a
# decision that raised the predicted chance of winning by five points (A = 0.1) weighs
# e times one that changed nothing, and a critic's noise of a point or two stays
# within a factor of about 1.5. At 100 the largest possible advantage moves a weight
# by a few percent: the weights are those of behavior cloning.
BETA_MIN, BETA_MAX = 0.1, 100.0
# What one sampled decision may weigh before its advantage, in even shares of the
# buffer (1 / its decisions). Demonstrations have tens of thousands of macro-actions in
# their smallest group; a round has phases that occur a few times in a thousand runs,
# and an equal share for such a group would hand a single decision the weight of
# several batches.
WEIGHT_CAP = 10.0


def buffer_weights(source_counts, cap=WEIGHT_CAP):
    """The share weight of a sampled decision of every (character, phase, source) of
    the buffer. Every group weighs the same, as `data.group_weights` has it, unless
    its decisions would then weigh more than `cap` even shares each: so small a group
    keeps `cap` even shares per decision, and the rest of its share goes to the other
    groups, which may bring the next smallest over the cap in turn. Without a cap,
    the plain rule."""
    weights = group_weights(source_counts)
    if not cap:
        return weights
    if cap < 1:
        raise ValueError("The cap is in even shares of the buffer: at least one")
    counts = {key: count for key, count in source_counts.items() if count}
    limit = cap / sum(counts.values())
    capped = set()
    while True:
        over = {key for key, weight in weights.items() if key not in capped and weight > limit * (1 + 1e-9)}
        if not over:
            return weights
        capped |= over
        rest = {key: count for key, count in counts.items() if key not in capped}
        left = 1.0 - limit * sum(counts[key] for key in capped)
        weights = {key: limit for key in capped}
        weights.update({key: left / (len(rest) * count) for key, count in rest.items()})


def multipliers(advantages, beta):
    """exp(A/β) up to a common factor: the largest is 1, so none overflows."""
    if math.isinf(beta):
        return [1.0] * len(advantages)
    top = max(advantages)
    return [math.exp((a - top) / beta) for a in advantages]


def effective_size(advantages, beta):
    """(Σw)²/Σw² of w = exp(A/β): as many equally weighted samples as these amount to."""
    weights = multipliers(advantages, beta)
    return sum(weights) ** 2 / sum(w * w for w in weights)


def tune(advantages, share=ESS_SHARE, low=BETA_MIN, high=BETA_MAX, steps=40):
    """The smallest β in [low, high] whose multipliers keep an effective sample size of
    at least `share` of the batch; `low` where that already holds, `high` where nothing
    in the range reaches it. The effective size does not fall as β grows: its
    derivative in 1/β is twice the mean advantage under exp(A/β) minus that under
    exp(2A/β), which is never positive."""
    if not advantages:
        raise ValueError("A batch without sampled decisions has no β")
    target = share * len(advantages)
    if effective_size(advantages, low) >= target:
        return low
    if effective_size(advantages, high) < target:
        return high
    below, above = math.log(low), math.log(high)
    for _ in range(steps):
        middle = (below + above) / 2
        if effective_size(advantages, math.exp(middle)) >= target:
            above = middle
        else:
            below = middle
    return math.exp(above)


@dataclass(frozen=True)
class Temperature:
    """How a batch's β is chosen: `fixed` (infinite for no weighting), or tuned to the
    batch when it is None."""

    fixed: float | None = None
    share: float = ESS_SHARE
    low: float = BETA_MIN
    high: float = BETA_MAX

    def __post_init__(self):
        if self.fixed is not None and not self.fixed > 0:
            raise ValueError("β must be positive")
        if not 0 < self.share <= 1 or not 0 < self.low <= self.high or math.isinf(self.high):
            raise ValueError("β is tuned to a share of the batch in (0, 1], within finite positive bounds")

    @classmethod
    def parse(cls, text, **bounds):
        """`auto` (tuned to each batch), `inf` (no weighting) or a number."""
        return cls(None if str(text) == "auto" else float(text), **bounds)

    def of(self, advantages):
        return tune(advantages, self.share, self.low, self.high) if self.fixed is None else self.fixed

    def describe(self):
        """The rule as a record keeps it: tuned, fixed at a β, or no weighting."""
        if self.fixed is None:
            return dict(rule="tuned", share=self.share, low=self.low, high=self.high)
        return dict(rule="none") if math.isinf(self.fixed) else dict(rule="fixed", beta=self.fixed)


def factors(advantages, base, beta):
    """What each sample's weight is multiplied by: exp(A/β), normalized so that the
    batch's total weight Σ base·factor stays Σ base."""
    raw = multipliers(advantages, beta)
    scale = sum(base) / sum(b * r for b, r in zip(base, raw))
    return [r * scale for r in raw]


def reweigh(logical, table, temperature, scale=1.0):
    """A logical batch of sampled decisions with its weights multiplied by their
    advantage factors and by `scale`, and what was done: β, the effective sample size
    and how many samples. `table` maps (run id, macro index) to the advantage."""
    items = [item for micro, _ in logical for item in micro]
    try:
        advantages = [table[item.run_id, item.index] for item in items]
    except KeyError as missing:
        raise ProtocolError(f"Sampled decision {missing.args[0]} has no advantage; compute the "
                            "advantages of every round of the buffer first") from None
    beta = temperature.of(advantages)
    factor = iter(factors(advantages, [item.weight for item in items], beta))
    weighted = [([replace(item, weight=item.weight * next(factor) * scale) for item in micro], prepared)
                for micro, prepared in logical]
    size = effective_size(advantages, beta)
    return weighted, dict(beta=beta, effective=size, samples=len(items), share=size / len(items))


def _again(factory):
    """The batches of a stream, begun anew whenever it ends."""
    while True:
        batches, empty = factory(), True
        try:
            for batch in batches:
                empty = False
                yield batch
        finally:
            if hasattr(batches, "close"):
                batches.close()
        if empty:
            raise ProtocolError("A data stream of the update holds no batch")


class Mixed:
    """The batches of one pass: each the weighted sampled half followed by a supervised
    half, as one logical batch of the trainer. `sampled` and `supervised` begin a
    stream of half-batches each time they are called. Without `updates` the pass is
    one pass over the sampled stream; with it, that many batches, either stream begun
    anew when it ends. `last` is what `reweigh` did to the batch just handed over,
    and `seen` every batch's."""

    def __init__(self, sampled, supervised, weigh, updates=None):
        if updates is not None and updates < 1:
            raise ValueError("A pass makes at least one update")
        self.sampled, self.supervised, self.weigh, self.updates = sampled, supervised, weigh, updates
        self.last, self.seen = None, []
        self.batches = self._batches()

    def __iter__(self):
        return self.batches

    def close(self):
        """End both streams, and with them the processes that feed them; the trainer
        calls this when it leaves the pass, however it leaves."""
        self.batches.close()

    def _batches(self):
        ours = _again(self.sampled) if self.updates is not None else iter(self.sampled())
        theirs = _again(self.supervised)
        try:
            while self.updates is None or len(self.seen) < self.updates:
                batch = next(ours, None)
                if batch is None:
                    break
                weighted, self.last = self.weigh(batch)
                self.seen.append(self.last)
                yield weighted + next(theirs)
        finally:
            for batches in (ours, theirs):
                if hasattr(batches, "close"):
                    batches.close()

    def summary(self):
        """β and the effective share over the batches of the pass."""
        if not self.seen:
            return dict(batches=0)
        betas = [entry["beta"] for entry in self.seen if not math.isinf(entry["beta"])]
        shares = [entry["share"] for entry in self.seen]
        return dict(
            batches=len(self.seen), samples=sum(entry["samples"] for entry in self.seen),
            beta=sum(betas) / len(betas) if betas else None,
            beta_min=min(betas) if betas else None, beta_max=max(betas) if betas else None,
            effective_share=sum(shares) / len(shares), effective_share_min=min(shares))


def pass_updates(rounds, config):
    """How many updates one pass over the training part of `rounds` makes: its sampled
    decisions, half a logical batch at a time."""
    training, _ = split_held_out(RunShards(rounds))
    return max(1, math.ceil(training.macros / (config.logical_batch_size // 2)))


def supervised_loss(sources):
    """The weighted mean loss of everything but the sampled decisions, from a trainer's
    per-source report."""
    rest = [entry for name, entry in sources.items() if name != SAMPLED]
    weight = sum(entry["weight"] for entry in rest)
    return sum(entry["loss"] * entry["weight"] for entry in rest) / weight if weight else None


def update(learner, rounds, supervised, *, temperature=None, updates=None, window_shards=8, on_update=None,
           weight_cap=WEIGHT_CAP):
    """One advantage-weighted pass of `learner`: half of every update the sampled
    decisions of `rounds` (the buffer's round directories, each with its advantages),
    half the supervised data of the `supervised` directories (demonstrations first,
    then imports of student battles). The held-out seeds of both are left out.
    `window_shards` shards are in memory at a time, between the two streams. Returns
    the pass's metrics, with `awr`: β, the effective share, both losses and how much
    data each half drew from."""
    config = learner.config
    half = config.logical_batch_size // 2
    if half < 1 or half * 2 != config.logical_batch_size:
        raise ValueError("An update of two halves needs an even logical batch size")
    temperature = temperature or Temperature()
    ours, _ = split_held_out(RunShards(rounds))
    theirs, _ = split_held_out(RunShards(supervised))
    if not ours.macros or not theirs.macros:
        raise ProtocolError("An advantage-weighted update needs sampled decisions and supervised samples")
    if any(source != SAMPLED for _, _, source in ours.source_counts):
        raise ProtocolError("The buffer's rounds must hold sampled runs only")
    table = advantage.read(rounds)
    # The trainer's loss scale is the supervised data's; the sampled weights join it.
    total, scale = theirs.macros, ours.macros / theirs.macros
    halves = replace(config, logical_batch_size=half, loader_feeders=max(1, config.loader_feeders // 2))

    shares = buffer_weights(ours.source_counts, weight_cap)
    capped = sorted("/".join(key[:2]) for key, weight in group_weights(ours.source_counts).items()
                    if shares[key] < weight)

    # The two streams share what one stream of behavior cloning has: half of the
    # feeders each, and half of the shards held in memory at a time.
    window = max(1, window_shards // 2)

    def batches(shards, weights):
        return lambda: stream(shards, learner.vocabulary, halves, window_shards=window, shuffle=True,
                              weights=weights)

    mixed = Mixed(batches(ours, shares), batches(theirs, theirs.weights),
                  lambda batch: reweigh(batch, table, temperature, scale), updates)

    def told(metrics):
        done = mixed.last
        on_update(dict(
            metrics, beta=None if math.isinf(done["beta"]) else done["beta"], effective_samples=done["effective"],
            effective_share=done["share"], sampled_samples=done["samples"],
            awr_loss=metrics["sources"].get(SAMPLED, {}).get("loss"),
            supervised_loss=supervised_loss(metrics["sources"])))

    metrics = learner.fit(mixed, total=total, on_update=told if on_update else None)
    metrics["awr"] = dict(
        mixed.summary(), awr_loss=metrics["sources"].get(SAMPLED, {}).get("loss"),
        supervised_loss=supervised_loss(metrics["sources"]), sampled=extent(ours), supervised=extent(theirs),
        temperature=temperature.describe(), weight_cap=weight_cap or None, capped_groups=capped)
    return metrics
