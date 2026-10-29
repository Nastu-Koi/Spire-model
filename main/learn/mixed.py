"""The objective of a mixed batch: every stream's samples learned as their source says.

One update's logical batch may hold samples of three kinds of source (`KINDS`), each a
stream or several of their own (`main/learn/grpo.py`):

- **act groups** (`ACT`): the decisions outside the fights of a round's act waves,
  sampled, with advantages; the clipped objective with a KL to the reference
  (`clipped.Clipped`);
- **fight groups** (`FIGHTS`): the decisions of the fight groups' segments, sampled,
  with advantages; the same clipped objective with its KL;
- **combat demonstrations** (`DEMOS`): the solver's fights (`main/learn/combat_demos.py`);
  the negative log-likelihood of every macro-action at temperature 1, as behavior
  cloning learns it (`Likelihood`), with no KL: the KL keeps what the samples teach
  from carrying the policy off, and a demonstration's label is no sample of it.

Every macro-action's loss is added on one scale: a sample weighs what its stream's
share weights and the trainer's scale give it, whatever its kind, with no coefficient
between the kinds (`trainer.Learner.fit`). The kinds' shares of the batch are their
shares of the gradient (`batches.stream_shares`).

What a step saw is tallied by kind on the device (`Mixed.flush`) and reported by
kind (`Mixed.means`): the clipped kinds' ratio, clip fraction, KL and share of zero
advantages, the demonstrations' NLL, and every kind's samples and accuracy. The
report's top-level entries are those of the clipped objective over every sampled
decision of the step, act groups and fight groups together, as an update of act
groups alone has always reported them.
"""

import torch

from main.env.protocol import ProtocolError
from main.learn.clipped import MAXIMA, TALLY, Clipped, combine, empty
from main.play.rollout import precision_context

ACT, FIGHTS, DEMOS = "act-groups", "fight-groups", "combat-demos"
# The kinds of source, in the order they are reported; the sampled ones are learned
# by the clipped objective.
KINDS = (ACT, FIGHTS, DEMOS)
SAMPLED = (ACT, FIGHTS)


class Likelihood:
    """The negative log-likelihood of every macro-action at temperature 1, with no KL:
    what behavior cloning learns a demonstration by (`trainer.Learner`). It keeps the
    NLL of what it saw on the device until `flush`."""

    # The tally's entries: the summed NLL.
    WIDTH = 1

    def __init__(self):
        self._seen, self._samples = None, 0

    def __call__(self, learner, prepared, micro):
        """(loss, entropy at temperature 1, accuracy counts) of every macro-action of a
        microbatch, the loss differentiable."""
        model = learner.model
        with precision_context(model, learner.config.precision):
            log_prob, entropy, accuracy = model.replay(prepared.to(model.device), with_accuracy=True)
        loss = -log_prob
        tally = loss.detach().float().sum().double().reshape(1)
        self._seen = tally if self._seen is None else self._seen + tally
        self._samples += len(micro)
        return loss, entropy, accuracy

    def flush(self):
        seen, samples = self._seen, self._samples
        self._seen, self._samples = None, 0
        return seen, samples

    @staticmethod
    def means(tally, samples):
        return dict(nll=float(tally[0]) / samples if samples else None)


class Mixed:
    """The objective of a batch whose streams are of several kinds: every microbatch
    is learned by its kind's objective. `kinds` maps a stream's name (what its samples
    are reported as, `Sample.source`) to its kind (`KINDS`); `reference` is the KL's
    reference of the sampled kinds, a frozen policy network or None for the
    distributions the decisions were drawn from (`clipped.Clipped`); `kl` is its
    coefficient, one for every sampled kind or {kind: coefficient}. A microbatch is
    one stream's: the streams are packed apart (`streams.Joint`)."""

    def __init__(self, kinds, reference, *, clip, kl):
        unknown = set(kinds.values()) - set(KINDS)
        if unknown:
            raise ValueError(f"Unknown kinds of source {sorted(unknown)}; they are {list(KINDS)}")
        self.kinds = dict(kinds)
        if isinstance(kl, dict) and set(kl) - set(SAMPLED):
            raise ValueError(f"KL coefficients are of the sampled kinds {list(SAMPLED)}")
        coefficient = lambda kind: kl.get(kind, 0.0) if isinstance(kl, dict) else kl  # noqa: E731
        self.objectives = {kind: Clipped(reference, clip=clip, kl=coefficient(kind)) if kind in SAMPLED
                           else Likelihood() for kind in KINDS if kind in self.kinds.values()}
        self._accuracy = dict.fromkeys(self.objectives)
        self._device = None
        # Everything flushed so far, for the pass's summary.
        self._all, self._samples_all = None, dict.fromkeys(self.objectives, 0)

    @staticmethod
    def width(objective):
        return len(TALLY) if isinstance(objective, Clipped) else objective.WIDTH

    def __call__(self, learner, prepared, micro):
        names = {item.source for item in micro}
        if len(names) != 1:
            raise ProtocolError(f"A microbatch holds samples of several streams: {sorted(map(str, names))}")
        name = names.pop()
        if name not in self.kinds:
            raise ProtocolError(f"Samples of stream {name!r}, which is none of the update's")
        kind = self.kinds[name]
        loss, entropy, accuracy = self.objectives[kind](learner, prepared, micro)
        counts = accuracy.sum(0).double()
        self._accuracy[kind] = counts if self._accuracy[kind] is None else self._accuracy[kind] + counts
        self._device = loss.device
        return loss, entropy, accuracy

    def flush(self):
        """Every kind's tally since the last flush (its objective's, then the correct
        and all decisions of its accuracy) in one tensor still on the device, and each
        kind's macro-actions: what the trainer reads back with the step (`means`)."""
        parts, samples = [], {}
        for kind, objective in self.objectives.items():
            seen, samples[kind] = objective.flush()
            accuracy, self._accuracy[kind] = self._accuracy[kind], None
            if self._device is None:
                continue
            zeros = lambda width: torch.zeros(width, dtype=torch.float64, device=self._device)  # noqa: E731
            nothing = empty(self._device) if isinstance(objective, Clipped) else zeros(objective.WIDTH)
            parts.append(torch.cat([nothing if seen is None else seen, zeros(2) if accuracy is None else accuracy]))
        tally = torch.cat(parts) if parts else None
        if tally is not None:
            self._all = combine(self._all, tally, self.maxima())
        for kind, count in samples.items():
            self._samples_all[kind] += count
        return tally, samples

    def maxima(self):
        """Where the entries combined by their maximum are in a flushed tally (`clipped.MAXIMA`)."""
        places, offset = [], 0
        for objective in self.objectives.values():
            if isinstance(objective, Clipped):
                places += [offset + index for index in MAXIMA]
            offset += self.width(objective) + 2
        return tuple(places)

    def means(self, tally, samples):
        """The report of a tally over each kind's `samples`: the clipped objective's means
        over every sampled kind together, and by kind (`mix`) its samples and accuracy
        with its objective's means. Waits for the tally."""
        offset, sampled, count, mix = 0, None, 0, {}
        for kind, objective in self.objectives.items():
            width, n = self.width(objective), samples.get(kind, 0)
            if tally is None:
                part, correct, decisions = None, 0, 0
            else:
                part = tally[offset:offset + width]
                correct, decisions = (float(value) for value in tally[offset + width:offset + width + 2].tolist())
            offset += width + 2
            if isinstance(objective, Clipped):
                report = Clipped.means(part, n)
                if part is not None:
                    sampled = combine(sampled, part)
                count += n
            else:
                report = objective.means(part, n) if part is not None else dict(nll=None)
            mix[kind] = dict(samples=n, accuracy=correct / decisions if decisions else None, **report)
        return dict(Clipped.means(sampled, count), mix=mix)

    def take(self):
        """What was seen since the last `take` or `flush`."""
        return self.means(*self.flush())

    def summary(self):
        """The same over everything flushed, with the sampled decisions' count."""
        report = self.means(self._all, self._samples_all)
        return dict(report, samples=sum(self._samples_all[kind] for kind in self.objectives if kind in SAMPLED))


def by_kind(mix, sources, kinds):
    """`mix` (`Mixed.means`) with every kind's loss and entropy at temperature 1, the
    weighted means of its streams' (`sources`, the trainer's report by stream;
    `kinds` maps a stream to its kind)."""
    result = {}
    for kind, report in mix.items():
        streams = [sources[name] for name, of in kinds.items() if of == kind and name in sources]
        weight = sum(entry["weight"] for entry in streams)
        mean = (lambda key: sum(entry[key] * entry["weight"] for entry in streams) / weight) if weight else None
        result[kind] = dict(report, loss=mean("loss") if mean else None, entropy=mean("entropy") if mean else None)
    return result
