"""The batches of an update of act groups: the waves' shares, their weights, their streams.

A round of act groups (`main/play/groups.py`) is one data set with a stream per wave:
every logical batch takes a part of each wave's segments, sized by the waves' shares,
and the joint batch is what the objective (`main/learn/grpo.py`) learns from.

- **Share weights.** Within a wave every (character, phase) group weighs the same, as
  the groups of demonstrations do, up to a cap on what one decision of a small group
  may weigh (`buffer_weights`); a wave's share weights add up to one.
- **Shares and scale.** A wave's share weights are multiplied by its share of the
  round's decisions, and the trainer scales every batch's loss by the round's decision
  count over the logical batch size: a decision then weighs 1/B every time it is in a
  batch, whatever its wave. How often it is in one is set by the waves' shares of the
  batch (`wave_shares`): by default a wave's share is its share of the decisions, so a
  pass of ΣN/B updates (`pass_updates`) sees every wave about once and every decision
  weighs the same; a wave given a larger share is seen more often over the pass and
  weighs more. A wave's share is its share of the gradient.
- **Streams.** Each wave is read by feeder processes of its own, at least two
  (`stream_feeders`): one feeder decoding its next shards would stall its stream, and
  the joint batch with it. A stream that ends before the pass does begins anew (`Joint`).
"""

import math
from dataclasses import replace

from main.data.index import read_index
from main.data.shards import RunShards, group_weights
from main.learn import advantage
from main.learn.loader import stream
from main.env.protocol import ProtocolError

SAMPLED = "rollout"
# What one sampled decision may weigh before its advantage, in even shares of its
# wave (1 / its decisions). Demonstrations have tens of thousands of macro-actions in
# their smallest group; a round has phases that occur a few times in a thousand runs,
# and an equal share for such a group would hand a single decision the weight of
# several batches.
WEIGHT_CAP = 10.0
# Feeder processes each stream of an update has at least.
STREAM_FEEDERS = 2


# How the decisions of a wave share its weight: by (character, phase) group, every
# group alike (`group`), or every decision alike (`even`).
PHASE_WEIGHTS = ("group", "even")


def buffer_weights(source_counts, cap=WEIGHT_CAP, phase_weights="group"):
    """The share weight of a sampled decision of every (character, phase, source) of
    a wave. With `phase_weights` `group`, every group weighs the same, as
    `data.group_weights` has it, unless its decisions would then weigh more than `cap`
    even shares each: so small a group keeps `cap` even shares per decision, and the
    rest of its share goes to the other groups, which may bring the next smallest over
    the cap in turn. Without a cap, the plain rule. With `even`, every decision weighs
    the same, whatever its phase: the plain policy gradient over the wave's decisions,
    where fights are three quarters of them."""
    if phase_weights not in PHASE_WEIGHTS:
        raise ValueError("Phase weights are `group` or `even`")
    counts = {key: count for key, count in source_counts.items() if count}
    if phase_weights == "even":
        total = sum(counts.values())
        return {key: 1 / total for key in counts}
    weights = group_weights(source_counts)
    if not cap:
        return weights
    if cap < 1:
        raise ValueError("The cap is in even shares of the data: at least one")
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


def wave_shares(macros, relative=None):
    """The shares of a batch of the waves of `macros` ({wave: decisions}): each wave's
    share of the decisions, or `relative` (one positive number per wave, in the waves'
    order) normalized."""
    names = list(macros)
    if relative is None:
        weights = [macros[name] for name in names]
    else:
        weights = [float(value) for value in relative]
        if len(weights) != len(names):
            raise ValueError(f"One share for each of the {len(names)} waves")
    if min(weights, default=0) <= 0:
        raise ValueError("A wave's share is positive")
    total = sum(weights)
    return {name: weight / total for name, weight in zip(names, weights)}


def pass_updates(macros, batch):
    """How many updates one pass over the waves' decisions makes, a logical batch at a
    time: at the default shares every wave is seen about once."""
    return max(1, math.ceil(sum(macros.values()) / batch))


def sizes(batch, shares):
    """How many samples of a logical batch of `batch` each stream gives, in proportion
    to its share, adding up to the batch."""
    names = list(shares)
    total = sum(shares.values())
    exact = {name: batch * shares[name] / total for name in names}
    counts = {name: int(exact[name]) for name in names}
    for name in sorted(names, key=lambda name: exact[name] - counts[name], reverse=True)[:batch - sum(counts.values())]:
        counts[name] += 1
    if min(counts.values()) < 1:
        raise ValueError("Every stream of an update needs a sample of every batch")
    return counts


def stream_feeders(config, streams):
    """How many feeder processes each of `streams` streams of an update reads with:
    the configuration's feeders divided among them, at least `STREAM_FEEDERS` each."""
    return max(STREAM_FEEDERS, config.loader_feeders // streams)


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


class Joint:
    """The batches of one pass over several streams: each logical batch takes a part
    of every stream (the waves of a round), each part weighed as its stream says. The
    `primary` stream sets the pass (one pass over it, or `updates` batches); the others
    begin anew when they end. `seen` keeps what weighing reported of every batch, by
    stream, and `last` the latest."""

    def __init__(self, streams, primary, updates=None):
        if updates is not None and updates < 1:
            raise ValueError("A pass makes at least one update")
        self.streams, self.primary, self.updates = streams, primary, updates
        self.seen = {name: [] for name in streams}
        self.last = {}
        self.batches = self._batches()

    def __iter__(self):
        return self.batches

    def close(self):
        self.batches.close()

    def _batches(self):
        sources = {name: (_again(factory) if name != self.primary or self.updates is not None else iter(factory()))
                   for name, (factory, _) in self.streams.items()}
        try:
            count = 0
            while self.updates is None or count < self.updates:
                parts = []
                for name, (_, weigh) in self.streams.items():
                    batch = next(sources[name], None)
                    if batch is None:
                        return
                    weighted, done = weigh(batch)
                    if done is not None:
                        self.seen[name].append(done)
                        self.last[name] = done
                    parts += weighted
                count += 1
                yield parts
        finally:
            for batches in sources.values():
                if hasattr(batches, "close"):
                    batches.close()


def batches(learner, shards, weights, size, window, feeders):
    """A stream of logical batches of `size` samples of `shards`, begun anew each call."""
    part = replace(learner.config, logical_batch_size=size, loader_feeders=feeders)
    return lambda: stream(shards, learner.vocabulary, part, window_shards=window, shuffle=True, weights=weights)


def priced(directory):
    """The segments of a wave of act groups that have an advantage, as a `RunShards`
    over the wave's directory, and their advantage table. A segment without one (it
    reached its cut and nothing finished from its save) is left out."""
    table = advantage.read([directory])
    ids = {run for run, _ in table}
    rows = [row for row in read_index(directory) if row["run_id"] in ids]
    return RunShards(directory, rows), table


def priced_macros(waves):
    """{wave: priced decisions} of a round's waves, from their advantage headers.
    `waves` maps a wave's name to its directory."""
    macros = {}
    for name, directory in waves.items():
        head = advantage.header(directory)
        if head is None:
            raise FileNotFoundError(f"Wave {directory} has no advantages; compute them first")
        macros[name] = head["decisions"]
    return macros


def priced_waves(waves):
    """{wave: (RunShards, advantage table)} of a round's waves, each checked to hold
    sampled segments with advantages."""
    data = {name: priced(directory) for name, directory in waves.items()}
    for name, (shards, _) in data.items():
        if not shards.macros:
            raise ProtocolError(f"Wave {name} holds no decision with an advantage")
        if any(source != SAMPLED for _, _, source in shards.source_counts):
            raise ProtocolError("A wave's segments must be sampled runs only")
    return data


def streams_of(data, shares, config):
    """What the streams of an update have in common: the shares of the batch (the
    waves by their decisions unless given), each stream's sample count and the round's
    decision count."""
    shares = wave_shares({name: shards.macros for name, (shards, _) in data.items()}) if shares is None else dict(shares)
    if set(shares) != set(data):
        raise ValueError("The shares name every wave, nothing else")
    counts = sizes(config.logical_batch_size, shares)
    total = sum(shards.macros for shards, _ in data.values())
    return shares, counts, total
