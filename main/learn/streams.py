"""The streams of sampled decisions an update reads: their share weights, a stream's
batches, several streams joined into one batch.

- **Share weights.** Within a stream every (character, phase) group weighs the same, as
  the groups of demonstrations do, up to a cap on what one decision of a small group
  may weigh (`buffer_weights`); or every decision alike. A stream's share weights add
  up to one.
- **Streams.** A stream is the sampled runs of a directory that have an advantage
  (`priced`), read by feeder processes of its own (`batches`), at least two
  (`stream_feeders`): one feeder decoding its next shards would stall its stream, and
  a joint batch with it. Several streams make one logical batch, each giving its part
  (`sizes`); a stream that ends before the pass does begins anew (`Joint`).

The act groups' composition of these, a stream per wave at the waves' shares, is
`main/learn/batches.py`.
"""

import math
from dataclasses import replace

from main.data.index import read_index
from main.data.shards import RunShards, group_weights
from main.env.protocol import ProtocolError
from main.learn import advantage
from main.learn.loader import stream

# What one sampled decision may weigh before its advantage, in even shares of its
# stream (1 / its decisions). Demonstrations have tens of thousands of macro-actions in
# their smallest group; a round has phases that occur a few times in a thousand runs,
# and an equal share for such a group would hand a single decision the weight of
# several batches.
WEIGHT_CAP = 10.0
# Feeder processes each stream of an update has at least.
STREAM_FEEDERS = 2


# How the decisions of a stream share its weight: by (character, phase) group, every
# group alike (`group`), or every decision alike (`even`).
PHASE_WEIGHTS = ("group", "even")


def buffer_weights(source_counts, cap=WEIGHT_CAP, phase_weights="group"):
    """The share weight of a sampled decision of every (character, phase, source) of
    a stream. With `phase_weights` `group`, every group weighs the same, as
    `data.group_weights` has it, unless its decisions would then weigh more than `cap`
    even shares each: so small a group keeps `cap` even shares per decision, and the
    rest of its share goes to the other groups, which may bring the next smallest over
    the cap in turn. Without a cap, the plain rule. With `even`, every decision weighs
    the same, whatever its phase: the plain policy gradient over the stream's
    decisions, most of which are fights."""
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


def stream_feeders(config, shares):
    """How many feeder processes each stream of an update reads with, by its share of
    the batch: the configuration's feeders in that proportion, at least
    `STREAM_FEEDERS` each. `shares` maps a stream to its share; returns the same map
    to feeder counts. A stream that fills most of every batch is the one whose
    feeders must keep up."""
    return {name: max(STREAM_FEEDERS, math.ceil(share * config.loader_feeders)) for name, share in shares.items()}


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
    """The sampled runs of a directory that have an advantage (`main/learn/advantage.py`),
    as a `RunShards` over the directory, and their advantage table. A run without one (a
    segment of act groups that reached its cut with nothing finished from its save, or
    one of a uniform group left out) is left out."""
    table = advantage.read([directory])
    ids = {run for run, _ in table}
    rows = [row for row in read_index(directory) if row["run_id"] in ids]
    return RunShards(directory, rows), table
