"""Combat demonstrations as a stream of an update: the solver's fights of a set of
imports, drawn to the pass's quota.

The battle behavior cloning of ADR 0001 learns the solver's fights in the same batches
as the act groups' decisions (`main/learn/grpo.py`, the objective in
`main/learn/mixed.py`). Its demonstrations are imports of independent samples
(`python -m main import-independent`), several directories read as one data set: the
fights the solver played from a few rounds' entry saves.

- **Fights only** (`fights`). Of every directory only the sample groups that hold
  combat macro-actions are taken, each whole, the selections made inside the fight
  with it: what `--combat-from` takes of a directory of demonstrations. The held-out
  seed groups (`data.held_out`) are trained on too: these demonstrations are scored
  elsewhere (the solver's fights of the fight evaluation set), and a split would
  throw away a fiftieth of every round's fights.
- **A quota** (`draw`). A pass takes as many demonstrated macro-actions as the
  demonstrations' share of its batches makes. More than the directories hold, and
  every fight is taken as many whole times as fit, the rest a part of them; fewer,
  and a part of them. The part is whole fights in an order drawn from `key`, taken
  until the quota is reached: the same directories, quota and key draw the same
  fights, whatever order the directories are given in.
- **Even weights** (`batches`). Every demonstrated macro-action weighs the same, each
  time it is in a batch as much as an act group's decision does on average: no
  coefficient between the kinds of source.
"""

import random
from dataclasses import dataclass, replace

from main.data.index import row_source
from main.data.shards import RunShards
from main.env.protocol import ProtocolError
from main.learn import streams
from main.learn.mixed import DEMOS

# The sources of runs an import of demonstrations holds (`data.index.row_source`).
DEMONSTRATED = ("independent", "demonstration", "recorder_bc")


def fights(directories):
    """The fights of these directories of demonstrations, as one data set: every run
    that holds combat macro-actions, whole, held-out seeds included."""
    every = RunShards([*directories])
    rows = [row for row in every.rows if "combat" in row["phases"]]
    sampled = {row_source(row) for row in rows} - set(DEMONSTRATED)
    if sampled:
        raise ProtocolError(f"Combat demonstrations are imported demonstrations, not runs of {sorted(sampled)}")
    if not rows:
        raise ProtocolError(f"No fight in the combat demonstrations {[str(path) for path in every.directories]}")
    return RunShards(every.directories, rows)


@dataclass(frozen=True)
class Draw:
    """What a pass takes of the fights `pool`: every fight `passes` times, then those
    of `part` once."""

    pool: RunShards
    quota: int
    key: str
    passes: int
    part: tuple

    @property
    def macros(self):
        """The demonstrated macro-actions drawn."""
        return self.passes * self.pool.macros + sum(sum(row["phases"].values()) for row in self.part)

    def describe(self):
        return dict(directories=[str(path) for path in self.pool.directories], fights=len(self.pool),
                    macros=self.pool.macros, quota=self.quota, key=self.key, passes=self.passes,
                    part_fights=len(self.part), part_macros=self.macros - self.passes * self.pool.macros,
                    drawn=self.macros, repeats=round(self.macros / self.pool.macros, 4))


def draw(pool, quota, key):
    """The fights of `pool` a pass of `quota` demonstrated macro-actions takes: every
    fight as many whole times as `quota` holds the pool, the rest in whole fights in
    the order `key` shuffles them to, up to the first that reaches it."""
    if type(quota) is not int or quota < 1:
        raise ValueError("A pass takes at least one demonstrated macro-action")
    passes, rest = divmod(quota, pool.macros)
    # In an order of their own, not of the directories' listing.
    order = sorted(pool.rows, key=lambda row: (str(pool.directories[row.get("root", 0)].resolve()), row["run_id"]))
    random.Random(str(key)).shuffle(order)
    part, taken = [], 0
    for row in order:
        if taken >= rest:
            break
        part.append(row)
        taken += sum(row["phases"].values())
    return Draw(pool, quota, str(key), passes, tuple(part))


def batches(learner, drawn, size, window, feeders):
    """The stream of logical batches of `size` samples of what `drawn` takes: a pass
    over every fight for each of its passes, then one over its part. Every sample
    weighs one over the pool's macro-actions."""
    pool = drawn.pool
    weights = streams.buffer_weights(pool.source_counts, phase_weights="even")
    parts = [pool] * drawn.passes + ([RunShards(pool.directories, list(drawn.part))] if drawn.part else [])

    def read():
        for shards in parts:
            yield from streams.batches(learner, shards, weights, size, window, feeders)()
    return read


def weigh(scale):
    """A demonstration stream's batches, reported as `DEMOS`, weights times `scale`."""
    def weighed(batch):
        parts = [([replace(item, source=DEMOS, weight=item.weight * scale) for item in micro], prepared)
                 for micro, prepared in batch]
        return parts, dict(samples=sum(len(micro) for micro, _ in parts))
    return weighed
