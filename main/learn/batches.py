"""The batches of an update of act groups: the waves' shares, their weights, their streams.

A round of act groups (`main/play/groups.py`) is one data set with a stream per wave:
every logical batch takes a part of each wave's segments, sized by the waves' shares,
and the joint batch is what the objective (`main/learn/grpo.py`) learns from.

- **Share weights.** Within a wave every (character, phase) group weighs the same, as
  the groups of demonstrations do, up to a cap on what one decision of a small group
  may weigh (`streams.buffer_weights`); a wave's share weights add up to one.
- **Shares and scale.** A wave's share weights are multiplied by its share of the
  round's decisions, and the trainer scales every batch's loss by the round's decision
  count over the logical batch size: a decision then weighs 1/B every time it is in a
  batch, whatever its wave. How often it is in one is set by the waves' shares of the
  batch (`wave_shares`): by default a wave's share is its share of the decisions, so a
  pass of ΣN/B updates (`pass_updates`) sees every wave about once and every decision
  weighs the same; a wave given a larger share is seen more often over the pass and
  weighs more. A wave's share is its share of the gradient.
- **Streams.** Each wave is a stream of its own (`main/learn/streams.py`): read by
  feeder processes of its own, at least two (`streams.stream_feeders`), and begun anew
  when it ends before the pass does (`streams.Joint`).
- **Other kinds of source.** A batch may also hold the fight groups' decisions and
  combat demonstrations (`main/learn/mixed.py`), each a stream of its own. The kinds
  share the batch by their decisions, or as given, and the act groups' share is split
  among the waves as above (`stream_shares`); every decision of any kind weighs 1/B
  each time it is in a batch, so a kind's share is its share of the gradient too.
"""

import math

from main.env.protocol import ProtocolError
from main.learn import advantage
from main.learn.mixed import ACT, KINDS
from main.learn.streams import priced

SAMPLED = "rollout"


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


def stream_shares(macros, others=None, *, waves=None, sources=None):
    """The shares of a batch of an update's streams. `macros` maps an act wave to its
    priced decisions, `others` a stream of another kind of source (`mixed.KINDS`: the
    fight groups, the combat demonstrations), named by its kind, to its decisions. The
    kinds share the batch by `sources` ({kind: positive number}, naming every kind the
    update has) normalized, or by their decisions; the act groups' share is split among
    their waves by `waves` (`wave_shares`). With the act groups alone these are the
    waves' shares."""
    others = dict(others or {})
    if ACT in others or set(others) - set(KINDS):
        raise ValueError(f"The other streams of an update are of the kinds {[kind for kind in KINDS if kind != ACT]}")
    decisions = {ACT: sum(macros.values()), **others}
    if sources is None:
        weights = decisions
    else:
        if set(sources) != set(decisions):
            raise ValueError(f"The shares of the sources name every kind of the update, {sorted(decisions)}, "
                             "nothing else")
        weights = {kind: float(sources[kind]) for kind in decisions}
    if min(weights.values()) <= 0:
        raise ValueError("A source's share is positive")
    total = sum(weights.values())
    kinds = {kind: weight / total for kind, weight in weights.items()}
    within = wave_shares(macros, waves)
    return {**{name: kinds[ACT] * share for name, share in within.items()}, **{kind: kinds[kind] for kind in others}}


def pass_updates(macros, batch):
    """How many updates one pass over the waves' decisions makes, a logical batch at a
    time: at the default shares every wave is seen about once."""
    return max(1, math.ceil(sum(macros.values()) / batch))


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

