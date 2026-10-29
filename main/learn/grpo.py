"""Group-relative policy optimization of act groups: the clipped objective over a round's
waves, in batches that may also hold fight groups and combat demonstrations.

What a round of act groups (`main/play/groups.py`) teaches, with the advantages of its
groups (`main/learn/groups.py`) and the batches of its waves (`main/learn/batches.py`),
by the clipped objective (`main/learn/clipped.py`): a clipped ratio of the tempered
policy and a KL to a reference.

The same batches may hold two other kinds of source, each a stream of its own beside
the waves (`main/learn/mixed.py`): the fight groups' segments, priced as a wave is
(`main/learn/fight_groups.py`) and learned by the same clipped objective with its KL,
and combat demonstrations (`main/learn/combat_demos.py`), learned by their negative
log-likelihood with no KL. Every stream's samples are learned as its kind says, all on
one scale.

The weights are each stream's share weights times its share of the decisions; the
trainer sums Σ w_i L_i on the scale of behavior cloning (`trainer.Learner.fit`).
"""

import math

from main.data.shards import extent
from main.env.protocol import ProtocolError
from main.learn import batches, combat_demos, streams
from main.learn.clipped import CLIP, KL, with_advantages
from main.learn.mixed import ACT, DEMOS, FIGHTS, SAMPLED, Mixed, by_kind


def group_update(learner, waves, reference, *, fights=None, demos=None, demo_key="demos", clip=CLIP, kl=KL,
                 updates=None, window_shards=8, on_update=None, weight_cap=streams.WEIGHT_CAP, shares=None,
                 accumulate=1, frozen=None):
    """One pass over a round of act groups with the clipped objective (`Clipped`): every
    logical batch draws from each wave's segments in proportion to `shares` (each
    wave's share of the round's decisions by default, `batches.wave_shares`) at the
    wave's capped share weights (`streams.buffer_weights`, by phase group); `reference` is the frozen policy network the KL is to
    (`Reference`), or `Reference.recorded()`: the distributions the decisions were drawn from.
    `waves` maps a wave's name to its directory. `updates` sets the pass in logical
    batches (`batches.pass_updates` sees every wave about once at the default shares);
    without it the largest wave does. Every `accumulate` logical batches make one
    optimizer step (`trainer.Learner.fit`).

    `fights` is the directory of a round's fight groups (a stream `fight-groups`,
    priced as a wave is, learned by the same objective at the same weights), or
    {stream: directory} of several streams of fight groups (the fight groups and their
    hindsight forks); `demos` a
    data set of combat demonstrations (`combat_demos.fights`: a stream `combat-demos`,
    every macro-action by its NLL, evenly weighted), of which the pass takes its share
    of every batch, drawn by `demo_key` (`combat_demos.draw`). `shares` then names every
    stream, and by default each has its share of the decisions, a demonstration
    stream that of its data set's. Returns the pass's metrics, with `groups`: what each
    stream drew from and what the objective saw, by kind of source (`mix`).

    `kl` is one coefficient for every sampled kind or {kind: coefficient} (`Mixed`).
    The streams of a kind in `frozen` (`mixed.ACT`, say) are learned with every advantage
    0: only their KL pulls, keeping what they do where the reference has it."""
    if not waves:
        raise ValueError("An update of act groups needs at least one wave")
    if reference.vocabulary_digest is not None and reference.vocabulary_digest != learner.vocabulary.digest:
        raise ProtocolError("The reference policy has another vocabulary than the policy it anchors")
    config = learner.config
    data = batches.priced_waves(waves)
    sampled = dict(data)
    fights = None if fights is None else fights if isinstance(fights, dict) else {FIGHTS: fights}
    if fights is not None:
        sampled.update(batches.priced_waves(fights))
    macros = {name: shards.macros for name, (shards, _) in sampled.items()}
    if demos is not None:
        macros[DEMOS] = demos.macros
    if shares is None:
        shares = batches.wave_shares(macros)
    shares = dict(shares)
    if set(shares) != set(macros):
        raise ValueError("The shares name every stream, nothing else")
    counts = streams.sizes(config.logical_batch_size, shares)
    total = sum(macros.values())
    primary = max(waves, key=lambda name: data[name][0].macros)
    window = max(1, window_shards // len(macros))
    feeders = streams.stream_feeders(config, shares)
    frozen = set(frozen or ())
    if frozen - set(SAMPLED):
        raise ValueError(f"Only sampled kinds {list(SAMPLED)} are frozen")
    of_kind = lambda name: ACT if name in waves else FIGHTS  # noqa: E731
    parts = {name: (streams.batches(learner, shards, streams.buffer_weights(shards.source_counts, weight_cap),
                                    counts[name], window, feeders[name]),
                    with_advantages(table, name, shards.macros / total, frozen=of_kind(name) in frozen))
             for name, (shards, table) in sampled.items()}
    drawn = None
    if demos is not None:
        # The pass's demonstrations: their part of every batch it makes.
        made = updates if updates is not None else math.ceil(data[primary][0].macros / counts[primary])
        drawn = combat_demos.draw(demos, made * counts[DEMOS], demo_key)
        parts[DEMOS] = (combat_demos.batches(learner, drawn, counts[DEMOS], window, feeders[DEMOS]),
                        combat_demos.weigh(demos.macros / total))
    kinds = {**dict.fromkeys(waves, ACT), **dict.fromkeys(fights or (), FIGHTS),
             **({DEMOS: DEMOS} if demos is not None else {})}
    joint = streams.Joint(parts, primary, updates)
    objective = Mixed(kinds, reference.model, clip=clip, kl=kl)

    def told(metrics):
        losses = {name: metrics["sources"].get(name, {}).get("loss") for name in parts}
        done = joint.last
        # The objective's means of the step (`Mixed.means`) come in `metrics`.
        on_update(dict(
            metrics, mix=by_kind(metrics["mix"], metrics["sources"], kinds),
            **{f"{name}_loss": losses[name] for name in waves},
            **{f"{name}_zero_share": done[name]["zero"] / max(done[name]["samples"], 1) for name in waves}))

    metrics = learner.fit(joint, total=total, on_update=told if on_update else None, objective=objective,
                          accumulate=accumulate)

    def wave(name, shards):
        seen = joint.seen[name]
        samples = sum(entry["samples"] for entry in seen)
        return dict(batches=len(seen), samples=samples,
                    zero_share=sum(entry["zero"] for entry in seen) / max(samples, 1),
                    advantage_abs=sum(entry["advantage_abs"] for entry in seen) / max(samples, 1),
                    data=extent(shards))

    summary = objective.summary()
    mix = by_kind(summary.pop("mix"), metrics["sources"], kinds)
    for kind in mix:
        mix[kind].update(decisions=sum(macros[name] for name, of in kinds.items() if of == kind),
                         share=sum(shares[name] for name, of in kinds.items() if of == kind),
                         size=sum(counts[name] for name, of in kinds.items() if of == kind))
    metrics["groups"] = dict(
        clip=clip, kl=kl, shares=shares, sizes=counts, decisions=total, accumulate=accumulate,
        batch=config.logical_batch_size * accumulate,
        waves={name: wave(name, shards) for name, (shards, _) in data.items()},
        losses={name: metrics["sources"].get(name, {}).get("loss") for name in parts},
        weight_cap=weight_cap or None, phase_weights="group", seen=summary, mix=mix,
        **({"fights": wave(FIGHTS, sampled[FIGHTS][0])} if fights is not None and FIGHTS in fights else {}),
        **({"fight_streams": {name: wave(name, sampled[name][0]) for name in fights}}
           if fights is not None and len(fights) > 1 else {}),
        **({"demos": dict(drawn.describe(), batches=len(joint.seen[DEMOS]),
                          samples=sum(entry["samples"] for entry in joint.seen[DEMOS]))} if drawn else {}))
    return metrics
