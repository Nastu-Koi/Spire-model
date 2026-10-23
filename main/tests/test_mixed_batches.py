"""Mixed batches (`main.learn.mixed`, `main.learn.combat_demos`, `grpo.group_update`):
act groups, fight groups and combat demonstrations in one update, every stream learned
as its kind says, on one scale."""

import copy
import gzip
import json
import random
from collections import Counter
from dataclasses import replace

import pytest
import torch

from main.data.index import HEADER, index_row
from main.data.shards import RunShards, extent, samples
from main.env.protocol import CHARACTERS, SCHEMA, ProtocolError, segment_key
from main.learn import batches, combat_demos, grpo, streams
from main.learn.clipped import CLIP, KL, Clipped, Reference, with_advantages
from main.learn.groups import advantages
from main.learn.mixed import ACT, DEMOS, FIGHTS, Mixed
from main.network.policy import prepare_replay
from main.play.groups import sample_groups
from main.play.sampling import Samplers
from main.tests.journeys import learner, walked
from main.tests.test_groups import BOUNDARIES, engine, runner


# ---- A directory of combat demonstrations ----

def demo_run(character, number, fight=True):
    """A demonstration of the fights of a walk through the rooms (each room's selection,
    in the combat phase), or, not `fight`, of its decisions outside them."""
    frames = walked(character)
    macros, current = [], None
    for frame in frames:
        if (frame["public"]["phase"] == "combat") != fight:
            current = None
            continue
        if segment_key(frame) != current:
            current = segment_key(frame)
            macros.append(dict(phase=frame["public"]["phase"], steps=[]))
        candidates = frame["legal"]["candidates"]
        macros[-1]["steps"].append(dict(frame=frame, candidate_ref=candidates[0]["candidate_ref"],
                                        forced=len(candidates) == 1))
    return dict(schema=SCHEMA, source="demonstration", teacher_visibility="public", status="complete", victory=True,
                ascension=0, character=character, seed=f"demo-{number}", run_id=f"{character}:demo-{number}",
                contract=frames[0]["contract"], macros=macros)


def demo_directory(path, fights=6, others=2, first=0):
    """A directory of demonstrations: `fights` fights of three macro-actions each and
    `others` runs of decisions outside the fights."""
    path.mkdir(parents=True)
    (path / "accepted").mkdir()
    runs = [demo_run(CHARACTERS[number % len(CHARACTERS)], first + number) for number in range(fights)]
    runs += [demo_run("Ironclad", first + fights + number, fight=False) for number in range(others)]
    with gzip.open(path / "accepted/00000.jsonl.gz", "wt") as stream:
        stream.writelines(json.dumps(run) + "\n" for run in runs)
    with gzip.open(path / "index.jsonl.gz", "wt") as stream:
        stream.write(json.dumps(HEADER) + "\n")
        stream.writelines(json.dumps(index_row(run, "accepted/00000.jsonl.gz", line)) + "\n"
                          for line, run in enumerate(runs))
    return path


# ---- The objective ----

def setup():
    """A policy, a reference that differs from it, sampled items of a run drawn by the
    policy at temperature 0.5 with advantages, and the same macro-actions as demonstrations."""
    policy = learner()
    run = demo_run("Silent", 0)
    items = samples([run])
    prepared = prepare_replay(policy.vocabulary, [item.macro["steps"] for item in items])
    with torch.no_grad():
        drawn = policy.model.replay(prepared, temperatures=[0.5] * len(items))[0]
    sampled = [replace(item, source="act1", temperature=0.5, old_log_prob=float(log_prob),
                       advantage=1.0 if index % 2 else -1.0) for index, (item, log_prob) in enumerate(zip(items, drawn))]
    demos = [replace(item, source=DEMOS) for item in items]
    reference = copy.deepcopy(policy.model)
    torch.manual_seed(1)
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.add_(0.05 * torch.randn_like(parameter))
    return policy, reference, prepared, sampled, demos


def gradients(policy, objective, prepared, items):
    policy.model.zero_grad(set_to_none=True)
    loss, _, _ = objective(policy, prepared, items)
    loss.sum().backward()
    return loss.detach(), torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten()
                                     for p in policy.model.parameters()])


def test_a_sampled_microbatch_is_learned_by_the_clipped_objective_with_its_kl():
    policy, reference, prepared, sampled, _ = setup()
    plain, mixed = Clipped(reference, clip=0.2, kl=0.1), Mixed({"act1": ACT, DEMOS: DEMOS}, reference, clip=0.2, kl=0.1)
    loss, gradient = gradients(policy, plain, prepared, sampled)
    again, regradient = gradients(policy, mixed, prepared, sampled)
    assert torch.equal(loss, again) and torch.equal(gradient, regradient)
    seen, report = plain.take(), mixed.take()
    assert seen["kl"] > 0
    # The top level is the clipped objective's over the sampled decisions, as it was.
    assert {key: report[key] for key in seen} == seen
    act = report["mix"][ACT]
    assert act["samples"] == len(sampled) and act["kl"] == seen["kl"] and act["clip_fraction"] == seen["clip_fraction"]
    assert 0 <= act["accuracy"] <= 1
    assert report["mix"][DEMOS] == dict(samples=0, accuracy=None, nll=None)


def test_a_demonstration_is_learned_by_its_nll_with_no_kl_and_needs_nothing_sampled():
    policy, reference, prepared, _, demos = setup()
    assert all(item.temperature is None and item.advantage is None for item in demos)
    with pytest.raises(ProtocolError, match="temperature"):
        Clipped(reference)(policy, prepared, demos)
    with torch.no_grad():
        nll = -policy.model.replay(prepared)[0]
    for anchor in (reference, copy.deepcopy(policy.model)):
        mixed = Mixed({"act1": ACT, DEMOS: DEMOS}, anchor, clip=0.2, kl=0.1)
        loss, _ = gradients(policy, mixed, prepared, demos)
        # The NLL at temperature 1, whatever the reference: no KL.
        assert torch.allclose(loss, nll, atol=1e-6)
        report = mixed.take()
        assert report["mix"][DEMOS]["samples"] == len(demos)
        assert report["mix"][DEMOS]["nll"] == pytest.approx(float(nll.mean()), abs=1e-5)
        assert "kl" not in report["mix"][DEMOS] and report["mix"][DEMOS]["accuracy"] is not None
        # No sampled decision was seen: nothing of the clipped objective to report.
        assert report["mix"][ACT]["samples"] == 0 and report["kl"] == 0.0
    # The same macro-actions as sampled ones carry their KL to the reference.
    mixed = Mixed({"act1": ACT, DEMOS: DEMOS}, reference, clip=0.2, kl=0.1)
    sampled = [replace(item, source="act1", temperature=1.0, old_log_prob=float(-value), advantage=0.0)
               for item, value in zip(demos, nll)]
    loss, _ = gradients(policy, mixed, prepared, sampled)
    assert (loss > 1e-6).all() and mixed.take()["kl"] > 0


def test_the_kinds_are_tallied_apart_and_the_sampled_ones_together():
    policy, reference, prepared, sampled, demos = setup()
    mixed = Mixed({"act1": ACT, FIGHTS: FIGHTS, DEMOS: DEMOS}, reference, clip=0.2, kl=0.1)
    fights = [replace(item, source=FIGHTS, advantage=0.0) for item in sampled]
    act, fight = Clipped(reference, clip=0.2, kl=0.1), Clipped(reference, clip=0.2, kl=0.1)
    for objective, items in ((mixed, sampled), (mixed, fights), (mixed, demos), (act, sampled), (fight, fights)):
        objective(policy, prepared, items)
    tally, counts = mixed.flush()
    assert counts == {ACT: len(sampled), FIGHTS: len(fights), DEMOS: len(demos)}
    report = mixed.means(tally, counts)
    alone, other = act.take(), fight.take()
    assert report["mix"][ACT]["kl"] == pytest.approx(alone["kl"]) and report["mix"][FIGHTS]["zero_share"] == 1.0
    assert report["mix"][FIGHTS]["kl"] == pytest.approx(other["kl"])
    # The top level is over every sampled decision: here half of them had no advantage.
    assert report["kl"] == pytest.approx((alone["kl"] + other["kl"]) / 2) and report["zero_share"] == 0.5
    summary = mixed.summary()
    assert summary["samples"] == len(sampled) + len(fights) and summary["mix"][DEMOS]["samples"] == len(demos)
    # A microbatch of two streams, or of a stream the update does not have, is refused.
    with pytest.raises(ProtocolError, match="several streams"):
        mixed(policy, prepared, sampled[:1] + demos[1:])
    with pytest.raises(ProtocolError, match="none of the update"):
        mixed(policy, prepared, [replace(item, source="act9") for item in sampled])
    with pytest.raises(ValueError, match="Unknown kinds"):
        Mixed({"act1": "solver"}, reference, clip=0.2, kl=0.1)


# ---- Shares and the draw of demonstrations ----

def test_the_kinds_share_a_batch_by_their_decisions_or_as_given_and_the_waves_split_the_act_groups_share():
    macros = {"act1": 100, "act2": 200, "act3": 700}
    # With the act groups alone: the waves' shares, to the bit.
    assert batches.stream_shares(macros) == batches.wave_shares(macros)
    assert batches.stream_shares(macros, waves=[1, 1, 2]) == batches.wave_shares(macros, [1, 1, 2])
    by_decisions = batches.stream_shares(macros, {DEMOS: 3000})
    assert by_decisions == pytest.approx({"act1": 0.025, "act2": 0.05, "act3": 0.175, DEMOS: 0.75})
    even = batches.stream_shares(macros, {DEMOS: 3000}, sources={ACT: 1, DEMOS: 1})
    assert even == pytest.approx({"act1": 0.05, "act2": 0.1, "act3": 0.35, DEMOS: 0.5})
    three = batches.stream_shares(macros, {FIGHTS: 500, DEMOS: 3000}, waves=[1, 1, 2], sources={ACT: 2, FIGHTS: 1, DEMOS: 1})
    assert three == pytest.approx({"act1": 0.125, "act2": 0.125, "act3": 0.25, FIGHTS: 0.25, DEMOS: 0.25})
    for wrong in ({ACT: 1}, {ACT: 1, DEMOS: 1, FIGHTS: 1}, {ACT: 1, DEMOS: 0}):
        with pytest.raises(ValueError):
            batches.stream_shares(macros, {DEMOS: 3000}, sources=wrong)
    with pytest.raises(ValueError):
        batches.stream_shares(macros, {"solver": 3000})


def test_the_demonstrations_are_the_fights_of_their_directories_held_out_seeds_and_all(tmp_path, world):
    first, second = demo_directory(tmp_path / "a"), demo_directory(tmp_path / "b", fights=4, first=10)
    pool = combat_demos.fights([first, second])
    # Only the runs that hold combat macro-actions, whole; none of the others.
    assert len(pool) == 10 and all(set(row["phases"]) == {"combat"} for row in pool.rows)
    assert pool.macros == 30 and len(RunShards([first, second])) == 14
    # No seed is held out: every fight is there, as `--include-heldout` has it.
    assert {row["run_id"] for row in pool.rows} == {row["run_id"] for row in RunShards([first, second]).rows
                                                   if "combat" in row["phases"]}
    # A directory without fights, or of sampled runs, is refused.
    with pytest.raises(ProtocolError, match="No fight"):
        combat_demos.fights([demo_directory(tmp_path / "c", fights=0, others=2, first=20)])
    with pytest.raises(ProtocolError, match="rollout"):
        combat_demos.fights([first, world["waves"]["act1"]])


def test_a_pass_draws_its_quota_whole_fights_repeated_when_short_and_a_part_when_not(tmp_path):
    pool = combat_demos.fights([demo_directory(tmp_path / "a")])  # six fights of three
    assert pool.macros == 18
    many = combat_demos.draw(pool, 45, "key")
    assert (many.passes, len(many.part), many.macros) == (2, 3, 45)
    few = combat_demos.draw(pool, 10, "key")
    # Whole fights up to the first that reaches the quota.
    assert (few.passes, len(few.part), few.macros) == (0, 4, 12)
    assert few.describe() == dict(directories=[str(tmp_path / "a")], fights=6, macros=18, quota=10, key="key",
                                  passes=0, part_fights=4, part_macros=12, drawn=12, repeats=0.6667)
    # Reproducible: the same key draws the same fights; another key, others.
    assert combat_demos.draw(pool, 10, "key").part == few.part
    assert {tuple(row["run_id"] for row in combat_demos.draw(pool, 10, f"key{n}").part) for n in range(8)} != {
        tuple(row["run_id"] for row in few.part)}
    assert combat_demos.draw(pool, 18, "key").passes == 1 and not combat_demos.draw(pool, 18, "key").part
    with pytest.raises(ValueError):
        combat_demos.draw(pool, 0, "key")
    # Whatever order the directories are given in.
    second = demo_directory(tmp_path / "b", fights=4, first=10)
    one, other = combat_demos.fights([tmp_path / "a", second]), combat_demos.fights([second, tmp_path / "a"])
    ids = lambda drawn: [row["run_id"] for row in drawn.part]  # noqa: E731
    assert ids(combat_demos.draw(one, 17, "k")) == ids(combat_demos.draw(other, 17, "k"))

    # The stream reads every fight once a pass and the part once more, every sample at
    # one over the pool's macro-actions.
    policy = learner()
    read = [part for batch in combat_demos.batches(policy, many, 4, 8, 2)() for part in batch]
    seen = Counter(item.run_id for micro, _ in read for item in micro)
    drawn = {row["run_id"] for row in many.part}
    assert seen == {row["run_id"]: 3 * (2 + (row["run_id"] in drawn)) for row in pool.rows}
    assert all(item.weight == pytest.approx(1 / 18) for micro, _ in read for item in micro)
    weighed, done = combat_demos.weigh(0.5)(read)
    assert done["samples"] == 45 and {item.source for micro, _ in weighed for item in micro} == {DEMOS}
    assert all(item.weight == pytest.approx(1 / 36) for micro, _ in weighed for item in micro)


# ---- One update ----

@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """A sampled round of act groups with its advantages, and a directory of combat demonstrations."""
    root = tmp_path_factory.mktemp("mixed")
    policy = learner()
    seeds = {character: [f"{character}-seed"] for character in CHARACTERS}
    with Samplers(engine, policy.model, policy.vocabulary, engines=4, processes=2) as pool:
        sample_groups(pool, runner(policy), seeds, root / "groups", n=2, m=2, boundaries=BOUNDARIES)
    advantages(root / "groups", processes=2)
    return dict(waves={wave: root / "groups" / wave for wave in ("act1", "act2", "act3")},
                demos=demo_directory(root / "demos"))


def anchored(policy):
    """A reference that differs from the policy: the KL is not zero."""
    model = copy.deepcopy(policy.model)
    generator = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.02 * torch.randn(parameter.shape, generator=generator))
    return Reference(model, policy.vocabulary.digest)


def update(policy, waves, **arguments):
    """One update from a fixed state of the generators; its records of every step."""
    steps = []
    random.seed(3)
    torch.manual_seed(3)
    metrics = (arguments.pop("by") if "by" in arguments else grpo.group_update)(
        policy, waves, anchored(policy), on_update=steps.append, **arguments)
    return metrics, steps


def update_before(learner, waves, reference, *, clip=CLIP, kl=KL, updates=None, window_shards=8, on_update=None,
                  weight_cap=streams.WEIGHT_CAP, shares=None, phase_weights="group", accumulate=1):
    """The update of act groups as it was before mixed batches, kept to hold the
    present one to it."""
    config = learner.config
    data = batches.priced_waves(waves)
    shares = batches.wave_shares({name: shards.macros for name, (shards, _) in data.items()}) if shares is None \
        else dict(shares)
    counts = streams.sizes(config.logical_batch_size, shares)
    total = sum(shards.macros for shards, _ in data.values())
    primary = max(waves, key=lambda name: data[name][0].macros)
    window = max(1, window_shards // len(waves))
    feeders = streams.stream_feeders(config, shares)
    parts = {name: (streams.batches(learner, shards,
                                    streams.buffer_weights(shards.source_counts, weight_cap, phase_weights),
                                    counts[name], window, feeders[name]),
                    with_advantages(table, name, shards.macros / total))
             for name, (shards, table) in data.items()}
    joint = streams.Joint(parts, primary, updates)
    objective = Clipped(reference.model, clip=clip, kl=kl)

    def told(metrics):
        losses = {name: metrics["sources"].get(name, {}).get("loss") for name in parts}
        done = joint.last
        on_update(dict(
            metrics,
            **{f"{name}_loss": losses[name] for name in waves},
            **{f"{name}_zero_share": done[name]["zero"] / max(done[name]["samples"], 1) for name in waves}))

    metrics = learner.fit(joint, total=total, on_update=told if on_update else None, objective=objective,
                          accumulate=accumulate)
    metrics["seen"] = objective.summary()
    return metrics


# What a step of an update of act groups has always recorded.
STEP = ("loss", "entropy", "accuracy", "ratio_mean", "clip_fraction", "kl", "zero_share", "explored", "samples",
        "weight", "last_grad_norm", "loss_scale", "act1_loss", "act2_loss", "act3_loss", "act1_zero_share",
        "act2_zero_share", "act3_zero_share", "updates")


def test_with_the_act_groups_alone_an_update_is_the_one_it_was_step_by_step(world):
    waves = world["waves"]
    macros = batches.priced_macros(waves)
    updates = batches.pass_updates(macros, 8)
    assert updates >= 3
    before, now = learner(), learner()
    old, old_steps = update(before, waves, by=update_before, updates=updates, kl=0.1)
    new, new_steps = update(now, waves, updates=updates, kl=0.1)
    assert len(old_steps) == len(new_steps) == updates
    for was, step in zip(old_steps, new_steps):
        # The same batches, losses, KL and clipping, to the bit.
        assert {key: step[key] for key in STEP} == {key: was[key] for key in STEP}
        assert step["sources"] == was["sources"]
        assert set(step["mix"]) == {ACT} and step["mix"][ACT]["kl"] == step["kl"]
    for one, other in zip(before.model.parameters(), now.model.parameters()):
        assert torch.equal(one, other)
    # The pass's means add the same tallies step by step instead of microbatch by microbatch.
    assert new["loss"] == old["loss"] and new["groups"]["seen"] == pytest.approx(old["seen"], rel=1e-12)
    assert new["groups"]["shares"] == batches.wave_shares(macros) and "demos" not in new["groups"]
    mix = new["groups"]["mix"]
    assert set(mix) == {ACT} and mix[ACT]["decisions"] == sum(macros.values()) and mix[ACT]["share"] == 1.0
    assert mix[ACT]["loss"] == pytest.approx(new["loss"]) and mix[ACT]["samples"] == new["groups"]["seen"]["samples"]


def test_demonstrations_share_the_batches_of_the_act_groups_one_to_one(world):
    waves = world["waves"]
    macros = batches.priced_macros(waves)
    pool = combat_demos.fights([world["demos"]])
    shares = batches.stream_shares(macros, {DEMOS: pool.macros}, sources={ACT: 1, DEMOS: 1})
    sizes = streams.sizes(8, shares)
    assert sizes[DEMOS] == 4 and sum(sizes[wave] for wave in waves) == 4
    # A pass sees every act decision about once; the demonstrations as many.
    updates = batches.pass_updates(macros, 4)
    metrics, steps = update(learner(), waves, demos=pool, demo_key="test:0", shares=shares, updates=updates, kl=0.1)
    done = metrics["groups"]
    drawn = done["demos"]
    assert drawn["quota"] == updates * 4 and drawn["quota"] <= drawn["drawn"] < drawn["quota"] + 3
    assert drawn["fights"] == 6 and drawn["macros"] == 18 and drawn["passes"] == updates * 4 // 18
    assert drawn["samples"] == sum(step["mix"][DEMOS]["samples"] for step in steps)
    assert abs(drawn["samples"] - updates * 4) <= 4 * (drawn["passes"] + 1)
    mix = done["mix"]
    assert set(mix) == {ACT, DEMOS} and mix[DEMOS]["share"] == 0.5 and mix[DEMOS]["size"] == 4
    assert mix[DEMOS]["decisions"] == 18 and mix[ACT]["decisions"] == sum(macros.values())
    assert mix[DEMOS]["loss"] == pytest.approx(mix[DEMOS]["nll"], rel=1e-6) and mix[DEMOS]["loss"] > 0
    assert mix[ACT]["kl"] > 0 and "kl" not in mix[DEMOS]
    assert abs(mix[DEMOS]["samples"] - mix[ACT]["samples"]) <= 4 * (drawn["passes"] + 2)
    # The top-level means are the sampled decisions'; the demonstrations are a source of the loss.
    assert done["seen"]["samples"] == mix[ACT]["samples"] and done["seen"]["kl"] == mix[ACT]["kl"]
    assert set(done["losses"]) == {"act1", "act2", "act3", DEMOS}
    for step in steps:
        assert set(step["mix"]) == {ACT, DEMOS} and step["mix"][DEMOS]["loss"] > 0
        assert step["kl"] == step["mix"][ACT]["kl"]
    # Every demonstrated macro-action weighs, each time, what an act decision does on average.
    assert metrics["sources"][DEMOS]["weight"] / mix[DEMOS]["samples"] == pytest.approx(
        sum(metrics["sources"][wave]["weight"] for wave in waves) / mix[ACT]["samples"], rel=0.25)
    # The same draw for the same key.
    again, _ = update(learner(), waves, demos=pool, demo_key="test:0", shares=shares, updates=updates, kl=0.1)
    assert again["groups"]["demos"] == drawn


def test_fight_groups_are_a_stream_of_their_own_learned_as_sampled_decisions(world):
    """Synthetic fight groups: a priced directory of the same format as their wave."""
    waves = {name: world["waves"][name] for name in ("act1", "act2")}
    fights = world["waves"]["act3"]
    macros = batches.priced_macros(waves)
    pool = combat_demos.fights([world["demos"]])
    others = {FIGHTS: batches.priced_macros({FIGHTS: fights})[FIGHTS], DEMOS: pool.macros}
    shares = batches.stream_shares(macros, others, sources={ACT: 2, FIGHTS: 1, DEMOS: 1})
    metrics, steps = update(learner(), waves, fights=fights, demos=pool, shares=shares, updates=3, kl=0.1)
    done = metrics["groups"]
    mix = done["mix"]
    assert set(mix) == {ACT, FIGHTS, DEMOS} and mix[FIGHTS]["share"] == pytest.approx(0.25)
    assert mix[FIGHTS]["samples"] > 0 and mix[FIGHTS]["kl"] > 0 and 0 <= mix[FIGHTS]["clip_fraction"] <= 1
    assert done["fights"]["samples"] == mix[FIGHTS]["samples"] and done["fights"]["data"] == extent(
        batches.priced_waves({FIGHTS: fights})[FIGHTS][0])
    assert done["seen"]["samples"] == mix[ACT]["samples"] + mix[FIGHTS]["samples"]
    assert all(set(step["mix"]) == {ACT, FIGHTS, DEMOS} for step in steps) and len(steps) == 3
    with pytest.raises(ValueError, match="every stream"):
        update(learner(), waves, fights=fights, shares=batches.stream_shares(macros), updates=1)


# ---- The driver ----

def test_an_iteration_learns_the_fights_of_combat_demonstrations_in_its_rounds_batches(tmp_path):
    from types import SimpleNamespace

    from main.cli import parser
    from main.iterate import Driver, Settings
    from main.tests import journeys
    from main.tests.test_iterate import chain, lines

    demos = demo_directory(tmp_path / "demos")
    listed = sorted(path.relative_to(demos) for path in demos.rglob("*"))
    checkpoint = journeys.checkpoint(tmp_path / "policy")

    def settings(**changes):
        return Settings(**{**dict(checkpoint=str(checkpoint), boundaries=[2, 3],
                                  group_n=2, group_m=[2, 2], runs_per_character=1, seed_root="test", workers=2,
                                  combat_demos=[str(demos)], source_shares={ACT: 1, DEMOS: 1}), **changes})

    world = SimpleNamespace(output=tmp_path / "iterate", checkpoints=tmp_path / "checkpoints/iterate")
    state = Driver(world.output, settings(), checkpoints=world.checkpoints, engine=chain, say=lambda _: None).run(1)
    entry = state["iterations"]["0"]
    assert entry["settings"]["combat_demos"] == [str(demos)]
    assert entry["settings"]["source_shares"] == {ACT: 1.0, DEMOS: 1.0}
    update = entry["update"]
    decisions = sum(head["decisions"] for head in entry["advantages"].values())
    # Half of every batch is demonstrations: a pass sees every act decision about once and as many of them.
    assert update["shares"][DEMOS] == 0.5 and update["sizes"][DEMOS] == 4
    assert update["pass_updates"] == -(-decisions // 4)
    drawn = update["demos"]
    assert drawn["key"] == "test:0" and drawn["directories"] == [str(demos)] and drawn["fights"] == 6
    assert drawn["quota"] == update["pass_updates"] * 4 and drawn["passes"] == drawn["quota"] // 18
    mix = update["mix"]
    assert set(mix) == {ACT, DEMOS} and mix[ACT]["decisions"] == decisions and mix[DEMOS]["decisions"] == 18
    assert mix[DEMOS]["loss"] > 0 and mix[DEMOS]["samples"] == drawn["samples"] > 0 and mix[ACT]["kl"] >= 0
    assert update["seen"]["kl"] == mix[ACT]["kl"]
    metrics = lines(world.output / "history.jsonl")[0]["metrics"]
    assert metrics["kl"] == metrics["act_groups_kl"] and metrics["combat_demos_loss"] == mix[DEMOS]["loss"]
    assert metrics["combat_demos_samples"] == mix[DEMOS]["samples"] and "combat_demos_kl" not in metrics
    assert metrics["combat_demos_repeats"] == drawn["repeats"] and metrics["combat_demos_accuracy"] is not None
    steps = lines(world.output / "updates.jsonl")
    assert len(steps) == update["pass_updates"]
    assert all(set(step["metrics"]["mix"]) == {ACT, DEMOS} for step in steps)
    # The demonstrations are read, never cleaned up.
    assert sorted(path.relative_to(demos) for path in demos.rglob("*")) == listed

    # What the settings refuse; shares that name a source the update does not have fail at the update.
    for wrong in (dict(source_shares={"solver": 1}), dict(source_shares={ACT: 0}), dict(source_shares={}),
                  dict(combat_demos=[])):
        with pytest.raises(ValueError):
            settings(**wrong)
    with pytest.raises(ValueError, match="every kind"):
        settings(combat_demos=None).group_shares({"act1": 10, "act2": 10, "act3": 10})
    assert settings(combat_demos=None, source_shares=None).group_shares({"act1": 10, "act2": 30}) == {
        "act1": 0.25, "act2": 0.75}
    common = ["iterate", "--checkpoint", "c", "--output", "o", "--iterations", "1"]
    args = parser().parse_args([*common, "--combat-demos", "a", "b", "--source-shares", "act-groups=1",
                                "combat-demos=0.5"])
    assert args.combat_demos == ["a", "b"] and args.source_shares == [(ACT, 1.0), (DEMOS, 0.5)]
    assert parser().parse_args(common).combat_demos is None and parser().parse_args(common).source_shares is None
    with pytest.raises(SystemExit):
        parser().parse_args([*common, "--source-shares", "act-groups:1"])
