"""Advantages of act groups without a critic (`main.learn.groups`): returns passed
back up the chain of waves, the tree of shared prefixes, the within-group ridge
baseline."""

import gc
import gzip
import json

import numpy as np
import pytest

from main.data.shards import RunShards
from main.env.protocol import CHARACTERS
from main.learn import advantage, batches
from main.learn.features import NAMES
from main.learn.groups import advantages, headers, share, wave_advantages
from main.learn.segments import Decision, Segment, fit, read_waves, returns, tree
from main.play import digests
from main.play.digests import where
from main.play.groups import sample_groups
from main.play.sampling import Samplers
from main.tests.journeys import learner
from main.tests.test_groups import BOUNDARIES, engine, runner


def decision(index, key, action, **features):
    row = [0.0] * len(NAMES)
    for name, value in features.items():
        row[NAMES.index(name)] = value
    return Decision(index, key, (action,), row)


def segment(run_id, group, outcome, decisions, character="Ironclad"):
    return Segment(run_id, group, character, outcome, decisions)


def test_the_state_summary_counts_the_deck_the_relics_and_the_potions():
    entities = [dict(entity_type="card", zone="hand"), dict(entity_type="card", zone="deck"),
                dict(entity_type="player", act=2, floor=20, hp=30, max_hp=80, gold=99),
                dict(entity_type="card", zone="deck"), dict(entity_type="relic"), dict(entity_type="potion"),
                dict(entity_type="player", act=3, floor=40), dict(entity_type="enemy")]
    assert where(dict(public=dict(entities=entities))) == (
        2, 20, dict(hp=30, max_hp=80, gold=99, deck=2, relics=1, potions=1))
    with pytest.raises(ValueError, match="player"):
        where(dict(public=dict(entities=entities[:2])))


def test_a_segments_return_is_what_the_group_played_from_its_save_came_to():
    last = [segment("f0:act2-0:act3-0", "f0:act2-0:act3", "win", []),
            segment("f0:act2-0:act3-1", "f0:act2-0:act3", "loss", []),
            segment("f0:act2-0:act3-2", "f0:act2-0:act3", "win", []),
            segment("f0:act2-1:act3-0", "f0:act2-1:act3", "loss", [])]
    middle = [segment("f0:act2-0", "f0:act2", "reached", []), segment("f0:act2-1", "f0:act2", "reached", []),
              segment("f0:act2-2", "f0:act2", "loss", []), segment("f0:act2-3", "f0:act2", "reached", [])]
    first = [segment("f0", "g", "reached", []), segment("f1", "g", "loss", []), segment("f2", "g", "reached", [])]
    table = returns([("act1", first), ("act2", middle), ("act3", last)])
    assert table["f0:act2-0:act3-0"] == 1.0 and table["f0:act2-0:act3-1"] == -1.0
    assert table["f0:act2-0"] == pytest.approx(1 / 3) and table["f0:act2-1"] == -1.0 and table["f0:act2-2"] == -1.0
    assert "f0:act2-3" not in table  # nothing finished from its save
    # The first wave's return is the mean of what was played from its save, two waves down.
    assert table["f0"] == pytest.approx((1 / 3 - 1.0 - 1.0) / 3) and table["f1"] == -1.0
    assert "f2" not in table
    # Two waves: the earlier takes the later's mean directly.
    assert returns([("act1", first), ("act3", last)]) == {s.run_id: (1.0 if s.outcome == "win" else -1.0) for s in last} | {"f1": -1.0}


def test_the_tree_compares_actions_at_the_frames_the_segments_share():
    # Four segments share the first frame; two chose a, two chose b. The a pair sees
    # the same second frame and chooses alike; the b pair's second frames differ.
    members = [
        segment("s0", "g", "win", [decision(0, "f0", "a"), decision(1, "f1a", "x")]),
        segment("s1", "g", "loss", [decision(0, "f0", "a"), decision(1, "f1a", "x")]),
        segment("s2", "g", "win", [decision(0, "f0", "b"), decision(1, "f1b", "x")]),
        segment("s3", "g", "win", [decision(0, "f0", "b"), decision(1, "f1c", "y")]),
    ]
    table = {"s0": 1.0, "s1": -1.0, "s2": 1.0, "s3": 1.0}
    found, counts = tree(members, table)
    assert found[("s0", 0)] == (0.5, -0.5) and found[("s1", 0)] == (0.5, -0.5)
    assert found[("s2", 0)] == (0.5, 0.5) and found[("s3", 0)] == (0.5, 0.5)
    assert found[("s0", 1)] == (0.0, 0.0) and found[("s1", 1)] == (0.0, 0.0)  # one action: nothing to compare
    assert ("s2", 1) not in found and ("s3", 1) not in found
    assert counts == {"nodes": 2, "decisions": 6, "branching_nodes": 1, "branching_decisions": 4}


def test_the_baseline_is_fitted_within_groups_and_its_corrections_sum_to_zero():
    # Within each group the return rises with HP; between groups the levels differ.
    members = {}
    for g, level in (("g0", 0.0), ("g1", 0.6)):
        for k, (hp, outcome) in enumerate(((0.2, "loss"), (0.9, "win"), (0.5, "loss"), (0.8, "win"))):
            members.setdefault(g, []).append(segment(
                f"{g}-{k}", g, outcome, [decision(0, f"{g}-{k}-0", "a", hp_ratio=hp),
                                         decision(1, f"{g}-{k}-1", "a", hp_ratio=hp + 0.05)]))
    table = {s.run_id: (1.0 if s.outcome == "win" else -1.0) + (0.6 if s.group == "g1" else 0.0)
             for group in members.values() for s in group}
    weights, r2, fitted = fit(members, table, ridge=0.01)
    assert fitted == 16 and r2 > 0.8 and weights[NAMES.index("hp_ratio")] > 1.0
    runs, stats = wave_advantages([s for group in members.values() for s in group], table, ridge=0.01)
    for group in members.values():
        corrections = [value - np.mean([table[s.run_id] for s in group])
                       for s in group for _, value, _ in runs[s.run_id]]
        assert sum(corrections) == pytest.approx(0.0, abs=1e-9)
    assert stats["groups"] == 2 and stats["segments"] == 8 and stats["decisions"] == 16
    assert stats["tree"]["nodes"] == 0 and stats["baseline"]["r2_within"] == pytest.approx(r2)
    assert stats["between_share"] is not None and stats["uniform_groups"] == 0


def test_dropping_the_uniform_groups_leaves_their_decisions_unpriced_and_out_of_the_baseline():
    # Two groups whose returns rise with HP, and two whose segments all came to one
    # return whatever their HP: fitted on, the latter pull the weight on HP towards zero.
    members = {}
    for g, outcomes in (("g0", ("loss", "win", "loss", "win")), ("g1", ("loss", "win", "loss", "win")),
                        ("g2", ("win",) * 4), ("g3", ("loss",) * 4)):
        for k, (hp, outcome) in enumerate(zip((0.2, 0.9, 0.5, 0.8), outcomes)):
            members.setdefault(g, []).append(segment(
                f"{g}-{k}", g, outcome, [decision(0, f"{g}-{k}-0", "a", hp_ratio=hp),
                                         decision(1, f"{g}-{k}-1", "a", hp_ratio=hp + 0.05)]))
    members["g0"][0].decisions[1].explored = members["g2"][0].decisions[1].explored = True
    table = {s.run_id: 1.0 if s.outcome == "win" else -1.0 for group in members.values() for s in group}
    wave = [s for group in members.values() for s in group]
    kept = members["g0"] + members["g1"]
    runs, stats = wave_advantages(wave, table, ridge=0.01)
    dropped_runs, dropped = wave_advantages(wave, table, ridge=0.01, drop_uniform=True)
    # Without the switch every group is priced.
    assert set(runs) == {s.run_id for s in wave} and "dropped" not in stats
    assert (stats["groups"], stats["uniform_groups"], stats["explored"]["decisions"]) == (4, 2, 2)
    # With it the uniform groups' segments have no advantage, as a segment without a return has none.
    assert set(dropped_runs) == {s.run_id for s in kept}
    assert dropped["dropped"] == dict(groups=2, segments=8, decisions=16)
    assert (dropped["segments"], dropped["decisions"], dropped["explored"]["decisions"]) == (8, 16, 1)
    # The groups kept are priced as though the wave held them alone: the baseline is theirs.
    alone_runs, alone = wave_advantages(kept, table, ridge=0.01)
    assert dropped_runs == alone_runs
    assert dropped["baseline"] == alone["baseline"] and dropped["baseline"]["fitted"] == 16
    assert dropped["tree"] == alone["tree"] and dropped["explored"] == alone["explored"]
    assert dropped["baseline"]["weights"]["hp_ratio"] > stats["baseline"]["weights"]["hp_ratio"]
    # What the groups came to is still told of all of them.
    for key in ("groups", "uniform_groups", "return_mean", "between_share"):
        assert dropped[key] == stats[key]
    for level, counts in stats["by_ascension"].items():
        assert {key: value for key, value in dropped["by_ascension"][level].items() if key != "explored"} == \
            {key: value for key, value in counts.items() if key != "explored"}
    # A wave without a uniform group is priced alike with the switch or without.
    assert wave_advantages(kept, table, ridge=0.01, drop_uniform=True)[0] == alone_runs


def test_the_between_group_share_tells_group_means_from_noise():
    rng = np.random.default_rng(0)
    apart = [[m + rng.normal(0, 0.1) for _ in range(8)] for m in np.linspace(-1, 1, 20)]
    alike = [[rng.normal(0, 1) for _ in range(8)] for _ in range(20)]
    assert share(apart)[0] > 0.9 and share(alike)[0] < 0.2
    assert share([[1.0]])[0] is None


def test_a_sampled_round_of_groups_gets_advantages_for_every_decision(tmp_path):
    policy = learner()
    seeds = {character: [f"{character}-seed"] for character in CHARACTERS}
    with Samplers(engine, policy.model, policy.vocabulary, engines=4, processes=2) as pool:
        sample_groups(pool, runner(policy), seeds, tmp_path / "groups", n=2, m=2, boundaries=BOUNDARIES)
    written = advantages(tmp_path / "groups", processes=2)
    assert list(written) == ["act1", "act2", "act3"] and headers(tmp_path / "groups") == written
    for wave in written:
        head = written[wave]
        assert head["kind"] == "act-groups" and head["wave"] == wave and head["decisions"] > 0
        assert head["boundaries"] == [2, 3] and "critic" not in head
        stats = head["stats"]
        assert stats["segments"] == head["runs"] and stats["decisions"] == head["decisions"]
        assert 0 <= stats["tree"]["share"] <= 1 and len(stats["baseline"]["weights"]) == len(NAMES)
        table = advantage.read([tmp_path / "groups" / wave])
        assert len(table) == head["decisions"]
    # Every segment reached its cut: its return is the mean of the group played from its save.
    assert [written[wave]["runs"] for wave in written] == [10, 20, 40]
    # The segments of a group start from one frame: the tree holds that node.
    assert written["act2"]["stats"]["tree"]["nodes"] >= 10 and written["act3"]["stats"]["tree"]["nodes"] >= 20
    # Written once: asking again reads the headers back.
    assert advantages(tmp_path / "groups") == written
    assert all(written[wave]["stats"]["forked"]["decisions"] == 0 for wave in written)


def test_a_round_priced_without_its_uniform_groups_trains_on_the_rest(tmp_path):
    policy = learner()
    seeds = {character: [f"{character}-seed"] for character in CHARACTERS}
    with Samplers(engine, policy.model, policy.vocabulary, engines=4, processes=2) as pool:
        sample_groups(pool, runner(policy), seeds, tmp_path / "groups", n=2, m=2, boundaries=BOUNDARIES)
    waves = ["act1", "act2", "act3"]
    plain = advantages(tmp_path / "groups", processes=2)
    kept = {wave: unzipped(tmp_path / "groups" / wave / advantage.FILE) for wave in waves}
    # Priced again with the switch: the round's advantages are another, and are computed anew.
    dropped = advantages(tmp_path / "groups", processes=2, drop_uniform=True)
    assert sum(dropped[wave]["stats"]["dropped"]["groups"] for wave in waves) > 0
    for wave in waves:
        head, before = dropped[wave], plain[wave]
        stats = head["stats"]
        assert head["drop_uniform_groups"] is True and "drop_uniform_groups" not in before
        assert stats["dropped"]["groups"] == stats["uniform_groups"] == before["stats"]["uniform_groups"]
        assert stats["groups"] == before["stats"]["groups"]
        assert head["runs"] + stats["dropped"]["segments"] == before["runs"]
        assert head["decisions"] + stats["dropped"]["decisions"] == before["decisions"]
        # What an update reads of the wave is the segments kept.
        shards, table = batches.priced(tmp_path / "groups" / wave)
        assert shards.macros == len(table) == head["decisions"] and len(shards.rows) == head["runs"]
    assert advantages(tmp_path / "groups", processes=2, drop_uniform=True) == dropped
    # Without the switch again: as they were, to the byte.
    assert advantages(tmp_path / "groups", processes=2) == plain
    assert {wave: unzipped(tmp_path / "groups" / wave / advantage.FILE) for wave in waves} == kept


def test_the_shards_of_every_wave_are_read_in_one_pool_as_in_one_process(tmp_path):
    policy = learner()
    seeds = {character: [f"{character}-seed"] for character in CHARACTERS}
    with Samplers(engine, policy.model, policy.vocabulary, engines=4, processes=2) as pool:
        sample_groups(pool, runner(policy), seeds, tmp_path / "groups", n=2, m=2, boundaries=BOUNDARIES,
                      shard_samples=20)
    waves = ["act1", "act2", "act3"]
    assert all(len(RunShards(tmp_path / "groups" / wave).window_names(shards=1)) > 1 for wave in waves)
    alone = read_waves(tmp_path / "groups", waves, processes=1)
    pooled = read_waves(tmp_path / "groups", waves, processes=3)
    # The same segments, wave by wave in the order of their shards, whoever read them.
    assert list(pooled) == waves and pooled == alone
    assert [len(alone[wave]) for wave in waves] == [10, 20, 40]
    # The pool's workers inherit the heap frozen; it is thawed again after them.
    assert gc.get_freeze_count() == 0


def unzipped(path):
    with gzip.open(path, "rb") as stream:
        return stream.read()


def test_a_round_keeps_a_digest_of_its_decisions_and_the_advantages_need_nothing_else(tmp_path, monkeypatch):
    policy = learner()
    seeds = {character: [f"{character}-seed"] for character in CHARACTERS}
    with Samplers(engine, policy.model, policy.vocabulary, engines=4, processes=2) as pool:
        sample_groups(pool, runner(policy), seeds, tmp_path / "groups", n=2, m=2, boundaries=BOUNDARIES,
                      shard_samples=20, fork_depth=1)
    waves = ["act1", "act2", "act3"]
    # Every segment a wave wrote has its line in the wave's digest.
    for wave in waves:
        listed = [json.loads(line)["run_id"] for line in gzip.open(tmp_path / "groups" / wave / digests.FILE, "rt")]
        assert sorted(listed) == sorted(RunShards(tmp_path / "groups" / wave).identities)
    # Read from the digests, the waves are what their shards say, and no shard is opened.
    for wave in waves:
        (tmp_path / "groups" / wave / digests.FILE).rename(tmp_path / f"{wave}.digest")
    from_shards = read_waves(tmp_path / "groups", waves, processes=1)
    for wave in waves:
        (tmp_path / f"{wave}.digest").rename(tmp_path / "groups" / wave / digests.FILE)

    def unread(*_):
        raise AssertionError("a shard was read")

    with monkeypatch.context() as patched:
        patched.setattr(RunShards, "runs", unread)
        assert read_waves(tmp_path / "groups", waves, processes=1) == from_shards
        written = advantages(tmp_path / "groups", processes=2)
    assert all(written[wave]["stats"]["forked"]["decisions"] > 0 for wave in waves)
    kept = {wave: unzipped(tmp_path / "groups" / wave / advantage.FILE) for wave in waves}
    # The same advantage files, to the byte, as from the shards.
    for wave in waves:
        (tmp_path / "groups" / wave / advantage.FILE).unlink()
        (tmp_path / "groups" / wave / digests.FILE).unlink()
    advantages(tmp_path / "groups", processes=2)
    assert {wave: unzipped(tmp_path / "groups" / wave / advantage.FILE) for wave in waves} == kept


def test_a_wave_whose_digest_misses_a_segment_is_read_from_its_shards(tmp_path):
    policy = learner()
    seeds = {character: [f"{character}-seed"] for character in CHARACTERS}
    with Samplers(engine, policy.model, policy.vocabulary, engines=4, processes=2) as pool:
        sample_groups(pool, runner(policy), seeds, tmp_path / "groups", n=2, m=2, boundaries=BOUNDARIES)
    waves = ["act1", "act2", "act3"]
    complete = read_waves(tmp_path / "groups", waves, processes=1)
    # A digest cut short (the round was killed between a shard's line and its digest's).
    path = tmp_path / "groups" / "act2" / digests.FILE
    lines = gzip.open(path, "rt").readlines()
    with gzip.open(path, "wt") as stream:
        stream.writelines(lines[:-1])
    assert read_waves(tmp_path / "groups", waves, processes=1) == complete


def test_a_forked_round_compares_the_two_halves_of_every_group_at_its_first_choice(tmp_path):
    policy = learner()
    seeds = {character: [f"{character}-seed"] for character in CHARACTERS}
    with Samplers(engine, policy.model, policy.vocabulary, engines=4, processes=2) as pool:
        summary = sample_groups(pool, runner(policy), seeds, tmp_path / "groups", n=2, m=2, boundaries=BOUNDARIES,
                                fork_depth=1)
    assert summary["fork_depth"] == 1
    # Every segment forked once: at its first decision with a choice.
    assert all(summary[wave]["forked"] == summary[wave]["written"] for wave in summary["waves"])
    written = advantages(tmp_path / "groups", processes=2)
    for wave in written:
        stats = written[wave]["stats"]
        assert stats["forked"]["decisions"] == stats["segments"]
        # The two segments of every group share their first frame and part there, one
        # each way: the tree has a node with two actions in every group.
        assert stats["tree"]["branching_nodes"] >= stats["groups"]
        # Each half is measured against the other: their advantages are opposite.
        assert stats["forked"]["positive"] == stats["forked"]["negative"]


def test_a_round_with_greedy_fights_prices_and_trains_the_decisions_outside_them_alone(tmp_path):
    from main.data.shards import samples
    from main.learn import streams
    from main.learn.clipped import with_advantages
    from main.tests.test_groups import greedy

    policy = learner()
    seeds = {character: [f"{character}-seed"] for character in CHARACTERS}
    directory = tmp_path / "groups"
    with Samplers(engine, policy.model, policy.vocabulary, engines=4, processes=2) as pool:
        sample_groups(pool, greedy(policy), seeds, directory, n=2, m=2, boundaries=BOUNDARIES)
    waves = ["act1", "act2", "act3"]
    written = advantages(directory, processes=2)
    for wave in waves:
        head = written[wave]
        # No decision of a fight has an advantage; the header counts those left out.
        assert head["stats"]["fights"]["decisions"] == 0 and head["stats"]["fights"]["excluded"] > 0
        shards, table = batches.priced(directory / wave)
        assert shards.macros == len(table) == head["decisions"] and len(shards.rows) == head["runs"]
        assert all(phase != "combat" for _, phase, _ in shards.source_counts)
        runs = [run for window in shards.windows(shards=8) for run in window]
        by_id = {run["run_id"]: run for run in runs}
        assert all(by_id[run_id]["macros"][index]["fight"] is None for run_id, index in table)
        # What an update's stream of the wave reads: the decisions outside the fights,
        # each with its advantage; the fights in no batch and no count.
        items = samples(runs, weights=streams.buffer_weights(shards.source_counts))
        assert {(item.run_id, item.index) for item in items} == set(table)
        assert all(item.macro["fight"] is None and item.temperature == 0.8 for item in items)
        weights = streams.buffer_weights(shards.source_counts)
        read = list(streams.batches(policy, shards, weights, 8, 8, 2)())
        priced, done = with_advantages(table, wave, 1.0)([part for batch in read for part in batch])
        assert done["samples"] == len(table)
        assert all(item.macro["fight"] is None for micro, _ in priced for item in micro)
    # Read from the shards, a wave is what its digest says, the fights left out alike.
    for wave in waves:
        (directory / wave / digests.FILE).rename(tmp_path / f"{wave}.digest")
    from_shards = read_waves(directory, waves, processes=1)
    for wave in waves:
        (tmp_path / f"{wave}.digest").rename(directory / wave / digests.FILE)
    assert read_waves(directory, waves, processes=1) == from_shards
    assert all(segment.frameless > 0 for wave in waves for segment in from_shards[wave])
    # Without greedy fights every fight's decision is priced and none is left out.
    with Samplers(engine, policy.model, policy.vocabulary, engines=4, processes=2) as pool:
        sample_groups(pool, runner(policy), seeds, tmp_path / "plain", n=2, m=2, boundaries=BOUNDARIES)
    plain = advantages(tmp_path / "plain", processes=2)
    assert all(plain[wave]["stats"]["fights"]["decisions"] > 0 and plain[wave]["stats"]["fights"]["excluded"] == 0
               for wave in waves)
