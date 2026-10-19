"""Advantages of act groups without a critic (`main.learn.groups`): returns passed
back up the chain of waves, the tree of shared prefixes, the within-group ridge
baseline."""

import numpy as np
import pytest

from main.env.protocol import CHARACTERS
from main.learn import advantage
from main.learn.features import NAMES
from main.learn.groups import Decision, Segment, advantages, fit, headers, returns, share, tree, wave_advantages
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
