"""Advantages of fight groups (`main.learn.fight_groups`): each segment's return its
fight's cost at the prices given, the group's mean as the baseline, the tree of shared
prefixes, the uniform groups, the diagnostics by kind of fight."""

import copy
import json

import pytest

from main.data.shards import samples
from main.env.protocol import CHARACTERS
from main.learn import advantage, batches, costs, streams
from main.learn.clipped import with_advantages
from main.learn.features import NAMES
from main.learn.fight_groups import KIND, advantages, describe, wave_advantages
from main.learn.segments import Decision, Segment
from main.play.fight_groups import WAVE, sample_fight_groups
from main.play.fights import FILE, fight_rows
from main.play.groups import sample_groups
from main.play.sampling import Samplers
from main.tests.journeys import learner
from main.tests.test_fight_groups import chain, runner
from main.iterate import fight_cost

FIRE, ROCK = "POTION.FIRE_POTION", "POTION.POTION_SHAPED_ROCK"


def decision(index, key, action):
    return Decision(index, key, (action,), [0.0] * len(NAMES))


def segment(run_id, group, outcome, moves):
    """A segment whose `moves` are (frame, action) pairs, its decisions in order."""
    return Segment(run_id, group, "Ironclad", outcome, [decision(i, key, action) for i, (key, action) in enumerate(moves)])


def record(run_id, group, kind="elite", act=1, entry=50, exit=40, died=False, potions=(), ascension=0, last=None):
    """A fight segment's record; won at the run's last boss it has no exit HP (`exit` None)."""
    return dict(run_id=run_id, group=group, fight=1, kind=kind, act=act, floor=7, second_boss=False,
                ascension=ascension, entry_hp=entry, entry_max_hp=80, exit_hp=0 if died else exit,
                exit_max_hp=None if exit is None and not died else 80, last_hp=exit if last is None else last,
                last_max_hp=80, potions_used=list(potions), died=died)


def elite_group(name="e"):
    """Four segments of one elite fight entered at 50 HP: two take `a` first and see the
    same frame after it, two take `b` and part at once; one of these uses a fire potion,
    the other dies."""
    members = [
        segment(f"{name}-0", name, "reached", [("f0", "a"), ("f1a", "x"), ("f2a0", "p")]),
        segment(f"{name}-1", name, "reached", [("f0", "a"), ("f1a", "x"), ("f2a1", "q")]),
        segment(f"{name}-2", name, "reached", [("f0", "b"), ("f1b", "y")]),
        segment(f"{name}-3", name, "loss", [("f0", "b"), ("f1c", "z")]),
    ]
    rows = [record(f"{name}-0", name, exit=45), record(f"{name}-1", name, exit=35),
            record(f"{name}-2", name, exit=48, potions=[FIRE]), record(f"{name}-3", name, died=True)]
    return members, rows


def uniform_group(name="n", kind="normal"):
    """Four segments of one fight that act differently and come to the same record."""
    members = [segment(f"{name}-{k}", name, "reached", [("g0", "ab"[k % 2]), (f"g1-{k}", "x")]) for k in range(4)]
    return members, [record(f"{name}-{k}", name, kind=kind, exit=43) for k in range(4)]


def by_decision(runs):
    return {(run, index): (value, adv) for run, entries in runs.items() for index, value, adv in entries}


def test_a_groups_mean_is_the_baseline_and_its_shared_prefix_gives_each_action_its_own():
    members, rows = elite_group()
    runs, stats = wave_advantages(members, rows, costs.FIXED)
    # The returns: −cost / 100, a fire potion at 7 HP, a death at 100 with the HP down to 0.
    returns = {"e-0": -0.05, "e-1": -0.15, "e-2": -0.09, "e-3": -1.5}
    mean = sum(returns.values()) / 4
    found = by_decision(runs)
    # At the first frame, which all four share: each action's mean against the node's.
    for run in ("e-0", "e-1"):
        assert found[run, 0] == pytest.approx((mean, -0.1 - mean))
    for run in ("e-2", "e-3"):
        assert found[run, 0] == pytest.approx((mean, (-0.09 - 1.5) / 2 - mean))
    # A node of one action teaches nothing.
    assert found["e-0", 1] == found["e-1", 1] == pytest.approx((-0.1, 0.0))
    # Parted, a decision is its segment's return against the group's mean.
    for run, index in (("e-0", 2), ("e-1", 2), ("e-2", 1), ("e-3", 1)):
        assert found[run, index] == pytest.approx((mean, returns[run] - mean))
    assert stats["tree"] == dict(nodes=2, decisions=6, branching_nodes=1, branching_decisions=4,
                                 share=pytest.approx(6 / 10), branching_share=pytest.approx(4 / 10))
    assert (stats["groups"], stats["segments"], stats["decisions"], stats["uniform_groups"]) == (1, 4, 10, 0)
    assert stats["return_mean"] == pytest.approx(mean, abs=1e-5)
    assert stats["within_variance"] == pytest.approx(
        sum((value - mean) ** 2 for value in returns.values()) / 3, abs=1e-6)


def test_a_uniform_groups_advantages_are_zero():
    members, rows = uniform_group()
    runs, stats = wave_advantages(members, rows, costs.FIXED)
    # Two actions at the shared frame, and nothing to tell them apart by.
    assert stats["tree"]["branching_nodes"] == 1 and stats["uniform_groups"] == 1
    assert all(adv == 0.0 and value == -0.07 for entries in runs.values() for _, value, adv in entries)
    assert sum(len(entries) for entries in runs.values()) == 8
    assert stats["within_variance"] == 0.0 and stats["kinds"]["normal"]["uniform_groups"] == 1


def test_other_prices_price_the_same_records_again():
    members, rows = elite_group()
    kept = copy.deepcopy(rows)
    fixed, _ = wave_advantages(members, rows, costs.FIXED)
    dear = costs.Prices(tiers=(30.0, 30.0, 30.0))
    runs, stats = wave_advantages(members, rows, dear)
    assert rows == kept
    # The potion now costs 30: the segment that used it comes to −0.32, and what took `b` pays.
    mean = (-0.05 - 0.15 - 0.32 - 1.5) / 4
    found = by_decision(runs)
    assert found["e-2", 0] == pytest.approx((mean, (-0.32 - 1.5) / 2 - mean))
    assert found["e-2", 1] == pytest.approx((mean, -0.32 - mean))
    assert found != by_decision(fixed)
    assert stats["kinds"]["elite"]["users_minus_others_cost"] == pytest.approx(32 - (5 + 15 + 150) / 3, abs=1e-5)
    # A price object of another kind serves as well, and says what it is in the header.
    class Free:
        def price(self, potion, fight):
            return 0.0

        def describe(self):
            return "free"

    free, _ = wave_advantages(members, rows, Free())
    assert by_decision(free)["e-2", 1] == pytest.approx(((-0.05 - 0.15 - 0.02 - 1.5) / 4, -0.02 + 1.72 / 4))
    assert describe(Free()) == "free" and describe(object()) == "object"
    assert describe(costs.FIXED) == dict(tiers=[7.0, 9.0, 11.0], final=[1.0, 1.0, 1.0], free=[ROCK])


def test_dropping_the_uniform_groups_leaves_them_out_whole():
    elite, elite_rows = elite_group()
    alike, alike_rows = uniform_group()
    wave, rows = elite + alike, elite_rows + alike_rows
    runs, stats = wave_advantages(wave, rows, costs.FIXED)
    dropped_runs, dropped = wave_advantages(wave, rows, costs.FIXED, drop_uniform=True)
    assert set(runs) == {s.run_id for s in wave} and "dropped" not in stats
    # Their segments have no advantage, and the groups kept are priced as though alone.
    assert set(dropped_runs) == {s.run_id for s in elite}
    assert dropped_runs == wave_advantages(elite, elite_rows, costs.FIXED)[0]
    assert dropped["dropped"] == dict(groups=1, segments=4, decisions=8)
    assert (dropped["segments"], dropped["decisions"]) == (4, 10)
    assert dropped["tree"] == wave_advantages(elite, elite_rows, costs.FIXED)[1]["tree"]
    # What the groups came to is still told of all of them.
    for key in ("groups", "uniform_groups", "return_mean", "between_share", "within_variance", "kinds"):
        assert dropped[key] == stats[key]
    assert stats["groups"] == 2 and stats["uniform_groups"] == 1


def test_a_fight_segment_needs_its_one_record():
    members, rows = elite_group()
    with pytest.raises(ValueError, match="no fight record"):
        wave_advantages(members, rows[1:], costs.FIXED)
    with pytest.raises(ValueError, match="more than one"):
        wave_advantages(members, rows + rows[:1], costs.FIXED)


def test_the_header_tells_each_kind_of_fight_how_its_groups_split():
    elite, elite_rows = elite_group()
    normal, normal_rows = uniform_group()
    # The run's last boss, at ascension 0: one segment wins it with two potions, at 30
    # HP of 60 (no decision after it), the other dies; another boss fight alike twice.
    final = [segment("b-0", "b", "win", [("h0", "a")]), segment("b-1", "b", "loss", [("h0", "b")])]
    final_rows = [record("b-0", "b", kind="boss", act=3, entry=60, exit=None, last=30, potions=[FIRE, FIRE]),
                  record("b-1", "b", kind="boss", act=3, entry=60, died=True)]
    second = [segment(f"c-{k}", "c", "reached", [("k0", "a")]) for k in range(2)]
    second_rows = [record(f"c-{k}", "c", kind="boss", act=2, entry=70, exit=50) for k in range(2)]
    wave = elite + normal + final + second
    rows = elite_rows + normal_rows + final_rows + second_rows
    _, stats = wave_advantages(wave, rows, costs.FIXED)
    kinds = stats["kinds"]
    assert kinds["normal"] == dict(
        groups=1, segments=4, uniform_groups=1, return_mean=-0.07, within_variance=0.0, death_rate=0.0,
        death_varies=0, potion_somewhere=0, potions_per_segment=0.0, potion_varies=0, users_minus_others_death=None,
        users_minus_others_hp_lost=None, n_hp_pairs=0, users_minus_others_cost=None)
    elite_kind = kinds["elite"]
    assert (elite_kind["groups"], elite_kind["death_rate"], elite_kind["death_varies"]) == (1, 0.25, 1)
    assert (elite_kind["potion_somewhere"], elite_kind["potions_per_segment"], elite_kind["potion_varies"]) == (1, 0.25, 1)
    # The user lived, the others died one in three; it lost 2 HP, the two others that lived 5 and 15.
    assert elite_kind["users_minus_others_death"] == pytest.approx(-1 / 3, abs=1e-5)
    assert elite_kind["users_minus_others_hp_lost"] == pytest.approx(2 - 10) and elite_kind["n_hp_pairs"] == 1
    assert elite_kind["users_minus_others_cost"] == pytest.approx(9 - (5 + 15 + 150) / 3, abs=1e-5)
    boss = kinds["boss"]
    assert (boss["groups"], boss["segments"], boss["uniform_groups"]) == (2, 4, 1)
    assert boss["death_rate"] == 0.25 and boss["death_varies"] == 1
    assert (boss["potion_somewhere"], boss["potions_per_segment"], boss["potion_varies"]) == (1, 0.5, 1)
    # Won at the run's end the fight has no HP after it to compare; its cost counts the HP
    # of its last decision and the potions at 1 each: 32 against the death's 60 + 100.
    assert boss["users_minus_others_death"] == -1.0 and boss["n_hp_pairs"] == 0
    assert boss["users_minus_others_hp_lost"] is None and boss["users_minus_others_cost"] == pytest.approx(32 - 160)
    # The second act's boss counts its HP a fifth: 20 HP is 4.
    assert boss["return_mean"] == pytest.approx((-0.32 - 1.6 - 0.04 - 0.04) / 4, abs=1e-5)
    assert boss["within_variance"] == pytest.approx(((-0.32 + 1.6) ** 2 / 2) / 2, abs=1e-6)
    assert stats["groups"] == 4 and stats["uniform_groups"] == 2 and stats["between_share"] is not None


# ---- A round's fight groups ----

def sampled_round(tmp_path):
    """A round of act groups with greedy fights and its fight groups, three segments each."""
    policy = learner()
    seeds = {character: [f"{character}-seed"] for character in CHARACTERS}
    greedy = runner(policy, greedy_fights=True)
    directory = tmp_path / "round"
    with Samplers(chain, policy.model, policy.vocabulary, engines=4, processes=2) as pool:
        sample_groups(pool, greedy, seeds, directory, n=2, m=[1, 1], fight_entries=True)
        sample_fight_groups(pool, greedy.variant(greedy_fights=False), directory, 11, cost=fight_cost,
                            key="test:0", size=3, fork_depth=1)
    return policy, directory


def test_a_rounds_fight_groups_are_priced_as_an_update_reads_an_act_wave(tmp_path):
    policy, directory = sampled_round(tmp_path)
    head = advantages(directory, costs.FIXED, processes=1)
    wave = directory / WAVE
    assert (head["format"], head["kind"], head["wave"]) == (advantage.FORMAT, KIND, WAVE)
    assert head["prices"] == describe(costs.FIXED) and head["written"] == 21 and "drop_uniform_groups" not in head
    stats = head["stats"]
    assert (stats["groups"], stats["segments"], head["runs"]) == (7, 21, 21)
    assert stats["decisions"] == head["decisions"] > 0 and stats["tree"]["nodes"] >= 7
    assert sum(kind["groups"] for kind in stats["kinds"].values()) == 7
    assert advantage.header(wave) == head
    # What an update reads of a wave: every decision of the segments priced, each once.
    shards, table = batches.priced(wave)
    assert shards.macros == len(table) == head["decisions"] and len(shards.rows) == head["runs"]
    runs = [run for window in shards.windows(shards=8) for run in window]
    items = samples(runs, weights=streams.buffer_weights(shards.source_counts))
    assert {(item.run_id, item.index) for item in items} == set(table)
    weights = streams.buffer_weights(shards.source_counts)
    read = list(streams.batches(policy, shards, weights, 8, 8, 2)())
    priced, done = with_advantages(table, WAVE, 1.0)([part for batch in read for part in batch])
    assert done["samples"] == len(table)
    assert batches.priced_waves({WAVE: wave})[WAVE][0].macros == len(table)
    # Say a segment of a normal fight used a potion: priced at other prices, its group's
    # advantages change and nothing else does, the records as they were.
    user = next(row for row in fight_rows(directory, WAVE) if row["kind"] == "normal")
    used = "".join(json.dumps(dict(row, potions_used=[FIRE]) if row["run_id"] == user["run_id"] else row) + "\n"
                   for row in map(json.loads, (wave / FILE).read_text().splitlines()))
    (wave / FILE).write_text(used)
    fixed = advantages(directory, costs.FIXED, processes=1)
    at_fixed = advantage.read([wave])
    dear = advantages(directory, costs.Prices(tiers=(50.0, 50.0, 50.0)), processes=1)
    at_dear = advantage.read([wave])
    assert (wave / FILE).read_text() == used and dear["prices"]["tiers"] == [50.0, 50.0, 50.0]
    assert dear["stats"]["return_mean"] < fixed["stats"]["return_mean"] < stats["return_mean"]
    group = {row["run_id"] for row in fight_rows(directory, WAVE) if row["group"] == user["group"]}
    changed = {run for run, index in at_fixed if at_fixed[run, index] != at_dear[run, index]}
    assert user["run_id"] in changed and changed <= group
    assert {key: value for key, value in at_fixed.items() if key[0] not in group} == {
        key: value for key, value in table.items() if key[0] not in group}
    # Left out, the uniform groups' segments are not read by an update.
    dropped = advantages(directory, costs.FIXED, drop_uniform=True, processes=1)
    assert dropped["drop_uniform_groups"] is True
    assert dropped["runs"] == 21 - dropped["stats"]["dropped"]["segments"]
    assert len(batches.priced(wave)[1]) == dropped["decisions"]


def test_a_round_without_fight_groups_has_none_to_price(tmp_path):
    policy = learner()
    seeds = {character: [f"{character}-seed"] for character in CHARACTERS}
    with Samplers(chain, policy.model, policy.vocabulary, engines=2) as pool:
        sample_groups(pool, runner(policy, greedy_fights=True), seeds, tmp_path / "round", n=1, m=[1, 1])
    with pytest.raises(FileNotFoundError, match="fight groups"):
        advantages(tmp_path / "round", costs.FIXED)
