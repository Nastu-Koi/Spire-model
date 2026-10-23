"""Fight groups: segments played from a fight's entry save to the fight's end
(`RolloutRunner.run(until_fight_end=True)`), the fights a round picks for them and the
wave they make (`main.play.fight_groups`)."""

import json
from collections import Counter
from pathlib import Path

import pytest

from main.cli import parser
from main.data.index import INDEX
from main.data.shards import RunShards
from main.env.protocol import CHARACTERS
from main.iterate import fight_cost
from main.learn.advantage import FILE as ADVANTAGES
from main.play import digests
from main.play.fight_groups import (PICKS, WAVE, candidates, group_name, jobs, pick, quotas, read_picks,
                                    sample_fight_groups)
from main.play.fights import FILE, ROW, SAVES, fight_rows
from main.play.groups import groups_summary, sample_groups
from main.play.rollout import RolloutRunner
from main.play.sampling import Samplers
from main.tests.journeys import TEMPERATURES, Journey, learner
from main.tests.test_iterate import driver, settings, world  # noqa: F401  (`world` is a fixture)


# ---- The pick ----

def test_the_kinds_share_the_fights_three_four_four():
    assert quotas(11) == dict(normal=3, elite=4, boss=4)
    assert quotas(450) == dict(normal=123, elite=164, boss=163)
    assert quotas(0) == dict(normal=0, elite=0, boss=0)
    assert quotas(1) == dict(normal=0, elite=1, boss=0)
    assert all(sum(quotas(count).values()) == count for count in range(100))


def fight(number, kind, cost, died=False, event=False, run="r"):
    """A fight's row whose cost is its `entry_hp` (`flat`)."""
    return dict(run_id=run, fight=number, kind=kind, died=died, entry_hp=cost,
                path=None if event else f"saves/{run}-{number}.json.gz")


def flat(row):
    return row["entry_hp"] + 100 * row["died"]


def test_half_of_each_kind_is_picked_by_priority_and_half_evenly_from_the_rest():
    rows = [fight(n, "normal", cost=n) for n in range(20)]
    rows += [fight(100 + n, "elite", cost=n, died=n == 3) for n in range(20)]
    rows += [fight(200 + n, "boss", cost=n) for n in range(3)]  # short of its quota
    rows += [fight(300 + n, kind, cost=1000, event=True) for n, kind in enumerate(["normal", "elite", "boss"])]
    picked, account = pick(rows, 22, flat, key="root:0")
    by = {kind: [row for row in picked if row["kind"] == kind] for kind in ("normal", "elite", "boss")}
    # 22 fights 3:4:4: six normal, eight elite and eight boss fights, three of which there are.
    assert account["kinds"] == dict(
        normal=dict(fights=20, died=0, quota=6, priority=3, uniform=3, short=0),
        elite=dict(fights=20, died=1, quota=8, priority=4, uniform=4, short=0),
        boss=dict(fights=3, died=0, quota=8, priority=3, uniform=0, short=5))
    assert (account["fights"], account["events"], account["picked"], account["short"]) == (46, 3, 17, ["boss"])
    # No fight an event began, however costly; no kind makes up for another.
    assert all(row["path"] is not None for row in picked) and len(picked) == 6 + 8 + 3
    # By priority: the deaths first, then the costliest.
    assert [row["fight"] for row in by["normal"] if row["reason"] == "priority"] == [19, 18, 17]
    assert [row["fight"] for row in by["elite"] if row["reason"] == "priority"] == [103, 119, 118, 117]
    assert [row["cost"] for row in by["elite"] if row["reason"] == "priority"] == [103.0, 19.0, 18.0, 17.0]
    # Evenly from the rest: none of those taken by priority, each once.
    for kind, chosen in by.items():
        first = {row["fight"] for row in chosen if row["reason"] == "priority"}
        rest = [row["fight"] for row in chosen if row["reason"] == "uniform"]
        assert not first & set(rest) and len(set(rest)) == len(rest)
    assert {row["fight"] for row in by["boss"]} == {200, 201, 202}
    # The draws are the key's: the same pick again, another for another round.
    assert pick(rows, 22, flat, key="root:0") == (picked, account)
    other = [row["fight"] for row in pick(rows, 22, flat, key="root:1")[0] if row["reason"] == "uniform"]
    assert other != [row["fight"] for row in picked if row["reason"] == "uniform"]
    # The share taken by priority can be another, rounded half up.
    everything, counts = pick(rows, 11, flat, key="root:0", priority=1.0)
    assert {row["reason"] for row in everything} == {"priority"}
    assert counts["kinds"]["normal"]["priority"] == 3
    assert pick(rows, 11, flat, key="root:0", priority=0.0)[1]["kinds"]["elite"]["priority"] == 0
    assert pick(rows, 11, flat, key="root:0")[1]["kinds"]["normal"] == dict(
        fights=20, died=0, quota=3, priority=2, uniform=1, short=0)
    for count, share in ((-1, 0.5), (2.0, 0.5), (3, 1.5), (3, True)):
        with pytest.raises(ValueError):
            pick(rows, count, flat, key="k", priority=share)


def test_a_fight_picked_is_played_from_its_entry_save_by_its_group():
    row = dict(fight(4, "boss", 30, run="Regent:S:act1-1"), character="Regent", seed="S", ascension=3)
    played = jobs([row], size=4, fork_depth=1)
    assert [spec["run_id"] for _, spec, _ in played] == [f"Regent:S:act1-1:fight4-{n}" for n in range(4)]
    for number, (character, spec, path) in enumerate(played):
        assert (character, path, spec["group"], spec["seed"], spec["ascension"]) == (
            "Regent", None, group_name(row), "S", 3)
        assert spec["resume"] == row["path"] and spec["until_fight_end"] is True
        assert spec["fork"] == dict(index=number, size=4, depth=1)
    assert "fork" not in jobs([row], size=2)[0][1]


# ---- Playing to the fight's end ----

class Rooms(Journey):
    """Rooms whose HP drops by 5 a room, and by 3 more once the room's fight is over;
    counts the saves it writes."""

    saves = 0

    def frame(self):
        frame = super().frame()
        hp = 70 - 5 * self.room - (3 if self.stage == "combat_play" else 0)
        frame["public"]["entities"][0]["hp"] = hp
        return frame

    def send(self, command):
        if command["cmd"] == "write_continue_save":
            self.saves += 1
        return super().send(command)


class Twin(Rooms):
    """The fights of the first two rooms follow each other: the second begins at the
    first decision after the first."""

    def send(self, command):
        reply = super().send(command)
        if self.stage == "combat_play" and self.room == 0:
            self.room += 1
            self.enter_room()
            self.events = reply["events"] + self.events
            return self.frame()
        return reply


class Ending(Rooms):
    """The last room's fight ends the run: lost, or won at its finish."""

    def __init__(self, *args, victory, **kwargs):
        super().__init__(*args, **kwargs)
        self.victory = victory

    def send(self, command):
        reply = super().send(command)
        if self.stage == "combat_play" and self.room == self.rooms - 1:
            self.stage, self.won = "terminal", self.victory
            self.events = [dict(type="run_completed", victory=self.victory, act=3, final_boss_defeated=self.victory)]
            return self.frame()
        return reply


def entry(room, kind="normal", act=1):
    """The entry save of a room's fight, as a round's fights write it."""
    return dict(at="fight", save=dict(seed="S", room=room, character="Regent", ascension=0),
                fight=dict(fight=room + 1, kind=kind, act=act, encounter="ENCOUNTER.SYNTHETIC", from_event=False,
                           second_boss=False, map_point_type="monster", floor=room + 1))


def runner(policy, **changes):
    return RolloutRunner(policy.model, policy.vocabulary, version=policy.policy_version, precision="no",
                         ascension=0, temperatures=TEMPERATURES, iteration=0, **changes)


def fight_decisions(run):
    return [macro["fight"] for macro in run["macros"]]


def test_a_segment_played_to_its_fights_end_stops_at_the_first_decision_after_it():
    policy = learner()
    played, engine = runner(policy), Rooms(4, 2, rooms=4)
    run = played.run(engine, "Regent", "S", resume=entry(1), until_fight_end=True, run_id="g-0", group="g")
    assert (run["status"], run["outcome"], run["victory"]) == ("complete", "reached", None)
    assert run["segment"] == dict(resumed=True, until_act=None, until_boss=None, until_fight_end=True)
    # The fight's decisions alone (the room's selection, one macro-action), drawn at the
    # fight's temperature; the decision after it would be the next segment's.
    assert fight_decisions(run) == [1] and run["macros"][0]["temperature"] == 0.5
    assert all(step["frame"]["public"]["phase"] == "combat" for macro in run["macros"] for step in macro["steps"])
    (record,) = run["fight_records"]
    assert {key: record[key] for key in ROW if key != "path"} == dict(
        fight=1, kind="normal", act=3, floor=2, encounter="ENCOUNTER.SYNTHETIC", second_boss=False, from_event=False,
        entry_hp=65, entry_max_hp=80, exit_hp=62, exit_max_hp=80, last_hp=65, last_max_hp=80, potions_used=[],
        opening_selection=True, died=False, decisions=3, run_id="g-0", group="g", character="Regent", seed="S",
        ascension=0)
    # It keeps its fight's record and writes no entry save: it was resumed from one.
    assert record["save"] is None and engine.saves == 0
    # A greedy segment stops at the same place (a fixed set of fights played greedily).
    greedy = played.run(Rooms(4, 2, rooms=4), "Regent", "S", sample=False, resume=entry(1), until_fight_end=True,
                        run_id="g-0", group="g")
    assert greedy["outcome"] == "reached" and fight_decisions(greedy) == [1]
    # With fight entries as well, still no save.
    again = played.run(Rooms(4, 2, rooms=4), "Regent", "S", resume=entry(1), until_fight_end=True, fight_entries=True)
    assert [record["save"] for record in again["fight_records"]] == [None]
    for wrong in (dict(), dict(resume=dict(at="act", save={})), dict(resume=entry(1), until_act=3),
                  dict(resume=entry(1), until_boss=2)):
        with pytest.raises(ValueError):
            played.run(Rooms(4, 2, rooms=4), "Regent", "S", until_fight_end=True, **wrong)


def test_a_fight_followed_at_once_by_another_stops_at_the_others_first_decision():
    policy = learner()
    run = runner(policy).run(Twin(4, 2, rooms=4), "Regent", "S", resume=entry(0), until_fight_end=True,
                             run_id="g-0", group="g")
    # The fight after it began with no decision between them: its first decision is not
    # the segment's, and it holds the HP the first fight left.
    assert (run["status"], run["outcome"]) == ("complete", "reached") and fight_decisions(run) == [1]
    (record,) = run["fight_records"]
    assert (record["fight"], record["entry_hp"], record["exit_hp"], record["decisions"]) == (1, 70, 65, 3)
    # Without the stop the segment plays on into the second fight.
    whole = runner(policy).run(Twin(4, 2, rooms=4), "Regent", "S", resume=entry(0), until_act=9, fight_entries=True)
    assert fight_decisions(whole)[:3] == [1, 2, None]


def test_a_fight_resumed_at_is_the_one_its_save_names():
    # Begun at the last act's second boss, the run has not seen the first: the save says it.
    save = entry(1, kind="boss")
    save["fight"].update(second_boss=True, map_point_type="boss")
    run = runner(learner()).run(Rooms(4, 2, rooms=4, acts=2), "Regent", "S", resume=save, until_fight_end=True)
    (record,) = run["fight_records"]
    assert (record["kind"], record["second_boss"], run["outcome"]) == ("boss", True, "reached")
    save["fight"]["second_boss"] = False
    run = runner(learner()).run(Rooms(4, 2, rooms=4, acts=2), "Regent", "S", resume=save, until_fight_end=True)
    assert run["fight_records"][0]["second_boss"] is False


@pytest.mark.parametrize("victory", [False, True])
def test_a_run_that_ends_in_the_fight_ends_the_segment_with_its_outcome(victory):
    policy = learner()
    engine = Ending(4, 2, rooms=3, victory=victory)
    run = runner(policy).run(engine, "Regent", "S", resume=entry(2), until_fight_end=True, run_id="g-0", group="g")
    assert (run["status"], run["victory"], run["outcome"]) == ("complete", victory, "win" if victory else "loss")
    (record,) = run["fight_records"]
    assert record["died"] is not victory and record["last_hp"] == 60 and record["save"] is None
    # Won, the terminal frame says nothing of the HP; lost, there is none left.
    assert (record["exit_hp"], record["exit_max_hp"]) == ((60, 80) if victory else (0, 80))


# ---- The wave ----

ROOMS, ACTS = 6, 2  # three acts of two rooms: a normal fight and the act's boss


def chain():
    return Rooms(4, 2, rooms=ROOMS, acts=ACTS)


class Idle:
    """Sampling processes that must not be asked to play."""

    def collect(self, *_, **__):
        raise AssertionError("A finished wave of fight groups is played again")


def test_a_rounds_fight_groups_play_its_fights_again_beside_its_act_waves(tmp_path):
    policy = learner()
    seeds = {character: [f"{character}-seed"] for character in CHARACTERS}
    greedy = runner(policy, greedy_fights=True)
    directory = tmp_path / "round"
    with Samplers(chain, policy.model, policy.vocabulary, engines=4, processes=2) as pool:
        sample_groups(pool, greedy, seeds, directory, n=2, m=[1, 1], fight_entries=True)
        before = {wave: (directory / wave / "summary.json").read_text()
                  for wave in groups_summary(directory)["waves"]}
        fights = greedy.variant(greedy_fights=False, explore_potions=[0.0, 0.5])
        summary = sample_fight_groups(pool, fights, directory, 11, cost=fight_cost, key="test:0", size=3,
                                      fork_depth=1)
    rows = candidates(directory, summary["waves"])
    assert len(rows) == 2 * sum(summary[wave]["written"] for wave in summary["waves"])
    wave = summary["fight_groups"]
    assert wave["wave"] == WAVE and summary["waves"] == ["act1", "act2", "act3"]
    # Rooms of a normal fight and a boss: the elites are short, and nothing makes up for them.
    assert wave["picks"]["kinds"]["elite"] == dict(fights=0, died=0, quota=4, priority=0, uniform=0, short=4)
    assert wave["picks"]["kinds"]["normal"]["quota"] == 3 and wave["picks"]["kinds"]["boss"]["quota"] == 4
    assert wave["picks"]["short"] == ["elite"] and wave["picks"]["picked"] == 7
    picked = read_picks(directory / WAVE)["picks"]
    assert [row["reason"] for row in picked] == ["priority", "priority", "uniform", "priority", "priority",
                                                 "uniform", "uniform"]
    assert wave["groups"] == 7 and wave["segments"] == wave["written"] == 21 and wave["failed"] == {}
    assert wave["plan"]["explore_potions"] == dict(normal=0.0, elite=0.5, boss=0.5)
    # Each segment plays its fight alone, every frame kept, and records it with no save.
    segments = {run["run_id"]: run for window in RunShards(directory / WAVE).windows(shards=1) for run in window}
    records = fight_rows(directory, WAVE)
    assert len(segments) == len(records) == 21 and wave["fights"]["saves"] == 0
    assert not (directory / WAVE / SAVES).exists()
    source = {group_name(row): row for row in picked}
    for record in records:
        run = segments[record["run_id"]]
        row = source[run["group"]]
        assert record["group"] == run["group"] and record["path"] is None
        assert (record["kind"], record["act"], record["floor"], record["entry_hp"]) == (
            row["kind"], row["act"], row["floor"], row["entry_hp"])
        assert run["segment"]["until_fight_end"] is True and "greedy_fights" not in run
        assert all(macro["fight"] == 1 for macro in run["macros"]) and run["outcome"] in ("reached", "win", "loss")
        assert run["fork"]["size"] == 3
    assert wave["phases"] == dict(combat=sum(len(run["macros"]) for run in segments.values()))
    # A digest line for every segment, as an act wave has.
    assert {entry["run_id"] for entry in digests.entries(directory / WAVE / digests.FILE)} == set(segments)
    # Not one of the round's waves: these stand as they were.
    assert groups_summary(directory)["fight_groups"] == wave
    assert {name: (directory / name / "summary.json").read_text() for name in summary["waves"]} == before
    # Played again on the same pick, the finished wave stands.
    assert sample_fight_groups(Idle(), fights, directory, 11, cost=fight_cost, key="test:0", size=3,
                               fork_depth=1)["fight_groups"] == wave


def test_a_cut_off_wave_of_fight_groups_goes_on_with_its_pick(tmp_path):
    policy = learner()
    seeds = {character: [f"{character}-seed"] for character in CHARACTERS}
    greedy = runner(policy, greedy_fights=True)
    directory = tmp_path / "round"
    with Samplers(chain, policy.model, policy.vocabulary, engines=2) as pool:
        sample_groups(pool, greedy, seeds, directory, n=1, m=[1, 1], fight_entries=True)
        fights = greedy.variant(greedy_fights=False)
        first = sample_fight_groups(pool, fights, directory, 4, cost=fight_cost, key="test:0", size=2)["fight_groups"]
        # Cut off after some of its segments: the pick stands, the rest are played.
        (directory / WAVE / "summary.json").unlink()
        kept = Counter()

        def counted(wave, number, run):
            kept[wave] += 1

        again = sample_fight_groups(pool, fights, directory, 4, cost=fight_cost, key="test:0", size=2,
                                    on_run=counted)["fight_groups"]
        assert len(RunShards(directory / WAVE).rows) == 4 and kept == {}
        assert {key: again[key] for key in ("picks", "segments", "written", "outcomes")} == {
            key: first[key] for key in ("picks", "segments", "written", "outcomes")}
        # Another pick sets the wave aside and plays it anew.
        other = sample_fight_groups(pool, fights, directory, 11, cost=fight_cost, key="test:0", size=2,
                                    on_run=counted)["fight_groups"]
    assert first["picks"]["picked"] == 2 and other["picks"]["picked"] == 7
    assert kept[WAVE] == other["written"] == 14
    assert any((directory / "interrupted").iterdir())
    # Only a round of greedy fights with fight entries has fights to pick.
    plain = tmp_path / "plain"
    with Samplers(chain, policy.model, policy.vocabulary, engines=2) as pool:
        sample_groups(pool, greedy, seeds, plain, n=1, m=[1, 1])
        with pytest.raises(ValueError, match="fight entries"):
            sample_fight_groups(pool, fights, plain, 4, cost=fight_cost, key="k")
    with pytest.raises(ValueError, match="greedy"):
        sample_fight_groups(Idle(), greedy, directory, 4, cost=fight_cost, key="k")
    with pytest.raises(ValueError, match="two segments"):
        sample_fight_groups(Idle(), fights, directory, 4, cost=fight_cost, key="k", size=1)


# ---- The iteration ----

def test_fight_groups_are_sampled_after_the_act_waves_priced_and_learned_beside_them(world):
    state = driver(world, eval_seeds=None, greedy_fights=True, fight_entries=True, fight_groups=11,
                   fight_group_size=2, fight_group_explore_potions=[0.0, 0.1]).run(1)
    entry = state["iterations"]["0"]
    sampled = entry["sample"]["fight_groups"]
    assert entry["settings"]["fight_groups"] == 11 and entry["settings"]["fight_group_size"] == 2
    assert sampled["picks"]["short"] == ["elite"] and sampled["picks"]["picked"] == 7
    assert sampled["segments"] == sampled["written"] == 14 and sampled["failed"] == 0
    assert sampled["decisions"] > 0 and sampled["fights"]["records"] == 14
    directory = world.output / "groups/round-0"
    assert groups_summary(directory)["fight_groups"]["picks"] == sampled["picks"]
    # The fight groups are priced beside the act waves (`main.learn.fight_groups`) and
    # learned in the same batches, a stream of their own by the same objective.
    priced, update = entry["advantages"], entry["update"]
    assert set(priced) == {"act1", "act2", "act3", WAVE} and set(update["waves"]) == {"act1", "act2", "act3"}
    assert update["decisions"] == sum(head["decisions"] for head in priced.values())
    assert set(update["shares"]) == {"act1", "act2", "act3", WAVE} and update["fights"]["samples"] > 0
    assert set(update["mix"]) == {"act-groups", "fight-groups"} and update["mix"]["fight-groups"]["kl"] is not None
    fights = priced[WAVE]
    assert fights["groups"] == 7 and fights["runs"] == 14 and fights["decisions"] == sampled["decisions"]
    assert fights["prices"]["tiers"] == [7.0, 9.0, 11.0] and fights["dropped"] is None
    assert {kind: head["groups"] for kind, head in fights["kinds"].items()} == dict(normal=3, elite=0, boss=4)
    # Read last by the update, the wave goes at the iteration's cleanup.
    for name in ("accepted", INDEX, digests.FILE, FILE, PICKS, ADVANTAGES):
        assert not (directory / WAVE / name).exists()
    assert json.loads((directory / WAVE / "summary.json").read_text())["kind"] == "fight_groups_wave"
    # It needs the round's greedy fights and their records.
    for wrong in (dict(fight_groups=4), dict(fight_groups=4, greedy_fights=True),
                  dict(fight_groups=4, fight_entries=True)):
        with pytest.raises(ValueError, match="greedy fights and fight entries"):
            settings(world, eval_seeds=None, **wrong)
    on = dict(greedy_fights=True, fight_entries=True, fight_groups=4)
    for wrong in (dict(fight_groups=-1), dict(fight_group_size=1, **on), dict(fight_group_priority=2, **on),
                  dict(fight_group_fork_depth=-1, **on), dict(fight_group_explore_potions=2, **on)):
        with pytest.raises(ValueError):
            settings(world, eval_seeds=None, **wrong)
    assert settings(world).fight_groups == 0
    common = ["iterate", "--checkpoint", "c", "--output", "o", "--iterations", "1"]
    parsed = parser().parse_args([*common, "--fight-groups", "450", "--fight-group-size", "4",
                                  "--fight-group-priority", "0.25", "--fight-group-fork-depth", "2",
                                  "--fight-group-explore-potions", "0", "0.05"])
    assert (parsed.fight_groups, parsed.fight_group_size, parsed.fight_group_priority,
            parsed.fight_group_fork_depth, parsed.fight_group_explore_potions) == (450, 4, 0.25, 2, [0.0, 0.05])
    defaults = parser().parse_args(common)
    assert (defaults.fight_groups, defaults.fight_group_size, defaults.fight_group_priority) == (0, 8, 0.5)


def test_an_iteration_cut_off_in_its_fight_groups_goes_on_from_them(world):
    from main.iterate import Driver

    calls = []

    def failing(self, iteration):
        calls.append(iteration)
        raise RuntimeError("cut off")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Driver, "advantages", failing)
        with pytest.raises(RuntimeError, match="cut off"):
            driver(world, eval_seeds=None, greedy_fights=True, fight_entries=True).run(1)
    directory = world.output / "groups/round-0"
    played = (directory / "summary.json").read_text()
    assert "fight_groups" not in json.loads(played) and not (directory / WAVE).exists()
    # The sample step was recorded without fight groups: asked for them now, it is not
    # made again. A round whose step was not recorded plays them after its act waves.
    state = json.loads((world.output / "state.json").read_text())
    del state["iterations"]["0"]["sample"]
    (world.output / "state.json").write_text(json.dumps(state))
    # Four fights of two segments are too few for a share by their decisions: half the batch.
    state = driver(world, eval_seeds=None, greedy_fights=True, fight_entries=True, fight_groups=4,
                   fight_group_size=2, source_shares={"act-groups": 1, "fight-groups": 1}).run(1)
    assert state["iterations"]["0"]["sample"]["fight_groups"]["written"] == 4
    assert calls == [0] and state["iterations"]["0"]["complete"]
