"""Act groups: segments of runs cut at the boundary acts and gone on from the saves
there, sampled wave by wave (`main.play.groups`)."""

import json
from collections import Counter

import pytest
import torch

from main.data.index import read_index
from main.data.shards import RunShards, validate_run
from main.env.control import policy_macros
from main.env.protocol import CHARACTERS, ProtocolError
from main.learn.groups import advantages
from main.learn.segments import read_waves
from main.network.policy import prepare_replay
from main.play import digests
from main.play.groups import cuts, groups_summary, per_wave, resume_jobs, sample_groups, save_rows, start_jobs, waves
from main.play.rollout import RolloutRunner, stand
from main.play.sampling import Samplers
from main.tests.journeys import TEMPERATURES, Journey, learner

ROOMS, ACTS = 6, 2  # three acts of two rooms
BOUNDARIES = (2, 3)


def engine():
    return Journey(4, 2, rooms=ROOMS, acts=ACTS)


def runner(policy, iteration=0):
    return RolloutRunner(policy.model, policy.vocabulary, version=policy.policy_version, precision="no",
                         ascension=0, temperatures=TEMPERATURES, iteration=iteration)


def frames(run):
    return [step["frame"] for macro in run["macros"] for step in macro["steps"]]


def test_a_segment_stops_at_the_act_and_the_next_goes_on_from_its_save():
    policy = learner()
    played = runner(policy)
    first = played.run(engine(), "Ironclad", "seed", until_act=2, run_id="Ironclad:seed:act1-0",
                       group="Ironclad:seed:act1")
    assert first["status"] == "complete" and first["outcome"] == "reached" and first["victory"] is None
    assert first["reached_act"] == 2 and first["segment"] == {"resumed": False, "until_act": 2, "until_boss": None}
    assert first["run_id"] == "Ironclad:seed:act1-0" and first["group"] == "Ironclad:seed:act1"
    # The first decision of the second act is not part of the segment; the save stands there.
    assert frames(first) and all(stand(frame)[0] == 1 for frame in frames(first))
    assert first["save"]["at"] == "act" and first["save"]["save"]["room"] == ACTS
    validate_run(first, on_policy=True)

    second = played.run(engine(), "Ironclad", "seed", resume=first["save"], until_act=3,
                        run_id="Ironclad:seed:act1-0:act2-0", group="Ironclad:seed:act1-0:act2")
    assert second["segment"] == {"resumed": True, "until_act": 3, "until_boss": None} and second["outcome"] == "reached"
    assert second["reached_act"] == 3 and second["save"]["save"]["room"] == 2 * ACTS
    assert stand(frames(second)[0]) == (2, 1, None) and all(stand(frame)[0] == 2 for frame in frames(second))
    validate_run(second, on_policy=True)

    last = played.run(engine(), "Ironclad", "seed", resume=second["save"], run_id="Ironclad:seed:act1-0:act2-0:act3-0",
                      group="Ironclad:seed:act1-0:act2-0:act3")
    assert last["segment"] == {"resumed": True, "until_act": None, "until_boss": None} and last["outcome"] in ("win", "loss")
    assert last["victory"] is (last["outcome"] == "win")
    assert stand(frames(last)[0]) == (3, 1, None) and all(stand(frame)[0] == 3 for frame in frames(last))
    assert "save" not in last
    validate_run(last, on_policy=True)


def test_a_segment_is_played_at_its_own_ascension_and_goes_on_at_its_saves():
    policy = learner()
    played = runner(policy)  # the runner's own ascension is 0
    first = played.run(engine(), "Ironclad", "seed", until_act=2, run_id="Ironclad:seed:act1-0",
                       group="Ironclad:seed:act1", ascension=5)
    assert first["ascension"] == 5 and all(frame["contract"]["fixed_ascension"] == 5 for frame in frames(first))
    validate_run(first, on_policy=True)
    second = played.run(engine(), "Ironclad", "seed", resume=first["save"], until_act=3, ascension=5,
                        run_id="Ironclad:seed:act1-0:act2-0", group="Ironclad:seed:act1-0:act2")
    assert second["ascension"] == 5 and all(frame["contract"]["fixed_ascension"] == 5 for frame in frames(second))
    validate_run(second, on_policy=True)
    # A save goes on at the ascension it was written at, not at another one asked for.
    other = played.run(engine(), "Ironclad", "seed", resume=first["save"], until_act=3, ascension=3,
                       run_id="Ironclad:seed:act1-0:act2-1", group="Ironclad:seed:act1-0:act2")
    assert other["status"] == "error" and "ascension" in other["error"]
    with pytest.raises(ValueError, match="Ascension"):
        played.run(engine(), "Ironclad", "seed", until_act=2, ascension=11, run_id="x", group="x")
    # Without one the run is at the runner's.
    plain = played.run(engine(), "Ironclad", "seed", until_act=2, run_id="Ironclad:seed:act1-1", group="Ironclad:seed:act1")
    assert plain["ascension"] == 0


def test_a_segment_stops_at_the_boss_fight_and_the_next_enters_it_again():
    policy = learner()
    played = runner(policy)
    # The third act's boss is the fight of its last room; the segment stops at the
    # fight's first decision, which belongs to the next segment.
    first = played.run(engine(), "Defect", "seed", resume=None, until_boss=3, run_id="a", group="g")
    assert first["outcome"] == "reached" and first["reached_boss"] == 3 and "reached_act" not in first
    assert first["segment"] == {"resumed": False, "until_act": None, "until_boss": 3}
    save = first["save"]
    assert save["at"] == "fight" and save["save"]["room"] == ROOMS - 1
    assert save["fight"]["kind"] == "boss" and save["fight"]["act"] == 3 and save["fight"]["second_boss"] is False
    assert save["fight"]["encounter"] == "ENCOUNTER.SYNTHETIC" and save["fight"]["floor"] == ACTS
    # Every frame of the segment is before the boss fight; the earlier bosses were fought.
    assert frames(first) and len(first["macros"]) == 2 * (ROOMS - 1)
    assert first["fights"]["wins"] == ROOMS - 1
    validate_run(first, on_policy=True)

    last = played.run(engine(), "Defect", "seed", resume=save, run_id="a:boss3-0", group="a:boss3")
    assert last["segment"] == {"resumed": True, "until_act": None, "until_boss": None}
    assert last["outcome"] in ("win", "loss") and last["victory"] is (last["outcome"] == "win")
    # The segment is the boss fight and the decision after it: the fight's selection first.
    assert frames(last)[0]["public"]["phase"] == "combat" and stand(frames(last)[0])[0] == 3
    assert last["fights"]["wins"] == 1
    validate_run(last, on_policy=True)

    # A boss cut before the second act's start: the boss wave then plays on to the act.
    boss2 = played.run(engine(), "Defect", "seed", until_boss=2, run_id="b", group="g")
    on = played.run(engine(), "Defect", "seed", resume=boss2["save"], until_act=3, run_id="b:boss2-0", group="b:boss2")
    assert on["outcome"] == "reached" and on["reached_act"] == 3 and on["save"]["at"] == "act"
    with pytest.raises(ValueError):
        played.run(engine(), "Defect", "seed", until_act=2, until_boss=2, run_id="c", group="g")


def test_a_run_without_an_outcome_is_only_a_segment_that_reached_its_act():
    policy = learner()
    first = runner(policy).run(engine(), "Silent", "seed", until_act=2, run_id="x", group="g")
    bogus = dict(first)
    del bogus["segment"]
    with pytest.raises(ProtocolError):
        validate_run(bogus, on_policy=True)


def test_the_waves_and_their_jobs_are_named_by_where_they_start():
    assert waves((2, 3)) == ["act1", "act2", "act3"] and waves((3,)) == ["act1", "act3"]
    # A boss cut comes before the next act's start; the wave from it is the boss's.
    assert cuts((2, 3), (3,)) == [("act", 2), ("act", 3), ("boss", 3)]
    assert cuts((2, 3), (1, 2, 3)) == [("boss", 1), ("act", 2), ("boss", 2), ("act", 3), ("boss", 3)]
    assert waves((2, 3), (3,)) == ["act1", "act2", "act3", "boss3"]
    assert waves((), (3,)) == ["act1", "boss3"] and waves((2,), (1,)) == ["act1", "boss1", "act2"]
    for wrong in ((), (1,), (3, 2), (2, 2)):
        with pytest.raises(ValueError):
            waves(wrong)
    for wrong in ((0,), (3, 1), (2, 2)):
        with pytest.raises(ValueError):
            waves((2, 3), wrong)
    assert per_wave(8, 2) == [8, 8] and per_wave([4, 8], 2) == [4, 8] and per_wave([5], 1) == [5]
    for wrong in (([4, 8, 2], 2), (0, 1), ([], 1)):
        with pytest.raises(ValueError):
            per_wave(*wrong)
    jobs = start_jobs({c: [f"{c}-1"] for c in CHARACTERS}, 2, "act1", ("act", 2))
    assert len(jobs) == 10
    character, spec, _ = jobs[1]
    assert spec == dict(seed=f"{character}-1", run_id=f"{character}:{character}-1:act1-1",
                        group=f"{character}:{character}-1:act1", until_act=2)
    rows = [dict(run_id="Ironclad:s:act1-0", group="Ironclad:s:act1", character="Ironclad", seed="s", act=2,
                 path="saves/x.json.gz")]
    jobs = resume_jobs(rows, 3, "act2", ("act", 3))
    assert [spec["run_id"] for _, spec, _ in jobs] == [f"Ironclad:s:act1-0:act2-{k}" for k in range(3)]
    assert all(spec["resume"] == "saves/x.json.gz" and spec["group"] == "Ironclad:s:act1-0:act2"
               and spec["until_act"] == 3 for _, spec, _ in jobs)
    assert all(spec["until_act"] is None for _, spec, _ in resume_jobs(rows, 1, "act3", None))
    assert all(spec["until_boss"] == 3 and "until_act" not in spec for _, spec, _ in resume_jobs(rows, 1, "act3", ("boss", 3)))
    # Forking jobs say each segment's place in its group; without a depth they say nothing.
    assert all("fork" not in spec for _, spec, _ in jobs)
    forking = start_jobs({c: [f"{c}-1"] for c in CHARACTERS}, 2, "act1", ("act", 2), fork_depth=2)
    assert [spec["fork"] for _, spec, _ in forking[:2]] == [dict(index=0, size=2, depth=2), dict(index=1, size=2, depth=2)]
    assert [spec["fork"] for _, spec, _ in resume_jobs(rows, 3, "act2", ("act", 3), fork_depth=1)] == [
        dict(index=k, size=3, depth=1) for k in range(3)]


def test_act_groups_sample_the_waves_in_turn_and_keep_the_chains_they_won(tmp_path):
    policy = learner()
    seeds = {character: [f"{character}-seed"] for character in CHARACTERS}
    directory = tmp_path / "groups"
    with Samplers(engine, policy.model, policy.vocabulary, engines=4, processes=2) as pool:
        summary = sample_groups(pool, runner(policy, iteration=1), seeds, directory, n=2, m=[2, 1],
                                boundaries=BOUNDARIES)
    assert summary == groups_summary(directory)
    assert summary["waves"] == ["act1", "act2", "act3"] and summary["boundaries"] == [2, 3] and summary["bosses"] == []
    assert summary["cuts"] == [["act", 2], ["act", 3]]
    assert (summary["groups"], summary["n"], summary["m"], summary["iteration"]) == (5, 2, [2, 1], 1)
    assert summary["saves"] == {"act1": 10, "act2": 20}
    first, second, last = (summary[wave] for wave in summary["waves"])
    assert (first["wave"], first["until_act"], first["segments"], first["written"]) == ("act1", 2, 10, 10)
    assert first["outcomes"] == {"reached": 10} and first["groups"] == 5
    assert (second["until_act"], second["segments"], second["groups"]) == (3, 20, 10)
    assert second["outcomes"] == {"reached": 20}
    assert (last["until_act"], last["segments"], last["groups"]) == (None, 20, 20)
    assert set(last["outcomes"]) <= {"win", "loss"} and sum(last["outcomes"].values()) == 20
    # Every segment keeps its fights' frames.
    assert all(summary[wave]["phases"].get("combat") for wave in summary["waves"])

    rows = save_rows(directory, "act1")
    assert len(rows) == 10 and all(row["act"] == 2 and row["at"] == "act" and row["boss"] is None for row in rows)
    assert all(row["act"] == 3 for row in save_rows(directory, "act2")) and save_rows(directory, "act3") == []
    first_rows = list(read_index(directory / "act1"))
    assert sorted(row["run_id"] for row in first_rows) == sorted(row["run_id"] for row in rows)
    assert all(row["outcome"] == "reached" and row["victory"] is None and row["group"].endswith(":act1")
               for row in first_rows)
    second_rows = list(read_index(directory / "act2"))
    assert {row["group"] for row in second_rows} == {row["run_id"] + ":act2" for row in rows}
    last_rows = list(read_index(directory / "act3"))
    assert {row["group"] for row in last_rows} == {row["run_id"] + ":act3" for row in save_rows(directory, "act2")}
    assert all(type(row["victory"]) is bool and row["outcome"] in ("win", "loss") for row in last_rows)

    # The shards read back as runs; a later wave's segment begins where its parent stopped.
    for wave, act in (("act2", 2), ("act3", 3)):
        for window_ in RunShards(directory / wave).windows(shards=1):
            for run in window_:
                assert run["segment"]["resumed"] is True and stand(frames(run)[0]) == (act, 1, None)
                assert run["group"].endswith(f":{wave}") and "save" not in run
                assert all(stand(frame)[0] == act for frame in frames(run))
    for window_ in RunShards(directory / "act1").windows(shards=1):
        for run in window_:
            assert run["outcome"] == "reached" and all(stand(frame)[0] == 1 for frame in frames(run))



def test_a_round_plays_each_group_at_its_seeds_ascension_and_its_later_waves_at_their_saves(tmp_path):
    policy = learner()
    seeds = {character: [f"{character}-a", f"{character}-b"] for character in CHARACTERS}
    ascensions = {character: {f"{character}-a": 0, f"{character}-b": 4} for character in CHARACTERS}
    directory = tmp_path / "groups"
    with Samplers(engine, policy.model, policy.vocabulary, engines=4, processes=2) as pool:
        summary = sample_groups(pool, runner(policy), seeds, directory, n=2, m=[1, 1], boundaries=BOUNDARIES,
                                ascensions=ascensions)
    assert summary["ascensions"] == "metadata/ascensions.json"
    assert json.loads((directory / "metadata/ascensions.json").read_text()) == ascensions
    for wave in summary["waves"]:
        rows = list(read_index(directory / wave))
        # Every segment is at its root seed's ascension, a later wave's through its save.
        assert all(row["ascension"] == ascensions[row["character"]][row["seed"]] for row in rows)
        counted = summary[wave]["by_ascension"]
        assert {level: entry["written"] for level, entry in counted.items()} == Counter(str(r["ascension"]) for r in rows)
        assert sum(sum(entry["outcomes"].values()) for entry in counted.values()) == summary[wave]["written"]
    assert all(row["ascension"] == ascensions[row["character"]][row["seed"]]
               for wave in ("act1", "act2") for row in save_rows(directory, wave))
    # The advantages count each ascension's groups apart.
    for wave, head in advantages(directory, processes=1).items():
        levels = head["stats"]["by_ascension"]
        assert set(levels) == {"0", "4"} and sum(entry["groups"] for entry in levels.values()) == head["stats"]["groups"]
        assert all(entry["segments"] > 0 and entry["return_mean"] is not None for entry in levels.values())


def test_a_round_cut_off_goes_on_from_the_wave_it_was_in(tmp_path):
    import gzip
    import json

    from main.data.index import INDEX

    policy = learner()
    seeds = {character: [f"{character}-seed"] for character in CHARACTERS}
    directory = tmp_path / "groups"
    with Samplers(engine, policy.model, policy.vocabulary, engines=4, processes=2) as pool:
        whole = sample_groups(pool, runner(policy), seeds, directory, n=2, m=[2, 2], boundaries=BOUNDARIES)
        rows_before = {wave: list(read_index(directory / wave)) for wave in whole["waves"]}
        stamps = {wave: (directory / wave / "summary.json").stat().st_mtime_ns for wave in ("act1", "act2")}

        # Cut off in the last wave after five segments: the shards and the index of
        # those stand, no summary says the wave or the round is finished.
        (directory / "summary.json").unlink()
        (directory / "act3" / "summary.json").unlink()
        with gzip.open(directory / "act3" / INDEX, "rt", encoding="utf-8") as stream:
            lines = stream.read().splitlines()
        with gzip.open(directory / "act3" / INDEX, "wt", encoding="utf-8") as stream:
            stream.write("\n".join(lines[:1 + 5]) + "\n")
        with pytest.raises(FileExistsError):
            sample_groups(pool, runner(policy), seeds, directory, n=2, m=[2, 2], boundaries=BOUNDARIES)
        again = sample_groups(pool, runner(policy), seeds, directory, n=2, m=[2, 2], boundaries=BOUNDARIES,
                              resume=True)
    assert again["waves"] == whole["waves"] and again["saves"] == whole["saves"]
    assert all((directory / wave / "summary.json").stat().st_mtime_ns == stamps[wave] for wave in stamps)
    assert again["act3"]["resumed"] == 5 and again["act3"]["written"] == 40 and again["act3"]["segments"] == 40
    rows = list(read_index(directory / "act3"))
    assert [row["run_id"] for row in rows[:5]] == [row["run_id"] for row in rows_before["act3"][:5]]
    assert sorted(row["run_id"] for row in rows) == sorted(row["run_id"] for row in rows_before["act3"])
    assert again["act3"]["shards"] >= 2 and set(again["act3"]["contracts"]) == set(whole["act3"]["contracts"])
    shards = RunShards(directory / "act3")
    assert len(shards) == 40 and sum(len(window_) for window_ in shards.windows(shards=1)) == 40
    for window_ in shards.windows(shards=1):
        for run in window_:
            validate_run(run)
    assert groups_summary(directory)["act3"]["written"] == 40
    assert again["act3"]["distributions"] and again["distributions"]
    # The digest goes on too: a segment played again is its later line.
    digested = read_waves(directory, ["act3"], processes=1)
    (directory / "act3" / digests.FILE).rename(tmp_path / "act3.digest")
    assert read_waves(directory, ["act3"], processes=1) == digested
    (tmp_path / "act3.digest").rename(directory / "act3" / digests.FILE)

    # A wave left without an index is set aside and played again.
    (directory / "summary.json").unlink()
    (directory / "act3" / "summary.json").unlink()
    (directory / "act3" / INDEX).unlink()
    with Samplers(engine, policy.model, policy.vocabulary, engines=4, processes=2) as pool:
        again = sample_groups(pool, runner(policy), seeds, directory, n=2, m=[2, 2], boundaries=BOUNDARIES,
                              resume=True)
    assert again["act3"]["resumed"] == 0 and again["act3"]["written"] == 40
    assert (directory / "interrupted" / "act3.0").is_dir() and len(list(read_index(directory / "act3"))) == 40


def test_a_segment_keeps_the_distribution_each_of_its_decisions_was_drawn_from(tmp_path):
    policy = learner()
    seeds = {character: [f"{character}-seed"] for character in CHARACTERS}
    with Samplers(engine, policy.model, policy.vocabulary, engines=4, processes=2) as pool:
        summary = sample_groups(pool, runner(policy), seeds, tmp_path / "groups", n=2, m=[2, 2],
                                boundaries=BOUNDARIES)
    assert summary["distributions"] and all(summary[wave]["distributions"] for wave in summary["waves"])
    shards = RunShards(tmp_path / "groups" / "act2")
    macros = [macro for window in shards.windows(shards=8) for run in window for macro in policy_macros(run)]
    for macro in macros:
        for step in macro["steps"]:
            candidates = step["frame"]["legal"]["candidates"]
            if step["forced"]:
                assert "log_probs" not in step
                continue
            # Every candidate's log-probability at the temperature it was drawn at: the
            # chosen one's is what the macro-action's own sums.
            assert len(step["log_probs"]) == len(candidates)
            chosen = [c["candidate_ref"] for c in candidates].index(step["candidate_ref"])
            if len(macro["steps"]) == 1:
                assert step["log_probs"][chosen] == pytest.approx(macro["old_log_prob"], abs=1e-5)
    # Replayed, they are the distributions the policy gives the same decisions again.
    prepared = prepare_replay(policy.vocabulary, [macro["steps"] for macro in macros])
    with torch.no_grad():
        again = policy.model.decision_log_probs(prepared, [macro["temperature"] for macro in macros])
    assert torch.equal(torch.isinf(prepared.reference), ~prepared.masks)
    assert (prepared.reference - again)[prepared.masks].abs().max() < 1e-4


def test_a_wave_cut_off_leaves_no_summary_and_is_gone_on_with(tmp_path):
    policy = learner()
    seeds = {character: [f"{character}-seed"] for character in CHARACTERS}
    directory = tmp_path / "groups"
    seen = []

    def cut(wave, index, run):
        seen.append(wave)
        if wave == "act3" and seen.count("act3") == 3:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        with Samplers(engine, policy.model, policy.vocabulary, engines=4, processes=2) as pool:
            sample_groups(pool, runner(policy), seeds, directory, n=2, m=[2, 2], boundaries=BOUNDARIES, on_run=cut)
    assert groups_summary(directory) is None and not (directory / "act3" / "summary.json").exists()
    assert (directory / "act1" / "summary.json").exists() and (directory / "act2" / "summary.json").exists()
    assert len(list(read_index(directory / "act3"))) >= 3
    with Samplers(engine, policy.model, policy.vocabulary, engines=4, processes=2) as pool:
        again = sample_groups(pool, runner(policy), seeds, directory, n=2, m=[2, 2], boundaries=BOUNDARIES,
                              resume=True)
    assert again["act3"]["resumed"] >= 3 and again["act3"]["written"] == 40 and groups_summary(directory) == again


def test_a_round_cut_at_a_boss_fight_plays_the_boss_wave_from_the_fights_saves(tmp_path):
    policy = learner()
    seeds = {character: [f"{character}-seed"] for character in CHARACTERS}
    directory = tmp_path / "groups"
    with Samplers(engine, policy.model, policy.vocabulary, engines=4, processes=2) as pool:
        summary = sample_groups(pool, runner(policy, iteration=1), seeds, directory, n=2, m=[1, 1, 2],
                                boundaries=BOUNDARIES, bosses=(3,))
    assert summary["waves"] == ["act1", "act2", "act3", "boss3"] and summary["bosses"] == [3]
    assert summary["cuts"] == [["act", 2], ["act", 3], ["boss", 3]] and summary["m"] == [1, 1, 2]
    third, boss = summary["act3"], summary["boss3"]
    # The third act's segments stop at its boss fight; the boss wave plays the fight to the end.
    assert (third["until_act"], third["until_boss"], third["until"]) == (None, 3, ["boss", 3])
    assert third["outcomes"] == {"reached": 10} and summary["saves"]["act3"] == 10
    assert (boss["until_act"], boss["until"], boss["segments"], boss["groups"]) == (None, None, 20, 10)
    assert set(boss["outcomes"]) <= {"win", "loss"} and sum(boss["outcomes"].values()) == 20
    rows = save_rows(directory, "act3")
    assert all(row["at"] == "fight" and row["boss"] == 3 and row["act"] is None for row in rows)
    for window_ in RunShards(directory / "boss3").windows(shards=1):
        for run in window_:
            assert run["segment"] == {"resumed": True, "until_act": None, "until_boss": None}
            # The boss wave's frames are the fight (in the last room of the act) and what follows.
            assert frames(run)[0]["public"]["phase"] == "combat" and stand(frames(run)[0]) == (3, ACTS, None)
            assert run["fights"]["wins"] + run["fights"]["losses"] == 1
            validate_run(run, on_policy=True)


def greedy(policy, iteration=0):
    return RolloutRunner(policy.model, policy.vocabulary, version=policy.policy_version, precision="no",
                         ascension=0, temperatures=TEMPERATURES, iteration=iteration, greedy_fights=True)


def test_a_round_with_greedy_fights_keeps_no_frames_of_its_fights(tmp_path):
    policy = learner()
    seeds = {character: [f"{character}-seed"] for character in CHARACTERS}
    directory = tmp_path / "groups"
    with Samplers(engine, policy.model, policy.vocabulary, engines=4, processes=2) as pool:
        summary = sample_groups(pool, greedy(policy), seeds, directory, n=2, m=2, boundaries=BOUNDARIES,
                                fork_depth=1)
        with pytest.raises(ValueError, match="boss"):
            sample_groups(pool, greedy(policy), seeds, tmp_path / "bosses", n=2, m=2, boundaries=BOUNDARIES,
                          bosses=(3,))
    assert summary["greedy_fights"] is True and summary["waves"] == ["act1", "act2", "act3"]
    for wave in summary["waves"]:
        # The round's decisions are those outside the fights: every segment forked at its
        # first one, the fights' selections counted nowhere.
        assert summary[wave]["phases"] and "combat" not in summary[wave]["phases"]
        assert summary[wave]["forked"] == summary[wave]["written"]
        runs = [run for window_ in RunShards(directory / wave).windows(shards=8) for run in window_]
        frameless = 0
        for run in runs:
            assert run["greedy_fights"] is True
            fights = [macro for macro in run["macros"] if macro["fight"] is not None]
            # A fight's macro-actions keep their counts and how likely their play was,
            # with no temperature and no frames; the rest are the run's samples.
            assert fights and all("steps" not in macro and macro["temperature"] is None
                                  and 1 <= macro["branches"] <= macro["decisions"] for macro in fights)
            assert all(macro["fight"] is None and macro["temperature"] == TEMPERATURES["other"]
                       for macro in policy_macros(run))
            assert all(frame["public"]["phase"] != "combat" for frame in frames(dict(macros=list(policy_macros(run)))))
            frameless += len(fights)
        # The digest has a line for every decision kept, none of them in a fight, and
        # counts the fights' macro-actions it has none for.
        lines = list(digests.entries(directory / wave / digests.FILE))
        assert not any(decision[digests.FIELDS.index("fight")] for line in lines for decision in line["decisions"])
        assert sum(line["frameless"] for line in lines) == frameless > 0
    # Without greedy fights the digest has none to count.
    plain = tmp_path / "plain"
    with Samplers(engine, policy.model, policy.vocabulary, engines=4, processes=2) as pool:
        sample_groups(pool, runner(policy), seeds, plain, n=2, m=2, boundaries=BOUNDARIES)
    assert all(line["frameless"] == 0 for line in digests.entries(plain / "act1" / digests.FILE))
