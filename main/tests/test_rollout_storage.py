"""A sampled round is written as training reads it: the frames of decisions outside
fights, the entry save of every fight, and shards with an index."""

import gzip
import json

import pytest
import torch

from main.play.buffer import sample_round
from main.cli import main
from main.env.control import policy_macros
from main.data.shards import RunShards, samples, validate_run
from main.data.index import INDEX, read_index, rebuild_index
from main.env.protocol import CHARACTERS, ProtocolError
from main.encoding.representation import observation
from main.play.rollout import RolloutRunner
from main.play.sampling import Samplers
from main.env.saves import FightTracker, anchor_commands, read_saves
from main.play.testing import SyntheticEngine
from main.tests.test_sampling import Failing, Fighting, setup_round

TEMPERATURES = {"combat": 0.5, "other": 0.8}


def fighting():
    return Fighting(4, 2)


def sampled_runner(iteration=3):
    learner, _, seeds = setup_round(TEMPERATURES)
    runner = RolloutRunner(learner.model, learner.vocabulary, version=learner.policy_version, precision="no",
                           ascension=0, temperatures=TEMPERATURES, iteration=iteration)
    return learner, runner, seeds


def started(index, kind="normal", act=1, from_event=False):
    return dict(type="encounter_started", encounter_id=f"e{index}", encounter="ENCOUNTER.X", act=act, kind=kind,
                from_event=from_event)


def completed(index):
    return dict(type="encounter_completed", encounter_id=f"e{index}", result="victory")


def test_fights_are_told_apart_by_the_engine_events():
    tracker = FightTracker()
    assert tracker.decision() == (None, None)
    tracker.events([started(1)])
    fight, entry = tracker.decision()
    assert fight == 1 and entry == dict(fight=1, encounter_id="e1", encounter="ENCOUNTER.X", kind="normal", act=1,
                                        from_event=False, map_point_type="monster", second_boss=False)
    # Later decisions of the fight belong to it without being its entry.
    assert tracker.decision() == (1, None)
    # The first decision after the win is outside: the fight ended with the win.
    tracker.events([completed(1)])
    assert tracker.decision() == (None, None)
    tracker.events([started(2, "boss"), ])
    assert tracker.decision()[1]["second_boss"] is False
    tracker.events([completed(2), started(3, "boss")])
    fight, entry = tracker.decision()
    assert (fight, entry["second_boss"], entry["map_point_type"]) == (3, True, "boss")
    tracker.events([completed(3), started(4, "elite", act=2, from_event=True)])
    assert tracker.decision()[1] == dict(fight=4, encounter_id="e4", encounter="ENCOUNTER.X", kind="elite", act=2,
                                         from_event=True, map_point_type="elite", second_boss=False)
    with pytest.raises(ProtocolError, match="identity"):
        FightTracker().events([dict(started(5), kind="event")])
    # How each fight ended for the policy: the run died in the fourth; cut off there
    # instead, the fight has no result.
    assert (tracker.results, tracker.tally()) == ({1: "win", 2: "win", 3: "win", 4: None},
                                                  dict(wins=3, losses=0, unfinished=1))
    tracker.finish(False)
    assert (tracker.results[4], tracker.tally()) == ("loss", dict(wins=3, losses=1, unfinished=0))


def test_a_sampled_run_keeps_frames_outside_fights_and_counts_inside():
    _, runner, _ = sampled_runner(iteration=9)
    run = runner.run(fighting(), "Ironclad", "stored", combat_frames=False)
    assert (run["status"], run["source"], run["iteration"], run["temperature"]) == ("complete", "rollout", 9, TEMPERATURES)
    inside, outside = run["macros"]
    # The fight's selection: how many decisions, how many with a choice, how likely; no frames.
    assert (inside["phase"], inside["fight"], inside["branches"], inside["decisions"]) == ("combat", 1, 2, 3)
    assert "steps" not in inside and isinstance(inside["old_log_prob"], float) and inside["old_log_prob"] < 0
    # The decision after the win, with its frame and full candidates.
    assert (outside["phase"], outside["fight"], len(outside["steps"])) == ("combat_play", None, 1)
    assert len(outside["steps"][0]["frame"]["legal"]["candidates"]) == 2
    assert [macro["phase"] for macro in policy_macros(run)] == ["combat_play"]
    assert validate_run(run, on_policy=True) is run
    (sample,) = samples([run])
    assert (sample.phase, sample.source, sample.weight, sample.branches) == ("combat_play", "rollout", 1.0, 1)
    assert run["fights"] == dict(wins=1, losses=0, unfinished=0)
    assert "climb" in run and run["environment_actions"] == []
    # Greedy twice, with and without the fight's frames: the same play, counted or kept.
    # With every frame the fight's macro-action is a sample too.
    counted = runner.run(fighting(), "Ironclad", "stored", sample=False, combat_frames=False)
    full = runner.run(fighting(), "Ironclad", "stored", sample=False)
    assert [(m["phase"], m["fight"], len(m["steps"])) for m in full["macros"]] == [("combat", 1, 3), ("combat_play", None, 1)]
    assert [m["phase"] for m in policy_macros(full)] == ["combat", "combat_play"]
    assert [m["old_log_prob"] for m in full["macros"]] == [m["old_log_prob"] for m in counted["macros"]]
    assert (counted["macros"][0]["branches"], counted["macros"][0]["decisions"]) == (2, 3)
    # Only a sampled run's fights go without frames, and only with their counts.
    with pytest.raises(ProtocolError, match="Only the fights"):
        validate_run(dict(run, source="evaluation", sampling="argmax", temperature=None))
    with pytest.raises(ProtocolError, match="counts"):
        validate_run(dict(run, macros=[dict(inside, branches=0), outside]))
    with pytest.raises(ProtocolError, match="Only the fights"):
        validate_run(dict(run, macros=[dict(inside, fight=None), outside]))


class Refusing(SyntheticEngine):
    """An engine that cannot write its save."""

    def send(self, command):
        if command["cmd"] == "write_continue_save":
            return {"type": "error", "message": "no room to save"}
        return super().send(command)


def test_entry_saves_name_each_fight_and_its_result(tmp_path):
    _, runner, _ = sampled_runner()
    saves = tmp_path / "run.saves.jsonl.gz"
    run = runner.run(SyntheticEngine(4, 2), "Ironclad", "saved", saves=saves, combat_frames=False)
    assert run["status"] == "complete"
    assert run["saves"] == dict(path="run.saves.jsonl.gz", schema="rollout-entry-saves-v1", fights=1, failed=0, from_event=0)
    (row,) = read_saves(saves)
    assert (row["run_id"], row["seed"], row["character"], row["ascension"]) == (run["run_id"], "saved", "Ironclad", 0)
    assert (row["fight"], row["encounter_id"], row["encounter"], row["kind"]) == (1, "ordinary", "ENCOUNTER.SYNTHETIC", "normal")
    assert (row["map_point_type"], row["second_boss"], row["act"], row["from_event"]) == ("monster", False, 1, False)
    # The save is the engine's own, taken at the fight's first decision (state version 0).
    assert row["save"]["synthetic"] is True and row["save"]["state_version"] == 0
    # The row says how the fight ended for the policy, and which relics it entered with.
    assert (row["result"], row["relics"]) == ("win", [])
    enter, room = anchor_commands(row)
    assert (enter["cmd"], enter["save"], enter["map_point_type"], enter["second_boss"]) == (
        "enter_anchor", row["save"], "monster", False)
    assert room == dict(cmd="anchor_room", room=dict(type="combat", encounter="SYNTHETIC"))
    # A refused save leaves the fight its row and its result, and says why.
    refused = runner.run(Refusing(4, 2), "Ironclad", "refused", saves=tmp_path / "r.saves.jsonl.gz")
    assert refused["status"] == "complete" and refused["saves"]["failed"] == 1
    (row,) = read_saves(tmp_path / "r.saves.jsonl.gz")
    assert row["save"] is None and "no room" in row["error"] and row["result"] == "win"
    with pytest.raises(ProtocolError, match="no entry save"):
        anchor_commands(row)


def replayed(model, vocabulary, run):
    """The stored log-probability of every macro-action that kept its frames, scored
    again from them at the temperature of its phase."""
    result = []
    for macro in run["macros"]:
        if "steps" not in macro:
            continue
        total = 0.0
        for step in macro["steps"]:
            if step["forced"]:
                continue
            frame = step["frame"]
            obs = observation(frame)
            slots = {c["candidate_ref"]: c["decoder_slot_ref"] for c in frame["legal"]["candidates"]}
            mask = torch.tensor([slot in set(slots.values()) for slot in obs.slot_refs])
            with torch.no_grad():
                output = model.decode_batch(model.encode([obs], vocabulary), [mask])[0]
            temperature = TEMPERATURES["combat" if frame["public"]["phase"] == "combat" else "other"]
            total += float((output.logits / temperature).log_softmax(-1)[obs.slot_refs.index(slots[step["candidate_ref"]])])
        result.append((macro["old_log_prob"], total))
    return result


def test_a_sampled_round_is_shards_the_trainer_reads(tmp_path):
    learner, runner, seeds = sampled_runner(iteration=3)
    directory = tmp_path / "round-0"
    with Samplers(fighting, runner.model, runner.vocabulary, engines=5, processes=3) as pool:
        summary = sample_round(pool, runner, seeds, directory)
    # The round's files: shards and index, the entry saves of every run, no run files.
    assert sorted(p.name for p in directory.iterdir() if p.is_dir()) == ["accepted", "metadata"]
    assert [p.name for p in (directory / "accepted").iterdir()] == ["00000.jsonl.gz"]
    assert (directory / INDEX).exists() and json.loads((directory / "metadata/seeds.json").read_text()) == seeds
    assert [p.name for p in directory.glob("*.json")] == ["summary.json"] and not list(directory.glob("*.jsonl"))
    assert len(list(directory.glob("*.saves.jsonl.gz"))) == 20 and not list(directory.glob("*.combats.jsonl.gz"))
    assert json.loads((directory / "summary.json").read_text()) == summary
    assert (summary["kind"], summary["iteration"], summary["policy_version"], summary["policy_digest"]) == (
        "rollout_round", 3, runner.version, runner.digest)
    assert (summary["runs"], summary["accepted_runs"], summary["quarantined_runs"], summary["shards"]) == (20, 20, 0, 1)
    assert (summary["fights"], summary["saves_failed"], summary["macros"], summary["fight_macros"]) == (20, 0, 20, 20)
    assert summary["counts"] == {c + "/combat_play": 4 for c in CHARACTERS} and summary["temperature"] == TEMPERATURES
    assert summary["report"]["comparable"] is True and summary["report"]["equal_character_win_rate"] == summary["wins"] / 20
    assert len(summary["contracts"]) == 1
    # The index names the policy, the iteration and the outcome of every run.
    rows = read_index(directory)
    assert len(rows) == 20 and all(
        (row["policy_version"], row["policy_digest"], row["iteration"], type(row["victory"])) == (runner.version, runner.digest, 3, bool)
        for row in rows)
    assert all(row["phases"] == {"combat_play": 1} for row in rows)
    # RunShards reads it with the weights the demonstrations get.
    shards = RunShards(directory)
    assert RunShards.is_at(directory) and len(shards) == 20
    assert shards.counts == {(c, "combat_play"): 4 for c in CHARACTERS}
    played = Fighting(4, 2)
    played.reset("Ironclad", "contract", 0)
    assert shards.metadata()["training_engine_contracts"] == [played.contract]
    (window,) = shards.windows(shards=8)
    items = samples(window, weights=shards.weights)
    assert len(items) == 20 and {item.source for item in items} == {"rollout"}
    assert all(item.weight == pytest.approx(1 / 20) and item.phase == "combat_play" and item.branches == 1 for item in items)
    # Each stored run is the sampled run: its fights counted, its other decisions scored as stored.
    for run in window:
        assert run["source"] == "rollout" and [m["fight"] for m in run["macros"]] == [1, None]
        assert all(abs(stored - again) < 1e-4 for stored, again in replayed(learner.model, learner.vocabulary, run))
    metrics = learner.bootstrap(shards, epochs=1, window_shards=1)
    assert metrics[0]["updates"] > 0 and metrics[0]["weight"] == pytest.approx(1.0)
    # Rebuilding the index keeps what it says of each sampled run.
    rebuild_index(directory)
    assert read_index(directory) == rows


def test_a_run_that_did_not_finish_is_quarantined_and_fails_the_round(tmp_path):
    _, runner, seeds = sampled_runner()
    directory = tmp_path / "round"
    with Samplers(lambda: Failing(4, 2), runner.model, runner.vocabulary, engines=4, processes=2) as pool:
        with pytest.raises(ProtocolError, match="Round incomplete.*Silent-0"):
            sample_round(pool, runner, seeds, directory, journals=True)
    with gzip.open(directory / "quarantine/Silent-0.json.gz", "rt") as stream:
        broken = json.loads(stream.read())
    assert broken["status"] == "error" and "broke off" in broken["error"]
    summary = json.loads((directory / "summary.json").read_text())
    assert (summary["accepted_runs"], summary["quarantined_runs"]) == (19, 1) and "Silent-0" in summary["quarantine"]
    assert summary["report"]["pooled"]["attempts"] == 20
    assert summary["report"]["comparable"] is False
    assert summary["report"]["equal_character_win_rate"] == pytest.approx(summary["wins"] / 20)
    # The rest of the round is readable, every run with its journal this time (the
    # broken run's journal was opened before its engine broke).
    assert len(RunShards(directory)) == 19 and len(list(directory.glob("*.jsonl"))) == 20


class Scripted:
    """What the engines of the command line are replaced with in the test."""

    def __init__(self, root=None):
        self.engine = Fighting(4, 2)

    def __getattr__(self, name):
        return getattr(self.engine, name)


def test_collect_writes_a_round_and_evaluate_writes_full_runs(tmp_path, monkeypatch, capsys):
    from main import cli as cli_module

    assert main(["smoke", "--output", str(tmp_path / "smoke")]) == 0
    checkpoint = tmp_path / "smoke/checkpoint"
    seeds = tmp_path / "seeds.json"
    seeds.write_text(json.dumps({c: [f"{c}-cli"] for c in CHARACTERS}))
    monkeypatch.setattr(cli_module, "CliEngine", Scripted)
    capsys.readouterr()
    assert main(["collect", "--checkpoint", str(checkpoint), "--seeds", str(seeds), "--output", str(tmp_path / "rounds"),
                 "--workers", "2", "--iteration", "5"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert (summary["iteration"], summary["accepted_runs"]) == (5, 5)
    shards = RunShards(tmp_path / "rounds/round-0")
    assert len(shards) == 5 and {row["iteration"] for row in shards.rows} == {5}
    # The next round of the same output goes to round-1, numbered as its iteration by default.
    assert main(["collect", "--checkpoint", str(checkpoint), "--seeds", str(seeds), "--output", str(tmp_path / "rounds"),
                 "--workers", "2", "--journals"]) == 0
    assert json.loads(capsys.readouterr().out)["iteration"] == 1
    assert len(list((tmp_path / "rounds/round-1").glob("*.jsonl"))) == 5
    # Evaluation keeps every run with every frame beside its journal, and no index.
    assert main(["evaluate", "--checkpoint", str(checkpoint), "--seeds", str(seeds), "--output", str(tmp_path / "eval"),
                 "--workers", "2"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["comparable"] is True
    runs = [json.loads(p.read_text()) for p in sorted((tmp_path / "eval").glob("*.json"))]
    assert len(runs) == 5 and all(run["source"] == "evaluation" and run["iteration"] is None for run in runs)
    assert all("steps" in macro for run in runs for macro in run["macros"])
    assert len(list((tmp_path / "eval").glob("*.jsonl"))) == 5 and not (tmp_path / "eval" / INDEX).exists()
    # An evaluation can play another ascension than the checkpoint was trained at.
    assert main(["evaluate", "--checkpoint", str(checkpoint), "--seeds", str(seeds), "--output", str(tmp_path / "a3"),
                 "--workers", "2", "--ascension", "3"]) == 0
    assert json.loads(capsys.readouterr().out)["ascensions"] == [3]
    runs = [json.loads(p.read_text()) for p in sorted((tmp_path / "a3").glob("*.json"))]
    assert {(run["ascension"], run["contract"]["fixed_ascension"]) for run in runs} == {(3, 3)}


def test_a_run_with_packed_frames_makes_the_shard_line_of_the_built_run():
    from collections import Counter

    from main.data.index import member
    from main.play.rollout import _pack, _unpack, packed_line

    _, runner, _ = sampled_runner()
    run = runner.run(fighting(), "Ironclad", "stored", packed=True)
    assert all(isinstance(step["frame"], bytes) for macro in run["macros"] for step in macro["steps"])
    run["environment_actions"] = [dict(frame=_pack(_unpack(run["macros"][0]["steps"][0]["frame"])), action="x")]
    built = dict(run, macros=[dict(macro, steps=[dict(step, frame=_unpack(step["frame"])) for step in macro["steps"]])
                              for macro in run["macros"]],
                 environment_actions=[dict(action, frame=_unpack(action["frame"]))
                                      for action in run["environment_actions"]])
    line, phases = packed_line(run)
    assert json.loads(gzip.decompress(line)) == json.loads(gzip.decompress(member(built)))
    assert phases == dict(Counter(macro["phase"] for macro in policy_macros(built))) == {"combat": 1, "combat_play": 1}
    # A field that reads as a frame's stand-in is refused, not spliced.
    with pytest.raises(ValueError, match="stand-in"):
        packed_line(dict(run, group="\0" + "0"))
