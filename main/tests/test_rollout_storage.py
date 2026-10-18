"""A sampled run is stored as training reads it: its fights told apart by the engine's
events, their frames kept or counted, the run packed as the line it makes in a shard;
a command-line evaluation keeps every run in full."""

import gzip
import json

import pytest

from main.cli import main
from main.env.control import policy_macros
from main.data.shards import samples, validate_run
from main.data.index import INDEX
from main.env.protocol import CHARACTERS, ProtocolError
from main.play.rollout import RolloutRunner
from main.env.saves import FightTracker
from main.tests.test_sampling import Fighting, setup_round

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
    assert tracker.kind is None
    tracker.events([started(1)])
    fight, entry = tracker.decision()
    assert fight == 1 and entry == dict(fight=1, encounter_id="e1", encounter="ENCOUNTER.X", kind="normal", act=1,
                                        from_event=False, map_point_type="monster", second_boss=False)
    # Later decisions of the fight belong to it without being its entry; the kind is the fight's.
    assert tracker.decision() == (1, None) and tracker.kind == "normal"
    # The first decision after the win is outside: the fight ended with the win.
    tracker.events([completed(1)])
    assert tracker.decision() == (None, None) and tracker.kind is None
    tracker.events([started(2, "boss"), ])
    assert tracker.decision()[1]["second_boss"] is False and tracker.kind == "boss"
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


class Scripted:
    """What the engines of the command line are replaced with in the test."""

    def __init__(self, root=None):
        self.engine = Fighting(4, 2)

    def __getattr__(self, name):
        return getattr(self.engine, name)


def test_evaluate_writes_full_runs(tmp_path, monkeypatch, capsys):
    from main import commands as cli_module

    assert main(["smoke", "--output", str(tmp_path / "smoke")]) == 0
    checkpoint = tmp_path / "smoke/checkpoint"
    seeds = tmp_path / "seeds.json"
    seeds.write_text(json.dumps({c: [f"{c}-cli"] for c in CHARACTERS}))
    monkeypatch.setattr(cli_module, "CliEngine", Scripted)
    capsys.readouterr()
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
