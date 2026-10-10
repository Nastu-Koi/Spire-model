"""The iteration driver: every step leaves its result under the output directory, and
a run that was cut off goes on from the step it was in."""

import gzip
import json
import os
import signal
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch._dynamo

import model
from model import advantage, awr
from model.advantage import FILE
from model.cli import _learner, main, parser
from model.data import RunShards
from model.dataset_index import read_index
from model.history import TrainingHistory
from model.independent import import_samples
from model.iterate import COMPILED_VARIANTS, Command, Driver, Settings, canonical, round_seeds, seeds_of
from model.protocol import CHARACTERS, ProtocolError
from model.tests import journeys
from model.tests.test_student_share import POLICY, rows
from spire_codex_data.sources import split_group


def test_a_rounds_seeds_are_its_own_and_avoid_the_seeds_kept_apart(tmp_path):
    seeds = round_seeds("root", 3, 4)
    assert set(seeds) == set(CHARACTERS) and all(len(listed) == 4 for listed in seeds.values())
    drawn = [seed for listed in seeds.values() for seed in listed]
    assert len(set(drawn)) == 20 and all(len(seed) == 16 and seed == canonical(seed) for seed in drawn)
    # The same round whenever it is asked for; another for another iteration or root.
    assert round_seeds("root", 3, 4) == seeds
    assert not set(drawn) & {seed for listed in round_seeds("root", 4, 4).values() for seed in listed}
    assert not set(drawn) & {seed for listed in round_seeds("other", 3, 4).values() for seed in listed}
    # A longer round begins with the shorter one.
    assert all(round_seeds("root", 3, 6)[c][:4] == seeds[c] for c in CHARACTERS)
    # A seed kept apart is passed over: as it is written, as the game reads it, or as
    # the split group demonstrations are recorded under.
    apart = {seeds["Ironclad"][0].lower(), seeds["Silent"][1].replace("0", "O"), split_group(seeds["Defect"][2])}
    again = round_seeds("root", 3, 4, apart)
    assert seeds["Ironclad"][0] not in again["Ironclad"] and again["Ironclad"][:3] == seeds["Ironclad"][1:]
    assert seeds["Silent"][1] not in again["Silent"] and seeds["Defect"][2] not in again["Defect"]
    assert again["Regent"] == seeds["Regent"] and all(len(listed) == 4 for listed in again.values())
    by_character, plain = tmp_path / "by-character.json", tmp_path / "plain.json"
    by_character.write_text(json.dumps(dict(Ironclad=["A1"], Silent=["B2", "C3"])))
    plain.write_text(json.dumps(["D4"]))
    assert seeds_of(by_character) == {"A1", "B2", "C3"} and seeds_of(plain) == {"D4"}


class Solver:
    """Stands in for the commands that label a round: the fights (`run`), the export
    and the import. `fail` names the ones that end badly."""

    def __init__(self, fail=()):
        self.fail, self.calls = set(fail), []

    def __call__(self, command, log):
        command = list(map(str, command))
        kind = "import" if "import-independent" in command else command[3]
        self.calls.append(kind)
        output = Path(command[command.index("--output") + 1])
        status = 1 if kind in self.fail else {"run": self.fight, "export": self.export, "import": self.bring}[kind](
            command, output)
        return SimpleNamespace(wait=lambda: status, stop=lambda: None)

    def fight(self, command, work):
        work.mkdir(parents=True, exist_ok=True)
        (work / "summary.json").write_text(json.dumps(dict(
            selection=dict(runs_selected=2), run_failures={}, counts=dict(fights=2, verified=2, outcome_win=2))))
        return 0

    def export(self, command, work):
        with gzip.open(work / "independent-training.jsonl.gz", "wt") as stream:
            for number in range(2):
                for row in rows(f"{work.name}-{number}", f"{work.name}-{number}", policy=POLICY):
                    stream.write(json.dumps(row) + "\n")
        return 0

    def bring(self, command, target):
        source = command[command.index("import-independent") + 1]
        return 0 if import_samples(source, target, 250, 1)["accepted_runs"] else 2


@pytest.fixture
def world(tmp_path):
    """A policy to start from, demonstrations, a critic profile and evaluation seeds,
    one of which a round would otherwise be played on."""
    listed = {character: [f"EVAL-{character}-{index}" for index in range(2)] for character in CHARACTERS}
    listed["Ironclad"][0] = round_seeds("test", 0, 2)["Ironclad"][0]
    seeds = tmp_path / "evaluation-seeds.json"
    seeds.write_text(json.dumps(listed))
    return SimpleNamespace(
        root=tmp_path, output=tmp_path / "iterate", checkpoint=journeys.checkpoint(tmp_path / "policy"),
        demonstrations=journeys.demonstrations(tmp_path / "demonstrations"),
        profile=journeys.profile(tmp_path / "profile.json"), seeds=seeds)


def settings(world, **changes):
    return Settings(**{**dict(
        checkpoint=str(world.checkpoint), demonstrations=[str(world.demonstrations)], runs_per_character=2,
        seed_root="test", workers=2, buffer_iterations=2, critic_config=str(world.profile), critic_gate=False,
        eval_seeds=str(world.seeds), evaluate_every=2), **changes})


def driver(world, launch=None, **changes):
    return Driver(world.output, settings(world, **changes), engine=journeys.engine, launch=launch or Solver(),
                  say=lambda _: None)


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def test_two_iterations_sample_label_value_weigh_update_and_evaluate(world):
    solver = Solver()
    state = driver(world, solver).run(2)
    first, second = state["iterations"]["0"], state["iterations"]["1"]
    assert first["complete"] and second["complete"] and read(world.output / "state.json") == state
    assert set(first) == {"settings", "sample", "labels", "critic", "update", "complete"}
    assert set(second) == set(first) | {"evaluation"}

    # Sampled: a round per iteration, by the policy as it stood, on seeds of the
    # iteration's own that are none of the evaluation's.
    start = _learner(world.checkpoint, "cpu")[0].policy_version
    apart = seeds_of(world.seeds)
    for number, entry in enumerate((first, second)):
        directory = world.output / "buffer" / f"round-{number}"
        seeds = read(world.output / "seeds" / f"iteration-{number:04d}.json")
        assert seeds == round_seeds("test", number, 2, apart) == read(directory / "metadata/seeds.json")
        assert not apart & {seed for listed in seeds.values() for seed in listed}
        assert (entry["sample"]["round"], entry["sample"]["runs"], entry["sample"]["decisions"]) == (
            f"round-{number}", 10, 30)
        assert entry["sample"]["policy_version"] == start + number
        assert {(row["iteration"], row["policy_version"]) for row in read_index(directory)} == {(number, start + number)}
        assert entry["sample"]["mean_floor"] == 3.0 and 0 <= entry["sample"]["win_rate"] <= 1
    assert read(world.output / "seeds/iteration-0000.json") != round_seeds("test", 0, 2)
    assert first["sample"]["policy_digest"] != second["sample"]["policy_digest"]

    # Labelled: the solver's commands in order, every round's import kept as data.
    assert solver.calls == ["run", "export", "import"] * 2
    assert first["labels"] == dict(fraction=0.25, selected_runs=2, failed_runs=0, fights=2, verified=2, wins=2,
                                   groups=2, macros=4, data="student-import/round-0")
    assert RunShards(world.output / "student-import/round-1").student_macros == 4

    # Valued: a critic per iteration, the second continuing the first over the buffer.
    assert (first["critic"]["directory"], first["critic"]["version"], first["critic"]["pretrained"]) == (
        "critic/iteration-0000", 1, None)
    assert (second["critic"]["version"], second["critic"]["rounds"]) == (2, ["round-0", "round-1"])
    assert second["critic"]["pretrained"] == str(world.output / "critic/iteration-0000")
    assert (first["critic"]["decisions"], second["critic"]["decisions"]) == (30, 60)

    # Weighed: both rounds of the buffer by the second critic, beside the rounds.
    for number in range(2):
        header = advantage.header(world.output / "buffer" / f"round-{number}")
        assert (header["kind"], header["lam"], header["critic"]["version"], header["decisions"]) == ("critic", 0.9, 2, 30)
    assert [entry["round"] for entry in second["update"]["advantages"]] == ["round-0", "round-1"]

    # Updated: the policy of each iteration, saved with what it was made from.
    demonstrations = str(world.demonstrations)
    assert first["update"]["supervised"] == [demonstrations, str(world.output / "student-import/round-0")]
    assert second["update"]["supervised"] == first["update"]["supervised"] + [str(world.output / "student-import/round-1")]
    assert (first["update"]["policy"], second["update"]["policy"]) == ("policy/iteration-0000", "policy/iteration-0001")
    assert (first["update"]["policy_version"], second["update"]["policy_version"]) == (start + 1, start + 2)
    assert first["update"]["gate"]["enforced"] is False and "heldout_seeds" in first["update"]["gate"]["failed"]
    assert first["update"]["temperature"]["rule"] == "tuned" and 0.1 <= first["update"]["beta"] <= 100
    assert 0 < first["update"]["effective_share_min"] <= first["update"]["effective_share"] <= 1
    assert {"rollout", "demonstration"} <= set(first["update"]["sources"]) <= {"rollout", "demonstration", "student_battle"}
    assert first["update"]["labelled"]["student_macros"] == 4 and second["update"]["labelled"]["student_macros"] == 8
    # As many updates as a pass over the newest round makes, drawn from the whole buffer.
    learner, manifest = _learner(world.output / "policy/current", "cpu")
    passes = [awr.pass_updates([world.output / "buffer" / f"round-{number}"], learner.config) for number in range(2)]
    assert [first["update"]["pass_updates"], second["update"]["pass_updates"]] == passes and min(passes) >= 7
    assert (first["update"]["updates"], second["update"]["updates"]) == (passes[0], sum(passes))
    assert second["update"]["sampled"]["macros"] > first["update"]["sampled"]["macros"]
    progress = manifest["progress"]
    assert (learner.policy_version, learner.updates) == (start + 2, sum(passes))
    assert (progress["iteration"], progress["advantage"], progress["critic"]) == (
        1, "critic", dict(version=2, directory="critic/iteration-0001"))
    assert set(seeds["Ironclad"]) <= set(progress["training_seeds"])
    assert read(world.output / "policy/iteration-0000/manifest.json")["progress"]["iteration"] == 0

    # Evaluated: after every second iteration, greedily, on the evaluation's seeds.
    assert second["evaluation"]["directory"] == "evaluation/iteration-0001" and second["evaluation"]["runs"] == 10
    report = read(world.output / "evaluation/iteration-0001/report.json")
    assert (report["policy_version"], report["comparable"], report["pooled"]["attempts"]) == (start + 2, True, 10)
    assert read(world.output / "evaluation/iteration-0001/metadata/seeds.json") == read(world.seeds)

    # What the monitor reads: an update record for every update, a summary for every iteration.
    updates, history = lines(world.output / "updates.jsonl"), lines(world.output / "history.jsonl")
    assert len(updates) == sum(passes) and {record["kind"] for record in updates + history} == {"iterate"}
    assert [record["round"] for record in updates] == [0] * passes[0] + [1] * passes[1]
    assert {record["stage"] for record in updates} == {"awr"}
    assert all(key in updates[0]["metrics"] for key in ("beta", "effective_share", "awr_loss", "supervised_loss", "loss"))
    assert [record["round"] for record in history] == [0, 1]
    metrics = history[1]["metrics"]
    assert metrics["updates"] == sum(passes) and metrics["policy_updated"] is True
    assert (metrics["sampled_runs"], metrics["sampled_mean_floor"], metrics["buffer_rounds"]) == (10, 3.0, 2)
    assert metrics["critic_version"] == 2 and metrics["critic_gate_passed"] is False
    assert all(key in metrics for key in (
        "sampled_win_rate", "critic_win_ev", "critic_act_ev", "beta", "effective_share", "awr_loss",
        "supervised_loss", "label_fights", "student_macros", "eval_win_rate", "eval_mean_floor"))
    assert "eval_win_rate" not in history[0]["metrics"]
    assert read(world.output / "status.json")["state"] == "completed"
    assert not (world.output / "interrupted").exists()
    # A process that samples, values and updates in turn may compile each way of running its layers.
    assert torch._dynamo.config.recompile_limit >= COMPILED_VARIANTS == 64


def test_a_run_goes_on_from_the_step_it_was_cut_off_in(world):
    first = driver(world).run(1)
    assert first["iterations"]["0"]["complete"] and "1" not in first["iterations"]
    kept = json.dumps(first["iterations"]["0"], sort_keys=True)
    sampled = (world.output / "buffer/round-0/summary.json").stat().st_mtime_ns

    # Cut off while the second round's labels were imported: the round stands, and so
    # does the first iteration.
    with pytest.raises(ProtocolError, match="import of round 1"):
        driver(world, Solver(fail=["import"])).run(2)
    state = read(world.output / "state.json")
    assert json.dumps(state["iterations"]["0"], sort_keys=True) == kept
    assert set(state["iterations"]["1"]) == {"settings", "sample"}
    status = read(world.output / "status.json")
    assert (status["state"], status["round"], status["step"]) == ("failed", 1, "labels") and "import" in status["error"]
    again = (world.output / "buffer/round-1/summary.json").stat().st_mtime_ns
    # What the cut-off import left is in the way of the second attempt.
    (world.output / "student-import/round-1").mkdir(parents=True)
    (world.output / "student-import/round-1/half-written").write_text("")

    solver = Solver()
    state = driver(world, solver).run(2)
    assert state["iterations"]["1"]["complete"] and json.dumps(state["iterations"]["0"], sort_keys=True) == kept
    # Neither round was played again; the labels were made from the fights anew.
    assert (world.output / "buffer/round-0/summary.json").stat().st_mtime_ns == sampled
    assert (world.output / "buffer/round-1/summary.json").stat().st_mtime_ns == again
    assert solver.calls == ["run", "export", "import"]
    assert (world.output / "interrupted/student-import-round-1.0/half-written").exists()
    assert state["iterations"]["1"]["labels"]["data"] == "student-import/round-1"

    # A round cut off while it was played is set aside and played again.
    (world.output / "buffer/round-2/metadata").mkdir(parents=True)
    (world.output / "buffer/round-2/metadata/seeds.json").write_text("{}")
    state = driver(world).run(3)
    assert state["iterations"]["2"]["sample"]["runs"] == 10
    assert (world.output / "interrupted/buffer-round-2.0/metadata/seeds.json").read_text() == "{}"
    # The buffer keeps two iterations: the first round is left on disk, unread.
    assert state["iterations"]["2"]["critic"]["rounds"] == ["round-1", "round-2"]
    assert [entry["round"] for entry in state["iterations"]["2"]["update"]["advantages"]] == ["round-1", "round-2"]
    assert (world.output / "buffer/round-0/summary.json").exists()
    # Labels are kept from every round, in the buffer or out of it.
    assert len(state["iterations"]["2"]["update"]["supervised"]) == 4

    # Cut off after the update was saved and before it was recorded: the policy in hand
    # is the updated one, and it is not updated a second time.
    recorded = state["iterations"]["2"]["update"]
    for step in ("update", "complete"):
        del state["iterations"]["2"][step]
    (world.output / "state.json").write_text(json.dumps(state))
    written = len(lines(world.output / "updates.jsonl"))
    state = driver(world).run(3)
    assert state["iterations"]["2"]["update"] == recorded and state["iterations"]["2"]["complete"]
    assert len(lines(world.output / "updates.jsonl")) == written
    assert _learner(world.output / "policy/current", "cpu")[0].updates == recorded["updates"]
    # Finished iterations are left alone; another checkpoint's or seed root's output is
    # not continued, nor one that another process is writing.
    assert driver(world).run(3) == state
    with pytest.raises(ValueError, match="another checkpoint or seed root"):
        driver(world, seed_root="other")
    held = TrainingHistory(world.output, "iterate", {})
    try:
        with pytest.raises(ValueError, match="Another trainer"):
            driver(world).run(3)
    finally:
        held.close()


def test_a_critic_that_fails_the_gate_leaves_the_policy_as_it_is(world):
    state = driver(world, critic_gate=True, labels=False).run(1)
    update = state["iterations"]["0"]["update"]
    assert update["skipped"] == "critic_gate" and "heldout_seeds" in update["failed"]
    assert update["gate"]["enforced"] is True and update["gate"]["passed"] is False
    assert not (world.output / "policy").exists() and not (world.output / "updates.jsonl").exists()
    assert not (world.output / "buffer/round-0" / FILE).exists()
    metrics = lines(world.output / "history.jsonl")[0]["metrics"]
    assert metrics["policy_updated"] is False and metrics["critic_gate_passed"] is False and "awr_loss" not in metrics
    # The next iteration samples with the same policy; with the gate off its critic weighs.
    state = driver(world, critic_gate=False, labels=False).run(2)
    assert state["iterations"]["1"]["sample"]["policy_version"] == state["iterations"]["0"]["sample"]["policy_version"]
    assert state["iterations"]["1"]["update"]["policy"] == "policy/iteration-0001"
    assert state["iterations"]["0"]["update"]["skipped"] == "critic_gate"
    # The floors are settings of the invocation.
    loose = driver(world, critic_gate=True, labels=False, gate=["min_heldout_seeds=1"])
    assert "heldout_seeds" not in loose.gate(state["iterations"]["1"]["critic"])["failed"]
    # Labels a finished iteration did without are not made for it afterwards.
    solver = Solver()
    assert driver(world, solver).run(2) == state and solver.calls == []


def test_the_controls_run_without_a_critic_and_without_weighting(world):
    state = driver(world, advantage="return", beta="inf", labels=False, updates="3", eval_seeds=None).run(2)
    for number in ("0", "1"):
        entry = state["iterations"][number]
        assert set(entry) == {"settings", "sample", "update", "complete"}
        assert entry["update"]["pass_updates"] == 3 and entry["update"]["gate"] is None
        assert entry["update"]["beta"] is None and entry["update"]["effective_share"] == 1.0
        assert entry["update"]["temperature"] == dict(rule="none")
        assert {item["kind"] for item in entry["update"]["advantages"]} == {"return"}
        assert entry["update"]["supervised"] == [str(world.demonstrations)]
    assert not (world.output / "critic").exists() and not (world.output / "evaluation").exists()
    assert advantage.header(world.output / "buffer/round-1")["baseline"].keys() == set(CHARACTERS)
    assert read(world.output / "policy/current/manifest.json")["progress"]["critic"] is None
    # A pass over the whole buffer instead of a fixed number of updates, weighted again.
    state = driver(world, advantage="return", labels=False, updates="buffer", eval_seeds=None).run(3)
    done = state["iterations"]["2"]["update"]
    assert done["pass_updates"] > 3 and done["rounds"] == ["round-1", "round-2"] and done["temperature"]["rule"] == "tuned"
    assert done["capped_groups"] == [] and state["iterations"]["2"]["settings"]["weight_cap"] == awr.WEIGHT_CAP
    for wrong in (dict(advantage="ppo"), dict(updates="none"), dict(updates="0"), dict(beta="-1"),
                  dict(label_fraction=2.0), dict(runs_per_character=0), dict(weight_cap=0.5)):
        with pytest.raises(ValueError):
            settings(world, **wrong)


def test_labels_made_in_the_background_enter_the_next_update(world):
    solver = Solver()
    state = driver(world, solver, labels_in_background=True).run(2)
    first, second = state["iterations"]["0"], state["iterations"]["1"]
    # The first update came before its round's labels; they were in hand before the
    # next round's engines ran.
    assert first["update"]["supervised"] == [str(world.demonstrations)]
    assert second["update"]["supervised"] == [str(world.demonstrations), str(world.output / "student-import/round-0")]
    assert first["labels"]["data"] == "student-import/round-0" and second["labels"]["data"] == "student-import/round-1"
    assert solver.calls == ["run", "export", "import"] * 2
    # A solver that fails is tried a second time before the iterations stop.
    failing = Solver(fail=["run"])
    with pytest.raises(ProtocolError, match="labelling of round 2 failed"):
        driver(world, failing).run(3)
    assert failing.calls == ["run", "run"]


def test_a_request_to_stop_ends_the_step_under_way_and_what_it_started(world, monkeypatch):
    started = []

    def launch(command, log):
        started.append(Command(command, log))
        return started[-1]

    monkeypatch.setattr(Driver, "label_command", lambda self, iteration: [
        sys.executable, "-c", "import time; time.sleep(60)"])
    threading.Timer(1.5, os.kill, (os.getpid(), signal.SIGTERM)).start()
    before = signal.getsignal(signal.SIGTERM)
    with pytest.raises(KeyboardInterrupt):
        driver(world, launch).run(1)
    # The solver's command went with the step, and the output says where it stopped.
    assert len(started) == 1 and started[0].process.poll() is not None
    status = read(world.output / "status.json")
    assert (status["state"], status["round"], status["step"]) == ("interrupted", 0, "labels")
    assert signal.getsignal(signal.SIGTERM) is before
    # The round it had sampled stands for the run that goes on.
    assert set(read(world.output / "state.json")["iterations"]["0"]) == {"settings", "sample"}
    monkeypatch.undo()
    assert driver(world).run(1)["iterations"]["0"]["complete"]


def test_the_command_line_runs_the_iterations(world, monkeypatch, capsys):
    import model.engine

    arguments = parser().parse_args([
        "iterate", "--checkpoint", "c", "--demonstrations", "d", "e", "--output", "o", "--iterations", "3"])
    assert (arguments.runs_per_character, arguments.buffer_iterations, arguments.evaluate_every) == (200, 10, 5)
    assert (arguments.label_fraction, arguments.lam, arguments.beta, arguments.updates) == (0.25, 0.9, "auto", "round")
    assert (arguments.advantage, arguments.critic_gate, arguments.ess_share) == ("critic", "on", 0.3)
    assert arguments.weight_cap == awr.WEIGHT_CAP == Settings(checkpoint="c", demonstrations=["d"]).weight_cap
    monkeypatch.setattr(model.engine, "CliEngine", lambda root=None: journeys.engine())
    common = ["iterate", "--checkpoint", str(world.checkpoint), "--demonstrations", str(world.demonstrations),
              "--output", str(world.output), "--iterations", "1"]
    assert main([*common, "--runs-per-character", "2", "--workers", "2", "--seed-root", "cli", "--no-labels",
                 "--advantage", "return", "--beta", "0.5", "--updates", "2", "--exclude-seeds", str(world.seeds)]) == 0
    printed = capsys.readouterr().out
    assert json.loads(printed[printed.rindex("\n{\n"):]) == dict(output=str(world.output), iterations=1)
    state = read(world.output / "state.json")
    assert state["iterations"]["0"]["update"]["beta"] == 0.5 and state["iterations"]["0"]["update"]["pass_updates"] == 2
    assert state["iterations"]["0"]["settings"]["exclude_seeds"] == [str(world.seeds)]
    # Every step said what it came to as it ended, one line each.
    steps = [json.loads(line)["step"] for line in printed[:printed.rindex("\n{\n")].splitlines()]
    assert steps == ["sample", "update", "complete"]
    # An invocation with nothing left to do, and one of another seed root: their statuses say so.
    assert main([*common, "--seed-root", "cli"]) == 0
    assert main([*common, "--seed-root", "another"]) == 1


def test_a_command_runs_this_code_and_is_stopped_with_what_it_started(tmp_path):
    log = tmp_path / "logs" / "command.log"
    command = Command([sys.executable, "-c", "import model, sys; print(model.__file__); sys.exit(3)"], log)
    assert command.wait() == 3
    assert log.read_text().startswith("$ ") and str(Path(model.__file__).resolve()) in log.read_text()
    sleeping = Command([sys.executable, "-c", "import time; time.sleep(60)"], log)
    sleeping.stop()
    assert sleeping.process.poll() is not None
