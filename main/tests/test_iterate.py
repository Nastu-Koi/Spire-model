"""The iteration driver: every step leaves its result under the output directory, and
a run that was cut off goes on from the step it was in."""

import json
import os
import signal
import threading
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch._dynamo

from main.learn import streams
from main.learn.advantage import FILE
from main.cli import main, parser
from main.learn.history import TrainingHistory
from main.learn.trainer import Learner
from main.network.config import LearningRates
from main.iterate import COMPILED_VARIANTS, Driver, Settings
from main.env.seeds import canonical, round_ascensions, round_seeds, seeds_of, split_group
from main.env.protocol import CHARACTERS, ProtocolError
from main.play import digests
from main.play.groups import groups_summary
from main.tests import journeys
from main.tests.journeys import Journey


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


@pytest.fixture
def world(tmp_path):
    """A policy to start from and evaluation seeds, one of which a round would
    otherwise be played on."""
    listed = {character: [f"EVAL-{character}-{index}" for index in range(2)] for character in CHARACTERS}
    listed["Ironclad"][0] = round_seeds("test", 0, 1)["Ironclad"][0]
    seeds = tmp_path / "evaluation-seeds.json"
    seeds.write_text(json.dumps(listed))
    return SimpleNamespace(root=tmp_path, output=tmp_path / "iterate", checkpoints=tmp_path / "checkpoints/iterate",
                           checkpoint=journeys.checkpoint(tmp_path / "policy"), seeds=seeds)


def chain():
    """Runs of two acts of six rooms: every segment reaches its cut."""
    return Journey(4, 2, rooms=6, acts=2)


def settings(world, **changes):
    return Settings(**{**dict(
        checkpoint=str(world.checkpoint), boundaries=[2, 3], group_n=2, group_m=[2, 2], runs_per_character=1,
        seed_root="test", workers=2, eval_seeds=str(world.seeds), evaluate_every=2), **changes})


def driver(world, **changes):
    return Driver(world.output, settings(world, **changes), checkpoints=world.checkpoints, engine=chain,
                  say=lambda _: None)


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def test_a_run_goes_on_from_the_step_it_was_cut_off_in(world):
    first = driver(world, eval_seeds=None).run(1)
    assert first["iterations"]["0"]["complete"] and "1" not in first["iterations"]
    kept = json.dumps(first["iterations"]["0"], sort_keys=True)
    sampled = (world.output / "groups/round-0/summary.json").stat().st_mtime_ns

    # Cut off while the second round's advantages were computed: the round stands, and
    # so does the first iteration.
    def failing(self, iteration):
        raise ProtocolError("advantages of round 1")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Driver, "advantages", failing)
        with pytest.raises(ProtocolError, match="advantages of round 1"):
            driver(world, eval_seeds=None).run(2)
    state = read(world.output / "state.json")
    assert json.dumps(state["iterations"]["0"], sort_keys=True) == kept
    assert set(state["iterations"]["1"]) == {"settings", "sample"}
    status = read(world.output / "status.json")
    assert (status["state"], status["round"], status["step"]) == ("failed", 1, "advantages")
    again = (world.output / "groups/round-1/summary.json").stat().st_mtime_ns

    state = driver(world, eval_seeds=None).run(2)
    assert state["iterations"]["1"]["complete"] and json.dumps(state["iterations"]["0"], sort_keys=True) == kept
    # Neither round was played again.
    assert (world.output / "groups/round-0/summary.json").stat().st_mtime_ns == sampled
    assert (world.output / "groups/round-1/summary.json").stat().st_mtime_ns == again

    # A round cut off on other seeds is set aside and played again.
    (world.output / "groups/round-2/metadata").mkdir(parents=True)
    (world.output / "groups/round-2/metadata/seeds.json").write_text("{}")
    state = driver(world, eval_seeds=None).run(3)
    assert state["iterations"]["2"]["sample"]["written"] == 70
    assert read(world.output / "groups/round-2/metadata/seeds.json") == read(world.output / "seeds/iteration-0002.json")
    # What it left was set aside for the second attempt and removed at the cleanup.
    assert not (world.output / "interrupted").exists()

    # Cut off after the update was saved and before it was recorded: the policy in hand
    # is the updated one, and it is not updated a second time.
    recorded = state["iterations"]["2"]["update"]
    for step in ("update", "cleanup", "complete"):
        del state["iterations"]["2"][step]
    (world.output / "state.json").write_text(json.dumps(state))
    written = len(lines(world.output / "updates.jsonl"))
    state = driver(world, eval_seeds=None).run(3)
    assert state["iterations"]["2"]["update"] == recorded and state["iterations"]["2"]["complete"]
    assert len(lines(world.output / "updates.jsonl")) == written
    assert Learner.load(world.checkpoints / "current", "cpu")[0].updates == recorded["updates"]
    # Finished iterations are left alone; another checkpoint's or seed root's output is
    # not continued, nor one that another process is writing.
    assert driver(world, eval_seeds=None).run(3) == state
    with pytest.raises(ValueError, match="another checkpoint or seed root"):
        driver(world, seed_root="other")
    held = TrainingHistory(world.output, "iterate", {})
    try:
        with pytest.raises(ValueError, match="Another trainer"):
            driver(world).run(3)
    finally:
        held.close()


def test_a_request_to_stop_ends_the_step_under_way(world, monkeypatch):
    def stopped(self, iteration):
        # The request arrives while the step is under way.
        threading.Timer(0.2, os.kill, (os.getpid(), signal.SIGTERM)).start()
        time.sleep(60)

    monkeypatch.setattr(Driver, "advantages", stopped)
    before = signal.getsignal(signal.SIGTERM)
    with pytest.raises(KeyboardInterrupt):
        driver(world).run(1)
    status = read(world.output / "status.json")
    assert (status["state"], status["round"], status["step"]) == ("interrupted", 0, "advantages")
    assert signal.getsignal(signal.SIGTERM) is before
    # The round it had sampled stands for the run that goes on.
    assert set(read(world.output / "state.json")["iterations"]["0"]) == {"settings", "sample"}
    monkeypatch.undo()
    assert driver(world).run(1)["iterations"]["0"]["complete"]


def test_the_command_line_runs_the_iterations(world, monkeypatch, capsys):
    from main.env import engine as engine_module

    arguments = parser().parse_args(["iterate", "--checkpoint", "c", "--output", "o", "--iterations", "3"])
    assert (arguments.runs_per_character, arguments.evaluate_every, arguments.updates) == (200, 5, "round")
    assert (arguments.clip, arguments.kl) == (0.2, 0.01) and arguments.wave_shares is None
    assert arguments.weight_cap == streams.WEIGHT_CAP == Settings(checkpoint="c").weight_cap
    monkeypatch.setattr(engine_module, "CliEngine", lambda root=None: chain())
    common = ["iterate", "--checkpoint", str(world.checkpoint), "--output", str(world.output),
              "--checkpoints", str(world.checkpoints), "--iterations", "1",
              "--boundary-acts", "2", "3", "--group-n", "2", "--group-m", "2", "2"]
    assert main([*common, "--runs-per-character", "1", "--workers", "2", "--seed-root", "cli",
                 "--kl", "0.05", "--updates", "2", "--exclude-seeds", str(world.seeds)]) == 0
    printed = capsys.readouterr().out
    assert json.loads(printed[printed.rindex("\n{\n"):]) == dict(output=str(world.output), iterations=1)
    state = read(world.output / "state.json")
    assert state["iterations"]["0"]["update"]["pass_updates"] == 2
    assert state["iterations"]["0"]["update"]["kl"] == 0.05
    assert state["iterations"]["0"]["settings"]["exclude_seeds"] == [str(world.seeds)]
    # Every step said what it came to as it ended, one line each.
    steps = [json.loads(line)["step"] for line in printed[:printed.rindex("\n{\n")].splitlines()]
    assert steps == ["sample", "advantages", "update", "cleanup", "complete"]
    # An invocation with nothing left to do, and one of another seed root: their statuses say so.
    assert main([*common, "--seed-root", "cli"]) == 0
    assert main([*common, "--seed-root", "another"]) == 1
    # The policies are in the checkpoints directory; the output holds only the records.
    assert (world.checkpoints / "current/manifest.json").exists() and not (world.output / "policy").exists()
    assert state["checkpoints"] == str(world.checkpoints)


def test_a_runs_checkpoints_go_under_checkpoints_and_stay_where_the_run_first_put_them(world, tmp_path, monkeypatch):
    from main.network.checkpoint import checkpoints_for

    # By default the output's path under runs/ is repeated under checkpoints/; an output
    # elsewhere names its own, and none is kept under runs/.
    monkeypatch.chdir(tmp_path)
    assert checkpoints_for("runs/grpo/train/v1") == Path("checkpoints/grpo/train/v1")
    assert checkpoints_for(tmp_path / "runs/grpo/v1") == Path("checkpoints/grpo/v1")
    assert checkpoints_for("elsewhere/v1", "mine") == Path("mine")
    for output, given in (("elsewhere/v1", None), ("runs", None), ("runs/v1", "runs/v1/policy")):
        with pytest.raises(ValueError):
            checkpoints_for(output, given)
    arguments = parser().parse_args(["iterate", "--checkpoint", "c", "--output", "o", "--iterations", "1"])
    assert arguments.checkpoints is None

    state = driver(world, eval_seeds=None).run(1)
    assert state["checkpoints"] == str(world.checkpoints) and (world.checkpoints / "iteration-0000").exists()
    assert state["iterations"]["0"]["update"]["policy"] == str(world.checkpoints / "iteration-0000")
    # Continued with another checkpoints directory: refused, not started over.
    with pytest.raises(ValueError, match="are in"):
        Driver(world.output, settings(world, eval_seeds=None), checkpoints=world.root / "moved", engine=chain,
               say=lambda _: None)
    # Its policies gone after an update: refused, where loading the starting checkpoint
    # would silently undo the updates made.
    (world.checkpoints / "current").rename(world.root / "aside")
    with pytest.raises(ValueError, match="is missing"):
        driver(world, eval_seeds=None).run(2)
    # A run of the old layout, its policies in the output: told where they go.
    (world.root / "aside").rename(world.output / "policy")
    with pytest.raises(ValueError, match="move them there"):
        driver(world, eval_seeds=None)


def test_act_group_iterations_sample_waves_price_them_update_and_clean_up(world):
    state = driver(world).run(2)
    # The driver's process may hold more compiled variants of a function than the default.
    assert any(getattr(torch._dynamo.config, name, 0) >= COMPILED_VARIANTS
               for name in ("recompile_limit", "cache_size_limit"))
    first, second = state["iterations"]["0"], state["iterations"]["1"]
    assert set(first) == {"settings", "sample", "advantages", "update", "cleanup", "complete"}
    assert set(second) == set(first) | {"evaluation"}
    assert first["settings"]["group_m"] == [2, 2] and first["settings"]["kl"] == 0.01

    # Sampled: a round of three waves on seeds of the iteration's own; every segment of
    # these rooms reaches its cut, so every save is played on from.
    sampled = first["sample"]
    assert sampled["round"] == "round-0" and sampled["waves"] == ["act1", "act2", "act3"]
    assert (sampled["groups"], sampled["written"], sampled["failed"]) == (5, 70, 0)
    assert sampled["saves"] == {"act1": 10, "act2": 20} and sampled["reach_rates"] == {"act1": 1.0, "act2": 1.0}
    assert 0 <= sampled["win_rate"] <= 1 and "wins" not in sampled
    assert all(sampled["collector"][wave]["passes"] > 0 for wave in sampled["waves"])
    assert read(world.output / "seeds/iteration-0000.json") == read(world.output / "groups/round-0/metadata/seeds.json")

    # Priced: every wave's advantages from its own groups.
    for wave, head in first["advantages"].items():
        assert head["runs"] > 0 and head["decisions"] > 0 and head["groups"] > 0 and head["r2_within"] is not None
    assert first["advantages"]["act3"]["groups"] == 20
    assert all(head["dropped"] is None for head in first["advantages"].values())
    assert first["advantages"]["act2"]["advantage"]["n"] > 0

    # Updated with the clipped objective: the waves by their decisions; as many
    # updates as a pass over the priced decisions makes, which sees every wave about once.
    update = first["update"]
    assert 0 < update["entropy"] and (update["clip"], update["kl"]) == (0.2, 0.01)
    assert update["phase_weights"] == "group"
    seen = update["seen"]
    assert seen["samples"] > 0 and 0 < seen["ratio_mean"] and 0 <= seen["clip_fraction"] <= 1 and seen["kl"] >= 0
    priced = {wave: head["decisions"] for wave, head in first["advantages"].items()}
    decisions = sum(priced.values())
    assert update["shares"] == {wave: pytest.approx(count / decisions) for wave, count in priced.items()}
    assert sum(update["sizes"].values()) == 8 and set(update["sizes"]) == {"act1", "act2", "act3"}
    assert set(update["losses"]) == {"act1", "act2", "act3"} and "anchor" not in update and "window" not in update
    assert update["pass_updates"] == -(-decisions // 8) == update["updates"] and "primary" not in update
    assert 0 <= update["wait_seconds"] <= update["seconds"] and 0 <= update["wait_share"] <= 1
    for wave, count in priced.items():
        assert update["waves"][wave]["data"]["macros"] == count
        assert abs(update["waves"][wave]["samples"] - count) <= update["pass_updates"] + 8
    assert all(0 <= update["waves"][wave]["zero_share"] <= 1 for wave in priced)
    assert second["update"]["updates"] == first["update"]["updates"] + second["update"]["pass_updates"]
    progress = read(world.checkpoints / "current/manifest.json")["progress"]
    assert (progress["iteration"], progress["advantage"]) == (1, "act-groups")
    assert set(read(world.output / "seeds/iteration-0001.json")["Ironclad"]) <= set(progress["training_seeds"])

    # Cleaned up: the frames, saves, indexes and advantages of the round are gone, its
    # summaries stay; the archive of an iteration nobody evaluated goes too.
    for wave in ("act1", "act2", "act3"):
        directory = world.output / "groups/round-0" / wave
        for name in ("accepted", "saves", "index.jsonl.gz", FILE, digests.FILE):
            assert not (directory / name).exists()
        assert (directory / "summary.json").exists()
    assert groups_summary(world.output / "groups/round-0")["waves"] == ["act1", "act2", "act3"]
    assert not (world.checkpoints / "iteration-0000").exists() and (world.checkpoints / "iteration-0001").exists()

    # Evaluated after the second iteration, on the evaluation's seeds.
    assert second["evaluation"]["runs"] == 10
    history = lines(world.output / "history.jsonl")
    metrics = history[1]["metrics"]
    assert all(key in metrics for key in (
        "sampled_segments", "sampled_win_rate", "reach_rate_act1", "between_share_act2", "r2_within_act3",
        "pass_updates", "loss", "ratio_mean", "clip_fraction", "kl", "zero_share", "act1_loss", "act2_zero_share",
        "entropy", "eval_win_rate", "update_wait_share"))
    assert 0 <= metrics["update_wait_share"] <= 1
    assert metrics["policy_updated"] is True and "beta" not in metrics and "anchor_loss" not in metrics
    updates = lines(world.output / "updates.jsonl")
    assert {record["stage"] for record in updates} == {"groups"}
    assert all(key in updates[0]["metrics"] for key in ("clip_fraction", "kl", "act1_loss", "act3_zero_share"))
    # Finished iterations are left alone.
    assert driver(world).run(2) == state



def test_a_window_plays_each_group_at_an_ascension_of_its_own_and_evaluates_at_the_checkpoints(world):
    seeds = round_seeds("root", 3, 7)
    levels = round_ascensions("root", 3, seeds, [0, 5])
    for character in CHARACTERS:
        assert set(levels[character]) == set(seeds[character])
        counts = Counter(levels[character].values())
        assert set(counts) <= set(range(6)) and max(counts.values()) - min(counts.get(a, 0) for a in range(6)) <= 1
    assert round_ascensions("root", 3, seeds, [0, 5]) == levels
    assert {a for listed in round_ascensions("root", 3, seeds, [2, 2]).values() for a in listed.values()} == {2}

    state = driver(world, ascensions=[0, 2], evaluate_every=1).run(1)
    first = state["iterations"]["0"]
    assert first["settings"]["ascensions"] == [0, 2]
    assigned = read(world.output / "seeds/iteration-0000-ascensions.json")
    assert assigned == read(world.output / "groups/round-0/metadata/ascensions.json")
    assert assigned == round_ascensions("test", 0, read(world.output / "seeds/iteration-0000.json"), [0, 2])
    for wave in first["sample"]["waves"]:
        counted = first["sample"]["by_ascension"][wave]
        assert set(counted) <= {str(a) for a in assigned_levels(assigned)}
        assert sum(entry["written"] for entry in counted.values()) == groups_summary(
            world.output / "groups/round-0")[wave]["written"]
        assert sum(entry["groups"] for entry in first["advantages"][wave]["by_ascension"].values()) == \
            first["advantages"][wave]["groups"]
    # The evaluation stays at the checkpoint's ascension.
    played = [read(path) for path in (world.output / "evaluation/iteration-0000").glob("*.json") if path.name[0].isupper()]
    assert played and {run["ascension"] for run in played} == {0}
    # Without a window every group is at the checkpoint's.
    assert settings(world).ascensions is None
    for wrong in ([3, 2], [0, 11], [-1, 2], [1], [0.5, 2]):
        with pytest.raises(ValueError, match="ascension window"):
            settings(world, ascensions=wrong)
    arguments = parser().parse_args(["iterate", "--checkpoint", "c", "--output", "o", "--iterations", "1",
                                     "--ascensions", "0", "5"])
    assert arguments.ascensions == [0, 5]
    assert parser().parse_args(["iterate", "--checkpoint", "c", "--output", "o", "--iterations", "1"]).ascensions is None


def assigned_levels(assigned):
    return {level for listed in assigned.values() for level in listed.values()}


def test_act_group_settings_resolve_their_defaults_and_refuse_what_cannot_be_run(world):
    plain = settings(world, group_m=[8])
    assert plain.group_m == [8, 8] and plain.wave_shares is None and (plain.clip, plain.kl) == (0.2, 0.01)
    # A boss cut is one more wave: `group_m` gives one number for each later wave.
    assert plain.bosses == [] and settings(world, bosses=[3], group_m=[4]).group_m == [4, 4, 4]
    assert settings(world, bosses=[3], group_m=[2, 2, 4]).group_m == [2, 2, 4]
    assert plain.phase_weights == "group" and settings(world, phase_weights="even").phase_weights == "even"
    # The waves share a batch by their priced decisions.
    macros = {"act1": 1000, "act2": 2000, "act3": 7000}
    assert plain.group_shares(macros) == {"act1": pytest.approx(0.1), "act2": pytest.approx(0.2), "act3": pytest.approx(0.7)}
    # Or as told, one relative number per wave: `1 1 1` is the waves alike.
    alike = settings(world, wave_shares=[1, 1, 1])
    assert alike.group_shares(macros) == {wave: pytest.approx(1 / 3) for wave in macros}
    assert settings(world, wave_shares=[1, 1, 2]).group_shares(macros) == {
        "act1": pytest.approx(0.25), "act2": pytest.approx(0.25), "act3": pytest.approx(0.5)}
    for wrong in (dict(boundaries=[3, 2]), dict(group_n=0), dict(group_m=[1, 2, 3]), dict(clip=1.0), dict(kl=-0.1),
                  dict(bosses=[0]), dict(bosses=[3], group_m=[2, 2]),
                  dict(updates="none"), dict(updates="buffer"), dict(updates="0"),
                  dict(runs_per_character=0), dict(weight_cap=0.5), dict(phase_weights="uniform"),
                  dict(wave_shares=[1, 1]), dict(wave_shares=[1, 0, 1]), dict(wave_shares=[1, 1, -1])):
        with pytest.raises(ValueError):
            settings(world, **wrong)
    arguments = parser().parse_args([
        "iterate", "--checkpoint", "c", "--output", "o", "--iterations", "1",
        "--boundary-acts", "2", "3", "--group-n", "4", "--group-m", "4", "2", "--wave-shares", "1", "2", "7",
        "--phase-weights", "even", "--boundary-bosses", "3"])
    assert (arguments.boundary_acts, arguments.boundary_bosses, arguments.group_n, arguments.group_m) == ([2, 3], [3], 4, [4, 2])
    assert arguments.wave_shares == [1.0, 2.0, 7.0] and arguments.phase_weights == "even"
    plain_arguments = parser().parse_args(["iterate", "--checkpoint", "c", "--output", "o", "--iterations", "1"])
    assert plain_arguments.wave_shares is None and plain_arguments.boundary_bosses == []


def test_a_run_may_leave_the_uniform_groups_out_of_its_updates(world):
    state = driver(world, eval_seeds=None, drop_uniform_groups=True).run(1)
    entry = state["iterations"]["0"]
    assert entry["settings"]["drop_uniform_groups"] is True
    priced, update = entry["advantages"], entry["update"]
    assert sum(head["dropped"]["groups"] for head in priced.values()) > 0
    for wave, head in priced.items():
        assert head["dropped"]["groups"] == head["uniform_groups"]
        assert update["waves"][wave]["data"]["macros"] == head["decisions"]
    # The shares and the pass are of the decisions kept.
    decisions = sum(head["decisions"] for head in priced.values())
    assert update["decisions"] == decisions and update["pass_updates"] == -(-decisions // 8)
    assert update["shares"] == {wave: pytest.approx(head["decisions"] / decisions) for wave, head in priced.items()}
    assert settings(world).drop_uniform_groups is False
    common = ["iterate", "--checkpoint", "c", "--output", "o", "--iterations", "1"]
    assert parser().parse_args([*common, "--drop-uniform-groups"]).drop_uniform_groups is True
    assert parser().parse_args(common).drop_uniform_groups is False


def test_act_group_iterations_stop_once_the_policy_has_lost_its_entropy(world):
    state = driver(world, eval_seeds=None, min_entropy=100.0).run(3)
    assert state["iterations"]["0"]["complete"] and "1" not in state["iterations"]
    assert "below 100.0" in state["halted"] and read(world.output / "status.json")["state"] == "halted"
    # Without the guard the same output goes on.
    state = driver(world, eval_seeds=None).run(2)
    assert state["iterations"]["1"]["complete"]


def test_act_group_iterations_with_the_clipped_objective_anchor_to_the_checkpoint(world):
    acts = dict(eval_seeds=None)
    # The run's own rates: half the checkpoint's in every group.
    saved = Learner.load(world.checkpoint, "cpu")[0]
    rates = world.root / "rates.json"
    rates.write_text(json.dumps({name: rate / 2 for name, rate in asdict(LearningRates.of(saved.config)).items()}))
    state = driver(world, **acts, learning_rates=str(rates)).run(2)
    first, second = state["iterations"]["0"], state["iterations"]["1"]
    assert first["complete"] and second["complete"]
    update = first["update"]
    assert (update["clip"], update["kl"]) == (0.2, 0.01) and 0 < update["entropy"]
    # The first batches are on-policy: the ratios start at one, the KL at zero.
    seen = update["seen"]
    assert seen["samples"] > 0 and 0 < seen["ratio_mean"] and 0 <= seen["clip_fraction"] <= 1 and seen["kl"] >= 0
    first_update = lines(world.output / "updates.jsonl")[0]["metrics"]
    # Every rate at half the checkpoint's, and the checkpoint the run saves keeps them.
    saved.scheduler.prepare()
    assert first_update["learning_rates"] == {
        group["name"]: pytest.approx(group["lr"] / 2) for group in saved.optimizer.param_groups}
    assert first["settings"]["learning_rates"] == str(rates)
    progress = read(world.checkpoints / "current/manifest.json")["training"]
    assert progress["muon_lr"] == pytest.approx(saved.config.muon_lr / 2)
    assert first_update["ratio_mean"] == pytest.approx(1.0, abs=1e-3) and first_update["kl"] == pytest.approx(0.0, abs=1e-5)
    decisions = sum(head["decisions"] for head in first["advantages"].values())
    assert update["pass_updates"] == -(-decisions // 8) == update["updates"]
    for wave in ("act1", "act2", "act3"):
        assert 0 <= update["waves"][wave]["zero_share"] <= 1 and update["waves"][wave]["data"]["macros"] > 0
    assert second["update"]["updates"] == update["updates"] + second["update"]["pass_updates"]
    metrics = lines(world.output / "history.jsonl")[1]["metrics"]
    assert all(key in metrics for key in ("ratio_mean", "clip_fraction", "kl", "zero_share", "act2_zero_share"))
    assert all(key in first_update for key in ("clip_fraction", "act1_loss", "act3_zero_share"))
    # What cannot be run is refused, a learning-rate file without every group among it.
    wrong_rates = world.root / "wrong-rates.json"
    wrong_rates.write_text(json.dumps(dict(muon=1e-4)))
    for wrong in (dict(clip=1.0), dict(kl=-0.1), dict(learning_rates=str(wrong_rates))):
        with pytest.raises(ValueError):
            settings(world, **{**acts, **wrong})
    arguments = parser().parse_args(["iterate", "--checkpoint", "c", "--output", "o", "--iterations", "1",
                                     "--clip", "0.1", "--kl", "0.02", "--learning-rates", "r.json"])
    assert (arguments.clip, arguments.kl, arguments.learning_rates) == (0.1, 0.02, "r.json")


def test_an_optimizer_step_accumulates_the_logical_batches_its_batch_holds(world):
    # The test policy's logical batch is 8: a step of 16 accumulates two of them.
    state = driver(world, eval_seeds=None, batch=16).run(1)
    update = state["iterations"]["0"]["update"]
    decisions = sum(head["decisions"] for head in state["iterations"]["0"]["advantages"].values())
    assert (update["batch"], update["accumulate"]) == (16, 2)
    # A pass sees every decision once, in half as many steps as batches.
    assert update["pass_updates"] == -(-(-(-decisions // 8)) // 2) == update["updates"]
    steps = [record["metrics"] for record in lines(world.output / "updates.jsonl") if record["round"] == 0]
    assert len(steps) == update["pass_updates"] and all(step["samples"] <= 16 for step in steps)
    assert sum(step["samples"] for step in steps) == update["waves"]["act1"]["samples"] + sum(
        update["waves"][wave]["samples"] for wave in ("act2", "act3"))
    assert steps[0]["loss_scale"] == pytest.approx(decisions / 16)
    assert state["iterations"]["0"]["settings"]["batch"] == 16
    # A batch that is no multiple of the logical batch is refused before anything is sampled.
    with pytest.raises(ValueError, match="multiple"):
        Driver(world.root / "other", settings(world, eval_seeds=None, batch=12), checkpoints=world.root / "other-c",
               engine=chain, say=lambda _: None).run(1)
    for wrong in (0, -16):
        with pytest.raises(ValueError, match="batch"):
            settings(world, eval_seeds=None, batch=wrong)
    assert parser().parse_args(["iterate", "--checkpoint", "c", "--output", "o", "--iterations", "1",
                                "--batch", "2048"]).batch == 2048
    assert parser().parse_args(["iterate", "--checkpoint", "c", "--output", "o", "--iterations", "1"]).batch is None


def test_a_checkpoint_out_of_another_runs_iteration_starts_this_runs_first_update(world):
    acts = dict(eval_seeds=None)
    first = driver(world, **acts).run(1)
    # The policy that iteration 0 of one run left says so in its metadata: another run
    # starting from it has still made no update of its own iteration 0.
    started = world.checkpoints / "current"
    state = Driver(world.root / "other", settings(world, **acts, checkpoint=str(started), seed_root="other"),
                   checkpoints=world.root / "checkpoints/other", engine=chain, say=lambda _: None).run(1)
    update = state["iterations"]["0"]["update"]
    assert update["updates"] == first["iterations"]["0"]["update"]["updates"] + update["pass_updates"]
    assert (world.root / "checkpoints/other/current").exists()


def test_every_round_and_evaluation_of_a_run_is_played_by_the_processes_forked_before_its_first_update(world):
    starts = world.root / "starts"

    def counted():
        # Told from a sampling process: one line for every engine started.
        with starts.open("a") as stream:
            stream.write(f"{os.getpid()}\n")
        return chain()

    state = Driver(world.output, settings(world), checkpoints=world.checkpoints, engine=counted,
                   say=lambda _: None).run(2)
    assert state["iterations"]["1"]["evaluation"]
    pids = starts.read_text().split()
    # Two engines for each round (its waves share them) and for the evaluation, all in
    # the same two processes: the engines rest between rounds, the processes stay.
    assert len(pids) == 2 * 3 and len(set(pids)) == 2


def test_the_kl_reference_is_renewed_every_so_many_iterations(world):
    acts = dict(eval_seeds=None, reference_every=2)
    state = driver(world, **acts).run(3)
    # The reference archive survives the cleanups that remove the others not evaluated,
    # for as long as the next update refers to it.
    archives = lambda: sorted(path.name for path in world.checkpoints.glob("iteration-*"))  # noqa: E731
    assert archives() == ["iteration-0001", "iteration-0002"]
    state = driver(world, **acts).run(4)
    assert archives() == ["iteration-0003"]
    updates = {number: entry["update"] for number, entry in state["iterations"].items()}
    # Iterations 0 and 1 anchor to the starting checkpoint; 2 and 3 to the policy after
    # iteration 1, as it stood when their stretch began.
    assert updates["0"]["reference"] == updates["1"]["reference"] == str(world.checkpoint)
    assert updates["2"]["reference"] == updates["3"]["reference"] == str(world.checkpoints / "iteration-0001")
    # Where the reference is the policy that played the round (the first of a stretch),
    # the KL is to the distributions its decisions were drawn from: no pass of it.
    assert [updates[str(number)]["reference_recorded"] for number in range(4)] == [True, False, True, False]
    # The first update of a stretch is on-policy with its reference: the KL starts at zero.
    first_of_stretch = [record["metrics"] for record in lines(world.output / "updates.jsonl") if record["round"] == 2][0]
    assert first_of_stretch["kl"] == pytest.approx(0.0, abs=1e-5)
    plain = driver(world, eval_seeds=None, reference_every=0)
    assert plain.reference(7) == Path(world.checkpoint)
    renewed = driver(world, **acts)
    assert renewed.reference(5) == world.checkpoints / "iteration-0003"
    with pytest.raises(ValueError):
        settings(world, reference_every=-1)
    arguments = parser().parse_args(["iterate", "--checkpoint", "c", "--output", "o", "--iterations", "1",
                                     "--reference-every", "3"])
    assert arguments.reference_every == 3
    # Unless given, the reference is renewed at the evaluation interval: an evaluated policy.
    assert parser().parse_args(["iterate", "--checkpoint", "c", "--output", "o", "--iterations", "1"]).reference_every is None
    assert Settings(checkpoint="c").reference_every == 5 and Settings(checkpoint="c", evaluate_every=3).reference_every == 3
    assert Settings(checkpoint="c", reference_every=0).reference_every == 0
    assert settings(world).reference_every == 2 and settings(world, reference_every=4).reference_every == 4



def test_an_evaluation_plays_at_the_precision_asked_and_sampling_at_the_checkpoints(world):
    arguments = parser().parse_args(["iterate", "--checkpoint", "c", "--output", "o", "--iterations", "1",
                                     "--eval-precision", "fp32"])
    assert arguments.eval_precision == "fp32"
    assert parser().parse_args(["iterate", "--checkpoint", "c", "--output", "o", "--iterations", "1"]).eval_precision is None
    assert parser().parse_args(["evaluate", "--checkpoint", "c", "--seeds", "s", "--output", "o",
                                "--precision", "fp32"]).precision == "fp32"
    with pytest.raises(SystemExit):
        parser().parse_args(["evaluate", "--checkpoint", "c", "--seeds", "s", "--output", "o", "--precision", "fp16"])
    with pytest.raises(ValueError, match="precision"):
        settings(world, eval_seeds=None, eval_precision="fp16")
    run = driver(world, eval_precision="fp32")
    state = run.run(2)
    assert run.sampler().precision == run.learner.config.precision and run.sampler(precision="fp32").precision == "fp32"
    evaluated = world.output / state["iterations"]["1"]["evaluation"]["directory"]
    played = [read(path) for path in evaluated.glob("*.json") if path.name != "report.json"]
    assert played and all(run_["precision"] == "fp32" for run_ in played)
    # Rounds keep the checkpoint's precision (the cleanup has removed their shards by now).
    assert run.learner.config.precision != "fp32"


def test_the_fork_depth_is_a_setting_of_the_run(world):
    state = driver(world, eval_seeds=None, fork_depth=1).run(1)
    entry = state["iterations"]["0"]
    # Every segment of the round forked once, and its advantages say what forking taught.
    assert all(entry["sample"]["forked"][wave] == count for wave, count in
               {wave: sum(outcomes.values()) for wave, outcomes in entry["sample"]["outcomes"].items()}.items())
    assert all(entry["advantages"][wave]["forked"]["decisions"] > 0 for wave in ("act1", "act2", "act3"))
    assert entry["settings"]["fork_depth"] == 1
    assert settings(world, eval_seeds=None).fork_depth == 0
    for wrong in (-1, 1.5):
        with pytest.raises(ValueError, match="fork"):
            settings(world, eval_seeds=None, fork_depth=wrong)
    assert parser().parse_args(["iterate", "--checkpoint", "c", "--output", "o", "--iterations", "1",
                                "--fork-depth", "2"]).fork_depth == 2
    assert parser().parse_args(["iterate", "--checkpoint", "c", "--output", "o", "--iterations", "1"]).fork_depth == 0


def test_the_potion_exploration_rate_is_a_setting_of_the_run(world):
    arguments = parser().parse_args(["iterate", "--checkpoint", "c", "--output", "o", "--iterations", "1",
                                     "--explore-potions", "0.25"])
    assert arguments.explore_potions == [0.25]
    assert parser().parse_args(["iterate", "--checkpoint", "c", "--output", "o", "--iterations", "1",
                                "--explore-potions", "0.01", "0.03"]).explore_potions == [0.01, 0.03]
    assert parser().parse_args(["iterate", "--checkpoint", "c", "--output", "o", "--iterations", "1"]).explore_potions == [0.0]
    assert settings(world, eval_seeds=None, explore_potions=0.25).explore_potions == {"normal": 0.25, "elite": 0.25, "boss": 0.25}
    assert settings(world, eval_seeds=None, explore_potions=[0.01, 0.03]).explore_potions == {"normal": 0.01, "elite": 0.03, "boss": 0.03}
    assert settings(world, eval_seeds=None).explore_potions == {"normal": 0.0, "elite": 0.0, "boss": 0.0}
    for wrong in (1.5, -0.1, [0.1, 0.2, 0.3, 0.4]):
        with pytest.raises(ValueError, match="exploration"):
            settings(world, eval_seeds=None, explore_potions=wrong)


def test_greedy_fights_are_a_setting_of_the_run_and_its_updates_learn_nothing_of_the_fights(world):
    state = driver(world, eval_seeds=None, greedy_fights=True, fork_depth=1).run(1)
    entry = state["iterations"]["0"]
    assert entry["settings"]["greedy_fights"] is True and entry["sample"]["greedy_fights"] is True
    assert groups_summary(world.output / "groups/round-0")["greedy_fights"] is True
    priced, update = entry["advantages"], entry["update"]
    for wave, head in priced.items():
        # The round's decisions are those outside the fights: no fight has an advantage,
        # and the fights left out are counted.
        assert entry["sample"]["decisions"][wave] == head["decisions"] > 0
        assert head["fights"]["decisions"] == 0 and head["fights"]["excluded"] > 0
        assert update["waves"][wave]["data"]["macros"] == head["decisions"]
    # The shares, the pass and the loss's scale are of those decisions alone.
    decisions = sum(head["decisions"] for head in priced.values())
    assert update["decisions"] == decisions and update["pass_updates"] == -(-decisions // 8)
    assert update["greedy_fights"] is True
    # Without it every fight's decision is priced, as before.
    assert settings(world).greedy_fights is False
    for wrong in (dict(greedy_fights=True, explore_potions=0.1), dict(greedy_fights=True, bosses=[3]),
                  dict(greedy_fights=1)):
        with pytest.raises(ValueError, match="Greedy"):
            settings(world, eval_seeds=None, **wrong)
    common = ["iterate", "--checkpoint", "c", "--output", "o", "--iterations", "1"]
    assert parser().parse_args([*common, "--greedy-fights"]).greedy_fights is True
    assert parser().parse_args(common).greedy_fights is False
