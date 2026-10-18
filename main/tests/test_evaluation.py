"""Difficulty, evaluation reports, paired comparisons and the held-out boundary."""

import gzip
import itertools
import json

import pytest

from main.cli import main
from main.commands import vocabulary_for
from main.network.config import ModelConfig, TrainConfig
from main.data.shards import HELDOUT_FRACTION, RunShards, extent, held_out, split_held_out
from main.data.index import HEADER, index_row
from main.play.evaluation import evaluate_runs, nll_report, sign_test, wilson_interval
from main.learn.loader import stream
from main.network.model import PolicyNetwork
from main.env.protocol import CHARACTERS
from main.encoding.representation import observation
from main.play.rollout import RolloutRunner, write_run
from main.env.seeds import validate_evaluation_seeds
from main.play.testing import SyntheticEngine, demonstration
from main.learn.trainer import Learner
from main.env.seeds import split_group


def test_unfinished_runs_are_not_wins():
    runs = [dict(character=c, status="complete" if i == 0 else "unresolved", victory=i == 0, ascension=10)
            for c in CHARACTERS for i in range(10)]
    report = evaluate_runs(runs)
    # Five wins in fifty attempts, whatever share of the finished runs they are.
    assert report["equal_character_win_rate"] == pytest.approx(0.1)
    assert report["equal_character_completed_win_rate"] == 1.0
    assert report["equal_character_completion_rate"] == pytest.approx(0.1)
    assert report["comparable"] is False and report["ascensions"] == [10]
    assert report["characters"]["Silent"]["errors_or_unresolved"] == 9
    # The five characters pooled, with the uncertainty of the pooled rates.
    pooled = report["pooled"]
    assert (pooled["attempts"], pooled["wins"], pooled["win_rate"]) == (50, 5, 0.1)
    assert pooled["wilson_95"] == wilson_interval(5, 50) == report["characters"]["Silent"]["wilson_95"] and 0.1 in [
        round(x, 3) for x in pooled["wilson_95"]] or pooled["wilson_95"][0] < 0.1 < pooled["wilson_95"][1]
    assert pooled["acts_passed"] == [0, 0, 5] and pooled["act_pass_wilson_95"][2] == pooled["wilson_95"]
    assert report["characters"]["Silent"]["act_pass_wilson_95"] == [wilson_interval(0, 10)] * 2 + [wilson_interval(1, 10)]
    finished = evaluate_runs([dict(r, status="complete") for r in runs])
    assert finished["comparable"] is True
    assert finished["equal_character_win_rate"] == finished["equal_character_completed_win_rate"]


def climbing_run(character, stops, victory=False):
    """A finished run whose recorded decisions stood at `stops`: (act, floor) in play order."""
    def frame(act, floor):
        return {"public": {"entities": [
            {"entity_type": "player", "act": act, "floor": floor},
            # The map names its rows from zero.
            *({"entity_type": "map_node", "floor": row} for row in range((17, 16, 15)[act - 1])),
        ]}}

    passed = sorted({act for act, _ in stops if act < stops[-1][0]} | ({3} if victory else set()))
    return dict(
        character=character, status="complete", victory=victory, ascension=0,
        macros=[{"steps": [{"frame": frame(*stop)}]} for stop in stops[:-1]],
        environment_actions=[{"frame": frame(*stops[-1])}],
        ledger={"bosses": [act for act in passed if not (victory and act == 3)],
                "components": {"combat": 0.0, "boss": float(sum(act for act in passed if act < 3)),
                               "victory": 5.0 if victory else 0.0}},
    )


def test_reports_how_far_finished_runs_climbed():
    runs = []
    for c in CHARACTERS:
        runs += [
            climbing_run(c, [(1, 1), (1, 9)]),
            # Floors count across acts: the whole first act, then five of the second.
            climbing_run(c, [(1, 17), (2, 0), (2, 5)]),
            climbing_run(c, [(1, 17), (2, 16), (3, 15)]),
            climbing_run(c, [(1, 17), (2, 16), (3, 15)], victory=True),
            dict(climbing_run(c, [(1, 1), (1, 3)]), status="unresolved"),
        ]
    report = evaluate_runs(runs)
    silent = report["characters"]["Silent"]
    # A defeat at the last boss and a win stand on the same floor; only the defeat is a death.
    assert silent["death_floors"] == [[9, 1], [22, 1], [48, 1]]
    assert silent["mean_floor"] == pytest.approx((9 + 22 + 48 + 48) / 4)
    # The win was confirmed by the run's end, not by a boss event: the act is passed all the same.
    assert silent["acts_passed"] == [3, 2, 1]
    assert silent["act_pass_rates"] == pytest.approx([3 / 5, 2 / 5, 1 / 5])
    assert silent["mean_score"] == pytest.approx((0 + 1 + 3 + 8) / 4)
    assert report["act_floors"] == [17, 16, 15]
    assert report["equal_character_act_pass_rates"] == pytest.approx([0.6, 0.4, 0.2])
    assert report["equal_character_act_pass_rates"][2] == pytest.approx(report["equal_character_win_rate"])
    assert report["equal_character_mean_floor"] == pytest.approx(31.75)
    assert report["equal_character_mean_score"] == pytest.approx(3.0)
    # Runs that recorded no decisions say nothing about floors; the wins are still counted.
    bare = evaluate_runs([dict(character=c, status="complete", victory=True) for c in CHARACTERS])
    assert bare["act_floors"] == [] and bare["equal_character_mean_floor"] is None
    assert bare["characters"]["Regent"]["death_floors"] == []
    assert bare["characters"]["Regent"]["acts_passed"] == [0, 0, 1]


def test_runs_are_read_once_so_they_can_come_one_at_a_time():
    runs = [climbing_run(c, stops, victory) for c in CHARACTERS for stops, victory in (
        ([(1, 1), (1, 9)], False), ([(1, 17), (2, 16), (3, 15)], True))]
    runs.append(dict(climbing_run("Silent", [(1, 1), (1, 3)]), status="unresolved"))
    read = []

    def one_at_a_time():
        for run in runs:
            read.append(run["character"])
            yield run

    assert evaluate_runs(one_at_a_time()) == evaluate_runs(runs)
    assert len(read) == len(runs)
    assert evaluate_runs(iter(()))["comparable"] is False


def test_rollouts_play_the_configured_ascension():
    with pytest.raises(ValueError):
        TrainConfig(ascension=11)
    config = ModelConfig.tiny()
    runs = [demonstration(c, "ascension", 4, 2) for c in CHARACTERS]
    vocabulary = vocabulary_for(runs, config)
    runner = RolloutRunner(PolicyNetwork(config), vocabulary, ascension=4)
    trace = runner.run(SyntheticEngine(4, 2), "Ironclad", "ascension-rollout")
    assert trace["ascension"] == 4 and trace["contract"]["fixed_ascension"] == 4
    assert evaluate_runs([trace])["ascensions"] == [4]


def test_ascension_on_the_player_is_model_input():
    frame = SyntheticEngine(4, 2).reset("Ironclad", "input")

    def digest(ascension):
        public = dict(frame["public"], entities=frame["public"]["entities"] + [
            dict(entity_type="player", ref="hero", character="IRONCLAD", ascension=ascension)])
        return observation(dict(frame, public=public)).digest

    assert digest(0) != digest(10)


def test_bc_records_all_training_seeds_for_manual_evaluation(tmp_path, capsys):
    first = tmp_path / "first"
    for c in CHARACTERS:
        write_run(first / f"{c}.json", demonstration(c, "seen-seed", 4, 2))
    common = ["--tiny", "--precision", "no", "--epochs", "1"]
    kept = tmp_path / "checkpoints" / "a"
    assert main(["bootstrap", "--data", str(first), "--output", str(tmp_path / "a"), "--checkpoints", str(kept),
                 *common]) == 0
    # The checkpoints are in their own directory; the output holds the records.
    assert not (tmp_path / "a" / "current").exists() and not any((tmp_path / "a").glob("epoch-*"))
    assert "seen-seed" in json.loads((kept / "current" / "manifest.json").read_text())["progress"]["training_seeds"]
    capsys.readouterr()
    record = json.loads((tmp_path / "a" / "history.jsonl").read_text())
    assert "validation" not in record and "validation_runs" not in record
    assert (kept / "epoch-000001" / "weights.pt").exists()
    progress = json.loads((kept / "current" / "manifest.json").read_text())["progress"]
    with pytest.raises(ValueError, match="overlaps"):
        validate_evaluation_seeds({c: ["seen-seed" if i == 0 else f"unseen-{i}"]
                                   for i, c in enumerate(CHARACTERS)}, progress)
    archived = (kept / "epoch-000001" / "weights.pt").read_bytes()
    assert main(["bootstrap", "--data", str(first), "--checkpoint", str(kept / "current"),
                 "--output", str(tmp_path / "a"), "--checkpoints", str(kept), "--epochs", "1"]) == 0
    assert (kept / "epoch-000002" / "weights.pt").exists()
    assert (kept / "epoch-000001" / "weights.pt").read_bytes() == archived
    capsys.readouterr()
    # A new initialization cannot replace the user's earlier comparison run.
    assert main(["bootstrap", "--data", str(first), "--output", str(tmp_path / "a"), "--checkpoints", str(kept),
                 *common]) == 1
    assert "new output directory" in capsys.readouterr().err
    # An output outside runs/ names its checkpoints directory; one of the old layout,
    # its checkpoints among its records, is told where they go.
    assert main(["bootstrap", "--data", str(first), "--output", str(tmp_path / "b"), *common]) == 1
    assert "--checkpoints" in capsys.readouterr().err
    (kept / "current").rename(tmp_path / "a" / "current")
    assert main(["bootstrap", "--data", str(first), "--checkpoint", str(tmp_path / "a" / "current"),
                 "--output", str(tmp_path / "a"), "--checkpoints", str(kept), "--epochs", "1"]) == 1
    assert "move them there" in capsys.readouterr().err



def test_an_evaluation_seed_is_refused_however_training_recorded_it():
    queue = {c: [f"SEED{i}"] for i, c in enumerate(CHARACTERS)}
    assert validate_evaluation_seeds(queue, {"training_seeds": ["OTHER", split_group("OTHER")]}) is queue
    # A rollout is kept under its game seed, an independent sample under the seed's split group.
    for known in ("SEED0", " seedo ", split_group("SEED0"), split_group("seedo")):
        with pytest.raises(ValueError, match="overlaps"):
            validate_evaluation_seeds(queue, {"training_seeds": [known]})

    queue[CHARACTERS[1]] = ["seedo"]
    with pytest.raises(ValueError, match="unique"):
        validate_evaluation_seeds(queue, {})


def test_a_sampled_seed_and_its_imported_labels_share_the_held_out_side(tmp_path):
    from main.demos.independent import import_samples
    from main.tests.test_independent import samples as exported_samples

    seeds = ["1553B38947E951FC", "C41AC5D6EA2F4572"]
    runs = [demonstration("Ironclad", seed, 4, 2) for seed in seeds]
    config = ModelConfig.tiny()
    runner = RolloutRunner(PolicyNetwork(config), vocabulary_for(runs, config))
    sampled = [runner.run(SyntheticEngine(4, 2), "Ironclad", seed) for seed in seeds]
    directory = tmp_path / "sampled"
    directory.mkdir()
    rollouts = sharded(directory, sampled)
    for source in ("historical_noncombat", "battle"):
        records = []
        for seed in seeds:
            rows = exported_samples(group=seed, seed=seed)
            for row in rows:
                row["metadata"].update(split_group=split_group(seed), anchor_kind=source)
            records.extend(rows)
        exported = tmp_path / f"{source}.jsonl"
        exported.write_text("".join(json.dumps(row) + "\n" for row in records))
        imported = tmp_path / source
        assert import_samples(exported, imported)["accepted_runs"] == 2
        labels = RunShards(imported)
        # Preserve the existing independent data's split; never hash its digest twice.
        assert [held_out(row) for row in labels.rows] == [False, True]
        for own_part, label_part in zip(rollouts.split(), labels.split()):
            assert {split_group(seed) for seed in own_part.seeds} == label_part.seeds
        assert [held_out(run) for run in sampled] == [held_out(row) for row in labels.rows]
        assert [held_out(run) for window in labels.windows() for run in window] == [False, True]
        # Legacy indexes had no explicit group; the oldest imports also omitted source.
        with gzip.open(imported / "index.jsonl.gz", "wt") as stream:
            stream.write(json.dumps(HEADER) + "\n")
            for row in labels.rows:
                stream.write(json.dumps({key: value for key, value in row.items()
                                         if key not in {"split_group", "source"}}) + "\n")
        assert [held_out(row) for row in RunShards(imported).rows] == [False, True]


def held_out_seed(prefix, start=0):
    """A seed named `<prefix>-<n>` that the fixed rule holds out."""
    return next(f"{prefix}-{n}" for n in itertools.count(start)
                if held_out({"seed": f"{prefix}-{n}", "run_id": ""}))


def test_the_held_out_side_is_a_function_of_the_seed_alone():
    seed = held_out_seed("held")
    # Every run and fragment of a seed falls on the same side, however it is stored.
    assert held_out({"seed": seed, "run_id": "a"}) and held_out({"seed": seed, "run_id": "b"})
    assert held_out({"run_id": seed})  # a run without a seed is keyed by its identity
    assert not held_out({"seed": "kept", "run_id": "a"}) and held_out({"seed": "kept", "run_id": "a"}, 1.0)
    # About one seed in fifty, whatever else the data set holds and in whatever order.
    seeds = [f"seed-{i}" for i in range(20000)]
    held = [s for s in seeds if held_out({"seed": s, "run_id": s})]
    assert 330 < len(held) < 470 and HELDOUT_FRACTION == 0.02
    assert held == [s for s in reversed(seeds) if held_out({"seed": s, "run_id": s})][::-1]
    training, part = split_held_out([{"seed": s, "run_id": s} for s in seeds])
    assert [r["seed"] for r in part] == held and len(training) + len(part) == len(seeds)


def sharded(directory, runs, per_shard=2):
    """`runs` written as gzip shards of `per_shard` runs with an index, as an import does."""
    rows = []
    for number, start in enumerate(range(0, len(runs), per_shard)):
        name = f"{number}.jsonl.gz"
        with gzip.open(directory / name, "wt") as out:
            for line, run in enumerate(runs[start:start + per_shard]):
                out.write(json.dumps(run) + "\n")
                rows.append(index_row(run, name, line))
    with gzip.open(directory / "index.jsonl.gz", "wt") as out:
        out.write(json.dumps(HEADER) + "\n")
        out.writelines(json.dumps(row) + "\n" for row in rows)
    return RunShards(directory)


def test_bootstrap_leaves_the_held_out_seeds_out_and_counts_them(tmp_path):
    config = ModelConfig.tiny()
    kept = [demonstration(c, "kept", 4, 2) for c in CHARACTERS]
    held = [demonstration(c, held_out_seed("held"), 4, 2) for c in CHARACTERS]
    runs = kept + held
    vocabulary = vocabulary_for(runs, config)
    training = TrainConfig(precision="no", loader_workers=0, logical_batch_size=4,
                           token_buckets=[32, 64], action_buckets=[8, 16])
    metrics = Learner(PolicyNetwork(config), vocabulary, training).bootstrap(runs, epochs=1)[0]
    assert metrics["held_out"] == {"fraction": HELDOUT_FRACTION, **extent(held)}
    assert metrics["training"] == extent(kept) and extent(held)["macros"] > 0
    # The loader delivered the training part and nothing else.
    assert metrics["samples"] == extent(kept)["macros"] == extent(runs)["macros"] - extent(held)["macros"]
    everything = Learner(PolicyNetwork(config), vocabulary, training).bootstrap(
        runs, epochs=1, include_held_out=True)[0]
    assert everything["samples"] == extent(runs)["macros"]
    assert everything["held_out"] == {"fraction": 0.0, "groups": 0, "macros": 0, "seeds": 0}
    # Sharded data: a held-out run shares its shard with trained runs and is still skipped.
    shards = sharded(tmp_path, [run for pair in zip(kept, held) for run in pair])
    part, rest = shards.split()
    assert len(part) == len(rest) == 5 and rest.seeds == {held[0]["seed"]} and not part.seeds & rest.seeds
    assert {row["shard"] for row in part.rows} == {row["shard"] for row in rest.rows}
    streamed = Learner(PolicyNetwork(config), vocabulary, training).bootstrap(shards, epochs=1, window_shards=2)[0]
    assert streamed["samples"] == streamed["training"]["macros"] == part.macros == shards.macros - rest.macros
    assert streamed["held_out"] == {"fraction": HELDOUT_FRACTION, **extent(rest)}
    delivered = {item.run_id for logical in stream(part, vocabulary, training, window_shards=2, shuffle=False)
                 for items, _ in logical for item in items}
    assert delivered == part.identities and not delivered & rest.identities


def test_held_out_nll_is_reported_by_character_and_phase(tmp_path, capsys):
    data = tmp_path / "data"
    for c in CHARACTERS:
        write_run(data / f"{c}-kept.json", demonstration(c, "kept", 4, 2))
        write_run(data / f"{c}-held.json", demonstration(c, held_out_seed("held"), 4, 2))
    assert main(["bootstrap", "--data", str(data), "--output", str(tmp_path / "bc"),
                 "--checkpoints", str(tmp_path / "checkpoints/bc"), "--tiny", "--precision", "no", "--epochs", "1"]) == 0
    capsys.readouterr()
    record = json.loads((tmp_path / "bc" / "history.jsonl").read_text())
    assert record["training_runs"] == 5 and record["metrics"]["held_out"]["groups"] == 5
    assert record["metrics"]["samples"] == record["metrics"]["training"]["macros"]
    progress = json.loads((tmp_path / "checkpoints/bc" / "current" / "manifest.json").read_text())["progress"]
    assert progress["held_out"] == record["metrics"]["held_out"]
    assert main(["heldout-nll", "--data", str(data), "--checkpoint", str(tmp_path / "checkpoints/bc" / "current")]) == 0
    report = json.loads(capsys.readouterr().out)
    held = [demonstration(c, held_out_seed("held"), 4, 2) for c in CHARACTERS]
    assert report["held_out"] == {"fraction": HELDOUT_FRACTION, **extent(held), "of": {
        "groups": 10, "macros": 2 * extent(held)["macros"], "seeds": 2}}
    assert report["all"]["macros"] == extent(held)["macros"] and set(report["characters"]) == set(CHARACTERS)
    for name, group in report["groups"].items():
        assert name.split("/")[0] in CHARACTERS and group["nll_per_branch"] == pytest.approx(group["nll"] / group["branches"])
        assert group["nll_per_macro"] == pytest.approx(group["nll"] / group["macros"])
    assert report["equal_group_nll_per_macro"] == pytest.approx(
        sum(g["nll_per_macro"] for g in report["groups"].values()) / len(report["groups"]))
    assert "a0" not in report and report["policy_version"] == 1


def test_nll_tables_follow_the_training_weighting():
    groups = {"Ironclad/combat": {"nll": 4.0, "branches": 8, "macros": 2, "nll_per_branch": 0.5},
              "Ironclad/map": {"nll": 1.0, "branches": 1, "macros": 1, "nll_per_branch": 1.0},
              "Silent/combat": {"nll": 6.0, "branches": 3, "macros": 3, "nll_per_branch": 2.0},
              "A0/Silent/combat": {"nll": 2.0, "branches": 1, "macros": 1, "nll_per_branch": 2.0}}
    report = nll_report(groups)
    assert report["all"] == {"nll": 11.0, "macros": 6, "branches": 12, "nll_per_macro": 11 / 6, "nll_per_branch": 11 / 12}
    assert report["characters"]["Ironclad"]["nll_per_macro"] == 5 / 3 and report["phases"]["combat"]["macros"] == 5
    # Every group weighs the same, as in the training loss.
    assert report["equal_group_nll_per_macro"] == pytest.approx((2.0 + 1.0 + 2.0) / 3)
    assert report["equal_group_nll_per_branch"] == pytest.approx((0.5 + 1.0 + 2.0) / 3)
    assert set(report["a0"]["groups"]) == {"Silent/combat"} and report["a0"]["all"]["macros"] == 1


def test_sign_test_counts_only_the_pairs_that_disagree():
    assert sign_test([1] * 8, [0] * 8) == {"pairs": 8, "first": 8, "second": 0, "only_first": 8,
                                           "only_second": 0, "p_value": 2 / 256}
    assert sign_test([0] * 8, [1] * 8)["p_value"] == sign_test([1] * 8, [0] * 8)["p_value"]
    assert sign_test([1, 0, 1], [1, 0, 1]) == {"pairs": 3, "first": 2, "second": 2, "only_first": 0,
                                               "only_second": 0, "p_value": 1.0}
    assert sign_test([1, 1, 0, 0, 1], [0, 1, 1, 0, 1])["p_value"] == 1.0  # one each way
    assert sign_test([1] * 5 + [0] * 300, [0] * 305)["p_value"] == pytest.approx(0.0625)


def test_compare_pairs_two_evaluations_seed_by_seed(tmp_path, capsys):
    seeds = {c: [f"{c}-{i}" for i in range(3)] for c in CHARACTERS}

    def evaluation(name, won, reached_act_3, unfinished=()):
        directory = tmp_path / name
        write_run(directory / "metadata" / "seeds.json", seeds)
        for c in CHARACTERS:
            for i in range(3):
                stops = [(1, 17), (2, 16), (3, 15)] if i in reached_act_3 else [(1, 17), (2, 5)]
                run = climbing_run(c, stops, victory=i in won)
                if (c, i) in unfinished:
                    run = dict(run, status="unresolved", victory=None)
                write_run(directory / f"{c}-{i}.json", run)
        return directory

    first = evaluation("first", won={0}, reached_act_3={0, 1})
    second = evaluation("second", won={0, 1}, reached_act_3={0, 1, 2}, unfinished={("Silent", 2)})
    assert main(["compare", str(first), str(second), "--head", "1"]) == 0
    report = json.loads(capsys.readouterr().out)
    whole, head = report["evaluations"][1]["report"], report["evaluations"][1]["head"]
    assert whole["pooled"]["wins"] == 10 and whole["comparable"] is False
    assert head["seeds_per_character"] == 1 and head["report"]["pooled"] == dict(
        attempts=5, wins=5, win_rate=1.0, wilson_95=wilson_interval(5, 5), acts_passed=[5, 5, 5],
        act_pass_rates=[1.0, 1.0, 1.0], act_pass_wilson_95=[wilson_interval(5, 5)] * 3)
    paired = report["paired"]
    assert paired["pairs"] == 15 and paired["unpaired"] == {"first": 0, "second": 0}
    assert paired["win"] == {"pairs": 15, "first": 5, "second": 10, "only_first": 0, "only_second": 5, "p_value": 0.0625}
    # The unfinished run passed nothing: act 1 is lost there, act 2 gained on the other four thirds.
    assert paired["act_1_passed"] == {"pairs": 15, "first": 15, "second": 14, "only_first": 1, "only_second": 0, "p_value": 1.0}
    assert paired["act_2_passed"] == {"pairs": 15, "first": 10, "second": 14, "only_first": 0, "only_second": 4, "p_value": 0.125}
    assert paired["head"]["seeds_per_character"] == 1 and paired["head"]["win"]["pairs"] == 5
    assert paired["head"]["win"]["p_value"] == 1.0 and paired["head"]["act_2_passed"]["only_second"] == 0
    # One evaluation reports alone; a missing run is an error, not a silent gap.
    assert main(["compare", str(first)]) == 0
    alone = json.loads(capsys.readouterr().out)
    assert "paired" not in alone and alone["evaluations"][0]["runs"] == 15
    (first / "Defect-1.json").unlink()
    assert main(["compare", str(first)]) == 1
    assert "missing" in capsys.readouterr().err
