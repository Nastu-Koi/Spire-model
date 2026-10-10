"""Difficulty, evaluation reports and the train/validation boundary."""

import json

import pytest

from model.cli import main, vocabulary_for
from model.config import ModelConfig, TrainConfig
from model.model import PolicyValue
from model.protocol import CHARACTERS
from model.representation import observation
from model.rollout import RolloutRunner, write_run
from model.seeds import validate_evaluation_seeds
from model.testing import SyntheticEngine, demonstration
from model.trainer import evaluate_runs
from spire_codex_data.sources import split_group


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
    runner = RolloutRunner(PolicyValue(config), vocabulary, ascension=4)
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
    assert main(["bootstrap", "--data", str(first), "--output", str(tmp_path / "a"),
                 *common]) == 0
    assert "seen-seed" in json.loads((tmp_path / "a" / "current" / "manifest.json").read_text())["progress"]["training_seeds"]
    capsys.readouterr()
    record = json.loads((tmp_path / "a" / "history.jsonl").read_text())
    assert "validation" not in record and "validation_runs" not in record
    assert (tmp_path / "a" / "epoch-000001" / "weights.pt").exists()
    progress = json.loads((tmp_path / "a" / "current" / "manifest.json").read_text())["progress"]
    with pytest.raises(ValueError, match="overlaps"):
        validate_evaluation_seeds({c: ["seen-seed" if i == 0 else f"unseen-{i}"]
                                   for i, c in enumerate(CHARACTERS)}, progress)
    archived = (tmp_path / "a" / "epoch-000001" / "weights.pt").read_bytes()
    assert main(["bootstrap", "--data", str(first), "--checkpoint", str(tmp_path / "a" / "current"),
                 "--output", str(tmp_path / "a"), "--epochs", "1"]) == 0
    assert (tmp_path / "a" / "epoch-000002" / "weights.pt").exists()
    assert (tmp_path / "a" / "epoch-000001" / "weights.pt").read_bytes() == archived
    capsys.readouterr()
    # A new initialization cannot replace the user's earlier comparison run.
    assert main(["bootstrap", "--data", str(first), "--output", str(tmp_path / "a"), *common]) == 1
    assert "new output directory" in capsys.readouterr().err



def test_an_evaluation_seed_is_refused_however_training_recorded_it():
    queue = {c: [f"SEED{i}"] for i, c in enumerate(CHARACTERS)}
    assert validate_evaluation_seeds(queue, {"training_seeds": ["OTHER", split_group("OTHER")]}) is queue
    # A rollout is kept under its game seed, an independent sample under the seed's split group.
    for known in ("SEED0", split_group("SEED0"), split_group("seedo")):
        with pytest.raises(ValueError, match="overlaps"):
            validate_evaluation_seeds(queue, {"training_seeds": [known]})
    with pytest.raises(ValueError, match="overlaps"):
        validate_evaluation_seeds(queue, {"random_seed_schedule": {"used_seeds": ["SEED4"]}})
