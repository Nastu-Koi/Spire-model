"""Difficulty, evaluation reports and the train/validation boundary."""

import json

import pytest

from model.cli import main, vocabulary_for
from model.config import ModelConfig, TrainConfig
from model.model import PolicyValue
from model.protocol import CHARACTERS
from model.representation import observation
from model.rollout import RolloutRunner, write_run
from model.testing import SyntheticEngine, demonstration
from model.trainer import evaluate_runs


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


def test_a_checkpoint_keeps_its_training_seeds_out_of_validation(tmp_path, capsys):
    first, second = tmp_path / "first", tmp_path / "second"
    for c in CHARACTERS:
        write_run(first / f"{c}.json", demonstration(c, "seen-seed", 4, 2))
        write_run(second / f"{c}.json", demonstration(c, "new-seed", 4, 2))
    common = ["--tiny", "--precision", "no"]
    assert main(["bootstrap", "--data", str(first), "--all-training-data", "--output", str(tmp_path / "a"),
                 *common]) == 0
    checkpoint = str(tmp_path / "a" / "current")
    assert "seen-seed" in json.loads((tmp_path / "a" / "current" / "manifest.json").read_text())["progress"]["training_seeds"]
    capsys.readouterr()
    # The data of this session does not overlap its validation set; the checkpoint's history does.
    assert main(["bootstrap", "--data", str(second), "--validation", str(first), "--checkpoint", checkpoint,
                 "--output", str(tmp_path / "b")]) == 1
    assert "checkpoint's training seeds" in capsys.readouterr().err
