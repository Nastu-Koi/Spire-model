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
