import json
import random

import pytest

from model import cli
from model.config import TrainConfig
from model.data import load_runs
from model.protocol import CHARACTERS
from model.rollout import collect_round
from model.seeds import RandomSeedSchedule
from model.testing import SyntheticEngine
from model.trainer import Learner


def test_160_rounds_have_equal_five_character_queues_and_no_reused_seeds():
    schedule = RandomSeedSchedule()
    seen = set()
    for _ in range(160):
        queue = schedule.next()
        assert set(queue) == set(CHARACTERS)
        assert {len(seeds) for seeds in queue.values()} == {4}
        seeds = [seed for group in queue.values() for seed in group]
        assert len(set(seeds)) == 20
        assert not seen.intersection(seeds)
        assert all(len(seed) == 16 and int(seed, 16) >= 0 for seed in seeds)
        seen.update(seeds)
    assert len(seen) == 3200
    assert schedule.state_dict()["next_round"] == 160


def test_schedule_roundtrip_is_independent_of_global_training_rng():
    original = RandomSeedSchedule(2)
    original.next()
    restored = RandomSeedSchedule(state=json.loads(json.dumps(original.state_dict())))
    random.seed(321)
    for _ in range(100):
        random.random()
    assert original.next() == restored.next()
    assert restored.runs_per_character == 2
    assert original.state_dict() == restored.state_dict()
    assert RandomSeedSchedule().root != RandomSeedSchedule().root


def test_cli_training_defaults_and_seed_source_exclusivity():
    args = cli.parser().parse_args(["train", "--checkpoint", "init", "--output", "run"])
    assert args.rounds == 160 and args.seeds is None
    with pytest.raises(SystemExit):
        cli.parser().parse_args(["train", "--checkpoint", "init", "--output", "run",
                                 "--seeds", "fixed.json", "--random-seeds"])
    with pytest.raises(SystemExit):
        cli.parser().parse_args(["evaluate", "--checkpoint", "init", "--output", "run"])


def test_seed_assignment_is_saved_before_even_a_failed_engine_start(tmp_path):
    seeds = RandomSeedSchedule(1).next()

    def unavailable():
        assert json.loads((tmp_path / "metadata/seeds.json").read_text()) == seeds
        raise OSError("engine unavailable")

    with pytest.raises(OSError, match="unavailable"):
        collect_round(unavailable, None, seeds, tmp_path)
    assert not list(tmp_path.glob("*.json"))
    with pytest.raises(FileExistsError, match="assignment"):
        collect_round(unavailable, None, seeds, tmp_path)


@pytest.mark.parametrize("warmup", [False, True])
def test_cli_random_rounds_checkpoint_resume_and_fixed_replay(setup, tmp_path, monkeypatch, warmup):
    model, vocabulary = setup
    learner = Learner(model, vocabulary, TrainConfig(precision="no", ppo_epochs=1))
    cli._save(tmp_path / "init", learner)
    monkeypatch.setattr(cli, "CliEngine", lambda **_: SyntheticEngine(5, 3))
    whole = tmp_path / "whole"
    options = ["--threads", "2", "train", "--checkpoint", str(tmp_path / "init"),
               "--random-seeds", "--runs-per-character", "1", "--rounds", "2", "--output", str(whole)]
    assert cli.main(options + (["--value-warmup"] if warmup else [])) == 0
    first = json.loads((whole / "round-0/metadata/seeds.json").read_text())
    second = json.loads((whole / "round-1/metadata/seeds.json").read_text())
    assert first != second
    assert len(load_runs(whole / "round-0")) == 5

    resumed = tmp_path / "resumed"
    assert cli.main(["--threads", "2", "train", "--checkpoint", str(whole / "current"),
                     "--rounds", "1", "--output", str(resumed)]) == 0
    assert json.loads((resumed / "round-2/metadata/seeds.json").read_text()) not in [first, second]
    progress = json.loads((resumed / "current/manifest.json").read_text())["progress"]
    assert progress["random_seed_schedule"]["next_round"] == 3
    assert progress["random_seed_schedule"]["runs_per_character"] == 1
    assert progress["completed_ppo_rounds"] == (2 if warmup else 3)

    fixed = tmp_path / "fixed"
    assert cli.main(["--threads", "2", "train", "--checkpoint", str(tmp_path / "init"),
                     "--seeds", str(whole / "round-0/metadata/seeds.json"), "--rounds", "1",
                     "--output", str(fixed)]) == 0
    assert json.loads((fixed / "round-0/metadata/seeds.json").read_text()) == first
    assert json.loads((fixed / "current/manifest.json").read_text())["progress"]["seed_mode"] == "fixed"
