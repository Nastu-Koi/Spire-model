"""Bounded MCTS batch and target workflows with fake native search."""

import json
import time
from pathlib import Path

import pytest

from combat_solver_cli import batch, mcts, target
from combat_solver_cli.client import InfrastructureError


def _accepted(character, seed):
    return {
        "character": character,
        "seed": seed,
        "provenance": {"verified_outcome": "A0_final_boss_victory"},
    }


def test_batch_uses_mcts_and_counts_failed_wall_time(tmp_path, monkeypatch, capsys):
    assert batch.search is mcts.search
    monkeypatch.setattr(batch, "load_runs", lambda path: [_accepted("Ironclad", "win")])
    seen = []

    def fake_search(config, character, seed, directory, **options):
        seen.append((seed, options))
        time.sleep(0.01)
        Path(directory).mkdir(parents=True)
        if seed == "win":
            (Path(directory) / "accepted.jsonl").write_text("verified\n")
            return {"status": "verified_victory", "algorithm": "mcts"}
        return {"status": "budget_exhausted", "algorithm": "mcts"}

    result = batch.generate(
        "config",
        [{"character": "Ironclad", "seed": seed} for seed in ("miss", "win")],
        tmp_path / "batch",
        ascension=0,
        search_fn=fake_search,
        progress=True,
        progress_interval=0.001,
    )
    assert [seed for seed, _ in seen] == ["miss", "win"]
    assert result["verified_victories"] == 1
    assert result["outcome_counts"]["budget_exhausted"] == 1
    assert result["unfinished_attempt_seconds"] > 0
    assert result["wall_seconds_per_verified_victory"] == result["wall_seconds"]
    assert "waiting" in capsys.readouterr().err
    assert (
        json.loads((tmp_path / "batch/manifest.json").read_text())["mode"]
        == "mcts_batch_v1"
    )


def test_target_stops_at_attempt_bound_then_resumes_tree(tmp_path, monkeypatch):
    assert target.search is mcts.search
    monkeypatch.setattr(
        target, "load_runs", lambda path: [_accepted("Ironclad", "seed")]
    )
    monkeypatch.setattr(target, "_new_seed", lambda used: "seed")
    seen = []

    def fake_search(config, character, seed, directory, *, resume, **options):
        seen.append(resume)
        Path(directory).mkdir(parents=True)
        (Path(directory) / "tree.json").write_text('{"schema":"mcts-tree-v1"}')
        if resume is None:
            return {"status": "budget_exhausted", "seconds": 0.01}
        (Path(directory) / "accepted.jsonl").write_text("verified\n")
        return {"status": "verified_victory", "seconds": 0.01}

    output = tmp_path / "target"
    first = target.generate_target(
        "config",
        output,
        1,
        search_fn=fake_search,
        max_attempts=1,
        max_total_seconds=1,
        progress=False,
        ascension=0,
    )
    assert first["status"] == "budget_exhausted"
    assert first["verified_trajectories"] == 0
    second = target.resume_target(output, search_fn=fake_search)
    assert second["status"] == "complete"
    assert second["attempts"] == 2
    assert second["unfinished_attempt_seconds"] > 0
    assert seen[0] is None
    assert seen[1].name == "tree.json"
    assert (output / "accepted.jsonl").read_text() == "verified\n"


def test_target_rejects_legacy_manifest(tmp_path):
    output = tmp_path / "legacy"
    output.mkdir()
    (output / "manifest.json").write_text('{"mode":"target_verified_trajectories"}')
    with pytest.raises(ValueError, match="compatible MCTS"):
        target.resume_target(output)


def test_target_records_infrastructure_failure_and_stops(tmp_path, monkeypatch):
    monkeypatch.setattr(target, "_new_seed", lambda used: "seed")
    calls = []

    def failing_search(config, character, seed, directory, **options):
        calls.append(directory)
        Path(directory).mkdir(parents=True)
        (Path(directory) / "tree.json").write_text('{"schema":"mcts-tree-v1"}')
        raise InfrastructureError("native unavailable")

    output = tmp_path / "target"
    result = target.generate_target(
        "config",
        output,
        1,
        search_fn=failing_search,
        max_attempts=3,
        max_total_seconds=1,
        progress=False,
    )
    assert len(calls) == 1
    assert result["status"] == "blocked"
    assert result["attempts"] == 1
    assert result["outcome_counts"]["native_error"] == 1
    assert result["unfinished_attempt_seconds"] > 0
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["games"][0]["segments"][0]["result"]["status"] == "job_error"
    assert (Path(calls[0]) / "tree.json").is_file()


def test_target_keeps_keyboard_interrupt_visible(tmp_path, monkeypatch):
    monkeypatch.setattr(target, "_new_seed", lambda used: "seed")

    def interrupted_search(config, character, seed, directory, **options):
        Path(directory).mkdir(parents=True)
        (Path(directory) / "tree.json").write_text('{"schema":"mcts-tree-v1"}')
        raise KeyboardInterrupt()

    output = tmp_path / "target"
    with pytest.raises(KeyboardInterrupt):
        target.generate_target(
            "config",
            output,
            1,
            search_fn=interrupted_search,
            max_attempts=1,
            max_total_seconds=1,
            progress=False,
        )
    assert next(output.glob("games/*/segments/0000/tree.json")).is_file()


@pytest.mark.parametrize("search_status", ["stopped", "interrupted"])
def test_target_does_not_retry_stopped_search(tmp_path, monkeypatch, search_status):
    monkeypatch.setattr(target, "_new_seed", lambda used: "seed")
    monkeypatch.setattr(
        target, "load_runs", lambda path: [_accepted("Ironclad", "seed")]
    )
    resumes = []

    def fake_search(config, character, seed, directory, *, resume, **options):
        resumes.append(resume)
        Path(directory).mkdir(parents=True)
        (Path(directory) / "tree.json").write_text('{"schema":"mcts-tree-v1"}')
        if resume is None:
            return {"status": search_status}
        (Path(directory) / "accepted.jsonl").write_text("verified\n")
        return {"status": "verified_victory"}

    output = tmp_path / "target"
    first = target.generate_target(
        "config",
        output,
        1,
        search_fn=fake_search,
        max_attempts=3,
        max_total_seconds=1,
        progress=False,
    )
    assert first["status"] == "stopped"
    assert first["attempts"] == 1
    assert len(resumes) == 1
    assert next(output.glob("games/*/segments/0000/tree.json")).is_file()
    second = target.resume_target(output, search_fn=fake_search)
    assert second["status"] == "complete"
    assert len(resumes) == 2
    assert resumes[1].name == "tree.json"


def test_target_marks_deadline_stop_as_budget_exhausted(tmp_path, monkeypatch):
    monkeypatch.setattr(target, "_new_seed", lambda used: "seed")
    calls = []

    def fake_search(config, character, seed, directory, **options):
        calls.append(directory)
        Path(directory).mkdir(parents=True)
        (Path(directory) / "tree.json").write_text('{"schema":"mcts-tree-v1"}')
        time.sleep(0.15)
        return {"status": "stopped"}

    result = target.generate_target(
        "config",
        tmp_path / "target",
        1,
        search_fn=fake_search,
        max_attempts=3,
        max_total_seconds=0.1,
        progress=False,
    )
    assert result["status"] == "budget_exhausted"
    assert result["attempts"] == len(calls) == 1
