"""Production CLI routes search and resume to the MCTS workflow."""

import json
import sys

import pytest

from combat_solver_cli import __main__ as cli
from combat_solver_cli import resume_bootstrap


def test_search_cli_calls_only_mcts(tmp_path, monkeypatch, capsys):
    seen = {}

    def fake_search(**options):
        seen.update(options)
        return {"status": "verified_victory", "algorithm": "mcts"}

    monkeypatch.setattr(cli, "search", fake_search)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "combat_solver_cli",
            "search",
            "--seed",
            "abc",
            "--output",
            str(tmp_path / "run"),
        ],
    )
    with pytest.raises(SystemExit) as stopped:
        cli.main()
    assert stopped.value.code == 0
    assert seen["character"] == "Ironclad"
    assert seen["ascension"] == 0
    assert seen["rollout_decisions"] == 256
    assert seen["search_lanes"] == 1
    assert "algorithm" not in seen and "weight" not in seen
    assert json.loads(capsys.readouterr().out)["algorithm"] == "mcts"


def test_search_help_has_no_algorithm_switch(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["combat_solver_cli", "search", "--help"])
    with pytest.raises(SystemExit) as stopped:
        cli.main()
    assert stopped.value.code == 0
    help_text = capsys.readouterr().out
    assert "--algorithm" not in help_text
    assert "--weight" not in help_text


def test_resume_batch_requires_mcts_manifest_and_tree(tmp_path):
    previous = tmp_path / "previous"
    previous.mkdir()
    (previous / "manifest.json").write_text(
        json.dumps({"jobs": [{"character": "Ironclad", "seed": "abc"}], "options": {}})
    )
    with pytest.raises(ValueError, match="incompatible with MCTS"):
        resume_bootstrap.resumable_jobs(previous)
    manifest = json.loads((previous / "manifest.json").read_text())
    manifest["mode"] = "mcts_batch_v1"
    (previous / "manifest.json").write_text(json.dumps(manifest))
    from model.protocol import fingerprint

    job = previous / "jobs" / ("Ironclad-" + fingerprint("abc")[:16])
    job.mkdir(parents=True)
    (job / "tree.json").write_text('{"schema":"mcts-tree-v1"}')
    jobs, _ = resume_bootstrap.resumable_jobs(previous)
    assert jobs[0]["resume"] == str((job / "tree.json").resolve())


def test_resume_preserves_nondefault_mcts_policy(tmp_path, monkeypatch, capsys):
    from model.protocol import fingerprint

    previous = tmp_path / "previous"
    previous.mkdir()
    prior = {
        "ascension": 0,
        "search_lanes": 1,
        "budget_ms": 1200,
        "boss_budget_ms": 8000,
        "max_expansions": 17,
        "max_seconds": 31,
        "max_steps": 250,
        "rollout_decisions": 8,
        "reuse_turn_plan": False,
        "local_repair": False,
        "risk_aware_rollout": False,
        "exploration": 0.7,
        "rollout_epsilon": 0.2,
    }
    (previous / "manifest.json").write_text(
        json.dumps(
            {
                "mode": "mcts_batch_v1",
                "config": "saved.json",
                "options": prior,
                "jobs": [{"character": "Ironclad", "seed": "abc"}],
            }
        )
    )
    job = previous / "jobs" / ("Ironclad-" + fingerprint("abc")[:16])
    job.mkdir(parents=True)
    (job / "tree.json").write_text('{"schema":"mcts-tree-v1"}')
    seen = {}

    def fake_generate(**options):
        seen.update(options)
        return {"verified_victories": 1}

    monkeypatch.setattr(resume_bootstrap, "generate", fake_generate)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "resume_bootstrap",
            "--previous",
            str(previous),
            "--output",
            str(tmp_path / "resumed"),
            "--max-expansions",
            "42",
        ],
    )
    with pytest.raises(SystemExit) as stopped:
        resume_bootstrap.main()
    assert stopped.value.code == 0
    assert seen["max_expansions"] == 42
    for key in (
        "boss_budget_ms",
        "local_repair",
        "risk_aware_rollout",
        "exploration",
        "rollout_epsilon",
        "rollout_decisions",
    ):
        assert seen[key] == prior[key]
    assert seen["max_seconds"] == 31
    assert seen["jobs"][0]["resume"].endswith("tree.json")
    assert json.loads(capsys.readouterr().out)["verified_victories"] == 1


def test_smoke_defaults_to_ironclad_a0(tmp_path, monkeypatch):
    seen = []

    def fake_first_combat(
        config, character, seed, budget_ms, reuse_turn_plan, ascension
    ):
        seen.append((character, ascension))
        return {"character": character, "ascension": ascension}

    monkeypatch.setattr(cli, "first_combat", fake_first_combat)
    monkeypatch.setattr(
        sys,
        "argv",
        ["combat_solver_cli", "smoke", "--output", str(tmp_path / "smoke.json")],
    )
    cli.main()
    assert seen == [("Ironclad", 0)]
