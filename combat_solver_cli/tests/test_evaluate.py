import copy
import json
from unittest.mock import patch

import pytest

from combat_solver_cli import evaluate


def manifest():
    return {
        "schema": "search-evaluation-v1",
        "character": "Ironclad",
        "ascension": 0,
        "seeds": {"dev": ["a", "b"], "holdout": ["never-use"]},
        "profiles": [
            {"name": "base", "options": {}},
            {"name": "budget", "options": {"budget_ms": 2000}},
        ],
        "repetitions": 1,
        "budget": {"max_seconds_per_run": 1, "max_expansions": 10, "max_steps": 100},
    }


def test_frozen_cases_include_errors_and_incomplete_cost_without_holdout(tmp_path):
    now = [0.0]
    cases = []

    def fake(config, character, seed, output, **options):
        cases.append((seed, options.get("budget_ms")))
        now[0] += 2
        if len(cases) == 2:
            raise RuntimeError("engine fault")
        return {"status": "budget_exhausted"}

    with (
        patch.object(evaluate, "source_hashes", return_value={"code": "fixed"}),
        patch.object(evaluate.time, "monotonic", side_effect=lambda: now[0]),
    ):
        report = evaluate.evaluate(manifest(), "dev", tmp_path / "out", search_fn=fake)
    assert cases == [("a", None), ("a", 2000), ("b", 2000), ("b", None)]
    assert report["wall_seconds"] == 8
    assert report["wall_seconds_per_verified_victory"] is None
    assert report["counts"] == {"budget_exhausted": 3, "infrastructure_error": 1}
    assert report["independent_seeds"] == 2
    assert all(row["budget_overrun_seconds"] == 1 for row in report["runs"])
    assert json.loads((tmp_path / "out/report.json").read_text()) == report


def test_victory_claim_without_independent_artifact_is_rejected(tmp_path):
    with patch.object(evaluate, "source_hashes", return_value={}):
        report = evaluate.evaluate(
            manifest(),
            "dev",
            tmp_path / "out",
            search_fn=lambda *a, **k: {"status": "verified_victory"},
        )
    assert report["verified_victories"] == 0
    assert report["counts"] == {"replay_failed": 4}


def test_source_changes_are_reported(tmp_path):
    with patch.object(
        evaluate, "source_hashes", side_effect=[{"code": "before"}, {"code": "after"}]
    ):
        report = evaluate.evaluate(
            manifest(),
            "dev",
            tmp_path / "out",
            search_fn=lambda *a, **k: {"status": "budget_exhausted"},
        )
    assert report["source_hash_mismatch"]


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d["seeds"].update(holdout=["a"]),
        lambda d: d["profiles"][0]["options"].update(depth=3),
        lambda d: d["profiles"][0]["options"].update(budget_ms=True),
        lambda d: d["budget"].update(max_seconds_per_run=float("nan")),
    ],
)
def test_invalid_frozen_config_rejected_before_output(tmp_path, change):
    data = copy.deepcopy(manifest())
    change(data)
    with pytest.raises(ValueError):
        evaluate.evaluate(data, "dev", tmp_path / "out")
    assert not (tmp_path / "out").exists()
