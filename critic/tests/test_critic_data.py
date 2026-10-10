"""Rows are the first choice of each out-of-combat macro-action, labelled by the run's
and the act's outcome, and split by seed."""

import json

import pytest

from critic.data import coverage, decisions, frame_of, load, split
from critic.metrics import auc, brier, calibration, explained_variance, reliability, summary
from critic.tests.conftest import STANDS, synthetic_run


def test_rows_are_the_first_choice_of_each_out_of_combat_macro_action():
    run = synthetic_run("Silent", "rows", victory=False, bosses={1, 2}, stands=STANDS, forced_first=True)
    rows = list(decisions(run))
    assert [(r.phase, r.act, r.floor) for r in rows] == list(STANDS)
    # The forced step before the first choice is not the decision; combat is left out.
    assert rows[0].decision_id == f"{run['run_id']}:0:1" and rows[1].decision_id == f"{run['run_id']}:1:0"
    assert not any(r.win for r in rows) and [r.act_passed for r in rows] == [True, True, True, False]
    assert [r.seed for r in rows] == ["rows"] * 4 and rows[0].character == "Silent"
    assert (rows[0].policy_version, rows[0].policy_digest) == (3, "synthetic")
    assert rows[3].state["gold"] == 120 and rows[3].state["deck"] == 0
    frame = frame_of(rows[2])
    assert frame["public"]["phase"] == "card_select" and len(frame["legal"]["candidates"]) == 4
    won = list(decisions(synthetic_run("Silent", "won", victory=True, bosses={1, 2, 3}, stands=STANDS)))
    assert all(r.win and r.act_passed for r in won)


def test_split_is_by_seed_and_needs_two_seeds(runs):
    rows = [r for run in runs for r in decisions(run)]
    train, test = split(rows, 0.5)
    assert len(train) == len(test) == 12
    assert not {r.seed for r in train} & {r.seed for r in test}
    assert len({r.seed for r in train}) == 3 and len({r.seed for r in test}) == 3
    with pytest.raises(ValueError, match="two seeds"):
        split(list(decisions(runs[0])), 0.5)
    report = coverage(rows)
    assert report["all"]["decisions"] == 24 and set(report["act"]) == {"1", "2", "3"}
    assert report["act"]["3"]["act_pass_rate"] == report["all"]["win_rate"]


def test_loading_skips_other_json_files_and_unfinished_runs(tmp_path, runs):
    directory = tmp_path / "evaluation"
    directory.mkdir()
    (directory / "report.json").write_text(json.dumps(dict(equal_character_win_rate=0.1)))
    (directory / "a.json").write_text(json.dumps(runs[0]))
    (directory / "b.json").write_text(json.dumps(dict(runs[1], status="unresolved")))
    (directory / "c.jsonl").write_text(json.dumps(runs[2]) + "\n" + json.dumps(runs[3]) + "\n")
    rows, summary_ = load([directory])
    assert (summary_["runs"], summary_["skipped_files"], summary_["incomplete_runs"]) == (3, 1, 1)
    assert summary_["decisions"] == len(rows) == 12 and summary_["seeds"] == 3
    assert summary_["policies"] == [dict(policy_version=3, policy_digest="synthetic", decisions=12)]
    with pytest.raises(ValueError, match="No out-of-combat decisions"):
        load([tmp_path / "evaluation" / "report.json"])


def test_metrics_of_the_constant_and_the_perfect_predictor():
    outcomes = [True, False, False, True, False]
    rate = 0.4
    assert explained_variance([rate] * 5, outcomes) == pytest.approx(0.0)
    assert explained_variance([1.0 if y else 0.0 for y in outcomes], outcomes) == pytest.approx(1.0)
    assert brier([rate] * 5, outcomes) == pytest.approx(rate * (1 - rate))
    assert auc([0.9, 0.1, 0.2, 0.8, 0.3], outcomes) == 1.0 and auc([0.5] * 5, outcomes) == 0.5
    assert explained_variance([0.5] * 3, [True, True, True]) is None
    report = summary([0.9, 0.1, 0.2, 0.8, 0.3], outcomes)
    assert report["n"] == 5 and report["rate"] == 0.4 and report["auc"] == 1.0 and report["explained_variance"] > 0.5
    table = calibration([1, 1, 2, 2, 2], [0.9, 0.1, 0.2, 0.8, 0.3], outcomes)
    assert [row["key"] for row in table] == [1, 2] and table[0]["actual"] == 0.5 and table[0]["gap"] == 0.0
    bins = reliability([0.05, 0.95, 0.96], [False, True, True], bins=10)
    assert [row["bin"] for row in bins] == [[0.0, 0.1], [0.9, 1.0]] and bins[1]["n"] == 2
