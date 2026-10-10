"""The critic reads sampled rounds and buffers of them, scores their decisions into
advantages, and is held to a gate before those weigh anything."""

import json
from copy import deepcopy

import pytest

from critic.advantage import identity, score_rounds
from critic.data import decisions, expand, frame_of, load
from critic.gate import Gate, check
from critic.model import load as load_critic
from critic.tests.conftest import STANDS, synthetic_run
from critic.train import constant_loss, main
from model import advantage
from model.advantage import FILE, gae
from model.buffer import buffer_rounds, round_summary
from model.tests import journeys


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """A buffer of three rounds of two iterations, a policy checkpoint and a critic profile."""
    root = tmp_path_factory.mktemp("rounds")
    policy = journeys.learner()
    for name, iteration in (("round-0", 0), ("round-1", 1), ("round-2", 1)):
        journeys.sampled_round(policy, root / "buffer" / name, iteration, runs=3, name=name)
    # What a round cut off leaves is not a round of the buffer.
    (root / "buffer" / "round-3" / "accepted").mkdir(parents=True)
    return dict(root=root, buffer=root / "buffer", checkpoint=journeys.checkpoint(root / "policy"),
                profile=journeys.profile(root / "profile.json"))


def test_a_buffer_is_the_finished_rounds_of_its_last_iterations(world):
    buffer = world["buffer"]
    assert [path.name for path in buffer_rounds(buffer)] == ["round-0", "round-1", "round-2"]
    assert [path.name for path in buffer_rounds(buffer, 1)] == ["round-1", "round-2"]
    assert buffer_rounds(buffer, 5) == buffer_rounds(buffer)
    assert round_summary(buffer / "round-3") is None and round_summary(buffer / "round-0")["iteration"] == 0
    with pytest.raises(ValueError):
        buffer_rounds(buffer, 0)
    # A buffer directory stands for its rounds; a round, a file or a directory of runs for itself.
    assert expand([buffer], 1) == buffer_rounds(buffer, 1) and expand([buffer / "round-0"]) == [buffer / "round-0"]
    assert expand([world["root"]]) == [world["root"]]


def test_rows_of_a_sampled_round_are_its_decisions_outside_fights(world):
    rows, summary = load([world["buffer"] / "round-0"])
    assert (summary["rounds"], summary["runs"], summary["decisions"]) == (1, 15, 45) and len(rows) == 45
    # Each run's decisions in the order it made them, named by their place among its macro-actions.
    by_run = {}
    for row in rows:
        by_run.setdefault(row.run_id, []).append(row)
    assert len(by_run) == 15 and all([row.macro for row in run] == [1, 3, 5] for run in by_run.values())
    assert all([row.floor for row in run] == [1, 2, 3] and {row.phase for row in run} == {"combat_play"}
               for run in by_run.values())
    assert {row.iteration for row in rows} == {0} and len({row.win for row in rows}) == 2
    assert all(len({row.win for row in run}) == 1 for run in by_run.values())
    # The buffer's rounds together, and only those of its last iterations.
    assert load([world["buffer"]])[1]["decisions"] == 135
    rows, summary = load([world["buffer"]], iterations=1)
    assert (summary["rounds"], summary["decisions"]) == (2, 90) and {row.iteration for row in rows} == {1}
    assert summary["sources"] == [str(world["buffer"] / name) for name in ("round-1", "round-2")]


def test_a_run_that_names_its_fights_keeps_selections_inside_them_out_of_its_rows():
    run = synthetic_run("Silent", "fights", victory=False, bosses={1}, stands=STANDS)
    # Written before fights were named: only the combat phase is a fight.
    assert [row.macro for row in decisions(run)] == [0, 1, 2, 3]
    named = deepcopy(run)
    for index, macro in enumerate(named["macros"]):
        macro["fight"] = 1 if index in (2, 4) else None
    # The selection made inside the fight (`card_select`) is the fight's, not a decision outside it.
    assert [(row.macro, row.phase) for row in decisions(named)] == [(0, "map"), (1, "rewards"), (3, "shop")]


def test_a_critic_trained_on_rounds_scores_them_into_advantages(world, tmp_path):
    saved = tmp_path / "critic"
    report = main([str(world["buffer"]), "--buffer-iterations", "1", "--checkpoint", str(world["checkpoint"]),
                   "--config", str(world["profile"]), "--device", "cpu", "--save", str(saved)])
    assert report["data"]["rounds"] == 2 and report["data"]["decisions"] == 90
    assert report["train"]["constant_loss"] > 0 and "baselines" in report
    manifest = json.loads((saved / "manifest.json").read_text())
    assert manifest["report"]["train"] == report["train"] and manifest["version"] == 1

    rounds = buffer_rounds(world["buffer"])
    shards = [sorted(path.read_bytes() for path in (directory / "accepted").iterdir()) for directory in rounds]
    headers = score_rounds(rounds, saved, lam=0.9)
    scorer = load_critic(saved)
    wanted = identity(saved, scorer, 0.9)
    assert wanted["critic"]["version"] == 1 and len(wanted["critic"]["weights"]) == 16
    assert [(header["kind"], header["lam"], header["critic"], header["decisions"]) for header in headers] == [
        ("critic", 0.9, wanted["critic"], 45)] * 3
    assert [sorted(path.read_bytes() for path in (directory / "accepted").iterdir()) for directory in rounds] == shards
    # Each advantage is the generalized advantage of the critic's own values along its run.
    table = advantage.read(rounds)
    rows, _ = load([rounds[0]])
    run = [row for row in rows if row.run_id == rows[0].run_id]
    p_win, _ = scorer.predict([frame_of(row) for row in run])
    expected = gae([2 * p - 1 for p in p_win], 1.0 if run[0].win else -1.0, 0.9)
    assert [table[row.run_id, row.macro] for row in run] == pytest.approx(expected, abs=1e-4)
    assert len(table) == 135
    # Rounds this critic scored at this λ are left as they are; another λ scores them again.
    written = [(directory / FILE).stat().st_mtime_ns for directory in rounds]
    assert score_rounds(rounds, saved, lam=0.9) == headers
    assert [(directory / FILE).stat().st_mtime_ns for directory in rounds] == written
    again = score_rounds(rounds[:1], saved, lam=0.5)
    assert again[0]["lam"] == 0.5 and (rounds[0] / FILE).stat().st_mtime_ns != written[0]


def passing():
    """A training report every floor of the default gate holds for."""
    def head(variance, rate):
        return dict(n=4000, rate=rate, predicted=rate + 0.01, brier=0.12, explained_variance=variance, auc=0.8,
                    log_loss=0.4)

    def table(keys):
        return [dict(key=key, n=1000, actual=0.2, predicted=0.22, gap=0.02) for key in keys]

    bins = [dict(bin=[0.0, 0.1], n=3900, actual=0.2, predicted=0.22, gap=0.02),
            dict(bin=[0.9, 1.0], n=100, actual=0.5, predicted=0.95, gap=0.45)]
    return dict(
        data=dict(split=dict(test_seeds=220)), train=dict(constant_loss=1.20),
        history=[dict(train_loss=1.10)],
        baselines=dict(constant=dict(win=dict(brier=0.16), act=dict(brier=0.24)),
                       logistic_act_floor=dict(win=dict(explained_variance=0.06), act=dict(explained_variance=0.05))),
        critic=dict(win=head(0.2, 0.2), act=head(0.15, 0.6), calibration=dict(
            by_act=dict(win=table([1, 2, 3]), act=table([1, 2, 3])),
            by_character=dict(win=table("abcde"), act=table("abcde")),
            by_probability=dict(win=deepcopy(bins), act=deepcopy(bins)))))


def failing(change):
    report = passing()
    change(report)
    return check(report)["failed"]


def test_the_gate_holds_a_critic_to_its_heldout_report():
    result = check(passing())
    assert result["passed"] and result["failed"] == [] and len(result["checks"]) == 14
    assert result["gate"] == Gate().__dict__
    # The far bin is a fortieth of the decisions: too few to be held to its gap.
    assert result["checks"]["win_bin_gap"]["value"] == 0.02
    assert result["checks"]["memorization"]["value"] == dict(train_loss_fall=0.1, heldout_brier_gain=0.16)

    assert failing(lambda r: r["data"]["split"].update(test_seeds=74)) == ["heldout_seeds"]
    assert failing(lambda r: r["critic"]["win"].update(explained_variance=0.08)) == [
        "win_explained_variance", "win_margin_over_baseline"]
    assert failing(lambda r: r["baselines"]["logistic_act_floor"]["act"].update(explained_variance=0.12)) == [
        "act_margin_over_baseline"]
    # Outcomes that do not vary on the held-out seeds measure nothing.
    assert failing(lambda r: r["critic"]["win"].update(explained_variance=None)) == [
        "win_explained_variance", "win_margin_over_baseline"]
    assert failing(lambda r: r["critic"]["act"].update(predicted=0.65)) == ["act_overall_gap"]
    assert failing(lambda r: r["critic"]["calibration"]["by_act"]["win"][2].update(gap=-0.2)) == ["win_act_gap"]
    assert failing(lambda r: r["critic"]["calibration"]["by_character"]["act"][0].update(gap=0.11)) == [
        "act_character_gap"]
    assert failing(lambda r: r["critic"]["calibration"]["by_probability"]["win"][1].update(n=400)) == ["win_bin_gap"]
    # Memorized runs: the training loss falls far more than the held-out Brier score gains.
    assert failing(lambda r: r["history"][-1].update(train_loss=0.6)) == ["memorization"]
    assert failing(lambda r: r["critic"]["win"].update(brier=0.2) or r["critic"]["act"].update(brier=0.3)) == [
        "memorization"]
    # A loss that did not fall memorized nothing, whatever the held-out seeds say.
    assert failing(lambda r: r["history"][-1].update(train_loss=1.3) or r["critic"]["win"].update(brier=0.2)) == []
    # Without baselines the margin and the memorization cannot be measured.
    assert set(failing(lambda r: r.pop("baselines"))) == {
        "win_margin_over_baseline", "act_margin_over_baseline", "memorization"}

    # Floors are settings: a gate for a small buffer, from the command line's overrides.
    small = Gate.of(["min_heldout_seeds=50", "min_explained_variance=0.05"])
    assert (small.min_heldout_seeds, small.min_explained_variance, small.max_act_gap) == (50, 0.05, 0.05)
    report = passing()
    report["data"]["split"]["test_seeds"] = 74
    assert check(report, small)["passed"] and not check(report)["passed"]
    for wrong in (["min_explained_variance"], ["floor=1"], ["min_bin_share=2"]):
        with pytest.raises(ValueError):
            Gate.of(wrong)


def test_the_constant_loss_is_the_entropy_of_the_base_rates(world):
    rows, _ = load([world["buffer"] / "round-0"])
    wins = sum(row.win for row in rows) / len(rows)
    passed = sum(row.act_passed for row in rows) / len(rows)

    def entropy(rate):
        import math

        return 0.0 if rate in (0, 1) else -(rate * math.log(rate) + (1 - rate) * math.log(1 - rate))

    assert constant_loss(rows, 0.5) == pytest.approx(entropy(wins) + 0.5 * entropy(passed))


def test_a_baseline_of_labels_that_do_not_vary_is_their_constant():
    from critic.baselines import fit_logistic

    features = [[float(index), float(index % 3)] for index in range(12)]
    assert fit_logistic(features, [1.0] * 12)(features[:3]) == [1.0] * 3
    assert fit_logistic(features, [0.0] * 12)(features[:2]) == [0.0] * 2
    varied = fit_logistic(features, [float(index > 5) for index in range(12)])(features)
    assert varied[0] < 0.1 and varied[-1] > 0.9
