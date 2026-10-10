"""Student battles in the training data: the solver's labels on the policy's own
fights are imported beside the demonstrations, told apart by source, and take a share
of their (character, phase) groups' weight."""

import json
from collections import Counter

import pytest

from model.cli import main
from model.data import (
    SHARE_RAMP_MACROS, STUDENT_BATTLE, RunShards, extent, group_weights, samples, student_share, weight_totals,
)
from model.dataset_index import index_row, read_index, rebuild_index, row_source, sample_source
from model.independent import convert_group, import_samples
from model.protocol import CHARACTERS, ProtocolError
from model.tests.test_independent import learner_for, samples as demo_rows

POLICY = dict(policy_version=3, policy_digest="a" * 64, iteration=2)


def student_rows(group, seed, policy=POLICY):
    """The rows `spire_codex_data.student_battles` exports for one fight, synthetic."""
    rows = demo_rows(group, visibility="privileged", seed=seed)
    for row in rows:
        row["metadata"].update(anchor_kind=STUDENT_BATTLE, label_source="combat_solver", split_group="split-" + seed,
                               **policy)
    return rows


def totals(items):
    """The total weight of every (character, phase, source) among `items`."""
    total = Counter()
    for item in items:
        total[item.character, item.phase, item.source] += item.weight
    return total


def test_the_student_share_starts_at_a_third_and_reaches_a_half():
    assert student_share(0) == pytest.approx(1 / 3)
    assert student_share(SHARE_RAMP_MACROS // 2) == pytest.approx(1 / 3 + 1 / 12)
    assert student_share(SHARE_RAMP_MACROS) == pytest.approx(1 / 2)
    assert student_share(10 * SHARE_RAMP_MACROS) == pytest.approx(1 / 2)
    values = [student_share(n) for n in range(0, 2 * SHARE_RAMP_MACROS, SHARE_RAMP_MACROS // 20)]
    assert values == sorted(values)
    with pytest.raises(ValueError):
        student_share(-1)


def test_group_weights_split_a_group_between_students_and_the_rest_by_the_share():
    counts = Counter({("Ironclad", "combat", "independent"): 400, ("Ironclad", "combat", STUDENT_BATTLE): 10,
                      ("Silent", "combat", "independent"): 50, ("Silent", "combat", "rollout"): 25,
                      ("Defect", "map", STUDENT_BATTLE): 7})
    weights = group_weights(counts, share=1 / 3)
    total = weight_totals(weights, counts)
    # Three groups of 1/3 each; the shared one gives students 1/3 of it, 1:2 against the rest.
    assert total["Ironclad", "combat", STUDENT_BATTLE] == pytest.approx(1 / 9)
    assert total["Ironclad", "combat", "independent"] == pytest.approx(2 / 9)
    # Without students the other sources share the group by count; alone, students take it whole.
    assert total["Silent", "combat", "independent"] == pytest.approx(2 / 9)
    assert total["Silent", "combat", "rollout"] == pytest.approx(1 / 9)
    assert weights["Silent", "combat", "independent"] == weights["Silent", "combat", "rollout"]
    assert total["Defect", "map", STUDENT_BATTLE] == pytest.approx(1 / 3)
    assert sum(total.values()) == pytest.approx(1.0)
    # The default share follows the student macro-actions counted: 17 of them is a third and a little.
    by_default = weight_totals(group_weights(counts), counts)
    assert by_default["Ironclad", "combat", STUDENT_BATTLE] == pytest.approx(student_share(17) / 3)
    with pytest.raises(ValueError):
        group_weights(counts, share=1.0)


def test_samples_tell_student_battles_apart_and_weigh_them_by_the_share():
    demos = [convert_group(demo_rows(f"d{i}", seed=f"d{i}")) for i in range(3)]
    students = [convert_group(student_rows(f"s{i}", f"s{i}")) for i in range(2)]
    assert [sample_source(run) for run in demos + students] == ["independent"] * 3 + [STUDENT_BATTLE] * 2
    assert students[0]["provenance"]["anchor_kind"] == STUDENT_BATTLE
    assert {k: students[0]["provenance"][k] for k in POLICY} == POLICY
    items = samples(demos + students)
    assert {i.source for i in items} == {"independent", STUDENT_BATTLE}
    total = totals(items)
    groups = {(c, p) for c, p, _ in total}
    share = student_share(sum(1 for i in items if i.source == STUDENT_BATTLE))
    for character, phase in groups:
        assert total[character, phase, STUDENT_BATTLE] == pytest.approx(share / len(groups))
        assert total[character, phase, "independent"] == pytest.approx((1 - share) / len(groups))
    assert sum(total.values()) == pytest.approx(1.0)
    # At a third exactly, the two kinds weigh 1:2 in every group they share.
    counts = Counter((i.character, i.phase, i.source) for i in items)
    exact = totals(samples(demos + students, weights=group_weights(counts, share=1 / 3)))
    for character, phase in groups:
        assert exact[character, phase, STUDENT_BATTLE] * 2 == pytest.approx(exact[character, phase, "independent"])
    # A student battle must name its policy.
    rows = student_rows("unnamed", "u")
    for row in rows:
        del row["metadata"]["policy_digest"]
    with pytest.raises(ProtocolError, match="names the policy"):
        convert_group(rows)


def test_old_index_rows_without_a_source_are_demonstrations_or_sampled_runs():
    assert row_source(dict(run_id="r")) == "independent"
    assert row_source(dict(run_id="r", victory=True, policy_version=1)) == "rollout"
    assert row_source(dict(run_id="r", source=STUDENT_BATTLE)) == STUDENT_BATTLE
    run = convert_group(student_rows("g", "g"))
    row = index_row(run)
    assert (row["source"], row["policy_version"], row["policy_digest"], row["iteration"]) == (
        STUDENT_BATTLE, POLICY["policy_version"], POLICY["policy_digest"], POLICY["iteration"])
    assert index_row(convert_group(demo_rows("d", seed="d")))["source"] == "independent"


def write_rows(path, groups):
    import gzip

    with gzip.open(path, "wt") as stream:
        stream.write("".join(json.dumps(r) + "\n" for rows in groups for r in rows))


def test_demonstrations_and_student_battles_are_read_together_with_their_shares(tmp_path):
    demos = [demo_rows(f"d{i}", seed=f"d{i}") for i in range(3)]
    for i, rows in enumerate(demos):
        for row in rows:
            row["metadata"]["split_group"] = f"split-d{i}"
    students = [student_rows(f"s{i}", f"s{i}") for i in range(2)]
    write_rows(tmp_path / "demos.jsonl.gz", demos)
    write_rows(tmp_path / "students.jsonl.gz", students)
    import_samples(tmp_path / "demos.jsonl.gz", tmp_path / "demo", shard_samples=len(demos[0]) + 1)
    summary = import_samples(tmp_path / "students.jsonl.gz", tmp_path / "student")
    assert summary["accepted_runs"] == 2
    # The student import's index names the source and the policy; rebuilding keeps them.
    rows = read_index(tmp_path / "student")
    assert all(row["source"] == STUDENT_BATTLE and {k: row[k] for k in POLICY} == POLICY for row in rows)
    rebuild_index(tmp_path / "student")
    assert read_index(tmp_path / "student") == rows
    assert all(row["source"] == "independent" for row in read_index(tmp_path / "demo"))
    shards = RunShards([tmp_path / "demo", tmp_path / "student"])
    assert len(shards) == 5 and len(shards.directories) == 2
    with pytest.raises(ValueError, match="several"):
        shards.directory
    assert sorted({root for root, _ in (name for names in shards.window_names(8) for name in names)}) == [0, 1]
    counts = shards.source_counts
    assert {source for _, _, source in counts} == {"independent", STUDENT_BATTLE}
    assert shards.student_macros == sum(n for (_, _, s), n in counts.items() if s == STUDENT_BATTLE) > 0
    grouped = Counter()
    for (character, phase, _), n in counts.items():
        grouped[character, phase] += n
    assert shards.counts == grouped
    assert extent(shards)["student_macros"] == shards.student_macros
    assert extent(shards)["student_share"] == pytest.approx(student_share(shards.student_macros))
    # Windows read across the directories; the weights are the whole data set's.
    everything = [run for window in shards.windows(shards=1) for run in window]
    assert len(everything) == 5 and {sample_source(r) for r in everything} == {"independent", STUDENT_BATTLE}
    items = [item for window in shards.windows(shards=2) for item in samples(window, weights=shards.weights)]
    total = totals(items)
    share = student_share(shards.student_macros)
    groups = {(c, p) for c, p, _ in total}
    for character, phase in groups:
        assert total[character, phase, STUDENT_BATTLE] / total[character, phase, "independent"] == pytest.approx(
            share / (1 - share))
    assert sum(total.values()) == pytest.approx(1.0)
    # The held-out rule reads the student rows' seeds like any other.
    train, held = shards.split(0.5)
    assert len(train) + len(held) == 5 and not train.seeds & held.seeds
    assert shards.metadata()["training_engine_contracts"] == [everything[0]["contract"]]
    learner = learner_for(everything)
    metrics = learner.bootstrap(shards, epochs=1, window_shards=1, include_held_out=True)
    assert metrics[0]["weight"] == pytest.approx(1.0) and metrics[0]["training"]["student_macros"] > 0


def test_bootstrap_takes_several_data_directories(tmp_path, capsys):
    assert main(["smoke", "--output", str(tmp_path / "smoke")]) == 0
    demos = [demo_rows(f"d{i}", seed=f"d{i}", character=c) for i, c in enumerate(CHARACTERS)]
    students = [student_rows(f"s{i}", f"s{i}") for i in range(2)]
    write_rows(tmp_path / "demos.jsonl.gz", demos)
    write_rows(tmp_path / "students.jsonl.gz", students)
    import_samples(tmp_path / "demos.jsonl.gz", tmp_path / "demo")
    import_samples(tmp_path / "students.jsonl.gz", tmp_path / "student")
    capsys.readouterr()
    assert main(["bootstrap", "--data", str(tmp_path / "demo"), str(tmp_path / "student"),
                 "--checkpoint", str(tmp_path / "smoke/checkpoint"), "--weights-only", "--precision", "no",
                 "--epochs", "1", "--include-heldout", "--output", str(tmp_path / "bc")]) == 0
    record = json.loads((tmp_path / "bc/history.jsonl").read_text())
    assert record["training_runs"] == 7 and record["metrics"]["training"]["student_macros"] > 0
    assert record["metrics"]["training"]["student_share"] == pytest.approx(1 / 3, abs=1e-3)
    # A path that is not sharded cannot be read together with sharded ones.
    assert main(["bootstrap", "--data", str(tmp_path / "demo"), str(tmp_path / "smoke"),
                 "--checkpoint", str(tmp_path / "smoke/checkpoint"), "--weights-only",
                 "--output", str(tmp_path / "bc2")]) == 1
    assert "sharded" in capsys.readouterr().err
