"""Student battles in the training data: the solver's labels on the policy's own
fights are imported beside the demonstrations, told apart by source, and take a share
of their character's combat group's weight."""

import gzip
import json
from collections import Counter

import pytest

from model.cli import main
from model.data import (
    SHARE_RAMP_MACROS, STUDENT_BATTLE, RunShards, extent, group_weights, samples, student_share, weight_totals,
)
from model.dataset_index import index_row, read_index, rebuild_index, row_source, sample_source
from model.independent import convert_group, import_samples
from model.protocol import CHARACTERS, ProtocolError, execution_command
from model.testing import SyntheticEngine
from model.tests.test_independent import learner_for
from spire_codex_data.anchor import row

POLICY = dict(policy_version=3, policy_digest="a" * 64, iteration=2)


class Fighting(SyntheticEngine):
    """An engine whose selection is a fight: its frames stand in the combat phase."""

    def frame(self):
        frame = super().frame()
        if self.stage == "selection":
            frame["public"]["phase"] = "combat"
        return frame


def rows(group, seed, character="IRONCLAD", policy=None):
    """The rows exported for one solver fight: a summary anchor's battle, or with
    `policy` a student battle of that policy. One macro-action in the combat phase (the
    fight's selection) and one decision after it, in another phase."""
    engine = Fighting(4, 2)
    frame = engine.reset("Ironclad", seed)
    kind = dict(anchor_kind=STUDENT_BATTLE, **policy) if policy else dict(anchor_kind="battle")
    result = []
    while frame["boundary"] != "terminal":
        candidate = frame["legal"]["candidates"][0]
        item = row(frame, candidate, "combat_solver")
        item["metadata"] = dict(
            run_hash="run", sample_group=group, split_group="split-" + seed, character=character, ascension=10,
            game_version="v0.111.0", teacher_visibility="privileged", label_source="combat_solver",
            native_replay_verified=True, independent=True, bc_only=True, routing=item.pop("routing"),
            contract=engine.contract, **kind)
        del item["before_hash"]
        result.append(item)
        frame = engine.send(execution_command(frame, candidate["candidate_ref"]))
    return result


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


def test_group_weights_split_a_combat_group_between_students_and_the_rest_by_the_share():
    counts = Counter({("Ironclad", "combat", "independent"): 400, ("Ironclad", "combat", STUDENT_BATTLE): 10,
                      ("Silent", "combat", "independent"): 50, ("Silent", "combat", "rollout"): 25,
                      ("Defect", "combat", STUDENT_BATTLE): 7,
                      ("Ironclad", "card_select", "independent"): 90, ("Ironclad", "card_select", STUDENT_BATTLE): 10})
    weights = group_weights(counts, share=1 / 3)
    total = weight_totals(weights, counts)
    # Four groups of 1/4 each. The combat group with both gives students 1/3 of it: 1:2 against the rest.
    assert total["Ironclad", "combat", STUDENT_BATTLE] == pytest.approx(1 / 12)
    assert total["Ironclad", "combat", "independent"] == pytest.approx(2 / 12)
    # Without students the other sources share the group by count; alone, students take it whole.
    assert total["Silent", "combat", "independent"] == pytest.approx(2 / 12)
    assert total["Silent", "combat", "rollout"] == pytest.approx(1 / 12)
    assert weights["Silent", "combat", "independent"] == weights["Silent", "combat", "rollout"]
    assert total["Defect", "combat", STUDENT_BATTLE] == pytest.approx(1 / 4)
    # Outside the combat phase a student battle's selections are counted with the rest of their group.
    assert weights["Ironclad", "card_select", STUDENT_BATTLE] == weights["Ironclad", "card_select", "independent"]
    assert total["Ironclad", "card_select", STUDENT_BATTLE] == pytest.approx(1 / 40)
    assert sum(total.values()) == pytest.approx(1.0)
    # The default share follows the student macro-actions counted: 27 of them is a third and a little.
    by_default = weight_totals(group_weights(counts), counts)
    assert by_default["Ironclad", "combat", STUDENT_BATTLE] == pytest.approx(student_share(27) / 4)
    with pytest.raises(ValueError):
        group_weights(counts, share=1.0)


def test_samples_tell_student_battles_apart_and_weigh_them_by_the_share():
    demos = [convert_group(rows(f"d{i}", f"d{i}")) for i in range(3)]
    students = [convert_group(rows(f"s{i}", f"s{i}", policy=POLICY)) for i in range(2)]
    assert [sample_source(run) for run in demos + students] == ["independent"] * 3 + [STUDENT_BATTLE] * 2
    assert students[0]["provenance"]["anchor_kind"] == STUDENT_BATTLE
    assert {k: students[0]["provenance"][k] for k in POLICY} == POLICY
    items = samples(demos + students)
    assert {i.source for i in items} == {"independent", STUDENT_BATTLE}
    assert {i.phase for i in items} == {"combat", "combat_play"}
    total = totals(items)
    share = student_share(sum(1 for i in items if i.source == STUDENT_BATTLE))
    # Two groups of a half each: the fight's, shared by the share, and the decision after it, by count.
    assert total["Ironclad", "combat", STUDENT_BATTLE] == pytest.approx(share / 2)
    assert total["Ironclad", "combat", "independent"] == pytest.approx((1 - share) / 2)
    assert [i.weight for i in items if i.phase == "combat_play"] == [pytest.approx(1 / 10)] * 5
    assert sum(total.values()) == pytest.approx(1.0)
    # At a third exactly, the two kinds weigh 1:2 in the character's combat group.
    counts = Counter((i.character, i.phase, i.source) for i in items)
    exact = totals(samples(demos + students, weights=group_weights(counts, share=1 / 3)))
    assert exact["Ironclad", "combat", STUDENT_BATTLE] * 2 == pytest.approx(exact["Ironclad", "combat", "independent"])
    assert exact["Ironclad", "combat", STUDENT_BATTLE] == pytest.approx(1 / 6)
    # A student battle must name its policy.
    unnamed = rows("unnamed", "u", policy=POLICY)
    for item in unnamed:
        del item["metadata"]["policy_digest"]
    with pytest.raises(ProtocolError, match="names the policy"):
        convert_group(unnamed)


def test_old_index_rows_without_a_source_are_demonstrations_or_sampled_runs():
    assert row_source(dict(run_id="r")) == "independent"
    assert row_source(dict(run_id="r", victory=True, policy_version=1)) == "rollout"
    assert row_source(dict(run_id="r", source=STUDENT_BATTLE)) == STUDENT_BATTLE
    indexed = index_row(convert_group(rows("g", "g", policy=POLICY)))
    assert (indexed["source"], indexed["policy_version"], indexed["policy_digest"], indexed["iteration"]) == (
        STUDENT_BATTLE, POLICY["policy_version"], POLICY["policy_digest"], POLICY["iteration"])
    assert index_row(convert_group(rows("d", "d")))["source"] == "independent"


def write_rows(path, groups):
    with gzip.open(path, "wt") as stream:
        stream.write("".join(json.dumps(r) + "\n" for group in groups for r in group))


def test_demonstrations_and_student_battles_are_read_together_with_their_shares(tmp_path):
    demos = [rows(f"d{i}", f"d{i}") for i in range(3)]
    students = [rows(f"s{i}", f"s{i}", policy=POLICY) for i in range(2)]
    write_rows(tmp_path / "demos.jsonl.gz", demos)
    write_rows(tmp_path / "students.jsonl.gz", students)
    import_samples(tmp_path / "demos.jsonl.gz", tmp_path / "demo", shard_samples=len(demos[0]) + 1)
    summary = import_samples(tmp_path / "students.jsonl.gz", tmp_path / "student")
    assert summary["accepted_runs"] == 2
    # The student import's index names the source and the policy; rebuilding keeps them.
    indexed = read_index(tmp_path / "student")
    assert all(r["source"] == STUDENT_BATTLE and {k: r[k] for k in POLICY} == POLICY for r in indexed)
    rebuild_index(tmp_path / "student")
    assert read_index(tmp_path / "student") == indexed
    assert all(r["source"] == "independent" for r in read_index(tmp_path / "demo"))
    shards = RunShards([tmp_path / "demo", tmp_path / "student"])
    assert len(shards) == 5 and len(shards.directories) == 2
    with pytest.raises(ValueError, match="several"):
        shards.directory
    assert sorted({root for names in shards.window_names(8) for root, _ in names}) == [0, 1]
    counts = shards.source_counts
    assert counts == {("Ironclad", "combat", "independent"): 3, ("Ironclad", "combat_play", "independent"): 3,
                      ("Ironclad", "combat", STUDENT_BATTLE): 2, ("Ironclad", "combat_play", STUDENT_BATTLE): 2}
    assert shards.counts == {("Ironclad", "combat"): 5, ("Ironclad", "combat_play"): 5} and shards.student_macros == 4
    assert extent(shards) == dict(groups=5, macros=10, seeds=5, student_macros=4, student_share=student_share(4))
    assert extent(RunShards(tmp_path / "demo"))["student_share"] is None
    # Windows read across the directories; the weights are the whole data set's.
    everything = [run for window in shards.windows(shards=1) for run in window]
    assert len(everything) == 5 and {sample_source(r) for r in everything} == {"independent", STUDENT_BATTLE}
    items = [item for window in shards.windows(shards=2) for item in samples(window, weights=shards.weights)]
    total = totals(items)
    share = student_share(4)
    assert total["Ironclad", "combat", STUDENT_BATTLE] / total["Ironclad", "combat", "independent"] == pytest.approx(
        share / (1 - share))
    assert total["Ironclad", "combat_play", STUDENT_BATTLE] / total["Ironclad", "combat_play", "independent"] == (
        pytest.approx(2 / 3))
    assert sum(total.values()) == pytest.approx(1.0)
    # The held-out rule reads the student rows' seeds like any other.
    train, held = shards.split(0.5)
    assert len(train) + len(held) == 5 and not train.seeds & held.seeds
    assert shards.metadata()["training_engine_contracts"] == [everything[0]["contract"]]
    # An index that says another source than its shard holds is stale.
    stale = RunShards([tmp_path / "demo", tmp_path / "student"],
                      [dict(r, source="independent") for r in shards.rows])
    with pytest.raises(ProtocolError, match="source disagrees"):
        list(stale.windows())
    learner = learner_for(everything)
    metrics = learner.bootstrap(shards, epochs=1, window_shards=1, include_held_out=True)
    assert metrics[0]["weight"] == pytest.approx(1.0) and metrics[0]["samples"] == 10
    assert metrics[0]["training"]["student_macros"] == 4


def test_bootstrap_takes_several_data_directories(tmp_path, capsys):
    assert main(["smoke", "--output", str(tmp_path / "smoke")]) == 0
    demos = [rows(f"d{i}", f"d{i}", character=c.upper()) for i, c in enumerate(CHARACTERS)]
    students = [rows(f"s{i}", f"s{i}", policy=POLICY) for i in range(2)]
    write_rows(tmp_path / "demos.jsonl.gz", demos)
    write_rows(tmp_path / "students.jsonl.gz", students)
    import_samples(tmp_path / "demos.jsonl.gz", tmp_path / "demo")
    import_samples(tmp_path / "students.jsonl.gz", tmp_path / "student")
    capsys.readouterr()
    assert main(["bootstrap", "--data", str(tmp_path / "demo"), str(tmp_path / "student"),
                 "--checkpoint", str(tmp_path / "smoke/checkpoint"), "--weights-only", "--precision", "no",
                 "--epochs", "1", "--include-heldout", "--output", str(tmp_path / "bc")]) == 0
    record = json.loads((tmp_path / "bc/history.jsonl").read_text())
    assert record["training_runs"] == 7 and record["metrics"]["training"]["student_macros"] == 4
    assert record["metrics"]["training"]["student_share"] == pytest.approx(student_share(4))
    assert record["metrics"]["weight"] == pytest.approx(1.0)
    # A path that is not sharded cannot be read together with sharded ones.
    assert main(["bootstrap", "--data", str(tmp_path / "demo"), str(tmp_path / "smoke"),
                 "--checkpoint", str(tmp_path / "smoke/checkpoint"), "--weights-only",
                 "--output", str(tmp_path / "bc2")]) == 1
    assert "sharded" in capsys.readouterr().err
