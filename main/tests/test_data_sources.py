"""Supervised data directories: several read as one data set, the runs of one
ascension, fights taken from other directories, and indexes rebuilt."""

import gzip
import json
from collections import Counter

import pytest

from main.cli import main
from main.data.shards import RunShards, at_ascension, combat_from, extent, group_weights, samples, weight_totals
from main.data.index import INDEX, index_row, read_index, rebuild_index, row_source, sample_source
from main.demos.independent import convert_group, import_samples
from main.env.protocol import CHARACTERS, ProtocolError, execution_command
from main.play.testing import SyntheticEngine
from main.tests.test_independent import learner_for
from spire_codex_data.anchor import row


class Fighting(SyntheticEngine):
    """An engine whose selection is a fight: its frames stand in the combat phase."""

    def frame(self):
        frame = super().frame()
        if self.stage == "selection":
            frame["public"]["phase"] = "combat"
        return frame


def rows(group, seed, character="IRONCLAD", ascension=10):
    """The rows exported for one solver fight of a summary anchor: one macro-action in
    the combat phase (the fight's selection) and one decision after it, in another phase."""
    engine = Fighting(4, 2)
    frame = engine.reset("Ironclad", seed, ascension)
    result = []
    while frame["boundary"] != "terminal":
        candidate = frame["legal"]["candidates"][0]
        item = row(frame, candidate, "combat_solver")
        item["metadata"] = dict(
            run_hash="run", sample_group=group, split_group="split-" + seed, character=character, ascension=ascension,
            game_version="v0.111.0", teacher_visibility="privileged", label_source="combat_solver",
            native_replay_verified=True, independent=True, bc_only=True, routing=item.pop("routing"),
            contract=engine.contract, anchor_kind="battle")
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


def test_old_index_rows_without_a_source_are_demonstrations_or_sampled_runs():
    assert row_source(dict(run_id="r")) == "independent"
    assert row_source(dict(run_id="r", victory=True, policy_version=1)) == "rollout"
    assert index_row(convert_group(rows("d", "d")))["source"] == "independent"


def write_rows(path, groups):
    with gzip.open(path, "wt") as stream:
        stream.write("".join(json.dumps(r) + "\n" for group in groups for r in group))


def test_several_directories_are_read_together_as_one_data_set(tmp_path):
    first = [rows(f"d{i}", f"d{i}") for i in range(3)]
    second = [rows(f"s{i}", f"s{i}") for i in range(2)]
    write_rows(tmp_path / "first.jsonl.gz", first)
    write_rows(tmp_path / "second.jsonl.gz", second)
    import_samples(tmp_path / "first.jsonl.gz", tmp_path / "first", shard_samples=len(first[0]) + 1)
    import_samples(tmp_path / "second.jsonl.gz", tmp_path / "second")
    shards = RunShards([tmp_path / "first", tmp_path / "second"])
    assert len(shards) == 5 and len(shards.directories) == 2
    with pytest.raises(ValueError, match="several"):
        shards.directory
    assert sorted({root for names in shards.window_names(8) for root, _ in names}) == [0, 1]
    assert shards.counts == {("Ironclad", "combat"): 5, ("Ironclad", "combat_play"): 5}
    assert extent(shards) == dict(groups=5, macros=10, seeds=5)
    # Windows read across the directories; the weights are the whole data set's.
    everything = [run for window in shards.windows(shards=1) for run in window]
    assert len(everything) == 5 and {sample_source(r) for r in everything} == {"independent"}
    items = [item for window in shards.windows(shards=2) for item in samples(window, weights=shards.weights)]
    total = totals(items)
    assert total["Ironclad", "combat", "independent"] == pytest.approx(0.5)
    assert sum(total.values()) == pytest.approx(1.0)
    train, held = shards.split(0.5)
    assert len(train) + len(held) == 5 and not train.seeds & held.seeds
    assert shards.metadata()["training_engine_contracts"] == [everything[0]["contract"]]
    # An index that says another source than its shard holds is stale.
    stale = RunShards([tmp_path / "first", tmp_path / "second"], [dict(r, source="rollout") for r in shards.rows])
    with pytest.raises(ProtocolError, match="source disagrees"):
        list(stale.windows())
    learner = learner_for(everything)
    metrics = learner.bootstrap(shards, epochs=1, window_shards=1, include_held_out=True)
    assert metrics[0]["weight"] == pytest.approx(1.0) and metrics[0]["samples"] == 10


def test_group_weights_share_a_group_equally_whatever_the_source():
    counts = Counter({("Ironclad", "combat", "independent"): 30, ("Ironclad", "combat", "rollout"): 10,
                      ("Silent", "map", "rollout"): 5, ("Silent", "shop", "rollout"): 0})
    weights = group_weights(counts)
    assert set(weights) == {("Ironclad", "combat", "independent"), ("Ironclad", "combat", "rollout"),
                            ("Silent", "map", "rollout")}
    assert weights["Ironclad", "combat", "independent"] == weights["Ironclad", "combat", "rollout"] == 1 / 80
    assert sum(weight_totals(weights, counts).values()) == pytest.approx(1.0)


def test_bootstrap_takes_several_data_directories(tmp_path, capsys):
    assert main(["smoke", "--output", str(tmp_path / "smoke")]) == 0
    demos = [rows(f"d{i}", f"d{i}", character=c.upper()) for i, c in enumerate(CHARACTERS)]
    others = [rows(f"s{i}", f"s{i}") for i in range(2)]
    write_rows(tmp_path / "demos.jsonl.gz", demos)
    write_rows(tmp_path / "others.jsonl.gz", others)
    import_samples(tmp_path / "demos.jsonl.gz", tmp_path / "demo")
    import_samples(tmp_path / "others.jsonl.gz", tmp_path / "other")
    capsys.readouterr()
    assert main(["bootstrap", "--data", str(tmp_path / "demo"), str(tmp_path / "other"),
                 "--checkpoint", str(tmp_path / "smoke/checkpoint"), "--weights-only", "--precision", "no",
                 "--epochs", "1", "--include-heldout", "--output", str(tmp_path / "bc")]) == 0
    record = json.loads((tmp_path / "bc/history.jsonl").read_text())
    assert record["training_runs"] == 7 and record["metrics"]["training"]["macros"] == 14
    assert record["metrics"]["weight"] == pytest.approx(1.0)
    # A path that is not sharded cannot be read together with sharded ones.
    assert main(["bootstrap", "--data", str(tmp_path / "demo"), str(tmp_path / "smoke"),
                 "--checkpoint", str(tmp_path / "smoke/checkpoint"), "--weights-only",
                 "--output", str(tmp_path / "bc2")]) == 1
    assert "sharded" in capsys.readouterr().err


def test_the_runs_of_one_ascension_are_a_data_set_of_their_own(tmp_path):
    groups = [rows(f"d{i}", f"d{i}", ascension=10 if i % 2 else i) for i in range(6)]
    write_rows(tmp_path / "demos.jsonl.gz", groups)
    import_samples(tmp_path / "demos.jsonl.gz", tmp_path / "demo", shard_samples=len(groups[0]) + 1)
    shards = RunShards(tmp_path / "demo")
    assert Counter(row["ascension"] for row in shards.rows) == {10: 3, 0: 1, 2: 1, 4: 1}
    top = at_ascension(shards, 10)
    assert len(top) == 3 and top.macros == shards.macros // 2
    # The weights are those of the part: its groups share the whole between them.
    assert sum(weight_totals(top.weights, top.source_counts).values()) == pytest.approx(1.0)
    assert {run["ascension"] for names in top.window_names(8) for run in top.read(names)} == {10}
    assert len(at_ascension(shards, 7)) == 0
    # Runs in memory are told apart the same way.
    held = [run for names in shards.window_names(8) for run in shards.read(names)]
    assert len(held) == 6 and len(at_ascension(held, 10)) == 3
    # An index written before ascensions were recorded cannot tell: it is rebuilt.
    path = tmp_path / "demo" / INDEX
    with gzip.open(path, "rt") as stream:
        old = [json.loads(line) for line in stream]
    with gzip.open(path, "wt") as stream:
        stream.write("".join(json.dumps({k: v for k, v in row.items() if k != "ascension"}) + "\n" for row in old))
    with pytest.raises(ProtocolError, match="reindex"):
        at_ascension(RunShards(tmp_path / "demo"), 10)
    rebuild_index(tmp_path / "demo")
    assert len(at_ascension(RunShards(tmp_path / "demo"), 10)) == 3


def test_fights_are_taken_from_the_directories_named_for_them(tmp_path):
    demos = [rows(f"d{i}", f"d{i}") for i in range(4)]
    write_rows(tmp_path / "demos.jsonl.gz", demos)
    import_samples(tmp_path / "demos.jsonl.gz", tmp_path / "demo", shard_samples=len(demos[0]) + 1)
    write_rows(tmp_path / "better.jsonl.gz", [rows(f"b{i}", f"b{i}") for i in range(2)])
    import_samples(tmp_path / "better.jsonl.gz", tmp_path / "better", shard_samples=len(demos[0]) + 1)
    # A run without combat macro-actions among the demonstrations: a decision outside a fight.
    path = tmp_path / "demo" / INDEX
    with gzip.open(path, "rt") as stream:
        header, *listed = stream.read().splitlines()
    outside = json.loads(listed[0])
    outside["phases"] = {phase: n for phase, n in outside["phases"].items() if phase != "combat"}
    assert outside["phases"]
    with gzip.open(path, "wt") as stream:
        stream.write("\n".join([header, json.dumps(outside), *listed[1:]]) + "\n")
    both = RunShards([tmp_path / "demo", tmp_path / "better"])
    assert len(both) == 6
    part = combat_from(both, [tmp_path / "better"])
    assert [(row.get("root", 0), "combat" in row["phases"]) for row in part.rows] == [(0, False), (1, True), (1, True)]
    assert len(combat_from(both, [tmp_path / "demo", tmp_path / "better"])) == 6
    assert sum(weight_totals(part.weights, part.source_counts).values()) == pytest.approx(1.0)
    with pytest.raises(ProtocolError, match="--data"):
        combat_from(both, [tmp_path / "elsewhere"])
    with pytest.raises(ProtocolError, match="sharded"):
        combat_from([dict(phases={})], [tmp_path / "better"])


def test_a_directory_sharing_shards_is_reindexed_as_the_part_it_lists(tmp_path):
    groups = [rows(f"d{i}", f"d{i}", ascension=10 if i % 2 else 0) for i in range(6)]
    write_rows(tmp_path / "demos.jsonl.gz", groups)
    import_samples(tmp_path / "demos.jsonl.gz", tmp_path / "whole", shard_samples=2 * len(groups[0]) + 1)
    # A part of the whole: its own index over the whole's shards, as a subset is kept.
    part = tmp_path / "part"
    part.mkdir()
    (part / "accepted").symlink_to(tmp_path / "whole" / "accepted")
    with gzip.open(tmp_path / "whole" / INDEX, "rt") as stream:
        header, *lines = stream.readlines()
    kept = [json.loads(line) for line in lines[::2]]
    with gzip.open(part / INDEX, "wt") as stream:
        stream.write(header + "".join(json.dumps({k: v for k, v in row.items() if k != "ascension"}) + "\n"
                                      for row in kept))
    with pytest.raises(ProtocolError, match="--listed-only"):
        rebuild_index(part)
    rebuild_index(part, listed_only=True)
    assert read_index(part) == kept
    assert len(read_index(tmp_path / "whole")) == 6
