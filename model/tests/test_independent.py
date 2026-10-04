"""Independent Spire Codex samples become supervised Bootstrap runs with engine-routed macros."""

import gzip
import json

import pytest

from model.config import ModelConfig, TrainConfig
from model.data import RunShards, load_runs, samples as data_samples, validate_run
from model.independent import convert_group, import_samples
from model.model import PolicyValue
from model.protocol import ProtocolError, action_semantics, execution_command
from model.representation import Vocabulary
from model.testing import SyntheticEngine, demonstration
from model.trainer import Learner
from spire_codex_data.anchor import row


def samples(group="run", character="IRONCLAD", visibility="recorded_public", seed="seed"):
    """The rows spire_codex_data exports for one group, from a synthetic run."""
    engine = SyntheticEngine(4, 2)
    frame = engine.reset("Ironclad", seed)
    rows = []
    while frame["boundary"] != "terminal":
        candidate = frame["legal"]["candidates"][0]
        item = row(frame, candidate, "historical_noncombat")
        item["metadata"] = dict(
            run_hash="run", sample_group=group, split_group="split", character=character, ascension=10,
            game_version="v0.111.0", teacher_visibility=visibility, native_replay_verified=True,
            independent=True, bc_only=True, routing=item.pop("routing"), contract=engine.contract)
        del item["before_hash"]
        rows.append(item)
        frame = engine.send(execution_command(frame, candidate["candidate_ref"]))
    return rows


def test_group_becomes_a_supervised_run_with_the_engine_macros():
    run = convert_group(samples())
    expected = demonstration("Ironclad", "seed", 4, 2)
    assert (run["source"], run["status"], run["victory"], run["character"]) == ("independent", "partial", None, "Ironclad")
    assert (run["seed"], run["run_id"], run["teacher_visibility"]) == ("split", "run", "recorded_public")
    # A buffered selection stays one macro, including its forced steps.
    assert [len(m["steps"]) for m in run["macros"]] == [len(m["steps"]) for m in expected["macros"]]
    assert max(len(m["steps"]) for m in run["macros"]) > 1
    for macro, reference in zip(run["macros"], expected["macros"]):
        for step, other in zip(macro["steps"], reference["steps"]):
            chosen = {c["candidate_ref"]: c for c in step["frame"]["legal"]["candidates"]}[step["candidate_ref"]]
            label = {c["candidate_ref"]: c for c in other["frame"]["legal"]["candidates"]}[other["candidate_ref"]]
            assert action_semantics(chosen) == action_semantics(label)
            assert step["frame"]["routing"] == other["frame"]["routing"]
    with pytest.raises(ProtocolError, match="supervised-only"):
        validate_run(run, on_policy=True)


def test_solver_samples_keep_privileged_visibility_and_reject_mixed_groups():
    rows = samples(visibility="privileged")
    assert convert_group(rows)["teacher_visibility"] == "privileged"
    rows[-1]["metadata"]["teacher_visibility"] = "recorded_public"
    with pytest.raises(ProtocolError, match="mixes teacher_visibility"):
        convert_group(rows)


def learner_for(runs):
    config = ModelConfig.tiny()
    frames = (s["frame"] for r in runs for m in r["macros"] for s in m["steps"])
    return Learner(PolicyValue(config), Vocabulary.from_frames(frames, config),
                   TrainConfig(precision="no", logical_batch_size=4, token_buckets=[32, 64], action_buckets=[8, 16]))


def test_import_quarantines_bad_groups_and_bootstrap_trains_on_the_rest(tmp_path):
    good, bad, unknown = samples("run"), samples("run:battle-000", seed="other"), samples("modded", character="WATCHER")
    bad[0]["label"] = dict(bad[0]["label"], verb="NOT_OFFERED")
    path = tmp_path / "independent-training.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in good + bad + unknown))
    summary = import_samples(path, tmp_path / "import")
    assert (summary["accepted_runs"], summary["quarantined_groups"], summary["samples"]) == (1, 2, 3 * len(good))
    assert set(summary["quarantine_reasons"]) == {"Label is not a unique legal option", "Missing character/run identity"}
    runs = load_runs(tmp_path / "import/accepted/00000.jsonl.gz")
    assert summary["macros"] == len(runs[0]["macros"])
    assert learner_for(runs).bootstrap(runs)[0]["updates"] > 0


def test_import_shards_are_read_a_window_at_a_time_with_global_weights(tmp_path):
    groups = [samples(f"g{i}", seed=f"seed{i}") for i in range(6)]
    for i, rows in enumerate(groups):
        for row in rows:
            row["metadata"]["split_group"] = f"split{i}"
    path = tmp_path / "independent-training.jsonl.gz"
    with gzip.open(path, "wt") as stream:
        stream.write("".join(json.dumps(r) + "\n" for rows in groups for r in rows))
    # A group is never split: every shard closes on a group boundary.
    summary = import_samples(path, tmp_path / "import", shard_samples=len(groups[0]) + 1)
    assert (summary["accepted_runs"], summary["shards"]) == (6, 3)
    shards = RunShards(tmp_path / "import")
    assert RunShards.is_at(tmp_path / "import") and len(shards) == 6
    everything = [run for window in shards.windows(shards=1) for run in window]
    assert [len(w) for w in shards.windows(shards=1)] == [2, 2, 2] and len(list(shards.windows(shards=8))) == 1
    # Weights of a window are those the whole data set gives its samples.
    whole = {(i.run_id, i.macro["phase"]): i.weight for i in data_samples(everything)}
    window = next(shards.windows(shards=1))
    assert all(i.weight == whole[i.run_id, i.macro["phase"]] for i in data_samples(window, counts=shards.counts))
    assert sum(i.weight for w in shards.windows(1) for i in data_samples(w, counts=shards.counts)) == pytest.approx(1.0)
    # A seed is held out whole, and the two parts never share a run.
    train, validation = shards.split(0.5)
    assert len(train) + len(validation) == 6 and not train.seeds & validation.seeds
    assert shards.metadata()["training_engine_contracts"] == [everything[0]["contract"]]
    learner = learner_for(everything)
    metrics = learner.bootstrap(shards, epochs=2, window_shards=1)
    assert len(metrics) == 2 and metrics[0]["weight"] == pytest.approx(1.0)
    assert metrics[1]["updates"] > metrics[0]["updates"] > 0
    report = learner.evaluate_bootstrap(shards, window_shards=2)
    assert sum(group["macros"] for group in report.values()) == summary["macros"]
