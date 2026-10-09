"""Independent Spire Codex samples become supervised Bootstrap runs with engine-routed macros."""

import gzip
import json

import pytest
import torch

from model.config import ModelConfig, TrainConfig
from model.data import RunShards, load_runs, samples as data_samples, validate_run
from model.independent import convert_group, import_samples
from model import loader
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


def test_the_run_keeps_how_far_its_source_and_states_are_known():
    rows = samples()
    for item in rows:
        item["metadata"].update(source_integrity="unverified_source", initialization="summary-anchor-v1",
                                label_source="historical_choice", state_sources=[])
    rows[0]["metadata"].update(state_sources=["derived", "sampled_counter"], verified_by="recorded_outcome",
                               native_replay_verified=False)
    rows[0]["metadata"]["recovery"] = dict(offer="recorded_single_chest_relic", matching_gold_rolls=[47])
    kept = convert_group(rows)["provenance"]
    assert (kept["source_integrity"], kept["initialization"]) == ("unverified_source", "summary-anchor-v1")
    assert kept["label_sources"] == {"historical_choice": len(rows)}
    assert kept["state_sources"] == {"derived": 1, "sampled_counter": 1}
    assert kept["verified_by"] == {"recorded_outcome": 1, "native_replay": len(rows) - 1}
    assert kept["recovery"] == rows[0]["metadata"]["recovery"]


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


def test_a0_validation_is_reported_separately_from_mixed_ascensions():
    high = convert_group(samples('high'))
    rows = samples('a0')
    for item in rows:
        item['metadata']['ascension'] = 0
        item['metadata']['contract']['fixed_ascension'] = 0
    low = convert_group(rows)
    learner = learner_for([high, low])
    learner.config.loader_workers = 0
    report = learner.evaluate_bootstrap([high, low], include_a0=True)
    for name, group in report.items():
        if name.startswith('A0/'):
            combined = report[name[3:]]
            assert group['macros'] * 2 == combined['macros']
            assert group['nll_per_branch'] == pytest.approx(combined['nll_per_branch'])
    assert any(name.startswith('A0/') for name in report)


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

    # Feeder processes hand over what the trainer would pack itself, without the frames.
    def packed(workers):
        learner.config.loader_workers = workers
        rows = []
        for logical in loader.stream(shards, learner.vocabulary, learner.config, window_shards=1, shuffle=False,
                                     counts=shards.counts):
            for micro, prepared in logical:
                with torch.no_grad():
                    scores = learner.model.replay(prepared)[0].tolist()
                rows += [(i.run_id, i.phase, i.branches, i.ascension, i.weight, round(score, 5), i.macro is None)
                         for i, score in zip(micro, scores)]
        return sorted(rows)
    here, fed = packed(0), packed(2)
    assert [row[:-1] for row in here] == [row[:-1] for row in fed] and len(here) == summary["macros"]
    assert not any(row[-1] for row in here) and all(row[-1] for row in fed)


def test_import_by_several_processes_writes_what_one_writes(tmp_path):
    groups = [samples(f"g{i}", seed=f"seed{i}") for i in range(5)] + [samples("g1", seed="again")]
    groups[3][0]["label"] = dict(groups[3][0]["label"], verb="NOT_OFFERED")
    path = tmp_path / "independent-training.jsonl.gz"
    with gzip.open(path, "wt") as stream:
        stream.write("".join(json.dumps(r) + "\n" for rows in groups for r in rows))
    one = import_samples(path, tmp_path / "one", shard_samples=len(groups[0]) + 1)
    several = import_samples(path, tmp_path / "several", shard_samples=len(groups[0]) + 1, workers=2)
    # A refused group and a repeated one are refused alike, and the runs land in the same shards.
    assert one["quarantine_reasons"] == {"Label is not a unique legal option": 1, "Duplicate sample group": 1}
    ignored = ("accepted", "quarantine")
    assert {k: v for k, v in one.items() if k not in ignored} == {k: v for k, v in several.items() if k not in ignored}
    names = sorted(p.name for p in (tmp_path / "one/accepted").iterdir())
    assert names == sorted(p.name for p in (tmp_path / "several/accepted").iterdir()) and len(names) == 2
    for name in ["index.jsonl.gz", *("accepted/" + n for n in names)]:
        assert gzip.open(tmp_path / "one" / name).read() == gzip.open(tmp_path / "several" / name).read()
    assert (tmp_path / "one/quarantine.jsonl").read_text() == (tmp_path / "several/quarantine.jsonl").read_text()
