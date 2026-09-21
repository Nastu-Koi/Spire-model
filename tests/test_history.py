import json
from pathlib import Path

import pytest
import torch

from model import cli
from model.checkpoint import load_model, save_checkpoint
from model.history import TrainingHistory
from model.monitor import snapshot


def test_current_replacement_preserves_last_good_checkpoint_on_failure(setup, tmp_path, monkeypatch):
    model, vocabulary = setup
    target = tmp_path / "current"
    save_checkpoint(target, model, vocabulary)
    before = model.bos.detach().clone()
    with torch.no_grad():
        model.bos.add_(1)
    original_save = torch.save
    def fail(*_, **__):
        raise OSError("disk full")
    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError, match="disk full"):
        save_checkpoint(target, model, vocabulary, overwrite=True)
    loaded, _, _ = load_model(target)
    assert torch.equal(loaded.bos, before)
    monkeypatch.setattr(torch, "save", original_save)
    save_checkpoint(target, model, vocabulary, overwrite=True)
    loaded, _, _ = load_model(target)
    assert torch.equal(loaded.bos, model.bos)
    assert [p.name for p in tmp_path.iterdir()] == ["current"]


def test_bootstrap_epochs_saved_and_resume_appends(demo, tmp_path):
    data = tmp_path / "data.json"
    data.write_text(json.dumps(demo))
    output = tmp_path / "bootstrap"
    assert cli.main(["bootstrap", "--data", str(data), "--tiny", "--precision", "no",
                     "--epochs", "2", "--output", str(output)]) == 0
    first = [json.loads(x) for x in (output / "history.jsonl").read_text().splitlines()]
    assert [r["round"] for r in first] == [1, 2]
    assert cli.main(["bootstrap", "--data", str(data), "--checkpoint", str(output / "current"),
                     "--epochs", "1", "--output", str(output)]) == 0
    records = [json.loads(x) for x in (output / "history.jsonl").read_text().splitlines()]
    assert records[:2] == first
    assert records[-1]["round"] == 3
    assert records[-1]["metrics"]["updates"] > first[-1]["metrics"]["updates"]
    manifest = json.loads((output / "current/manifest.json").read_text())
    assert manifest["progress"]["completed_bootstrap_epochs"] == 3
    assert not list(output.glob("checkpoint-*"))
    monitored = snapshot(tmp_path)
    assert monitored[0]["history"] == records
    assert monitored[0]["status"]["state"] == "completed"


def test_history_lock_and_partial_line(tmp_path):
    history = TrainingHistory(tmp_path, "bootstrap", {})
    with pytest.raises(ValueError, match="Another trainer"):
        TrainingHistory(tmp_path, "ppo", {})
    history.append(1, {"loss": 1.})
    with (tmp_path / "history.jsonl").open("a") as stream:
        stream.write('{"partial":')
    assert len(snapshot(tmp_path)[0]["history"]) == 1
    history.status("interrupted")
    resumed = TrainingHistory(tmp_path, "bootstrap", {})
    resumed.append(2, {"loss": .5})
    assert [r["round"] for r in snapshot(tmp_path)[0]["history"]] == [1, 2]
    resumed.status("completed")


def test_legacy_training_configuration_and_progress_are_readable(setup, tmp_path):
    from model.config import TrainConfig
    model, vocabulary = setup
    path = tmp_path / "legacy"
    save_checkpoint(path, model, vocabulary, training=TrainConfig(precision="no"),
                    progress={"completed_bc_epochs": 5, "policy_version": 5})
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["training"]["bc_coef"] = manifest["training"].pop("bootstrap_coef")
    manifest_path.write_text(json.dumps(manifest))
    learner, _ = cli._learner(path, "cpu")
    assert learner.config.bootstrap_coef == 0
    assert learner.checkpoint_metadata["completed_bootstrap_epochs"] == 5
    assert "completed_bc_epochs" not in learner.checkpoint_metadata
