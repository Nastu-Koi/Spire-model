"""End-to-end training on a few fight outcomes; a saved model rebuilds from its manifest and can be continued."""

import gzip
import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from combat_outcome.data import SCHEMA, entities, load_data, split_runs
from combat_outcome.frames import frame_for
from combat_outcome.model import load_model
from combat_outcome.train import main
from model.config import ModelConfig
from model.representation import observation

FIXTURE = Path(__file__).resolve().parents[2] / "combat_outcome/tests/fixtures/rest_site.json"


def deck():
    return [e for e in json.loads(FIXTURE.read_text())["public"]["entities"]
            if e.get("entity_type") in {"player", "card", "relic"}]


def fight(seed, index, **values):
    encounter = ("ENCOUNTER.A", "ENCOUNTER.B")[index % 2]
    row = dict(schema=SCHEMA, source="summary_anchor", seed=seed, character="SILENT", act=1, kind="regular",
               encounter=encounter, entities=deck(), solver=dict(budget_ms=1000, potions=True), result="win",
               max_hp=70, start_hp=60, end_hp=45 - index, hp_lost=15 + index)
    return dict(row, **values)


@pytest.fixture
def profile(tmp_path):
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(dict(model=asdict(ModelConfig.tiny()),
                                    training=dict(epochs=1, batch_size=4, loader_workers=0))))
    return path


@pytest.fixture
def labels(tmp_path):
    """Three seeds, four fights each: one heals more than it costs, one is lost."""
    path = tmp_path / "anchors" / "combat-outcomes.jsonl.gz"
    path.parent.mkdir()
    with gzip.open(path, "wt") as sink:
        for seed in ("s0", "s1", "s2"):
            for i in range(4):
                extra = dict(end_hp=66, hp_lost=-6) if i == 2 else \
                    dict(end_hp=0, hp_lost=60, result="loss", kind="boss") if i == 3 else {}
                sink.write(json.dumps(fight(seed, i, **extra)) + "\n")
    return path


def test_pretraining_saves_a_model_that_new_labels_continue(labels, profile, tmp_path):
    saved = tmp_path / "pretrained"
    main([str(labels), "--config", str(profile), "--device", "cpu", "--summary-entity", "--save", str(saved)])
    report = json.loads((labels.parent / "model_report.json").read_text())
    assert (report["schema"], report["rows"], report["runs"], report["test_runs"]) == (SCHEMA, 12, 3, 1)
    assert report["sources"] == ["summary_anchor"] and report["loss_rate"] == 0.25
    assert set(report["baselines"]) == {"encounter_only", "ridge_summary", "ridge_summary_bag"}
    assert set(report["entity_model"]["test"]) == {"all", "regular", "boss", "defeat"}

    net, vocabulary, manifest = load_model(saved)
    assert manifest["model"] == asdict(ModelConfig.tiny()) and manifest["training"]["summary_entity"]
    assert net.body.config == ModelConfig.tiny() and not net.training
    with torch.no_grad():
        frac, logit = net([observation(frame_for(deck(), "ENCOUNTER.A", summary=True))], vocabulary)
    # A signed fraction: a fight can give back more HP than it takes.
    assert -1 < float(frac) < 1 and torch.isfinite(logit).all()

    # Fights recorded later, in the recorder's layout, continue from the saved model and its vocabulary.
    online = tmp_path / "online"
    for seed in ("r0", "r1"):
        (online / seed).mkdir(parents=True)
        (online / seed / "combats.jsonl").write_text(
            "".join(json.dumps(fight(seed, i, source="rollout")) + "\n" for i in range(2)))
    tuned = tmp_path / "tuned"
    main([str(online), "--config", str(profile), "--device", "cpu", "--summary-entity", "--pretrained", str(saved),
          "--no-baselines", "--save", str(tuned), "--report", str(tmp_path / "tuned.json")])
    assert load_model(tuned)[2]["vocabulary"] == manifest["vocabulary"]
    assert json.loads((tmp_path / "tuned.json").read_text())["sources"] == ["rollout"]
    with pytest.raises(ValueError, match="different profile"):
        main([str(online), "--config", str(profile), "--device", "cpu", "--pretrained", str(saved)])


def test_the_ascension_a_fight_was_fought_at_is_model_input():
    def digest(ascension):
        state = [dict(e, ascension=ascension) if e["entity_type"] == "player" else e for e in deck()]
        return observation(frame_for(state, "ENCOUNTER.A")).digest

    assert digest(0) != digest(10)


def test_rows_keep_entities_packed_and_split_by_seed(labels):
    rows = load_data(labels)
    assert isinstance(rows[0]["packed"], bytes) and entities(rows[0]) == deck()
    assert [r["y"] for r in rows[:4]] == [15.0, 16.0, -6.0, 60.0] and rows[3]["lost"]
    train, test, runs, held = split_runs(rows, 0, 0.34)
    assert len(runs) == 3 and len(held) == 1 and not {r["run"] for r in train} & {r["run"] for r in test}


def test_one_seed_cannot_be_split_and_labels_must_add_up(tmp_path):
    single = tmp_path / "single.jsonl"
    single.write_text("".join(json.dumps(fight("only", i)) + "\n" for i in range(4)))
    rows = load_data(single)
    with pytest.raises(ValueError, match="two independent"):
        split_runs(rows, 0)
    broken = tmp_path / "broken.jsonl"
    broken.write_text(json.dumps(fight("s", 0, hp_lost=99)) + "\n")
    with pytest.raises(ValueError, match="Inconsistent"):
        load_data(broken)
