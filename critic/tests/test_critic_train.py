"""End-to-end training on synthetic runs: the saved critic scores frames, continues
as the next version, and shares the policy's body and optimizer grouping."""

import json
from dataclasses import asdict

import pytest
import torch

from critic.batches import pack, parse
from critic.config import Budgets, CriticConfig, TrainingConfig
from critic.data import decisions
from critic.model import Critic, Scorer, load
from critic.tests.conftest import frames_of
from critic.train import build_optimizer, main, set_head_bias, trainable
from model.config import ModelConfig
from model.model import PolicyValue
from model.representation import Vocabulary


def test_training_saves_a_critic_that_scores_frames_and_continues(runs_dir, runs, profile, checkpoint, tmp_path):
    saved = tmp_path / "critic-1"
    report = main([str(runs_dir), "--checkpoint", str(checkpoint), "--config", str(profile), "--device", "cpu",
                   "--holdout", "0.5", "--save", str(saved), "--report", str(tmp_path / "report.json")])
    assert report["data"]["runs"] == 6 and report["data"]["split"] == dict(
        holdout=0.5, train=12, train_seeds=3, test=12, test_seeds=3)
    assert set(report["baselines"]) == {"constant", "logistic_act_floor", "logistic_state"}
    assert report["baselines"]["constant"]["win"]["explained_variance"] <= 0.0
    assert set(report["critic"]) == {"win", "act", "calibration"} and len(report["history"]) == 2
    assert {row["key"] for row in report["critic"]["calibration"]["by_act"]["act"]} == {1, 2, 3}
    assert report["model"] == dict(json.loads((tmp_path / "report.json").read_text())["model"])
    assert report["model"]["version"] == 1 and report["model"]["init"] == "checkpoint"

    scorer = load(saved)
    manifest = scorer.manifest
    assert manifest["model"] == asdict(ModelConfig.tiny()) and manifest["version"] == 1
    assert manifest["policy"]["vocabulary_hash"] == scorer.vocabulary.digest
    assert manifest["policy"]["policy_version"] == 3 and manifest["training"]["precision"] == "no"
    assert manifest["data"]["decisions"] == 24 and manifest["report"]["critic"] == report["critic"]
    # Only the shared body and the head are saved: the policy's decision heads are not.
    names = set(torch.load(saved / "weights.pt", weights_only=True))
    assert "head.weight" in names and all(not n.startswith(("body.query", "body.key", "body.value")) for n in names)
    frames = frames_of(runs[0])[:5]
    p_win, p_act = scorer.predict(frames)
    assert len(p_win) == len(p_act) == 5 and all(0 < p < 1 for p in p_win + p_act)
    assert scorer.score([]).shape == (0, 2)
    scored = scorer.score(parse(frames))
    assert scored[:, 0].tolist() == pytest.approx(p_win) and scored[:, 1].tolist() == pytest.approx(p_act)

    # The next iteration continues from the saved critic: same vocabulary, next version.
    continued = tmp_path / "critic-2"
    report = main([str(runs_dir), "--pretrained", str(saved), "--config", str(profile), "--device", "cpu",
                   "--holdout", "0.5", "--epochs", "1", "--no-baselines", "--save", str(continued)])
    assert report["model"]["version"] == 2 and report["model"]["pretrained"] == str(saved)
    assert load(continued).manifest["vocabulary"] == manifest["vocabulary"]
    assert "baselines" not in report

    # A random body takes only the vocabulary from the checkpoint.
    report = main([str(runs_dir), "--checkpoint", str(checkpoint), "--config", str(profile), "--device", "cpu",
                   "--holdout", "0.5", "--epochs", "1", "--init", "random", "--no-baselines"])
    assert report["model"]["init"] == "random"
    with pytest.raises(SystemExit):
        main([str(runs_dir), "--config", str(profile), "--device", "cpu"])
    other = tmp_path / "other.json"
    other.write_text(json.dumps(dict(model=asdict(ModelConfig(vocabulary_size=4096, backend="reference")))))
    with pytest.raises(ValueError, match="differs from the checkpoint"):
        main([str(runs_dir), "--checkpoint", str(checkpoint), "--config", str(other), "--device", "cpu",
              "--holdout", "0.5", "--no-baselines"])


def test_the_profile_gives_the_model_and_the_critic_section(profile):
    config = CriticConfig.load(profile)
    assert config.model == ModelConfig.tiny()
    assert (config.training.epochs, config.training.logical_batch_size, config.training.init) == (2, 4, "checkpoint")
    with pytest.raises(ValueError):
        TrainingConfig(init="policy")
    with pytest.raises(ValueError):
        TrainingConfig(holdout=1.0)


def test_the_body_is_the_policy_weights_and_the_optimizer_groups_are_the_policy_s():
    config = ModelConfig.tiny()
    policy = PolicyValue(config)
    net = Critic(config)
    net.load_body(policy.state_dict())
    for name, tensor in policy.state_dict().items():
        if name.startswith(("encoder.", "blocks.", "relation.", "final_norm.")):
            assert torch.equal(net.state()["body." + name], tensor)
    assert not any(name.startswith(("body.query", "body.key", "body.value")) for name in net.state())
    with pytest.raises(ValueError, match="unexpected"):
        net.load_state(dict(net.state(), **{"body.query.weight": policy.query.weight}))
    with pytest.raises(ValueError, match="missing"):
        net.load_state({k: v for k, v in net.state().items() if k != "head.weight"})

    optimizer = build_optimizer(net, TrainingConfig(precision="no"))
    names = {group["name"] for group in optimizer.param_groups}
    assert names == {"muon", "embedding_no_decay", "head_decay", "head_no_decay", "backbone_decay", "backbone_no_decay"}
    grouped = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert grouped == {id(p) for _, p in net.trainable()}
    assert not any(id(p) in grouped for p in (net.body.query.weight, net.body.key.weight))
    muon = next(group for group in optimizer.param_groups if group["name"] == "muon")
    linears = [m for n, m in net.body.named_modules() if isinstance(m, torch.nn.Linear)
               and (".attention." in n or ".ffn." in n) and n.startswith(("blocks.", "encoder.local_blocks."))]
    assert len(muon["params"]) == len(linears) and all(p.ndim == 2 for p in muon["params"])
    tables = next(group for group in optimizer.param_groups if group["name"] == "embedding_no_decay")
    assert len(tables["params"]) == 4 and tables["lr"] == 1e-3 and tables["weight_decay"] == 0.0

    # A linear probe trains the head alone, with AdamW alone; `train_blocks` adds the top layers.
    probe = TrainingConfig(precision="no", train_body=False)
    assert [name for name, _ in trainable(net, probe)] == ["head.weight", "head.bias"]
    assert {group["name"] for group in build_optimizer(net, probe).param_groups} == {"head_decay", "head_no_decay"}
    top = [name for name, _ in trainable(net, TrainingConfig(precision="no", train_blocks=1))]
    assert all(name.startswith(("head.", "body.final_norm.", "body.blocks.1.")) for name in top)
    assert any(name.startswith("body.blocks.1.") for name in top) and not any("blocks.0." in name for name in top)
    with pytest.raises(ValueError):
        TrainingConfig(train_blocks=0)


def test_the_untrained_head_predicts_the_base_rates(runs):
    config = ModelConfig.tiny()
    frames = frames_of(runs[0])[:3]
    vocabulary = Vocabulary.from_frames(frames, config)
    net = Critic(config)
    set_head_bias(net, [0.2, 0.6])
    scored = Scorer(net, vocabulary, precision="no").score(parse(frames))
    assert scored.shape == (3, 2)
    assert scored[:, 0].tolist() == pytest.approx([0.2] * 3, abs=1e-5)
    assert scored[:, 1].tolist() == pytest.approx([0.6] * 3, abs=1e-5)
    # Frames are packed under the budgets: a tight one splits them, every frame once.
    parts = pack(parse(frames), vocabulary, Budgets(microbatch_size=2, token_buckets=[64, 128], action_buckets=[8]))
    assert sorted(p for positions, _ in parts for p in positions) == [0, 1, 2] and len(parts) == 2
    rows = list(decisions(runs[0]))
    assert len(rows) == 4
