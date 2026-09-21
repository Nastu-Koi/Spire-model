from copy import deepcopy
import json

import pytest
import torch

from model.checkpoint import load_model, restore_training, save_checkpoint
from model.config import TrainConfig
from model.data import audit_files, microbatches, samples, split_runs, validate_run
from model.policy import replay
from model.protocol import CHARACTERS, ProtocolError
from model.rewards import MilestoneLedger
from model.rollout import RolloutRunner
from model.testing import SyntheticEngine, demonstration
from model.trainer import Learner, ppo_objective


def test_ledger_caps_dedup_restore_and_boss_exclusivity():
    ledger = MilestoneLedger()
    events = [{"type": "encounter_completed", "act": 1, "kind": "elite", "result": "victory", "encounter_id": str(i)} for i in range(10)]
    assert ledger.apply(events) == pytest.approx(.1)
    assert ledger.apply(events) == 0
    restored = MilestoneLedger.restore(json.loads(json.dumps(ledger.state_dict())))
    assert restored.apply(events) == 0
    boss = {"type": "encounter_completed", "act": 1, "kind": "boss", "result": "victory", "encounter_id": "boss", "final_in_act": True}
    assert restored.apply([boss, boss]) == .1
    assert restored.components == {"combat": pytest.approx(.1), "boss": .1, "victory": 0.}
    win = {"type": "run_completed", "victory": True, "act": 3, "final_boss_defeated": True}
    assert restored.apply([win, win]) == 1
    with pytest.raises(ProtocolError):
        MilestoneLedger().apply([dict(win, act=2)])


def test_rollout_reward_assignment_returns_and_old_replay(setup):
    model, vocab = setup
    runner = RolloutRunner(model, vocab)
    run = runner.run(SyntheticEngine(5, 3), "Ironclad", "test")
    assert run["status"] == "complete", run.get("error")
    assert len(run["macros"]) == 2
    a, b = run["macros"]
    assert a["reward"] == .005  # Reward after forced STOP belongs to preceding macro.
    assert a["return"] == b["return"] + .005
    assert run["automatic_steps"] == 1
    validate_run(run, on_policy=True)
    for macro in run["macros"]:
        new, value, _ = replay(model, vocab, macro["steps"])
        assert float(new.detach()) == pytest.approx(macro["old_log_prob"], abs=1e-6)
        assert float(value.detach()) == pytest.approx(macro["old_value"], abs=1e-6)


def test_unresolved_never_produces_returns(setup):
    model, vocab = setup
    run = RolloutRunner(model, vocab, max_steps=1).run(SyntheticEngine(5, 3), "Ironclad", "timeout")
    assert run["status"] == "unresolved"
    assert "return" not in run["macros"][0]
    with pytest.raises(ProtocolError, match="Incomplete"):
        validate_run(run, on_policy=True)


def test_terminal_at_exact_step_limit_is_complete(setup):
    model, vocab = setup
    run = RolloutRunner(model, vocab, max_steps=5).run(SyntheticEngine(5, 3), "Ironclad", "exact-limit")
    assert run["status"] == "complete", run.get("error")


def test_native_waits_do_not_consume_action_budget(setup):
    class WaitingEngine(SyntheticEngine):
        pending = None

        def send(self, command):
            if command["cmd"] == "advance_to_boundary":
                result, self.pending = self.pending, None
                return result
            self.pending = super().send(command)
            waiting = deepcopy(self.pending)
            waiting.update(boundary="waiting", events=[])
            waiting["legal"] = {"candidates": []}
            return waiting

    model, vocab = setup
    run = RolloutRunner(model, vocab, max_steps=5).run(WaitingEngine(5, 3), "Ironclad", "native-waits")
    assert run["status"] == "complete", run.get("error")
    validate_run(run, on_policy=True)


def test_checkpoint_cli_preserves_initialization_and_records_dataset(setup, demo, tmp_path):
    from model.cli import _data_metadata, _learner, _save
    model, vocab = setup
    learner = Learner(model, vocab, TrainConfig(precision="no"))
    _save(tmp_path / "init", learner, {"engine_contract": demo["contract"], "initialization": "test"})
    restored, _ = _learner(tmp_path / "init", "cpu")
    _save(tmp_path / "trained", restored, _data_metadata([demo]))
    manifest = json.loads((tmp_path / "trained" / "manifest.json").read_text())
    progress = manifest["progress"]
    assert progress["engine_contract"] == demo["contract"]
    assert progress["training_engine_contracts"] == [demo["contract"]]
    assert progress["reward_version"] == "milestones-v1"
    assert progress["initialization"] == "test"
    assert len(progress["training_seed_pool_hash"]) == 64


def test_forced_only_run_has_no_training_samples_or_model_calls(setup):
    class ForcedEngine(SyntheticEngine):
        def frame(self):
            frame = super().frame()
            if frame["boundary"] == "decision":
                frame["legal"]["candidates"] = frame["legal"]["candidates"][:1]
            return frame

    model, vocab = setup
    run = RolloutRunner(model, vocab).run(ForcedEngine(5, 3), "Ironclad", "all-forced")
    assert run["status"] == "complete", run.get("error")
    assert run["macros"] == []
    assert run["initial_reward"] == pytest.approx(1.005)
    assert model.encoder_calls == model.decoder_calls == 0


def test_value_warmup_does_not_change_actor(setup):
    model, vocab = setup
    learner = Learner(model, vocab, TrainConfig(precision="no"))
    runs = [RolloutRunner(model, vocab).run(SyntheticEngine(5, 3), c, "warmup") for c in CHARACTERS]
    actor = {name: p.detach().clone() for name, p in model.named_parameters() if not name.startswith("value.")}
    before_value = model.value.weight.detach().clone()
    learner.value_warmup(runs)
    for name, p in model.named_parameters():
        if name in actor:
            torch.testing.assert_close(actor[name], p, atol=0, rtol=0)
    assert not torch.equal(before_value, model.value.weight)


def test_ppo_complete_trajectory_clip_is_once():
    lp = torch.tensor([.1, .2], requires_grad=True)
    loss, _, _ = ppo_objective(lp.sum(), 0., 2., .2)
    assert loss.item() == pytest.approx(-2.4)
    assert loss.item() != pytest.approx(-2 * (lp.exp().clamp(.8, 1.2)).sum().item())


def test_bootstrap_learns_and_all_optimizer_steps_are_logical(setup, demo):
    model, vocab = setup
    learner = Learner(model, vocab, TrainConfig(precision="no", backbone_lr=.002, head_lr=.002,
                                               logical_batch_size=32, microbatch_size=1))
    before = -float(replay(model, vocab, demo["macros"][0]["steps"])[0].detach())
    learner.bootstrap([demo], epochs=6)
    after = -float(replay(model, vocab, demo["macros"][0]["steps"])[0].detach())
    assert after < before
    assert learner.updates == 6


def test_microbatch_partition_preserves_update(setup, demo):
    model, vocab = setup
    other = deepcopy(model)
    a = Learner(model, vocab, TrainConfig(precision="no", logical_batch_size=32, microbatch_size=1))
    b = Learner(other, vocab, TrainConfig(precision="no", logical_batch_size=32, microbatch_size=4))
    items = samples([demo])
    a._optimize(items, "bootstrap")
    b._optimize(items, "bootstrap")
    for p, q in zip(model.parameters(), other.parameters()):
        torch.testing.assert_close(p, q, atol=0, rtol=0)
    assert a.updates == b.updates == 1


def test_checkpoint_restores_weights_optimizer_rng_and_scheduler(setup, demo, tmp_path):
    model, vocab = setup
    learner = Learner(model, vocab, TrainConfig(precision="no"))
    learner.bootstrap([demo])
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(checkpoint, model, vocab, learner.optimizer, learner.scheduler,
                    training=learner.config, progress={"policy_version": learner.policy_version})
    expected_random = torch.rand(3)
    loaded, vocabulary, manifest = load_model(checkpoint)
    restored = Learner(loaded, vocabulary, TrainConfig(**manifest["training"]))
    restore_training(checkpoint, restored.optimizer, restored.scheduler)
    torch.testing.assert_close(torch.rand(3), expected_random)
    before = replay(model, vocab, demo["macros"][0]["steps"])[0]
    after = replay(loaded, vocabulary, demo["macros"][0]["steps"])[0]
    torch.testing.assert_close(before, after, atol=0, rtol=0)
    learner._optimize(samples([demo]), "bootstrap")
    restored._optimize(samples([demo]), "bootstrap")
    for a, b in zip(model.parameters(), loaded.parameters()):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert learner.scheduler.state_dict() == restored.scheduler.state_dict()


def test_ppo_round_is_frozen_and_equal_characters(setup):
    model, vocab = setup
    learner = Learner(model, vocab, TrainConfig(precision="no", ppo_epochs=1))
    runner = RolloutRunner(model, vocab, version=0)
    runs = [runner.run(SyntheticEngine(5, 3), c, "round") for c in CHARACTERS]
    metrics = learner.ppo(runs)
    assert metrics[0]["old_replay_max_error"] < 1e-5
    assert learner.policy_version == 1
    with pytest.raises(ProtocolError, match="differs"):
        learner.ppo(runs)
    with pytest.raises(ProtocolError, match="five"):
        samples(runs[:1], ppo=True)
    demonstration_run = deepcopy(runs[0])
    demonstration_run["source"] = "demonstration"
    with pytest.raises(ProtocolError, match="on-policy"):
        validate_run(demonstration_run, on_policy=True)


def test_run_weight_does_not_normalize_by_length(setup):
    model, vocab = setup
    runner = RolloutRunner(model, vocab)
    runs = [runner.run(SyntheticEngine(5, 3), c, "round") for c in CHARACTERS]
    items = samples(runs, ppo=True, horizon_scale=100)
    assert all(x.weight == .002 for x in items)


def test_audit_isolates_missing_legal_labels_and_unverified_teacher(demo, tmp_path):
    accepted = tmp_path / "accepted.jsonl"
    quarantine = tmp_path / "quarantine.jsonl"
    good = tmp_path / "good.json"
    good.write_text(json.dumps(demo))
    bad_run = deepcopy(demo)
    bad_run["macros"][0]["steps"][0]["candidate_ref"] = "absent"
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(bad_run))
    counts = audit_files([good, bad], accepted, quarantine)
    assert counts["accepted_runs"] == counts["quarantined_runs"] == 1
    assert "legal" in json.loads(quarantine.read_text())["reason"]
    demo["teacher_visibility"] = "unverified"
    with pytest.raises(ProtocolError, match="visibility"):
        validate_run(demo)


def test_seed_split_keeps_fragments_together():
    runs = [{"seed": str(i//3), "run_id": str(i)} for i in range(100)]
    train, validation = split_runs(runs)
    assert not {r["seed"] for r in train} & {r["seed"] for r in validation}
