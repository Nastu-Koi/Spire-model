from copy import deepcopy

import pytest
import torch
from torch import nn

from model.config import ModelConfig, TrainConfig
from model.model import PolicyValue
from model.optim import HybridOptimizer, build_optimizer
from model.policy import SessionPolicy, choose_batch, replay, replay_batch
from model.protocol import CHARACTERS, execution_command
from model.representation import Vocabulary, symbols_from_frames
from model.rollout import numeric_backend, precision_context
from model.testing import SyntheticEngine, demonstration
from model.trainer import Learner


def small_config(**changes):
    values = dict(hidden_size=16, num_heads=2, layers=1, ffn_size=24,
                  local_size=8, local_heads=2, local_layers=1,
                  vocabulary_size=1024, backend="reference")
    values.update(changes)
    return ModelConfig(**values)


def model_and_data(*, checkpoint_layers=False):
    runs = [demonstration(seed="short", count=4, select=2),
            demonstration(seed="long", count=5, select=3)]
    steps = [run["macros"][0]["steps"] for run in runs]
    frames = [step["frame"] for sequence in steps for step in sequence]
    vocabulary = Vocabulary(symbols_from_frames(frames), 1024)
    model = PolicyValue(small_config(checkpoint_layers=checkpoint_layers))
    return model, vocabulary, steps


def objective(outputs):
    return sum(log_prob + value + .01 * entropy for log_prob, value, entropy in outputs)


def test_replay_batch_matches_serial_values_and_gradients():
    batched, vocabulary, steps = model_and_data()
    serial = deepcopy(batched)

    batch_outputs = replay_batch(batched, vocabulary, steps)
    serial_outputs = [replay(serial, vocabulary, sequence) for sequence in steps]
    for actual, expected in zip(batch_outputs, serial_outputs):
        for actual_tensor, expected_tensor in zip(actual, expected):
            torch.testing.assert_close(actual_tensor, expected_tensor, rtol=2e-5, atol=2e-6)

    objective(batch_outputs).backward()
    objective(serial_outputs).backward()
    for (actual_name, actual), (expected_name, expected) in zip(
            batched.named_parameters(), serial.named_parameters()):
        assert actual_name == expected_name
        if actual.grad is not None or expected.grad is not None:
            assert actual.grad is not None and expected.grad is not None
            torch.testing.assert_close(actual.grad, expected.grad, rtol=2e-4, atol=3e-6)


def test_bf16_replay_batch_matches_serial_replay():
    batched, vocabulary, steps = model_and_data()
    serial = deepcopy(batched)
    with precision_context(batched, "bf16"):
        batch_outputs = replay_batch(batched, vocabulary, steps)
    with precision_context(serial, "bf16"):
        serial_outputs = [replay(serial, vocabulary, sequence) for sequence in steps]
    for actual, expected in zip(batch_outputs, serial_outputs):
        for actual_tensor, expected_tensor in zip(actual, expected):
            torch.testing.assert_close(actual_tensor, expected_tensor, rtol=2e-2, atol=2e-2)


def test_choose_batch_forced_bypass_variable_sessions_and_cache_isolation():
    engines = [SyntheticEngine(count=5, select=3, forced_first=True),
               SyntheticEngine(count=6, select=3, forced_first=True)]
    frames = [engine.reset("Ironclad", f"batch-{index}") for index, engine in enumerate(engines)]
    vocabulary = Vocabulary(symbols_from_frames(frames), 1024)
    model = PolicyValue(small_config())
    policies = [SessionPolicy(model, vocabulary), SessionPolicy(model, vocabulary)]
    encoded_batch_sizes = []
    original_encode = model.encode

    def recording_encode(observations, used_vocabulary, *, pad_to=None):
        encoded_batch_sizes.append(len(observations))
        return original_encode(observations, used_vocabulary, pad_to=pad_to)

    model.encode = recording_encode
    forced = choose_batch(policies, frames, teachers=["c0", "c0"])
    assert all(choice.log_prob is None for choice in forced)
    assert encoded_batch_sizes == []

    frames = [engine.send(execution_command(frame, choice.candidate_ref))
              for engine, frame, choice in zip(engines, frames, forced)]
    teachers = [frame["legal"]["candidates"][0]["candidate_ref"] for frame in frames]
    choices = choose_batch(policies, frames, teachers=teachers)
    assert encoded_batch_sizes == [2]
    assert all(choice.value is not None for choice in choices)
    assert policies[0].hidden.data_ptr() != policies[1].hidden.data_ptr()

    frames = [engine.send(execution_command(frame, choice.candidate_ref))
              for engine, frame, choice in zip(engines, frames, choices)]
    teachers = [frame["legal"]["candidates"][0]["candidate_ref"] for frame in frames]
    next_choices = choose_batch(policies, frames, teachers=teachers)
    assert encoded_batch_sizes == [2]
    assert all(choice.value is None for choice in next_choices)


def test_checkpointed_batch_replay_has_gradients_and_uses_batch_encode():
    model, vocabulary, steps = model_and_data(checkpoint_layers=True)
    sizes = []
    original_encode = model.encode

    def recording_encode(observations, used_vocabulary, *, pad_to=None):
        sizes.append(len(observations))
        return original_encode(observations, used_vocabulary, pad_to=pad_to)

    model.encode = recording_encode
    objective(replay_batch(model, vocabulary, steps, pad_to=16)).backward()
    assert sizes == [2]
    assert model.blocks[0].attention.qkv.weight.grad is not None


def test_hybrid_optimizer_assigns_only_block_linear_weights_to_muon():
    model = PolicyValue(small_config())
    config = TrainConfig(optimizer="muon_adamw", precision="no", muon_ns_steps=2)
    optimizer = build_optimizer(model, config)
    assert isinstance(optimizer, HybridOptimizer)
    muon_ids = {id(parameter) for group in optimizer.muon.param_groups for parameter in group["params"]}
    adamw_ids = {id(parameter) for group in optimizer.adamw.param_groups for parameter in group["params"]}
    assert not muon_ids & adamw_ids
    assert muon_ids | adamw_ids == {id(parameter) for parameter in model.parameters()}

    expected = {id(module.weight) for name, module in model.named_modules()
                if isinstance(module, nn.Linear)
                and (name.startswith("blocks.") or name.startswith("encoder.local_blocks."))
                and (".attention." in name or ".ffn." in name)}
    assert muon_ids == expected
    embedding_ids = {id(module.weight) for module in model.modules() if isinstance(module, nn.Embedding)}
    assert embedding_ids <= adamw_ids

    before_muon = {id(parameter): parameter.detach().clone() for group in optimizer.muon.param_groups
                   for parameter in group["params"]}
    before_adamw = model.value.weight.detach().clone()
    loss = sum(parameter.square().sum() for parameter in model.parameters())
    loss.backward()
    optimizer.step()
    assert any(not torch.equal(parameter, before_muon[id(parameter)])
               for group in optimizer.muon.param_groups for parameter in group["params"])
    assert not torch.equal(model.value.weight, before_adamw)
    restored = build_optimizer(model, config)
    restored.load_state_dict(optimizer.state_dict())
    assert len(restored.state) == len(optimizer.state)


def _on_policy_runs(model, vocabulary):
    runs = []
    with torch.no_grad():
        for index, character in enumerate(CHARACTERS):
            source = demonstration(character=character, seed=f"ppo-{index}", count=3, select=1)
            macro = deepcopy(source["macros"][0])
            log_prob, value, _ = replay(model, vocabulary, macro["steps"])
            macro.update({"old_log_prob": float(log_prob), "old_value": float(value),
                          "reward": 1., "return": 1., "advantage": 1. - float(value)})
            runs.append({"schema": source["schema"], "source": "on_policy", "sampling": "full_distribution",
                         "status": "complete", "victory": True, "ascension": 10, "character": character,
                         "seed": source["seed"], "run_id": source["run_id"], "contract": source["contract"],
                         "policy_version": 0, "precision": "no", "vocabulary_hash": vocabulary.digest,
                         "numeric_backend": numeric_backend(model), "macros": [macro]})
    return runs


def test_ppo_microbatch_smoke_uses_one_logical_update():
    source_runs = [demonstration(character=character, seed=f"ppo-{index}", count=3, select=1)
                   for index, character in enumerate(CHARACTERS)]
    frames = [step["frame"] for run in source_runs for macro in run["macros"] for step in macro["steps"]]
    vocabulary = Vocabulary(symbols_from_frames(frames), 1024)
    model = PolicyValue(small_config())
    runs = _on_policy_runs(model, vocabulary)
    config = TrainConfig(optimizer="muon_adamw", muon_ns_steps=2, precision="no", ppo_epochs=1,
                         logical_batch_size=5, microbatch_size=5, token_buckets=[16, 32],
                         action_buckets=[4, 8], token_budget=256, pair_budget=4096)
    learner = Learner(model, vocabulary, config)
    before_calls = model.encoder_calls
    result = learner.ppo(runs)
    assert learner.updates == 1
    assert result[0]["updates"] == 1
    assert result[0]["old_replay_max_error"] < 1e-4
    assert model.encoder_calls - before_calls == 2  # check_old_policy + the PPO microbatch


def test_value_warmup_leaves_muon_parameters_unchanged():
    source_runs = [demonstration(character=character, seed=f"value-{index}", count=3, select=1)
                   for index, character in enumerate(CHARACTERS)]
    frames = [step["frame"] for run in source_runs for macro in run["macros"] for step in macro["steps"]]
    vocabulary = Vocabulary(symbols_from_frames(frames), 1024)
    model = PolicyValue(small_config())
    runs = _on_policy_runs(model, vocabulary)
    config = TrainConfig(optimizer="muon_adamw", muon_ns_steps=2, precision="no",
                         logical_batch_size=5, microbatch_size=5, token_buckets=[16, 32],
                         action_buckets=[4, 8], token_budget=256, pair_budget=4096)
    learner = Learner(model, vocabulary, config)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    metrics = learner.value_warmup(runs)
    assert len(metrics) == 1
    assert learner.updates == 1
    assert any(not torch.equal(parameter, before[name])
               for name, parameter in model.named_parameters() if name.startswith("value."))
    assert all(torch.equal(parameter, before[name])
               for name, parameter in model.named_parameters() if not name.startswith("value."))
