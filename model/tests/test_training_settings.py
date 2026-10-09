"""BC numerical scale, complete logical batches, and explicit restart/resume."""

import copy
import gzip
import json
import random
from collections import Counter
from dataclasses import replace

import pytest
import torch

from model.checkpoint import load_model, save_checkpoint
from model.cli import _learner, parser, vocabulary_for
from model.config import ModelConfig, TrainConfig
from model.data import RunShards, samples
from model.loader import batches, stream
from model.model import PolicyValue
from model.optim import TRAINING_VERSION, BootstrapSchedule
from model.representation import Vocabulary
from model.testing import demonstration
from model.trainer import Learner


def setup():
    runs = [demonstration("Ironclad", f"numerical-{i}", 4 + i, 2) for i in range(2)]
    config = ModelConfig.tiny()
    vocabulary = vocabulary_for(runs, config)
    torch.manual_seed(7)
    model = PolicyValue(config)
    training = TrainConfig(optimizer="adamw", precision="no", loader_workers=0,
                           logical_batch_size=3, microbatch_size=3, bootstrap_warmup_updates=0,
                           weight_decay=0, max_grad_norm=1e6, update_stats_every=1)
    return runs, vocabulary, model, training


def gradients(model):
    return torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])


def test_dataset_size_and_micro_splitting_do_not_shrink_a_batch_update():
    runs, vocabulary, base, training = setup()
    original = samples(runs)
    grads, parameters, updates = [], [], []
    for duplication, micro in ((1, 3), (10, 3), (10, 1)):
        learner = Learner(copy.deepcopy(base), vocabulary, replace(training, microbatch_size=micro))
        # Same first logical batch; the rest of the dataset is duplicated.
        items = [replace(item, weight=item.weight / duplication) for item in original[:3]]
        result = learner._optimize(batches(items, vocabulary, learner.config), "bootstrap",
                                   bootstrap_samples=len(original) * duplication)
        grads.append(gradients(learner.model))
        parameters.append(torch.cat([p.detach().flatten() for p in learner.model.parameters()]))
        updates.append(result["parameter_updates"])
        assert result["updates"] == 1 and result["parameter_updates"]
    for actual in grads[1:]:
        torch.testing.assert_close(actual, grads[0], rtol=1e-4, atol=2e-6)
    torch.testing.assert_close(parameters[1], parameters[0], rtol=1e-4, atol=2e-6)
    # Splitting changes FP32 summation order. The mathematically zero key-bias
    # gradient can change by ~1e-9 and Adam amplifies it; check the actual group
    # update scale rather than requiring identical redundant bias coordinates.
    for group, expected in updates[0].items():
        assert updates[2][group]["mean_delta_l2"] == pytest.approx(expected["mean_delta_l2"], rel=.01)


def test_short_tail_keeps_nominal_batch_divisor():
    runs, vocabulary, base, training = setup()
    # Repeating the same macro ensures gradient direction is identical.
    item = samples(runs)[0]
    values = []
    for count in (1, 3):
        learner = Learner(copy.deepcopy(base), vocabulary, training)
        logical = batches([copy.copy(item) for _ in range(count)], vocabulary, training)
        learner._optimize(logical, "bootstrap", bootstrap_samples=12)
        values.append(gradients(learner.model))
    torch.testing.assert_close(values[0] * 3, values[1], rtol=1e-4, atol=2e-6)


def sharded(tmp_path):
    from model.dataset_index import HEADER, index_row

    rows, runs = [], []
    for i in range(5):
        run = demonstration("Ironclad", f"shard-{i}", 4, 2)
        name = f"{i}.jsonl.gz"
        with gzip.open(tmp_path / name, "wt") as out:
            out.write(json.dumps(run) + "\n")
        rows.append(index_row(run, name, 0))
        runs.append(run)
    with gzip.open(tmp_path / "index.jsonl.gz", "wt") as out:
        out.write(json.dumps(HEADER) + "\n")
        out.writelines(json.dumps(row) + "\n" for row in rows)
    return RunShards(tmp_path), runs


@pytest.mark.parametrize("workers", [0, 1])
def test_shard_windows_leave_only_one_epoch_tail(tmp_path, workers):
    shards, runs = sharded(tmp_path)
    vocabulary = vocabulary_for(runs, ModelConfig.tiny())
    training = TrainConfig(precision="no", loader_workers=workers, logical_batch_size=3, microbatch_size=2)
    logicals = list(stream(shards, vocabulary, training, window_shards=2, shuffle=False))
    sizes = [sum(len(items) for items, _ in logical) for logical in logicals]
    assert sizes == [3, 3, 3, 1]
    actual = Counter((item.run_id, item.phase) for logical in logicals for items, _ in logical for item in items)
    assert actual == Counter((item.run_id, item.phase) for item in samples(runs))
    assert sum(item.weight for logical in logicals for items, _ in logical for item in items) == pytest.approx(1)


@pytest.mark.parametrize("optimizer", ["adamw", "muon_adamw"])
def test_every_lookup_table_trains_at_the_embedding_rate(optimizer):
    _, vocabulary, model, training = setup()
    learner = Learner(model, vocabulary, replace(training, optimizer=optimizer, embedding_lr=0.01))
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    tables = {name for name, module in model.named_modules() if isinstance(module, torch.nn.Embedding)}
    group = next(g for g in learner.optimizer.param_groups if g["name"] == "embedding_no_decay")
    assert {names[id(parameter)] for parameter in group["params"]} == {name + ".weight" for name in tables}
    assert group["initial_lr"] == 0.01 and group["weight_decay"] == 0


@pytest.mark.parametrize("optimizer", ["adamw", "muon_adamw"])
def test_resume_continues_warmup_and_weights_only_restarts_it(tmp_path, optimizer):
    runs, vocabulary, model, training = setup()
    training = replace(training, optimizer=optimizer, bootstrap_warmup_updates=6, logical_batch_size=4)
    learner = Learner(model, vocabulary, training)
    first = learner.bootstrap(runs)[0]
    assert first["learning_rates"]["embedding_no_decay"] == pytest.approx(training.embedding_lr / 6)
    saved = tmp_path / "saved"
    save_checkpoint(saved, model, vocabulary, learner.optimizer, learner.scheduler, training=training,
                    progress={"updates": learner.updates, "policy_version": learner.policy_version})
    resumed, _ = _learner(saved, "cpu")
    assert resumed.updates == 1 and resumed.scheduler.completed_updates == 1
    state = random.getstate()
    expected = learner.bootstrap(runs)[0]
    random.setstate(state)
    actual = resumed.bootstrap(runs)[0]
    assert actual["learning_rates"] == expected["learning_rates"]
    for a, b in zip(learner.model.parameters(), resumed.model.parameters()):
        torch.testing.assert_close(a, b)
    changed = replace(training, embedding_lr=training.embedding_lr * 4)
    with pytest.raises(ValueError, match="Strict resume"):
        _learner(saved, "cpu", training=changed)
    fresh, _ = _learner(saved, "cpu", weights_only=True, training=changed)
    assert fresh.updates == fresh.scheduler.completed_updates == 0
    assert not fresh.optimizer.state
    assert next(g for g in fresh.optimizer.param_groups if g["name"] == "embedding_no_decay")["lr"] == changed.embedding_lr
    initial = torch.load(saved / "weights.pt", weights_only=True)
    for name, parameter in fresh.model.named_parameters():
        torch.testing.assert_close(parameter, initial[name])
    # Editing the manifest must not quietly restore an old LR or other config.
    manifest = json.loads((saved / "manifest.json").read_text())
    manifest["training"]["head_lr"] *= 2
    (saved / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="conflicts"):
        _learner(saved, "cpu")


def test_legacy_encoding_can_only_start_new_training_without_reindexing(tmp_path):
    _, vocabulary, model, training = setup()
    old = vocabulary.state()
    old["symbols"] = [s for s in old["symbols"] if not s.startswith("rider_effect=")]
    vocabulary = Vocabulary.from_state(old, model.config)
    saved = tmp_path / "old-init"
    save_checkpoint(saved, model, vocabulary, training=training)
    manifest = json.loads((saved / "manifest.json").read_text())
    del manifest["encoding_version"], manifest["training_version"]
    (saved / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="encoding changed"):
        load_model(saved)
    fresh, _ = _learner(saved, "cpu", weights_only=True)
    assert fresh.vocabulary.symbols[:len(old["symbols"])] == tuple(old["symbols"])
    assert fresh.vocabulary.encode("rider_effect=Energized") >= len(old["symbols"])
    for a, b in zip(model.parameters(), fresh.model.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert fresh.updates == 0 and not fresh.optimizer.state


def test_decay_runs_bc_rates_to_zero_and_resumes_only_at_its_length(tmp_path):
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([parameter], lr=4e-5)
    scheduler = BootstrapSchedule(optimizer, 0, decay_updates=4)
    rates = []
    while not scheduler.exhausted:
        scheduler.prepare("bootstrap")
        rates.append(optimizer.param_groups[0]["lr"])
        scheduler.step()
    assert rates == pytest.approx([4e-5, 3e-5, 2e-5, 1e-5]) and scheduler.decayed_updates == 4
    scheduler.prepare("ppo")
    assert optimizer.param_groups[0]["lr"] == 4e-5
    constant = {"kind": TRAINING_VERSION, "base_lrs": [4e-5], "warmup_updates": 0, "completed_updates": 9,
                "decay_updates": 0, "decayed_updates": 0, "mode": "bootstrap"}
    fresh = BootstrapSchedule(optimizer, 0, decay_updates=4)
    fresh.load_state_dict(constant)
    assert (fresh.completed_updates, fresh.decayed_updates, optimizer.param_groups[0]["lr"]) == (9, 0, 4e-5)
    for other in (BootstrapSchedule(optimizer, 0, decay_updates=2), BootstrapSchedule(optimizer, 0)):
        with pytest.raises(ValueError, match="decay of 4"):
            other.load_state_dict(scheduler.state_dict())
    # A decayed BC run stops after its updates; the checkpoint continues the optimizer.
    runs, vocabulary, model, training = setup()
    learner = Learner(model, vocabulary, training)
    learner.bootstrap(runs)
    saved = tmp_path / "constant"
    save_checkpoint(saved, model, vocabulary, learner.optimizer, learner.scheduler, training=training,
                    progress={"updates": learner.updates, "policy_version": learner.policy_version})
    decaying, _ = _learner(saved, "cpu", decay_updates=1)
    assert decaying.scheduler.decay_updates == 1 and decaying.optimizer.state
    metrics = decaying.bootstrap(runs, epochs=None)
    assert decaying.updates == learner.updates + 1 and len(metrics) == 1
    # The finished decay cannot be trained further, and is loaded to play as it is.
    final = tmp_path / "final"
    save_checkpoint(final, decaying.model, vocabulary, decaying.optimizer, decaying.scheduler, training=training,
                    progress={"updates": decaying.updates, "policy_version": decaying.policy_version})
    with pytest.raises(ValueError, match="inside a decay"):
        _learner(final, "cpu")
    played, _ = _learner(final, "cpu", inference=True)
    assert played.updates == decaying.updates and played.policy_version == decaying.policy_version
    for a, b in zip(decaying.model.parameters(), played.model.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert metrics[0]["decay"] == {"updates": 1, "completed": 1} and decaying.scheduler.exhausted
    with pytest.raises(ValueError, match="decay has finished"):
        decaying.bootstrap(runs)
    assert parser().parse_args(["bootstrap", "--data", "d", "--output", "o", "--decay-updates", "5"]).decay_updates == 5


def test_tables_fall_geometrically_over_their_decay_then_hold(tmp_path):
    parameters = [torch.nn.Parameter(torch.ones(1)) for _ in range(2)]
    optimizer = torch.optim.AdamW([{"params": [parameters[0]], "lr": 1e-2, "name": "embedding_no_decay"},
                                   {"params": [parameters[1]], "lr": 3e-5, "name": "head_decay"}])
    scheduler = BootstrapSchedule(optimizer, 0, table_final_ratio=0.1, table_decay_updates=4)
    tables, heads = [], []
    for _ in range(6):
        scheduler.prepare("bootstrap")
        tables.append(optimizer.param_groups[0]["lr"])
        heads.append(optimizer.param_groups[1]["lr"])
        scheduler.step()
    assert tables == pytest.approx([1e-2, 1e-2 * 0.1 ** .25, 1e-2 * 0.1 ** .5, 1e-2 * 0.1 ** .75, 1e-3, 1e-3])
    assert heads == [3e-5] * 6
    scheduler.prepare("ppo")
    assert optimizer.param_groups[0]["lr"] == 1e-2
    # The table decay is part of the schedule a strict resume must match.
    with pytest.raises(ValueError, match="table decay"):
        BootstrapSchedule(optimizer, 0).load_state_dict(scheduler.state_dict())
    for bad in (dict(embedding_lr_final=1e-3), dict(embedding_decay_updates=5),
                dict(embedding_lr_final=2e-2, embedding_decay_updates=5)):
        with pytest.raises(ValueError):
            TrainConfig(embedding_lr=1e-2, **bad)
    runs, vocabulary, model, training = setup()
    training = replace(training, embedding_lr=1e-2, embedding_lr_final=1e-3, embedding_decay_updates=2,
                       logical_batch_size=4)
    learner = Learner(model, vocabulary, training)
    assert learner.scheduler.table_final_ratio == pytest.approx(0.1)
    # One update per epoch here: the third update is the first at the final rate.
    metrics = learner.bootstrap(runs, epochs=3)
    assert [m["learning_rates"]["embedding_no_decay"] for m in metrics] == pytest.approx([1e-2, 1e-2 * 0.1 ** .5, 1e-3])
    assert metrics[-1]["learning_rates"]["head_decay"] == pytest.approx(training.head_lr)
    saved = tmp_path / "tables"
    save_checkpoint(saved, model, vocabulary, learner.optimizer, learner.scheduler, training=training,
                    progress={"updates": learner.updates, "policy_version": learner.policy_version})
    resumed, _ = _learner(saved, "cpu")
    assert resumed.scheduler.table_decay_updates == 2 and resumed.scheduler.completed_updates == learner.updates


def test_warmup_reaches_old_rates_and_does_not_scale_ppo_or_value():
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([parameter], lr=3e-5)
    scheduler = BootstrapSchedule(optimizer, 2)
    rates = []
    for mode in ("bootstrap", "value", "bootstrap", "ppo", "bootstrap"):
        scheduler.prepare(mode)
        rates.append(optimizer.param_groups[0]["lr"])
        scheduler.step()
    assert rates == pytest.approx([1.5e-5, 3e-5, 3e-5, 3e-5, 3e-5])
    assert scheduler.completed_updates == 3


def test_unlimited_bc_continues_until_manual_interrupt():
    runs, vocabulary, model, training = setup()
    learner = Learner(model, vocabulary, training)
    epochs = []

    def stop(index, metrics):
        epochs.append(index)
        if index == 2:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        learner.bootstrap(runs, epochs=None, on_epoch=stop)
    assert epochs == [1, 2] and learner.policy_version == 2


def test_bc_defaults_have_no_automatic_validation_or_stopping_rule():
    args = parser().parse_args(["bootstrap", "--data", "provided", "--output", "new-run"])
    assert args.epochs is None
    assert not hasattr(args, "holdout") and not hasattr(args, "validation")
    assert TrainConfig().logical_batch_size == 512


def test_feeders_share_tensors_by_name_and_a_dead_feeder_is_reported(tmp_path, monkeypatch):
    import os
    import torch.multiprocessing
    from model import loader

    shards, runs = sharded(tmp_path)
    vocabulary = vocabulary_for(runs, ModelConfig.tiny())
    training = TrainConfig(precision="no", loader_workers=1, loader_feeders=2, logical_batch_size=3,
                           microbatch_size=2)
    items = samples(runs)
    # Two feeders interleave, so the batches arrive in either order; every sample arrives once,
    # and its tensors came through named shared memory, with no descriptor to serve later.
    fed, shared = [], []
    for logical in stream(shards, vocabulary, training, window_shards=2, shuffle=False):
        for batch_items, prepared in logical:
            fed += [item.run_id for item in batch_items]
            shared.append(prepared.labels.is_shared() and prepared.batch.tokens.ids.is_shared())
    assert Counter(fed) == Counter(item.run_id for item in items)
    # The feeders' tails are packed by the trainer itself, after every feeder batch.
    assert len(shared) > 1 and all(shared[:-1]) and not shared[-1]
    assert torch.multiprocessing.get_sharing_strategy() == "file_system"

    def vanish(connection, *args):
        os._exit(0)

    monkeypatch.setattr(loader, "_feed", vanish)
    with pytest.raises(RuntimeError, match="exited without finishing"):
        list(stream(shards, vocabulary, training, window_shards=2, shuffle=False))


def test_stalled_loader_workers_are_replaced_without_losing_or_repeating_batches(monkeypatch):
    from model import loader

    class Numbers(torch.utils.data.Dataset):
        def __len__(self):
            return 7

        def __getitem__(self, index):
            return index

    started = []

    class Stalling:
        def __init__(self, dataset, **options):
            self.dataset = dataset
            started.append((len(dataset), options["num_workers"], options["timeout"]))

        def __iter__(self):
            stall = 3 if len(started) == 1 else None
            for index in range(len(self.dataset)):
                if index == stall:
                    raise RuntimeError(f"DataLoader timed out after {loader.STALL_SECONDS} seconds")
                yield self.dataset[index]

    monkeypatch.setattr(loader, "DataLoader", Stalling)
    assert list(loader._stream(Numbers(), 4)) == list(range(7))
    assert started == [(7, 4, loader.STALL_SECONDS), (4, 4, loader.STALL_SECONDS)]


def test_feeders_and_their_workers_end_when_the_trainer_is_killed(tmp_path, monkeypatch):
    import os
    import signal
    import time
    import torch.multiprocessing as multiprocessing
    from model import loader

    shards, runs = sharded(tmp_path)
    vocabulary = vocabulary_for(runs, ModelConfig.tiny())
    training = TrainConfig(precision="no", loader_workers=1, loader_feeders=2, logical_batch_size=3,
                           microbatch_size=2)
    packing = loader._stream

    def busy(dataset, workers):
        # A feeder in the middle of a window, its workers alive.
        for value in packing(dataset, workers):
            yield value
            time.sleep(60)

    monkeypatch.setattr(loader, "_stream", busy)

    def trainer(connection):
        batches = stream(shards, vocabulary, training, window_shards=2, shuffle=False)
        next(batches)
        connection.send([child.pid for child in multiprocessing.active_children()])
        time.sleep(60)

    context = multiprocessing.get_context("fork")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=trainer, args=(sender,))
    process.start()
    feeders = receiver.recv()
    assert len(feeders) == 2

    def group_alive(leader):
        try:
            os.killpg(leader, 0)
        except ProcessLookupError:
            return False
        return True

    assert all(group_alive(pid) for pid in feeders)
    # No cleanup runs in a killed trainer: the feeders must notice by themselves.
    os.kill(process.pid, signal.SIGKILL)
    process.join()
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline and any(group_alive(pid) for pid in feeders):
        time.sleep(0.2)
    assert not any(group_alive(pid) for pid in feeders)
