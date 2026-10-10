"""A round sampled by several processes is the round one process would sample."""

import copy
import gc
import json
import os
import pickle
import random
from dataclasses import fields, replace as changed

import pytest
import torch

from model.batch import Packed
from model.cli import _read_runs, vocabulary_for
from model.config import ModelConfig, TrainConfig
from model.data import sample_lengths, samples
from model.model import PolicyValue
from model.policy import prepare_decisions, prepare_replay
from model.protocol import CHARACTERS, ProtocolError
from model.rollout import RolloutRunner, collect_round
from model.sampling import Samplers
from model.testing import SyntheticEngine, demonstration
from model.tests.test_encoding import build, duel, logits, map_frame
from model.trainer import Learner, climb, evaluate_runs
from model.wire import GROUPED, pack, replay, size, unpack


def setup_round():
    demonstrations = [demonstration(character, "sampling", 4, 2) for character in CHARACTERS]
    config = ModelConfig.tiny()
    vocabulary = vocabulary_for(demonstrations, config)
    training = TrainConfig(optimizer="adamw", precision="no", loader_workers=0, logical_batch_size=8,
                           microbatch_size=4, bootstrap_warmup_updates=0, ascension=0)
    torch.manual_seed(0)
    learner = Learner(PolicyValue(config), vocabulary, training)
    # A value head that says something, so that stored values are worth comparing.
    torch.nn.init.normal_(learner.model.value.weight)
    runner = RolloutRunner(learner.model, vocabulary, version=learner.policy_version,
                           precision="no", ascension=0)
    seeds = {character: [f"{character}-{index}" for index in range(4)] for character in CHARACTERS}
    return learner, runner, seeds


def engine():
    return SyntheticEngine(4, 2)


def read(paths):
    return [json.loads(path.read_text()) for path in paths]


read_runs = read


def test_choosing_for_a_packed_batch_scores_as_decision_by_decision():
    frames = [map_frame(), duel("enemy:0"), map_frame(("map:1:2", "Elite"))]
    model, vocabulary = build(frames)
    torch.nn.init.normal_(model.value.weight)
    _, decisions = prepare_decisions(vocabulary, frames)
    with torch.no_grad():
        slots, log_probs, values = model.choose(decisions, sample=False)
        drawn = model.choose(decisions)[0]
    for index, expected in enumerate(logits(model, vocabulary, frames)):
        assert int(slots[index]) == int(expected.argmax())
        assert torch.allclose(log_probs[index], expected.log_softmax(-1).max(), atol=1e-5)
        assert bool(decisions.masks[index, drawn[index]])
    assert values.abs().min() > 0


def same(a, b):
    """Packed values with the same tensors."""
    if isinstance(a, torch.Tensor):
        return a.dtype == b.dtype and a.shape == b.shape and bool((a == b).all())
    if isinstance(a, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, Packed):
        return all(same(getattr(a, f.name), getattr(b, f.name)) for f in fields(a))
    return a == b


FRAMES = [map_frame(), duel("enemy:0"), map_frame(("map:1:2", "Elite")), duel("enemy:1"),
          map_frame(("map:0:2", "Shop"))]


def test_a_pack_crosses_to_the_collecting_process_with_its_programs_regrouped():
    model, vocabulary = build(FRAMES)
    torch.nn.init.normal_(model.value.weight)
    for frames in (FRAMES[:1], FRAMES[1:2], FRAMES):
        decisions = prepare_decisions(vocabulary, frames)[1]
        message = pickle.loads(pickle.dumps(pack(decisions)))
        arrived, unregistered = unpack([message], "cpu")
        assert size(message) == len(frames) and unregistered == []
        # Everything but the program groups arrives as it was packed.
        assert same(arrived.masks, decisions.masks)
        assert all(same(getattr(arrived.batch, f.name), getattr(decisions.batch, f.name))
                   for f in fields(arrived.batch) if f.name != "programs")
        # The programs are the same ones in fewer groups, each still at its place.
        groups = decisions.batch.programs, arrived.batch.programs
        assert len(groups[1]) <= min(len(groups[0]), len(GROUPED))
        for name in ("lengths", "slots", "binding_sources", "binding_roles"):
            assert all(sorted(torch.cat([getattr(g, name) for g in a]).tolist())
                       == sorted(torch.cat([getattr(g, name) for g in groups[0]]).tolist()) for a in groups)
        with torch.no_grad():
            for a, b in zip(model.choose(arrived, sample=False), model.choose(decisions, sample=False)):
                assert torch.allclose(a.float(), b.float(), atol=1e-5)


def test_packs_of_several_processes_are_chosen_for_as_one_pack():
    model, vocabulary = build(FRAMES)
    torch.nn.init.normal_(model.value.weight)
    together = prepare_decisions(vocabulary, FRAMES)[1]
    # As the collecting process receives them: packed by different processes.
    messages = [pack(prepare_decisions(vocabulary, part)[1])
                for part in (FRAMES[:1], FRAMES[1:3], FRAMES[3:])]
    joined, _ = unpack(messages, "cpu")
    assert joined.masks.shape == together.masks.shape
    assert same(joined.batch.lengths, together.batch.lengths)
    with torch.no_grad():
        expected = model.choose(together, sample=False)
        found = model.choose(joined, sample=False)
    assert found[0].tolist() == expected[0].tolist()
    for a, b in zip(found[1:], expected[1:]):
        assert torch.allclose(a, b, atol=1e-5)


def test_names_the_vocabulary_never_registered_reach_the_collecting_process():
    _, vocabulary = build(FRAMES[:1])
    messages = [pack(prepare_decisions(vocabulary, [frame])[1]) for frame in FRAMES[:2]]
    assert messages[0][3] == [] and messages[1][3]
    assert unpack(messages, "cpu")[1] == sorted(messages[1][3])


def samplers(runner, factory=engine, *, engines=4, processes=2, keep=False):
    return Samplers(factory, runner.model, runner.vocabulary, engines=engines, processes=processes, keep=keep)


def test_a_round_sampled_in_processes_replays_as_it_was_stored(tmp_path):
    learner, runner, seeds = setup_round()
    done = []
    with samplers(runner, engines=5, processes=3) as pool:
        paths = collect_round(engine, runner, seeds, tmp_path / "round", samplers=pool,
                              on_run=lambda *progress: done.append(progress))
    runs = read(paths)
    assert [run["status"] for run in runs] == ["complete"] * 20
    assert [(completed, total) for _, completed, total in done] == [(n, 20) for n in range(1, 21)]
    assert sorted(character for character, _, _ in done) == sorted(list(CHARACTERS) * 4)
    assert all(path.with_suffix(".jsonl").exists() for path in paths)
    # Exact arithmetic: every stored probability and value is the learner's own.
    found = learner.check_old_policy(runs, samples(runs, ppo=True, horizon_scale=learner.config.horizon_scale))
    assert max(found.values()) < 1e-4
    assert any(macro["old_value"] for run in runs for macro in run["macros"])


def steps(run):
    return [(step["frame"]["routing"]["decision_id"], step["candidate_ref"])
            for macro in run["macros"] for step in macro["steps"]]


def test_greedy_runs_in_processes_take_the_steps_one_process_takes(tmp_path):
    _, runner, seeds = setup_round()
    alone = read(collect_round(engine, runner, seeds, tmp_path / "alone", sample=False))
    with samplers(runner) as pool:
        apart = read(collect_round(engine, runner, seeds, tmp_path / "apart", samplers=pool, sample=False))
    assert [steps(run) for run in apart] == [steps(run) for run in alone]
    assert [run["victory"] for run in apart] == [run["victory"] for run in alone]


def test_sampling_processes_stay_for_the_rounds_of_later_policies(tmp_path):
    learner, runner, seeds = setup_round()
    started = []

    def counted():
        started.append(os.getpid())
        return engine()

    with samplers(runner, counted, engines=2, processes=2) as pool:
        collect_round(engine, runner, seeds, tmp_path / "first", samplers=pool)
        with torch.no_grad():
            for parameter in learner.model.parameters():
                parameter.add_(0.05 * torch.randn_like(parameter))
        learner.policy_version += 1
        later = RolloutRunner(learner.model, runner.vocabulary, version=learner.policy_version,
                              precision="no", ascension=0)
        runs = read(collect_round(engine, later, seeds, tmp_path / "second", samplers=pool))
    assert {run["policy_version"] for run in runs} == {later.version}
    assert {run["policy_digest"] for run in runs} == {later.digest} != {runner.digest}
    # The second round is the changed policy's own, played by the processes of the first.
    found = learner.check_old_policy(runs, samples(runs, ppo=True, horizon_scale=learner.config.horizon_scale))
    assert max(found.values()) < 1e-4
    # The factory ran in the sampling processes only, not here.
    assert started == []


def test_a_kept_round_updates_as_the_round_read_back_from_its_frames(tmp_path):
    learner, runner, seeds = setup_round()
    with samplers(runner, engines=5, processes=5, keep=True) as pool:
        paths = collect_round(engine, runner, seeds, tmp_path / "round", samplers=pool)
        kept = pool.runs()
    read = read_runs(paths)
    # The same runs, their macro-actions holding packed decisions in place of frames.
    assert [run["run_id"] for run in kept] == [run["run_id"] for run in read]
    assert evaluate_runs(kept) == evaluate_runs(read)
    for ours, theirs in zip(kept, read):
        assert "environment_actions" not in ours and len(ours["macros"]) == len(theirs["macros"])
        for packed, framed in zip(ours["macros"], theirs["macros"]):
            assert "steps" not in packed and len(packed["packed"]) == len(packed["slots"])
            assert len(packed["packed"]) == sum(not step["forced"] for step in framed["steps"])
            assert {key: packed[key] for key in ("old_log_prob", "old_value", "return", "advantage", "phase")} \
                == {key: framed[key] for key in ("old_log_prob", "old_value", "return", "advantage", "phase")}
    horizon = learner.config.horizon_scale
    ours, theirs = samples(kept, ppo=True, horizon_scale=horizon), samples(read, ppo=True, horizon_scale=horizon)
    assert [(item.weight, item.branches, sample_lengths(item)) for item in ours] \
        == [(item.weight, item.branches, sample_lengths(item)) for item in theirs]
    # One batch either way: the packed decisions replay as the frames do.
    with torch.no_grad():
        packed = learner.model.replay(replay([(item.macro["packed"], item.macro["slots"]) for item in ours], "cpu"))
        framed = learner.model.replay(prepare_replay(learner.vocabulary, [item.macro["steps"] for item in theirs]))
    for a, b in zip(packed, framed):
        assert torch.allclose(a, b, atol=1e-5)
    assert max(learner.check_old_policy(kept, ours).values()) < 1e-4
    # And the update they give is the update the frames give.
    weights = {name: value.clone() for name, value in learner.model.state_dict().items()}
    state = copy.deepcopy(learner.optimizer.state_dict())
    results = []
    for runs in (kept, read):
        learner.model.load_state_dict(weights)
        learner.optimizer.load_state_dict(copy.deepcopy(state))
        learner.policy_version, learner.updates = runner.version, 0
        random.seed(3)
        results.append(learner.ppo(runs))
    assert [epoch["updates"] for epoch in results[0]] == [epoch["updates"] for epoch in results[1]]
    for ours, theirs in zip(*results):
        for key in ("loss", "kl", "entropy", "value_mse", "clip_fraction"):
            assert ours[key] == pytest.approx(theirs[key], abs=1e-4)


def test_a_round_handed_over_in_memory_leaves_only_its_fights_on_disk(tmp_path):
    learner, runner, seeds = setup_round()
    with samplers(runner, engines=5, processes=5, keep=True) as pool:
        assert collect_round(engine, runner, seeds, tmp_path / "round", samplers=pool, save=False) == []
        kept = pool.runs()
    written = sorted(path.name for path in (tmp_path / "round").iterdir())
    assert written == sorted(["metadata"] + [f"{c}-{i}.combats.jsonl.gz" for c in CHARACTERS for i in range(4)])
    assert json.loads((tmp_path / "round/metadata/seeds.json").read_text()) == seeds
    assert [run["status"] for run in kept] == ["complete"] * 20
    found = learner.check_old_policy(kept, samples(kept, ppo=True, horizon_scale=learner.config.horizon_scale))
    assert max(found.values()) < 1e-4
    with pytest.raises(ValueError, match="Only sampling processes"):
        collect_round(engine, runner, seeds, tmp_path / "threads", save=False)


class Climbing(SyntheticEngine):
    """An engine whose frames say where the run stands: a floor further every decision."""

    def frame(self):
        frame = super().frame()
        entities = frame["public"]["entities"]
        entities[0].update(act=1, floor=self.version + 1)
        entities.append({"ref": "map:0:16", "entity_type": "map_node", "content_id": "Boss",
                         "floor": 16, "col": 0, "current": False, "visited": False})
        return frame


def test_a_run_that_keeps_no_frames_still_says_where_it_stood():
    _, runner, _ = setup_round()
    framed = runner.run(Climbing(4, 2), "Ironclad", "climb", sample=False)
    bare = runner.run(Climbing(4, 2), "Ironclad", "climb", sample=False, frames=False)
    assert climb(framed) == bare["climb"] == ({1: max(bare["climb"][0].values())}, {1: 17})
    assert bare["climb"][0][1] > 1
    for kept, dropped in zip(framed["macros"], bare["macros"]):
        assert [(step["candidate_ref"], step["forced"]) for step in kept["steps"]] \
            == [(step["candidate_ref"], step["forced"]) for step in dropped["steps"]]
        assert all("frame" not in step for step in dropped["steps"])
        assert {key: kept[key] for key in ("old_log_prob", "old_value", "return", "advantage")} \
            == {key: dropped[key] for key in ("old_log_prob", "old_value", "return", "advantage")}
    assert evaluate_runs([framed]) == evaluate_runs([bare])


class Failing(SyntheticEngine):
    """An engine whose first Silent run breaks off."""

    def reset(self, character="Ironclad", seed="test", ascension=10):
        if (character, seed) == ("Silent", "Silent-0"):
            raise ProtocolError("the run broke off")
        return super().reset(character, seed, ascension)


def failing():
    return Failing(4, 2)


@pytest.mark.parametrize("processes", [0, 2])
def test_a_run_that_did_not_finish_blocks_the_update_and_is_written_for_inspection(tmp_path, processes):
    _, runner, seeds = setup_round()
    if not processes:
        with pytest.raises(ProtocolError, match="Round incomplete.*Silent-0.json"):
            collect_round(failing, runner, seeds, tmp_path / "round", workers=3)
        return
    with samplers(runner, failing, keep=True) as pool:
        with pytest.raises(ProtocolError, match="Round incomplete.*Silent-0.json"):
            collect_round(failing, runner, seeds, tmp_path / "round", samplers=pool, save=False)
    # Nothing else of the round was written.
    assert [path.name for path in (tmp_path / "round").glob("*.json")] == ["Silent-0.json"]
    broken = json.loads((tmp_path / "round/Silent-0.json").read_text())
    assert broken["status"] == "error" and "broke off" in broken["error"]


def test_a_round_is_handed_over_only_where_it_was_kept(tmp_path):
    _, runner, seeds = setup_round()
    with samplers(runner) as pool:
        collect_round(engine, runner, seeds, tmp_path / "round", samplers=pool)
        with pytest.raises(RuntimeError, match="No round was kept"):
            pool.runs()


def test_runs_are_read_back_with_the_collector_left_as_it_was(tmp_path):
    _, runner, seeds = setup_round()
    paths = collect_round(engine, runner, seeds, tmp_path / "round")
    assert gc.isenabled()
    assert [run["seed"] for run in _read_runs(paths)] == [seed for c in CHARACTERS for seed in seeds[c]]
    assert gc.isenabled()


def test_sampling_processes_serve_one_model(tmp_path):
    _, runner, seeds = setup_round()
    _, other, _ = setup_round()
    with samplers(runner) as pool:
        with pytest.raises(ValueError, match="differ"):
            collect_round(engine, other, seeds, tmp_path / "round", samplers=pool)


class Broken(SyntheticEngine):
    def reset(self, *args):
        raise RuntimeError("the engine broke")


class Vanishing(SyntheticEngine):
    def reset(self, *args):
        os._exit(3)


@pytest.mark.parametrize("factory, message", [
    (Broken, "the engine broke"),
    (Vanishing, "exited without finishing"),
])
def test_a_sampling_process_that_fails_stops_the_round_and_the_processes(tmp_path, factory, message):
    _, runner, seeds = setup_round()
    with samplers(runner, factory) as pool:
        with pytest.raises(RuntimeError, match=message):
            collect_round(factory, runner, seeds, tmp_path / "round", samplers=pool)
        with pytest.raises(RuntimeError, match="have ended"):
            collect_round(factory, runner, seeds, tmp_path / "again", samplers=pool)


def cooled_round(temperature):
    """The round's learner and runner, sampling at a temperature."""
    learner, runner, seeds = setup_round()
    learner.config = changed(learner.config, sampling_temperature=temperature)
    runner = RolloutRunner(learner.model, learner.vocabulary, version=learner.policy_version,
                           precision="no", ascension=0, temperature=temperature)
    return learner, runner, seeds


def test_choices_at_a_temperature_are_drawn_from_the_scores_divided_by_it():
    frames = [map_frame(), duel("enemy:0"), map_frame(("map:1:2", "Elite"))]
    model, vocabulary = build(frames)
    _, decisions = prepare_decisions(vocabulary, frames)
    torch.manual_seed(0)
    with torch.no_grad():
        slots, log_probs, _ = model.choose(decisions, temperature=0.5)
        most_likely = model.choose(decisions, sample=False, temperature=0.5)[0]
        cold = [model.choose(decisions, temperature=1e-6)[0] for _ in range(8)]
    for index, expected in enumerate(logits(model, vocabulary, frames)):
        assert torch.allclose(log_probs[index], (expected / 0.5).log_softmax(-1)[slots[index]], atol=1e-5)
        # The most likely action is the same at every temperature, and all that is left of
        # the distribution as the temperature falls.
        assert int(most_likely[index]) == int(expected.argmax())
        assert {int(drawn[index]) for drawn in cold} == {int(expected.argmax())}


@pytest.mark.parametrize("workers,processes", [(1, 0), (3, 0), (5, 3)])
def test_a_round_sampled_at_a_temperature_replays_at_it_and_at_no_other(tmp_path, workers, processes):
    learner, runner, seeds = cooled_round(0.5)
    if processes:
        with samplers(runner, engines=workers, processes=processes) as pool:
            paths = collect_round(engine, runner, seeds, tmp_path / "round", samplers=pool)
    else:
        paths = collect_round(engine, runner, seeds, tmp_path / "round", workers=workers)
    runs = read(paths)
    assert {run["temperature"] for run in runs} == {0.5}
    items = samples(runs, ppo=True, horizon_scale=learner.config.horizon_scale)
    assert max(learner.check_old_policy(runs, items).values()) < 1e-4
    # A learner at another temperature is another policy: the round is not its own, and
    # its probabilities are not the ones the round stored.
    learner.config = changed(learner.config, sampling_temperature=1.0)
    with pytest.raises(ProtocolError, match="temperature differs"):
        learner.check_old_policy(runs, items)
    for run in runs:
        run["temperature"] = 1.0
    with pytest.raises(ProtocolError, match="do not replay"):
        learner.check_old_policy(runs, items)


def test_the_update_of_a_cooled_round_starts_from_the_policy_that_sampled_it(tmp_path):
    learner, runner, seeds = cooled_round(0.5)
    with samplers(runner, engines=5, processes=5, keep=True) as pool:
        collect_round(engine, runner, seeds, tmp_path / "round", samplers=pool, save=False)
        kept = pool.runs()
    demonstrations = [demonstration(character, "sampling", 4, 2) for character in CHARACTERS]
    learned_at_one = learner.evaluate_bootstrap(demonstrations)
    updates = []
    learner.ppo(kept, on_update=lambda update, *, epoch, stage: updates.append(update))
    # Before any update the new policy is the old one: at the round's temperature their
    # probabilities agree, so nothing has moved and nothing is clipped.
    assert updates[0]["kl"] < 1e-6 and updates[0]["clip_fraction"] == 0
    # Demonstrations are scored at temperature 1, whatever rounds are sampled at.
    learner, _, _ = setup_round()
    assert learner.evaluate_bootstrap(demonstrations) == learned_at_one


def test_greedy_runs_have_no_temperature():
    learner, runner, _ = cooled_round(0.5)
    run = runner.run(engine(), "Ironclad", "greedy", sample=False)
    assert run["status"] == "complete" and run["temperature"] is None
    with pytest.raises(ValueError, match="temperature"):
        RolloutRunner(learner.model, learner.vocabulary, temperature=0)
    with pytest.raises(ValueError, match="temperature"):
        TrainConfig(sampling_temperature=1.5)

