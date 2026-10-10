"""A round sampled by several processes is the round the runner plays alone."""

import gzip
import json
import os
import pickle
from dataclasses import fields

import pytest
import torch

from model.batch import Packed
from model.cli import vocabulary_for
from model.config import ModelConfig, TrainConfig
from model.data import validate_run
from model.model import PolicyNetwork
from model.policy import prepare_decisions
from model.protocol import CHARACTERS, ProtocolError
from model.representation import observation
from model.rollout import RolloutRunner, collect_round
from model.sampling import Samplers, _without_frames
from model.testing import SyntheticEngine, demonstration
from model.tests.test_encoding import build, duel, logits, map_frame
from model.trainer import Learner, climb, evaluate_runs
from model.wire import GROUPED, pack, size, unpack


def setup_round(temperatures=None):
    demonstrations = [demonstration(character, "sampling", 4, 2) for character in CHARACTERS]
    config = ModelConfig.tiny()
    vocabulary = vocabulary_for(demonstrations, config)
    training = TrainConfig(optimizer="adamw", precision="no", loader_workers=0, logical_batch_size=8,
                           microbatch_size=4, bootstrap_warmup_updates=0, ascension=0)
    torch.manual_seed(0)
    learner = Learner(PolicyNetwork(config), vocabulary, training)
    runner = RolloutRunner(learner.model, vocabulary, version=learner.policy_version,
                           precision="no", ascension=0, temperatures=temperatures)
    seeds = {character: [f"{character}-{index}" for index in range(4)] for character in CHARACTERS}
    return learner, runner, seeds


def engine():
    return SyntheticEngine(4, 2)


def read(paths):
    return [json.loads(path.read_text()) for path in paths]


def scored(model, vocabulary, run, temperatures=None):
    """Each macro-action's log-probability scored again from its frames, decision by
    decision at the temperature of its phase (at 1 without temperatures)."""
    result = []
    for macro in run["macros"]:
        total = 0.0
        for step in macro["steps"]:
            if step["forced"]:
                continue
            frame = step["frame"]
            obs = observation(frame)
            slots = {c["candidate_ref"]: c["decoder_slot_ref"] for c in frame["legal"]["candidates"]}
            mask = torch.tensor([slot in set(slots.values()) for slot in obs.slot_refs])
            with torch.no_grad():
                output = model.decode_batch(model.encode([obs], vocabulary), [mask])[0]
            phase = "combat" if frame["public"]["phase"] == "combat" else "other"
            temperature = temperatures[phase] if temperatures else 1.0
            chosen = obs.slot_refs.index(slots[step["candidate_ref"]])
            total += float((output.logits / temperature).log_softmax(-1)[chosen])
        result.append(total)
    return result


def largest_gap(model, vocabulary, runs, temperatures=None):
    """How far a stored log-probability is, at most, from its replay."""
    return max(abs(macro["old_log_prob"] - log_prob)
               for run in runs for macro, log_prob in zip(run["macros"], scored(model, vocabulary, run, temperatures)))


def test_choosing_for_a_packed_batch_scores_as_decision_by_decision():
    frames = [map_frame(), duel("enemy:0"), map_frame(("map:1:2", "Elite"))]
    model, vocabulary = build(frames)
    _, decisions = prepare_decisions(vocabulary, frames)
    with torch.no_grad():
        slots, log_probs = model.choose(decisions, sample=False)
        drawn = model.choose(decisions)[0]
    for index, expected in enumerate(logits(model, vocabulary, frames)):
        assert int(slots[index]) == int(expected.argmax())
        assert torch.allclose(log_probs[index], expected.log_softmax(-1).max(), atol=1e-5)
        assert bool(decisions.masks[index, drawn[index]])


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
    assert torch.allclose(found[1], expected[1], atol=1e-5)


def test_names_the_vocabulary_never_registered_reach_the_collecting_process():
    _, vocabulary = build(FRAMES[:1])
    messages = [pack(prepare_decisions(vocabulary, [frame])[1]) for frame in FRAMES[:2]]
    assert messages[0][3] == [] and messages[1][3]
    assert unpack(messages, "cpu")[1] == sorted(messages[1][3])


def test_each_decision_is_drawn_at_its_own_temperature():
    frames = [map_frame(), duel("enemy:0"), map_frame(("map:1:2", "Elite"))]
    model, vocabulary = build(frames)
    _, decisions = prepare_decisions(vocabulary, frames)
    temperatures = torch.tensor([0.5, 1.0, 0.25])
    torch.manual_seed(0)
    with torch.no_grad():
        slots, log_probs = model.choose(decisions, temperature=temperatures)
        most_likely = model.choose(decisions, sample=False, temperature=temperatures)[0]
        cold = [model.choose(decisions, temperature=1e-6)[0] for _ in range(8)]
    for index, expected in enumerate(logits(model, vocabulary, frames)):
        assert torch.allclose(log_probs[index], (expected / temperatures[index]).log_softmax(-1)[slots[index]], atol=1e-5)
        # The most likely action is the same at every temperature, and all that is left of
        # the distribution as the temperature falls.
        assert int(most_likely[index]) == int(expected.argmax())
        assert {int(drawn[index]) for drawn in cold} == {int(expected.argmax())}


def samplers(runner, factory=engine, *, engines=4, processes=2):
    return Samplers(factory, runner.model, runner.vocabulary, engines=engines, processes=processes)


def test_a_round_sampled_in_processes_replays_as_it_was_stored(tmp_path):
    learner, runner, seeds = setup_round()
    done = []
    with samplers(runner, engines=5, processes=3) as pool:
        paths = collect_round(pool, runner, seeds, tmp_path / "round",
                              on_run=lambda *progress: done.append(progress))
    runs = read(paths)
    assert [run["status"] for run in runs] == ["complete"] * 20
    assert [(completed, total) for _, completed, total in done] == [(n, 20) for n in range(1, 21)]
    assert sorted(character for character, _, _ in done) == sorted(list(CHARACTERS) * 4)
    # Every run beside its journal and its fights; the assignment before any of them.
    assert all(path.with_suffix(".jsonl").exists() and path.with_suffix(".combats.jsonl.gz").exists()
               for path in paths)
    assert json.loads((tmp_path / "round/metadata/seeds.json").read_text()) == seeds
    for run in runs:
        validate_run(run, on_policy=True)
    # Exact arithmetic: every stored probability is the learner's own, at the
    # temperatures the round was sampled at, which every run records.
    assert all(run["temperature"] == {"combat": 1.0, "other": 1.0} for run in runs)
    assert largest_gap(learner.model, learner.vocabulary, runs) < 1e-4


def steps(run):
    return [(step["frame"]["routing"]["decision_id"], step["candidate_ref"])
            for macro in run["macros"] for step in macro["steps"]]


def test_greedy_runs_in_processes_take_the_steps_the_runner_takes_alone(tmp_path):
    _, runner, seeds = setup_round()
    alone = [runner.run(engine(), character, seed, sample=False)
             for character in CHARACTERS for seed in seeds[character]]
    with samplers(runner) as pool:
        apart = read(collect_round(pool, runner, seeds, tmp_path / "apart", sample=False))
    assert [steps(run) for run in apart] == [steps(run) for run in alone]
    assert [run["victory"] for run in apart] == [run["victory"] for run in alone]
    assert [run["temperature"] for run in apart] == [None] * 20


def test_sampling_processes_stay_for_the_rounds_of_later_policies(tmp_path):
    learner, runner, seeds = setup_round()
    started = []

    def counted():
        started.append(os.getpid())
        return engine()

    with samplers(runner, counted, engines=2, processes=2) as pool:
        collect_round(pool, runner, seeds, tmp_path / "first")
        with torch.no_grad():
            for parameter in learner.model.parameters():
                parameter.add_(0.05 * torch.randn_like(parameter))
        learner.policy_version += 1
        later = RolloutRunner(learner.model, runner.vocabulary, version=learner.policy_version,
                              precision="no", ascension=0)
        runs = read(collect_round(pool, later, seeds, tmp_path / "second"))
    assert {run["policy_version"] for run in runs} == {later.version}
    assert {run["policy_digest"] for run in runs} == {later.digest} != {runner.digest}
    # The second round is the changed policy's own, played by the processes of the first.
    assert largest_gap(learner.model, learner.vocabulary, runs) < 1e-4
    # The factory ran in the sampling processes only, not here.
    assert started == []


class Climbing(SyntheticEngine):
    """An engine whose frames say where the run stands: a floor further every decision."""

    def frame(self):
        frame = super().frame()
        entities = frame["public"]["entities"]
        entities[0].update(act=1, floor=self.version + 1)
        entities.append({"ref": "map:0:16", "entity_type": "map_node", "content_id": "Boss",
                         "floor": 16, "col": 0, "current": False, "visited": False})
        return frame


def test_a_run_handed_over_without_its_frames_still_says_where_it_stood():
    _, runner, _ = setup_round()
    run = runner.run(Climbing(4, 2), "Ironclad", "climb", sample=False)
    handed = _without_frames(run)
    assert "environment_actions" not in handed and all("steps" not in macro for macro in handed["macros"])
    assert [macro["branches"] for macro in handed["macros"]] \
        == [sum(not step["forced"] for step in macro["steps"]) for macro in run["macros"]]
    assert handed["climb"] == climb(run) == ({1: max(handed["climb"][0].values())}, {1: 17})
    assert handed["climb"][0][1] > 1
    assert evaluate_runs([handed]) == evaluate_runs([run])


class Failing(SyntheticEngine):
    """An engine whose first Silent run breaks off."""

    def reset(self, character="Ironclad", seed="test", ascension=10):
        if (character, seed) == ("Silent", "Silent-0"):
            raise ProtocolError("the run broke off")
        return super().reset(character, seed, ascension)


def failing():
    return Failing(4, 2)


def test_a_run_that_did_not_finish_blocks_what_follows_and_is_written_for_inspection(tmp_path):
    _, runner, seeds = setup_round()
    with samplers(runner, failing) as pool:
        with pytest.raises(ProtocolError, match="Round incomplete.*Silent-0.json"):
            collect_round(pool, runner, seeds, tmp_path / "round")
    broken = json.loads((tmp_path / "round/Silent-0.json").read_text())
    assert broken["status"] == "error" and "broke off" in broken["error"]
    # The rest of the round was played and written all the same.
    assert len(list((tmp_path / "round").glob("*-?.json"))) == 20


def test_sampling_processes_serve_one_model(tmp_path):
    _, runner, seeds = setup_round()
    _, other, _ = setup_round()
    with samplers(runner) as pool:
        with pytest.raises(ValueError, match="differ"):
            collect_round(pool, other, seeds, tmp_path / "round")


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
            collect_round(pool, runner, seeds, tmp_path / "round")
        with pytest.raises(RuntimeError, match="have ended"):
            collect_round(pool, runner, seeds, tmp_path / "again")


class Fighting(SyntheticEngine):
    """An engine whose selection is a fight: its frames stand in the combat phase."""

    def frame(self):
        frame = super().frame()
        if self.stage == "selection":
            frame["public"]["phase"] = "combat"
        return frame


def fighting():
    return Fighting(4, 2)


def test_a_round_sampled_at_phase_temperatures_records_them_and_replays_at_them(tmp_path):
    temperatures = {"combat": 0.5, "other": 0.8}
    learner, runner, seeds = setup_round(temperatures)
    with samplers(runner, fighting, engines=5, processes=3) as pool:
        paths = collect_round(pool, runner, seeds, tmp_path / "round")
    runs = read(paths)
    assert all(run["temperature"] == temperatures for run in runs)
    # Both phases occur in every run: the fight's selections, then the choice after it.
    assert all({macro["phase"] for macro in run["macros"]} == {"combat", "combat_play"} for run in runs)
    assert largest_gap(learner.model, learner.vocabulary, runs, temperatures) < 1e-4
    # Scored at any other temperatures the round is another policy's.
    assert largest_gap(learner.model, learner.vocabulary, runs) > 1e-3
    assert largest_gap(learner.model, learner.vocabulary, runs, {"combat": 0.8, "other": 0.5}) > 1e-3
    # The fights of a run name the temperatures with the rest of the policy.
    with gzip.open(paths[0].with_suffix(".combats.jsonl.gz"), "rt") as stream:
        rows = [json.loads(line) for line in stream]
    assert rows and all(row["policy"]["temperature"] == temperatures for row in rows)


def test_greedy_runs_have_no_temperature():
    learner, runner, _ = setup_round({"combat": 0.5, "other": 0.8})
    run = runner.run(fighting(), "Ironclad", "greedy", sample=False)
    assert run["status"] == "complete" and run["temperature"] is None
    # Its probabilities are those of the most likely actions at temperature 1.
    assert largest_gap(learner.model, learner.vocabulary, [run]) < 1e-5
    with pytest.raises(ValueError, match="temperature"):
        RolloutRunner(learner.model, learner.vocabulary, temperatures={"combat": 0, "other": 1})
    with pytest.raises(ValueError, match="temperature"):
        RolloutRunner(learner.model, learner.vocabulary, temperatures={"combat": 0.5})
    with pytest.raises(ValueError, match="temperature"):
        TrainConfig(combat_temperature=1.5)
    assert TrainConfig(combat_temperature=0.3, other_temperature=0.7).temperatures == {"combat": 0.3, "other": 0.7}
