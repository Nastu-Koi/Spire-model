"""A round sampled by several processes is the round the runner plays alone."""

import json
import os
import pickle
from copy import deepcopy
from dataclasses import fields

import numpy as np
import pytest
import torch

from main.encoding.batch import Packed
from main.commands import vocabulary_for
from main.network.config import ModelConfig, TrainConfig
from main.data.shards import validate_run
from main.network.model import PolicyNetwork
from main.network.policy import prepare_decisions, prepare_replay
from main.env.protocol import CHARACTERS, ProtocolError, execution_command
from main.encoding.representation import observation
from main.play.rollout import RolloutRunner, climbed, collect_round, stand
from main.play.sampling import Samplers, _without_frames
from main.play.testing import SyntheticEngine, demonstration
from main.tests.test_encoding import build, duel, logits, map_frame
from main.play.evaluation import climb, evaluate_runs
from main.learn.trainer import Learner
from main.play import wire
from main.play.wire import GROUPED, Sent, pack, size, tables, unpack


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


def scored(model, vocabulary, run, temperatures=None, by_phase=False):
    """Each macro-action's log-probability scored again from its frames, decision by
    decision at the temperature it was drawn at: the fight's inside a fight, the
    rest's outside (at 1 without temperatures). With `by_phase` a fight is only what
    stands in the combat phase, which leaves a selection made inside one out of it."""
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
            inside = frame["public"]["phase"] == "combat" or (macro.get("fight") is not None and not by_phase)
            temperature = temperatures["combat" if inside else "other"] if temperatures else 1.0
            chosen = obs.slot_refs.index(slots[step["candidate_ref"]])
            total += float((output.logits / temperature).log_softmax(-1)[chosen])
        result.append(total)
    return result


def largest_gap(model, vocabulary, runs, temperatures=None, by_phase=False):
    """How far a stored log-probability is, at most, from its replay."""
    return max(abs(macro["old_log_prob"] - log_prob)
               for run in runs
               for macro, log_prob in zip(run["macros"], scored(model, vocabulary, run, temperatures, by_phase)))


def replayed_gap(model, vocabulary, runs):
    """How far a stored log-probability is, at most, from the trainer's replay of its
    macro-action at the temperature the macro-action records."""
    macros = [macro for run in runs for macro in run["macros"]]
    prepared = prepare_replay(vocabulary, [macro["steps"] for macro in macros])
    with torch.no_grad():
        log_probs = model.replay(prepared, temperatures=[macro["temperature"] for macro in macros])[0]
    return max(abs(macro["old_log_prob"] - float(log_prob)) for macro, log_prob in zip(macros, log_probs))


def test_choosing_for_a_packed_batch_scores_as_decision_by_decision():
    frames = [map_frame(), duel("enemy:0"), map_frame(("map:1:2", "Elite"))]
    model, vocabulary = build(frames)
    _, decisions = prepare_decisions(vocabulary, frames)
    with torch.no_grad():
        slots, log_probs, _ = model.choose(decisions, sample=False)
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


def crossed(model, *packs):
    """Decisions as the collecting process receives them in one pass: each
    (observations, decisions) packed by a sampling process of its own at the start of a
    round, through a pickle as a pipe sends it."""
    messages = [pickle.loads(pickle.dumps(pack(decisions, observations, Sent(index))))
                for index, (observations, decisions) in enumerate(packs)]
    return unpack(messages, tables(model, len(packs)))


def assert_chosen_alike(model, arrived, decisions):
    """The table pass encodes and chooses as the pass of the decisions' own batch."""
    with torch.no_grad():
        assert torch.allclose(model._entities(arrived.batch), model._entities(decisions.batch), atol=1e-5)
        found, expected = model.choose(arrived, sample=False), model.choose(decisions, sample=False)
    assert found[0].tolist() == expected[0].tolist()
    assert torch.allclose(found[1], expected[1], atol=1e-5)


def test_packs_of_several_processes_are_chosen_for_as_one_pack():
    model, vocabulary = build(FRAMES)
    together = prepare_decisions(vocabulary, FRAMES)[1]
    parts = [prepare_decisions(vocabulary, part) for part in (FRAMES[:1], FRAMES[1:3], FRAMES[3:])]
    joined, unregistered = crossed(model, *parts)
    assert unregistered == [] and joined.batch.program_count == 0
    assert joined.masks.shape == together.masks.shape
    assert same(joined.batch.lengths, together.batch.lengths) and same(joined.batch.relations, together.batch.relations)
    # Each process sent all it holds, its programs regrouped into few groups of their
    # real nodes, each at its row of the padded layout.
    assert len(joined.batch.new_token_rows) == len(joined.batch.token_rows.unique())
    assert len(joined.batch.programs) <= len(GROUPED)
    for group in joined.batch.programs:
        rows = torch.cat([torch.arange(n) + p * group.length for p, n in enumerate(group.lengths.tolist())])
        assert same(group.nodes, rows) and len(group.fields.ids) == len(rows)
    assert [size(pack(d, o, Sent(0))) for o, d in parts] == [1, 2, 2]
    assert_chosen_alike(model, joined, together)


def test_a_process_sends_each_token_and_program_once_a_round():
    model, vocabulary = build(FRAMES)
    sent, table = Sent(0), tables(model, 1)
    sent_tokens, sent_programs = [], []
    for frames in (FRAMES[:2], FRAMES[1:5], FRAMES):
        observations, decisions = prepare_decisions(vocabulary, frames)
        arrived, _ = unpack([pack(decisions, observations, sent)], table)
        sent_tokens.append(len(arrived.batch.new_token_rows))
        sent_programs.append(sum(len(group.lengths) for group in arrived.batch.programs))
        # What was sent before is read from the tables, what is new is written to them.
        assert_chosen_alike(model, arrived, decisions)
    # The second pass shows a map the first did not (in its frontier's program) and
    # sends that alone; the last shows only what was sent: nothing travels but rows.
    assert sent_tokens[0] > 0 and sent_programs[0] > sent_programs[1] > 0
    assert sent_tokens[2] == 0 and sent_programs[2] == 0


def bound_duel(hp):
    """A duel whose card's program binds the enemy it hits, of `hp` hit points."""
    frame = duel("enemy:0")
    for entity in frame["public"]["entities"]:
        if entity["ref"] == "enemy:0":
            entity["hp"] = hp
        if entity["ref"] == "card:0":
            entity["semantic_program"] = {"op": "DAMAGE", "amount": 6,
                                          "target": {"kind": "entity", "ref": "enemy:0"}}
    return frame


def test_a_bound_program_is_sent_again_when_what_it_binds_changes():
    frames = [bound_duel(40), bound_duel(40), bound_duel(30)]
    model, vocabulary = build(frames)
    assert any(effect.bindings for effect in observation(frames[0]).effects)
    sent, table = Sent(0), tables(model, 1)
    sent_programs = []
    for frame in frames:
        observations, decisions = prepare_decisions(vocabulary, [frame])
        arrived, _ = unpack([pack(decisions, observations, sent)], table)
        sent_programs.append(sum(len(group.lengths) for group in arrived.batch.programs))
        assert_chosen_alike(model, arrived, decisions)
    assert sent_programs[0] > 0 and sent_programs[1] == 0
    # The enemy's hit points changed: the program that binds it, and nothing else, again.
    assert sent_programs[2] == 1


def test_a_process_that_fills_its_rows_starts_over_without_spoiling_what_it_sent(monkeypatch):
    """Rows are filled half by half; a pass that changes halves between two packs of one
    process reads the rows the first pack named as they were."""
    monkeypatch.setattr(wire, "TOKEN_ROWS", 52)
    monkeypatch.setattr(wire, "PROGRAM_ROWS", 12)
    frames = FRAMES + [bound_duel(hp) for hp in range(40, 0, -4)]
    model, vocabulary = build(frames)
    sent, table = Sent(1), tables(model, 2)
    sides = set()
    for start in range(12):
        # Two engines of one process: two packs in one message.
        chosen = [frames[(start + k) % len(frames)] for k in (0, 3)]
        parts = [prepare_decisions(vocabulary, [frame]) for frame in chosen]
        messages = []
        for observations, decisions in parts:
            messages.append(pack(decisions, observations, sent))
            sides.add((sent.tokens.side, sent.programs.side))
        arrived, _ = unpack(messages, table)
        assert_chosen_alike(model, arrived, prepare_decisions(vocabulary, chosen)[1])
    assert {tokens for tokens, _ in sides} == {programs for _, programs in sides} == {0, 1}
    observations, decisions = prepare_decisions(vocabulary, frames)
    with pytest.raises(ValueError, match="half"):
        pack(decisions, observations, Sent(0))


def test_a_pack_of_fields_left_as_rows_is_the_pack_of_their_tensors(monkeypatch):
    """A sampling process makes tensors of only the fields it sends (`tensors=False`):
    what it sends is the same, names never registered and a change of halves included."""
    monkeypatch.setattr(wire, "TOKEN_ROWS", 52)
    monkeypatch.setattr(wire, "PROGRAM_ROWS", 12)
    frames = FRAMES + [bound_duel(hp) for hp in range(40, 0, -4)]
    _, vocabulary = build(frames[:1])
    sides = {True: set(), False: set()}
    met = False
    eager, deferred = (deepcopy(vocabulary), Sent(1)), (deepcopy(vocabulary), Sent(1))
    for frame in frames * 2:
        packs = []
        for tensors, (known, sent) in ((True, eager), (False, deferred)):
            observations, decisions = prepare_decisions(known, [frame], tensors=tensors)
            packs.append(pack(decisions, observations, sent))
            sides[tensors].add(sent.tokens.side)
        (header, integers, numbers, names), (header_, integers_, numbers_, names_) = packs
        assert np.array_equal(header, header_) and np.array_equal(integers, integers_)
        assert integers.dtype == integers_.dtype and numbers.dtype == numbers_.dtype
        assert np.array_equal(numbers, numbers_) and names == names_
        met |= bool(names)
    assert met and sides[True] == sides[False] == {0, 1}


def test_names_the_vocabulary_never_registered_reach_the_collecting_process():
    model, vocabulary = build(FRAMES[:1])
    parts = [prepare_decisions(vocabulary, [frame]) for frame in FRAMES[:2]]
    messages = [pack(decisions, observations, Sent(index)) for index, (observations, decisions) in enumerate(parts)]
    assert messages[0][3] == [] and messages[1][3]
    assert unpack(messages, tables(model, 2))[1] == sorted(messages[1][3])


def test_each_decision_is_drawn_at_its_own_temperature():
    frames = [map_frame(), duel("enemy:0"), map_frame(("map:1:2", "Elite"))]
    model, vocabulary = build(frames)
    _, decisions = prepare_decisions(vocabulary, frames)
    temperatures = torch.tensor([0.5, 1.0, 0.25])
    torch.manual_seed(0)
    with torch.no_grad():
        slots, log_probs, base = model.choose(decisions, temperature=temperatures)
        most_likely = model.choose(decisions, sample=False, temperature=temperatures)[0]
        cold = [model.choose(decisions, temperature=1e-6)[0] for _ in range(8)]
    for index, expected in enumerate(logits(model, vocabulary, frames)):
        assert torch.allclose(log_probs[index], (expected / temperatures[index]).log_softmax(-1)[slots[index]], atol=1e-5)
        # Beside it, the same action's log-probability at temperature 1: what a sample is
        # corrected by when it was drawn at another.
        assert torch.allclose(base[index], expected.log_softmax(-1)[slots[index]], atol=1e-5)
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
                              on_run=lambda index, run: done.append((index, run["character"], run["status"])))
        timing = pool.timing
    runs = read(paths)
    # The collecting process accounts for its round: passes of the network, the
    # decisions they carried, and where its seconds went.
    assert timing["passes"] > 0 and timing["decisions"] >= timing["passes"]
    assert timing["decisions"] == sum(len(macro["steps"]) for run in runs for macro in run["macros"]) - sum(
        1 for run in runs for macro in run["macros"] for step in macro["steps"]
        if len(step["frame"]["legal"]["candidates"]) == 1)
    assert timing["decisions_per_pass"] == pytest.approx(timing["decisions"] / timing["passes"], abs=1e-3)
    parts = sum(timing[key] for key in ("waiting_seconds", "unpacking_seconds", "choosing_seconds", "sending_seconds",
                                        "other_seconds"))
    assert 0 < timing["seconds"] and parts == pytest.approx(timing["seconds"], abs=0.01)
    # Each stage of a pass, and where the sampling processes' time went, summed over
    # them: every run played and handed over, every decision with a choice observed,
    # prepared, packed and waited for, every frame played cleaned.
    assert all(timing["collecting"][name]["count"] == timing["passes"] for name in (
        "network_encoder", "network_layout", "network_layers", "network_heads", "readback"))
    sampling = timing["sampling"]
    assert sampling["playing"]["count"] == sampling["handing_over"]["count"] == 20
    assert all(sampling[name]["count"] == timing["decisions"]
               for name in ("observation", "preparing", "wire_packing", "choice_wait"))
    assert sampling["cleaning"]["count"] >= timing["decisions"]
    assert sampling["collector_wait"]["count"] >= timing["passes"]
    assert sampling["playing"]["ms_per_decision"] == pytest.approx(
        1000 * sampling["playing"]["seconds"] / timing["decisions"], abs=1e-2)
    assert [run["status"] for run in runs] == ["complete"] * 20
    # Every job is announced once as it ends, by its index, with the run's summary.
    assert sorted(done) == [(index, run["character"], "complete") for index, run in enumerate(runs)]
    # Every run beside its journal; the assignment before any of them.
    assert all(path.with_suffix(".jsonl").exists() for path in paths)
    assert json.loads((tmp_path / "round/metadata/seeds.json").read_text()) == seeds
    for run in runs:
        validate_run(run, on_policy=True)
    # Exact arithmetic: every stored probability is the learner's own, at the
    # temperatures the round was sampled at, which every run records.
    assert all(run["temperature"] == {"combat": 1.0, "other": 1.0} for run in runs)
    assert largest_gap(learner.model, learner.vocabulary, runs) < 1e-4
    # At temperature 1 the log-probability a macro-action was drawn with is its own.
    macros = [macro for run in runs for macro in run["macros"] if "old_log_prob" in macro]
    assert macros and all(abs(macro["base_log_prob"] - macro["old_log_prob"]) < 1e-5 for macro in macros)


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


def test_resting_ends_the_engines_and_the_next_round_starts_them_again(tmp_path):
    _, runner, seeds = setup_round()
    starts = tmp_path / "starts"

    def counted():
        # Told from a sampling process: one line for every engine started.
        with starts.open("a") as stream:
            stream.write(f"{os.getpid()}\n")
        return engine()

    with samplers(runner, counted, engines=2, processes=2) as pool:
        first = read(collect_round(pool, runner, seeds, tmp_path / "first", sample=False))
        pool.rest()
        pool.rest()
        second = read(collect_round(pool, runner, seeds, tmp_path / "second", sample=False))
        processes = [process.pid for process in pool.samplers]
    lines = starts.read_text().split()
    # Each process started one engine for each round, and stayed.
    assert len(lines) == 4 and set(map(int, lines)) == set(processes)
    assert [run["status"] for run in first + second] == ["complete"] * (len(first) + len(second))


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
    assert [(macro["branches"], macro["decisions"]) for macro in handed["macros"]] \
        == [(sum(not step["forced"] for step in macro["steps"]), len(macro["steps"])) for macro in run["macros"]]
    # A run records where it stood as it is played; the record reads back as the frames do.
    assert climb(handed) == climb(run) == ({1: max(run["climb"]["reached"].values())}, {1: 17})
    assert climb(run) == climbed(stand(step["frame"]) for macro in run["macros"] for step in macro["steps"])
    assert climb(run)[0][1] > 1 and climb(json.loads(json.dumps(run))) == climb(run)
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
    # Every macro-action records the temperature it was drawn at; replayed at it, as
    # the trainer does, it has the log-probability it was drawn with.
    assert {(macro["phase"], macro["temperature"]) for run in runs for macro in run["macros"]} == {
        ("combat", 0.5), ("combat_play", 0.8)}
    assert replayed_gap(learner.model, learner.vocabulary, runs) < 1e-4
    # Scored at any other temperatures the round is another policy's.
    assert largest_gap(learner.model, learner.vocabulary, runs) > 1e-3
    assert largest_gap(learner.model, learner.vocabulary, runs, {"combat": 0.8, "other": 0.5}) > 1e-3


def test_a_selection_inside_a_fight_is_drawn_at_the_fights_temperature(tmp_path):
    temperatures = {"combat": 0.5, "other": 0.8}
    learner, runner, seeds = setup_round(temperatures)
    # The plain engine's fight is a selection in the `card_select` phase: the engine's
    # encounter events, not the phase, say that it is inside a fight.
    with samplers(runner, engines=5, processes=3) as pool:
        runs = read(collect_round(pool, runner, seeds, tmp_path / "round"))
    assert all([(macro["phase"], macro["fight"], macro["temperature"]) for macro in run["macros"]]
               == [("card_select", 1, 0.5), ("combat_play", None, 0.8)] for run in runs)
    assert largest_gap(learner.model, learner.vocabulary, runs, temperatures) < 1e-4
    assert replayed_gap(learner.model, learner.vocabulary, runs) < 1e-4
    assert largest_gap(learner.model, learner.vocabulary, runs, temperatures, by_phase=True) > 1e-3
    # The runner alone draws at the same temperatures as the sampling processes.
    alone = runner.run(engine(), "Ironclad", "alone")
    assert largest_gap(learner.model, learner.vocabulary, [alone], temperatures) < 1e-4
    frame = alone["macros"][0]["steps"][0]["frame"]
    assert (runner.temperature_for(frame), runner.temperature_for(frame, fight=1)) == (0.8, 0.5)
    assert runner.temperature_for(frame, sample=False, fight=1) == 1.0


def test_greedy_runs_have_no_temperature():
    learner, runner, _ = setup_round({"combat": 0.5, "other": 0.8})
    run = runner.run(fighting(), "Ironclad", "greedy", sample=False)
    assert run["status"] == "complete" and run["temperature"] is None
    assert all(macro["temperature"] == 1.0 for macro in run["macros"])
    # Its probabilities are those of the most likely actions at temperature 1.
    assert largest_gap(learner.model, learner.vocabulary, [run]) < 1e-5
    with pytest.raises(ValueError, match="temperature"):
        RolloutRunner(learner.model, learner.vocabulary, temperatures={"combat": 0, "other": 1})
    with pytest.raises(ValueError, match="temperature"):
        RolloutRunner(learner.model, learner.vocabulary, temperatures={"combat": 0.5})
    with pytest.raises(ValueError, match="temperature"):
        TrainConfig(combat_temperature=1.5)
    assert TrainConfig(combat_temperature=0.3, other_temperature=0.7).temperatures == {"combat": 0.3, "other": 0.7}


def test_the_shard_line_of_a_run_goes_to_on_run_and_is_not_kept(tmp_path):
    from main.play.rollout import assign

    _, runner, seeds = setup_round()
    jobs = assign(seeds, tmp_path / "round")
    seen = []
    with samplers(runner, engines=5, processes=3) as pool:
        played = pool.collect(runner, jobs, sample=True,
                              on_run=lambda index, run: seen.append(("member" in run, run.get("digest"))),
                              layout=dict(trajectory=False, journal=False, combat_frames=False))
    assert len(seen) == len(jobs) and all(member for member, _ in seen)
    # These frames show no player's act and floor: no digest, the advantages read the shards.
    assert all(digest is None for _, digest in seen)
    assert all(run["status"] == "complete" and "member" not in run for run in played)


def test_the_graphed_global_layers_choose_as_the_eager_ones_do():
    """On the GPU the global layers of a sampling pass replay as CUDA graphs on a
    padded batch; the greedy choices are the eager pass's and the log-probabilities
    within the padding's rounding. On the CPU the switch does nothing."""
    frames = [map_frame(), duel("enemy:0"), map_frame(("map:1:2", "Elite"))]
    model, vocabulary = build(frames)
    observations, decisions = prepare_decisions(vocabulary, frames)
    model.eval()
    with torch.no_grad():
        eager = model.choose(decisions, sample=False)
        model.graphed(True)
        assert model._graphs is not None
        again = model.choose(decisions, sample=False)
        assert torch.equal(eager[0], again[0]) and torch.allclose(eager[1], again[1])
        assert not model._graphs.graphs  # nothing is captured on the CPU
        if torch.cuda.is_available():
            model.to("cuda")
            cuda, _ = crossed(model, (observations, decisions))
            model.graphed(False)
            plain = model.choose(cuda, sample=False)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                half = model.choose(cuda, sample=False)
            model.graphed(True)
            graphed = model.choose(cuda, sample=False)
            assert len(model._graphs.graphs) == 1
            assert torch.equal(plain[0], graphed[0]) and torch.allclose(plain[1], graphed[1], atol=1e-2)
            # A second pass of the same shape replays the one graph.
            graphed = model.choose(cuda, sample=False)
            assert len(model._graphs.graphs) == 1 and torch.equal(plain[0], graphed[0])
            # A pass under autocast is another graph of the shape: what it captured
            # is the autocast pass, and the plain graph stays the plain pass.
            with torch.autocast("cuda", dtype=torch.bfloat16):
                half_graphed = model.choose(cuda, sample=False)
            assert len(model._graphs.graphs) == 2
            assert torch.equal(half[0], half_graphed[0]) and torch.allclose(half[1], half_graphed[1], atol=1e-2)
            again = model.choose(cuda, sample=False)
            assert len(model._graphs.graphs) == 2 and torch.allclose(plain[1], again[1], atol=1e-2)
        model.graphed(False)
        assert model._graphs is None


class Drinking(Fighting):
    """A fighting engine whose fight offers a potion once: using it is a decision of
    its own that changes nothing else."""

    def reset(self, character="Ironclad", seed="test", ascension=10):
        self.potion = True
        return super().reset(character, seed, ascension)

    def frame(self):
        frame = super().frame()
        if self.stage == "selection" and self.potion:
            frame["public"]["entities"].append({"ref": "potion:0", "entity_type": "potion", "slot": 0,
                                                "content_id": "POTION.BLOCK_POTION", "owner_ref": "player"})
            use = {"verb": "USE_POTION", "decoder_slot_ref": "potion:0", "source_refs": ["potion:0"],
                   "target_refs": [], "effect_coverage": "opaque"}
            frame["public"]["decoder_bank"].append(use)
            candidates = frame["legal"]["candidates"]
            candidates.append(dict(use, candidate_ref=f"c{len(candidates)}"))
        return frame

    def send(self, command):
        before = self.frame()
        chosen = next((c for c in before["legal"]["candidates"] if c["candidate_ref"] == command.get("candidate_ref")), None)
        if chosen is not None and chosen["verb"] == "USE_POTION":
            if command != execution_command(before, command["candidate_ref"]):
                raise ProtocolError("stale_decision")
            self.commands.append(deepcopy(command))
            self.version += 1
            self.potion = False
            return self.frame()
        return super().send(command)


def drinking():
    return Drinking(4, 2)


def explored_steps(run):
    return [step for macro in run["macros"] for step in macro["steps"] if step.get("explored")]


def verb_of(step):
    return next(c["verb"] for c in step["frame"]["legal"]["candidates"] if c["candidate_ref"] == step["candidate_ref"])


def test_a_decision_marked_to_explore_is_drawn_among_its_marked_slots_with_the_policys_log_probability():
    frames = [map_frame(), duel("enemy:0"), map_frame(("map:1:2", "Elite"))]
    model, vocabulary = build(frames)
    observations, decisions = prepare_decisions(vocabulary, frames, restrict=[None, ["play:enemy:1"], None])
    slot = observation(frames[1]).slot_refs.index("play:enemy:1")
    assert decisions.explore[1].nonzero().tolist() == [[slot]] and not decisions.explore[[0, 2]].any()
    expected = logits(model, vocabulary, frames)[1]
    torch.manual_seed(0)
    with torch.no_grad():
        for _ in range(8):
            slots, log_probs, base = model.choose(decisions, temperature=0.5)
            # The marked decision always draws its marked slot, and is recorded with the
            # whole distribution's log-probability of it, as if the policy had drawn it.
            assert int(slots[1]) == slot
            assert torch.allclose(log_probs[1], (expected / 0.5).log_softmax(-1)[slot], atol=1e-5)
            assert torch.allclose(base[1], expected.log_softmax(-1)[slot], atol=1e-5)
            assert all(bool(decisions.masks[i, slots[i]]) for i in (0, 2))
        # Greedy choices ignore the marks.
        assert int(model.choose(decisions, sample=False)[0][1]) == int(expected.argmax())
    # The marks cross to the collecting process with the pack.
    arrived, _ = crossed(model, (observations, decisions))
    assert same(arrived.explore, decisions.explore) and same(arrived.masks, decisions.masks)
    with pytest.raises(ProtocolError, match="restriction"):
        prepare_decisions(vocabulary, frames, restrict=[None, ["play:enemy:9"], None])


def test_a_decision_marked_with_a_rank_takes_the_slot_of_that_rank_with_the_policys_log_probability():
    frames = [map_frame(), duel("enemy:0"), map_frame(("map:1:2", "Elite"))]
    model, vocabulary = build(frames)
    observations, decisions = prepare_decisions(vocabulary, frames, rank=[1, None, 0])
    assert decisions.rank.tolist() == [1, -1, 0]
    expected = logits(model, vocabulary, frames)
    legal = [observation(frame).slot_refs for frame in frames]
    torch.manual_seed(0)
    with torch.no_grad():
        for _ in range(4):
            slots, log_probs, base = model.choose(decisions, temperature=0.5)
            for row, rank in ((0, 1), (2, 0)):
                masked = expected[row].masked_fill(~decisions.masks[row, :len(expected[row])], float("-inf"))
                wanted = int(masked.argsort(descending=True)[rank])
                # The slot of the rank, whatever the draw, recorded with the whole tempered
                # distribution's log-probability of it.
                assert int(slots[row]) == wanted
                assert torch.allclose(log_probs[row], (masked / 0.5).log_softmax(-1)[wanted], atol=1e-5)
                assert torch.allclose(base[row], masked.log_softmax(-1)[wanted], atol=1e-5)
            assert bool(decisions.masks[1, slots[1]])
        # Greedy choices ignore the ranks.
        assert int(model.choose(decisions, sample=False)[0][0]) == int(expected[0].argmax())
    assert len(legal[0]) >= 2
    # The ranks cross to the collecting process with the pack, beside other packs.
    arrived, _ = crossed(model, (observations, decisions), prepare_decisions(vocabulary, frames[:1]))
    assert arrived.rank.tolist() == [1, -1, 0, -1]
    # The single-process path takes the same slot.
    from main.network.policy import SessionPolicy
    for row, rank in ((0, 1), (2, 0)):
        routed = dict(frames[row], routing={"episode_id": "e", "decision_id": "d", "state_version": 1})
        choice = SessionPolicy(model, vocabulary).choose(routed, temperature=0.5, rank=rank)
        slot = next(c["decoder_slot_ref"] for c in frames[row]["legal"]["candidates"]
                    if c["candidate_ref"] == choice.candidate_ref)
        assert observation(frames[row]).slot_refs.index(slot) == int(slots[row]) and choice.forked
        assert not choice.explored
    with pytest.raises(ProtocolError, match="rank"):
        prepare_decisions(vocabulary, frames, rank=[len(legal[0]), None, None])
    with pytest.raises(ProtocolError, match="rank"):
        prepare_decisions(vocabulary, frames, rank=[1, None, 0], restrict=[["move:map:0:1"], None, None])


def test_a_run_explores_potion_use_at_the_rate_asked_and_records_the_policys_log_probability():
    learner, _, _ = setup_round({"combat": 0.5, "other": 0.8})
    exploring = RolloutRunner(learner.model, learner.vocabulary, precision="no", ascension=0,
                              temperatures={"combat": 0.5, "other": 0.8}, explore_potions=1.0)
    run = exploring.run(drinking(), "Ironclad", "drink")
    assert run["status"] == "complete" and run["explore_potions"] == {"normal": 1.0, "elite": 1.0, "boss": 1.0}
    # The fight's first decision could use the potion; exploring at rate 1 it does, once,
    # and the step says so. The decision is still the policy's as the record has it: its
    # log-probability is what the tempered policy gives the action, so the trainer's
    # replay matches it like any other.
    explored = explored_steps(run)
    assert len(explored) == 1 and verb_of(explored[0]) == "USE_POTION"
    assert explored[0] is run["macros"][0]["steps"][0]
    assert all(verb_of(step) != "USE_POTION" for macro in run["macros"] for step in macro["steps"] if not step.get("explored"))
    assert replayed_gap(learner.model, learner.vocabulary, [run]) < 1e-4
    assert _without_frames(run)["explored"] == 1
    validate_run(run, on_policy=True)
    # The training sample of the macro-action says it was explored.
    from main.data.shards import samples as training_samples
    flagged = [item.explored for item in training_samples([run])]
    assert flagged[0] and not any(flagged[1:])
    assert not any(item.forked for item in training_samples([run]))
    # Outside sampling, or at rate 0, nothing is explored and nothing is marked.
    greedy = exploring.run(drinking(), "Ironclad", "greedy", sample=False)
    assert greedy["explore_potions"] is None and not explored_steps(greedy)
    plain = RolloutRunner(learner.model, learner.vocabulary, precision="no", ascension=0,
                          temperatures={"combat": 0.5, "other": 0.8})
    run = plain.run(drinking(), "Ironclad", "plain")
    assert run["explore_potions"] == {"normal": 0.0, "elite": 0.0, "boss": 0.0}
    assert not explored_steps(run) and _without_frames(run)["explored"] == 0
    assert all("explored" not in step for macro in run["macros"] for step in macro["steps"])
    for wrong in (-0.1, 1.5, [0.1, 0.2, 0.3, 0.4], {"event": 0.1}):
        with pytest.raises(ValueError, match="exploration"):
            RolloutRunner(learner.model, learner.vocabulary, explore_potions=wrong)


def branching_steps(run):
    return [step for macro in run["macros"] for step in macro["steps"] if not step["forced"]]


def rank_of(model, vocabulary, step):
    """Where the step's candidate stands in the policy's order of the frame's legal candidates."""
    frame = step["frame"]
    obs = observation(frame)
    slots = {c["candidate_ref"]: c["decoder_slot_ref"] for c in frame["legal"]["candidates"]}
    mask = torch.tensor([slot in set(slots.values()) for slot in obs.slot_refs])
    with torch.no_grad():
        output = model.decode_batch(model.encode([obs], vocabulary), [mask])[0]
    order = output.logits.argsort(descending=True).tolist()
    return order.index(obs.slot_refs.index(slots[step["candidate_ref"]]))


def test_a_segment_forks_its_first_decisions_by_its_place_in_the_group(tmp_path):
    learner, runner, _ = setup_round({"combat": 0.5, "other": 0.8})
    model, vocabulary = learner.model, learner.vocabulary
    runs = [runner.run(engine(), "Ironclad", "fork", fork=dict(index=j, size=4, depth=1)) for j in range(4)]
    for j, run in enumerate(runs):
        assert run["status"] == "complete" and run["fork"] == dict(index=j, size=4, depth=1)
        first, *rest = branching_steps(run)
        # The first half of the group takes the policy's most likely candidate, the
        # other half its second; the step says it was forked, the rest are drawn.
        assert first["forked"] and rank_of(model, vocabulary, first) == (0 if j < 2 else 1)
        assert not any(step.get("forked") for step in rest) and not first.get("explored")
        assert _without_frames(run)["forked"] == 1
        validate_run(run, on_policy=True)
    # Recorded with the policy's own log-probability, as a draw would be.
    assert replayed_gap(model, vocabulary, runs) < 1e-4
    from main.data.shards import samples as training_samples
    forked_sample = training_samples([runs[2]])[0]
    assert forked_sample.forked and not forked_sample.explored
    # A depth of two parts each half again; greedy runs and runs without a fork never fork.
    deeper = [runner.run(engine(), "Ironclad", "fork", fork=dict(index=j, size=4, depth=2)) for j in range(4)]
    assert [[rank_of(model, vocabulary, step) for step in branching_steps(run)[:2]] for run in deeper] == [
        [0, 0], [0, 1], [1, 0], [1, 1]]
    greedy = runner.run(engine(), "Ironclad", "fork", sample=False, fork=dict(index=3, size=4, depth=1))
    assert greedy["fork"] is None and not any(step.get("forked") for step in branching_steps(greedy))
    plain = runner.run(engine(), "Ironclad", "fork")
    assert plain["fork"] is None and not any(step.get("forked") for step in branching_steps(plain))
    for wrong in (dict(index=4, size=4, depth=1), dict(index=0, size=4, depth=-1), dict(index=0, size=4)):
        with pytest.raises(ValueError, match="fork"):
            runner.run(engine(), "Ironclad", "fork", fork=wrong)
    # The sampling processes fork as the runner alone does.
    from main.play.rollout import SEGMENTS
    jobs = [("Ironclad", dict(seed="fork", run_id=f"g-{j}", group="g", fork=dict(index=j, size=4, depth=1)), None)
            for j in range(4)]
    with samplers(runner, engine, engines=4, processes=2) as pool:
        played = []
        pool.collect(runner, jobs, on_run=lambda index, run: played.append((index, run)), layout=SEGMENTS)
    assert sorted(run["forked"] for _, run in played) == [1, 1, 1, 1]


class EliteDrinking(Drinking):
    """The same fight, an elite's."""

    def enter(self, encounter, act=1, kind="normal"):
        super().enter(encounter, act, "elite")


def test_the_exploration_rate_is_the_fights_kinds(tmp_path):
    learner, _, seeds = setup_round({"combat": 0.5, "other": 0.8})
    # Two rates: normal fights, then elite and boss fights.
    runner = RolloutRunner(learner.model, learner.vocabulary, version=learner.policy_version, precision="no",
                           ascension=0, temperatures={"combat": 0.5, "other": 0.8}, explore_potions=[0.0, 1.0])
    assert runner.explore_potions == {"normal": 0.0, "elite": 1.0, "boss": 1.0}
    assert not explored_steps(runner.run(drinking(), "Ironclad", "normal-fight"))
    elite = runner.run(EliteDrinking(4, 2), "Ironclad", "elite-fight")
    assert len(explored_steps(elite)) == 1 and verb_of(explored_steps(elite)[0]) == "USE_POTION"
    assert elite["explore_potions"] == {"normal": 0.0, "elite": 1.0, "boss": 1.0}
    # By kind, with a kind left out exploring nothing; and in the sampling processes alike.
    by_kind = RolloutRunner(learner.model, learner.vocabulary, version=learner.policy_version, precision="no",
                            ascension=0, temperatures={"combat": 0.5, "other": 0.8}, explore_potions={"elite": 1.0})
    assert by_kind.explore_potions == {"normal": 0.0, "elite": 1.0, "boss": 0.0}
    with samplers(by_kind, lambda: EliteDrinking(4, 2), engines=5, processes=3) as pool:
        runs = read(collect_round(pool, by_kind, seeds, tmp_path / "round"))
    assert all(len(explored_steps(run)) == 1 for run in runs)
    with samplers(by_kind, drinking, engines=5, processes=3) as pool:
        runs = read(collect_round(pool, by_kind, seeds, tmp_path / "normal"))
    assert not any(explored_steps(run) for run in runs)


def test_sampling_processes_explore_potion_use_as_the_runner_alone_does(tmp_path):
    learner, _, seeds = setup_round({"combat": 0.5, "other": 0.8})
    runner = RolloutRunner(learner.model, learner.vocabulary, version=learner.policy_version, precision="no",
                           ascension=0, temperatures={"combat": 0.5, "other": 0.8}, explore_potions=1.0)
    with samplers(runner, drinking, engines=5, processes=3) as pool:
        runs = read(collect_round(pool, runner, seeds, tmp_path / "round"))
    assert all(run["explore_potions"] == {"normal": 1.0, "elite": 1.0, "boss": 1.0} for run in runs)
    assert all(len(explored_steps(run)) == 1 and verb_of(explored_steps(run)[0]) == "USE_POTION" for run in runs)
    assert replayed_gap(learner.model, learner.vocabulary, runs) < 1e-4


def greedy_runner(learner, temperatures, **settings):
    return RolloutRunner(learner.model, learner.vocabulary, version=learner.policy_version, precision="no",
                         ascension=0, temperatures=temperatures, greedy_fights=True, **settings)


class Asking:
    """A policy that says how each decision was asked of it: its phase, whether it is
    drawn and at what temperature."""

    def __init__(self, policy):
        self.policy, self.asked = policy, []

    def reset(self):
        self.policy.reset()

    def choose(self, frame, **asked):
        self.asked.append((frame["public"]["phase"], asked["sample"], asked["temperature"]))
        return self.policy.choose(frame, **asked)


def test_greedy_fights_take_the_fights_most_likely_actions_and_draw_the_rest(tmp_path):
    from main.network.policy import SessionPolicy

    temperatures = {"combat": 0.5, "other": 0.8}
    learner, runner, seeds = setup_round(temperatures)
    model, vocabulary = learner.model, learner.vocabulary
    greedy = greedy_runner(learner, temperatures)
    # The plain engine's fight is its selection (`card_select`); the decision after it is outside.
    asking = Asking(SessionPolicy(model, vocabulary))
    run = greedy.run(engine(), "Ironclad", "asked", _policy=asking)
    assert {asked for asked in asking.asked} == {("card_select", False, 1.0), ("combat_play", True, 0.8)}
    # Without the switch every decision is drawn, at its phase's temperature, as before.
    plain = Asking(SessionPolicy(model, vocabulary))
    runner.run(engine(), "Ironclad", "asked", _policy=plain)
    assert {asked for asked in plain.asked} == {("card_select", True, 0.5), ("combat_play", True, 0.8)}
    assert "greedy_fights" not in runner.run(engine(), "Ironclad", "plain")

    torch.manual_seed(0)
    alone = [greedy.run(engine(), "Ironclad", f"greedy-{index}") for index in range(8)]
    with samplers(greedy, engines=5, processes=3) as pool:
        apart = read(collect_round(pool, greedy, seeds, tmp_path / "round"))
    for run in alone + apart:
        assert run["status"] == "complete" and run["greedy_fights"] is True and run["temperature"] == temperatures
        fights = [macro for macro in run["macros"] if macro["fight"] is not None]
        rest = [macro for macro in run["macros"] if macro["fight"] is None]
        assert fights and rest
        # Inside the fight the most likely action, in the sampling processes as alone; the
        # macro-action records no temperature: it was not drawn.
        assert all(rank_of(model, vocabulary, step) == 0 for macro in fights for step in branching_steps(dict(macros=[macro])))
        assert all(macro["temperature"] is None for macro in fights)
        assert all(macro["temperature"] == 0.8 for macro in rest)
        assert not any(step.get("explored") or step.get("forked") for step in branching_steps(run))
        validate_run(run, on_policy=True)
    # Its log-probability is the action's at 1, as a greedy run's; the rest's at its temperature.
    assert largest_gap(model, vocabulary, alone + apart, {"combat": 1.0, "other": 0.8}) < 1e-4
    # Outside the fight the decisions are drawn: not always the most likely candidate.
    outside = [rank_of(model, vocabulary, step) for run in alone for macro in run["macros"] if macro["fight"] is None
               for step in branching_steps(dict(macros=[macro]))]
    assert outside and any(outside)
    # Greedy runs are greedy throughout, as they were; the switch takes no exploration.
    assert "greedy_fights" not in greedy.run(engine(), "Ironclad", "evaluated", sample=False)
    with pytest.raises(ValueError, match="explore"):
        greedy_runner(learner, temperatures, explore_potions=0.1)
    with pytest.raises(ValueError, match="Greedy"):
        RolloutRunner(model, vocabulary, greedy_fights=1)


def test_greedy_fights_fork_at_the_first_decisions_outside_them(tmp_path):
    learner, _, _ = setup_round({"combat": 0.5, "other": 0.8})
    model, vocabulary = learner.model, learner.vocabulary
    greedy = greedy_runner(learner, {"combat": 0.5, "other": 0.8})
    runs = [greedy.run(engine(), "Ironclad", "fork", fork=dict(index=j, size=4, depth=1)) for j in range(4)]
    for j, run in enumerate(runs):
        fight, after = run["macros"]
        assert fight["fight"] is not None and after["fight"] is None
        # The fight's decisions come first and are taken by argmax, unforked; the fork
        # falls on the first decision with a choice outside it.
        assert not any(step.get("forked") for step in fight["steps"])
        assert all(rank_of(model, vocabulary, step) == 0 for step in branching_steps(dict(macros=[fight])))
        first = branching_steps(dict(macros=[after]))[0]
        assert first["forked"] and rank_of(model, vocabulary, first) == (0 if j < 2 else 1)
        assert _without_frames(run)["forked"] == 1
    # The sampling processes fork as the runner alone does.
    from main.play.rollout import SEGMENTS
    jobs = [("Ironclad", dict(seed="fork", run_id=f"g-{j}", group="g", fork=dict(index=j, size=4, depth=1)), None)
            for j in range(4)]
    with samplers(greedy, engine, engines=4, processes=2) as pool:
        played = []
        pool.collect(greedy, jobs, on_run=lambda index, run: played.append((index, run)), layout=SEGMENTS)
    assert sorted(run["forked"] for _, run in played) == [1, 1, 1, 1]
