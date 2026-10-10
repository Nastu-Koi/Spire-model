"""Advantages along a run's decisions outside fights, the weights they give the
sampled half of an update, and the batches that mix it with the supervised half."""

import gzip
import json
import math
from dataclasses import replace

import pytest

from model import advantage, awr
from model.advantage import FILE, from_values, gae
from model.data import RunShards, Sample, group_weights, samples, split_held_out
from model.protocol import ProtocolError
from model.tests import journeys


def test_advantages_are_discounted_sums_of_temporal_differences_ending_in_the_return():
    values = [0.1, -0.2, 0.4]
    # One step: the next decision's value, or the return after the last decision.
    assert gae(values, 1.0, lam=0.0) == pytest.approx([-0.3, 0.6, 0.6])
    # All the way: the return minus the value, whatever lies between.
    assert gae(values, 1.0, lam=1.0) == pytest.approx([0.9, 1.2, 0.6])
    assert gae(values, -1.0, lam=1.0) == pytest.approx([-1.1, -0.8, -1.4])
    last = 1.0 - 0.4
    middle = (0.4 + 0.2) + 0.9 * last
    assert gae(values, 1.0) == pytest.approx([(-0.2 - 0.1) + 0.9 * middle, middle, last])
    assert gae([], 1.0) == [] and gae([0.25], -1.0) == pytest.approx([-1.25])
    with pytest.raises(ValueError):
        gae(values, 1.0, lam=1.5)


def test_the_next_state_is_the_next_decision_outside_a_fight():
    # Decisions at macro-actions 1, 3 and 5 of a run: the fights between them hold
    # none, so each difference spans a fight.
    decisions = [("a", 1, True), ("a", 3, True), ("a", 5, True), ("b", 0, False), ("b", 4, False)]
    values = [0.0, 0.5, 0.25, -0.5, 0.0]
    runs = from_values(decisions, values, lam=0.5)
    assert [index for index, _, _ in runs["a"]] == [1, 3, 5]
    assert [a for _, _, a in runs["a"]] == pytest.approx(gae([0.0, 0.5, 0.25], 1.0, 0.5))
    assert runs["a"][0][2] == pytest.approx((0.5 - 0.0) + 0.5 * ((0.25 - 0.5) + 0.5 * (1.0 - 0.25)))
    # A lost run ends at -1, and its first decision's difference is to the decision after the fight.
    assert [a for _, _, a in runs["b"]] == pytest.approx([(0.0 + 0.5) + 0.5 * (-1.0 - 0.0), -1.0])
    with pytest.raises(ValueError, match="order"):
        from_values([("a", 3, True), ("a", 1, True)], [0.0, 0.0])
    with pytest.raises(ValueError, match="one outcome"):
        from_values([("a", 1, True), ("a", 3, False)], [0.0, 0.0])
    with pytest.raises(ValueError):
        from_values(decisions, values[:-1])


def test_a_rounds_advantages_are_kept_beside_it_with_what_made_them(tmp_path):
    policy = journeys.learner()
    directory = tmp_path / "round-0"
    summary = journeys.sampled_round(policy, directory)
    shards = sorted(path.read_bytes() for path in (directory / "accepted").iterdir())
    # The samples of a round are its decisions outside fights: every other macro-action.
    places = list(advantage.decision_places(directory))
    assert len(places) == summary["accepted_runs"] == 10 and all(place == [1, 3, 5] for _, _, _, place in places)
    values = {run: [0.1 * index for index in place] for run, _, _, place in places}
    identity = dict(kind=advantage.CRITIC, lam=0.9, critic=dict(version=3, weights="abc"))
    runs = from_values([(run, index, won) for run, _, won, place in places for index in place],
                       [value for run, _, _, _ in places for value in values[run]])
    header = advantage.write(directory, identity, runs)
    assert (header["format"], header["kind"], header["critic"], header["lam"]) == (
        "rollout-advantages-v1", "critic", dict(version=3, weights="abc"), 0.9)
    assert (header["runs"], header["decisions"], header["advantage"]["n"]) == (10, 30, 30)
    assert advantage.header(directory) == header and advantage.current(directory, identity)
    assert not advantage.current(directory, dict(identity, lam=0.8))
    assert not advantage.current(directory, dict(identity, critic=dict(version=4, weights="abc")))
    table = advantage.read([directory])
    run, _, won, _ = places[0]
    assert set(table) == {(run, index) for run, _, _, place in places for index in place}
    assert [table[run, index] for index in (1, 3, 5)] == pytest.approx(
        gae([0.1, 0.3, 0.5], 1.0 if won else -1.0), abs=1e-6)
    # The shards are as they were, and every sample of the round finds its advantage.
    assert sorted(path.read_bytes() for path in (directory / "accepted").iterdir()) == shards
    items = samples(next(RunShards(directory).windows(8)))
    assert {(item.run_id, item.index) for item in items} == set(table)
    with pytest.raises(FileNotFoundError, match="no advantages"):
        advantage.read([tmp_path / "round-1"])


def test_without_a_critic_the_advantage_is_the_return_minus_the_characters_mean(tmp_path):
    policy = journeys.learner()
    first, second = tmp_path / "round-0", tmp_path / "round-1"
    journeys.sampled_round(policy, first, 0)
    journeys.sampled_round(policy, second, 1)
    headers = advantage.write_returns([first, second])
    baseline = advantage.mean_returns([first, second])
    assert [header["kind"] for header in headers] == ["return", "return"] and headers[0]["critic"] is None
    assert headers[0]["baseline"] == pytest.approx(baseline)
    table = advantage.read([first, second])
    for directory in (first, second):
        for run, character, won, place in advantage.decision_places(directory):
            assert [table[run, index] for index in place] == pytest.approx(
                [(1.0 if won else -1.0) - baseline[character]] * 3, abs=1e-6)
    # A round that holds them is left alone until the baseline moves.
    written = (first / FILE).stat().st_mtime_ns
    assert advantage.write_returns([first, second]) == headers and (first / FILE).stat().st_mtime_ns == written
    with gzip.open(first / FILE, "rt") as stream:
        assert json.load(stream)["format"] == advantage.FORMAT


def spread(count, largest):
    return [largest * index / (count - 1) for index in range(count)]


def test_the_effective_sample_size_grows_with_beta():
    advantages = spread(40, 2.0) + [5.0]
    sizes = [awr.effective_size(advantages, beta) for beta in (0.01, 0.1, 0.3, 1.0, 10.0, 1000.0)]
    assert sizes == sorted(sizes) and sizes[0] == pytest.approx(1.0, abs=1e-6) and sizes[-1] == pytest.approx(41, abs=0.1)
    assert awr.effective_size(advantages, math.inf) == 41
    assert awr.effective_size([0.3] * 7, 0.01) == pytest.approx(7)
    # Shifting every advantage changes nothing: only their differences weigh.
    assert awr.effective_size([a - 100 for a in advantages], 0.3) == pytest.approx(awr.effective_size(advantages, 0.3))


def test_beta_is_the_smallest_that_keeps_the_effective_share_within_its_bounds():
    advantages = spread(63, 1.0) + [3.0]
    target = 0.3 * len(advantages)
    beta = awr.tune(advantages)
    assert awr.BETA_MIN < beta < awr.BETA_MAX
    assert awr.effective_size(advantages, beta) >= target > awr.effective_size(advantages, beta * 0.99)
    # A larger share needs flatter weights.
    assert awr.tune(advantages, share=0.6) > beta > awr.tune(advantages, share=0.1)
    # Advantages already even enough stay at the lower bound: small differences are not sharpened further.
    assert awr.tune(spread(64, 0.05)) == awr.BETA_MIN and awr.tune([0.0] * 8) == awr.BETA_MIN
    assert awr.tune(spread(64, 0.05), low=0.001) < 0.05
    # What no β of the range can even out ends at the upper bound.
    assert awr.tune([0.0] * 9 + [1e6]) == awr.BETA_MAX
    assert awr.tune(advantages, low=0.5, high=0.5) == 0.5
    with pytest.raises(ValueError):
        awr.tune([])
    rule = awr.Temperature.parse("auto", share=0.6)
    assert rule.of(advantages) == pytest.approx(awr.tune(advantages, share=0.6)) and rule.describe()["rule"] == "tuned"
    assert awr.Temperature.parse("0.25").of(advantages) == 0.25 and awr.Temperature.parse("0.25").describe() == dict(rule="fixed", beta=0.25)
    assert math.isinf(awr.Temperature.parse("inf").of(advantages)) and awr.Temperature.parse("inf").describe() == dict(rule="none")
    for wrong in (dict(fixed=0.0), dict(share=0.0), dict(low=2.0, high=1.0), dict(high=math.inf)):
        with pytest.raises(ValueError):
            awr.Temperature(**wrong)


def item(run, index, weight, source="rollout"):
    return Sample(None, "Ironclad", run, weight, "map", 1, 0, source, index)


def test_a_decision_of_a_small_group_of_the_buffer_weighs_no_more_than_the_cap():
    def group(character, phase):
        return character, phase, "rollout"

    # A phase met twice in a round would give each of its decisions a sixth of the buffer's weight.
    counts = {group("Ironclad", "map"): 1000, group("Ironclad", "shop"): 500, group("Silent", "bundle_select"): 2}
    plain = awr.buffer_weights(counts, cap=0)
    assert plain == group_weights(counts) and plain[group("Silent", "bundle_select")] == pytest.approx(1 / 6)
    even = 1 / 1502
    weights = awr.buffer_weights(counts)
    assert weights[group("Silent", "bundle_select")] == pytest.approx(awr.WEIGHT_CAP * even)
    # What it leaves of its share is the other groups', equally.
    left = 1 - 2 * awr.WEIGHT_CAP * even
    assert weights[group("Ironclad", "map")] == pytest.approx(left / 2 / 1000)
    assert weights[group("Ironclad", "shop")] == pytest.approx(left / 2 / 500)
    assert sum(weights[key] * count for key, count in counts.items()) == pytest.approx(1.0)
    # Groups that can carry their share at the cap keep the plain rule.
    even_groups = {group("Ironclad", "map"): 300, group("Silent", "map"): 100, group("Defect", "shop"): 50}
    assert awr.buffer_weights(even_groups) == group_weights(even_groups)
    # The share a capped group gives up may bring the next smallest over the cap in turn.
    counts = {group("Ironclad", "map"): 1000, group("Ironclad", "event"): 40, group("Silent", "bundle_select"): 2}
    limit = awr.WEIGHT_CAP / 1042
    assert group_weights(counts)[group("Ironclad", "event")] < limit
    weights = awr.buffer_weights(counts)
    assert weights[group("Ironclad", "event")] == weights[group("Silent", "bundle_select")] == pytest.approx(limit)
    assert weights[group("Ironclad", "map")] == pytest.approx((1 - 42 * limit) / 1000) and weights[group("Ironclad", "map")] < limit
    assert sum(weights[key] * count for key, count in counts.items()) == pytest.approx(1.0)
    assert awr.buffer_weights(counts, cap=1.0) == {key: pytest.approx(1 / 1042) for key in counts}
    with pytest.raises(ValueError):
        awr.buffer_weights(counts, cap=0.5)


def test_a_samples_weight_is_its_share_weight_times_its_normalized_advantage_factor():
    # Two groups of the buffer with different share weights; advantages by (run, macro).
    items = [item("a", 1, 0.02), item("a", 3, 0.02), item("b", 1, 0.05), item("b", 5, 0.05)]
    table = {("a", 1): 0.4, ("a", 3): -0.2, ("b", 1): 0.0, ("b", 5): 0.4}
    logical = [(items[:3], "first part"), (items[3:], "second part")]
    weighted, done = awr.reweigh(logical, table, awr.Temperature(fixed=0.5), scale=0.25)
    assert [prepared for _, prepared in weighted] == ["first part", "second part"]
    after = [entry.weight for micro, _ in weighted for entry in micro]
    base = [entry.weight for entry in items]
    # The batch's total weight is the share weights' total, on the trainer's scale.
    assert sum(after) == pytest.approx(0.25 * sum(base))
    # Each weight is share × exp(A/β) × the batch's normalizer: equal advantages keep the
    # ratio of the shares, equal shares differ by exp(ΔA/β).
    assert after[0] / after[3] == pytest.approx(0.02 / 0.05)
    assert after[0] / after[1] == pytest.approx(math.exp(0.6 / 0.5))
    assert after[3] / after[2] == pytest.approx(math.exp(0.4 / 0.5))
    assert (done["beta"], done["samples"]) == (0.5, 4)
    assert done["effective"] == pytest.approx(awr.effective_size(list(table.values()), 0.5)) and done["share"] == done["effective"] / 4
    # The samples themselves are not changed, and nothing else of them is.
    assert [entry.weight for entry in items] == base
    assert [replace(entry, weight=0) for micro, _ in weighted for entry in micro] == [replace(entry, weight=0) for entry in items]
    # Without weighting every sample keeps its share weight: behavior cloning of the policy's own decisions.
    plain, done = awr.reweigh(logical, table, awr.Temperature.parse("inf"))
    assert [entry.weight for micro, _ in plain for entry in micro] == pytest.approx(base)
    assert done["effective"] == 4 and math.isinf(done["beta"])
    # Tuned: the batch's own β.
    tuned, done = awr.reweigh(logical, table, awr.Temperature(share=0.9))
    assert done["beta"] == pytest.approx(awr.tune(list(table.values()), share=0.9)) and done["share"] >= 0.9
    assert awr.factors([0.0, 0.0, 0.0], [1.0, 2.0, 3.0], 0.1) == pytest.approx([1.0, 1.0, 1.0])
    with pytest.raises(ProtocolError, match="no advantage"):
        awr.reweigh([([item("c", 0, 0.1)], None)], table, awr.Temperature())


class Stream:
    """Half-batches of a source, counting how often it was begun and closed."""

    def __init__(self, batches):
        self.batches, self.begun, self.closed = batches, 0, 0

    def __call__(self):
        self.begun += 1
        try:
            yield from self.batches
        finally:
            self.closed += 1


def halves(source, count):
    return [[([item(f"{source}{number}", 0, 1.0, source)], f"{source}{number}")] for number in range(count)]


def test_every_batch_is_a_weighted_sampled_half_followed_by_a_supervised_half():
    def weigh(batch):
        return [([replace(entry, weight=2.0) for entry in micro], prepared) for micro, prepared in batch], dict(
            beta=0.5, share=0.5, samples=1, effective=0.5)

    # One pass over the sampled stream; the shorter supervised stream is begun anew.
    sampled, supervised = Stream(halves("rollout", 3)), Stream(halves("independent", 2))
    mixed = awr.Mixed(sampled, supervised, weigh)
    batches = list(mixed)
    assert [[prepared for _, prepared in batch] for batch in batches] == [
        ["rollout0", "independent0"], ["rollout1", "independent1"], ["rollout2", "independent0"]]
    assert [[entry.weight for micro, _ in batch for entry in micro] for batch in batches] == [[2.0, 1.0]] * 3
    assert (sampled.begun, sampled.closed, supervised.begun, supervised.closed) == (1, 1, 2, 2)
    assert mixed.summary() == dict(batches=3, samples=3, beta=0.5, beta_min=0.5, beta_max=0.5, effective_share=0.5,
                                   effective_share_min=0.5)
    # A fixed number of updates begins either stream anew, and stops both when it is reached.
    sampled, supervised = Stream(halves("rollout", 2)), Stream(halves("independent", 3))
    mixed = awr.Mixed(sampled, supervised, weigh, updates=5)
    assert [batch[0][1] for batch in mixed] == ["rollout0", "rollout1", "rollout0", "rollout1", "rollout0"]
    assert (sampled.begun, sampled.closed, supervised.begun, supervised.closed) == (3, 3, 2, 2)
    # A trainer that leaves early closes the pass, and with it both streams.
    sampled, supervised = Stream(halves("rollout", 4)), Stream(halves("independent", 4))
    mixed = awr.Mixed(sampled, supervised, weigh)
    assert next(iter(mixed))[0][1] == "rollout0" and mixed.last["beta"] == 0.5
    mixed.close()
    assert (sampled.closed, supervised.closed) == (1, 1)
    with pytest.raises(ProtocolError, match="no batch"):
        list(awr.Mixed(Stream(halves("rollout", 1)), Stream([]), weigh))
    with pytest.raises(ValueError):
        awr.Mixed(sampled, supervised, weigh, updates=0)
    assert awr.supervised_loss(dict(rollout=dict(loss=9.0, weight=1.0), independent=dict(loss=1.0, weight=1.0),
                                    student_battle=dict(loss=4.0, weight=2.0))) == pytest.approx(3.0)
    assert awr.supervised_loss(dict(rollout=dict(loss=9.0, weight=1.0))) is None


def buffer(tmp_path, policy, rounds=2):
    directories = [tmp_path / "buffer" / f"round-{number}" for number in range(rounds)]
    for number, directory in enumerate(directories):
        journeys.sampled_round(policy, directory, number, runs=3)
    advantage.write_returns(directories)
    return directories


def test_an_update_is_half_weighted_decisions_of_the_buffer_and_half_supervised_samples(tmp_path):
    policy = journeys.learner()
    rounds = buffer(tmp_path, policy)
    demonstrations = journeys.demonstrations(tmp_path / "demonstrations")
    ours, _ = split_held_out(RunShards(rounds))
    theirs, _ = split_held_out(RunShards(demonstrations))
    assert awr.pass_updates(rounds, policy.config) == math.ceil(ours.macros / 4)
    told = []
    before = policy.policy_version
    metrics = awr.update(policy, rounds, [demonstrations], on_update=told.append)
    done = metrics["awr"]
    # One pass over the buffer's decisions, four of them with four supervised samples in every update.
    assert metrics["updates"] == done["batches"] == len(told) == math.ceil(ours.macros / 4) and done["samples"] == ours.macros
    assert policy.policy_version == before + 1 and metrics["loss_scale"] == theirs.macros / 8
    assert all(update["samples"] <= 8 and update["sampled_samples"] <= 4 for update in told)
    assert told[0]["sources"]["rollout"]["samples"] == 4 and told[0]["sources"]["demonstration"]["samples"] == 4
    # The sampled half carries the buffer's whole weight over a pass, on the supervised
    # data's scale, whatever β does to single decisions: half of every update's loss.
    assert metrics["sources"]["rollout"]["weight"] == pytest.approx(ours.macros / theirs.macros)
    # In units of the loss both halves weigh half a batch for every four samples they held.
    assert metrics["sources"]["rollout"]["weight"] * metrics["loss_scale"] == pytest.approx(ours.macros / 8)
    supervised = metrics["sources"]["demonstration"]
    assert supervised["weight"] * metrics["loss_scale"] == pytest.approx(supervised["samples"] / 8)
    assert all(awr.BETA_MIN <= update["beta"] <= awr.BETA_MAX and 0 < update["effective_share"] <= 1 for update in told)
    assert all(update["effective_share"] >= 0.3 - 1e-9 or update["beta"] == awr.BETA_MAX for update in told)
    assert done["awr_loss"] == pytest.approx(metrics["sources"]["rollout"]["loss"])
    assert done["supervised_loss"] == pytest.approx(metrics["sources"]["demonstration"]["loss"])
    assert (done["sampled"]["macros"], done["supervised"]["macros"]) == (ours.macros, theirs.macros)
    assert done["temperature"] == dict(rule="tuned", share=0.3, low=0.1, high=100.0)
    # No group of this buffer is small enough to be held to the cap on a decision's share weight.
    assert (done["weight_cap"], done["capped_groups"]) == (awr.WEIGHT_CAP, [])
    assert told[0]["awr_loss"] is not None and told[0]["supervised_loss"] is not None and told[0]["beta"] is not None
    json.dumps(told, allow_nan=False)

    # A fixed number of updates, without weighting: the weights of behavior cloning.
    plain = awr.update(policy, rounds, [demonstrations], temperature=awr.Temperature.parse("inf"), updates=3,
                       on_update=told.append)
    assert plain["awr"]["batches"] == 3 and plain["awr"]["beta"] is None and plain["awr"]["effective_share"] == 1.0
    assert told[-1]["beta"] is None and told[-1]["effective_share"] == 1.0
    assert plain["awr"]["temperature"] == dict(rule="none")
    json.dumps(told, allow_nan=False)
    with pytest.raises(FileNotFoundError, match="no advantages"):
        (rounds[0] / FILE).unlink()
        awr.update(policy, rounds, [demonstrations])
    with pytest.raises(ValueError, match="even"):
        awr.update(replace_config(policy, logical_batch_size=7), rounds, [demonstrations])


def replace_config(policy, **settings):
    policy.config = replace(policy.config, **settings)
    return policy
