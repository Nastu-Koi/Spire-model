"""The advantage file of a wave, and the batches of an update of act groups: the
waves' share weights, their shares of a batch, their streams."""

import gzip
import json

import pytest

from main.learn import advantage
from main.learn import batches as awr
from main.learn.advantage import FILE
from main.data import shards
from main.data.shards import RunShards, group_weights, samples
from main.tests import journeys


def test_a_waves_advantages_are_kept_beside_it_with_what_made_them(tmp_path):
    policy = journeys.learner()
    directory = tmp_path / "round-0"
    summary = journeys.sampled_round(policy, directory)
    shards = sorted(path.read_bytes() for path in (directory / "accepted").iterdir())
    # The samples of a round are its decisions outside fights: every other macro-action.
    items = samples(next(RunShards(directory).windows(8)))
    places = {}
    for entry in items:
        places.setdefault(entry.run_id, []).append(entry.index)
    assert len(places) == summary["accepted_runs"] == 10 and all(sorted(p) == [1, 3, 5] for p in places.values())
    runs = {run: [(index, 0.1 * index, 0.5 - 0.1 * index) for index in sorted(place)] for run, place in places.items()}
    identity = dict(kind="act-groups", ridge=1.0)
    header = advantage.write(directory, identity, runs)
    assert (header["format"], header["kind"], header["ridge"]) == ("rollout-advantages-v1", "act-groups", 1.0)
    assert (header["runs"], header["decisions"], header["advantage"]["n"]) == (10, 30, 30)
    assert advantage.header(directory) == header and advantage.current(directory, identity)
    assert not advantage.current(directory, dict(identity, ridge=2.0))
    table = advantage.read([directory])
    run = next(iter(places))
    assert set(table) == {(entry.run_id, entry.index) for entry in items}
    assert [table[run, index] for index in (1, 3, 5)] == pytest.approx([0.4, 0.2, 0.0], abs=1e-6)
    # The shards are as they were.
    assert sorted(path.read_bytes() for path in (directory / "accepted").iterdir()) == shards
    with gzip.open(directory / FILE, "rt") as stream:
        assert json.load(stream)["format"] == advantage.FORMAT
    with pytest.raises(FileNotFoundError, match="no advantages"):
        advantage.read([tmp_path / "round-1"])


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
    # Even phase weights: every decision alike, the cap beside the point; an empty group has no weight.
    even = awr.buffer_weights({**counts, group("Defect", "shop"): 0}, phase_weights="even")
    assert even == {key: pytest.approx(1 / 1042) for key in counts}
    assert awr.buffer_weights(counts, cap=0, phase_weights="even") == even
    with pytest.raises(ValueError):
        awr.buffer_weights(counts, phase_weights="uniform")


def test_a_batch_is_shared_out_in_proportion_and_adds_up():
    assert awr.sizes(512, {"anchor": 0.2, "act1": 0.4, "act2": 0.4}) == {"anchor": 102, "act1": 205, "act2": 205}
    assert sum(awr.sizes(7, {"a": 1, "b": 1, "c": 1}).values()) == 7
    with pytest.raises(ValueError):
        awr.sizes(2, {"a": 1, "b": 1, "c": 1})


def test_every_stream_of_an_update_reads_with_at_least_two_feeders():
    from main.network.config import TrainConfig

    # The profile's four feeders among three waves would leave one each: a feeder
    # decoding its next shards stalls its stream, and the joint batch with it.
    assert awr.stream_feeders(TrainConfig(loader_feeders=4), 3) == 2
    assert awr.stream_feeders(TrainConfig(loader_feeders=4), 4) == 2
    assert awr.stream_feeders(TrainConfig(loader_feeders=9), 3) == 3
    assert awr.stream_feeders(TrainConfig(loader_feeders=1), 1) == awr.STREAM_FEEDERS == 2


def test_the_waves_share_a_batch_by_their_decisions_unless_told_otherwise():
    macros = {"act1": 1000, "act2": 2000, "act3": 7000}
    # By default a wave's share is its share of the round's decisions: over a pass of
    # ΣN / B batches every wave is seen once, and the shares are what the waves weigh.
    assert awr.wave_shares(macros) == {"act1": pytest.approx(0.1), "act2": pytest.approx(0.2), "act3": pytest.approx(0.7)}
    # Relative shares of the waves, in their order: `1 1 1` is the waves alike.
    assert awr.wave_shares(macros, [1, 1, 1]) == {wave: pytest.approx(1 / 3) for wave in macros}
    assert awr.wave_shares(macros, [1, 3, 4]) == {
        "act1": pytest.approx(0.125), "act2": pytest.approx(0.375), "act3": pytest.approx(0.5)}
    for wrong in ([1, 1], [1, 0, 1], [1, -1, 1]):
        with pytest.raises(ValueError):
            awr.wave_shares(macros, wrong)
    with pytest.raises(ValueError):
        awr.wave_shares({"act1": 0, "act2": 0})
    # A pass of ΣN / B batches at these shares sees each wave's decisions once, up to
    # the rounding of each batch's sizes and of the pass to whole batches.
    sizes = awr.sizes(512, awr.wave_shares(macros))
    updates = awr.pass_updates(macros, 512)
    assert updates == -(-10000 // 512)
    for wave, count in macros.items():
        assert abs(sizes[wave] * updates - count) <= updates + 512


