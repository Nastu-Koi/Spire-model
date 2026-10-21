"""The update's loader processes: feeders forked from the feeder server, not from the
trainer, so that they copy nothing of the trainer's heap; packers forked from a feeder
with what it holds frozen from its collector."""

import gc
import os
from dataclasses import dataclass
from types import SimpleNamespace

from main.learn import loader


@dataclass(frozen=True)
class Config:
    loader_workers: int = 1
    loader_feeders: int = 2
    logical_batch_size: int = 4


class Shards:
    weights = None

    def window_names(self, shards, shuffle):
        return [["a"], ["b"]]


class Vocabulary:
    def report(self, unregistered):
        pass


# What the trainer holds and a process forked from it would see.
HELD = []


def seen_in_feeder(connection, shards, windows, weights, vocabulary, config, shuffle, seed):
    """Stands in for `_feed`: reports what it sees of the trainer's heap and its parent."""
    feeder, _ = windows
    os.setsid()
    prepared = SimpleNamespace(batch=SimpleNamespace(unregistered=()))
    connection.send(("batch", [((list(HELD), os.getppid()), prepared)]))
    connection.send(("done", feeder))
    connection.close()


def test_feeders_are_forked_from_the_server_and_hold_nothing_of_the_trainer(monkeypatch):
    monkeypatch.setattr(loader, "_feed", seen_in_feeder)
    HELD.append("the trainer's")
    try:
        seen = [part[0] for logical in loader.stream(Shards(), Vocabulary(), Config(), window_shards=2, shuffle=False)
                for part in logical]
    finally:
        HELD.clear()
    assert len(seen) == 2
    # Not the trainer's children, and none of its objects: the server's module state.
    assert all(held == [] and parent != os.getpid() for held, parent in seen)
    assert len({parent for _, parent in seen}) == 1


def test_packers_in_a_feeder_leave_what_it_inherited_frozen():
    # A feeder runs with the trainer's heap frozen; its packers' stream must not thaw it.
    gc.unfreeze()
    gc.freeze()
    try:
        assert list(loader._stream([1, 2], 1)) == [1, 2]
        assert gc.get_freeze_count() > 0  # frozen objects freed meanwhile leave the count
    finally:
        gc.unfreeze()
