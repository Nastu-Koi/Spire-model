"""Where a process's time goes: seconds and counts by what it was doing, summed over
its threads. A block is timed as it runs (`timed`, `Laps`), at the cost of two clock
reads; the process that reports it reads the totals twice and takes the difference
(`totals`, `since`), so that nothing has to be reset under a thread that is adding."""

import threading
import time

# Every thread's own totals: a thread adds only to its own, so nothing is lost between
# threads, and a reading sums them all.
_THREADS = []
_LOCAL = threading.local()


def add(name, seconds):
    """Count `seconds` under `name`, once."""
    own = getattr(_LOCAL, "totals", None)
    if own is None:
        own = _LOCAL.totals = {}
        _THREADS.append(own)
    total = own.get(name)
    own[name] = (seconds, 1) if total is None else (total[0] + seconds, total[1] + 1)


class timed:
    """Count the time of a `with` block under `name`."""

    __slots__ = ("name", "began")

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.began = time.perf_counter()

    def __exit__(self, *_):
        add(self.name, time.perf_counter() - self.began)


class Laps:
    """The time from one point of a block to the next, each counted under the name
    given at the later point."""

    __slots__ = ("last",)

    def __init__(self):
        self.last = time.perf_counter()

    def __call__(self, name):
        now = time.perf_counter()
        add(name, now - self.last)
        self.last = now


def totals():
    """Seconds and counts by name so far, over every thread of this process (and of
    the process it was forked from, before the fork)."""
    result = {}
    for own in list(_THREADS):
        for name, (seconds, count) in list(own.items()):
            before = result.get(name, (0.0, 0))
            result[name] = (before[0] + seconds, before[1] + count)
    return result


def since(before):
    """What was counted since the reading `before` of `totals`, by name."""
    result = {}
    for name, (seconds, count) in totals().items():
        earlier = before.get(name, (0.0, 0))
        if count != earlier[1]:
            result[name] = (seconds - earlier[0], count - earlier[1])
    return result
