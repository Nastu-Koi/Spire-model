"""Where a process's time goes (`main.env.clock`)."""

import threading
import time

from main.env import clock


def test_blocks_are_counted_by_name_over_every_thread_since_a_reading():
    before = clock.totals()

    def work():
        with clock.timed("test_block"):
            time.sleep(0.01)

    threads = [threading.Thread(target=work) for _ in range(3)]
    for thread in threads:
        thread.start()
    work()
    for thread in threads:
        thread.join()
    laps = clock.Laps()
    time.sleep(0.01)
    laps("test_lap")
    laps("test_lap")
    taken = clock.since(before)
    seconds, count = taken["test_block"]
    assert count == 4 and seconds >= 0.04
    seconds, count = taken["test_lap"]
    assert count == 2 and seconds >= 0.01
    # A reading after it sees nothing new; one before it sees all of it again.
    assert clock.since(clock.totals()) == {}
    assert clock.since(before)["test_block"] == taken["test_block"]
