"""A process's share of the GPU."""

import pytest

from main.env.runtime import gpu_budget

GiB = 1024**3


def test_a_process_takes_the_cap_when_the_gpu_has_room_for_it_and_the_reserve():
    # The training machine: 95.6 GiB, about 2.7 of them the desktop's.
    assert gpu_budget(92.9 * GiB) == 84 * GiB


def test_the_reserve_stays_free_when_less_than_the_cap_is_left():
    assert gpu_budget(90 * GiB) == 82 * GiB


def test_a_gpu_without_room_beyond_the_reserve_is_refused():
    with pytest.raises(RuntimeError):
        gpu_budget(8 * GiB)
