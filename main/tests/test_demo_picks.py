"""The fights a round hands to the solver for combat demonstrations (`main.play.demo_picks`):
3:4:4 by number of fights, each kind at its natural frequency, the fights set up by a
selection first up to their share, and the selection the solver reads."""

import json
from collections import Counter

import pytest

from main.play.demo_picks import natural_weights, pick, write_selection


def row(run, fight, kind="normal", *, wave="act1", act=1, character="Regent", seed="S", path=True, opening=False):
    return dict(run_id=run, fight=fight, kind=kind, wave=wave, act=act, floor=fight, encounter="ENCOUNTER.X",
                second_boss=False, character=character, seed=seed, ascension=0, died=False,
                opening_selection=opening, path=f"/saves/{run}-{fight}.json.gz" if path else None)


def test_a_fight_weighs_one_over_the_segments_of_its_run_in_its_wave():
    rows = [row("a-0", 1), row("a-0", 2), row("b-0", 1, seed="T")]
    rows += [row(f"a-{n}", 3, wave="act2", act=2) for n in range(4)]
    assert natural_weights(rows) == [1.0, 1.0, 1.0] + [0.25] * 4


def test_the_kinds_share_the_count_three_four_four_and_events_are_left_out():
    rows = [row(f"r{n}", 1, kind) for kind in ("normal", "elite", "boss") for n in range(50)]
    rows += [row(f"e{n}", 1, "elite", path=False) for n in range(50)]
    picked, account = pick(rows, 22, "k")
    assert Counter(fight["kind"] for fight in picked) == dict(normal=6, elite=8, boss=8)
    assert all(fight["path"] is not None for fight in picked)
    assert account["events"] == 50 and account["short"] == []
    # A kind short of its quota gives what it has; no other kind makes up for it.
    picked, account = pick(rows[:150 - 47], 22, "k")
    assert Counter(fight["kind"] for fight in picked)["boss"] == 3 and account["short"] == ["boss"]


def test_each_run_counts_once_in_each_act():
    """Twenty runs, each with one segment at the first act and eight at the third: drawn
    at their natural frequency the two acts come about even, not one to eight."""
    rows = [row(f"r{n}-a", 1, character="Silent", seed=f"S{n}") for n in range(400)]
    rows += [row(f"r{n}-c{m}", 9, wave="act3", act=3, character="Silent", seed=f"S{n}")
             for n in range(400) for m in range(8)]
    picked, _ = pick(rows, 330, "k", opening=0.0)  # 90 normal fights
    acts = Counter(fight["act"] for fight in picked if fight["kind"] == "normal")
    assert sum(acts.values()) == 90 and 30 <= acts[1] <= 60


def test_fights_set_up_by_a_selection_come_first_up_to_their_share():
    rows = [row(f"r{n}", 1, kind) for kind in ("normal", "elite", "boss") for n in range(40)]
    rows += [row(f"o{n}", 1, "normal", opening=True) for n in range(30)]
    rows += [row(f"p{n}", 1, "boss", opening=True) for n in range(2)]
    picked, account = pick(rows, 110, "k", opening=0.2)  # quotas 30 / 40 / 40, 22 first
    first = [fight for fight in picked if fight["reason"] == "opening"]
    assert len(first) == 22 and all(fight["opening_selection"] for fight in first)
    assert account["kinds"]["normal"]["opening"] + account["kinds"]["boss"]["opening"] == 22
    assert account["kinds"]["boss"]["opening"] == 2
    # Within its kind's quota: the normal fights' 30 hold at most 30 of them.
    picked, account = pick(rows, 11, "k", opening=1.0)  # quotas 3 / 4 / 4
    assert account["kinds"]["normal"]["opening"] == 3 and account["kinds"]["normal"]["picked"] == 3


def test_a_pick_is_the_same_by_its_key():
    rows = [row(f"r{n}", 1, kind) for kind in ("normal", "elite", "boss") for n in range(30)]
    ids = lambda picked: [(fight["run_id"], fight["fight"]) for fight in picked]
    assert ids(pick(rows, 22, "a")[0]) == ids(pick(rows, 22, "a")[0])
    assert ids(pick(rows, 22, "a")[0]) != ids(pick(rows, 22, "b")[0])


def test_the_selection_the_solver_reads(tmp_path):
    picked, account = pick([row("r0", 1), row("r0", 2, "boss"), row("r1", 1, "elite", seed="T")], 11, "k")
    path = write_selection(picked, account, tmp_path / "demos" / "selection.json", key="k", checkpoint="c")
    written = json.loads(path.read_text())
    assert written["extra"] == [] and written["key"] == "k" and written["checkpoint"] == "c"
    assert sorted(fight["id"] for fight in written["fights"]) == ["r0-001", "r0-002", "r1-001"]
    assert all(fight["path"].startswith("/") and {"character", "seed", "kind"} <= set(fight)
               for fight in written["fights"])
    with pytest.raises(ValueError, match="same id"):
        write_selection(picked + picked[:1], account, tmp_path / "again.json")


@pytest.mark.parametrize("count, opening", [(-1, 0.2), (1.5, 0.2), (10, 1.5), (10, True)])
def test_what_a_pick_takes(count, opening):
    with pytest.raises(ValueError):
        pick([], count, "k", opening=opening)
