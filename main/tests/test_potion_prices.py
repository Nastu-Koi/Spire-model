"""Potion prices fitted to what keeping a potion saves (`main.learn.potion_prices`):
the features of a state, the linear prices and their smoothing, and the fit on pairs of
continuations, the acts fitted backward."""

import json
import random

import numpy as np
import pytest

from main.env.protocol import CHARACTERS
from main.learn import costs
from main.learn.fight_groups import describe
from main.learn.potion_prices import (
    SLOPES, Design, LinearPrices, Smoothed, features, fight_features, fit, from_json, held_value, load, to_json)

DECK = dict(cards=20, attacks=8, skills=9, powers=2, curses=1, statuses=0, upgraded=4)


def state(act=1, hp=60, max_hp=80, potions=("POTION.FIRE_POTION",), capacity=3, floor=5, deck=DECK, relics=3):
    return dict(act=act, floor=floor, act_floors=17, hp=hp, max_hp=max_hp, potions=list(potions), capacity=capacity,
                deck=dict(deck), relics=relics)


def fight(potions=("POTION.FIRE_POTION",), *, act=1, kind="normal", hp=60, entry=None, **changes):
    return dict(dict(kind=kind, act=act, floor=5, second_boss=False, ascension=0, character="Silent",
                     entry_hp=hp, entry_max_hp=80, exit_hp=hp - 10, exit_max_hp=80, last_hp=hp - 10, last_max_hp=80,
                     potions_used=list(potions), died=False,
                     entry_state=state(act, hp) if entry is None else entry), **changes)


def test_the_features_of_a_state():
    found = features(state(act=2, hp=60, potions=("POTION.A", "POTION.B", "POTION.C")), "Regent", 4)
    assert (found["act"], found["full"], found["held"], found["floors_left"], found["ascension"]) == (2, True, 3, 12, 4)
    assert (found["hp_missing_2"], found["hp_missing_1"], found["hp_missing_3"]) == (0.25, 0.0, 0.0)
    assert (found["cards"], found["attacks"], found["curses"], found["upgraded"], found["relics"]) == (20, 8, 1, 4, 3)
    assert features(state(capacity=None), "Regent", 0)["full"] is False
    with pytest.raises(ValueError, match="entry state"):
        fight_features(fight(entry={}))


def priced_by(coefficients, rows=None):
    rows = rows or [dict(features(state(hp=hp), "Silent", 0), potion="POTION.FIRE_POTION") for hp in (40, 60, 80)]
    design = Design.of(rows)
    return LinearPrices(design, tuple(coefficients(design)))


def test_linear_prices_price_a_fight_by_its_entry_state():
    def coefficients(design):
        beta = np.zeros(len(design.columns))
        beta[design.columns.index("base_1_room")] = 5.0
        beta[design.columns.index("hp_missing_1")] = 2.0
        beta[design.columns.index("tier_high")] = 3.0
        return beta

    prices = priced_by(coefficients)
    missing = Design.of([dict(features(state(hp=hp), "Silent", 0), potion="P") for hp in (40, 60, 80)])
    standardized = (0.25 - missing.means["hp_missing_1"]) / missing.scales["hp_missing_1"]
    assert prices.price("POTION.FIRE_POTION", fight(hp=60)) == pytest.approx(5.0 + 2.0 * standardized)
    # A high-tier potion: its tier's offset.
    assert prices.price("POTION.SWIFT_POTION", fight(hp=60)) == pytest.approx(8.0 + 2.0 * standardized)
    # The run's last fight and the free potions keep the fixed prices.
    last = fight(act=3, kind="boss")
    assert prices.price("POTION.FIRE_POTION", last) == 1.0
    assert prices.price("POTION.POTION_SHAPED_ROCK", fight()) == 0.0
    # The same prices back from their file.
    again = from_json(json.loads(json.dumps(to_json(prices))))
    assert again.price("POTION.SWIFT_POTION", fight(hp=60)) == pytest.approx(prices.price("POTION.SWIFT_POTION",
                                                                                         fight(hp=60)))
    assert describe(prices)["kind"] == "linear" and describe(prices)["digest"] == describe(again)["digest"]


class Flat:
    """Prices of one value everywhere."""

    def __init__(self, value):
        self.value = value

    def price(self, potion, fight):
        return self.value


def test_a_fit_moves_a_price_halfway_and_by_at_most_thirty_percent():
    assert Smoothed(Flat(7.0), Flat(1.0)).price("POTION.FIRE_POTION", fight()) == pytest.approx(4.9)
    assert Smoothed(Flat(7.0), Flat(8.0)).price("POTION.FIRE_POTION", fight()) == pytest.approx(7.5)
    assert Smoothed(Flat(7.0), Flat(20.0)).price("POTION.FIRE_POTION", fight()) == pytest.approx(9.1)
    # Fit on fit: each layer from the one before.
    twice = Smoothed(Smoothed(Flat(7.0), Flat(1.0)), Flat(1.0))
    assert twice.price("POTION.FIRE_POTION", fight()) == pytest.approx(4.9 * 0.7)
    # The run's last fight and the free potions keep the previous prices.
    fixed = Smoothed(costs.FIXED, Flat(100.0))
    assert fixed.price("POTION.FIRE_POTION", fight(act=3, kind="boss")) == 1.0
    assert fixed.price("POTION.POTION_SHAPED_ROCK", fight()) == 0.0
    assert fixed.price("POTION.FIRE_POTION", fight()) == pytest.approx(9.1)
    # Read back from a file, the fixed prices under it too.
    assert load("fixed") is costs.FIXED
    assert costs.fight_return(fight(), fixed) != costs.fight_return(fight(), costs.FIXED)
    assert costs.fight_return(fight(()), fixed) == costs.fight_return(fight(()), costs.FIXED)


def record(cost, act, *, kind="normal"):
    """A fight that cost `cost` HP (no potion used)."""
    return dict(kind=kind, act=act, floor=3, second_boss=False, ascension=0, character="Silent", entry_hp=70,
                entry_max_hp=80, exit_hp=70 - cost, exit_max_hp=80, last_hp=70 - cost, last_max_hp=80,
                potions_used=[], died=False)


POTIONS = ("POTION.FIRE_POTION", "POTION.BLOCK_POTION", "POTION.SWIFT_POTION", "POTION.CLARITY")


def synthetic(truth, *, count=900, seed=7, carried=None):
    """Points and continuations whose pairs save `truth(features, potion)` HP plus
    noise, with what is held at the next act's start (`carried`: potions kept, potions
    the potion taken out also loses) for the first two acts."""
    draw = random.Random(seed)
    points, branches = [], []
    for number in range(count):
        act = draw.choice((1, 2, 3))
        held = [draw.choice(POTIONS) for _ in range(draw.randint(1, 3))]
        at = state(act=act, hp=draw.randint(10, 80), potions=held, floor=draw.randint(1, 15))
        character = draw.choice(CHARACTERS)
        point = dict(id=f"p{number}", character=character, seed=f"s{number % 150}", ascension=draw.randint(0, 5),
                     state=at)
        points.append(point)
        cut = None if act == 3 else state(act=act + 1, hp=72, potions=("POTION.FIRE_POTION",), floor=1)
        branches.append(dict(point=point["id"], branch="keep", dropped=None, status="complete", outcome="reached",
                             records=[record(10, act)], cut_state=cut))
        for index, potion in enumerate(held):
            saved = truth(features(at, character, point["ascension"]), potion) + draw.gauss(0, 3)
            branches.append(dict(point=point["id"], branch=f"drop{index}", dropped=potion, status="complete",
                                 outcome="reached", records=[record(10 + saved, act)],
                                 cut_state=None if cut is None else dict(cut, potions=[])))
    return points, branches


def test_the_fit_finds_the_prices_the_pairs_were_made_with():
    def truth(found, potion):
        base = {1: 6.0, 2: 9.0, 3: 4.0}[found["act"]] + (2.0 if found["full"] else 0.0)
        return base + 12.0 * found[f"hp_missing_{found['act']}"] + (5.0 if potion == "POTION.SWIFT_POTION" else 0.0)

    points, branches = synthetic(truth)
    # The pairs of the first two acts also lose the potion held at the next act's start
    # in the branch it was taken out of: its price there is part of the label.
    prices, report = fit(points, branches, costs.FIXED, grid=(1e-4, 1e-2, 1.0))
    assert report["kept"] == "model" and report["held_out_error"] < report["base_held_out_error"]
    assert report["rounds"] >= 2 and report["moved"] < 0.05
    low = fight(hp=20, entry=state(act=3, hp=20))
    high = fight(hp=80, entry=state(act=3, hp=80))
    # The last act's labels are its own: the slope in missing HP, the potion's offset.
    assert prices.price("POTION.FIRE_POTION", low) - prices.price("POTION.FIRE_POTION", high) == pytest.approx(
        12.0 * 0.75, abs=1.5)
    assert prices.price("POTION.SWIFT_POTION", low) - prices.price("POTION.FIRE_POTION", low) == pytest.approx(
        5.0, abs=1.5)
    # The second act's pairs hold a fire potion at the third act's start that the
    # taken-out branch does not: their labels are the truth plus its price there.
    at_third = prices.price("POTION.FIRE_POTION", fight(entry=state(act=3, hp=72, floor=1)))
    second = fight(act=2, hp=60, entry=state(act=2, hp=60))
    expected = truth(features(second["entry_state"], "Silent", 0), "POTION.FIRE_POTION") + at_third
    assert prices.price("POTION.FIRE_POTION", second) == pytest.approx(expected, abs=2.0)
    assert set(report["coefficients"]) == set(prices.design.columns) and len(SLOPES) == 13


def test_where_nothing_but_the_act_and_belt_matters_the_base_prices_are_kept():
    points, branches = synthetic(lambda found, potion: 7.0, count=600, seed=3)
    prices, report = fit(points, branches, costs.FIXED, grid=(1e-4, 1.0))
    assert report["kept"] == "base"
    assert all(value["value"] == 0.0 for name, value in report["coefficients"].items() if not name.startswith("base"))
    assert prices.price("POTION.FIRE_POTION", fight(act=3, entry=state(act=3))) == pytest.approx(7.0, abs=1.0)
