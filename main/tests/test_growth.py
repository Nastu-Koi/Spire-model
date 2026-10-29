"""The growth reward (ticket 15 of .scratch/training-plan): what a fight leaves the run to
grow on, recorded by `main.play.fights.FightLog` (`growth`, `relic_counters`) and priced
by `main.learn.costs` (`growth_events`, `Growing`)."""

import json
from pathlib import Path

import pytest

from main.cli import parser
from main.env.saves import FightTracker
from main.learn import costs, potion_prices
from main.learn.costs import FIXED, Growing, cost, growth_events, with_growth
from main.learn.fight_report import tally
from main.play.fights import FightLog

GA, SCYTHE, FEED, GREED, HUNT = (f"CARD.{name}" for name in ("GENETIC_ALGORITHM", "THE_SCYTHE", "FEED", "HAND_OF_GREED",
                                                             "THE_HUNT"))
DECK = [GA, GA, SCYTHE, FEED, GREED, HUNT, "CARD.STRIKE"]
END = dict(verb="END_TURN", candidate_ref="end", source_refs=[])


class Saving:
    def __init__(self):
        self.saves = 0

    def send(self, command):
        self.saves += 1
        Path(command["path"]).write_text(json.dumps({"number": self.saves}))
        return {"type": "save_result", "success": True}


def frame(hand=(), *, max_hp=80, gold=99, hunt=0, relics=(), phase="combat"):
    """A decision with `hand` to play (each card a PLAY_CARD candidate, `p<i>`), the deck
    `DECK`, the player's max HP, gold and stacks of The Hunt's power, and relics showing counts."""
    entities = [dict(ref="player", entity_type="player", hp=50, max_hp=max_hp, gold=gold, act=1, floor=3, capacity=3)]
    if hunt:
        entities.append(dict(entity_type="power", content_id="POWER.THE_HUNT_POWER", owner_ref="player", stacks=hunt))
    # Refs are the frame's own: the same names for other cards in another frame.
    entities += [dict(ref=f"card:{i}", entity_type="card", zone="hand", content_id=card) for i, card in enumerate(hand)]
    entities += [dict(ref=f"card:{len(hand) + i}", entity_type="card", zone="deck", content_id=card)
                 for i, card in enumerate(DECK)]
    entities += [dict(ref=f"relic:{i}", entity_type="relic", content_id=relic, counter=count)
                 for i, (relic, count) in enumerate(relics)]
    candidates = [dict(verb="PLAY_CARD", candidate_ref=f"p{i}", source_refs=[f"card:{i}"]) for i in range(len(hand))]
    return dict(public=dict(phase=phase, entities=entities), legal=dict(candidates=candidates + [END]))


class Fight:
    """A FightLog told of decisions as `RolloutRunner.run` tells it."""

    def __init__(self):
        self.engine, self.tracker, self.log = Saving(), FightTracker(), FightLog()

    def decide(self, seen, events=(), choice="end"):
        self.tracker.events(list(events))
        fight, entry = self.tracker.decision()
        self.log.passed(seen, fight)
        self.log.entered(self.engine, seen, fight, entry, 3)
        self.log.chose(seen, fight, choice)

    def record(self):
        return self.log.records(dict(run_id="r", group="g", character="Defect", seed="S", ascension=9))


STARTED = dict(type="encounter_started", encounter_id="e1", encounter="ENCOUNTER.1", act=1, kind="normal",
               from_event=False)
WON = dict(type="encounter_completed", encounter_id="e1", result="victory")


def test_cards_grow_by_play_or_fatal_blow_each_copy_once_a_fight():
    fight = Fight()
    fight.decide(frame([GA]), [STARTED], "p0")
    fight.decide(frame([GA]), choice="p0")
    fight.decide(frame([GA]), choice="p0")  # a third play of two copies: still two
    fight.decide(frame([SCYTHE, "CARD.STRIKE"]), choice="p0")
    fight.decide(frame([FEED]), choice="p0")
    fight.decide(frame([FEED]), choice="p0")  # the first Feed was not fatal: max HP the same
    fight.decide(frame([GREED], max_hp=83), choice="p0")  # the second was
    fight.decide(frame([GREED]), choice="p0")  # no gold: not fatal
    fight.decide(frame([HUNT], gold=119), choice="p0")  # gold: fatal; The Hunt played next
    # The Hunt's blow ended the fight: fatal, though its power went with the fight.
    fight.decide(frame(phase="rewards", relics=[("RELIC.PEN_NIB", 8), ("RELIC.KUNAI", 1)]), [WON])
    (record,) = fight.record()
    assert record["growth"] == {FEED: 1, GA: 2, GREED: 1, HUNT: 1, SCYTHE: 1}
    assert record["relic_counters"] == {"RELIC.PEN_NIB": 8, "RELIC.KUNAI": 1}
    assert not {"events", "copies", "pending", "played"} & set(record)


def test_the_hunt_is_fatal_while_the_fight_goes_on_by_its_power():
    fight = Fight()
    fight.decide(frame([HUNT, HUNT]), [STARTED], "p0")
    fight.decide(frame([HUNT], hunt=1), choice="p0")  # a stack more: fatal
    fight.decide(frame([], hunt=1))  # no more: not
    fight.decide(frame(phase="rewards"), [WON])
    assert fight.record()[0]["growth"] == {HUNT: 1}


def test_a_card_not_in_the_deck_grows_nothing():
    fight = Fight()
    fight.decide(frame(["CARD.STRIKE", "CARD.STRIKE"]), [STARTED], "p1")
    fight.decide(frame(phase="rewards"), [WON])
    assert fight.record()[0]["growth"] == {}


def record(**changes):
    found = dict(kind="normal", act=1, second_boss=False, ascension=9, entry_hp=50, entry_max_hp=70, exit_hp=45,
                 exit_max_hp=70, potions_used=[], died=False, growth={GA: 1, FEED: 1},
                 relic_counters={"RELIC.PEN_NIB": 8, "RELIC.HAPPY_FLOWER": 2, "RELIC.KUNAI": 1})
    return dict(found, **changes)


@pytest.mark.parametrize("relic, threshold, counted", [
    ("RELIC.HAPPY_FLOWER", 3, {2}), ("RELIC.FAKE_HAPPY_FLOWER", 5, {4}), ("RELIC.PENDULUM", 3, {2}),
    ("RELIC.POLLINOUS_CORE", 4, {3}), ("RELIC.IRON_CLUB", 4, {3}), ("RELIC.JOSS_PAPER", 5, {4}),
    ("RELIC.NUNCHAKU", 10, {7, 8, 9}), ("RELIC.PEN_NIB", 10, {7, 8, 9}), ("RELIC.TUNING_FORK", 10, {7, 8, 9}),
    ("RELIC.GALACTIC_DUST", 10, {7, 8, 9}), ("RELIC.KUNAI", 3, set())])
def test_a_relic_counts_left_near_its_trigger(relic, threshold, counted):
    for count in range(threshold):
        assert growth_events(dict(relic_counters={relic: count})) == (count in counted)


def test_each_growth_event_takes_the_bonus_off_but_not_where_the_run_ends():
    assert growth_events(record()) == 4  # two cards, Pen Nib at 8, Happy Flower at 2
    assert growth_events(dict(kind="normal")) == 0  # a record from before
    growing = with_growth(FIXED, 0.5)
    assert cost(record(), growing) == (5.0, 0.0, 0.0, 0.0, -2.0) and cost(record(), growing).total == 3.0
    assert cost(record(), FIXED).growth == 0.0 and with_growth(FIXED, 0) is FIXED
    assert cost(record(exit_hp=0, died=True), growing).growth == 0.0
    final = record(kind="boss", act=3, exit_hp=None, exit_max_hp=None, last_hp=40, last_max_hp=70)
    assert costs.relief(final) == costs.RUN_ENDING and cost(final, growing).growth == 0.0
    assert growing.price("POTION.FIRE", record()) == FIXED.price("POTION.FIRE", record())
    with pytest.raises(ValueError):
        Growing(FIXED, -1.0)


def test_growth_rides_with_the_prices_and_their_digest():
    growing = with_growth(FIXED, 0.5)
    again = potion_prices.from_json(json.loads(json.dumps(potion_prices.to_json(growing))))
    assert again == growing and potion_prices.digest(again) == potion_prices.digest(growing)
    assert potion_prices.digest(growing) != potion_prices.digest(FIXED)


def test_a_report_shows_the_growth_item_and_the_cost_without_it():
    row = dict(status="complete", kind="normal", act=1, floor=3, second_boss=False, ascension=9, id="f1",
               record=dict(record(), last_hp=45, last_max_hp=70, opening_selection=False, decisions=4), behavior=None)
    found = tally([row], with_growth(FIXED, 0.5))
    assert found["items"]["growth"] == -2.0 and found["cost"] == 3.0 and found["cost_without_growth"] == 5.0
    assert tally([row])["items"]["growth"] == 0.0


def test_the_cli_takes_a_growth_reward():
    common = ["iterate", "--checkpoint", "c", "--output", "o", "--iterations", "1"]
    assert parser().parse_args([*common, "--growth-bonus", "0.5"]).growth_bonus == 0.5
    assert parser().parse_args(common).growth_bonus == 0.0
    fight_eval = ["fight-eval", "--checkpoint", "c", "--fights", "f", "--output", "o", "--growth-bonus", "0.5"]
    assert parser().parse_args(fight_eval).growth_bonus == 0.5
