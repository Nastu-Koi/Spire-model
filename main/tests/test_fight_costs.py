"""The cost of a fight and its return (`main.learn.costs`), item by item against the
table of the spec: the HP weight and the potion prices by kind of fight, free potions,
maximum HP, death, a revival that costs nothing, the final boss, and prices that can be replaced."""

from copy import deepcopy

import pytest

from main.learn import costs
from main.learn.costs import ACT_CLEAR_HEAL, FIXED, NONE, RUN_ENDING, Prices, cost, fight_return, relief

BASE, ELEVATED, HIGH = "POTION.FIRE_POTION", "POTION.CLARITY", "POTION.SWIFT_POTION"
ROCK = "POTION.POTION_SHAPED_ROCK"


def fight(**changes):
    """A fight's record as the fight segments write it: a normal fight of act 1 at
    ascension 0 that cost nothing unless changed."""
    record = dict(run_id="r0", ascension=0, kind="normal", act=1, floor=3, encounter="SHRINKER_BEETLE_WEAK",
                  second_boss=False, from_event=False, entry_hp=60, entry_max_hp=80, exit_hp=60, exit_max_hp=80,
                  potions_used=[], died=False, decisions=12)
    return record | changes


NORMAL = dict(kind="normal", act=1)
ELITE = dict(kind="elite", act=2)
ACT1_BOSS = dict(kind="boss", act=1)
ACT2_BOSS = dict(kind="boss", act=2)
FINAL_BOSS = dict(kind="boss", act=3, ascension=0)
A10_FIRST_BOSS = dict(kind="boss", act=3, ascension=10, second_boss=False)
A10_SECOND_BOSS = dict(kind="boss", act=3, ascension=10, second_boss=True)


@pytest.mark.parametrize("where, relieved, weight, prices", [
    (NORMAL, NONE, 1.0, (7, 9, 11)),
    (ELITE, NONE, 1.0, (7, 9, 11)),
    (ACT1_BOSS, ACT_CLEAR_HEAL, 0.2, (7, 9, 11)),
    (ACT2_BOSS, ACT_CLEAR_HEAL, 0.2, (7, 9, 11)),
    (FINAL_BOSS, RUN_ENDING, 1.0, (1, 1, 1)),
    (dict(FINAL_BOSS, ascension=9), RUN_ENDING, 1.0, (1, 1, 1)),
    (A10_FIRST_BOSS, NONE, 1.0, (7, 9, 11)),
    (A10_SECOND_BOSS, RUN_ENDING, 1.0, (1, 1, 1)),
])
def test_the_hp_weight_and_the_price_of_each_tier_by_kind_of_fight(where, relieved, weight, prices):
    assert relief(fight(**where)) == relieved
    lost = fight(**where, exit_hp=40)
    assert cost(lost, FIXED) == (pytest.approx(20 * weight), 0.0, 0.0, 0.0)
    for potion, price in zip((BASE, ELEVATED, HIGH), prices):
        assert cost(fight(**where, potions_used=[potion]), FIXED).potions == price
    both = fight(**where, exit_hp=40, potions_used=[BASE, HIGH, BASE])
    assert cost(both, FIXED).total == pytest.approx(20 * weight + 2 * prices[0] + prices[2])


def test_the_solvers_tiers():
    assert [costs.tier(p) for p in (BASE, ELEVATED, HIGH, "POTION.GLOWWATER_POTION", "POTION.TOUCH_OF_INSANITY")] == [
        0, 1, 2, 2, 1]
    assert len(costs.HIGH) == 6 and len(costs.ELEVATED) == 7 and not costs.HIGH & costs.ELEVATED


def test_a_potion_at_an_act_boss_pays_where_it_saves_five_times_its_price():
    """The last column of the spec's table: at the boss of the first two acts a base
    potion is worth 35 HP, at the final boss 1."""
    used = fight(**ACT1_BOSS, exit_hp=50, potions_used=[BASE])
    kept = fight(**ACT1_BOSS, exit_hp=15)
    assert cost(used, FIXED).total == pytest.approx(cost(kept, FIXED).total)
    assert cost(fight(**FINAL_BOSS, exit_hp=59, potions_used=[HIGH]), FIXED).total == cost(
        fight(**FINAL_BOSS, exit_hp=58), FIXED).total


@pytest.mark.parametrize("where", [NORMAL, ELITE, ACT2_BOSS, FINAL_BOSS, A10_FIRST_BOSS])
def test_the_potion_shaped_rock_is_free(where):
    assert cost(fight(**where, potions_used=[ROCK, ROCK]), FIXED).potions == 0.0
    assert FIXED.price(ROCK, fight(**where)) == 0.0


def test_maximum_hp_counts_whole_and_a_gain_lowers_the_cost():
    assert cost(fight(exit_max_hp=74, exit_hp=54), FIXED) == (6.0, 6.0, 0.0, 0.0)
    # Not discounted at an act boss, where the HP counts a fifth.
    assert cost(fight(**ACT2_BOSS, exit_max_hp=75, exit_hp=50), FIXED) == (pytest.approx(2.0), 5.0, 0.0, 0.0)
    gained = cost(fight(exit_max_hp=87, exit_hp=67), FIXED)
    assert gained.max_hp == -7.0 and gained.hp == -7.0 and gained.total == -14.0
    assert fight_return(fight(exit_max_hp=87, exit_hp=67), FIXED) == pytest.approx(0.14)


@pytest.mark.parametrize("where, weight", [(NORMAL, 1.0), (ACT1_BOSS, 0.2), (FINAL_BOSS, 1.0)])
def test_a_death_counts_the_hp_down_to_zero_and_a_hundred_on_top(where, weight):
    died = fight(**where, exit_hp=0, exit_max_hp=80, died=True, potions_used=[BASE])
    price = FIXED.price(BASE, died)
    assert cost(died, FIXED) == (pytest.approx(60 * weight), 0.0, price, 100.0)
    assert fight_return(died, FIXED) == pytest.approx(-(60 * weight + price + 100) / 100)
    # Whatever HP the record has at the death, and without it.
    assert cost(fight(**where, exit_hp=None, exit_max_hp=None, died=True), FIXED) == (
        pytest.approx(60 * weight), 0.0, 0.0, 100.0)
    assert cost(fight(**where, exit_hp=12, died=True), FIXED).hp == pytest.approx(60 * weight)


def test_a_revival_is_not_priced():
    """The solver's premium of 9 per HP a revival restored is left out: at about 24 HP
    it would cost more than dying. A fairy in a bottle brings the player back at 24 HP:
    the fight counts the HP lost down to 24 and nothing for the revival."""
    assert costs.Cost._fields == ("hp", "max_hp", "potions", "death")
    revived = fight(kind="elite", exit_hp=24, revived_hp=24)
    assert cost(revived, FIXED) == (36.0, 0.0, 0.0, 0.0)


def test_the_final_boss_won_counts_no_hp_when_no_decision_follows():
    won = fight(**FINAL_BOSS, exit_hp=None, exit_max_hp=None, potions_used=[HIGH, ELEVATED])
    assert cost(won, FIXED) == (0.0, 0.0, 2.0, 0.0)
    assert fight_return(won, FIXED) == pytest.approx(-0.02)
    # With the exit HP recorded it counts whole, as the solver compares the final boss.
    assert cost(fight(**FINAL_BOSS, exit_hp=25), FIXED).hp == 35.0
    # The same goes for the second boss of act 3 at ascension 10; its first is an ordinary fight.
    assert cost(fight(**A10_SECOND_BOSS, exit_hp=None, exit_max_hp=None), FIXED).total == 0.0
    with pytest.raises(ValueError, match="exit_hp"):
        cost(fight(**A10_FIRST_BOSS, exit_hp=None), FIXED)


def test_the_final_boss_won_counts_the_hp_of_its_last_decision():
    """The fight segments record the HP at a fight's last decision: where the run ends
    with the fight won, it stands for the exit HP the end frame cannot give."""
    won = fight(**FINAL_BOSS, exit_hp=None, exit_max_hp=None, last_hp=41, last_max_hp=78, potions_used=[HIGH])
    assert cost(won, FIXED) == (19.0, 2.0, 1.0, 0.0)
    # Only there: a death counts down to 0, and a fight the run goes on from needs its exit.
    assert cost(fight(**FINAL_BOSS, exit_hp=None, last_hp=41, died=True), FIXED).hp == 60.0
    with pytest.raises(ValueError, match="no exit"):
        cost(fight(**ELITE, exit_hp=None, last_hp=41, last_max_hp=80), FIXED)


@pytest.mark.parametrize("record", [
    fight(exit_hp=None), fight(exit_max_hp=None), fight(**ACT2_BOSS, exit_hp=None), fight(**ELITE, exit_max_hp=None)])
def test_a_fight_the_run_goes_on_from_must_have_its_exit(record):
    with pytest.raises(ValueError, match="no exit"):
        cost(record, FIXED)


def test_records_that_cannot_be_counted():
    with pytest.raises(ValueError, match="kind"):
        cost(fight(kind="event"), FIXED)
    with pytest.raises(ValueError, match="ascension"):
        relief(fight(kind="boss", act=3, ascension=None))
    # The second boss is the last fight whatever the ascension says.
    assert relief(fight(kind="boss", act=3, ascension=None, second_boss=True)) == RUN_ENDING
    for bad in ((7, 9), (7, -1, 11)):
        with pytest.raises(ValueError, match="three prices"):
            Prices(tiers=bad)


class Flat:
    """Prices of another shape: anything with `price(potion, fight)` serves."""

    def price(self, potion, fight):
        return 4.0 if fight["kind"] == "elite" else 2.0


def test_other_prices_price_the_same_records_again():
    records = [fight(exit_hp=50, potions_used=[BASE]), fight(**ELITE, exit_hp=30, potions_used=[HIGH, ELEVATED]),
               fight(**ACT2_BOSS, exit_hp=10, potions_used=[ELEVATED]),
               fight(**FINAL_BOSS, exit_hp=None, exit_max_hp=None, potions_used=[BASE, BASE]),
               fight(**ELITE, exit_hp=0, died=True, potions_used=[ROCK])]
    kept = deepcopy(records)
    fixed = [fight_return(r, FIXED) for r in records]
    assert fixed == pytest.approx([-0.17, -0.5, -0.19, -0.02, -1.6])
    solver = Prices(tiers=(9, 14, 18))
    assert [fight_return(r, solver) for r in records] == pytest.approx([-0.19, -0.62, -0.24, -0.02, -1.6])
    cheaper = Prices(tiers=(1, 2, 3), final=(0, 0, 0), free=frozenset())
    assert [fight_return(r, cheaper) for r in records] == pytest.approx([-0.11, -0.35, -0.12, 0.0, -1.61])
    assert [fight_return(r, Flat()) for r in records] == pytest.approx([-0.12, -0.38, -0.12, -0.04, -1.64])
    assert records == kept
