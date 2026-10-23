"""The cost of a fight, from its record, and the prices of the potions it used.

A fight is counted in HP the way CombatSolver 0.44 counts a route
(`CombatBeamSolver.HealthResourceCost`, `ActEndingBossPolicy`, `PotionUsePolicy`):

    cost = w·(entry HP − exit HP) + (entry max HP − exit max HP)
           + Σ prices of the potions used + 100·death

and a fight segment's return is −cost / 100: a death is about −1 and a point of HP
about −0.01, the scale of the act groups' ±1. The weight w is 1/5 at the boss of the
first and second act, whose HP the act's end heals back, and 1 elsewhere (`relief`).
A potion's price is the price table's (`Prices`): the solver's three tiers at 7, 9
and 11, and 1 in the run's last fight, where nothing is left to keep a potion for.
Anything with the same `price(potion, fight)` serves as the prices: the returns are
counted from the records whenever they are wanted, so other prices (a fitted table)
price the same fights again without playing them again.

A fight's record is a dict with its kind (normal, elite, boss), act, `second_boss`,
the run's `ascension`, the entry and exit HP and maximum HP (and those at its last
decision, for a run that ends with it), the content ids of the potions used
(`potions_used`) and whether the player died (`died`). Nothing else of it is read.

The solver's premium on a revival (`DeathSavePremium`, 9 per HP it restored) is left
out: at the 20-odd HP a fairy brings back it would make a revival cost more than
dying. A revival costs the HP lost down to what it restored; the fairy it takes would
be priced as a used potion, but the engine does not tell when one is spent.
"""

from dataclasses import dataclass
from typing import NamedTuple

KINDS = ("normal", "elite", "boss")
LAST_ACT = 3
DOUBLE_BOSS = 10  # AscensionLevel.DoubleBoss: from it on, the last act has a second boss
DEATH = 100.0  # the cost of dying, in HP
SCALE = 100.0  # a fight's return is −cost / SCALE

# What the HP a fight leaves is worth after it (`ActEndingBossPolicy.ResolveHpRelief`):
# as it is; healed back at the end of the act; or nothing, the run being over.
NONE, ACT_CLEAR_HEAL, RUN_ENDING = "none", "act_clear_heal", "run_ending"
HP_WEIGHTS = {NONE: 1.0, ACT_CLEAR_HEAL: 0.2, RUN_ENDING: 1.0}

# The solver's tiers of potions (`PotionUsePolicy`); every other potion is of the base tier.
HIGH = frozenset("POTION." + name for name in (
    "GLOWWATER_POTION", "SWIFT_POTION", "GAMBLERS_BREW", "DUPLICATOR", "OROBIC_ACID", "POT_OF_GHOULS"))
ELEVATED = frozenset("POTION." + name for name in (
    "DISTILLED_CHAOS", "CLARITY", "RADIANT_TINCTURE", "CURE_ALL", "LIQUID_MEMORIES", "BOTTLED_POTENTIAL",
    "TOUCH_OF_INSANITY"))
# The solver prices a potion of rarity Token (5) at 0, and the potion-shaped rock where
# Petrified Toad brings a new one each fight. The engine names a potion's rarity in its
# `content_info` reply; of the game's potions only the rock is a Token, so the rarity
# frees it with or without the toad.
TOKEN = frozenset({"POTION.POTION_SHAPED_ROCK"})


def relief(fight):
    """What the HP a fight leaves is worth after it, as the solver decides it
    (`ActEndingBossPolicy.ResolveHpRelief`): the boss of an act before the last is
    healed after (ACT_CLEAR_HEAL); the boss that ends the run, the last act's or its
    second where it has two (ascension 10), is RUN_ENDING; every other fight is NONE,
    the first of the last act's two bosses among them."""
    kind = fight["kind"]
    if kind not in KINDS:
        raise ValueError(f"Unknown kind of fight: {kind!r}")
    if kind != "boss":
        return NONE
    if fight["act"] < LAST_ACT:
        return ACT_CLEAR_HEAL
    if fight["second_boss"]:
        return RUN_ENDING
    if fight.get("ascension") is None:
        raise ValueError("The boss of the last act needs the run's ascension: it may have a second")
    return NONE if fight["ascension"] >= DOUBLE_BOSS else RUN_ENDING


def tier(potion):
    """The solver's tier of a potion (a content id): 0 base, 1 elevated, 2 high."""
    return 2 if potion in HIGH else 1 if potion in ELEVATED else 0


@dataclass(frozen=True, slots=True)
class Prices:
    """The price of a potion used in a fight, in HP, by the solver's tier (base,
    elevated, high): `tiers` in every fight but the run's last, `final` in that one;
    the potions in `free` cost nothing anywhere. At the boss of the first or second
    act the HP counts a fifth, so a potion there has to save five times its price."""

    tiers: tuple = (7.0, 9.0, 11.0)
    final: tuple = (1.0, 1.0, 1.0)
    free: frozenset = TOKEN

    def __post_init__(self):
        for name in ("tiers", "final"):
            prices = getattr(self, name)
            if len(prices) != 3 or any(price < 0 for price in prices):
                raise ValueError(f"{name} must be three prices in HP, none negative")

    def price(self, potion, fight):
        if potion in self.free:
            return 0.0
        return float((self.final if relief(fight) == RUN_ENDING else self.tiers)[tier(potion)])


FIXED = Prices()  # the fixed prices, those the solver's demonstrations are solved at


class Cost(NamedTuple):
    """A fight's cost by item, in HP; `total` is the cost."""

    hp: float  # w·(entry HP − exit HP)
    max_hp: float  # entry max HP − exit max HP: a gain lowers the cost
    potions: float  # the prices of the potions used
    death: float  # 100 for dying

    @property
    def total(self):
        return self.hp + self.max_hp + self.potions + self.death


def cost(fight, prices):
    """A fight's cost by item (`Cost`) at `prices` (`Prices`, or anything with its
    `price(potion, fight)`).

    The exit HP is the first decision's after the fight, post-combat healing in it. A
    death counts the HP down to 0, whatever the record says, and DEATH on top. A run
    that ends with the fight won (the final boss) has no decision after it: there the
    HP at the fight's last decision (`last_hp`, `last_max_hp`) stands for the exit, and
    a record without either counts that item as 0. Anywhere else a missing exit is an
    error."""
    stance = relief(fight)
    died = bool(fight["died"])
    ended = died or stance == RUN_ENDING  # no decision after the fight to read the HP from

    def after(field):
        value = fight.get(field)
        if value is None and not ended:
            raise ValueError(f"Fight at floor {fight.get('floor')} of {fight.get('run_id')} has no {field}")
        if value is None and not died:
            value = fight.get(field.replace("exit_", "last_"))
        return value

    hp_after = 0 if died else after("exit_hp")
    max_hp_after = after("exit_max_hp")
    return Cost(
        hp=0.0 if hp_after is None else HP_WEIGHTS[stance] * (fight["entry_hp"] - hp_after),
        max_hp=0.0 if max_hp_after is None else float(fight["entry_max_hp"] - max_hp_after),
        potions=sum((prices.price(potion, fight) for potion in fight.get("potions_used") or ()), 0.0),
        death=DEATH if died else 0.0)


def fight_return(fight, prices):
    """A fight segment's return: −cost / SCALE."""
    return -cost(fight, prices).total / SCALE
