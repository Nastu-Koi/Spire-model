"""Potion prices from what keeping a potion saves (`.scratch/training-plan` ticket 10).

A potion used in a fight costs its price in HP (`costs.cost`). Fitted, the price is what
keeping the potion saves from where the player stands to the next act's start: the
fights' costs on the way (without potion prices), and the potions still held there at
the next act's prices. The labels are pairs of greedy continuations of the same map
save, one as saved and one with the potion taken out (`main/play/potion_value.py`): a
run is determined by its actions, so all the two differ by is the potion and what came
of it. In the last act the pair runs to the run's end, where a potion is worth nothing.

The price is linear in where the player stands and what they have (`FEATURES`, from
`fights.state` with the run's character and ascension), the state a fight was entered
in for a fight, the map decision for a pair:

- a base price for each act with the belt full or not (not penalized);
- a slope in the share of HP missing for each act; slopes in the floors left to the
  act's boss, the potions held, the ascension, the deck's counts and the relics;
- an offset for each character;
- an offset for each tier of the solver (`costs.tier`) and one for each potion, the
  potion's shrunk toward its tier's.

It is fitted by ridge regression, each group of columns with its own penalty chosen by
cross-validation, the folds made of whole source runs so that no run is on both sides
(`fit`). The acts are fitted backward: a pair of the first or second act prices the
potions it holds at the next act's start by the model, so the fit goes round until the
model stops changing (the last act's labels need no model, the second's then stand,
then the first's). The fit is set against the base prices alone (the mean of each act
and belt), and kept only where it predicts held-out pairs better.

The prices a fight is priced at (`Smoothed`): the mean of the previous prices and the
fitted ones, held within 30% of the previous. The run's last fight and the free potions
keep the previous prices (`costs.FIXED`'s 1/1/1 and 0).
"""

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from main.env.protocol import CHARACTERS
from main.learn import costs

ACTS = (1, 2, 3)
# The floors of each act where a state does not say (its map's rows and one).
ACT_FLOORS = (17, 16, 15)
# The slopes, by name: the share of HP missing at each act, then the rest.
SLOPES = ("hp_missing_1", "hp_missing_2", "hp_missing_3", "floors_left", "held", "ascension", "cards", "attacks",
          "skills", "powers", "upgraded", "curses", "relics")
TIERS = ("elevated", "high")  # the base tier is the base prices'
# The groups of columns, each with its own penalty; the base prices have none.
GROUPS = ("state", "character", "tier", "potion")
# What the penalties are chosen from, as multiples of the pairs fitted.
GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
FOLDS = 5
CAP = 0.3  # how far a fit moves a price at once, as a share of the previous price
# Fights cost here without the potions' prices: a pair's difference in potions is what
# it holds at the next act's start.
UNPRICED = costs.Prices(tiers=(0.0, 0.0, 0.0), final=(0.0, 0.0, 0.0), free=frozenset())


def features(state, character, ascension):
    """What a price depends on, from where the player stands and what they have
    (`fights.state`) and the run's character and ascension."""
    act = state["act"]
    if act not in ACTS:
        raise ValueError(f"A state of act {act!r} has no price")
    floors = state.get("act_floors") or ACT_FLOORS[act - 1]
    potions = list(state.get("potions") or ())
    capacity = state.get("capacity")
    deck = state.get("deck") or {}
    found = dict(act=act, full=capacity is not None and len(potions) >= capacity, character=character,
                 floors_left=max(floors - (state.get("floor") or 0), 0), held=len(potions),
                 ascension=ascension or 0, cards=deck.get("cards", 0), attacks=deck.get("attacks", 0),
                 skills=deck.get("skills", 0), powers=deck.get("powers", 0), upgraded=deck.get("upgraded", 0),
                 curses=deck.get("curses", 0) + deck.get("statuses", 0), relics=state.get("relics", 0))
    missing = 1.0 - state["hp"] / state["max_hp"] if state.get("max_hp") else 0.0
    for number in ACTS:
        found[f"hp_missing_{number}"] = missing if number == act else 0.0
    return found


def fight_features(fight):
    """`features` of the state a fight was entered in (its record's `entry_state`)."""
    state = fight.get("entry_state")
    if not state:
        raise ValueError(f"The fight at floor {fight.get('floor')} of {fight.get('run_id')} has no entry state: "
                         "a fitted price needs one (fights recorded since ticket 10's stage 2)")
    return features(state, fight.get("character"), fight.get("ascension"))


@dataclass
class Design:
    """The columns of the linear price: the base prices, the slopes (standardized by
    `means` and `scales`), the characters, the tiers and the potions named in `potions`
    (a potion it does not name has its tier's offset)."""

    potions: tuple
    means: dict
    scales: dict

    @classmethod
    def of(cls, rows):
        """The design of `rows` (each `features` with its `potion`)."""
        means, scales = {}, {}
        for name in SLOPES:
            values = [row[name] for row in rows]
            if name.startswith("hp_missing_"):
                # Each act's slope is fitted on its own act's rows: the others are zero.
                values = [row[name] for row in rows if row["act"] == int(name[-1])] or [0.0]
            mean = sum(values) / len(values)
            spread = math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
            means[name], scales[name] = mean, spread if spread > 1e-9 else 1.0
        return cls(tuple(sorted({row["potion"] for row in rows})), means, scales)

    @property
    def columns(self):
        return ([f"base_{act}_{'full' if full else 'room'}" for act in ACTS for full in (False, True)]
                + list(SLOPES) + [f"character_{c}" for c in CHARACTERS] + [f"tier_{t}" for t in TIERS]
                + [f"potion_{p}" for p in self.potions])

    def groups(self):
        """Each column's group: None for the base prices."""
        return ([None] * 6 + ["state"] * len(SLOPES) + ["character"] * len(CHARACTERS) + ["tier"] * len(TIERS)
                + ["potion"] * len(self.potions))

    def row(self, found, potion):
        """The columns of a potion priced at `found` (`features`)."""
        values = np.zeros(len(self.columns))
        values[(found["act"] - 1) * 2 + int(found["full"])] = 1.0
        for index, name in enumerate(SLOPES):
            value = found[name]
            if name.startswith("hp_missing_") and found["act"] != int(name[-1]):
                continue  # zero in another act: not centered there
            values[6 + index] = (value - self.means[name]) / self.scales[name]
        start = 6 + len(SLOPES)
        if found["character"] in CHARACTERS:
            values[start + CHARACTERS.index(found["character"])] = 1.0
        start += len(CHARACTERS)
        tier = costs.tier(potion)
        if tier:
            values[start + tier - 1] = 1.0
        start += len(TIERS)
        if potion in self.potions:
            values[start + self.potions.index(potion)] = 1.0
        return values

    def to_json(self):
        return dict(potions=list(self.potions), means=self.means, scales=self.scales)


@dataclass
class LinearPrices:
    """Prices linear in a fight's entry state (`Design`), as fitted (`fit`). In the
    run's last fight `final` by tier and the potions in `free` at 0, as `costs.FIXED`."""

    design: Design
    coefficients: tuple
    final: tuple = costs.FIXED.final
    free: frozenset = costs.FIXED.free
    info: dict = field(default_factory=dict)

    def value(self, potion, found):
        """The fitted price of `potion` at `found` (`features`): what keeping it saves, in HP."""
        return float(self.design.row(found, potion) @ np.asarray(self.coefficients))

    def price(self, potion, fight):
        if potion in self.free:
            return 0.0
        if costs.relief(fight) == costs.RUN_ENDING:
            return float(self.final[costs.tier(potion)])
        return self.value(potion, fight_features(fight))

    def to_json(self):
        return dict(kind="linear", design=self.design.to_json(), columns=self.design.columns,
                    coefficients=list(self.coefficients), final=list(self.final), free=sorted(self.free),
                    info=self.info)

    def describe(self):
        return dict(kind="linear", digest=digest(self), columns=len(self.coefficients), info=self.info)


@dataclass
class Smoothed:
    """The prices a fit moves to (`fitted`) from `previous`: their mean, held within
    `cap` of the previous price. The run's last fight and the free potions keep the
    previous prices."""

    previous: object
    fitted: object
    cap: float = CAP

    def price(self, potion, fight):
        old = self.previous.price(potion, fight)
        if potion in costs.FIXED.free or costs.relief(fight) == costs.RUN_ENDING:
            return old
        middle = (old + self.fitted.price(potion, fight)) / 2
        return min(max(middle, old * (1 - self.cap)), old * (1 + self.cap))

    def to_json(self):
        return dict(kind="smoothed", cap=self.cap, previous=to_json(self.previous), fitted=to_json(self.fitted))

    def describe(self):
        return dict(kind="smoothed", digest=digest(self), cap=self.cap, fitted=self.fitted.describe())


def to_json(prices):
    if isinstance(prices, costs.Prices):
        return dict(kind="fixed", tiers=list(prices.tiers), final=list(prices.final), free=sorted(prices.free))
    if isinstance(prices, costs.Growing):
        return dict(kind="growing", prices=to_json(prices.prices), bonus=prices.bonus)
    return prices.to_json()


def from_json(data):
    kind = data.get("kind")
    if kind == "fixed":
        return costs.Prices(tiers=tuple(data["tiers"]), final=tuple(data["final"]), free=frozenset(data["free"]))
    if kind == "linear":
        design = Design(tuple(data["design"]["potions"]), data["design"]["means"], data["design"]["scales"])
        if data["columns"] != design.columns:
            raise ValueError("The price model's columns are not this code's: refit it")
        return LinearPrices(design, tuple(data["coefficients"]), tuple(data["final"]), frozenset(data["free"]),
                            data.get("info", {}))
    if kind == "smoothed":
        return Smoothed(from_json(data["previous"]), from_json(data["fitted"]), data["cap"])
    if kind == "growing":
        return costs.Growing(from_json(data["prices"]), data["bonus"])
    raise ValueError(f"Unknown kind of prices: {kind!r}")


def digest(prices):
    text = json.dumps(to_json(prices), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load(spec):
    """Prices by name: "fixed" (`costs.FIXED`) or a file a fit wrote (`save`)."""
    if spec in (None, "fixed"):
        return costs.FIXED
    return from_json(json.loads(Path(spec).read_text(encoding="utf-8")))


def save(prices, path):
    Path(path).write_text(json.dumps(to_json(prices), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


# ---- The labels ----

def branch_cost(branch):
    """What a continuation's fights cost, without the potions' prices (`UNPRICED`)."""
    return sum(costs.cost(record, UNPRICED).total for record in branch["records"])


def held_value(branch, act, prices, character, ascension):
    """What a continuation holds at the next act's start at `prices`: nothing in the
    last act, or where it ended before."""
    state = branch.get("cut_state")
    if act == ACTS[-1] or branch.get("outcome") != "reached" or not state:
        return 0.0
    fight = dict(kind="normal", act=state["act"], second_boss=False, ascension=ascension, character=character,
                 entry_state=state, floor=state.get("floor"))
    return sum(prices.price(potion, fight) for potion in state.get("potions") or ())


def pairs(points, branches):
    """The pairs of continuations of each point (`potion_value.play_branches`'s rows):
    as saved against each with a potion taken out, both played through. Each with the
    point's features, the potion and the difference of the fights' costs (the potion
    taken out less as saved)."""
    by_point = defaultdict(dict)
    for branch in branches:
        by_point[branch["point"]][branch["branch"]] = branch
    found = []
    for point in points:
        played = by_point.get(point["id"], {})
        keep = played.get("keep")
        if keep is None or keep["status"] != "complete":
            continue
        for name, branch in played.items():
            if not name.startswith("drop") or branch["status"] != "complete":
                continue
            found.append(dict(point=point["id"], run=f"{point['character']}:{point['seed']}",
                              character=point["character"], ascension=point["ascension"],
                              features=features(point["state"], point["character"], point["ascension"]),
                              potion=branch["dropped"], costs=branch_cost(branch) - branch_cost(keep),
                              keep=keep, drop=branch))
    return found


def labels(found, prices):
    """Each pair's label at `prices` for what is held at the next act's start: the
    costs it saves and the potions it keeps."""
    return np.array([pair["costs"]
                     + held_value(pair["keep"], pair["features"]["act"], prices, pair["character"], pair["ascension"])
                     - held_value(pair["drop"], pair["features"]["act"], prices, pair["character"], pair["ascension"])
                     for pair in found])


def played_alike(first, second):
    """Whether two continuations came to the same fights and the same end, whatever
    they were named."""
    def facts(row):
        return [{key: value for key, value in record.items() if key not in ("run_id", "group")}
                for record in row["records"]], row["cut_state"]
    return facts(first) == facts(second)


def determinism(results):
    """How far the points played twice part (`potion_value.play_branches`' `keep-again`):
    of the points whose continuation as saved was played again, how many came to the
    same records, and the spread of the difference of their fights' costs."""
    by_point = defaultdict(dict)
    for row in results:
        by_point[row["point"]][row["branch"]] = row
    differences, same = [], 0
    for played in by_point.values():
        first, second = played.get("keep"), played.get("keep-again")
        if not first or not second or first["status"] != "complete" or second["status"] != "complete":
            continue
        same += played_alike(first, second)
        differences.append(branch_cost(second) - branch_cost(first))
    spread = float(np.std(differences, ddof=1)) if len(differences) > 1 else None
    return dict(points=len(differences), same=same, cost_difference_sd=spread)


# ---- The fit ----

def _penalty(design, chosen, count):
    groups = design.groups()
    return np.array([0.0 if group is None else chosen[group] * count for group in groups])


def _solve(gram, moment, penalty):
    return np.linalg.solve(gram + np.diag(penalty) + np.eye(len(penalty)) * 1e-9, moment)


def folds_of(found, folds=FOLDS):
    """Each pair's fold, whole source runs to a fold, the same every time."""
    return np.array([int(hashlib.sha256(pair["run"].encode()).hexdigest(), 16) % folds for pair in found])


def cross_validate(matrix, target, fold, design, grid=GRID, base_only=False):
    """The penalties of the groups with the least held-out squared error, and that error
    (mean per pair). With `base_only`, the base prices alone: each act and belt's mean."""
    count = len(target)
    numbers = sorted(set(fold.tolist()))
    pieces = []
    for number in numbers:
        train, test = fold != number, fold == number
        pieces.append((matrix[train].T @ matrix[train], matrix[train].T @ target[train], matrix[test], target[test]))
    if base_only:
        keep = np.array([group is None for group in design.groups()])
        error = 0.0
        for gram, moment, held, truth in pieces:
            beta = np.zeros(len(keep))
            beta[keep] = np.linalg.lstsq(gram[np.ix_(keep, keep)], moment[keep], rcond=None)[0]
            error += float(((held @ beta - truth) ** 2).sum())
        return None, error / count
    best = None
    for state in grid:
        for character in grid:
            for tier in grid:
                for potion in grid:
                    chosen = dict(state=state, character=character, tier=tier, potion=potion)
                    penalty = _penalty(design, chosen, count)
                    error = sum(float(((held @ _solve(gram, moment, penalty) - truth) ** 2).sum())
                                for gram, moment, held, truth in pieces)
                    if best is None or error < best[1]:
                        best = (chosen, error)
    return best[0], best[1] / count


def fit(points, branches, previous, *, grid=GRID, folds=FOLDS, rounds=6, tolerance=0.05, info=None):
    """The linear prices of the pairs of `points` and `branches` (`pairs`), the potions
    held at the next act's start priced first at `previous` and then at each round's
    fit until it moves no fitted price by more than `tolerance` HP. Returns the prices
    and the report: the penalties and the held-out error against the base prices
    alone's, the coefficients with their standard errors, the pairs by act. Where the
    base prices alone predict held-out pairs as well, they are the fit."""
    found = pairs(points, branches)
    if not found:
        raise ValueError("No pair of continuations to fit the prices on")
    rows = [dict(pair["features"], potion=pair["potion"]) for pair in found]
    design = Design.of(rows)
    matrix = np.stack([design.row(row, row["potion"]) for row in rows])
    fold = folds_of(found, folds)
    valuer, fitted, moved, rounds_made = previous, None, None, 0
    for rounds_made in range(1, rounds + 1):
        target = labels(found, valuer)
        chosen, error = cross_validate(matrix, target, fold, design, grid)
        beta = _solve(matrix.T @ matrix, matrix.T @ target, _penalty(design, chosen, len(target)))
        model = LinearPrices(design, tuple(beta.tolist()))
        moved = None if fitted is None else float(np.abs(matrix @ (beta - np.asarray(fitted.coefficients))).max())
        fitted, valuer = model, model
        if moved is not None and moved < tolerance:
            break
    target = labels(found, fitted)
    chosen, error = cross_validate(matrix, target, fold, design, grid)
    _, base_error = cross_validate(matrix, target, fold, design, base_only=True)
    kept = "model" if error < base_error else "base"
    # The base prices alone are the model with every other column held at zero.
    used = np.array([kept == "model" or group is None for group in design.groups()])
    part = matrix[:, used]
    penalty = _penalty(design, chosen, len(target))[used]
    inverse = np.linalg.inv(part.T @ part + np.diag(penalty) + np.eye(len(penalty)) * 1e-9)
    beta, errors = np.zeros(len(used)), np.zeros(len(used))
    beta[used] = inverse @ part.T @ target
    residual = target - matrix @ beta
    variance = float(residual @ residual) / max(len(target) - int(used.sum()), 1)
    errors[used] = np.sqrt(np.clip(np.diag(variance * inverse @ (part.T @ part) @ inverse), 0, None))
    report = dict(
        pairs=len(found), points=len({pair["point"] for pair in found}), runs=len({pair["run"] for pair in found}),
        rounds=rounds_made, moved=moved, penalties=chosen, held_out_error=error, base_held_out_error=base_error,
        kept=kept, label_sd=float(target.std()),
        by_act={act: dict(pairs=int(sum(row["act"] == act for row in rows)),
                          mean=float(target[[row["act"] == act for row in rows]].mean())
                          if any(row["act"] == act for row in rows) else None) for act in ACTS},
        coefficients={name: dict(value=float(value), error=float(error))
                      for name, value, error in zip(design.columns, beta, errors)})
    return LinearPrices(design, tuple(beta.tolist()), info=dict(info or {}, pairs=len(found), kept=kept)), report
