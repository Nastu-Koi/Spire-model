"""Combat demonstrations by the round: fights a round of act groups played, picked for the solver.

In the BC phase every round of act groups (fights greedy, with fight entries:
`groups.sample_groups(fight_entries=True)`) hands some of the fights it played to the
solver (`spire_codex_data.entry_battles`), whose fights of them are the round's combat
demonstrations. The pick (`pick`) is `count` fights not begun by an event, shared
among normal, elite and boss fights 3:4:4 by number of fights (`fight_groups.quotas`).

Within a kind the fights are drawn at their natural frequency: a round of act groups
plays every act of a run from several segments (a group of `n` at the first act, each
of those going on as a group of `m`), so the later acts hold many more segments of a
run than the first. A fight is weighed by one over the segments its run (character and
seed) has in its wave (`natural_weights`), so that each run counts once in each act,
as a run is played.

Fights set up by a selection before their first play (`opening_selection`: a gambling
chip's discard, toasty mittens, a card of several to keep) are drawn first, up to an
`opening` share of the count, each within its kind's quota: the old demonstrations
labelled those choices by a fixed rule, and a round holds only a few of them at their
natural frequency. The rest of each kind's quota is drawn from its other fights.

The draws are seeded by `key` (the run's seed root and the round), so a round picks the
same fights whenever it is picked again. The selection is written for the solver
(`write_selection`): every fight with its id, its run's character and seed, and the
absolute path of its entry save, which the round's clean-up deletes after the update.
"""

import json
import random
from collections import Counter
from pathlib import Path

from main.play.fight_groups import SHARES, candidates, quotas

DIRECTORY = "demos"  # a round's demonstrations, beside its waves
SELECTION = "selection.json"
COUNT = 2000
OPENING = 0.2  # the share of the count the fights set up by a selection may take first


def natural_weights(rows):
    """Each row's weight: one over the segments its run (character and seed) has in its
    wave, as many as its fights there name."""
    segments = {}
    for row in rows:
        segments.setdefault((row["wave"], row["character"], row["seed"]), set()).add(row["run_id"])
    return [1.0 / len(segments[(row["wave"], row["character"], row["seed"])]) for row in rows]


def _drawn(pairs, rng):
    """`(row, weight)` pairs in the order of a draw without replacement at those weights
    (each the larger of its key u ** (1 / weight) first)."""
    return [row for _, row in sorted(((rng.random() ** (1.0 / weight), row) for row, weight in pairs),
                                     key=lambda item: -item[0])]


def pick(rows, count, key, opening=OPENING, shares=SHARES):
    """The fights of `rows` (`fight_groups.candidates`) for the solver, `count` of them
    shared among the kinds by `shares`, and an account of the pick. Each fight picked
    is its row with its `weight` and `reason` (`opening` or `natural`)."""
    if type(count) is not int or count < 0:
        raise ValueError("Combat demonstrations: a number of fights, 0 or more")
    if isinstance(opening, bool) or not isinstance(opening, (int, float)) or not 0 <= opening <= 1:
        raise ValueError("The share of the fights set up by a selection is in [0, 1]")
    entered = [row for row in rows if row["path"] is not None]
    weighed = list(zip(entered, natural_weights(entered)))
    quota = quotas(count, shares)
    cap = int(count * opening)
    taken = {kind: [] for kind in quota}
    first = [(row, weight) for row, weight in weighed if row.get("opening_selection")]
    weights = {id(row): weight for row, weight in weighed}
    for row in _drawn(first, random.Random(f"{key}:opening")):
        if sum(map(len, taken.values())) >= cap:
            break
        if len(taken[row["kind"]]) < quota[row["kind"]]:
            taken[row["kind"]].append(dict(row, weight=weights[id(row)], reason="opening"))
    picked, kinds = [], {}
    for kind in quota:
        chosen = {(row["run_id"], row["fight"]) for row in taken[kind]}
        pool = [(row, weight) for row, weight in weighed if row["kind"] == kind]
        rest = [(row, weight) for row, weight in pool if (row["run_id"], row["fight"]) not in chosen]
        more = _drawn(rest, random.Random(f"{key}:{kind}"))[:quota[kind] - len(taken[kind])]
        found = taken[kind] + [dict(row, weight=weights[id(row)], reason="natural") for row in more]
        picked += found
        kinds[kind] = dict(fights=len(pool), opening_fights=sum(bool(row.get("opening_selection")) for row, _ in pool),
                           quota=quota[kind], opening=len(taken[kind]), picked=len(found),
                           short=quota[kind] - len(found), acts=dict(sorted(Counter(row["act"] for row in found).items())))
    account = dict(fights=len(rows), events=len(rows) - len(entered), picked=len(picked), opening_cap=cap,
                   kinds=kinds, short=[kind for kind, counts in kinds.items() if counts["short"]])
    return picked, account


def fight_id(row):
    """A fight picked, named by its segment and its place there."""
    return f"{row['run_id']}-{int(row['fight']):03d}"


def write_selection(picked, account, path, **about):
    """The selection the solver fights (`entry_battles run --selection`): `fights`, each
    with its id, its run's character and seed and the absolute path of its entry save;
    the account of the pick and what else `about` says (the round, the checkpoint)."""
    fights = [dict(id=fight_id(row), path=str(Path(row["path"]).resolve()), character=row["character"],
                   seed=row["seed"], ascension=row.get("ascension"), kind=row["kind"], act=row["act"],
                   floor=row["floor"], encounter=row["encounter"], second_boss=row["second_boss"],
                   opening_selection=bool(row.get("opening_selection")), wave=row["wave"], weight=row["weight"],
                   reason=row["reason"]) for row in picked]
    if len({fight["id"] for fight in fights}) != len(fights):
        raise ValueError("Two fights picked have the same id")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(about, account=account, fights=fights, extra=[]), ensure_ascii=False, indent=1),
                    encoding="utf-8")
    return path


def pick_round(directory, waves, count, key, opening=OPENING):
    """Pick the fights of the round in `directory` (its `waves`' fight records) and write
    the selection under `<directory>/demos/`; the selection's path and the account."""
    picked, account = pick(candidates(directory, waves), count, key, opening=opening)
    path = write_selection(picked, account, Path(directory) / DIRECTORY / SELECTION, key=key)
    return path, account
