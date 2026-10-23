"""The report of a fight evaluation (`main/play/fight_eval.py`): what the fights cost.

Each fight is priced from its record (`costs.cost`) at the prices given, so that other
prices report the same fights again without playing them again. The report:

- `summary`: by kind of fight (and all together) the mean cost and its items, the
  death rate, the potions used per fight and the share of fights that used one, the
  deaths outside the run's last fight with a potion to drink left unused
  (`held_at_death`), and the danger decisions (below 30% of max
  HP with a potion to use): how often a potion was used there and, for the policy, its
  probability of using one at temperature 1 and at the fight temperature;
- `gap`: against the solver's fights of the same saves (`fight_eval.solver_rows`),
  fight by fight, the mean difference of the costs by kind;
- `behavior`: the table of how the two played the fights both finished, a row for each
  kind of action and what it did: each potion (the fights that held it, those that
  used it), each choice that sets a fight up (by the relic that offers it: how many
  cards Gambling Chip discarded, which card Toasty Mittens exhausted or Toolbox took)
  and each other selection (by what opened it: how many cards it picked);
- `paired`: two plays of the same set fight by fight (`paired_plays`): the fights
  played with other actions or come to another record, and the mean difference of
  their costs.

A fight the run ended with, won, has no decision after it: its HP after is the HP at
its last decision, for the solver's fights as for the policy's (`record`).
"""

import math
from collections import Counter, defaultdict

from main.learn.costs import FIXED, KINDS, RUN_ENDING, cost, relief

ALL = "all"
# What a row's record is compared on between two plays of a fight.
OUTCOME = ("exit_hp", "exit_max_hp", "last_hp", "last_max_hp", "potions_used", "died", "decisions")
# Potions that act by themselves and are never drunk: the engine offers USE_POTION only
# for a potion of usage AnyTime or CombatOnly, and a bottled fairy is neither.
AUTOMATIC = frozenset({"POTION.FAIRY_IN_A_BOTTLE"})


def record(row):
    """A row's fight record as `costs.cost` reads it. A fight won that ended the run
    takes the HP at its last decision for the HP after it, whatever else was told."""
    found = dict(row["record"], kind=row["kind"], act=row["act"], floor=row["floor"],
                 second_boss=row["second_boss"], ascension=row["ascension"], run_id=row["id"])
    if not found["died"] and relief(found) == RUN_ENDING:
        found.update(exit_hp=None, exit_max_hp=None)
    return found


def finished(rows):
    """The rows of the fights that were played to their end."""
    return [row for row in rows if row["status"] == "complete" and row.get("record")]


def priced(row, prices=FIXED):
    """A row with its cost by item and in all (`cost`), in HP."""
    items = cost(record(row), prices)
    return dict(row, cost=dict(items._asdict(), total=items.total))


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else None


def mean_error(values):
    """The mean of `values` and its standard error (None for fewer than two)."""
    values = list(values)
    if not values:
        return None, None
    center = sum(values) / len(values)
    if len(values) < 2:
        return center, None
    spread = sum((value - center) ** 2 for value in values) / (len(values) - 1)
    return center, math.sqrt(spread / len(values))


def _round(value, digits=3):
    return None if value is None else round(value, digits)


def by_kind(rows):
    groups = {kind: [row for row in rows if row["kind"] == kind] for kind in KINDS}
    groups[ALL] = list(rows)
    return groups


def held_at_death(rows):
    """Of the deaths in `rows` (finished) where a potion is priced, those not in the run's
    last fight: how many there were, and in how many the belt held a potion that could be
    drunk at the fight's first decision and none was used. Fitting the potions' prices
    waits on this share (ticket 10 of .scratch/training-plan): a policy that dies with its
    potions in the belt gains next to nothing from keeping one."""
    deaths = [row for row in rows if row["record"]["died"] and relief(record(row)) != RUN_ENDING]
    holding = sum(any(potion not in AUTOMATIC for potion in (row.get("behavior") or {}).get("held", ()))
                  and not row["record"]["potions_used"] for row in deaths)
    return dict(deaths=len(deaths), holding=holding, share=_round(holding / len(deaths) if deaths else None))


def tally(rows, prices=FIXED):
    """The account of a kind's fights (`summary`)."""
    done = finished(rows)
    costs = [priced(row, prices)["cost"] for row in done]
    center, error = mean_error(cost["total"] for cost in costs)
    deaths = sum(bool(row["record"]["died"]) for row in done)
    used = [len(row["record"]["potions_used"] or ()) for row in done]
    with_potion = sum(count > 0 for count in used)
    danger = [entry for row in done for entry in (row.get("behavior") or {}).get("danger", ())]
    scored = [entry for entry in danger if "probability" in entry]
    return dict(
        fights=len(rows), finished=len(done), unfinished=len(rows) - len(done), deaths=deaths,
        death_rate=_round(deaths / len(done) if done else None, 4),
        cost=_round(center), cost_error=_round(error),
        items={item: _round(mean(cost[item] for cost in costs)) for item in ("hp", "max_hp", "potions", "death")},
        potions_per_fight=_round(mean(used)), fights_with_potion=with_potion,
        fights_with_potion_share=_round(with_potion / len(done) if done else None),
        held_at_death=held_at_death(done),
        danger=dict(decisions=len(danger), fights=sum(bool((row.get("behavior") or {}).get("danger")) for row in done),
                    used=sum(entry["used"] for entry in danger),
                    use_rate=_round(mean(entry["used"] for entry in danger), 4),
                    probability=_round(mean(entry["probability"] for entry in scored), 4),
                    tempered=_round(mean(entry["tempered"] for entry in scored), 4)))


def summary(rows, prices=FIXED):
    """`tally` by kind of fight and of all of them."""
    return {kind: tally(group, prices) for kind, group in by_kind(rows).items()}


def pairs(first, second):
    """The fights both played to their end, as (first's row, second's row) by id, in
    the first's order."""
    other = {row["id"]: row for row in finished(second)}
    return [(row, other[row["id"]]) for row in finished(first) if row["id"] in other]


def gap(policy, solver, prices=FIXED):
    """The policy's fights against the solver's, fight by fight over those both played
    to their end: by kind, the mean costs, their mean difference (policy − solver)
    with its standard error, the deaths and the potions per fight of each."""
    matched = pairs(policy, solver)
    found = {}
    for kind, group in by_kind([first for first, _ in matched]).items():
        ids = {row["id"] for row in group}
        chosen = [(a, b) for a, b in matched if a["id"] in ids]
        ours = [priced(a, prices)["cost"]["total"] for a, _ in chosen]
        theirs = [priced(b, prices)["cost"]["total"] for _, b in chosen]
        center, error = mean_error(a - b for a, b in zip(ours, theirs))
        found[kind] = dict(
            fights=len(chosen), policy=_round(mean(ours)), solver=_round(mean(theirs)), difference=_round(center),
            difference_error=_round(error),
            policy_deaths=sum(bool(a["record"]["died"]) for a, _ in chosen),
            solver_deaths=sum(bool(b["record"]["died"]) for _, b in chosen),
            policy_potions=_round(mean(len(a["record"]["potions_used"] or ()) for a, _ in chosen)),
            solver_potions=_round(mean(len(b["record"]["potions_used"] or ()) for _, b in chosen)),
            costlier=sum(a > b for a, b in zip(ours, theirs)), cheaper=sum(a < b for a, b in zip(ours, theirs)))
    played = {row["id"] for row in solver}
    found["unpaired"] = dict(
        policy_unfinished=sum(row["status"] != "complete" or not row.get("record") for row in policy),
        solver_unfinished=sum(row["status"] != "complete" or not row.get("record") for row in solver),
        solver_missing=sum(row["id"] not in played for row in policy))
    return found


def setup_content(setup):
    """What a choice that sets a fight up did: how many cards it picked where it may
    pick several (Gambling Chip's discards), the card it picked where one (or none)."""
    if setup["max"] is None or setup["max"] > 1:
        return f"picked {setup['picks']}"
    return setup["picked"][0] if setup["picked"] else "none"


def behavior_counts(rows):
    """How a side played its fights, counted: for each potion the fights that held it
    and those that used it (and its uses), for each setup source and each selection
    trigger the selections by what they did."""
    held, used, uses = Counter(), Counter(), Counter()
    setups, selections = defaultdict(Counter), defaultdict(Counter)
    for row in rows:
        found = row["behavior"]
        held.update(set(found["held"]))
        used.update(set(found["used"]))
        uses.update(found["used"])
        for setup in found["setups"]:
            setups[setup["relic"] or f"other {setup['source']}/{setup['operation']} {setup['min']}-{setup['max']}"][
                setup_content(setup)] += 1
        for selection in found["selections"]:
            key = f"{selection['trigger']} {selection['operation']} {selection['min']}-{selection['max']}"
            selections[key][f"picked {selection['picks']}"] += 1
    return dict(held=held, used=used, uses=uses, setups=setups, selections=selections)


def _share(count, total):
    return dict(count=count, of=total, share=_round(count / total if total else None, 4))


def behavior(policy, solver):
    """The table of how the policy and the solver played the fights both played to
    their end (`pairs`): a row for each kind of action (`potion`, `setup`,
    `selection`), its key (the potion; the relic that offers the setup, or its shape;
    what opened the selection, its operation and bounds) and what it did (`content`),
    with each side's count, out of what (`of`), and the share. A potion's row counts
    the fights that used it out of those that held it at their start; a setup's or a
    selection's counts the selections that did this out of that side's selections of
    the key, which may differ between the sides when their plays differ."""
    matched = pairs(policy, solver)
    ours, theirs = behavior_counts([a for a, _ in matched]), behavior_counts([b for _, b in matched])
    table = []
    for potion in sorted(set(ours["held"]) | set(theirs["held"])):
        table.append(dict(kind="potion", key=potion, content="used",
                          policy=dict(_share(ours["used"][potion], ours["held"][potion]), uses=ours["uses"][potion]),
                          solver=dict(_share(theirs["used"][potion], theirs["held"][potion]),
                                      uses=theirs["uses"][potion])))
    for kind, name in (("setup", "setups"), ("selection", "selections")):
        for key in sorted(set(ours[name]) | set(theirs[name])):
            mine, other = ours[name][key], theirs[name][key]
            for content in sorted(set(mine) | set(other), key=lambda value: (-(mine[value] + other[value]), value)):
                table.append(dict(kind=kind, key=key, content=content,
                                  policy=_share(mine[content], sum(mine.values())),
                                  solver=_share(other[content], sum(other.values()))))
    return dict(fights=len(matched), rows=table)


def paired_plays(first, second, prices=FIXED):
    """Two plays of the same fights, fight by fight over those both played to their
    end: by kind, the fights played with other actions (`actions`), the fights whose
    record differs (`OUTCOME`), and the mean difference of the costs (second − first)
    with its standard error; and the ids of the fights that differ."""
    matched = pairs(first, second)
    found = {}
    for kind, group in by_kind([a for a, _ in matched]).items():
        ids = {row["id"] for row in group}
        chosen = [(a, b) for a, b in matched if a["id"] in ids]
        center, error = mean_error(priced(b, prices)["cost"]["total"] - priced(a, prices)["cost"]["total"]
                                   for a, b in chosen)
        found[kind] = dict(
            fights=len(chosen), other_actions=sum(a["actions"] != b["actions"] for a, b in chosen),
            other_records=sum(any(a["record"][key] != b["record"][key] for key in OUTCOME) for a, b in chosen),
            other_deaths=sum(bool(a["record"]["died"]) != bool(b["record"]["died"]) for a, b in chosen),
            difference=_round(center), difference_error=_round(error))
    found["differing"] = [a["id"] for a, b in matched if a["actions"] != b["actions"]]
    found["unpaired"] = dict(first=len(first) - len(matched), second=len(second) - len(matched))
    return found


def report(policy, solver=None, prices=FIXED):
    """The report of the policy's fights of a set and, with the solver's, the gap and
    the table of behavior."""
    found = dict(prices=dict(tiers=list(prices.tiers), final=list(prices.final), free=sorted(prices.free))
                 if hasattr(prices, "tiers") else None, summary=summary(policy, prices))
    if solver:
        found.update(solver=summary(solver, prices), gap=gap(policy, solver, prices), behavior=behavior(policy, solver))
    return found


def _cells(values):
    return "| " + " | ".join("" if value is None else str(value) for value in values) + " |"


def text(found):
    """The tables of a report (`report`, with `paired` if there is one) as text."""
    lines = ["## policy", "",
             _cells(("kind", "fights", "deaths", "death rate", "cost", "±", "hp", "max hp", "potions", "death",
                     "potions/fight", "with potion", "died holding", "danger", "danger used", "P(use) T=1",
                     "P(use) T")),
             _cells(("---",) * 17)]
    for kind, entry in found["summary"].items():
        danger, held = entry["danger"], entry["held_at_death"]
        lines.append(_cells((kind, entry["finished"], entry["deaths"], entry["death_rate"], entry["cost"],
                             entry["cost_error"], *entry["items"].values(), entry["potions_per_fight"],
                             entry["fights_with_potion_share"], f"{held['holding']}/{held['deaths']}",
                             danger["decisions"], danger["used"], danger["probability"], danger["tempered"])))
    if "gap" in found:
        lines += ["", "## solver", "",
                  _cells(("kind", "fights", "deaths", "death rate", "cost", "±", "potions/fight", "danger",
                          "danger used")), _cells(("---",) * 9)]
        for kind, entry in found["solver"].items():
            lines.append(_cells((kind, entry["finished"], entry["deaths"], entry["death_rate"], entry["cost"],
                                 entry["cost_error"], entry["potions_per_fight"], entry["danger"]["decisions"],
                                 entry["danger"]["used"])))
        lines += ["", "## policy − solver", "",
                  _cells(("kind", "fights", "policy", "solver", "difference", "±", "policy deaths", "solver deaths",
                          "policy potions", "solver potions")), _cells(("---",) * 10)]
        for kind in (*KINDS, ALL):
            entry = found["gap"][kind]
            lines.append(_cells((kind, entry["fights"], entry["policy"], entry["solver"], entry["difference"],
                                 entry["difference_error"], entry["policy_deaths"], entry["solver_deaths"],
                                 entry["policy_potions"], entry["solver_potions"])))
        lines += ["", f"## behavior ({found['behavior']['fights']} fights)", "",
                  _cells(("kind", "key", "content", "policy", "of", "share", "solver", "of", "share")),
                  _cells(("---",) * 9)]
        for row in found["behavior"]["rows"]:
            lines.append(_cells((row["kind"], row["key"], row["content"], row["policy"]["count"], row["policy"]["of"],
                                 row["policy"]["share"], row["solver"]["count"], row["solver"]["of"],
                                 row["solver"]["share"])))
    if "paired" in found:
        lines += ["", "## paired plays", "",
                  _cells(("kind", "fights", "other actions", "other records", "other deaths", "difference", "±")),
                  _cells(("---",) * 7)]
        for kind in (*KINDS, ALL):
            entry = found["paired"][kind]
            lines.append(_cells((kind, entry["fights"], entry["other_actions"], entry["other_records"],
                                 entry["other_deaths"], entry["difference"], entry["difference_error"])))
    return "\n".join(lines) + "\n"
