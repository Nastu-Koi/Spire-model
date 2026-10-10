"""Evaluation reports: the win rates of played runs, two evaluations of the same seeds
set against each other, and the tables of the held-out NLL."""

import json
import math
from collections import defaultdict
from pathlib import Path

from .protocol import CHARACTERS
from .rollout import climbed, stand


def wilson_interval(wins, n, z=1.96):
    """The Wilson score interval of a proportion; None without attempts."""
    if not n:
        return None
    rate = wins / n
    center = (rate + z * z / (2 * n)) / (1 + z * z / n)
    radius = z * math.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [max(0.0, center - radius), min(1.0, center + radius)]


def climb(run):
    """Where a run stood: the furthest floor of each act it entered, and how many
    floors those acts have. A run records it over every decision it played (`climb`,
    with the acts as JSON keys); one from before that is read from the frames it
    kept. Both are empty when the frames do not say."""
    if "climb" in run:
        stored = run["climb"]
        if isinstance(stored, dict):
            return ({int(act): floor for act, floor in stored["reached"].items()},
                    {int(act): length for act, length in stored["lengths"].items()})
        return stored
    frames = [step["frame"] for macro in run.get("macros", ()) for step in macro.get("steps", ())]
    frames += [action["frame"] for action in run.get("environment_actions", ())]
    return climbed(map(stand, frames))


class RunTally:
    """Win rates over every attempt: a run that did not finish is not a win.

    The rate among finished runs is reported beside it, and `comparable` says
    whether every attempt finished; only then do the two agree and can reports
    be set against each other.

    Beside the wins stands how far the finished runs got: the acts they passed,
    the reward they were paid, and the floor each defeat stopped at, counted across
    acts. A win is far rarer than a floor climbed, so these move before it does.

    Runs are added one at a time and nothing of a run is kept but what it adds to
    the report: an evaluation is more runs than memory holds, so they are passed
    from where they were written, and one pass can feed several tallies.
    """

    def __init__(self):
        self.tallies = {
            character: {"attempts": 0, "complete": 0, "wins": 0, "passed": [0, 0, 0],
                        "scores": [], "floors": [], "deaths": {}}
            for character in CHARACTERS
        }
        self.act_floors, self.ascensions = {}, set()
        self.attempts = self.unfinished = 0

    def add(self, run):
        self.attempts += 1
        self.unfinished += run["status"] != "complete"
        if "ascension" in run:
            self.ascensions.add(run["ascension"])
        tally = self.tallies.get(run["character"])
        if tally is None:
            return
        tally["attempts"] += 1
        if run["status"] != "complete":
            return
        tally["complete"] += 1
        tally["wins"] += run.get("victory", False)
        ledger = run.get("ledger", {})
        # The last act is passed by winning the run, whichever event confirmed it.
        for act in set(ledger.get("bosses", ())) | ({3} if run.get("victory") else set()):
            tally["passed"][act - 1] += 1
        if "components" in ledger:
            tally["scores"].append(sum(ledger["components"].values()))
        reached, lengths = climb(run)
        for act, length in lengths.items():
            self.act_floors[act] = max(self.act_floors.get(act, 0), length)
        if reached:
            floor = sum(reached.values())
            tally["floors"].append(floor)
            if not run.get("victory"):
                tally["deaths"][floor] = tally["deaths"].get(floor, 0) + 1

    def report(self):
        report = {}
        for character, tally in self.tallies.items():
            total, n, wins = tally["attempts"], tally["complete"], tally["wins"]
            scores, floors, deaths = tally["scores"], tally["floors"], tally["deaths"]
            report[character] = {
                "attempts": total,
                "complete": n,
                "wins": wins,
                "win_rate": wins / total if total else None,
                "wilson_95": wilson_interval(wins, total),
                "completed_win_rate": wins / n if n else None,
                "completion_rate": n / total if total else None,
                "errors_or_unresolved": total - n,
                "acts_passed": tally["passed"],
                "act_pass_rates": [count / total for count in tally["passed"]] if total else None,
                "act_pass_wilson_95": [wilson_interval(count, total) for count in tally["passed"]],
                "mean_score": sum(scores) / len(scores) if scores else None,
                "mean_floor": sum(floors) / len(floors) if floors else None,
                # [floor, runs] for the finished runs that were lost, floors counted across acts.
                "death_floors": [[floor, deaths[floor]] for floor in sorted(deaths)],
            }

        def mean(key):
            values = [x[key] for x in report.values()]
            return sum(values) / 5 if all(v is not None for v in values) else None

        rates = [x["win_rate"] for x in report.values()]
        passes = [x["act_pass_rates"] for x in report.values()]
        attempts = sum(x["attempts"] for x in report.values())
        wins = sum(x["wins"] for x in report.values())
        passed = [sum(acts) for acts in zip(*(x["acts_passed"] for x in report.values()))]
        return {
            "characters": report,
            "ascensions": sorted(self.ascensions),
            "comparable": bool(self.attempts) and not self.unfinished,
            "equal_character_win_rate": mean("win_rate"),
            "worst_character_win_rate": min(rates) if all(r is not None for r in rates) else None,
            "equal_character_completed_win_rate": mean("completed_win_rate"),
            "equal_character_completion_rate": mean("completion_rate"),
            "equal_character_act_pass_rates": (
                [sum(act) / 5 for act in zip(*passes)] if all(p is not None for p in passes) else None
            ),
            "equal_character_mean_score": mean("mean_score"),
            "equal_character_mean_floor": mean("mean_floor"),
            # All attempts of the five characters together, with the uncertainty of their
            # rates. With equal attempts per character, as a round assigns them, the pooled
            # rates are the equal-character rates.
            "pooled": {
                "attempts": attempts,
                "wins": wins,
                "win_rate": wins / attempts if attempts else None,
                "wilson_95": wilson_interval(wins, attempts),
                "acts_passed": passed,
                "act_pass_rates": [count / attempts for count in passed] if attempts else None,
                "act_pass_wilson_95": [wilson_interval(count, attempts) for count in passed],
            },
            # Floors of acts 1, 2, ... as far as any finished run entered them.
            "act_floors": [self.act_floors.get(act) for act in range(1, max(self.act_floors, default=0) + 1)],
        }


def evaluate_runs(runs):
    """The report of `RunTally` over `runs`, which are read once, one at a time."""
    tally = RunTally()
    for run in runs:
        tally.add(run)
    return tally.report()


def outcome(run):
    """What a run came to, for setting it against the same seed of another evaluation:
    whether it finished, won and passed each act. A run that did not finish won and
    passed nothing, as the win rate over attempts counts it."""
    complete = run.get("status") == "complete"
    victory = bool(complete and run.get("victory"))
    bosses = set(run.get("ledger", {}).get("bosses", ())) if complete else set()
    return {"complete": complete, "win": victory,
            "acts_passed": [1 in bosses, 2 in bosses, 3 in bosses or victory]}


def sign_test(first, second):
    """The two-sided exact sign test of paired yes/no outcomes: `first` and `second`
    are the outcomes of the same seeds in two evaluations, in the same order. Only the
    pairs that disagree count; the p-value is the chance of a split at least this
    uneven when either side is equally likely to win a disagreement."""
    only_first = sum(bool(a) and not b for a, b in zip(first, second))
    only_second = sum(bool(b) and not a for a, b in zip(first, second))
    n = only_first + only_second
    if n:
        tail = sum(math.comb(n, k) for k in range(min(only_first, only_second) + 1))
        p_value = min(1.0, 2 * tail / 2**n)
    else:
        p_value = 1.0
    return {"pairs": len(first), "first": sum(map(bool, first)), "second": sum(map(bool, second)),
            "only_first": only_first, "only_second": only_second, "p_value": p_value}


PAIRED_OUTCOMES = {
    "win": lambda outcome: outcome["win"],
    "act_1_passed": lambda outcome: outcome["acts_passed"][0],
    "act_2_passed": lambda outcome: outcome["acts_passed"][1],
}


def paired_comparison(first, second):
    """Two evaluations set against each other seed by seed. `first` and `second` map
    (character, seed) to `outcome`s; over the seeds both played, the sign test of
    winning and of passing the first two acts. Seeds only one of them played are
    counted and left out."""
    keys = sorted(set(first) & set(second))
    report = {"pairs": len(keys),
              "unpaired": {"first": len(first) - len(keys), "second": len(second) - len(keys)}}
    for name, pick in PAIRED_OUTCOMES.items():
        report[name] = sign_test([pick(first[key]) for key in keys], [pick(second[key]) for key in keys])
    return report


def evaluation_runs(directory):
    """The runs of an evaluation directory as `collect_round` wrote it, one at a time in
    the order of its seed assignment: (character, index, seed, run)."""
    directory = Path(directory)
    seeds = json.loads((directory / "metadata" / "seeds.json").read_text())
    for character in CHARACTERS:
        for index, seed in enumerate(seeds.get(character, [])):
            path = directory / f"{character}-{index}.json"
            if not path.is_file():
                raise FileNotFoundError(f"Evaluation is missing its run {path}")
            yield character, index, str(seed), json.loads(path.read_text())


def summarize_evaluation(directory, head=60):
    """The report of an evaluation and of its first `head` seeds of every character, in
    one pass over its runs; with the outcome of every run keyed by (character, seed),
    for pairing."""
    whole, first = RunTally(), RunTally()
    outcomes = {}
    for character, index, seed, run in evaluation_runs(directory):
        whole.add(run)
        if index < head:
            first.add(run)
        outcomes[character, seed] = dict(outcome(run), index=index)
    summary = {"directory": str(directory), "runs": len(outcomes), "report": whole.report(),
               "head": {"seeds_per_character": head, "report": first.report()}}
    return summary, outcomes


def compare_evaluations(directories, head=60):
    """The reports of one or two evaluations, whole and over the first `head` seeds of
    every character; two are also paired seed by seed, whole and over the head."""
    if not 1 <= len(directories) <= 2:
        raise ValueError("Compare one evaluation, or two played on the same seeds")
    summaries, outcomes = [], []
    for directory in directories:
        summary, played = summarize_evaluation(directory, head)
        summaries.append(summary)
        outcomes.append(played)
    report = {"evaluations": summaries}
    if len(directories) == 2:
        first, second = outcomes
        report["paired"] = paired_comparison(first, second)
        report["paired"]["head"] = dict(
            seeds_per_character=head,
            **paired_comparison({key: value for key, value in first.items() if value["index"] < head},
                                {key: value for key, value in second.items() if value["index"] < head}))
    return report


def _nll_tables(groups):
    def total(entries):
        nll = sum(entry["nll"] for entry in entries)
        macros = sum(entry["macros"] for entry in entries)
        branches = sum(entry["branches"] for entry in entries)
        return {"nll": nll, "macros": macros, "branches": branches,
                "nll_per_macro": nll / macros if macros else None,
                "nll_per_branch": nll / branches if branches else None}

    by_character, by_phase = defaultdict(list), defaultdict(list)
    for name, entry in groups.items():
        character, phase = name.split("/", 1)
        by_character[character].append(entry)
        by_phase[phase].append(entry)
    per_macro = [entry["nll"] / entry["macros"] for entry in groups.values() if entry["macros"]]
    per_branch = [entry["nll"] / entry["branches"] for entry in groups.values() if entry["branches"]]
    return {
        "groups": {name: total([entry]) for name, entry in sorted(groups.items())},
        "characters": {name: total(entries) for name, entries in sorted(by_character.items())},
        "phases": {name: total(entries) for name, entries in sorted(by_phase.items())},
        "all": total(list(groups.values())),
        "equal_group_nll_per_macro": sum(per_macro) / len(per_macro) if per_macro else None,
        "equal_group_nll_per_branch": sum(per_branch) / len(per_branch) if per_branch else None,
    }


def nll_report(groups):
    """The tables of what `Learner.evaluate_bootstrap` returns: every character/phase
    group, then the characters, the phases and the whole, each as the NLL per
    macro-action and per decision (`nll_per_branch`). `equal_group_nll_per_macro` is
    the mean of the groups' per-macro NLL: the training loss weighs every group the
    same, so this is the number it is set against. Groups named `A0/...` are the
    ascension-0 subset and form tables of their own under `a0`."""
    report = _nll_tables({name: entry for name, entry in groups.items() if not name.startswith("A0/")})
    a0 = {name[3:]: entry for name, entry in groups.items() if name.startswith("A0/")}
    if a0:
        report["a0"] = _nll_tables(a0)
    return report
