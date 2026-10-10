"""Advantages of the decisions a policy made outside fights, kept beside its rounds.

A sampled run pays once, when it ends: +1 for a victory, -1 otherwise. Its decisions
outside fights, in order, are s_0 .. s_{T-1}; a value estimate V(s) of that ±1 return
(the critic's `2 * p_win - 1`) gives each decision a temporal difference

    δ_t = V(s_{t+1}) - V(s_t)      for t < T-1
    δ_t = R - V(s_t)               for the last decision, the run's return R = ±1

and a generalized advantage A_t = Σ_k λ^k δ_{t+k}, without discount. The next state
s_{t+1} is the run's next decision outside a fight: a fight between two of them holds
no such decision, so the difference spans it and counts what the fight cost. Nothing
follows the last decision but the outcome, whose value is the return itself.

Advantages are written once per round, into the round's directory beside its shards
(`advantages.json.gz`), with what they were computed from: the critic's version and
weights, or the control that uses no critic (the return minus the mean return of the
character's runs). The shards are not touched. A decision is named by its run and its
place among the run's macro-actions, as `data.Sample` names it.
"""

import gzip
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

from .control import policy_macros
from .data import RunShards
from .dataset_index import read_index

FILE = "advantages.json.gz"
FORMAT = "rollout-advantages-v1"
CRITIC, RETURN = "critic", "return"
LAMBDA = 0.9


def gae(values, outcome, lam=LAMBDA):
    """The generalized advantages of one run's decisions outside fights, in order, from
    their values and the run's return (+1 or -1). Undiscounted; the value of what
    follows the last decision is the return."""
    if not 0 <= lam <= 1:
        raise ValueError("The advantage's λ is between 0 and 1")
    advantages, following = [0.0] * len(values), 0.0
    for t in reversed(range(len(values))):
        after = outcome if t == len(values) - 1 else values[t + 1]
        following = after - values[t] + lam * following
        advantages[t] = following
    return advantages


def weights_digest(path):
    """What tells one critic's weights from another's with the same version number."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()[:16]


def summary(values):
    """Count, mean, deviation and range of a list of advantages."""
    if not values:
        return dict(n=0)
    mean = sum(values) / len(values)
    return dict(n=len(values), mean=round(mean, 5),
                std=round(math.sqrt(sum((v - mean) ** 2 for v in values) / len(values)), 5),
                min=round(min(values), 5), max=round(max(values), 5))


def write(directory, identity, runs):
    """Write a round's advantages. `identity` says what they were computed from;
    `runs` maps a run id to its decisions in order, each (macro index, value,
    advantage)."""
    path = Path(directory) / FILE
    decisions = [entry for entries in runs.values() for entry in entries]
    document = dict(format=FORMAT, **identity, runs=len(runs), decisions=len(decisions),
                    advantage=summary([entry[2] for entry in decisions]),
                    advantages={run: [[index, round(value, 6), round(advantage, 6)]
                                      for index, value, advantage in entries] for run, entries in runs.items()})
    temporary = path.with_name(path.name + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as stream:
        json.dump(document, stream, allow_nan=False, separators=(",", ":"))
    temporary.replace(path)
    return {key: value for key, value in document.items() if key != "advantages"}


def header(directory):
    """What a round's advantage file says of itself, or None without one."""
    path = Path(directory) / FILE
    if not path.is_file():
        return None
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        document = json.load(stream)
    if document.get("format") != FORMAT:
        raise ValueError(f"Unsupported advantage file {path}")
    return {key: value for key, value in document.items() if key != "advantages"}


def current(directory, identity):
    """Whether a round already holds the advantages `identity` describes."""
    found = header(directory)
    return found is not None and all(found.get(key) == value for key, value in identity.items())


def read(directories):
    """{(run id, macro index): advantage} of the rounds' advantage files."""
    table = {}
    for directory in directories:
        path = Path(directory) / FILE
        if not path.is_file():
            raise FileNotFoundError(f"Round {directory} has no advantages; compute them first")
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            document = json.load(stream)
        if document.get("format") != FORMAT:
            raise ValueError(f"Unsupported advantage file {path}")
        for run, entries in document["advantages"].items():
            for index, _, advantage in entries:
                table[run, index] = advantage
    return table


def from_values(decisions, values, lam=LAMBDA):
    """{run id: [(macro index, value, advantage)]} from the decisions of whole runs.
    `decisions` are (run id, macro index, won) in the order each run made them, the
    decisions of a run together; `values` their value estimates."""
    runs, order = defaultdict(list), {}
    for (run, index, won), value in zip(decisions, values, strict=True):
        if runs[run] and runs[run][-1][0] >= index:
            raise ValueError("A run's decisions must be given in the order they were made")
        if order.setdefault(run, won) != won:
            raise ValueError("A run has one outcome")
        runs[run].append((index, float(value)))
    return {run: [(index, value, advantage) for (index, value), advantage in zip(
        entries, gae([value for _, value in entries], 1.0 if order[run] else -1.0, lam))]
        for run, entries in runs.items()}


def decision_places(directory):
    """(run id, character, won, the macro index of each of its samples) of every run
    of a round: the decisions `data.samples` makes samples of, a shard at a time."""
    for window in RunShards(directory).windows(shards=1):
        for run in window:
            place = {id(macro): index for index, macro in enumerate(run["macros"])}
            yield run["run_id"], run["character"], bool(run["victory"]), [
                place[id(macro)] for macro in policy_macros(run)]


def mean_returns(directories):
    """The mean ±1 return of every character's runs over the rounds, from their
    indexes: the baseline of the control that uses no critic."""
    totals = defaultdict(lambda: [0.0, 0])
    for directory in directories:
        for row in read_index(directory):
            entry = totals[row["character"]]
            entry[0] += 1.0 if row["victory"] else -1.0
            entry[1] += 1
    return {character: total / count for character, (total, count) in sorted(totals.items())}


def write_returns(directories):
    """The control without a critic: every decision's advantage is its run's return
    minus the mean return of its character's runs over these rounds. Returns each
    round's header."""
    baseline = mean_returns(directories)
    identity = dict(kind=RETURN, critic=None, baseline={c: round(v, 6) for c, v in baseline.items()})
    headers = []
    for directory in directories:
        if current(directory, identity):
            headers.append(header(directory))
            continue
        runs = {}
        for run, character, won, places in decision_places(directory):
            advantage = (1.0 if won else -1.0) - baseline[character]
            runs[run] = [(index, baseline[character], advantage) for index in places]
        headers.append(write(directory, identity, runs))
    return headers
