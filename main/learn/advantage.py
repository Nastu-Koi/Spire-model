"""The advantage file of a round of act groups, kept beside each wave's shards.

A wave's advantages (`main/learn/groups.py`) are written once, into the wave's
directory beside its shards (`advantages.json.gz`), with what they were computed from.
The shards are not touched. A decision is named by its run and its place among the
run's macro-actions, as `data.Sample` names it; each entry keeps the decision's value
(its baseline) and its advantage.
"""

import gzip
import json
import math
from pathlib import Path

FILE = "advantages.json.gz"
FORMAT = "rollout-advantages-v1"


def summary(values):
    """Count, mean, deviation and range of a list of advantages."""
    if not values:
        return dict(n=0)
    mean = sum(values) / len(values)
    return dict(n=len(values), mean=round(mean, 5),
                std=round(math.sqrt(sum((v - mean) ** 2 for v in values) / len(values)), 5),
                min=round(min(values), 5), max=round(max(values), 5))


def write(directory, identity, runs):
    """Write a wave's advantages. `identity` says what they were computed from;
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
