"""Combat-outcome labels: solver fights at the HP they were entered with.

One row per completed fight (`combat-outcomes-v1`): the entry state's input
entities, the encounter and how the fight ended. Rows come from summary-anchor
battles (`spire_codex_data.anchor export`) and from fights recorded during
rollouts (`online.CombatRecorder`). Entities stay compressed in memory and are
unpacked a batch at a time, so a full export fits.
"""

import gzip
import json
import random
import zlib
from pathlib import Path

SCHEMA = "combat-outcomes-v1"
FILES = ("combats.jsonl", "combats.jsonl.gz", "combat-outcomes.jsonl.gz")


def entities(row):
    return json.loads(zlib.decompress(row["packed"]))


def read(path):
    with (gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open()) as stream:
        for index, line in enumerate(stream, 1):
            if line.strip():
                yield index, json.loads(line)


def load_data(source):
    """Rows of every label file under `source` (or of that one file).

    `y` is the HP lost; a lost fight costs all the HP it was entered with.
    `run` is the split key: the seed group, so runs of one seed stay together.
    """
    source = Path(source)
    files = [source] if source.is_file() else sorted(p for name in FILES for p in source.rglob(name))
    rows = []
    for path in files:
        for index, f in read(path):
            if f.get("schema") != SCHEMA:
                raise ValueError(f"Unsupported combat schema in {path}:{index}")
            if f.get("result") not in {"win", "loss"} or "error" in f:
                continue
            if f["max_hp"] <= 0 or f["hp_lost"] != f["start_hp"] - f["end_hp"]:
                raise ValueError(f"Inconsistent combat label in {path}:{index}")
            rows.append(dict(
                key=len(rows), run=str(f["seed"]), encounter=f["encounter"], kind=f["kind"],
                character=f["character"], ascension=f.get("ascension"), act=f["act"], max_hp=f["max_hp"],
                start_hp=f["start_hp"],
                y=float(f["hp_lost"]), lost=f["result"] == "loss", source=f.get("source"),
                packed=zlib.compress(json.dumps(f["entities"], separators=(",", ":")).encode(), 1)))
    if not rows:
        raise ValueError(f"No completed combat labels in {source}")
    return rows


def split_runs(rows, seed, holdout=0.2, folds=None, fold=0):
    runs = sorted({r["run"] for r in rows})
    if len(runs) < 2:
        raise ValueError("At least two independent seeds/runs are required for a leakage-free train/test split")
    if not 0 < holdout < 1 or (folds is not None and not 2 <= folds <= len(runs)):
        raise ValueError("Invalid holdout or fold count")
    if folds is not None and not 0 <= fold < folds:
        raise ValueError("Invalid fold index")
    random.Random(seed).shuffle(runs)
    test_runs = set(runs[fold::folds] if folds else runs[:max(1, int(len(runs) * holdout))])
    return ([r for r in rows if r["run"] not in test_runs],
            [r for r in rows if r["run"] in test_runs], runs, test_runs)
