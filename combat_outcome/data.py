"""Combat-outcome labels: fights at the HP they were entered with.

One row per completed fight (`combat-outcomes-v2`): who fought it, the entry
state's input entities, the encounter, how the fight ended, and the public
decision frames of the fight. Rows come from summary-anchor battles fought by the
solver (`spire_codex_data.anchor export`; the row points at the frames, which
stay in the anchor's sample file) and from fights a policy played in rollouts
(`online.CombatRecorder`; frames inline). A solver's fights say what the solver
can do, a policy's what that policy can do: the actor is part of every row.
Entities stay compressed in memory and are unpacked a batch at a time, so a full
export fits.

python -m combat_outcome.data runs/ppo      # coverage of the labels under a directory
"""

import argparse
import gzip
import json
import random
import zlib
from collections import Counter
from pathlib import Path

SCHEMA = "combat-outcomes-v2"
PATTERNS = ("*combats.jsonl", "*combats.jsonl.gz", "combat-outcomes.jsonl.gz")
# Who fights in each source. A solver sees hidden state; its fights are never a policy's.
ACTORS = {"summary_anchor": "combat_solver", "rollout": "policy"}


def entities(row):
    return json.loads(zlib.decompress(row["packed"]))


def read(path):
    with (gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open()) as stream:
        for index, line in enumerate(stream, 1):
            if line.strip():
                yield index, json.loads(line)


def files(source):
    source = Path(source)
    return [source] if source.is_file() else sorted({p for pattern in PATTERNS for p in source.rglob(pattern)})


def labels(source):
    """(path, line, row) of every completed fight under `source`, checked."""
    for path in files(source):
        for index, f in read(path):
            if f.get("schema") != SCHEMA:
                raise ValueError(f"Unsupported combat schema in {path}:{index}")
            if f.get("actor") != ACTORS.get(f.get("source")):
                raise ValueError(f"Combat label without a declared source and actor in {path}:{index}")
            if f.get("result") not in {"win", "loss"} or "error" in f:
                continue
            if f["max_hp"] <= 0 or f["hp_lost"] != f["start_hp"] - f["end_hp"]:
                raise ValueError(f"Inconsistent combat label in {path}:{index}")
            yield path, index, f


def policy_key(f):
    """The policy a fight measures: its weights, or the solver and its budget."""
    return (f.get("policy") or {}).get("digest") if f["actor"] == "policy" else "combat_solver"


def load_data(source):
    """Rows of every label file under `source` (or of that one file).

    `y` is the HP lost; a lost fight costs all the HP it was entered with.
    `run` is the split key: the seed group, so runs of one seed stay together.
    """
    rows = []
    for _, _, f in labels(source):
        rows.append(dict(
            key=len(rows), run=str(f["seed"]), encounter=f["encounter"], kind=f["kind"],
            character=f["character"], ascension=f.get("ascension"), act=f["act"], max_hp=f["max_hp"],
            start_hp=f["start_hp"],
            y=float(f["hp_lost"]), lost=f["result"] == "loss", source=f["source"], actor=f["actor"],
            policy=policy_key(f), act_result=f.get("act_result"),
            packed=zlib.compress(json.dumps(f["entities"], separators=(",", ":")).encode(), 1)))
    if not rows:
        raise ValueError(f"No completed combat labels in {source}")
    return rows


def fights(source):
    """(label row, frames) of every fight under `source`, frames in the order they were decided.

    A frame is the public observation of one decision of the fight and the action
    taken there; the first is the entry state. What followed a frame is in the
    label row only. All frames of a fight, and all fights of a seed, belong to one
    side of a split: split on the row's `seed`.
    """
    from model.representation import clean_public

    cache = (None, {})
    for path, index, f in labels(source):
        frames = f.get("frames")
        if isinstance(frames, dict):
            # Solver fights: the anchor's sample file holds one row per decision.
            samples = path.parent / frames["file"]
            if cache[0] != samples:
                groups = {}
                for _, item in read(samples):
                    groups.setdefault(item["metadata"]["sample_group"], []).append(item)
                cache = (samples, groups)
            items = cache[1].get(frames["group"], [])
            frames = [dict(step=i, forced=len(item["options"]) == 1, public=clean_public(item["observation"]),
                           action=item["label"]) for i, item in enumerate(items)]
            if len(frames) != f["frames"]["count"]:
                raise ValueError(f"Fight frames are missing for {path}:{index}")
        if not frames:
            raise ValueError(f"Fight without decision frames in {path}:{index}")
        yield {k: v for k, v in f.items() if k != "frames"}, frames


def coverage(source):
    """How many fights there are of each source, policy, character, ascension, act, kind and result."""
    counts, acts = Counter(), {}
    for _, _, f in labels(source):
        policy = (f.get("policy") or {})
        key = dict(source=f["source"], actor=f["actor"], policy=policy_key(f), policy_version=policy.get("version"),
                   character=f["character"], ascension=f.get("ascension"), act=f["act"], kind=f["kind"],
                   result=f["result"])
        counts[json.dumps(key, sort_keys=True)] += 1
        if f["actor"] == "policy":
            # One act of one run, however many fights it held.
            acts[f["run_id"], f["act"]] = (policy_key(f), f["character"], f["act"], f.get("act_result"))
    passed = Counter(json.dumps(dict(policy=p, character=c, act=a, act_result=r), sort_keys=True)
                     for p, c, a, r in acts.values())
    return dict(schema=SCHEMA, fights=[dict(json.loads(k), fights=n) for k, n in sorted(counts.items())],
                acts=[dict(json.loads(k), runs=n) for k, n in sorted(passed.items())])


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Coverage of the combat-outcome labels under a file or directory")
    parser.add_argument("source", type=Path)
    print(json.dumps(coverage(parser.parse_args().source), ensure_ascii=False, indent=1))
