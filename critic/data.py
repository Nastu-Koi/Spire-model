"""Out-of-combat decisions labelled with how their run and their act ended.

A row is one decision of the policy outside combat: the frame it decided on, whether
the run was won (`win`), whether the act the frame stands in was passed
(`act_passed`: that act's boss fell, or the run was won) and where it stood. The frame
is the first step with a choice of a macro-action whose phase is not `combat`; the
forced steps before it and the controller's steps are not decisions. Rows come from
complete runs as the run loop writes them (`spire-trajectory-v1`: one JSON file per
run, or one run per line of a `.jsonl`/`.jsonl.gz`); the rollout buffer of issue 15
is read here too once it exists, with nothing else to change. Frames stay compressed
in memory and are parsed a batch at a time where the loader runs.

The seed is the split key: every decision of a seed is on one side.

python -m critic.data runs/eval-a0/dev-v8      # coverage of the decisions under a directory
"""

import argparse
import gzip
import json
import statistics as st
import zlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from model.control import is_policy_frame
from model.data import held_out, validate_run
from model.protocol import SCHEMA

DATA_VERSION = "critic-decisions-v1"
STATE_COUNTS = {"deck": lambda e: e.get("entity_type") == "card" and e.get("zone") == "deck",
                "relics": lambda e: e.get("entity_type") == "relic",
                "potions": lambda e: e.get("entity_type") == "potion"}


@dataclass(slots=True)
class Row:
    packed: bytes  # zlib of the frame's public observation and legal candidates
    win: bool
    act_passed: bool
    seed: str
    run_id: str
    decision_id: str
    character: str
    act: int
    floor: int
    phase: str
    policy_version: int | None
    policy_digest: str | None
    state: dict  # hp, max_hp, gold and entity counts: the tabular baselines' features
    _lengths: tuple | None = None  # (tokens, action slots) once parsed; the packer's cache


def frame_of(row):
    return json.loads(zlib.decompress(row.packed))


def _pack(frame):
    return zlib.compress(json.dumps({"public": frame["public"], "legal": frame["legal"]},
                                    separators=(",", ":")).encode(), 1)


def where(frame):
    """The player's act, floor and the state summary of a frame."""
    entities = frame["public"]["entities"]
    player = next((e for e in entities if e.get("entity_type") == "player"), None)
    if player is None or type(player.get("act")) is not int or type(player.get("floor")) is not int:
        raise ValueError("A decision frame needs a player entity with an act and a floor")
    state = {key: player.get(key) for key in ("hp", "max_hp", "gold")}
    state.update({name: sum(1 for e in entities if test(e)) for name, test in STATE_COUNTS.items()})
    return player["act"], player["floor"], state


def decisions(run):
    """Rows of every out-of-combat decision of a complete, validated run."""
    validate_run(run)
    victory = bool(run["victory"])
    bosses = set(run.get("ledger", {}).get("bosses", ()))
    for macro in run["macros"]:
        if macro["phase"] == "combat":
            continue
        step = next((s for s in macro.get("steps", ()) if "frame" in s and is_policy_frame(s["frame"])), None)
        if step is None:
            if "packed" in macro:
                raise ValueError("Packed macro-actions carry no frames; the critic needs the decision frames")
            continue
        frame = step["frame"]
        act, floor, state = where(frame)
        yield Row(_pack(frame), victory, victory or act in bosses, str(run.get("seed", run["run_id"])),
                  run["run_id"], frame["routing"]["decision_id"], run["character"], act, floor,
                  macro["phase"], run.get("policy_version"), run.get("policy_digest"), state)


def _read(path):
    """The JSON objects a file holds: one, or one per line."""
    if path.name.endswith((".jsonl", ".jsonl.gz")):
        with (gzip.open(path, "rt", encoding="utf-8") if path.name.endswith(".gz") else path.open()) as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)
    else:
        data = json.loads(path.read_text())
        yield from data if isinstance(data, list) else [data]


def files(sources):
    paths = []
    for source in map(Path, sources):
        if source.is_dir():
            paths += sorted(p for p in source.iterdir()
                            if p.is_file() and (p.suffix == ".json" or p.name.endswith((".jsonl", ".jsonl.gz")))
                            and not p.name.endswith(("combats.jsonl", "combats.jsonl.gz")))
        elif source.is_file():
            paths.append(source)
        else:
            raise FileNotFoundError(source)
    return paths


def load(sources):
    """Rows of every complete run under `sources`, and what was read: files that hold
    no run (a report, a journal) are skipped, so are runs that did not finish."""
    rows, summary = [], Counter()
    policies = {}
    for path in files(sources):
        for item in _read(path):
            if not isinstance(item, dict) or item.get("schema") != SCHEMA:
                summary["skipped_files"] += 1
                break
            if item.get("status") != "complete":
                summary["incomplete_runs"] += 1
                continue
            count = len(rows)
            rows.extend(decisions(item))
            summary["runs"] += 1
            summary["wins"] += bool(item["victory"])
            key = (item.get("policy_version"), item.get("policy_digest"))
            policies[key] = policies.get(key, 0) + len(rows) - count
    if not rows:
        raise ValueError(f"No out-of-combat decisions of complete runs under {list(map(str, sources))}")
    summary["decisions"] = len(rows)
    summary["seeds"] = len({r.seed for r in rows})
    return rows, dict(summary, version=DATA_VERSION, sources=[str(s) for s in sources],
                      policies=[dict(policy_version=v, policy_digest=d, decisions=n)
                                for (v, d), n in sorted(policies.items(), key=str)])


def split(rows, holdout):
    """(train, test) by seed, with the policy's rule: a seed's side does not change as
    data is added, and it is the side the seed has in the policy's validation split."""
    seeds = {r.seed for r in rows}
    if len(seeds) < 2:
        raise ValueError("At least two seeds are required for a leakage-free split")
    parts = [], []
    for row in rows:
        parts[held_out({"seed": row.seed, "run_id": row.run_id}, holdout)].append(row)
    if not parts[0] or not parts[1]:
        raise ValueError("The split left one side empty; add seeds or change the holdout share")
    return parts


def coverage(rows):
    """Decisions, win and act-pass rates by act, character and phase."""
    def rates(group):
        return dict(decisions=len(group), win_rate=round(st.mean(r.win for r in group), 4),
                    act_pass_rate=round(st.mean(r.act_passed for r in group), 4),
                    seeds=len({r.seed for r in group}))

    def by(key):
        groups = {}
        for row in rows:
            groups.setdefault(key(row), []).append(row)
        return {str(k): rates(v) for k, v in sorted(groups.items())}

    return dict(version=DATA_VERSION, all=rates(rows), act=by(lambda r: r.act),
                character=by(lambda r: r.character), phase=by(lambda r: r.phase))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Coverage of the out-of-combat decisions under runs")
    parser.add_argument("sources", type=Path, nargs="+")
    loaded, summary = load(parser.parse_args().sources)
    print(json.dumps(dict(summary, coverage=coverage(loaded)), ensure_ascii=False, indent=1))
