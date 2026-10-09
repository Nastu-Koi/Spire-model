"""Import Spire Codex independent supervised samples as Bootstrap runs.

A sample group is the outside-combat decisions of one public run, or one battle
the solver fought from a restored entry state. Each group becomes a
supervised-only run. Steps that the engine routed as one buffered selection form
one macro; every other decision is a macro of its own.
"""

import gzip
import json
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from itertools import groupby
from pathlib import Path

from .control import controller_for
from .data import validate_run
from .dataset_index import HEADER, INDEX
from .protocol import CHARACTERS, SCHEMA, ProtocolError, clean_frame, fingerprint, segment_key, validate_frame

SAMPLE_SCHEMA = "spire-independent-decisions-v1"
IMPORT_VERSION = "independent-import-v6"
# Level 6 packs runs within a few percent of level 9 in a third of the time.
COMPRESSION = 6
# Samples handed to a worker at a time, in whole groups.
BATCH_SAMPLES = 64


def require(condition, message):
    if not condition:
        raise ProtocolError(message)


def step_for(row):
    require(row.get("schema") == SAMPLE_SCHEMA, "Unsupported sample schema")
    metadata, options = row["metadata"], row["options"]
    # A solver battle is verified by replaying it; a recorded choice by reproducing the
    # recorded outcome, or by being a legal move where the outcome is not executed.
    verified = metadata.get("native_replay_verified") is True or metadata.get("verified_by") in (
        "recorded_outcome", "legal_label")
    require(verified and metadata.get("bc_only") is True, "Sample is not a verified supervised-only decision")
    chosen = [i for i, option in enumerate(options) if option == row["label"]]
    require(len(chosen) == 1, "Label is not a unique legal option")
    frame = {
        "type": "decision_frame",
        "contract": metadata["contract"],
        "boundary": "decision",
        "routing": metadata["routing"],
        "public": row["observation"],
        "legal": {
            "candidates": [
                dict(option, candidate_ref=f"c{i}") for i, option in enumerate(options)
            ]
        },
        "events": [],
    }
    return {
        "frame": clean_frame(validate_frame(frame)),
        "candidate_ref": f"c{chosen[0]}",
        "forced": len(options) == 1,
    }


def convert_group(rows):
    """One supervised run from the ordered samples of a group."""
    metadata = rows[0]["metadata"]
    for key in ("character", "ascension", "teacher_visibility", "split_group", "contract"):
        require(
            all(r["metadata"].get(key) == metadata.get(key) for r in rows),
            f"Sample group mixes {key}",
        )
    character = {c.upper(): c for c in CHARACTERS}.get(str(metadata["character"]).upper())
    macros, automatic, environment = [], 0, []
    entries = ((row, step_for(row)) for row in rows)
    for _, members in groupby(entries, key=lambda pair: segment_key(pair[1]["frame"])):
        members = list(members)
        steps = [step for _, step in members]
        controllers = [controller_for(s["frame"]["public"]["phase"], s["frame"]["legal"]["candidates"])
                       for s in steps]
        if any(controllers):
            require(all(controllers),
                    "Macro mixes policy and environment steps")
            environment.extend(dict(actor=row.get("actor", "unknown"), **step) for row, step in members)
            automatic += len(steps)
        elif all(s["forced"] for s in steps):
            automatic += len(steps)
        else:
            macros.append({"phase": steps[0]["frame"]["public"]["phase"], "steps": steps})
    require(macros or environment, "Sample group contains no branching decisions")
    return validate_run(
        {
            "schema": SCHEMA,
            "source": "independent",
            "teacher_visibility": metadata["teacher_visibility"],
            "status": "partial",
            "victory": None,
            "ascension": metadata["ascension"],
            "character": character,
            "run_id": metadata["sample_group"],
            "seed": metadata["split_group"],
            "contract": metadata["contract"],
            "macros": macros,
            "automatic_steps": automatic,
            "environment_actions": environment,
            "provenance": {
                "importer": IMPORT_VERSION,
                "bc_only": True,
                "run_hash": metadata["run_hash"],
                "game_version": metadata.get("game_version"),
                "actors": dict(Counter(r["actor"] for r in rows)),
                # How far the source run is audited and which states were filled in
                # rather than recorded. Kept for reports, never a model input.
                "source_integrity": metadata.get("source_integrity"),
                "initialization": metadata.get("initialization"),
                "label_sources": dict(Counter(r["metadata"].get("label_source") for r in rows)),
                "verified_by": dict(Counter(
                    "native_replay" if r["metadata"].get("native_replay_verified") is True
                    else r["metadata"].get("verified_by") for r in rows)),
                "state_sources": dict(Counter(
                    source for r in rows for source in r["metadata"].get("state_sources") or [])),
                **({"recovery": metadata["recovery"]} if "recovery" in metadata else {}),
            },
        }
    )


def convert_lines(batch):
    """The groups of `batch`, each as (group, samples, run, None) or (group, samples,
    None, why it is refused). A run is its line of a shard as a gzip member of its own,
    with what the index and the summary say of it."""
    results = []
    for group, lines in batch:
        try:
            run = convert_group([json.loads(line) for line in lines])
            text = json.dumps(run, ensure_ascii=False, allow_nan=False) + "\n"
            results.append((group, len(lines), {
                "member": gzip.compress(text.encode("utf-8"), COMPRESSION),
                "run_id": run["run_id"], "seed": run["seed"], "character": run["character"],
                "phases": dict(Counter(m["phase"] for m in run["macros"])),
                "contract": run["contract"], "contract_key": fingerprint(run["contract"]),
                "automatic_steps": run["automatic_steps"]}, None))
        except (ProtocolError, ValueError, KeyError, TypeError) as exc:
            results.append((group, len(lines), None, str(exc)))
    return results


def in_order(pool, function, tasks, ahead):
    """`function` of every task, in order, with at most `ahead` tasks handed out and
    not yet read: the tasks are read from a file far larger than memory."""
    pending = deque()
    for task in tasks:
        pending.append(pool.submit(function, task))
        if len(pending) >= ahead:
            yield pending.popleft().result()
    while pending:
        yield pending.popleft().result()


class ShardWriter:
    """Accepted runs as gzip shards of bounded size, with one index row per run. A shard
    is the gzip members of its runs one after another, which reads as one stream."""

    def __init__(self, directory, shard_samples):
        self.directory, self.limit = Path(directory), shard_samples
        (self.directory / "accepted").mkdir()
        self.index = gzip.open(self.directory / INDEX, "xt", encoding="utf-8")
        self.index.write(json.dumps(HEADER) + "\n")
        self.stream, self.filled, self.shards, self.contracts = None, 0, [], {}
        self.line = 0

    def write(self, run, samples):
        if self.stream is None or self.filled >= self.limit:
            self.close_shard()
            self.shards.append(f"accepted/{len(self.shards):05}.jsonl.gz")
            self.stream = (self.directory / self.shards[-1]).open("xb")
        self.stream.write(run["member"])
        self.filled += samples
        self.contracts[run["contract_key"]] = run["contract"]
        self.index.write(json.dumps({
            "run_id": run["run_id"], "seed": run["seed"], "character": run["character"],
            "phases": run["phases"], "contract": run["contract_key"],
            "shard": self.shards[-1], "line": self.line}) + "\n")
        self.line += 1

    def close_shard(self):
        if self.stream is not None:
            self.stream.close()
        self.stream, self.filled = None, 0
        self.line = 0

    def close(self):
        self.close_shard()
        self.index.close()


def batches(groups):
    """Whole groups, in order, about BATCH_SAMPLES samples at a time."""
    batch, samples = [], 0
    for group, lines in groups:
        batch.append((group, lines))
        samples += len(lines)
        if samples >= BATCH_SAMPLES:
            yield batch
            batch, samples = [], 0
    if batch:
        yield batch


def converted(tasks, workers):
    """convert_lines of every batch, in order; in this process when there is one worker."""
    if workers <= 1:
        yield from map(convert_lines, tasks)
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        yield from in_order(pool, convert_lines, tasks, 4 * workers)


def import_samples(path, output, shard_samples=2000, workers=1):
    """Create an immutable import batch from an exported sample file.

    The batch is a set of gzip shards (a sample group is never split) plus an
    index, read back through model.data.RunShards a few shards at a time. Groups
    are converted by `workers` processes and written in the order of the file.
    """
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=False)
    counts, reasons, identities = Counter(), Counter(), set()
    source = Path(path)
    writer = ShardWriter(directory, shard_samples)
    with (
        # Summary-anchor exports are gzip files; the format is otherwise the same.
        (gzip.open(source, "rt", encoding="utf-8") if source.suffix == ".gz" else source.open(encoding="utf-8")) as stream,
        (directory / "quarantine.jsonl").open("x", encoding="utf-8") as bad,
    ):
        # The export writes every group contiguously and in decision order.
        rows = ((json.loads(line)["metadata"]["sample_group"], line) for line in stream if line.strip())
        groups = ((group, [line for _, line in members]) for group, members in groupby(rows, key=lambda row: row[0]))
        for results in converted(batches(groups), workers):
            for group, samples, run, reason in results:
                counts["samples"] += samples
                if group in identities:
                    reason = "Duplicate sample group"
                if reason is None:
                    identities.add(group)
                    writer.write(run, samples)
                    counts["accepted_runs"] += 1
                    counts["macros"] += sum(run["phases"].values())
                    counts["automatic_steps"] += run["automatic_steps"]
                    for phase, macros in run["phases"].items():
                        counts[run["character"] + "/" + phase] += macros
                else:
                    reasons[reason] += 1
                    counts["quarantined_groups"] += 1
                    bad.write(
                        json.dumps({"sample_group": group, "reason": reason}, ensure_ascii=False)
                        + "\n"
                    )
    writer.close()
    summary = {
        "importer": IMPORT_VERSION,
        "source": str(source.resolve()),
        "accepted_runs": 0,
        **dict(counts),
        "quarantine_reasons": dict(reasons),
        "accepted": str(directory),
        "shards": len(writer.shards),
        "shard_samples": shard_samples,
        "contracts": writer.contracts,
        "quarantine": str(directory / "quarantine.jsonl"),
    }
    (directory / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
