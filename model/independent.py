"""Import Spire Codex independent supervised samples as Bootstrap runs.

A sample group is the outside-combat decisions of one public run, or one battle
the solver fought from a restored entry state. Each group becomes a
supervised-only run. Steps that the engine routed as one buffered selection form
one macro; every other decision is a macro of its own.
"""

import gzip
import json
from collections import Counter
from itertools import groupby
from pathlib import Path

from .data import INDEX, validate_run
from .protocol import CHARACTERS, SCHEMA, ProtocolError, clean_frame, fingerprint, segment_key

SAMPLE_SCHEMA = "spire-independent-decisions-v1"
IMPORT_VERSION = "independent-import-v2"


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
        "frame": clean_frame(frame),
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
    macros, automatic = [], 0
    for _, steps in groupby(map(step_for, rows), key=lambda s: segment_key(s["frame"])):
        steps = list(steps)
        if all(s["forced"] for s in steps):
            automatic += len(steps)
        else:
            macros.append({"phase": steps[0]["frame"]["public"]["phase"], "steps": steps})
    require(macros, "Sample group contains no branching decisions")
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
            "provenance": {
                "importer": IMPORT_VERSION,
                "bc_only": True,
                "run_hash": metadata["run_hash"],
                "game_version": metadata.get("game_version"),
                "actors": dict(Counter(r["actor"] for r in rows)),
            },
        }
    )


class ShardWriter:
    """Accepted runs as gzip shards of bounded size, with one index row per run."""

    def __init__(self, directory, shard_samples):
        self.directory, self.limit = Path(directory), shard_samples
        (self.directory / "accepted").mkdir()
        self.index = gzip.open(self.directory / INDEX, "xt", encoding="utf-8")
        self.stream, self.filled, self.shards, self.contracts = None, 0, [], {}

    def write(self, run, samples):
        if self.stream is None or self.filled >= self.limit:
            self.close_shard()
            self.shards.append(f"accepted/{len(self.shards):05}.jsonl.gz")
            self.stream = gzip.open(self.directory / self.shards[-1], "xt", encoding="utf-8")
        self.stream.write(json.dumps(run, ensure_ascii=False, allow_nan=False) + "\n")
        self.filled += samples
        contract = fingerprint(run["contract"])
        self.contracts[contract] = run["contract"]
        self.index.write(json.dumps({
            "run_id": run["run_id"], "seed": run["seed"], "character": run["character"],
            "phases": dict(Counter(m["phase"] for m in run["macros"])), "contract": contract,
            "shard": self.shards[-1]}) + "\n")

    def close_shard(self):
        if self.stream is not None:
            self.stream.close()
        self.stream, self.filled = None, 0

    def close(self):
        self.close_shard()
        self.index.close()


def import_samples(path, output, shard_samples=2000):
    """Create an immutable import batch from an exported sample file.

    The batch is a set of gzip shards (a sample group is never split) plus an
    index, read back through model.data.RunShards a few shards at a time.
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
        rows = (json.loads(line) for line in stream if line.strip())
        # The export writes every group contiguously and in decision order.
        for group, members in groupby(rows, key=lambda r: r["metadata"]["sample_group"]):
            members = list(members)
            counts["samples"] += len(members)
            try:
                require(group not in identities, "Duplicate sample group")
                run = convert_group(members)
                identities.add(group)
                writer.write(run, len(members))
                counts["accepted_runs"] += 1
                counts["macros"] += len(run["macros"])
                counts["automatic_steps"] += run["automatic_steps"]
                for macro in run["macros"]:
                    counts[run["character"] + "/" + macro["phase"]] += 1
            except (ProtocolError, ValueError, KeyError, TypeError) as exc:
                reason = str(exc)
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
