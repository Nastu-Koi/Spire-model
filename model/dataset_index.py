"""Versioned policy counts over immutable demonstration shards."""

import gzip
import json
import os
import shutil
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .control import CONTROL_VERSION, policy_macros
from .protocol import ProtocolError, fingerprint


INDEX = "index.jsonl.gz"
HEADER = dict(kind="policy_index", control_version=CONTROL_VERSION)


def read_index(directory):
    path = Path(directory) / INDEX
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        if json.loads(next(stream, "null")) != HEADER:
            raise ProtocolError(
                "Missing or stale policy index; rebuild it with "
                f"python -m model reindex --data {Path(directory)}"
            )
        return [json.loads(line) for line in stream if line.strip()]


def index_row(run, shard, line):
    return dict(run_id=run["run_id"], seed=run["seed"], character=run["character"],
                phases=dict(Counter(m["phase"] for m in policy_macros(run))),
                contract=fingerprint(run["contract"]), shard=shard, line=line)


def _index_shard(task):
    directory, name = task
    rows, counts, excluded = [], Counter(), Counter()
    raw_macros, training_groups = 0, 0
    with gzip.open(directory / name, "rb") as stream:
        for line, payload in enumerate(stream):
            run = json.loads(payload)
            row = index_row(run, name, line)
            rows.append(json.dumps(row, ensure_ascii=False) + "\n")
            raw_macros += len(run["macros"])
            training_groups += bool(row["phases"])
            counts.update({run["character"] + "/" + phase: n for phase, n in row["phases"].items()})
            removed = Counter(m["phase"] for m in run["macros"])
            removed.subtract(row["phases"])
            excluded.update({phase: n for phase, n in removed.items() if n})
    return dict(member=gzip.compress("".join(rows).encode(), compresslevel=6),
                groups=len(rows), training_groups=training_groups, raw_macros=raw_macros,
                counts=counts, excluded=excluded)


def rebuild_index(directory, *, workers=1):
    """Recount all policy macros, then atomically publish an index with this rule version.

    A failed scan leaves the previous index in place. Source shards and import
    summaries are not changed; the previous index is retained before replacement.
    """
    directory = Path(directory)
    if workers < 1:
        raise ValueError("Index workers must be positive")
    names = sorted(p.relative_to(directory).as_posix() for p in (directory / "accepted").glob("*.jsonl.gz"))
    if not names:
        raise ProtocolError("No accepted gzip shards to index")
    descriptor, temporary = tempfile.mkstemp(prefix=".policy-index-", suffix=".gz", dir=directory)
    temporary = Path(temporary)
    counts, excluded, totals = Counter(), Counter(), Counter()
    pool = None
    try:
        with os.fdopen(descriptor, "wb") as stream:
            tasks = ((directory, name) for name in names)
            if workers > 1:
                pool = ProcessPoolExecutor(max_workers=workers)
                results = pool.map(_index_shard, tasks)
            else:
                results = map(_index_shard, tasks)
            stream.write(gzip.compress((json.dumps(HEADER) + "\n").encode(), compresslevel=6))
            for result in results:
                stream.write(result["member"])
                counts.update(result["counts"])
                excluded.update(result["excluded"])
                for key in ("groups", "training_groups", "raw_macros"):
                    totals[key] += result[key]
            stream.flush()
            os.fsync(stream.fileno())
        target = directory / INDEX
        backup = directory / (INDEX + ".before-" + CONTROL_VERSION)
        if target.exists() and not backup.exists():
            shutil.copy2(target, backup)
        temporary.replace(target)
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
        temporary.unlink(missing_ok=True)
    report = dict(control_version=CONTROL_VERSION, shards=len(names), **totals,
                  policy_macros=sum(counts.values()), counts=dict(sorted(counts.items())),
                  excluded_macros=dict(sorted(excluded.items())))
    report_path = directory / "policy-index-summary.json"
    stage = report_path.with_suffix(".tmp")
    stage.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    stage.replace(report_path)
    return report
