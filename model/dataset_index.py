"""Versioned policy counts over immutable shards of runs: imported demonstrations and
sampled rounds alike."""

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
from .seeds import seed_group


INDEX = "index.jsonl.gz"
HEADER = dict(kind="policy_index", control_version=CONTROL_VERSION)
# What the index says of a sampled run beyond its identity: the buffer keeps the
# rounds of the last few iterations, and reports count by policy and outcome.
ROLLOUT_FIELDS = ("policy_version", "policy_digest", "iteration", "victory")
# The solver's labels on the fights the policy itself walked into are imported as
# independent samples like the demonstrations, and told apart by their anchor kind:
# the trainer gives them a share of their groups' weight (`data.group_weights`). The
# index names the policy whose fights they are, as it names a sampled run's.
STUDENT_BATTLE = "student_battle"
STUDENT_FIELDS = ("policy_version", "policy_digest", "iteration")
# Level 6 packs runs within a few percent of level 9 in a third of the time.
COMPRESSION = 6


def sample_source(run):
    """What the trainer weighs a run's samples as: `student_battle` for the solver's
    labels on the policy's own fights, otherwise the run's source (demonstrations are
    `independent`, sampled runs `rollout`)."""
    if (run.get("provenance") or {}).get("anchor_kind") == STUDENT_BATTLE:
        return STUDENT_BATTLE
    return run.get("source")


def row_source(row):
    """The source of an index row. An index written before sources were recorded
    holds demonstrations, or a sampled round, which carries its policy fields."""
    source = row.get("source")
    if source is None:
        return "rollout" if "victory" in row else "independent"
    return source


def read_index(directory):
    path = Path(directory) / INDEX
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        if json.loads(next(stream, "null")) != HEADER:
            raise ProtocolError(
                "Missing or stale policy index; rebuild it with "
                f"python -m model reindex --data {Path(directory)}"
            )
        rows = [json.loads(line) for line in stream if line.strip()]
    for row in rows:
        # Older imports omit source; their seed is already a split group. Old
        # sampled rounds are distinguished by the outcome fields (`row_source`).
        if "split_group" not in row:
            row["split_group"] = seed_group(dict(row, source=row_source(row)))
    return rows


def member(run):
    """A run as one line of a shard, as a gzip member of its own: a shard is the
    members of its runs one after another, which reads as one stream."""
    text = json.dumps(run, ensure_ascii=False, allow_nan=False) + "\n"
    return gzip.compress(text.encode("utf-8"), COMPRESSION)


def index_row(run, shard=None, line=None, phases=None):
    """The index's row of a run. `phases` is the count of its policy macro-actions by
    phase, counted here from its frames unless the caller counted them already. The
    shard and line are left out when the caller does not know them yet."""
    if phases is None:
        phases = Counter(m["phase"] for m in policy_macros(run))
    source = sample_source(run)
    row = dict(run_id=run["run_id"], seed=run["seed"], character=run["character"],
               ascension=run.get("ascension"), phases=dict(phases),
               contract=fingerprint(run["contract"]), source=source, split_group=seed_group(run))
    if run.get("source") == "rollout":
        row.update({name: run.get(name) for name in ROLLOUT_FIELDS})
    if source == STUDENT_BATTLE:
        row.update({name: run["provenance"].get(name) for name in STUDENT_FIELDS})
    if shard is not None:
        row.update(shard=shard, line=line)
    return row


class ShardWriter:
    """Accepted runs as gzip shards of bounded size, with one index row per run."""

    def __init__(self, directory, shard_samples, *, flush=False):
        """With `flush`, every row reaches the index file as it is written: a round
        that is killed keeps the index of what it finished. An import of a million
        runs is better off buffering."""
        if type(shard_samples) is not int or shard_samples < 1:
            raise ValueError("Shard size must be a positive number of samples")
        self.directory, self.limit = Path(directory), shard_samples
        (self.directory / "accepted").mkdir()
        self.index = gzip.open(self.directory / INDEX, "xt", encoding="utf-8")
        self.index.write(json.dumps(HEADER) + "\n")
        self.flush = flush
        self.stream, self.filled, self.shards, self.contracts = None, 0, [], {}
        self.line = 0

    def write(self, member, row, samples, contract):
        """A run: its gzip member, its index row without shard and line, how many
        samples (decision steps) it holds, and the engine contract it names."""
        if self.stream is None or self.filled >= self.limit:
            self.close_shard()
            self.shards.append(f"accepted/{len(self.shards):05}.jsonl.gz")
            self.stream = (self.directory / self.shards[-1]).open("xb")
        self.stream.write(member)
        self.filled += samples
        self.contracts[row["contract"]] = contract
        self.index.write(json.dumps(dict(row, shard=self.shards[-1], line=self.line), ensure_ascii=False) + "\n")
        self.line += 1
        if self.flush:
            self.stream.flush()
            self.index.flush()

    def close_shard(self):
        if self.stream is not None:
            self.stream.close()
        self.stream, self.filled = None, 0
        self.line = 0

    def close(self):
        self.close_shard()
        if not self.index.closed:
            self.index.close()


def _index_shard(task):
    directory, name, listed = task
    rows, counts, excluded = [], Counter(), Counter()
    raw_macros, training_groups = 0, 0
    with gzip.open(directory / name, "rb") as stream:
        for line, payload in enumerate(stream):
            if listed is not None and line not in listed:
                continue
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


def listed_runs(directory):
    """The lines of every shard that the present index of `directory` lists."""
    listed = {}
    with gzip.open(Path(directory) / INDEX, "rt", encoding="utf-8") as stream:
        next(stream, None)
        for line in stream:
            if line.strip():
                row = json.loads(line)
                listed.setdefault(row["shard"], set()).add(row["line"])
    return listed


def rebuild_index(directory, *, workers=1, listed_only=False):
    """Recount all policy macros, then atomically publish an index with this rule version.

    A failed scan leaves the previous index in place. Source shards and import
    summaries are not changed; the previous index is retained before replacement.

    A directory that shares another's shards holds a part of them: what its index
    lists. `listed_only` keeps to those runs; without it such a directory is refused,
    since its index would come out as the whole of the other's.
    """
    directory = Path(directory)
    if workers < 1:
        raise ValueError("Index workers must be positive")
    if (directory / "accepted").is_symlink() and not listed_only:
        raise ProtocolError("The shards belong to another directory, of which this index lists a part; "
                            "rebuild it with --listed-only")
    names = sorted(p.relative_to(directory).as_posix() for p in (directory / "accepted").glob("*.jsonl.gz"))
    if not names:
        raise ProtocolError("No accepted gzip shards to index")
    listed = listed_runs(directory) if listed_only else None
    if listed is not None:
        if set(listed) - set(names):
            raise ProtocolError("The index lists shards that are missing")
        names = sorted(listed)
    descriptor, temporary = tempfile.mkstemp(prefix=".policy-index-", suffix=".gz", dir=directory)
    temporary = Path(temporary)
    counts, excluded, totals = Counter(), Counter(), Counter()
    pool = None
    try:
        with os.fdopen(descriptor, "wb") as stream:
            tasks = ((directory, name, None if listed is None else listed[name]) for name in names)
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
