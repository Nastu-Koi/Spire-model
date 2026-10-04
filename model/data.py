"""Audited complete-run data. Never invent candidates from a teacher's label."""

import gzip
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .protocol import (
    CHARACTERS,
    SCHEMA,
    ProtocolError,
    clean_frame,
    segment_key,
    validate_frame,
)


def validate_run(run, *, on_policy=False):
    recorder_bc = run.get("source") == "recorder_bc"
    if recorder_bc and (
        on_policy
        or run.get("provenance", {}).get("bc_only") is not True
        or run.get("teacher_visibility") != "unverified"
    ):
        raise ProtocolError(
            "Recorder Bootstrap data is supervised-only with unverified teacher visibility"
        )
    independent = run.get("source") == "independent"
    if independent and (
        on_policy
        or run.get("provenance", {}).get("bc_only") is not True
        or run.get("teacher_visibility") not in {"recorded_public", "privileged"}
    ):
        raise ProtocolError(
            "Independent samples are supervised-only with a declared teacher visibility"
        )
    ascension = run.get("ascension")
    if (
        run.get("schema") != SCHEMA
        or type(ascension) is not int
        or not 0 <= ascension <= 10
    ):
        raise ProtocolError("Unsupported trajectory schema or ascension")
    if recorder_bc:
        if run.get("status") != "partial" or run.get("victory") is not None:
            raise ProtocolError(
                "Recorder Bootstrap decisions must not claim a complete outcome"
            )
    elif independent:
        if run.get("status") != "partial" or run.get("victory") is not None:
            raise ProtocolError(
                "Independent samples must not claim a complete outcome"
            )
    elif run.get("status") != "complete" or type(run.get("victory")) is not bool:
        raise ProtocolError("Incomplete, erroneous or unresolved run")
    if run.get("character") not in CHARACTERS or not run.get("run_id"):
        raise ProtocolError("Missing character/run identity")
    if run.get("source") not in {
        "demonstration",
        "recorder_bc",
        "independent",
        "on_policy",
        "evaluation",
    }:
        raise ProtocolError("Unknown trajectory source")
    if on_policy and (
        run.get("source") != "on_policy" or run.get("sampling") != "full_distribution"
    ):
        raise ProtocolError(
            "PPO requires autonomous full-distribution on-policy samples"
        )
    if (
        not on_policy
        and run.get("source") == "demonstration"
        and run.get("teacher_visibility") != "public"
    ):
        raise ProtocolError("Teacher visibility has not been verified")
    seen = set()
    for macro in run.get("macros", []):
        steps = macro.get("steps", [])
        if not steps:
            raise ProtocolError("Empty macro")
        segment = segment_key(steps[0]["frame"])
        branches = 0
        for step in steps:
            frame = step["frame"]
            validate_frame(frame)
            if frame["contract"].get("fixed_ascension") != ascension:
                raise ProtocolError("Frame ascension disagrees with run")
            if frame["contract"] != run["contract"]:
                raise ProtocolError("Mixed engine contracts")
            if segment_key(frame) != segment:
                raise ProtocolError("Macro crosses a reveal/reset boundary")
            decision = frame["routing"]["decision_id"]
            if decision in seen:
                raise ProtocolError("Repeated decision in a run")
            seen.add(decision)
            candidates = frame["legal"]["candidates"]
            if step["candidate_ref"] not in {c["candidate_ref"] for c in candidates}:
                raise ProtocolError("Label is not in the full legal candidate set")
            forced = len(candidates) == 1
            if step.get("forced") != forced:
                raise ProtocolError("Forced-step classification disagrees with engine")
            branches += not forced
        if not branches:
            raise ProtocolError("Forced-only macro must not be a training sample")
        if on_policy:
            for key in ("old_log_prob", "old_value", "return", "advantage", "reward"):
                if not isinstance(macro.get(key), (float, int)) or not math.isfinite(
                    macro[key]
                ):
                    raise ProtocolError(f"Missing or non-finite PPO {key}")
            if abs(macro["advantage"] - (macro["return"] - macro["old_value"])) > 1e-5:
                raise ProtocolError(
                    "Advantage must equal complete return minus old value"
                )
    if on_policy:
        total = 0.0
        for macro in reversed(run["macros"]):
            total += macro["reward"]
            if abs(total - macro["return"]) > 1e-5:
                raise ProtocolError("Invalid complete-run Monte Carlo return")
    return run


def audit_files(paths, accepted, quarantine):
    accepted, quarantine = Path(accepted), Path(quarantine)
    accepted.parent.mkdir(parents=True, exist_ok=True)
    quarantine.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    identities = set()
    with accepted.open("w") as good, quarantine.open("w") as bad:
        for path in paths:
            try:
                run = json.loads(Path(path).read_text())
                validate_run(run)
                identity = run["run_id"]
                if identity in identities:
                    raise ProtocolError("Duplicate complete-run identity")
                identities.add(identity)
                for macro in run["macros"]:
                    for step in macro["steps"]:
                        step["frame"] = clean_frame(step["frame"])
                    counts[run["character"] + "/" + macro["phase"]] += 1
                good.write(json.dumps(run, allow_nan=False) + "\n")
                counts["accepted_runs"] += 1
            except (ValueError, KeyError, TypeError, OSError, ProtocolError) as exc:
                bad.write(json.dumps({"path": str(path), "reason": str(exc)}) + "\n")
                counts["quarantined_runs"] += 1
    return dict(counts)


def load_runs(path):
    """Every run of a small data set, in memory. Imports too large for that are
    read through RunShards instead."""
    path = Path(path)
    if path.is_dir():
        runs = [json.loads(p.read_text()) for p in sorted(path.glob("*.json"))]
    elif path.name.endswith(".jsonl.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            runs = [json.loads(line) for line in stream if line.strip()]
    elif path.suffix == ".jsonl":
        runs = [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
    else:
        data = json.loads(path.read_text())
        runs = data if isinstance(data, list) else [data]
    for run in runs:
        validate_run(run)
    identities = [r["run_id"] for r in runs]
    if len(identities) != len(set(identities)):
        raise ProtocolError("Duplicate run identity")
    return runs


def held_out(run, validation_fraction):
    """Whether a run's seed falls in the validation part. Every run and fragment of
    one seed lands on the same side."""
    from .protocol import fingerprint

    key = str(run.get("seed", run["run_id"]))
    return int(fingerprint(key)[:8], 16) / 2**32 < validation_fraction


def split_runs(runs, validation_fraction=0.2):
    """Group every seed across characters/fragments; never split adjacent decisions."""
    train, validation = [], []
    for run in runs:
        (validation if held_out(run, validation_fraction) else train).append(run)
    return train, validation


INDEX = "index.jsonl.gz"


class RunShards:
    """Imported runs kept as gzip shards and read a few shards at a time.

    The index holds one small row per run (identity, seed, character, macro counts
    by phase, shard), so splits and sample weights need no pass over the data.
    """

    def __init__(self, directory, rows=None):
        self.directory = Path(directory)
        if rows is None:
            with gzip.open(self.directory / INDEX, "rt", encoding="utf-8") as stream:
                rows = [json.loads(line) for line in stream if line.strip()]
        self.rows = rows

    @staticmethod
    def is_at(path):
        return (Path(path) / INDEX).is_file()

    def __len__(self):
        return len(self.rows)

    def split(self, validation_fraction):
        parts = [], []
        for row in self.rows:
            parts[held_out(row, validation_fraction)].append(row)
        return RunShards(self.directory, parts[0]), RunShards(self.directory, parts[1])

    @property
    def counts(self):
        """Macros per (character, phase): what the sample weights are normalized by."""
        counts = Counter()
        for row in self.rows:
            for phase, macros in row["phases"].items():
                counts[row["character"], phase] += macros
        return counts

    @property
    def seeds(self):
        return {str(row["seed"]) for row in self.rows}

    @property
    def identities(self):
        return {row["run_id"] for row in self.rows}

    def metadata(self):
        from .protocol import fingerprint

        known = json.loads((self.directory / "summary.json").read_text())["contracts"]
        return {
            "training_engine_contracts": [known[key] for key in sorted({r["contract"] for r in self.rows})],
            "training_seeds": sorted(self.seeds),
            "training_seed_pool_hash": fingerprint(sorted((r["character"], str(r["seed"])) for r in self.rows)),
        }

    def windows(self, shards=8, shuffle=False):
        """Lists of validated runs, each from `shards` shard files."""
        wanted = defaultdict(set)
        for row in self.rows:
            wanted[row["shard"]].add(row["run_id"])
        names = sorted(wanted)
        if shuffle:
            random.shuffle(names)
        for start in range(0, len(names), shards):
            runs = []
            for name in names[start : start + shards]:
                with gzip.open(self.directory / name, "rt", encoding="utf-8") as stream:
                    for line in stream:
                        run = json.loads(line)
                        if run["run_id"] in wanted[name]:
                            runs.append(validate_run(run))
            yield runs


@dataclass
class Sample:
    macro: dict
    character: str
    run_id: str
    weight: float
    _lengths: tuple[tuple[int, int], ...] | None = field(
        default=None, repr=False, compare=False
    )


def samples(runs, *, ppo=False, horizon_scale=100.0, counts=None):
    """Weighted samples. `counts` gives the macros per (character, phase) of the
    whole data set when `runs` is only a window of it."""
    result = []
    if ppo:
        counts = Counter(r["character"] for r in runs)
        if set(counts) != set(CHARACTERS) or len(set(counts.values())) != 1:
            raise ProtocolError(
                "PPO round requires equal complete-run counts for all five characters"
            )
        versions = {
            (r["policy_version"], r["precision"], r["vocabulary_hash"]) for r in runs
        }
        if len(versions) != 1:
            raise ProtocolError("Mixed policy/precision/vocabulary rollout versions")
        from .protocol import fingerprint

        if len({fingerprint(r["contract"]) for r in runs}) != 1:
            raise ProtocolError("PPO round mixes engine/content/effect contracts")
        for run in runs:
            validate_run(run, on_policy=True)
            for macro in run["macros"]:
                result.append(
                    Sample(
                        macro,
                        run["character"],
                        run["run_id"],
                        1 / (5 * counts[run["character"]] * horizon_scale),
                    )
                )
    else:
        if counts is None:
            counts = Counter(
                (r["character"], m["phase"]) for r in runs for m in r["macros"]
            )
        for run in runs:
            validate_run(run)
            for macro in run["macros"]:
                result.append(
                    Sample(
                        macro,
                        run["character"],
                        run["run_id"],
                        1 / (len(counts) * counts[run["character"], macro["phase"]]),
                    )
                )
    return result


def bucket_size(length, buckets):
    return next((b for b in buckets if b >= length), length)


def sample_lengths(item):
    """(tokens, action slots) of every branching decision of a sample."""
    from .representation import observation

    lengths = []
    for step in item.macro["steps"]:
        if not step["forced"]:
            obs = observation(step["frame"])
            lengths.append((len(obs.tokens), len(obs.slot_refs)))
    return tuple(lengths)


def sample_pad_size(item, config):
    """Return the token bucket for a sample, caching only input dimensions."""
    if item._lengths is None:
        item._lengths = sample_lengths(item)
    result = 0
    for tokens, actions in item._lengths:
        action_pad = bucket_size(actions, config.action_buckets)
        result = max(
            result, bucket_size(tokens - actions + action_pad, config.token_buckets)
        )
    return result


def batch_pad_size(batch, config):
    return max((sample_pad_size(item, config) for item in batch), default=0)


def microbatches(logical, config):
    """Capacity changes computation order only, never statistical sample weights.

    A replay holds one row per branching decision, so a sample costs as many rows
    as it has of them. A sample is never split: one that exceeds the budgets on
    its own forms a batch of its own.
    """
    result, batch, max_n, rows = [], [], 0, 0
    for item in logical:
        n = sample_pad_size(item, config)
        candidate_n = max(max_n, n)
        count = rows + len(item._lengths)
        if batch and (
            len(batch) + 1 > config.microbatch_size
            or count * candidate_n > config.token_budget
            or count * candidate_n**2 > config.pair_budget
        ):
            result.append(batch)
            batch, max_n, rows = [], 0, 0
        batch.append(item)
        max_n = max(max_n, n)
        rows += len(item._lengths)
    if batch:
        result.append(batch)
    return result
