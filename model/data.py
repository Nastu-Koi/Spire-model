"""Audited complete-run data. Never invent candidates from a teacher's label."""

import json
import math
from collections import Counter
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
    elif run.get("status") != "complete" or type(run.get("victory")) is not bool:
        raise ProtocolError("Incomplete, erroneous or unresolved run")
    if run.get("character") not in CHARACTERS or not run.get("run_id"):
        raise ProtocolError("Missing character/run identity")
    if run.get("source") not in {
        "demonstration",
        "recorder_bc",
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
    path = Path(path)
    if path.is_dir():
        runs = [json.loads(p.read_text()) for p in sorted(path.glob("*.json"))]
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


def split_runs(runs, validation_fraction=0.2):
    """Group every seed across characters/fragments; never split adjacent decisions."""
    from .protocol import fingerprint

    train, validation = [], []
    for run in runs:
        key = str(run.get("seed", run["run_id"]))
        fraction = int(fingerprint(key)[:8], 16) / 2**32
        (validation if fraction < validation_fraction else train).append(run)
    return train, validation


@dataclass
class Sample:
    macro: dict
    character: str
    run_id: str
    weight: float
    _lengths: tuple[tuple[int, int], ...] | None = field(
        default=None, repr=False, compare=False
    )


def samples(runs, *, ppo=False, horizon_scale=100.0):
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


def sample_pad_size(item, config):
    """Return the token bucket for a sample, caching only input dimensions."""
    if item._lengths is None:
        from .representation import observation

        lengths = []
        for step in item.macro["steps"]:
            if not step["forced"]:
                obs = observation(step["frame"])
                lengths.append((len(obs.tokens), len(obs.slot_refs)))
        item._lengths = tuple(lengths)
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
    """Capacity changes computation order only, never statistical sample weights."""
    result, batch, max_n = [], [], 0
    for item in logical:
        n = sample_pad_size(item, config)
        candidate_n = max(max_n, n)
        count = len(batch) + 1
        if batch and (
            count > config.microbatch_size
            or count * candidate_n > config.token_budget
            or count * candidate_n**2 > config.pair_budget
        ):
            result.append(batch)
            batch, max_n = [], 0
        # Oversized samples remain whole; caller can checkpoint/recompute or report capacity.
        batch.append(item)
        max_n = max(max_n, n)
    if batch:
        result.append(batch)
    return result
