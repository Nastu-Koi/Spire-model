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
from .rewards import require_reward_version
from .control import CONTROL_VERSION, controller_for, is_policy_frame, policy_macros
from .dataset_index import INDEX, read_index


SOURCES = {"demonstration", "recorder_bc", "independent", "rollout", "evaluation"}


def _finite(macro):
    """What a sampled run records of each macro-action: the log-probability it was
    played with and the reward paid after it."""
    for key in ("old_log_prob", "reward"):
        if not isinstance(macro.get(key), (float, int)) or not math.isfinite(macro[key]):
            raise ProtocolError(f"Missing or non-finite sampled {key}")


def validate_run(run, *, on_policy=False):
    """Check a run as data. A sampled run (`source == "rollout"`) is held to the
    sampled checks whether or not `on_policy` asks for one: its versions, its
    log-probabilities, and that controller steps are not among its samples. The
    macro-actions it played inside fights keep counts instead of frames."""
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
    if run.get("source") not in SOURCES:
        raise ProtocolError("Unknown trajectory source")
    sampled = run.get("source") == "rollout"
    if (on_policy and not sampled) or (sampled and run.get("sampling") != "full_distribution"):
        raise ProtocolError(
            "A sampled run is an autonomous full-distribution rollout"
        )
    if (
        not on_policy
        and run.get("source") == "demonstration"
        and run.get("teacher_visibility") != "public"
    ):
        raise ProtocolError("Teacher visibility has not been verified")
    seen = set()
    for macro in run.get("macros", []):
        if "steps" not in macro:
            # A fight of a sampled run: taught by the solver from its entry save, it
            # keeps how many decisions it held and how likely its play was, no frames.
            if not sampled or macro.get("fight") is None:
                raise ProtocolError("Only the fights of a sampled run keep no frames")
            branches, decisions = macro.get("branches"), macro.get("decisions")
            if type(branches) is not int or type(decisions) is not int or not 1 <= branches <= decisions:
                raise ProtocolError("A macro-action without frames needs its decision counts")
            _finite(macro)
            continue
        steps = macro["steps"]
        if not steps:
            raise ProtocolError("Empty macro")
        segment = segment_key(steps[0]["frame"])
        branches = 0
        for step in steps:
            frame = step["frame"]
            validate_frame(frame)
            phase = frame["public"]["phase"]
            if sampled and controller_for(phase, frame["legal"]["candidates"]) is not None:
                raise ProtocolError(
                    "Controller actions are environment steps, not policy samples"
                )
            if phase != macro["phase"]:
                raise ProtocolError("Macro phase disagrees with its decision frames")
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
        if sampled:
            _finite(macro)
    if sampled:
        from .public_history import require_history_version

        require_history_version(run.get("contract", {}).get("public_history_version"))
        if run.get("control_version") != CONTROL_VERSION:
            raise ProtocolError("Sampled runs require the current policy control version; collect new rollouts")
        require_reward_version(run.get("reward_version"))
        require_reward_version(run.get("ledger", {}).get("version"))
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
                    for step in macro.get("steps", ()):
                        step["frame"] = clean_frame(step["frame"])
                counts.update(run["character"] + "/" + macro["phase"] for macro in policy_macros(run))
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


class RunShards:
    """Imported runs kept as gzip shards and read a few shards at a time.

    The index holds one small row per run (identity, seed, character, macro counts
    by phase, shard), so splits and sample weights need no pass over the data.
    """

    def __init__(self, directory, rows=None):
        self.directory = Path(directory)
        if rows is None:
            rows = read_index(self.directory)
        self.rows = [row for row in rows if row["phases"]]

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

    def window_names(self, shards=8, shuffle=False):
        """The shard files of every window, `shards` to a window."""
        names = sorted({row["shard"] for row in self.rows})
        if shuffle:
            random.shuffle(names)
        return [names[start : start + shards] for start in range(0, len(names), shards)]

    def read(self, names):
        """The validated runs of these shard files that belong to this part of the data.
        Lines of other runs are skipped unparsed: a held-out part is a tenth of a shard."""
        names = list(names)
        selected = set(names)
        wanted = defaultdict(dict)
        for row in self.rows:
            if row["shard"] in selected:
                wanted[row["shard"]][row["line"]] = row
        runs = []
        for name in names:
            found = 0
            with gzip.open(self.directory / name, "rt", encoding="utf-8") as stream:
                for number, line in enumerate(stream):
                    if number in wanted[name]:
                        run = json.loads(line)
                        row = wanted[name][number]
                        if run["run_id"] != row["run_id"]:
                            raise ProtocolError("Shard does not hold its runs in the order of the index")
                        validate_run(run)
                        if Counter(m["phase"] for m in policy_macros(run)) != row["phases"]:
                            raise ProtocolError("Policy index counts disagree with the shard; rebuild the index")
                        runs.append(run)
                        found += 1
            if found != len(wanted[name]):
                raise ProtocolError("Shard is missing indexed runs; rebuild the index")
        return runs

    def windows(self, shards=8, shuffle=False):
        """Lists of validated runs, each from `shards` shard files."""
        for names in self.window_names(shards, shuffle):
            yield self.read(names)


@dataclass
class Sample:
    """A macro-action to learn from, with what the weighted NLL trainer and its reports
    need of it. `source` is the run's: demonstrations, solver labels and sampled runs are
    the sources the trainer weighs. `macro` is None once a feeder has packed it: the
    fields after it travel on alone."""

    macro: dict | None
    character: str
    run_id: str
    weight: float
    phase: str
    branches: int  # decisions of the macro-action that had a choice
    ascension: int | None
    source: str | None = None
    _lengths: tuple[tuple[int, int], ...] | None = field(
        default=None, repr=False, compare=False
    )


def _sample(run, macro, weight):
    branches = sum(is_policy_frame(step["frame"]) for step in macro["steps"])
    return Sample(macro, run["character"], run["run_id"], weight, macro["phase"], branches,
                  run.get("ascension"), run.get("source"))


def samples(runs, *, counts=None):
    """Weighted samples of the policy macro-actions of `runs`: every (character, phase)
    group weighs the same, and its macro-actions share that weight equally.

    `counts` gives the policy macros per (character, phase) of the whole data set, from
    its index, when `runs` is only a window of it; the weights follow the whole set.
    Shares between the sources of a group are the index's to give through `counts`.
    """
    if counts is None:
        counts = Counter(
            (r["character"], m["phase"]) for r in runs for m in policy_macros(r)
        )
    result = []
    for run in runs:
        validate_run(run)
        for macro in policy_macros(run):
            result.append(_sample(run, macro, 1 / (len(counts) * counts[run["character"], macro["phase"]])))
    return result


def bucket_size(length, buckets):
    return next((b for b in buckets if b >= length), length)


def sample_lengths(item, observe=None):
    """(tokens, action slots) of every branching decision of a sample."""
    if observe is None:
        from .representation import observation as observe

    lengths = []
    for step in item.macro["steps"]:
        if is_policy_frame(step["frame"]):
            obs = observe(step["frame"])
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
    its own forms a batch of its own. Samples are taken shortest first, so that
    where a logical batch must be split the short ones are not padded to the long.
    """
    result, batch, max_n, rows = [], [], 0, 0
    for item in sorted(logical, key=lambda item: sample_pad_size(item, config)):
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
