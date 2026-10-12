"""Audited complete-run data. Never invent candidates from a teacher's label."""

import gzip
import json
import math
import os
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
from .dataset_index import INDEX, STUDENT_BATTLE, read_index, row_source, sample_source
from .seeds import seed_group


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


# The share of seeds whose runs are never trained on: about one in fifty, which of the
# imported demonstrations is about a hundred thousand macro-actions, hundreds in every
# (character, phase) group. They are only ever scored (`Learner.evaluate_bootstrap`).
HELDOUT_FRACTION = 0.02


def held_out(run, fraction=HELDOUT_FRACTION):
    """Whether a run's seed falls in the held-out part. The side is a function of the
    seed group alone (the first 32 bits of its fingerprint, as a fraction of 2**32, under
    `fraction`): every run and fragment of one seed lands on the same side, in whatever
    shard it is stored, whatever else the data set holds and however it was imported."""
    from .protocol import fingerprint

    key = seed_group(run)
    return int(fingerprint(key)[:8], 16) / 2**32 < fraction


def split_runs(runs, fraction=HELDOUT_FRACTION):
    """(training, held-out) runs in memory: every seed goes whole to one side, and
    adjacent decisions are never split."""
    train, held = [], []
    for run in runs:
        (held if held_out(run, fraction) else train).append(run)
    return train, held


def macro_count(runs):
    """The policy macro-actions of runs in memory."""
    return sum(1 for run in runs for _ in policy_macros(run))


# The share of a character's combat group's weight that the solver's labels on the
# policy's own fight entries (`student_battle`) take where the group also holds other
# samples: a third at first, rising with the student macro-actions the training data
# holds in all, linearly, to a half at SHARE_RAMP_MACROS and no further. The share is
# the combat phase's: the few selections a student battle holds in another phase
# (`card_select` inside a fight) are counted with the rest of their group. The rule
# and its grounds: ARCHITECTURE.md, 批次整理与数据加载.
SHARE_START, SHARE_CAP, SHARE_RAMP_MACROS = 1 / 3, 1 / 2, 1_000_000
SHARE_PHASES = ("combat",)


def student_share(student_macros):
    """The share of a combat group's weight the student samples take, from how many
    student macro-actions the training data holds in all."""
    if student_macros < 0:
        raise ValueError("A count of macro-actions is not negative")
    return SHARE_START + (SHARE_CAP - SHARE_START) * min(1.0, student_macros / SHARE_RAMP_MACROS)


def group_weights(source_counts, share=None):
    """The weight of one macro-action of each (character, phase, source), from the
    macro-actions of each. Every (character, phase) group weighs 1/G and its
    macro-actions share that equally, except in the combat phase: there the student
    samples share `share` of the group equally and the other sources the rest, equally
    per macro-action, where the group holds both. `share` defaults to `student_share`
    of the student macro-actions counted."""
    groups = defaultdict(Counter)
    for (character, phase, source), n in source_counts.items():
        if n:
            groups[character, phase][source] += n
    if share is None:
        share = student_share(sum(n for (_, _, source), n in source_counts.items() if source == STUDENT_BATTLE))
    if not 0 < share < 1:
        raise ValueError("The student share is a fraction of a group's weight")
    weights = {}
    for group, by_source in groups.items():
        student = by_source.get(STUDENT_BATTLE, 0)
        rest = sum(by_source.values()) - student
        shared = group[1] in SHARE_PHASES and student and rest
        for source in by_source:
            if not shared:
                part, among = 1.0, student + rest
            elif source == STUDENT_BATTLE:
                part, among = share, student
            else:
                part, among = 1 - share, rest
            weights[group + (source,)] = part / (len(groups) * among)
    return weights


def weight_totals(weights, source_counts):
    """The total weight of every (character, phase, source): what a report shows of
    the shares."""
    return {key: weights[key] * n for key, n in source_counts.items() if n}


def source_counts_of(runs):
    """Policy macro-actions per (character, phase, source) of runs in memory."""
    return Counter((r["character"], m["phase"], sample_source(r)) for r in runs for m in policy_macros(r))


class RunShards:
    """Imported runs kept as gzip shards and read a few shards at a time.

    The index holds one small row per run (identity, seed, character, macro counts
    by phase, source, shard), so splits and sample weights need no pass over the
    data. `directory` is one data directory or a list of them, read as one data
    set: an import of demonstrations with the imports of student battles, or the
    rounds of a buffer. A row of a directory after the first remembers which it is
    (`root`); a shard is named by (root, file). The first directory's rows are kept
    as they are read, so the large one goes first: every feeder holds the index.
    """

    def __init__(self, directory, rows=None):
        if isinstance(directory, (str, os.PathLike)):
            directory = [directory]
        self.directories = [Path(d) for d in directory]
        if not self.directories:
            raise ValueError("A data set is at least one directory")
        if rows is None:
            rows = [dict(row, root=root) if root else row
                    for root, path in enumerate(self.directories) for row in read_index(path)]
        self.rows = [row for row in rows if row["phases"]]

    @property
    def directory(self):
        """The directory of a data set that is one directory."""
        if len(self.directories) != 1:
            raise ValueError("The data set spans several directories")
        return self.directories[0]

    @staticmethod
    def is_at(path):
        return (Path(path) / INDEX).is_file()

    def __len__(self):
        return len(self.rows)

    def split(self, fraction=HELDOUT_FRACTION):
        """(training, held-out) parts, by seed (`held_out`), over the same shards."""
        parts = [], []
        for row in self.rows:
            parts[held_out(row, fraction)].append(row)
        return RunShards(self.directories, parts[0]), RunShards(self.directories, parts[1])

    def at_ascension(self, ascension):
        """The runs played at `ascension`, over the same shards."""
        if any("ascension" not in row for row in self.rows):
            raise ProtocolError("The policy index predates ascensions; rebuild it with python -m model reindex")
        return RunShards(self.directories, [row for row in self.rows if row["ascension"] == ascension])

    def combat_from(self, directories):
        """The data set with its fights taken from `directories` alone, which are some
        of its own: a run that holds combat macro-actions stays where it is one of
        theirs, and every run without them stays. A fight is a sample group of its own,
        so what is left of the other directories is their decisions outside combat."""
        own = [path.resolve() for path in self.directories]
        named = [Path(path).resolve() for path in directories]
        if any(path not in own for path in named):
            raise ProtocolError("--combat-from names directories of --data")
        roots = {own.index(path) for path in named}
        return RunShards(self.directories, [row for row in self.rows
                                            if "combat" not in row["phases"] or row.get("root", 0) in roots])

    @property
    def counts(self):
        """Macros per (character, phase): the groups the sample weights are normalized by."""
        counts = Counter()
        for row in self.rows:
            for phase, macros in row["phases"].items():
                counts[row["character"], phase] += macros
        return counts

    @property
    def source_counts(self):
        """Macros per (character, phase, source): what the shares within a group are
        computed from."""
        counts = Counter()
        for row in self.rows:
            source = row_source(row)
            for phase, macros in row["phases"].items():
                counts[row["character"], phase, source] += macros
        return counts

    @property
    def student_macros(self):
        return sum(n for (_, _, source), n in self.source_counts.items() if source == STUDENT_BATTLE)

    @property
    def weights(self):
        """The weight of a macro-action of every (character, phase, source) of this data."""
        return group_weights(self.source_counts)

    @property
    def macros(self):
        return sum(self.counts.values())

    @property
    def seeds(self):
        return {str(row["seed"]) for row in self.rows}

    @property
    def identities(self):
        return {row["run_id"] for row in self.rows}

    def metadata(self):
        from .protocol import fingerprint

        known = {}
        for directory in self.directories:
            known.update(json.loads((directory / "summary.json").read_text())["contracts"])
        return {
            "training_engine_contracts": [known[key] for key in sorted({r["contract"] for r in self.rows})],
            "training_seeds": sorted(self.seeds),
            "training_seed_pool_hash": fingerprint(sorted((r["character"], str(r["seed"])) for r in self.rows)),
        }

    @staticmethod
    def _key(name):
        """A shard's key, (root, file); a bare file name is one of the first directory."""
        return (0, name) if isinstance(name, str) else tuple(name)

    def window_names(self, shards=8, shuffle=False):
        """The shards of every window, `shards` to a window, each as (root, file)."""
        names = sorted({(row.get("root", 0), row["shard"]) for row in self.rows})
        if shuffle:
            random.shuffle(names)
        return [names[start : start + shards] for start in range(0, len(names), shards)]

    def read(self, names):
        """The validated runs of these shards that belong to this part of the data.
        Lines of other runs are skipped unparsed: a held-out part is a tenth of a shard."""
        names = [self._key(name) for name in names]
        # The files asked for of each directory: most of a million rows match none.
        selected = defaultdict(set)
        for root, name in names:
            selected[root].add(name)
        wanted = defaultdict(dict)
        for row in self.rows:
            root = row.get("root", 0)
            if row["shard"] in selected[root]:
                wanted[root, row["shard"]][row["line"]] = row
        runs = []
        for key in names:
            found = 0
            root, name = key
            with gzip.open(self.directories[root] / name, "rt", encoding="utf-8") as stream:
                for number, line in enumerate(stream):
                    if number in wanted[key]:
                        run = json.loads(line)
                        row = wanted[key][number]
                        if run["run_id"] != row["run_id"]:
                            raise ProtocolError("Shard does not hold its runs in the order of the index")
                        validate_run(run)
                        if Counter(m["phase"] for m in policy_macros(run)) != row["phases"]:
                            raise ProtocolError("Policy index counts disagree with the shard; rebuild the index")
                        if sample_source(run) != row_source(row):
                            raise ProtocolError("Policy index source disagrees with the shard; rebuild the index")
                        runs.append(run)
                        found += 1
            if found != len(wanted[key]):
                raise ProtocolError("Shard is missing indexed runs; rebuild the index")
        return runs

    def windows(self, shards=8, shuffle=False):
        """Lists of validated runs, each from `shards` shard files."""
        for names in self.window_names(shards, shuffle):
            yield self.read(names)


def split_held_out(runs, fraction=HELDOUT_FRACTION):
    """(training part, held-out part) of `runs`, a list in memory or RunShards: the
    one rule, `held_out`, whatever form the data takes."""
    if isinstance(runs, RunShards):
        return runs.split(fraction)
    return split_runs(runs, fraction)


def at_ascension(runs, ascension):
    """The runs of `runs` played at `ascension`, a list in memory or RunShards."""
    if isinstance(runs, RunShards):
        return runs.at_ascension(ascension)
    return [run for run in runs if run.get("ascension") == ascension]


def combat_from(runs, directories):
    """`runs` with its fights taken from `directories` alone (RunShards.combat_from)."""
    if not isinstance(runs, RunShards):
        raise ProtocolError("--combat-from tells sharded data directories apart")
    return runs.combat_from(directories)


def extent(runs):
    """How much data `runs` is: its sample groups, policy macro-actions and seeds, and
    of the macro-actions how many are student battles and the share they get."""
    if isinstance(runs, RunShards):
        student = runs.student_macros
        result = {"groups": len(runs), "macros": runs.macros, "seeds": len(runs.seeds)}
    else:
        student = sum(n for (_, _, source), n in source_counts_of(runs).items() if source == STUDENT_BATTLE)
        result = {"groups": len(runs), "macros": macro_count(runs),
                  "seeds": len({str(run["seed"]) for run in runs})}
    return dict(result, student_macros=student, student_share=student_share(student) if student else None)


@dataclass
class Sample:
    """A macro-action to learn from, with what the weighted NLL trainer and its reports
    need of it. `source` is what the weights tell apart (`sample_source`): the run's
    source, or `student_battle` for the solver's labels on the policy's own fights.
    `index` is its place among its run's macro-actions: with `run_id` it names the
    sample, which is how a sampled decision finds its advantage (`model/awr.py`).
    `macro` is None once a feeder has packed it: the fields after it travel on alone."""

    macro: dict | None
    character: str
    run_id: str
    weight: float
    phase: str
    branches: int  # decisions of the macro-action that had a choice
    ascension: int | None
    source: str | None = None
    index: int | None = None
    _lengths: tuple[tuple[int, int], ...] | None = field(
        default=None, repr=False, compare=False
    )


def _sample(run, macro, weight, source, index):
    branches = sum(is_policy_frame(step["frame"]) for step in macro["steps"])
    return Sample(macro, run["character"], run["run_id"], weight, macro["phase"], branches,
                  run.get("ascension"), source, index)


def samples(runs, *, weights=None):
    """Weighted samples of the policy macro-actions of `runs`: every (character, phase)
    group weighs the same; within a group the student battles take their share and the
    rest is shared equally (`group_weights`).

    `weights` is the weight of a macro-action of every (character, phase, source) of
    the whole data set (`RunShards.weights`), when `runs` is only a window of it; the
    weights follow the whole set, not the window.
    """
    if weights is None:
        weights = group_weights(source_counts_of(runs))
    result = []
    for run in runs:
        validate_run(run)
        source = sample_source(run)
        place = {id(macro): index for index, macro in enumerate(run["macros"])}
        for macro in policy_macros(run):
            result.append(_sample(run, macro, weights[run["character"], macro["phase"], source], source,
                                  place[id(macro)]))
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
