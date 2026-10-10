"""Sampled rounds as training data: the run buffer's shards.

A sampled round is written as the imports are (`model/independent.py`): gzip shards of
runs under `accepted/`, one index row per run, a `summary.json` naming the contract.
`data.RunShards` reads it, with the sample weights and the controller filter the
demonstrations get; the index row of a sampled run also names its policy, its
iteration and its outcome, so that a buffer can keep the rounds of the last few
iterations and reports can count by policy.

What a run keeps is what training reads of it: every decision outside a fight with
its frame, its full legal candidates, the action taken and the log-probability it
was taken with; of each fight only its macro-actions' counts and log-probabilities,
because a fight is taught by the solver from the engine's save at its entry
(`model/saves.py`, `<run>.saves.jsonl.gz` beside `<run>.combats.jsonl.gz`); the
acts' results, the ledger and the outcome. A run that did not finish is not data:
it goes under `quarantine/` with its reason, and a sampled round with one fails.

The sampling processes hand each run over compressed as it ends; this process
appends it to the current shard and holds nothing of the round.
"""

import json
from collections import Counter
from pathlib import Path

from .control import CONTROL_VERSION
from .dataset_index import ShardWriter, index_row
from .protocol import ProtocolError
from .rewards import REWARD_VERSION
from .rollout import OUTSIDE_FRAMES, assign
from .evaluation import evaluate_runs

ROUND_FORMAT = "rollout-round-v1"


class RoundWriter:
    """The files of one sampled round: the shards and index of its finished runs, the
    quarantine of the rest, and the summary once the round is over."""

    def __init__(self, directory, shard_samples=2000):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.shards = ShardWriter(self.directory, shard_samples, flush=True)
        self.runs, self.quarantine = [], {}
        self.counts = Counter()
        self.totals = Counter({key: 0 for key in (
            "accepted_runs", "quarantined_runs", "wins", "macros", "fight_macros", "stored_steps",
            "automatic_steps", "fights", "saves_failed", "fights_from_event")})

    def add(self, job, run):
        """A run as its sampling process handed it over: its summary and, as `member`,
        the compressed line it makes in a shard."""
        _, _, path = job
        name = path.stem
        member = run["member"]
        if run["status"] != "complete":
            directory = self.directory / "quarantine"
            directory.mkdir(exist_ok=True)
            (directory / (name + ".json.gz")).write_bytes(member)
            self.quarantine[name] = run.get("error") or run["status"]
            self.totals["quarantined_runs"] += 1
            return
        self.shards.write(member, index_row(run, phases=run["phases"]), run["stored_steps"], run["contract"])
        for phase, macros in run["phases"].items():
            self.counts[run["character"] + "/" + phase] += macros
        self.totals["accepted_runs"] += 1
        self.totals["macros"] += sum(run["phases"].values())
        self.totals["fight_macros"] += sum(macro["fight"] is not None for macro in run["macros"])
        self.totals["stored_steps"] += run["stored_steps"]
        self.totals["automatic_steps"] += run["automatic_steps"]
        self.totals["wins"] += bool(run["victory"])
        saves = run.get("saves") or {}
        self.totals["fights"] += saves.get("fights", 0)
        self.totals["saves_failed"] += saves.get("failed", 0)
        self.totals["fights_from_event"] += saves.get("from_event", 0)
        # What the round's report reads of a run; the rest of the summary is let go.
        self.runs.append({key: run[key] for key in ("status", "victory", "character", "ascension", "ledger", "climb")})

    def close(self, runner, seeds_path="metadata/seeds.json"):
        """Close the shards and write the summary; returns it."""
        self.shards.close()
        summary = dict(
            kind="rollout_round", format=ROUND_FORMAT,
            iteration=runner.iteration, policy_version=runner.version, policy_digest=runner.digest,
            temperature=dict(runner.temperatures), ascension=runner.ascension, precision=runner.precision,
            vocabulary_hash=runner.vocabulary.digest,
            control_version=CONTROL_VERSION, reward_version=REWARD_VERSION,
            runs=self.totals["accepted_runs"] + self.totals["quarantined_runs"],
            **dict(self.totals), quarantine=dict(self.quarantine),
            counts=dict(sorted(self.counts.items())),
            shards=len(self.shards.shards), shard_samples=self.shards.limit,
            contracts=self.shards.contracts, seeds=seeds_path,
            report=evaluate_runs(self.runs) if self.runs else None,
        )
        path = self.directory / "summary.json"
        stage = path.with_suffix(".tmp")
        stage.write_text(json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
                         encoding="utf-8")
        stage.replace(path)
        return summary


def sample_round(samplers, runner, seeds, directory, *, journals=False, shard_samples=2000, on_run=None):
    """Sample one round into `directory` as the buffer reads it. The round is played
    by the engines of `samplers` (`sampling.Samplers`) at the runner's temperatures;
    every run is appended to the shards as it ends. With `journals` each run's journal
    (every frame with its reward) is written beside its fights as well. Returns the
    summary; a run that did not finish fails the round after the summary is written."""
    jobs = assign(seeds, directory)
    writer = RoundWriter(directory, shard_samples)

    def arrived(index, run):
        writer.add(jobs[index], run)
        if on_run:
            on_run(index, run)

    try:
        samplers.collect(runner, jobs, sample=True, on_run=arrived,
                         layout=dict(OUTSIDE_FRAMES, journal=journals))
    finally:
        # A round that failed still leaves readable shards and index of what finished.
        writer.shards.close()
    summary = writer.close(runner)
    if writer.quarantine:
        raise ProtocolError(
            "Round incomplete; a sampled round must finish every run. Inspect: "
            + ", ".join(f"{name} ({reason})" for name, reason in writer.quarantine.items())
        )
    return summary
