"""Frozen dev/holdout comparisons for the single MCTS + CombatSolver backend."""

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path

from model.data import load_runs
from model.protocol import CHARACTERS

from .client import DEFAULT_CONFIG, configuration
from .mcts import search
from .search_support import write_json

PROFILE_OPTIONS = {
    "budget_ms",
    "boss_budget_ms",
    "search_lanes",
    "reuse_turn_plan",
    "rollout_decisions",
    "exploration",
    "rollout_epsilon",
    "local_repair",
    "risk_aware_rollout",
}


def manifest_data(manifest):
    data = (
        json.loads(Path(manifest).read_text())
        if isinstance(manifest, (str, Path))
        else manifest
    )
    if not isinstance(data, dict) or data.get("schema") != "search-evaluation-v1":
        raise ValueError(
            "Expected search-evaluation-v1 manifest; public-policy manifests are historical"
        )
    if set(data) != {
        "schema",
        "character",
        "ascension",
        "seeds",
        "profiles",
        "repetitions",
        "budget",
    }:
        raise ValueError("Unexpected or missing manifest fields")
    if (
        data["character"] not in CHARACTERS
        or type(data["ascension"]) is not int
        or not 0 <= data["ascension"] <= 10
    ):
        raise ValueError("Invalid character or ascension")
    seeds = data["seeds"]
    if not isinstance(seeds, dict) or set(seeds) != {"dev", "holdout"}:
        raise ValueError("Specify disjoint dev and holdout seeds")
    for values in seeds.values():
        if (
            not isinstance(values, list)
            or not values
            or any(not isinstance(s, str) or not s for s in values)
            or len(set(values)) != len(values)
        ):
            raise ValueError("Each split needs unique nonempty seed strings")
    if set(seeds["dev"]) & set(seeds["holdout"]):
        raise ValueError("Development and holdout seeds overlap")
    if type(data["repetitions"]) is not int or data["repetitions"] < 1:
        raise ValueError("repetitions must be positive")
    budget = data["budget"]
    if not isinstance(budget, dict) or set(budget) != {
        "max_seconds_per_run",
        "max_expansions",
        "max_steps",
    }:
        raise ValueError("Specify per-run seconds, expansions and steps")
    for key, value in budget.items():
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or value <= 0
            or (key != "max_seconds_per_run" and type(value) is not int)
        ):
            raise ValueError("Invalid evaluation budget")
    profiles = data["profiles"]
    if not isinstance(profiles, list) or not profiles:
        raise ValueError("Specify profiles")
    names = set()
    for profile in profiles:
        if not isinstance(profile, dict) or set(profile) != {"name", "options"}:
            raise ValueError("Each profile needs name and options")
        name, options = profile["name"], profile["options"]
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("Profile names must be nonempty and unique")
        names.add(name)
        if not isinstance(options, dict) or options.keys() - PROFILE_OPTIONS:
            raise ValueError("Unknown MCTS profile option")
        for key, value in options.items():
            if key in {"reuse_turn_plan", "local_repair", "risk_aware_rollout"}:
                valid = type(value) is bool
            elif key == "search_lanes":
                valid = type(value) is int and value in (1, 2, 4)
            elif key in {"budget_ms", "boss_budget_ms"}:
                valid = type(value) is int and 1 <= value <= 120000
            elif key == "rollout_decisions":
                valid = type(value) is int and value >= 0
            else:
                valid = (
                    type(value) in (int, float)
                    and math.isfinite(value)
                    and (0 <= value <= 1 if key == "rollout_epsilon" else value > 0)
                )
            if not valid:
                raise ValueError(f"Invalid profile option: {key}")
    return data


def source_hashes(config):
    pinned = configuration(config)
    package = Path(__file__).parent
    paths = list(package.glob("*.py")) + list(package.glob("*.cs"))
    result = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
    for key in ("solver_dll", "game_dll", "worker_dll"):
        path = Path(pinned[key])
        result[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    worker = Path(pinned["worker_dll"])
    for name in ("Sts2Headless.dll", "GodotSharp.dll"):
        result[name] = hashlib.sha256(worker.with_name(name).read_bytes()).hexdigest()
    for directory in pinned["dependency_dirs"]:
        for path in sorted(Path(directory).glob("*.dll")):
            result[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def verified_run(directory, character, seed, ascension):
    runs = load_runs(directory / "accepted.jsonl")
    if len(runs) != 1:
        raise ValueError("Expected one independently verified trajectory")
    run = runs[0]
    provenance = run.get("provenance", {})
    if (
        (run["character"], run["seed"], run["ascension"])
        != (character, seed, ascension)
        or run["source"] != "recorder_bc"
        or run["teacher_visibility"] != "unverified"
        or provenance.get("verified_outcome") != f"A{ascension}_final_boss_victory"
        or provenance.get("verified_by") != "fresh_native_seed_replay"
        or provenance.get("bosses") != [1, 2, 3]
        or provenance.get("raw_sha256")
        != hashlib.sha256((directory / "verified_trace.jsonl").read_bytes()).hexdigest()
    ):
        raise ValueError("Missing or inconsistent native victory evidence")
    return run


def evaluate(manifest, split, output, *, config=DEFAULT_CONFIG, search_fn=None):
    data = manifest_data(manifest)
    if split not in data["seeds"]:
        raise ValueError("Unknown seed split")
    search_fn = search_fn or search
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    started = time.monotonic()
    hashes = source_hashes(config)
    output.mkdir(parents=True)
    write_json(output / "manifest.json", data)
    runs, accepted = [], []
    # Alternate profile order across cases to reduce systematic timing order bias.
    for repetition in range(data["repetitions"]):
        for index, seed in enumerate(data["seeds"][split]):
            profiles = (
                data["profiles"]
                if (index + repetition) % 2 == 0
                else list(reversed(data["profiles"]))
            )
            for profile in profiles:
                directory = output / "runs" / f"{len(runs) + 1:06d}"
                begin = time.monotonic()
                try:
                    result = search_fn(
                        config,
                        data["character"],
                        seed,
                        directory,
                        ascension=data["ascension"],
                        max_seconds=data["budget"]["max_seconds_per_run"],
                        max_expansions=data["budget"]["max_expansions"],
                        max_steps=data["budget"]["max_steps"],
                        **profile["options"],
                    )
                    if result["status"] == "verified_victory":
                        try:
                            accepted.append(
                                verified_run(
                                    directory,
                                    data["character"],
                                    seed,
                                    data["ascension"],
                                )
                            )
                        except (OSError, ValueError, KeyError, RuntimeError) as exc:
                            result = {
                                **result,
                                "status": "replay_failed",
                                "verification_error": str(exc),
                            }
                except (OSError, ValueError, KeyError, RuntimeError) as exc:
                    result = {"status": "infrastructure_error", "error": str(exc)}
                elapsed = time.monotonic() - begin
                directory.mkdir(parents=True, exist_ok=True)
                row = {
                    "profile": profile["name"],
                    "seed": seed,
                    "repetition": repetition + 1,
                    "run_dir": str(directory.relative_to(output)),
                    "result": result,
                    "wall_seconds": elapsed,
                    "budget_overrun_seconds": max(
                        0.0, elapsed - data["budget"]["max_seconds_per_run"]
                    ),
                }
                write_json(directory / "evaluation.json", row)
                runs.append(row)
                write_json(output / "progress.json", runs)
    # Repeated profiles may share run identities. Keep per-case verified exports;
    # evaluation repetitions must not silently become duplicated training data.
    write_json(
        output / "accepted_paths.json",
        [
            str(Path(row["run_dir"]) / "accepted.jsonl")
            for row in runs
            if row["result"]["status"] == "verified_victory"
        ],
    )
    after = source_hashes(config)
    counts = Counter(row["result"]["status"] for row in runs)
    profiles = []
    for profile in data["profiles"]:
        subset = [row for row in runs if row["profile"] == profile["name"]]
        successes = sum(row["result"]["status"] == "verified_victory" for row in subset)
        measured = sum(row["wall_seconds"] for row in subset)
        profiles.append(
            {
                "name": profile["name"],
                "attempts": len(subset),
                "verified_victories": successes,
                "counts": dict(Counter(row["result"]["status"] for row in subset)),
                "run_wall_seconds": measured,
                "run_wall_seconds_per_verified_victory": measured / successes
                if successes
                else None,
            }
        )
    report = {
        "schema": "search-evaluation-report-v1",
        "algorithm": "mcts",
        "split": split,
        "independent_seeds": len(data["seeds"][split]),
        "attempts": len(runs),
        "counts": dict(counts),
        "verified_victories": len(accepted),
        "profile_summaries": profiles,
        "runs": runs,
        "source_hashes": hashes,
        "source_hash_mismatch": hashes != after,
        "manifest_sha256": hashlib.sha256(
            (output / "manifest.json").read_bytes()
        ).hexdigest(),
        "wall_seconds": 0.0,
        "wall_seconds_per_verified_victory": None,
    }
    # Include initial report serialization and accepted export in the timed work.
    write_json(output / "report.json", report)
    report["wall_seconds"] = time.monotonic() - started
    report["wall_seconds_per_verified_victory"] = (
        report["wall_seconds"] / len(accepted) if accepted else None
    )
    write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--split", choices=("dev", "holdout"), required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate(args.manifest, args.split, args.output, config=args.config),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
