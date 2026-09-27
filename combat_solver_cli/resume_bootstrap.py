"""Resume every unfinished random-seed job from a previous batch checkpoint."""

import argparse
import json
from pathlib import Path

from model.protocol import fingerprint

from .batch import generate
from .client import DEFAULT_CONFIG


def resumable_jobs(previous):
    previous = Path(previous)
    manifest = json.loads((previous / "manifest.json").read_text())
    if manifest.get("mode") != "mcts_batch_v1":
        raise ValueError("Previous batch manifest is incompatible with MCTS resume")
    jobs = []
    for job in manifest["jobs"]:
        character, seed = job["character"], str(job["seed"])
        directory = previous / "jobs" / (character + "-" + fingerprint(seed)[:16])
        tree = directory / "tree.json"
        summary = directory / "summary.json"
        finished = (
            summary.is_file()
            and json.loads(summary.read_text()).get("status") == "verified_victory"
        )
        if tree.is_file() and not finished:
            jobs.append(
                {
                    "character": character,
                    "seed": seed,
                    "resume": str(tree.resolve()),
                }
            )
    if not jobs:
        raise ValueError("Previous batch has no resumable MCTS tree checkpoints")
    return jobs, manifest


def main():
    parser = argparse.ArgumentParser(
        description="Resume unfinished Bootstrap trajectory jobs"
    )
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--workers", type=int, choices=[1], default=1)
    parser.add_argument("--search-lanes", type=int, choices=[1, 2, 4])
    parser.add_argument("--budget-ms", type=int)
    parser.add_argument("--max-expansions", type=int)
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--rollout-decisions", type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--progress-interval", type=float, default=10)
    parser.add_argument(
        "--reuse-turn-plan", action=argparse.BooleanOptionalAction, default=None
    )
    args = parser.parse_args()
    jobs, manifest = resumable_jobs(args.previous)
    prior = manifest["options"]
    if not isinstance(prior, dict):
        raise TypeError("Previous batch options are invalid")
    options = dict(prior)
    for key in (
        "search_lanes",
        "budget_ms",
        "max_expansions",
        "max_seconds",
        "max_steps",
        "rollout_decisions",
        "reuse_turn_plan",
    ):
        value = getattr(args, key)
        if value is not None:
            options[key] = value
    options.update(
        config=args.config or Path(manifest.get("config", DEFAULT_CONFIG)),
        jobs=jobs,
        output=args.output,
        workers=args.workers,
        progress=not args.quiet,
        progress_interval=args.progress_interval,
    )
    result = generate(**options)
    print(json.dumps(result, ensure_ascii=False))
    raise SystemExit(0 if result["verified_victories"] else 2)


if __name__ == "__main__":
    main()
