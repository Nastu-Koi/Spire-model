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
    jobs = []
    for job in manifest["jobs"]:
        character, seed = job["character"], str(job["seed"])
        directory = previous / "jobs" / (character + "-" + fingerprint(seed)[:16])
        frontier = directory / "frontier.json"
        summary = directory / "summary.json"
        finished = (
            summary.is_file()
            and json.loads(summary.read_text()).get("status") == "verified_victory"
        )
        if frontier.is_file() and not finished:
            jobs.append(
                {
                    "character": character,
                    "seed": seed,
                    "resume": str(frontier.resolve()),
                }
            )
    if not jobs:
        raise ValueError("Previous batch has no resumable frontier checkpoints")
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
    parser.add_argument("--weight", type=float)
    parser.add_argument("--max-expansions", type=int, default=4000)
    parser.add_argument("--max-seconds", type=float, default=1200)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--rollout-decisions", type=int)
    parser.add_argument(
        "--reuse-turn-plan", action=argparse.BooleanOptionalAction, default=None
    )
    args = parser.parse_args()
    jobs, manifest = resumable_jobs(args.previous)
    prior = manifest["options"]
    options = {
        "config": args.config or Path(manifest.get("config", DEFAULT_CONFIG)),
        "jobs": jobs,
        "output": args.output,
        "workers": args.workers,
        "ascension": prior.get("ascension", 10),
        "search_lanes": (
            args.search_lanes
            if args.search_lanes is not None
            else prior.get("search_lanes", 2)
        ),
        "budget_ms": args.budget_ms
        if args.budget_ms is not None
        else prior.get("budget_ms", 2500),
        "weight": args.weight if args.weight is not None else prior.get("weight", 12),
        "max_expansions": args.max_expansions,
        "max_seconds": args.max_seconds,
        "max_steps": args.max_steps,
        "rollout_decisions": (
            args.rollout_decisions
            if args.rollout_decisions is not None
            else prior.get("rollout_decisions", 256)
        ),
        "reuse_turn_plan": (
            args.reuse_turn_plan
            if args.reuse_turn_plan is not None
            else prior.get("reuse_turn_plan", True)
        ),
    }
    result = generate(**options)
    print(json.dumps(result, ensure_ascii=False))
    raise SystemExit(0 if result["verified_victories"] else 2)


if __name__ == "__main__":
    main()
