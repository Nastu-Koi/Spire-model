"""Generate random or reproducible seed batches and export only replay-verified Bootstrap traces."""

import argparse
import json
import secrets
from pathlib import Path

from model.protocol import CHARACTERS

from .batch import generate
from .client import DEFAULT_CONFIG
from .target import generate_target


def make_jobs(seeds_per_character, characters=CHARACTERS, seed_prefix=None):
    if seeds_per_character < 1:
        raise ValueError("seeds_per_character must be positive")
    if seed_prefix:
        seeds = [f"{seed_prefix}-{index:06d}" for index in range(seeds_per_character)]
    else:
        seeds = []
        while len(seeds) < seeds_per_character:
            seed = secrets.token_hex(8).upper()
            if seed not in seeds:
                seeds.append(seed)
    return [
        {"character": character, "seed": seed}
        for seed in seeds
        for character in characters
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Batch-generate independently verified Bootstrap trajectories"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--ascension", type=int, choices=range(11), default=0)
    parser.add_argument(
        "--characters", nargs="+", choices=CHARACTERS, default=list(CHARACTERS)
    )
    parser.add_argument(
        "--seed-prefix", help="Use deterministic seeds for a reproducible experiment"
    )
    count = parser.add_mutually_exclusive_group()
    count.add_argument("--target-trajectories", type=int)
    count.add_argument("--seeds-per-character", type=int)
    parser.add_argument("--workers", type=int, choices=[1], default=1)
    parser.add_argument("--search-lanes", type=int, choices=[1, 2, 4], default=2)
    parser.add_argument("--budget-ms", type=int, default=2500)
    parser.add_argument("--weight", type=float, default=12)
    parser.add_argument("--max-expansions", type=int, default=2000)
    parser.add_argument("--max-seconds", type=float, default=900)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--rollout-decisions", type=int, default=256)
    parser.add_argument(
        "--reuse-turn-plan", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    if args.seeds_per_character is not None:
        jobs = make_jobs(args.seeds_per_character, args.characters, args.seed_prefix)
        options = vars(args).copy()
        options.pop("seed_prefix")
        options.pop("seeds_per_character")
        options.pop("target_trajectories")
        options.pop("characters")
        result = generate(jobs=jobs, **options)
        success = bool(result["verified_victories"])
    else:
        target = (
            args.target_trajectories if args.target_trajectories is not None else 100
        )
        options = vars(args).copy()
        for key in (
            "seed_prefix",
            "seeds_per_character",
            "target_trajectories",
            "characters",
        ):
            options.pop(key)
        if args.seed_prefix:
            raise ValueError(
                "--seed-prefix is only available with --seeds-per-character; target mode uses one random seed per trajectory"
            )
        result = generate_target(
            target_trajectories=target, characters=args.characters, **options
        )
        success = result["status"] == "complete"
    print(json.dumps(result, ensure_ascii=False))
    raise SystemExit(0 if success else 2)


if __name__ == "__main__":
    main()
