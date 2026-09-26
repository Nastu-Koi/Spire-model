"""Resume an interrupted exact-target Bootstrap generation in place."""

import argparse
import json
from pathlib import Path

from .target import resume_target


def main():
    parser = argparse.ArgumentParser(
        description="Resume an exact-target trajectory generation"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = resume_target(args.output)
    print(json.dumps(result, ensure_ascii=False))
    raise SystemExit(0 if result["status"] == "complete" else 2)


if __name__ == "__main__":
    main()
