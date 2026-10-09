"""Five complete A0 flow regressions per character against an isolated build."""

import argparse
import json
import os
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "sts2-cli/python")]
import play_full_run
from model.dotnet_runtime import runtime_command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    engine = args.engine.resolve()
    if not engine.is_file():
        raise FileNotFoundError(engine)
    play_full_run.runtime_command = lambda _default, **_kwargs: runtime_command(engine)
    os.environ["STS2_LIB"] = str(ROOT / "sts2-cli/lib")
    os.environ["STS2_GAME_DIR"] = str(ROOT / "sts2-cli/lib")
    cases = [(c, i) for c in play_full_run.VALID_CHARACTERS for i in range(5)]

    def run(case):
        character, index = case
        return dict(character=character, **play_full_run.play_run(
            f"public-history-a0-{character}-{index}", character, verbose=False, log=False, ascension=0))

    with ThreadPoolExecutor(max_workers=5) as pool:
        rows = list(pool.map(run, cases))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2))
    if any(r.get("error") or r.get("timeout") for r in rows):
        raise RuntimeError(f"Incomplete regression; see {args.output}")
    print(f"PASS: {len(rows)} complete A0 games, five per character, no crashes or unresolved runs.")


if __name__ == "__main__":
    main()
