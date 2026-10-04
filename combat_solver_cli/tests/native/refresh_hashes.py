"""Re-record the public-state hashes of the captured prefixes.

`before_hash` fingerprints the whole public state, so any change to what the
engine publishes invalidates it. Replay each prefix on the current engine and
store the hashes it produces; the recorded actions themselves are not changed.
A prefix whose actions no longer replay is reported and left for a new capture.

    python combat_solver_cli/tests/native/refresh_hashes.py [prefix.json ...]
"""

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from combat_solver_cli.client import DEFAULT_CONFIG, SolverEngine  # noqa: E402
from combat_solver_cli.search_support import (  # noqa: E402
    ReplayMismatch,
    collapse_noncombat_cancels,
    resolve,
    state_key,
)
from model.protocol import execution_command  # noqa: E402


def settle(engine, frame):
    deadline = time.monotonic() + 10
    while frame.get("boundary") == "waiting" and time.monotonic() < deadline:
        time.sleep(0.01)
        frame = engine.send({"cmd": "advance_to_boundary"})
    return frame


def refresh(path, config):
    data = json.loads(path.read_text())
    changed = 0
    with SolverEngine(config) as engine:
        frame = engine.reset(data["character"], data["seed"], data.get("ascension", 0))
        # Non-combat cancel excursions are not offered any more; their records
        # stay in the file for the tests which index it, but are not replayed.
        kept = collapse_noncombat_cancels(data["records"])
        for index, record in enumerate(kept):
            frame = settle(engine, frame)
            if frame.get("boundary") != "decision":
                return f"stops at replayed record {index}: boundary {frame.get('boundary')}"
            key = state_key(frame)
            changed += record.get("before_hash") != key
            record["before_hash"] = key
            try:
                candidate = resolve(frame, record["action"])
            except ReplayMismatch:
                return f"diverges at replayed record {index} ({frame['public']['phase']})"
            frame = engine.send(execution_command(frame, candidate["candidate_ref"]))
    path.write_text(json.dumps(data))
    return f"{len(kept)} records replayed, {changed} hashes updated"


def main(argv):
    config = os.environ.get("COMBAT_SOLVER_CONFIG", DEFAULT_CONFIG)
    here = Path(__file__).resolve().parent
    paths = [Path(p) for p in argv] or sorted(here.glob("*/*-prefix.json"))
    failed = False
    for path in paths:
        result = refresh(path, config)
        failed |= "records replayed" not in result
        print(f"{path.name}: {result}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
