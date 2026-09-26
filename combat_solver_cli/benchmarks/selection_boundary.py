"""Native regression: replay a captured branch and continue an unplanned selection.

Usage: python -m combat_solver_cli.benchmarks.selection_boundary --branch capture.json
"""
import argparse
import json
import time
from pathlib import Path
from combat_solver_cli.astar import ReplayWorker, from_records, state_key
from combat_solver_cli.client import DEFAULT_CONFIG
from model.protocol import action_semantics


def check(branch, config=DEFAULT_CONFIG):
    data = json.loads(Path(branch).read_text())
    started = time.monotonic()
    worker = ReplayWorker(config, data['character'], data['seed'], 1000,
                          started + 45, 10000, True, data['ascension'])
    try:
        worker.restore(from_records(data['records']))
        if data.get('pending_action'):
            worker.execute(data['pending_action'])
        frame = worker.combat_and_forced()
        assert frame['public']['phase'] == 'card_select', frame['public']['phase']
        candidates = frame['legal']['candidates']
        assert len(candidates) == 5, len(candidates)
        assert worker.solver_steps == 0, 'Selection must return without solving'
        before = state_key(frame)
        worker.execute(action_semantics(candidates[0]))
        worker.settle()
        assert state_key(worker.frame) != before, 'Selection did not advance'
        # Complete the real selection protocol (including confirmation) before solving.
        for _ in range(10):
            if worker.frame['public']['phase'] == 'combat':
                break
            frame = worker.settle()
            if frame['public']['phase'] == 'combat' or frame['boundary'] == 'terminal':
                break
            options = frame['legal']['candidates']
            choice = next((c for c in options if c['verb'] == 'FINISH_SELECTION'), options[0])
            worker.execute(action_semantics(choice))
            worker.settle()
        assert worker.frame['public']['phase'] not in ('card_select', 'card_reward'), 'Selection stuck'
        print(json.dumps(dict(status='passed', seed=data['seed'], candidates=len(candidates),
                              next_phase=worker.frame['public']['phase'],
                              seconds=time.monotonic()-started)))
    finally:
        worker.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--branch', required=True, type=Path)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    check(args.branch, args.config)
