"""Read completed frozen reports and journals; never mutate a running search."""
import argparse
import json
from collections import Counter
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('output', type=Path)
args = parser.parse_args()
report = json.loads((args.output / 'report.json').read_text())
rows = []
for row in report['runs']:
    directory = args.output / row['run_dir']
    events = [json.loads(line) for line in (directory / 'search.jsonl').read_text().splitlines()]
    result = row['result']
    rows.append({
        'seed': row['seed'], 'profile': row['profile'],
        'status': result['status'], 'wall_seconds': row['wall_seconds'],
        'max_journal_progress': max(([e.get('act', 0) or 0, e.get('floor', 0) or 0] for e in events), default=[0, 0]),
        'terminal_events': dict(Counter(e['result'] for e in events if 'result' in e)),
        'error_events': [e for e in events if 'error' in e],
        'expanded': result.get('expanded'), 'simulations': result.get('simulations'),
        'solver_steps': result.get('solver_steps'), 'replayed_steps': result.get('replayed_steps'),
        'search_seconds': result.get('search_seconds'),
        'post_search_seconds': row['wall_seconds'] - result.get('search_seconds', row['wall_seconds']),
    })
print(json.dumps({'counts':report['counts'], 'wall_seconds':report['wall_seconds'],
    'cost_per_success':report['wall_seconds_per_verified_victory'],
    'source_hash_mismatch':report['source_hash_mismatch'], 'runs':rows}, indent=2))
