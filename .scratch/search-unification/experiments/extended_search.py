"""Resume two existing trees with extended seed budgets; retain cumulative cost."""
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from combat_solver_cli.client import DEFAULT_CONFIG
from combat_solver_cli.evaluate import source_hashes, verified_run
from combat_solver_cli.mcts import search

ROOT = Path('combat_solver_cli/artifacts/unification')
OUT = ROOT / 'extended-seeds'

def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')

def run(index):
    prior = ROOT / 'performance-holdout' / 'runs' / f'{index:06d}'
    prior_record = json.loads((prior / 'evaluation.json').read_text())
    seed = prior_record['seed']
    output = OUT / f'seed-{index:02d}'
    started = time.monotonic()
    try:
        result = search(DEFAULT_CONFIG, 'Ironclad', seed, output,
            resume=prior/'tree.json', ascension=0, max_seconds=1800,
            max_expansions=100000, max_steps=10000, search_lanes=1,
            budget_ms=1000, boss_budget_ms=5000, reuse_turn_plan=True,
            rollout_decisions=256, exploration=2**0.5, rollout_epsilon=0.1,
            local_repair=True, risk_aware_rollout=True)
        if result['status'] == 'verified_victory':
            verified_run(output, 'Ironclad', seed, 0)
    except Exception as exc:
        result = {'status':'experiment_error','error':repr(exc)}
    elapsed = time.monotonic()-started
    record = {'seed':seed,'prior_wall_seconds':prior_record['wall_seconds'],
        'extension_wall_seconds':elapsed,
        'cumulative_seed_wall_seconds':prior_record['wall_seconds']+elapsed,
        'result':result}
    write(OUT/f'seed-{index:02d}-result.json',record)
    print(json.dumps(record),flush=True)
    return record

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=False)
    hashes=source_hashes(DEFAULT_CONFIG)
    write(OUT/'plan.json',{'seeds':[f'UNIFICATION-HOLDOUT-{i:02d}' for i in (1,2)],
        'additional_seconds_per_seed':1800,'parallel_seeds':2,
        'purpose':'feasibility, not holdout performance acceptance',
        'source_hashes':hashes})
    started=time.monotonic()
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows=list(pool.map(run,(1,2)))
    write(OUT/'report.json',{'runs':rows,'extension_batch_wall_seconds':time.monotonic()-started,
        'verified_victories':sum(r['result']['status']=='verified_victory' for r in rows),
        'source_hash_mismatch':hashes!=source_hashes(DEFAULT_CONFIG)})
