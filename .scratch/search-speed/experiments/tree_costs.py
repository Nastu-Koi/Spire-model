"""Offline cost evidence; public-state equality is not proof of cache safety."""
import hashlib,json
from collections import Counter
from functools import lru_cache
from pathlib import Path
rows=[]
for index in (1,2):
 root=Path(f'combat_solver_cli/artifacts/unification/extended-seeds/seed-{index:02d}')
 tree=json.loads((root/'tree.json').read_text());prefixes=tree['prefixes']
 @lru_cache(None)
 def history(i):
  if i==-1:return 'root'
  r=prefixes[i]
  return hashlib.sha256(json.dumps([history(r['parent']),r['before_hash'],r['action']],sort_keys=True).encode()).hexdigest()
 combats=[r for r in prefixes if r['actor']=='combat_solver']
 states=Counter(r['before_hash'] for r in combats)
 histories=Counter(history(r['parent']) for r in combats)
 winning=json.loads((root/'winning_prefix.json').read_text())
 prior=json.loads(Path(f'combat_solver_cli/artifacts/unification/performance-holdout/runs/{index:06d}/summary.json').read_text())
 later=json.loads((root/'summary.json').read_text())
 rows.append({'seed_index':index,'total_solver_steps':prior['solver_steps']+later['solver_steps'],
  'winning_solver_steps':sum(r['actor']=='combat_solver' for r in winning['records']),
  'retained_combat_prefix_records':len(combats),'public_state_duplicates':sum(n-1 for n in states.values()),
  'exact_history_duplicates':sum(n-1 for n in histories.values()),
  'total_branch_defeats':prior['deaths']+later['deaths'],
  'total_unresolved_branches':prior['unresolved_branches']+later['unresolved_branches']})
print(json.dumps(rows,indent=2))
