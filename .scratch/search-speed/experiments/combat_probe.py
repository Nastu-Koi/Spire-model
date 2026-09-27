"""Diagnostic only: replay a real winning prefix and re-solve one fixed combat."""
import argparse
import json
import time
from pathlib import Path
from unittest.mock import patch
from combat_solver_cli.client import DEFAULT_CONFIG, SolverEngine
from combat_solver_cli.search_support import ReplayWorker, from_records, player, state_key

p=argparse.ArgumentParser()
p.add_argument('--seed-index',type=int,default=1)
p.add_argument('--act',type=int,default=1)
p.add_argument('--combat-index',type=int,default=0)
p.add_argument('--budget-ms',type=int,default=1000)
p.add_argument('--repeats',type=int,default=2)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--max-combat-seconds',type=float)
p.add_argument('--expected-after-hash')
a=p.parse_args()
d=json.loads(Path(f'combat_solver_cli/artifacts/unification/extended-seeds/seed-{a.seed_index:02d}/winning_prefix.json').read_text())
starts=[];prev=None
for i,r in enumerate(d['records']):
 key=(r['act'],r['floor'])
 if r['actor']=='combat_solver' and r['act']==a.act and key!=prev:
  starts.append(i);prev=key
idx=starts[a.combat_index]; rows=[]
original=SolverEngine.step
for repeat in range(a.repeats):
 metrics=[]
 def measured(self,frame,**kw):
  start=time.monotonic();res=original(self,frame,**kw)
  metrics.append({'seconds':time.monotonic()-start,'search':res.get('search') is not None,'type':res.get('type')})
  return res
 w=ReplayWorker(DEFAULT_CONFIG,d['character'],d['seed'],a.budget_ms,time.monotonic()+180,10000,True,0,5000)
 try:
  start=time.monotonic();before=w.restore(from_records(d['records'][:idx]));restore=time.monotonic()-start
  assert state_key(before)==d['records'][idx]['before_hash']
  before_player=player(before)
  start=time.monotonic()
  with patch.object(SolverEngine,'step',measured): after=w.combat_and_forced()
  elapsed=time.monotonic()-start
  row={'repeat':repeat,'seed':d['seed'],'index':idx,'act':a.act,'floor':d['records'][idx]['floor'],'before_hash':state_key(before),'budget_ms':a.budget_ms,'restore_seconds':restore,'combat_seconds':elapsed,'before_hp':before_player.get('hp'),'after_hp':player(after).get('hp',w.last_player.get('hp')),'after_phase':after['public']['phase'],'boundary':after['boundary'],'after_hash':state_key(after),'calls':metrics}
  rows.append(row);print(json.dumps(row),flush=True)
 finally:w.close()
a.output.write_text(json.dumps(rows,indent=2)+'\n')

if a.max_combat_seconds is not None:
    assert all(r['combat_seconds'] <= a.max_combat_seconds for r in rows), 'Combat latency exceeds diagnostic threshold'
if a.expected_after_hash is not None:
    assert all(r['after_hash'] == a.expected_after_hash for r in rows), 'Combat quality/state changed'
