"""Fixed-seed native timing harness; total_seconds includes independent verification."""
import argparse
from collections import defaultdict
from functools import wraps
import json
from pathlib import Path
from time import perf_counter
from combat_solver_cli import astar
from combat_solver_cli.client import SolverEngine, DEFAULT_CONFIG

parser=argparse.ArgumentParser()
parser.add_argument('--output',required=True,type=Path)
parser.add_argument('--seconds',type=float,default=180)
parser.add_argument('--budget',type=int,default=1000)
parser.add_argument('--boss-budget',type=int,default=5000)
parser.add_argument('--early-route-diversity',action=argparse.BooleanOptionalAction,default=True)
parser.add_argument('--rollout',type=int,default=256)
parser.add_argument('--lanes',type=int,choices=(1,2,4),default=1)
parser.add_argument('--prefix',type=Path)
parser.add_argument('--resume',type=Path)
parser.add_argument('--require-victory',action='store_true')
parser.add_argument('--expansions',type=int,default=10000)
parser.add_argument('--reuse',action=argparse.BooleanOptionalAction,default=True)
args=parser.parse_args()
metrics=defaultdict(lambda:dict(calls=0,seconds=0.))
def wrap(cls,name,label=None):
 original=getattr(cls,name)
 @wraps(original)
 def measured(*a,**kw):
  start=perf_counter()
  try:return original(*a,**kw)
  finally:
   entry=metrics[label or name]
   entry['calls']+=1;entry['seconds']+=perf_counter()-start
 setattr(cls,name,measured)
for name in ('restore','execute','combat_and_forced'):
 wrap(astar.ReplayWorker,name)
wrap(SolverEngine,'step','solver_step')
wrap(astar,'heuristic')
wrap(astar,'preference')
wrap(astar,'save_frontier')
start=perf_counter()
try:
 result=astar.search(DEFAULT_CONFIG,'Ironclad','7E4A91CDAE1225F0',args.output,
   ascension=0,budget_ms=args.budget,boss_budget_ms=args.boss_budget,early_route_diversity=args.early_route_diversity,weight=12,max_expansions=args.expansions,max_seconds=args.seconds,prefix_path=args.prefix,resume=args.resume,
   reuse_turn_plan=args.reuse,rollout_decisions=args.rollout,search_lanes=args.lanes)
finally:
 args.output.mkdir(parents=True,exist_ok=True)
 report=dict(total_seconds=perf_counter()-start,timings=dict(metrics))
 (args.output/'profile.json').write_text(json.dumps(report,indent=2))
 print(json.dumps(report))
print(json.dumps(result))

if args.require_victory and result['status'] != 'verified_victory':
 raise SystemExit(2)
