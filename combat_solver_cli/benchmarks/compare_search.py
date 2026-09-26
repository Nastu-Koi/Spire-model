"""Run one controlled A*/MCTS trial; repeat in alternating order for comparison."""
import argparse
from collections import defaultdict
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import resource
import threading
import time

from combat_solver_cli import astar
from combat_solver_cli.client import DEFAULT_CONFIG, SolverEngine, configuration


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--algorithm',choices=('astar','mcts'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seconds',type=float,default=900)
    parser.add_argument('--seed',default='7E4A91CDAE1225F0')
    parser.add_argument('--lanes',type=int,choices=(1,2,4),default=4)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--local-repair',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--risk-aware-rollout',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--cpus',default='0,1,2,3')
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    cpus={int(x) for x in args.cpus.split(',')}
    os.sched_setaffinity(0,cpus)
    os.environ['DOTNET_PROCESSOR_COUNT']=str(len(cpus))
    fn=astar.search
    if args.algorithm=='mcts':
        from combat_solver_cli.mcts import search
        fn=search
    metrics=defaultdict(lambda:dict(calls=0,seconds=0.))
    lock=threading.Lock()
    def wrap(cls,name):
        original=getattr(cls,name)
        @wraps(original)
        def measured(*a,**kw):
            started=time.monotonic()
            try:
                result=original(*a,**kw)
                if name=='step' and isinstance(result,dict):
                    category='native_search' if result.get('search') is not None else 'plan_or_selection'
                    with lock: metrics[category]['calls']+=1
                return result
            finally:
                with lock:
                    metrics[name]['calls']+=1
                    metrics[name]['seconds']+=time.monotonic()-started
        setattr(cls,name,measured)
    for name in ('restore','combat_and_forced'): wrap(astar.ReplayWorker,name)
    wrap(SolverEngine,'step')
    options=dict(ascension=0,budget_ms=1000,boss_budget_ms=5000,weight=12,
                 max_expansions=100000,max_seconds=args.seconds,max_steps=10000,
                 reuse_turn_plan=True,rollout_decisions=256,search_lanes=args.lanes,
                 early_route_diversity=True)
    if args.algorithm == 'mcts':
        options.update(local_repair=args.local_repair,risk_aware_rollout=args.risk_aware_rollout)
    hashes={}
    for name in ('astar.py','mcts.py','mcts_lanes.py','lanes.py','diversity.py','SolverAdapter.cs'):
        p=Path('combat_solver_cli')/name
        if p.exists(): hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest()
    config=configuration(DEFAULT_CONFIG)
    hashes['worker_dll']=hashlib.sha256(Path(config['worker_dll']).read_bytes()).hexdigest()
    before=resource.getrusage(resource.RUSAGE_CHILDREN)
    start=time.monotonic();start_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    result=fn(DEFAULT_CONFIG,'Ironclad',args.seed,args.output,**options,
              **({'resume':args.resume} if args.resume else {}))
    elapsed=time.monotonic()-start
    after=resource.getrusage(resource.RUSAGE_CHILDREN)
    report=dict(algorithm=args.algorithm,seed=args.seed,character='Ironclad',start_utc=start_utc,
                total_seconds=elapsed,status=result['status'],cpus=sorted(cpus),options=options,
                code_sha256=hashes,timings=dict(metrics),
                native_cpu_seconds=(after.ru_utime+after.ru_stime)-(before.ru_utime+before.ru_stime),
                expanded=result.get('expanded'),deaths=result.get('deaths'),
                unresolved_branches=result.get('unresolved_branches'))
    if args.resume: report['resume']=str(args.resume.resolve())
    astar.write_json(args.output/'benchmark.json',report)
    print(json.dumps(report),flush=True)


if __name__=='__main__': main()
