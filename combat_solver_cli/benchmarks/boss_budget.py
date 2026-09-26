"""Replay a full native victory; audit actual solve budgets at all three bosses."""
import argparse
import json
import time
from pathlib import Path
from combat_solver_cli.astar import ReplayWorker, state_key, player
from combat_solver_cli.client import DEFAULT_CONFIG


def check(prefix_path, output):
    data=json.loads(prefix_path.read_text())
    started=time.monotonic()
    worker=ReplayWorker(DEFAULT_CONFIG,data['character'],data['seed'],1000,
                        started+240,10000,True,data['ascension'],5000)
    reports=[]; checked=set(); normal=False; bosses=set()
    try:
        worker.restore(None)
        for record in data['records']:
            frame=worker.settle()
            assert state_key(frame)==record['before_hash'], 'Replay diverged'
            p=player(frame); location=(p.get('act'),p.get('floor'))
            if frame['public']['phase']=='combat' and location not in checked:
                checked.add(location)
                info=worker.engine.send({'cmd':'solver_info'})
                is_boss=info['boss_combat']
                if (is_boss and location[0] not in bosses) or (not is_boss and not normal):
                    begin=time.monotonic()
                    result=worker.engine.solve(frame,budget_ms=1000,boss_budget_ms=5000,
                                               potions=True,reuse_turn_plan=True)
                    assert result['type']=='solver_result', result
                    assert result['budget_ms']==(5000 if is_boss else 1000), result['budget_ms']
                    assert result['boss_combat']==is_boss
                    # The remaining recorded actions also verify solver state isolation.
                    reports.append(dict(act=location[0],floor=location[1],boss=is_boss,
                                        budget_ms=result['budget_ms'],seconds=time.monotonic()-begin))
                    if is_boss: bosses.add(location[0])
                    else: normal=True
            worker.execute(record['action'],record['actor'])
        final=worker.settle()
        assert final['boundary']=='terminal' and final['public']['outcome']['victory']
        assert bosses=={1,2,3} and normal, reports
        report=dict(status='passed',seed=data['seed'],checks=reports,seconds=time.monotonic()-started)
        output.write_text(json.dumps(report,indent=2));print(json.dumps(report))
    finally: worker.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prefix',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path);args=parser.parse_args()
    check(args.prefix,args.output)
