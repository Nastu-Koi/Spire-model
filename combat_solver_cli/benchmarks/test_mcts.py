import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from combat_solver_cli import mcts
from combat_solver_cli.astar import Prefix, append
from combat_solver_cli.client import InfrastructureError


def candidate(name):
    return dict(candidate_ref=name,verb='CHOOSE_EVENT_OPTION',decoder_slot_ref=name,source_refs=[name],target_refs=[])


def frame(terminal=None):
    p=dict(entity_type='player',act=1,floor=1,hp=80,max_hp=80)
    public=dict(phase='event',entities=[p])
    if terminal is not None: public['outcome']={'victory':terminal}
    return dict(boundary='terminal' if terminal is not None else 'decision',public=public,
                legal={'candidates':[] if terminal is not None else [candidate('bad'),candidate('good')]})


class Worker:
    fail_once=False
    def __init__(self,*args):
        self.prefix=None;self.frame=frame();self.last_player=self.frame['public']['entities'][0]
        self.solver_steps=self.replayed=0;self.stop_path=None
    def check(self): pass
    def close(self): pass
    def native_diagnostics(self): return ''
    def restore(self,prefix):
        self.prefix=prefix
        self.frame=frame(None if prefix is None else prefix.action['source_refs']==['good'])
    def execute(self,action,actor):
        if Worker.fail_once:
            Worker.fail_once=False
            raise InfrastructureError('test worker disconnected')
        c=next(c for c in self.frame['legal']['candidates'] if c['source_refs']==action['source_refs'])
        self.prefix=append(self.prefix,self.frame,c,actor)
        self.frame=frame(c['source_refs']==['good'])
    def combat_and_forced(self): return self.frame


class MctsTests(unittest.TestCase):
    def test_uct_visits_untried_child_and_backs_up_ancestors(self):
        tree=mcts.Tree(1);root=tree.add(None,None,None);a=tree.add(root,None,{'a':1},2);b=tree.add(root,None,{'b':1},1)
        tree.nodes[root].expanded=True
        self.assertEqual(tree.select(),[root,a])
        tree.backup([root,a],.4)
        self.assertEqual(tree.select(),[root,b])
        self.assertEqual(tree.nodes[root].visits,1)
        self.assertEqual(tree.nodes[a].value_sum,.4)

    def test_closed_children_propagate_without_discarding_siblings(self):
        t=mcts.Tree();r=t.add(None,None,None);a=t.add(r,None,None);b=t.add(r,None,None)
        t.nodes[r].expanded=True;t.nodes[a].closed=True;t.backup([r,a],0)
        self.assertFalse(t.nodes[r].closed)
        self.assertEqual(t.select(),[r,b])
        t.nodes[b].closed=True;t.backup([r,b],0)
        self.assertTrue(t.nodes[r].closed);self.assertEqual(t.select(),[])

    def test_checkpoint_preserves_stats_rng_and_shared_history(self):
        t=mcts.Tree(22);p=Prefix(None,'hash',{'action':1},'mcts',1,'event',1,1)
        q=Prefix(p,'hash2',{'action':2},'mcts',2,'map',1,1)
        a=t.add(None,p,None);b=t.add(a,q,None);t.backup([a,b],.6)
        with tempfile.TemporaryDirectory() as directory:
            f=Path(directory)/'tree.json';mcts.save_tree(f,t,{'seed':'x'},{'simulations':1})
            restored,stats=mcts.load_tree(f,{'seed':'x'},math.sqrt(2))
        self.assertIs(restored.nodes[b].prefix.parent,restored.nodes[a].prefix)
        self.assertEqual(restored.nodes[a].visits,1)
        self.assertEqual(restored.nodes[b].value_sum,.6)
        self.assertEqual(restored.rng.random(),t.rng.random())
        self.assertEqual(stats['simulations'],1)

    def test_repair_alternates_global_and_moves_to_earlier_floors(self):
        t=mcts.Tree(1)
        root=t.add(None,None,None);mid=t.add(root,None,None);late=t.add(mid,None,None);dead=t.add(late,None,None)
        early_alt=t.add(root,None,{'verb':'SKIP'})
        mid_alt=t.add(mid,None,{'verb':'SKIP'})
        late_alt=t.add(late,None,{'verb':'SKIP'})
        other=t.add(None,None,{'verb':'SKIP'},100)
        for index,floor in ((root,2),(mid,6),(late,8)):
            t.nodes[index].expanded=True;t.nodes[index].decision=dict(act=1,floor=floor,phase='map')
        t.nodes[dead].closed=True
        path=[root,mid,late,dead]
        t.schedule_repair(path,1,9)
        self.assertEqual(t.select(),[root,mid,late,late_alt])
        self.assertEqual(t.select(),[other])  # Explicit global exploration turn.
        t.schedule_repair(path,1,9);t.schedule_repair(path,1,9)
        self.assertEqual(t.select(),[root,mid,mid_alt])  # Third failure widens to two floors.
        self.assertFalse(t.nodes[early_alt].closed)
        t.schedule_repair(path,2,9)
        self.assertEqual(t.failure_counts,{'1:9':3,'2:9':1})

    def test_repair_checkpoint_preserves_schedule_and_exhausted_fallback(self):
        t=mcts.Tree(1);r=t.add(None,None,None);dead=t.add(r,None,None);alt=t.add(r,None,{'verb':'SKIP'})
        t.nodes[r].expanded=True;t.nodes[r].decision=dict(act=1,floor=2,phase='map');t.nodes[dead].closed=True
        t.schedule_repair([r,dead],1,3)
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'tree.json';mcts.save_tree(p,t,{}, {})
            copy,_=mcts.load_tree(p,{},math.sqrt(2))
        self.assertEqual(copy.failure_counts,t.failure_counts)
        self.assertEqual(copy.select(),t.select())
        copy.nodes[alt].closed=True;copy.nodes[r].closed=True
        copy.repair_due=True
        self.assertEqual(copy.select(),[])

    def test_risk_rollout_keeps_all_actions_but_limits_random_pool(self):
        t=mcts.Tree(1);r=t.add(None,None,None)
        best=t.add(r,None,{'verb':'CHOOSE_EVENT_OPTION'},5)
        near=t.add(r,None,{'verb':'CHOOSE_EVENT_OPTION'},4.5)
        bad=t.add(r,None,{'verb':'CHOOSE_EVENT_OPTION'},-12)
        f=frame()
        with patch.object(t.rng,'random',return_value=.05), patch.object(t.rng,'choice',side_effect=lambda xs:xs[-1]):
            self.assertEqual(t.rollout_child(r,.1,f),near)
            f['public']['entities'][0]['hp']=10
            self.assertEqual(t.rollout_child(r,.1,f),best)
            self.assertEqual(t.rollout_child(r,.1),bad)  # Legacy policy remains available.
        self.assertEqual(t.nodes[r].children,[best,near,bad])
        self.assertFalse(t.nodes[bad].closed)

    def test_rest_and_boss_adjacent_choices_reduce_randomness(self):
        for condition in ('rest','boss','remove'):
            t=mcts.Tree(1);r=t.add(None,None,None);f=frame()
            verb='CHOOSE_REST_OPTION' if condition=='rest' else 'CHOOSE_EVENT_OPTION'
            best=t.add(r,None,{'verb':verb},5);t.add(r,None,{'verb':verb},4.5)
            if condition=='boss':
                f['public']['entities'] += [dict(entity_type='map_node',ref='here',current=True),dict(entity_type='map_node',ref='boss',content_id='BOSS')]
                f['public']['relations']=[dict(role='map_edge',source='here',target='boss')]
            if condition=='remove':f['public']['selection_context']={'operation':'remove'}
            with patch.object(t.rng,'random',return_value=.05),patch.object(t.rng,'choice',side_effect=lambda xs:xs[-1]):
                self.assertEqual(t.rollout_child(r,.1,f),best)

    def test_policy_change_requires_matching_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(mcts,'ReplayWorker',Worker), patch('combat_solver_cli.trajectory.verify_and_export'):
            p=Path(directory);self.run_search(p/'old',local_repair=False,risk_aware_rollout=False)
            with self.assertRaisesRegex(ValueError,'identity or policy'):
                self.run_search(p/'new',resume=p/'old/tree.json')
            result=self.run_search(p/'legacy',resume=p/'old/tree.json',local_repair=False,risk_aware_rollout=False)
            self.assertEqual(result['status'],'verified_victory')

    def test_only_actual_victory_gets_reward_one(self):
        self.assertEqual(mcts.rollout_value(frame(True)),1)
        self.assertLess(mcts.rollout_value(frame(False)),1)
        self.assertLess(mcts.rollout_value(frame()),1)

    def run_search(self,path,**options):
        return mcts.search(None,'Ironclad','fixed',path,ascension=0,rollout_decisions=4,
                           rollout_epsilon=0,max_expansions=10,**options)

    def test_real_search_loop_explores_after_defeat_then_verifies(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(mcts,'ReplayWorker',Worker), \
             patch.object(mcts,'preference',side_effect=lambda f,c: int(c['source_refs']==['bad'])), \
             patch('combat_solver_cli.trajectory.verify_and_export') as verify:
            p=Path(directory)/'run';result=self.run_search(p)
            self.assertEqual(result['status'],'verified_victory')
            self.assertEqual(result['deaths'],1)
            self.assertEqual(result['simulations'],2)
            verify.assert_called_once()
            self.assertEqual(json.loads((p/'winning_prefix.json').read_text())['records'][-1]['action']['source_refs'],['good'])

    def test_infrastructure_failure_preserves_pending_action_for_resume(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(mcts,'ReplayWorker',Worker), \
             patch.object(mcts,'preference',return_value=0), \
             patch('combat_solver_cli.trajectory.verify_and_export'):
            p=Path(directory);Worker.fail_once=True
            result=self.run_search(p/'first')
            self.assertEqual(result['status'],'infrastructure_error')
            data=json.loads((p/'first/tree.json').read_text())
            self.assertTrue(any(n['pending'] is not None and not n['closed'] for n in data['nodes']))
            result=self.run_search(p/'resume',resume=p/'first/tree.json')
            self.assertEqual(result['status'],'verified_victory')

    def test_cli_mcts_default_rollout_and_explicit_zero(self):
        from combat_solver_cli.__main__ import main
        for extra,expected in (([],256),(['--rollout-decisions','0'],0)):
            argv=['combat_solver_cli','search','--algorithm','mcts','--seed','fixed','--output','unused']+extra
            with patch('sys.argv',argv), patch.object(mcts,'search',return_value={'status':'budget_exhausted'}) as run, patch('builtins.print'):
                with self.assertRaises(SystemExit): main()
            self.assertEqual(run.call_args.kwargs['rollout_decisions'],expected)
            self.assertEqual(run.call_args.kwargs['boss_budget_ms'],5000)

    def test_cli_policy_flags_do_not_leak_into_astar(self):
        from combat_solver_cli.__main__ import main
        argv=['combat_solver_cli','search','--seed','fixed','--output','unused']
        with patch('sys.argv',argv),patch('combat_solver_cli.__main__.search',return_value={'status':'budget_exhausted'}) as run,patch('builtins.print'):
            with self.assertRaises(SystemExit):main()
        self.assertNotIn('local_repair',run.call_args.kwargs)
        self.assertNotIn('risk_aware_rollout',run.call_args.kwargs)
        argv += ['--algorithm','mcts','--no-local-repair','--no-risk-aware-rollout']
        with patch('sys.argv',argv),patch.object(mcts,'search',return_value={'status':'budget_exhausted'}) as run,patch('builtins.print'):
            with self.assertRaises(SystemExit):main()
        self.assertFalse(run.call_args.kwargs['local_repair'])
        self.assertFalse(run.call_args.kwargs['risk_aware_rollout'])

    def test_verification_failure_is_not_accepted_and_resume_retries(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(mcts,'ReplayWorker',Worker), \
             patch.object(mcts,'preference',return_value=0):
            p=Path(directory)
            with patch('combat_solver_cli.trajectory.verify_and_export',side_effect=RuntimeError('reject')):
                result=self.run_search(p/'first')
            self.assertEqual(result['status'],'replay_failed')
            self.assertFalse((p/'first/accepted.jsonl').exists())
            with patch('combat_solver_cli.trajectory.verify_and_export') as verify:
                result=self.run_search(p/'resume',resume=p/'first/tree.json')
            self.assertEqual(result['status'],'verified_victory')
            self.assertEqual(result['expanded'],0)
            verify.assert_called_once()


if __name__=='__main__': unittest.main()
