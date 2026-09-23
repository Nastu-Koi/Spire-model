import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from combat_solver_cli.astar import Prefix, load_frontier, save_frontier, search, resolve, state_key
from combat_solver_cli.client import InfrastructureError


def frame(terminal=False, victory=False):
    return dict(boundary='terminal' if terminal else 'decision', public=dict(phase='map', outcome=dict(victory=victory), entities=[dict(entity_type='player', act=1, floor=1, hp=80, max_hp=80), dict(entity_type='map_node',ref='a',content_id='Treasure',floor=2), dict(entity_type='map_node',ref='b',content_id='Monster',floor=2)]), legal=dict(candidates=[dict(candidate_ref='a', verb='MOVE_TO_NODE', source_refs=['a']), dict(candidate_ref='b', verb='MOVE_TO_NODE', source_refs=['b'])]))


class TinyWorker:
    fail = False
    def __init__(self, *args): self.prefix=None; self.replayed=self.solver_steps=0
    def check(self): pass
    def close(self): pass
    def restore(self, prefix):
        if self.fail: raise InfrastructureError('worker disappeared')
        self.prefix=prefix
    def execute(self, action, actor="astar"): self.prefix=Prefix(self.prefix, state_key(frame()), action, 'astar', 1)
    def combat_and_forced(self):
        return frame() if self.prefix is None else frame(True, self.prefix.action['source_refs']==['b'])


class SearchTests(unittest.TestCase):
    def test_semantic_resolution(self):
        f=frame(); action=dict(f['legal']['candidates'][0]); action.pop('candidate_ref')
        f['legal']['candidates'][0]['candidate_ref']='new'
        self.assertEqual(resolve(f, action)['candidate_ref'], 'new')

    def test_checkpoint_preserves_distinct_histories(self):
        with tempfile.TemporaryDirectory() as d:
            a=Prefix(None,'same',{'verb':'a'},'astar',1)
            b=Prefix(None,'same',{'verb':'b'},'astar',1)
            queue=[(1.,1,.1,a,None),(1.,2,.1,b,None)]
            path=Path(d)/'frontier.json'; save_frontier(path,queue,'Ironclad','s',5)
            loaded=load_frontier(path,'Ironclad','s',5)
            self.assertIsNot(loaded[0][3],loaded[1][3])
            self.assertEqual(loaded[0][3].records(),a.records())

    def test_checkpoint_preserves_backjump_scheduler_state(self):
        with tempfile.TemporaryDirectory() as d:
            prefix = Prefix(None, 'state', {'verb': 'a'}, 'astar', 1,
                            phase='rewards', act=2, floor=13)
            path = Path(d)/'frontier.json'
            save_frontier(path, [(1., 1, .1, prefix, None)], 'Ironclad', 's', 5,
                          search_state={'boss_failures': 3, 'backtrack_before': 17})
            loaded, state = load_frontier(path, 'Ironclad', 's', 5, with_state=True)
            self.assertEqual(state, {'boss_failures': 3, 'backtrack_before': 17})
            self.assertEqual((loaded[0][3].phase, loaded[0][3].act, loaded[0][3].floor),
                             ('rewards', 2, 13))

    def test_alternative_after_death(self):
        with tempfile.TemporaryDirectory() as d, patch('combat_solver_cli.astar.ReplayWorker',TinyWorker), patch('combat_solver_cli.trajectory.verify_and_export') as export:
            result=search('config','Ironclad','s',Path(d)/'run')
            self.assertEqual(result['status'],'verified_victory')
            self.assertEqual(result['deaths'],1); export.assert_called_once()

    def test_budget_then_resume(self):
        with tempfile.TemporaryDirectory() as d, patch('combat_solver_cli.astar.ReplayWorker',TinyWorker), patch('combat_solver_cli.trajectory.verify_and_export'):
            first=Path(d)/'first'
            result=search('config','Ironclad','s',first,max_expansions=1)
            self.assertEqual(result['status'],'budget_exhausted')
            self.assertFalse((first/'accepted.jsonl').exists())
            result=search('config','Ironclad','s',Path(d)/'second',resume=first/'frontier.json')
            self.assertEqual(result['status'],'verified_victory')

    def test_infrastructure_failure_preserves_active_node(self):
        with tempfile.TemporaryDirectory() as d, patch('combat_solver_cli.astar.ReplayWorker',TinyWorker), patch.object(TinyWorker,'fail',True):
            directory=Path(d)/'run'; result=search('config','Ironclad','s',directory)
            self.assertEqual(result['status'],'infrastructure_error')
            self.assertEqual(result['expanded'],1)
            self.assertEqual(result['unresolved_branches'],0)
            self.assertEqual(len(load_frontier(directory/'frontier.json','Ironclad','s',5)),1)

    def test_failed_verification_not_accepted(self):
        with tempfile.TemporaryDirectory() as d, patch('combat_solver_cli.astar.ReplayWorker',TinyWorker), patch('combat_solver_cli.trajectory.verify_and_export',side_effect=RuntimeError('divergence')):
            directory=Path(d)/'run'; result=search('config','Ironclad','s',directory)
            self.assertEqual(result['status'],'replay_failed'); self.assertFalse((directory/'accepted.jsonl').exists())
            self.assertEqual(len(load_frontier(directory/'frontier.json','Ironclad','s',5)), 1)

    def test_entering_next_act_improves_progress_estimate(self):
        from combat_solver_cli.astar import heuristic
        before=frame(); before['public']['entities'][0].update(act=1,floor=17)
        after=frame(); after['public']['entities'][0].update(act=2,floor=0)
        self.assertLess(heuristic(after),heuristic(before))

    def test_shop_card_score_follows_offers_relation(self):
        from combat_solver_cli.astar import preference
        f=frame()
        f['public']['relations']=[dict(source='merchant:0',target='offered-card',role='offers')]
        f['public']['entities'] += [dict(entity_type='shop_item',ref='merchant:0',content_id='CARD.INFLAME'),dict(entity_type='card',ref='offered-card',content_id='CARD.INFLAME',card_type='Power',rarity='Uncommon',stats={'StrengthPower':2})]
        candidate=dict(verb='BUY_ITEM',source_refs=['merchant:0'])
        strong=preference(f,candidate)
        f['public']['entities'][-1].update(card_type='Curse',stats={})
        self.assertGreater(strong,preference(f,candidate))

    def test_boss_reward_heal_estimate_continues_at_next_act_start(self):
        from combat_solver_cli.astar import heuristic
        before=frame(); before['public']['phase']='rewards'
        before['public']['entities'][0].update(act=1,floor=17,hp=5)
        before['public']['entities'].append(dict(entity_type='map_node',content_id='Boss',current=True,floor=16))
        after=frame(); after['public']['entities'][0].update(act=2,floor=0,hp=5)
        self.assertLess(heuristic(after),heuristic(before))

    def test_greedy_probe_retains_alternative_after_death(self):
        with tempfile.TemporaryDirectory() as d, patch('combat_solver_cli.astar.ReplayWorker',TinyWorker), patch('combat_solver_cli.trajectory.verify_and_export'):
            result=search('config','Ironclad','s',Path(d)/'run',rollout_decisions=8)
            self.assertEqual(result['status'],'verified_victory')
            self.assertEqual(result['deaths'],1)
            self.assertEqual(result['algorithm'],'weighted_astar_failure_backjump_v1')

    def test_minimal_terminal_death_packet_is_not_unresolved(self):
        class MinimalTerminal(TinyWorker):
            def combat_and_forced(self):
                result=super().combat_and_forced()
                if result['boundary']=='terminal': result['public'].pop('entities')
                return result
        with tempfile.TemporaryDirectory() as d, patch('combat_solver_cli.astar.ReplayWorker',MinimalTerminal), patch('combat_solver_cli.trajectory.verify_and_export'):
            result=search('config','Ironclad','s',Path(d)/'run')
            self.assertEqual(result['deaths'],1)
            self.assertEqual(result['unresolved_branches'],0)
            self.assertEqual(result['status'],'verified_victory')

    def test_map_preference_accounts_for_later_elite_and_rest(self):
        from combat_solver_cli.astar import preference
        f=frame(); f['public']['entities'] += [dict(entity_type='map_node',ref=ref,content_id=kind) for ref,kind in [('a','Monster'),('b','Monster'),('elite','Elite'),('rest','RestSite'),('boss','Boss')]]
        f['public']['relations']=[dict(source=a,target=b,role='map_edge') for a,b in [('a','elite'),('b','rest'),('elite','boss'),('rest','boss')]]
        risky=preference(f,dict(verb='MOVE_TO_NODE',source_refs=['a']))
        safe=preference(f,dict(verb='MOVE_TO_NODE',source_refs=['b']))
        self.assertGreater(safe,risky)


class ReplayErrorGateTests(unittest.TestCase):
    def test_native_error_is_detected_across_read_boundary(self):
        from types import SimpleNamespace
        from combat_solver_cli.trajectory import native_replay_error
        with tempfile.TemporaryFile() as stream:
            stream.write(b'x'*65533+b'[ERROR] System.MissingMethodException: visual method\n')
            stream.flush()
            before=stream.tell()
            error=native_replay_error(SimpleNamespace(stderr=stream))
            self.assertIn("MissingMethodException", error)
            self.assertEqual(stream.tell(),before)

    def test_clean_native_log_is_accepted(self):
        from types import SimpleNamespace
        from combat_solver_cli.trajectory import native_replay_error
        with tempfile.TemporaryFile() as stream:
            stream.write(b'[INFO] combat ended with victory\n')
            self.assertIsNone(native_replay_error(SimpleNamespace(stderr=stream)))

class AscensionTests(unittest.TestCase):
    def test_checkpoint_rejects_different_ascension(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'frontier.json'
            save_frontier(path, [], 'Ironclad', 's', 5, ascension=0)
            self.assertEqual(load_frontier(path, 'Ironclad', 's', 5, ascension=0), [])
            with self.assertRaises(ValueError):
                load_frontier(path, 'Ironclad', 's', 5, ascension=10)

    def test_generated_jobs_are_deterministic_and_balanced(self):
        from combat_solver_cli.generate_bootstrap import make_jobs
        jobs = make_jobs(2, ('Ironclad', 'Silent'), seed_prefix='a0-test')
        self.assertEqual(jobs, [
            {'character': 'Ironclad', 'seed': 'a0-test-000000'},
            {'character': 'Silent', 'seed': 'a0-test-000000'},
            {'character': 'Ironclad', 'seed': 'a0-test-000001'},
            {'character': 'Silent', 'seed': 'a0-test-000001'},
        ])

class A0DataTests(unittest.TestCase):
    def test_bootstrap_schema_accepts_matching_a0_frames(self):
        from model.data import validate_run
        from model.testing import demonstration
        run = demonstration(count=3, select=1)
        run['ascension'] = 0
        run['contract']['fixed_ascension'] = 0
        for macro in run['macros']:
            for step in macro['steps']:
                step['frame']['contract']['fixed_ascension'] = 0
        self.assertIs(validate_run(run), run)

    def test_heuristic_penalizes_extreme_deck_bloat(self):
        from combat_solver_cli.astar import heuristic
        compact = frame()
        card = dict(entity_type='card', zone='deck', content_id='CARD.TEST', card_type='Attack', cost=1,
                    rarity='Common', target_type='SingleEnemy', stats={'Damage': 8})
        compact['public']['entities'] += [dict(card, ref=f'card:{i}') for i in range(20)]
        bloated = json.loads(json.dumps(compact))
        bloated['public']['entities'] += [dict(card, ref=f'extra:{i}') for i in range(25)]
        self.assertGreater(heuristic(bloated), heuristic(compact))

    def test_elite_preference_drops_when_health_is_low(self):
        from combat_solver_cli.astar import preference
        f = frame()
        elite = next(e for e in f['public']['entities'] if e.get('ref') == 'a')
        elite['content_id'] = 'Elite'
        candidate = f['legal']['candidates'][0]
        healthy = preference(f, candidate)
        next(e for e in f['public']['entities'] if e.get('entity_type') == 'player')['hp'] = 20
        self.assertGreater(healthy, preference(f, candidate))

class ResumeBatchTests(unittest.TestCase):
    def test_resume_batch_selects_only_existing_frontiers(self):
        from combat_solver_cli.resume_bootstrap import resumable_jobs
        from model.protocol import fingerprint
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            jobs = [{'character': 'Ironclad', 'seed': 'one'}, {'character': 'Silent', 'seed': 'two'}]
            (root/'manifest.json').write_text(json.dumps({'jobs': jobs, 'options': {'ascension': 0}}))
            directory = root/'jobs'/('Ironclad-'+fingerprint('one')[:16])
            directory.mkdir(parents=True)
            (directory/'frontier.json').write_text('{}')
            selected, manifest = resumable_jobs(root)
            self.assertEqual([(j['character'], j['seed']) for j in selected], [('Ironclad', 'one')])
            self.assertEqual(Path(selected[0]['resume']), (directory/'frontier.json').resolve())
            self.assertEqual(manifest['options']['ascension'], 0)

class RandomSeedTests(unittest.TestCase):
    def test_random_seed_generation_retries_collisions(self):
        from combat_solver_cli.generate_bootstrap import make_jobs
        with patch('combat_solver_cli.generate_bootstrap.secrets.token_hex', side_effect=['aa', 'aa', 'bb']):
            jobs = make_jobs(2, ('Ironclad',))
        self.assertEqual([j['seed'] for j in jobs], ['AA', 'BB'])

class TwoLaneTests(unittest.TestCase):
    def test_two_lanes_run_concurrently_and_merge_disjoint_frontiers(self):
        import heapq
        import threading
        from combat_solver_cli.lanes import search_parallel
        active = 0
        maximum = 0
        lock = threading.Lock()
        barrier = threading.Barrier(2)
        resumes = []

        def fake_search(config, character, seed, output, **options):
            nonlocal active, maximum
            output = Path(output); output.mkdir(parents=True)
            queue, state = load_frontier(options['resume'], character, seed,
                                         options['weight'], options['ascension'], with_state=True)
            resumes.append({node[3].action['verb'] for node in queue})
            with lock:
                active += 1; maximum = max(maximum, active)
            barrier.wait(timeout=2)
            save_frontier(output/'frontier.json', queue, character, seed,
                          options['weight'], options['ascension'], search_state=state)
            with lock: active -= 1
            return {'status': 'budget_exhausted', 'expanded': 1, 'deaths': 0,
                    'unresolved_branches': 0, 'boss_failures': state.get('boss_failures', 0)}

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            queue = []
            for index in range(4):
                prefix = Prefix(None, str(index), {'verb': str(index)}, 'astar', 1,
                                phase='map', act=1, floor=index)
                queue.append((float(index), index, .1, prefix, None))
            heapq.heapify(queue)
            checkpoint = root/'input.json'
            save_frontier(checkpoint, queue, 'Ironclad', 's', 5, ascension=0)
            result = search_parallel('config', 'Ironclad', 's', root/'out',
                                     weight=5, ascension=0, resume=checkpoint,
                                     search_lanes=2, serial_search=fake_search)
            self.assertEqual(maximum, 2)
            self.assertEqual(len(resumes), 2)
            self.assertTrue(resumes[0].isdisjoint(resumes[1]))
            self.assertEqual(result['frontier'], 4)
            self.assertEqual(len(load_frontier(root/'out'/'frontier.json',
                                               'Ironclad', 's', 5, ascension=0)), 4)

    def test_four_lanes_partition_and_merge(self):
        import heapq
        from combat_solver_cli.lanes import search_parallel
        seen = []
        def fake_search(config, character, seed, output, **options):
            output = Path(output); output.mkdir()
            queue, state = load_frontier(options['resume'], character, seed,
                                         options['weight'], options['ascension'], with_state=True)
            seen.append({node[3].action['verb'] for node in queue})
            save_frontier(output/'frontier.json', queue, character, seed,
                          options['weight'], options['ascension'], search_state=state)
            return {'status': 'budget_exhausted', 'expanded': 1}
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            queue = [(float(i), i, .1, Prefix(None, str(i), {'verb': str(i)},
                     'astar', 1), None) for i in range(8)]
            heapq.heapify(queue)
            checkpoint = root/'input.json'
            save_frontier(checkpoint, queue, 'Ironclad', 's', 5, ascension=0)
            result = search_parallel('config', 'Ironclad', 's', root/'out',
                                     weight=5, ascension=0, resume=checkpoint,
                                     search_lanes=4, serial_search=fake_search)
            self.assertEqual(len(seen), 4)
            self.assertEqual(set.union(*seen), {str(i) for i in range(8)})
            self.assertEqual(sum(map(len, seen)), 8)
            self.assertEqual(result['search_lanes'], 4)
            self.assertEqual(result['frontier'], 8)

    def test_failed_lane_retains_its_input_shard(self):
        from combat_solver_cli.lanes import search_parallel
        def fake_search(config, character, seed, output, **options):
            queue = load_frontier(options['resume'], character, seed,
                                  options['weight'], options['ascension'])
            if queue[0][3].action['verb'] == '0':
                raise RuntimeError('lane crashed')
            output = Path(output); output.mkdir()
            save_frontier(output/'frontier.json', queue, character, seed,
                          options['weight'], options['ascension'])
            return {'status': 'budget_exhausted', 'expanded': 1}
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            queue = [(float(i), i, .1, Prefix(None, str(i), {'verb': str(i)},
                     'astar', 1), None) for i in range(4)]
            checkpoint = root/'input.json'
            save_frontier(checkpoint, queue, 'Ironclad', 's', 5, ascension=0)
            result = search_parallel('config', 'Ironclad', 's', root/'out',
                                     weight=5, ascension=0, resume=checkpoint,
                                     search_lanes=2, serial_search=fake_search)
            self.assertEqual(result['status'], 'infrastructure_error')
            self.assertEqual(result['recovered_lanes'], [0])
            self.assertEqual(result['frontier'], 4)

class SingleGameSchedulingTests(unittest.TestCase):
    def test_batch_rejects_parallel_games(self):
        from combat_solver_cli.batch import generate
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError, 'one game'):
                generate('config', [{'character': 'Ironclad', 'seed': 's'}], Path(d)/'out', workers=2)

    def test_failure_backjump_selects_earlier_decision(self):
        from combat_solver_cli.astar import pop_frontier
        early = Prefix(None, 'early', {'verb': 'TAKE_CARD_REWARD'}, 'astar', 10,
                       phase='rewards', act=1, floor=8)
        late = Prefix(None, 'late', {'verb': 'CHOOSE_REST_OPTION'}, 'astar', 100,
                      phase='rest_site', act=3, floor=14)
        frontier = [(100., 1, 1., early, {'verb': 'early'}),
                    (1., 2, 1., late, {'verb': 'late'})]
        import heapq
        heapq.heapify(frontier)
        selected = pop_frontier(frontier, backtrack_before=12)
        self.assertIs(selected[3], early)
        self.assertIs(frontier[0][3], late)

class TargetTrajectoryTests(unittest.TestCase):
    def test_target_quotas_balance_all_characters(self):
        from combat_solver_cli.target import target_quotas, target_schedule
        quotas = target_quotas(12, ('Ironclad', 'Silent', 'Defect', 'Necrobinder', 'Regent'))
        self.assertEqual(quotas, {'Ironclad': 3, 'Silent': 3, 'Defect': 2,
                                  'Necrobinder': 2, 'Regent': 2})
        schedule = target_schedule(quotas)
        self.assertEqual(schedule[:5], ['Ironclad', 'Silent', 'Defect', 'Necrobinder', 'Regent'])
        self.assertEqual(len(schedule), 12)

    def test_target_retries_same_seed_before_next_balanced_game(self):
        from combat_solver_cli.target import generate_target
        calls = []
        attempts = {}
        def fake_search(config, character, seed, output, resume=None, **options):
            output = Path(output); output.mkdir(parents=True)
            calls.append((character, seed, resume))
            attempts[seed] = attempts.get(seed, 0) + 1
            if attempts[seed] == 1:
                (output/'frontier.json').write_text('{}')
                return {'status': 'budget_exhausted'}
            (output/'accepted.jsonl').write_text('{}\n')
            return {'status': 'verified_victory'}
        def loaded(path):
            character, seed, _ = calls[-1]
            return [{'character': character, 'seed': seed,
                     'provenance': {'verified_outcome': 'A0_final_boss_victory'}}]
        with tempfile.TemporaryDirectory() as d, \
             patch('combat_solver_cli.target.load_runs', side_effect=loaded), \
             patch('combat_solver_cli.target.secrets.token_hex',
                   side_effect=[f'{i:016x}' for i in range(20)]):
            result = generate_target('config', Path(d)/'out', 6,
                                     search_fn=fake_search, ascension=0,
                                     search_lanes=2, max_seconds=1)
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(result['completed'], {'Ironclad': 2, 'Silent': 1,
                'Defect': 1, 'Necrobinder': 1, 'Regent': 1})
            self.assertEqual(len(calls), 12)
            for index in range(0, len(calls), 2):
                self.assertEqual(calls[index][:2], calls[index+1][:2])
                self.assertIsNone(calls[index][2])
                self.assertIsNotNone(calls[index+1][2])

class TargetResumeTests(unittest.TestCase):
    def test_target_resume_keeps_active_random_seed(self):
        from combat_solver_cli.target import generate_target, resume_target
        calls = []
        def fake_search(config, character, seed, output, resume=None, **options):
            output = Path(output); output.mkdir(parents=True)
            calls.append((character, seed, resume))
            if len(calls) == 1:
                (output/'frontier.json').write_text('{}')
                output.parents[3].joinpath('STOP').touch()
                return {'status': 'stopped'}
            (output/'accepted.jsonl').write_text('{}\n')
            return {'status': 'verified_victory'}
        with tempfile.TemporaryDirectory() as d, \
             patch('combat_solver_cli.target.load_runs', return_value=[{
                 'character': 'Ironclad', 'seed': 'A',
                 'provenance': {'verified_outcome': 'A0_final_boss_victory'}}]), \
             patch('combat_solver_cli.target.secrets.token_hex', return_value='a'):
            root = Path(d)/'out'
            first = generate_target('config', root, 1, characters=('Ironclad',),
                                    search_fn=fake_search, ascension=0, search_lanes=2)
            self.assertEqual(first['status'], 'stopped')
            second = resume_target(root, search_fn=fake_search)
            self.assertEqual(second['status'], 'complete')
            self.assertEqual(calls[0][1], calls[1][1])
            self.assertIsNotNone(calls[1][2])

    def test_resume_reconciles_completed_orphan_segment(self):
        from combat_solver_cli.target import generate_target, resume_target
        calls = []
        def fake_search(config, character, seed, output, resume=None, **options):
            output = Path(output); output.mkdir(parents=True)
            calls.append(resume)
            (output/'frontier.json').write_text('{}')
            (output/'summary.json').write_text(json.dumps({'status': 'budget_exhausted'}))
            if len(calls) == 1: raise RuntimeError('interrupted before manifest update')
            (output/'accepted.jsonl').write_text('{}\n')
            return {'status': 'verified_victory'}
        with tempfile.TemporaryDirectory() as d, \
             patch('combat_solver_cli.target.load_runs', return_value=[{
                 'character': 'Ironclad',
                 'provenance': {'verified_outcome': 'A0_final_boss_victory'}}]), \
             patch('combat_solver_cli.target.secrets.token_hex', return_value='a'):
            root = Path(d)/'out'
            with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                generate_target('config', root, 1, characters=('Ironclad',),
                                search_fn=fake_search, ascension=0)
            result = resume_target(root, search_fn=fake_search)
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(len(calls), 2)
            self.assertTrue(calls[1].is_file())

    def test_resume_publishes_recorded_winner_without_search(self):
        from combat_solver_cli.target import generate_target, resume_target
        calls = []
        def fake_search(config, character, seed, output, resume=None, **options):
            output = Path(output); output.mkdir(parents=True)
            (output/'accepted.jsonl').write_text('{}\n')
            calls.append(seed)
            return {'status': 'verified_victory'}
        with tempfile.TemporaryDirectory() as d, \
             patch('combat_solver_cli.target.load_runs', return_value=[{
                 'character': 'Ironclad',
                 'provenance': {'verified_outcome': 'A0_final_boss_victory'}}]), \
             patch('combat_solver_cli.target.secrets.token_hex', return_value='a'):
            root = Path(d)/'out'
            with patch('combat_solver_cli.target._publish_verified', side_effect=RuntimeError('interrupted')):
                with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                    generate_target('config', root, 1, characters=('Ironclad',),
                                    search_fn=fake_search, ascension=0)
            result = resume_target(root, search_fn=fake_search)
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(len(calls), 1)
            self.assertTrue((root/'accepted.jsonl').read_text().strip())

class GeneratorCliTests(unittest.TestCase):
    def test_default_target_mode_keeps_seed_prefix_attribute(self):
        from combat_solver_cli.generate_bootstrap import main
        with tempfile.TemporaryDirectory() as d, \
             patch('sys.argv', ['generate_bootstrap', '--output', str(Path(d)/'out')]), \
             patch('combat_solver_cli.generate_bootstrap.generate_target',
                   return_value={'status': 'complete'}) as generate:
            with self.assertRaises(SystemExit) as stopped:
                main()
            self.assertEqual(stopped.exception.code, 0)
            self.assertEqual(generate.call_args.kwargs['target_trajectories'], 100)
            self.assertNotIn('seed_prefix', generate.call_args.kwargs)
