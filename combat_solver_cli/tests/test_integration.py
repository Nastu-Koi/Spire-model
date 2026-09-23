import json
from pathlib import Path
import tempfile
import time
import unittest
from combat_solver_cli.client import DEFAULT_CONFIG, SolverEngine
from combat_solver_cli.__main__ import first_combat
from combat_solver_cli.trajectory import native_replay_error, verify_and_export
from combat_solver_cli.astar import ReplayMismatch, ReplayWorker, from_records
from model.protocol import CHARACTERS


@unittest.skipUnless(DEFAULT_CONFIG.is_file(), 'Local game/solver configuration required')
class NativeTests(unittest.TestCase):
    def test_five_characters_native_first_combat(self):
        for character in CHARACTERS:
            with self.subTest(character=character):
                result=first_combat(DEFAULT_CONFIG,character,'combat-solver-cli-smoke',500)
                self.assertEqual(result['status'],'victory'); self.assertTrue(result['training_ready'])

    def test_solve_read_only_and_step_versioned(self):
        with SolverEngine() as engine:
            engine.reset('Ironclad','solver-test')
            frame=engine.send(dict(cmd='enter_room',type='combat',encounter='SHRINKER_BEETLE_WEAK',decision_protocol=True))
            solved=engine.solve(frame,budget_ms=300)
            self.assertNotEqual(solved.get('type'),'error')
            after=engine.send(dict(cmd='advance_to_boundary'))
            for key in ('routing','public','legal','contract'): self.assertEqual(frame[key],after[key])
            result=engine.step(frame,budget_ms=300)
            self.assertEqual(result['type'],'solver_step')
            self.assertIn(result['candidate_ref'],[c['candidate_ref'] for c in frame['legal']['candidates']])
            duplicate=engine.step(frame,budget_ms=300)
            self.assertEqual(duplicate['type'],'error')
            self.assertIn('stale',duplicate['message'].lower())

    def test_incomplete_prefix_cannot_be_exported(self):
        with tempfile.TemporaryDirectory() as d:
            prefix=Path(d)/'prefix.json'; prefix.write_text(json.dumps(dict(character='Ironclad',seed='incomplete',records=[])))
            output=Path(d)/'output'
            with self.assertRaises(ReplayMismatch): verify_and_export(DEFAULT_CONFIG,prefix,output)
            self.assertFalse((output/'accepted.jsonl').exists())
            self.assertFalse((output/'verified_trace.jsonl').exists())

    def test_native_neutralize_damage_and_weak(self):
        with SolverEngine() as engine:
            engine.reset('Silent','solver-test')
            engine.send(dict(cmd='set_player',deck=['NEUTRALIZE']*5,relics=[]))
            frame=engine.send(dict(cmd='enter_room',type='combat',encounter='SHRINKER_BEETLE_WEAK',decision_protocol=True))
            result=engine.step(frame,budget_ms=500)
            self.assertEqual(result['type'],'solver_step')
            candidate=next(c for c in frame['legal']['candidates'] if c['candidate_ref']==result['candidate_ref'])
            target=candidate['target_refs'][0]
            before=next(e for e in frame['public']['entities'] if e.get('ref')==target)
            after=next(e for e in result['frame']['public']['entities'] if e.get('ref')==target)
            self.assertEqual(before['hp']-after['hp'],3)
            self.assertTrue(any(e.get('content_id')=='POWER.WEAK_POWER' and e.get('owner_ref')==target for e in result['frame']['public']['entities']))
            self.assertFalse(result['frame']['contract']['training_ready'])

    def test_self_targeted_potion_maps_native_candidate(self):
        with SolverEngine() as engine:
            engine.reset('Ironclad','solver-test')
            engine.send(dict(cmd='set_player',deck=['STRIKE_IRONCLAD']*5,hp=1,relics=[],potions=['STRENGTH_POTION']))
            frame=engine.send(dict(cmd='enter_room',type='combat',encounter='SHRINKER_BEETLE_WEAK',decision_protocol=True))
            for _ in range(20):
                result=engine.step(frame,budget_ms=500,potions=True,potion_policy='RequireAtLeastOne')
                self.assertEqual(result['type'],'solver_step',result)
                candidate=next(c for c in frame['legal']['candidates'] if c['candidate_ref']==result['candidate_ref'])
                if candidate['verb']=='USE_POTION':
                    self.assertEqual(candidate['target_refs'],['player']); return
                frame=result['frame']
                if frame['boundary']!='decision' or frame['public']['phase']!='combat': break
            self.fail('Solver did not execute required strength potion')

    def test_turn_plan_reuse_and_replan_on_next_turn(self):
        with SolverEngine() as engine:
            engine.reset('Ironclad','solver-test')
            engine.send(dict(cmd='set_player',deck=['STRIKE_IRONCLAD']*5,relics=[]))
            frame=engine.send(dict(cmd='enter_room',type='combat',encounter='SHRINKER_BEETLE_WEAK',decision_protocol=True))
            reused=searched=0
            turn=None
            for _ in range(8):
                result=engine.step(frame,budget_ms=1000,reuse_turn_plan=True)
                self.assertEqual(result['type'],'solver_step',result)
                self.assertIn(result['candidate_ref'],[c['candidate_ref'] for c in frame['legal']['candidates']])
                if result.get('search'): searched+=1
                else: reused+=1
                frame=result['frame']
                if frame['boundary']!='decision' or frame['public']['phase']!='combat': break
            self.assertGreater(reused,0)
            self.assertGreaterEqual(searched,2)

    def test_native_complete_route_survives_refinement_failure(self):
        fixture = json.loads((Path(__file__).parent/'fixtures/defect_refinement.json').read_text())
        worker = ReplayWorker(DEFAULT_CONFIG, fixture['character'], fixture['seed'], 2500,
                              time.monotonic()+90, 2000, True)
        try:
            worker.restore(from_records(fixture['records']))
            worker.combat_and_forced()
            self.assertNotEqual(worker.frame['boundary'], 'terminal')
            self.assertTrue(worker.frame['contract']['training_ready'])
            self.assertGreater(worker.solver_steps, 0)
        finally:
            worker.close()

    def test_test_subject_native_phase_change_settles(self):
        fixture = json.loads((Path(__file__).parent/'fixtures/test_subject_phase_change.json').read_text())
        worker = ReplayWorker(DEFAULT_CONFIG, fixture['character'], fixture['seed'], 2500,
                              time.monotonic()+90, 2000)
        try:
            worker.restore(from_records(fixture['records']))
            worker.execute(fixture['pending_action'])
            worker.settle()
            self.assertNotEqual(worker.frame['boundary'], 'waiting')
            self.assertTrue(worker.frame['contract']['training_ready'])
        finally:
            worker.close()

    def test_crusher_native_entry_settles_without_engine_errors(self):
        with SolverEngine() as engine:
            engine.reset('Ironclad', 'crusher-visual-regression')
            frame = engine.send(dict(cmd='enter_room', type='combat', encounter='KAISER_CRAB_BOSS', decision_protocol=True))
            deadline = time.monotonic()+5
            while frame.get('boundary') == 'waiting' and time.monotonic()<deadline:
                frame = engine.send(dict(cmd='advance_to_boundary'))
            self.assertEqual(frame.get('boundary'), 'decision', frame)
            self.assertEqual(frame['public']['phase'], 'combat')
            self.assertFalse(frame['contract']['training_ready'])


    def test_a0_native_contract_and_first_combat(self):
        with SolverEngine() as engine:
            frame = engine.reset('Ironclad', 'combat-solver-cli-a0-contract', ascension=0)
            self.assertEqual(frame['contract']['fixed_ascension'], 0)
            self.assertTrue(frame['contract']['training_ready'])
        result = first_combat(DEFAULT_CONFIG, 'Ironclad', 'combat-solver-cli-a0-contract', 500, ascension=0)
        self.assertEqual(result['status'], 'victory')
        self.assertEqual(result['ascension'], 0)

    def test_kaiser_crab_full_combat_has_no_headless_errors(self):
        with SolverEngine() as engine:
            engine.reset('Ironclad', 'kaiser-crab-full-regression', ascension=0)
            engine.send(dict(cmd='set_player', hp=999, max_hp=999, deck=['BLUDGEON']*5, relics=[]))
            frame = engine.send(dict(cmd='enter_room', type='combat', encounter='KAISER_CRAB_BOSS', decision_protocol=True))
            for _ in range(200):
                if frame.get('boundary') == 'waiting':
                    frame = engine.send(dict(cmd='advance_to_boundary'))
                    continue
                if frame.get('boundary') in ('terminal', 'error') or frame.get('public', {}).get('phase') != 'combat':
                    break
                result = engine.step(frame, budget_ms=1000, potions=True, reuse_turn_plan=True)
                self.assertEqual(result.get('type'), 'solver_step', result)
                frame = result['frame']
            self.assertEqual(frame.get('public', {}).get('phase'), 'rewards', frame)
            self.assertIsNone(native_replay_error(engine))
