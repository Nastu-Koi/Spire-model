"""Selection handoff at the ReplayWorker macro boundary."""
import time
import unittest
from unittest.mock import Mock
from combat_solver_cli.astar import ReplayWorker, ReplayMismatch


def worker(phase='card_select', count=2):
    w = ReplayWorker(None, 'Ironclad', 'fixed', 1000, time.monotonic()+5, 1000)
    w.frame = {'boundary': 'decision', 'public': {'phase': phase, 'entities': [
        {'entity_type': 'player', 'act': 2, 'floor': 2}]},
        'legal': {'candidates': [{'candidate_ref': str(i), 'verb': 'SELECT_ONE',
            'decoder_slot_ref': str(i), 'source_refs': [], 'target_refs': []}
            for i in range(count)]}}
    w.settle = Mock(side_effect=lambda: w.frame)
    w.engine = Mock()
    w.engine.send.return_value = {'type': 'solver_info', 'combat_in_progress': True}
    return w


class SelectionHandoffTests(unittest.TestCase):
    def test_unplanned_selection_retains_all_candidates_and_prefix(self):
        w = worker()
        current = w.frame
        w.engine.step.return_value = {'type': 'solver_selection_required', 'frame': current}
        self.assertIs(w.combat_and_forced(), current)
        self.assertEqual(len(current['legal']['candidates']), 2)
        self.assertIsNone(w.prefix)
        self.assertEqual(w.solver_steps, 0)

    def test_planned_selection_is_executed_by_solver(self):
        w = worker()
        terminal = {'boundary': 'terminal'}
        w.engine.step.return_value = {'type': 'solver_step', 'candidate_ref': '0', 'frame': terminal}
        self.assertIs(w.combat_and_forced(), terminal)
        self.assertEqual(w.prefix.actor, 'combat_solver')
        self.assertEqual(w.solver_steps, 1)
        self.assertEqual(w.engine.step.call_args.kwargs['budget_ms'], 1000)
        self.assertEqual(w.engine.step.call_args.kwargs['boss_budget_ms'], 5000)

    def test_solver_errors_are_not_treated_as_unplanned_selection(self):
        w = worker()
        w.engine.step.return_value = {'type': 'error', 'message': 'isolation failed'}
        with self.assertRaises(ReplayMismatch): w.combat_and_forced()

    def test_unplanned_single_choice_is_forced(self):
        w = worker(count=1)
        terminal = {'boundary': 'terminal'}
        w.engine.step.return_value = {'type': 'solver_selection_required'}
        def execute(*args): w.frame = terminal
        w.execute = Mock(side_effect=execute)
        self.assertIs(w.combat_and_forced(), terminal)
        self.assertEqual(w.execute.call_args.args[1], 'forced')

    def test_handoff_rejected_outside_selection(self):
        w = worker(phase='combat')
        w.engine.step.return_value = {'type': 'solver_selection_required'}
        with self.assertRaises(ReplayMismatch): w.combat_and_forced()


if __name__ == '__main__': unittest.main()
