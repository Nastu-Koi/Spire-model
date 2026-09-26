"""Failure scheduling must escape repeat non-boss deaths without losing branches."""
import heapq
import tempfile
import unittest
from pathlib import Path
from combat_solver_cli import astar


class FailureBackjumpTests(unittest.TestCase):
    def test_non_boss_failure_moves_to_earlier_decision_and_keeps_others(self):
        failures = {}
        cutoff = astar.failure_backtrack(3, 6, failures)
        early = astar.Prefix(None, '', {}, 'astar', 1, 'map', 3, 3)
        late = astar.Prefix(None, '', {}, 'astar', 1, 'map', 3, 5)
        frontier = [(1., 1, 0., late, {}), (100., 2, 0., early, {})]
        heapq.heapify(frontier)
        selected = astar.pop_frontier(frontier, cutoff)
        self.assertIs(selected[3], early)
        self.assertEqual(len(frontier), 1)
        self.assertIs(frontier[0][3], late)

    def test_repeated_location_backtracks_further_but_other_location_starts_fresh(self):
        failures = {}
        self.assertEqual(astar.failure_backtrack(3, 6, failures), 43)
        self.assertEqual(astar.failure_backtrack(3, 6, failures), 41)
        self.assertEqual(astar.failure_backtrack(2, 18, failures), 35)
        self.assertEqual(astar.failure_backtrack(None, None, failures), None)

    def test_resume_preserves_per_location_counts(self):
        failures = {}
        astar.failure_backtrack(3, 6, failures)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'frontier.json'
            astar.save_frontier(path, [], 'Ironclad', 'fixed', 12, 0,
                                {'failure_counts': failures})
            _, state = astar.load_frontier(path, 'Ironclad', 'fixed', 12, 0, with_state=True)
        self.assertEqual(astar.failure_backtrack(3, 6, state['failure_counts']), 41)


    def test_lane_merge_counts_only_new_failures(self):
        from combat_solver_cli.lanes import _merge_checkpoints
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outputs, inputs = [], []
            base = {'failure_counts': {'3:6': 2}, 'route_visits': {'opening': 2}}
            for i, count in enumerate((3, 4)):
                out = root/str(i)
                out.mkdir()
                source = root/f'input-{i}.json'
                astar.save_frontier(source, [], 'Ironclad', 'fixed', 12, 0, base)
                astar.save_frontier(out/'frontier.json', [], 'Ironclad', 'fixed', 12, 0,
                                    {'failure_counts': {'3:6': count}, 'route_visits': {'opening': count}})
                outputs.append(out)
                inputs.append(source)
            _, state, recovered = _merge_checkpoints(outputs, inputs,
                [{'status': 'budget_exhausted'}]*2, 'Ironclad', 'fixed', 12, 0, root, base)
            self.assertEqual(state['failure_counts'], {'3:6': 5})
            self.assertEqual(state['route_visits'], {'opening': 5})
            self.assertEqual(recovered, [])


if __name__ == '__main__': unittest.main()
