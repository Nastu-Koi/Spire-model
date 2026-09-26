import heapq
import tempfile
import unittest
from pathlib import Path
from combat_solver_cli.astar import Prefix, pop_frontier, save_frontier, load_frontier
from combat_solver_cli.diversity import RouteDiversity, partition_routes


def action(name, verb='CHOOSE_EVENT_OPTION'):
    return {'verb': verb, 'source_refs': [name], 'target_refs': []}


def prefix(name, parent=None, verb='CHOOSE_EVENT_OPTION', act=1, floor=1):
    return Prefix(parent, '', action(name, verb), 'astar', 1 if parent is None else parent.length+1, 'event', act, floor)


class DiversityTests(unittest.TestCase):
    def test_untried_opening_beats_deeper_repeat_without_pruning(self):
        d = RouteDiversity()
        a = (1., 0, 0., prefix('A'), None)
        b = (100., 1, 0., prefix('B'), None)
        d.begin(); d.observe(a)
        q = [a,b]; heapq.heapify(q)
        self.assertIs(pop_frontier(q, diversity=d), b)
        self.assertEqual(q, [a])

    def test_same_opening_explores_untried_early_map(self):
        root = prefix('A')
        a = (1., 0, 0., prefix('left', root, 'MOVE_TO_NODE'), None)
        b = (100., 1, 0., prefix('right', root, 'MOVE_TO_NODE'), None)
        d = RouteDiversity(); d.begin(); d.observe(a); d.observe(a)
        self.assertEqual(list(d.visits.values()), [1,1])
        self.assertEqual(min([a,b], key=d.rank), b)

    def test_pending_opening_and_replayed_history_have_same_key(self):
        d = RouteDiversity()
        self.assertEqual(d.keys(None, action('A')), d.keys(prefix('A')))
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'frontier.json';d.begin();d.observe((0,0,0,prefix('A'),None))
            save_frontier(p, [(0,0,0,prefix('A'),None)], 'Ironclad','seed',12,0,{'route_visits':d.visits})
            q,s=load_frontier(p,'Ironclad','seed',12,0,with_state=True)
            restored=RouteDiversity(s['route_visits'])
            self.assertEqual(restored.rank(q[0])[:3], (1,1,1))

    def test_taking_and_skipping_early_reward_are_distinct(self):
        d=RouteDiversity();root=prefix('A')
        take=prefix('card',root,'TAKE_CARD_REWARD',floor=2)
        skip=prefix('skip',root,'SKIP',floor=2)
        self.assertEqual(len(d.keys(take)),2)
        self.assertEqual(len(d.keys(skip)),2)
        self.assertNotEqual(d.keys(take),d.keys(skip))

    def test_late_choices_do_not_create_new_early_routes(self):
        d=RouteDiversity();a=prefix('A');b=prefix('late',a,act=2,floor=1)
        self.assertEqual(d.keys(a),d.keys(b))

    def test_partition_separates_openings_and_preserves_every_node(self):
        q=[(float(i),i,0.,prefix('A' if i%2==0 else 'B'),None) for i in range(8)]
        shards=partition_routes(q,2)
        self.assertEqual(sorted(n[1] for s in shards for n in s),list(range(8)))
        for s in shards:self.assertEqual(len({n[3].action['source_refs'][0] for n in s}),1)

    def test_partition_can_fill_lanes_when_only_one_route_exists(self):
        q=[(float(i),i,0.,prefix('A'),None) for i in range(8)]
        shards=partition_routes(q,4)
        self.assertTrue(all(shards))
        self.assertEqual(sorted(n[1] for s in shards for n in s),list(range(8)))


if __name__ == '__main__': unittest.main()
