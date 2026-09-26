import unittest
from combat_solver_cli.astar import preference


class SelectionPreferenceTests(unittest.TestCase):
    def test_exhaust_preserves_strong_card_before_basic_strike(self):
        strike = {'ref': 'strike', 'entity_type': 'card', 'content_id': 'CARD.STRIKE_IRONCLAD',
                  'cost': 1, 'stats': {'Damage': 6}, 'zone': 'hand'}
        scaling = {'ref': 'scaling', 'entity_type': 'card', 'content_id': 'CARD.SETUP_STRIKE',
                   'cost': 1, 'stats': {'Damage': 7, 'StrengthPower': 3}, 'zone': 'hand'}
        frame = {'public': {'phase': 'card_select', 'selection_context': {'operation': 'exhaust'},
                 'entities': [{'entity_type': 'player', 'hp': 50, 'max_hp': 80}, strike, scaling]}}
        def candidate(ref): return {'verb': 'SELECT_ONE', 'source_refs': [ref]}
        self.assertGreater(preference(frame, candidate('strike')), preference(frame, candidate('scaling')))


if __name__ == '__main__': unittest.main()
