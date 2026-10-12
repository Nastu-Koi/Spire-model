"""The solver pays the potion prices of the request, in the search itself."""
import os
import time

import pytest


@pytest.mark.engine
def test_search_pays_requested_potion_prices():
    config = os.environ.get('COMBAT_SOLVER_CONFIG')
    if not config:
        pytest.skip('Pinned native config required')
    from combat_solver_cli.client import SolverEngine
    with SolverEngine(config) as engine:
        engine.send(dict(cmd='start_run', character='IRONCLAD', seed='potion-costs-check', ascension=0))
        # One potion of the solver's base tier and one of its dearest.
        engine.send(dict(cmd='set_player', hp=60, potions=['FIRE_POTION', 'SWIFT_POTION']))
        frame = engine.send(dict(cmd='enter_room', type='combat', encounter='TOADPOLES_WEAK', decision_protocol=True))
        deadline = time.monotonic() + 15
        while frame.get('boundary') == 'waiting' and time.monotonic() < deadline:
            time.sleep(.01)
            frame = engine.send(dict(cmd='advance_to_boundary'))
        belt = {e['content_id'].split('.', 1)[1]: e['slot'] for e in frame['public']['entities']
                if e.get('entity_type') == 'potion'}
        assert set(belt) == {'FIRE_POTION', 'SWIFT_POTION'}

        def paid(potion, **prices):
            # The route has to use this potion and no other: what it cost is its price.
            reply = engine.solve(frame, budget_ms=300, potion_policy='Disabled', **prices,
                                 potion_directives=[dict(slot=belt[potion], potion=potion, directive='Force')])
            assert reply['type'] == 'solver_result', reply
            assert [a['potion_id'] for a in reply['actions'] if a['potion_id']] == [potion]
            return reply['potion_costs'], reply['potion_strategic_cost']

        assert paid('FIRE_POTION') == ([9, 14, 18], 9)
        assert paid('FIRE_POTION', potion_costs=[5, 7, 9]) == ([5, 7, 9], 5)
        assert paid('SWIFT_POTION', potion_costs=[5, 7, 9]) == ([5, 7, 9], 9)
        # The next request without prices searches at the solver's own again.
        assert paid('SWIFT_POTION') == ([9, 14, 18], 18)
        refused = engine.solve(frame, budget_ms=300, potions=True, potion_costs=[9, 7, 5])
        assert refused['type'] == 'error' and 'potion_costs' in refused['message']
