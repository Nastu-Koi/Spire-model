"""`reuse_route`: a turn that starts in exactly the state the last search predicted continues
that route without searching; without the flag every turn searches. The worker leaves null
fields out of its replies."""
import os
import time

import pytest


def settle(engine, frame):
    deadline = time.monotonic() + 15
    while frame.get('boundary') == 'waiting' and time.monotonic() < deadline:
        time.sleep(.01)
        frame = engine.send(dict(cmd='advance_to_boundary'))
    assert frame.get('type') == 'decision_frame', frame
    return frame


def fight(config, budget_ms=1000, **options):
    """The search part of every solver step of one fight, in order."""
    from combat_solver_cli.client import SolverEngine
    with SolverEngine(config) as engine:
        engine.send(dict(cmd='start_run', character='IRONCLAD', seed='route-reuse-check', ascension=0))
        engine.send(dict(cmd='set_player', hp=80))
        frame = settle(engine, engine.send(dict(cmd='enter_room', type='combat', encounter='TOADPOLES_WEAK',
                                                decision_protocol=True)))
        searches = []
        while frame['boundary'] != 'terminal' and frame['public']['phase'] == 'combat':
            reply = engine.step(frame, budget_ms=budget_ms, potions=True, reuse_turn_plan=True, **options)
            assert reply['type'] == 'solver_step', reply
            if reply.get('search') is not None:
                searches.append(reply['search'])
            frame = settle(engine, reply['frame'])
        return searches


@pytest.mark.engine
def test_later_turns_continue_the_searched_route():
    config = os.environ.get('COMBAT_SOLVER_CONFIG')
    if not config:
        pytest.skip('Pinned native config required')
    plain = fight(config)
    assert len(plain) >= 2, 'the fight has to last more than one turn'
    assert all(s.get('reused_from_turn') is None and s['expanded_nodes'] > 0 for s in plain)
    reused = fight(config, reuse_route=True)
    continued = [s for s in reused if s.get('reused_from_turn') is not None]
    assert continued, [s['expanded_nodes'] for s in reused]
    # A continued turn searched nothing and came from a search on an earlier turn.
    assert all(s['expanded_nodes'] == 0 for s in continued)
    assert reused[0].get('reused_from_turn') is None


@pytest.mark.engine
def test_routes_of_searches_cut_by_the_budget_are_not_continued():
    config = os.environ.get('COMBAT_SOLVER_CONFIG')
    if not config:
        pytest.skip('Pinned native config required')
    # So small a budget cuts the first search of the fight; the short ones near its end finish.
    searches = fight(config, budget_ms=20, reuse_route=True, reuse_within_budget=True)
    assert any(s['within_budget'] is False for s in searches), searches
    searched = None
    for s in searches:
        if s.get('reused_from_turn') is None:
            searched = s
        else:
            assert searched['within_budget'] is True
