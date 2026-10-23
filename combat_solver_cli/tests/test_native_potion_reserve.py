"""A Smart search may keep part of its budget from the potion-free search for the potion
layers (`potion_reserve`), when the belt holds a potion."""
import os
import time

import pytest


def logged(engine, ask):
    """The reply to `ask()` and what the worker wrote to stderr meanwhile."""
    fd = engine.stderr.fileno()
    start = os.fstat(fd).st_size
    reply = ask()
    return reply, os.pread(fd, os.fstat(fd).st_size - start, start).decode('utf-8', 'replace')


@pytest.mark.engine
def test_a_search_keeps_the_reserve_for_the_potion_layers():
    config = os.environ.get('COMBAT_SOLVER_CONFIG')
    if not config:
        pytest.skip('Pinned native config required')
    from combat_solver_cli.client import SolverEngine
    with SolverEngine(config) as engine:
        engine.send(dict(cmd='start_run', character='IRONCLAD', seed='potion-reserve-check', ascension=0))
        engine.send(dict(cmd='set_player', hp=60, potions=['FIRE_POTION', 'BLOCK_POTION']))
        # A boss the starter deck cannot beat without losing HP: the potion layers have work to do.
        frame = engine.send(dict(cmd='enter_room', type='combat', encounter='VANTOM_BOSS', decision_protocol=True))
        deadline = time.monotonic() + 15
        while frame.get('boundary') == 'waiting' and time.monotonic() < deadline:
            time.sleep(.01)
            frame = engine.send(dict(cmd='advance_to_boundary'))

        reply, err = logged(engine, lambda: engine.solve(frame, budget_ms=4000, potions=True, beam_portfolio=False,
                                                         potion_reserve=0.5))
        assert reply['type'] == 'solver_result', reply
        assert reply['potion_reserve_ms'] == 2000
        # The potion-free part stopped at half the budget; the audits started with time left.
        audits = [line for line in err.splitlines() if 'potion_reserve_audits' in line]
        assert audits and int(audits[0].split('elapsed_ms=')[1].split()[0]) < 3500, err[-2000:]
        assert 'SMART_POTION_GRADIENT layer=' in err

        # Without a share, or when no potion may be used, nothing is kept.
        reply, err = logged(engine, lambda: engine.solve(frame, budget_ms=1000, potions=True, beam_portfolio=False))
        assert reply['potion_reserve_ms'] == 0 and 'potion_reserve_audits' not in err
        reply = engine.solve(frame, budget_ms=1000, potions=False, potion_reserve=0.5)
        assert reply['potion_reserve_ms'] == 0
        refused = engine.solve(frame, budget_ms=1000, potions=True, potion_reserve=0.95)
        assert refused['type'] == 'error' and 'potion_reserve' in refused['message']


@pytest.mark.engine
def test_an_empty_belt_keeps_nothing():
    config = os.environ.get('COMBAT_SOLVER_CONFIG')
    if not config:
        pytest.skip('Pinned native config required')
    from combat_solver_cli.client import SolverEngine
    with SolverEngine(config) as engine:
        engine.send(dict(cmd='start_run', character='IRONCLAD', seed='potion-reserve-check', ascension=0))
        engine.send(dict(cmd='set_player', hp=60, potions=[]))
        frame = engine.send(dict(cmd='enter_room', type='combat', encounter='TOADPOLES_WEAK', decision_protocol=True))
        deadline = time.monotonic() + 15
        while frame.get('boundary') == 'waiting' and time.monotonic() < deadline:
            time.sleep(.01)
            frame = engine.send(dict(cmd='advance_to_boundary'))
        reply = engine.solve(frame, budget_ms=1000, potions=True, potion_reserve=0.5)
        assert reply['type'] == 'solver_result', reply
        assert reply['potion_reserve_ms'] == 0
