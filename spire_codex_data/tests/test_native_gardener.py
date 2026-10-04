"""Native ABI regression: Enlarge must finish and actually grant Strength."""
import os
import time
import pytest


@pytest.mark.engine
def test_gardener_enlarge_reaches_next_turn():
    if os.environ.get('SPIRE_CODEX_DATA_NATIVE_TESTS')!='1':
        pytest.skip('Set SPIRE_CODEX_DATA_NATIVE_TESTS=1 with the local solver configured')
    from combat_solver_cli.client import SolverEngine
    from model.protocol import execution_command
    from combat_solver_cli.trajectory import native_replay_error
    with SolverEngine() as engine:
        engine.send(dict(cmd='start_run',character='IRONCLAD',seed='gardener-abi-check',ascension=0))
        frame=engine.send(dict(cmd='enter_room',type='combat',encounter='PHANTASMAL_GARDENERS_ELITE',decision_protocol=True))
        c=next(c for c in frame['legal']['candidates'] if c['verb']=='END_TURN')
        frame=engine.send(execution_command(frame,c['candidate_ref']))
        deadline=time.monotonic()+15
        while frame.get('boundary')=='waiting' and time.monotonic()<deadline:
            time.sleep(.01)
            frame=engine.send(dict(cmd='advance_to_boundary'))
        assert frame['boundary']=='decision'
        assert frame['public']['phase']=='combat'
        p=next(e for e in frame['public']['entities'] if e.get('entity_type')=='player')
        assert p['round']==2
        assert any(e.get('content_id')=='POWER.STRENGTH_POWER' for e in frame['public']['entities'])
        assert native_replay_error(engine) is None
