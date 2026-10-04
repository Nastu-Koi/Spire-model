"""Reward UI must keep the game's AnyTime potion actions, including healing."""
import os
import time
import pytest


@pytest.mark.engine
def test_blood_potion_usable_from_rewards():
    if os.environ.get('SPIRE_CODEX_DATA_NATIVE_TESTS')!='1':
        pytest.skip('Set SPIRE_CODEX_DATA_NATIVE_TESTS=1 with the local solver configured')
    from combat_solver_cli.client import SolverEngine
    from model.protocol import execution_command
    from combat_solver_cli.trajectory import native_replay_error
    with SolverEngine() as e:
        e.send(dict(cmd='start_run',character='IRONCLAD',seed='reward-potion-check',ascension=0))
        e.send(dict(cmd='set_player',hp=60,potions=['BLOOD_POTION','ENERGY_POTION']))
        # Finish a native fight with held potions; leave its rewards open.
        f=e.send(dict(cmd='enter_room',type='combat',encounter='TOADPOLES_WEAK',decision_protocol=True))
        for _ in range(100):
            if f.get('boundary')=='waiting':
                time.sleep(.01);f=e.send(dict(cmd='advance_to_boundary'));continue
            if f['public']['phase']=='rewards':break
            reply=e.step(f,budget_ms=100,potions=False,reuse_turn_plan=True)
            assert reply['type']=='solver_step'
            f=reply['frame']
        assert f['public']['phase']=='rewards'
        before=next(x for x in f['public']['entities'] if x.get('entity_type')=='player')['hp']
        entities={x['ref']:x for x in f['public']['entities'] if 'ref' in x}
        uses=[c for c in f['legal']['candidates'] if c['verb']=='USE_POTION']
        assert len(uses)==1 and entities[uses[0]['source_refs'][0]]['content_id']=='POTION.BLOOD_POTION'
        f=e.send(execution_command(f,uses[0]['candidate_ref']))
        deadline=time.monotonic()+15
        while f.get('boundary')=='waiting' and time.monotonic()<deadline:
            time.sleep(.01);f=e.send(dict(cmd='advance_to_boundary'))
        after=next(x for x in f['public']['entities'] if x.get('entity_type')=='player')['hp']
        assert after>before
        assert f['public']['phase']=='rewards'
        assert not any(x.get('content_id')=='POTION.BLOOD_POTION' for x in f['public']['entities'])
        assert native_replay_error(e) is None
