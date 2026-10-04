"""Event reward commands must finish before publishing travel decisions."""
import json
import os
import time
from pathlib import Path

import pytest

from combat_solver_cli.client import SolverEngine
from combat_solver_cli.search_support import resolve, state_key
from combat_solver_cli.trajectory import native_replay_error
from model.protocol import execution_command, validate_frame


@pytest.mark.engine
@pytest.mark.parametrize('take_reward', [True, False])
def test_battleworn_reward_discard_cannot_leave_room(take_reward):
    config = os.environ.get('COMBAT_SOLVER_CONFIG')
    if not config:
        pytest.skip('Pinned native configuration required')
    data = json.loads(Path('combat_solver_cli/tests/native/fixtures/battleworn-rewards-prefix.json').read_text())
    with SolverEngine(config) as engine:
        def settle(frame):
            deadline = time.monotonic() + 5
            while frame.get('boundary') == 'waiting' and time.monotonic() < deadline:
                time.sleep(.01)
                frame = engine.send({'cmd': 'advance_to_boundary'})
            validate_frame(frame)
            assert frame['boundary'] != 'waiting'
            assert native_replay_error(engine) is None
            return frame

        def act(frame, verb):
            candidate = next(c for c in frame['legal']['candidates'] if c['verb'] == verb)
            return settle(engine.send(execution_command(frame, candidate['candidate_ref'])))

        # records[:split] reach the event's potion reward. The capture then discards
        # a potion and takes the reward, which this test does itself both ways;
        # records[resume:] continue from the map through the treasure room into
        # the next fight.
        split, resume, potion = data['reward_step'], data['resume_step'], data['reward_potion']
        frame = engine.reset(data['character'], data['seed'], 0)
        for record in data['records'][:split]:
            frame = settle(frame)
            assert state_key(frame) == record['before_hash']
            candidate = resolve(frame, record['action'])
            frame = engine.send(execution_command(frame, candidate['candidate_ref']))
        frame = settle(frame)
        assert frame['public']['phase'] == 'rewards'
        frame = act(frame, 'DISCARD_POTION')
        assert frame['public']['phase'] == 'rewards', 'Discarding a potion must not open the map'
        assert any(e.get('content_id') == potion
                   and e['entity_type'] == 'reward' for e in frame['public']['entities'])
        frame = act(frame, 'TAKE_REWARD' if take_reward else 'LEAVE_REWARDS')
        # Some native continuations present the completed event before the map.
        if frame['public']['phase'] == 'event':
            frame = act(frame, 'CHOOSE_EVENT_OPTION')
        assert frame['public']['phase'] == 'map'
        candidate = resolve(frame, data['records'][resume]['action'])
        frame = settle(engine.send(execution_command(frame, candidate['candidate_ref'])))
        potions = [e for e in frame['public']['entities'] if e['entity_type'] == 'potion']
        assert any(e.get('content_id') == potion for e in potions) == take_reward
        # Continue the recorded treasure/next-combat actions: an abandoned event
        # reward task would surface its exception only in that fight.
        for record in data['records'][resume + 1:]:
            candidate = resolve(frame, record['action'])
            frame = settle(engine.send(execution_command(frame, candidate['candidate_ref'])))
        assert frame['public']['phase'] == 'combat'
        assert native_replay_error(engine) is None
