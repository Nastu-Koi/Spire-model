"""Removal rewards outside combat: opening commits, declining means not opening."""
import json
import os
import time
from collections import Counter
from pathlib import Path

import pytest

from combat_solver_cli.client import SolverEngine
from combat_solver_cli.search_support import collapse_noncombat_cancels, resolve
from combat_solver_cli.trajectory import native_replay_error
from model.protocol import execution_command, validate_frame

PREFIX = Path('combat_solver_cli/tests/native/fixtures/reward-cancel-prefix.json')


def session(config, records):
    """Replay records and return (engine, frame, helpers) at the resulting boundary."""
    data = json.loads(PREFIX.read_text())
    engine = SolverEngine(config)

    def settle(frame):
        deadline = time.monotonic() + 5
        while frame.get('boundary') == 'waiting' and time.monotonic() < deadline:
            time.sleep(.01)
            frame = engine.send({'cmd': 'advance_to_boundary'})
        validate_frame(frame)
        assert frame['boundary'] != 'waiting'
        return frame

    # The fixture holds a rest-site smith -> cancel detour, which the protocol does
    # not offer; it is collapsed. The kept TAKE_REWARD reaches the removal prompt.
    frame = engine.reset(data['character'], data['seed'], 0)
    for record in collapse_noncombat_cancels(data['records'][:records]):
        frame = settle(frame)
        candidate = resolve(frame, record['action'])
        frame = engine.send(execution_command(frame, candidate['candidate_ref']))
    return engine, settle(frame), settle


def act(engine, settle, frame, verb):
    candidate = next(c for c in frame['legal']['candidates'] if c['verb'] == verb)
    return settle(engine.send(execution_command(frame, candidate['candidate_ref'])))


def deck(frame):
    return Counter(e['content_id'] for e in frame['public']['entities']
                   if e.get('entity_type') == 'card' and e.get('zone') == 'deck')


def removal(frame):
    refs = {e['ref'] for e in frame['public']['entities']
            if e.get('content_id') == 'CardRemovalReward'}
    return [c for c in frame['legal']['candidates']
            if c['verb'] == 'TAKE_REWARD' and set(c['source_refs']) & refs]


def config():
    value = os.environ.get('COMBAT_SOLVER_CONFIG')
    if not value:
        pytest.skip('Pinned native configuration required')
    return value


@pytest.mark.engine
def test_opened_removal_reward_offers_no_cancel_and_completes():
    engine, frame, settle = session(config(), -2)
    with engine:
        assert frame['public']['phase'] == 'card_select'
        original = deck(frame)
        assert sum(original.values()) == 21
        assert frame['public']['selection_context']['can_cancel'] is False
        assert 'CANCEL' not in {c['verb'] for c in frame['legal']['candidates']}
        candidate = next(c for c in frame['legal']['candidates'] if c['verb'] == 'SELECT_ONE')
        chosen = next(e['content_id'] for e in frame['public']['entities']
                      if e.get('ref') == candidate['source_refs'][0])
        frame = settle(engine.send(execution_command(frame, candidate['candidate_ref'])))
        assert [c['verb'] for c in frame['legal']['candidates']] == ['FINISH_SELECTION']
        frame = act(engine, settle, frame, 'FINISH_SELECTION')
        expected = original.copy()
        expected.subtract([chosen])
        assert deck(frame) == +expected
        assert not removal(frame)
        if frame['public']['phase'] == 'rewards':
            frame = act(engine, settle, frame, 'LEAVE_REWARDS')
        assert frame['public']['phase'] == 'map'
        assert native_replay_error(engine) is None


@pytest.mark.engine
def test_removal_reward_is_declined_by_not_opening_it():
    engine, frame, settle = session(config(), -3)
    with engine:
        assert frame['public']['phase'] == 'rewards'
        assert removal(frame)
        original = deck(frame)
        frame = act(engine, settle, frame, 'LEAVE_REWARDS')
        assert frame['public']['phase'] == 'map'
        assert deck(frame) == original
        assert native_replay_error(engine) is None
