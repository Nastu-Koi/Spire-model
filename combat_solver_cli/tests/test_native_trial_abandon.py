"""Trial abandonment is a legitimate loss, not a native engine fault."""
import json
import os
import time
from pathlib import Path

import pytest
from combat_solver_cli.client import SolverEngine
from combat_solver_cli.search_support import resolve
from combat_solver_cli.trajectory import native_replay_error
from model.protocol import execution_command, validate_frame


@pytest.mark.engine
@pytest.mark.parametrize('abandon', [True, False])
def test_trial_reject_choices(abandon):
    config = os.environ.get('COMBAT_SOLVER_CONFIG')
    if not config:
        pytest.skip('Pinned native configuration required')
    data = json.loads(Path('combat_solver_cli/tests/native/fixtures/trial-abandon-prefix.json').read_text())
    with SolverEngine(config) as engine:
        def settle(frame):
            deadline = time.monotonic() + 5
            while frame.get('boundary') == 'waiting' and time.monotonic() < deadline:
                time.sleep(.01)
                frame = engine.send({'cmd': 'advance_to_boundary'})
            validate_frame(frame)
            assert frame['boundary'] != 'waiting'
            return frame
        frame = engine.reset(data['character'], data['seed'], 0)
        for record in data['records'][:-1]:
            frame = settle(frame)
            candidate = resolve(frame, record['action'])
            frame = engine.send(execution_command(frame, candidate['candidate_ref']))
        frame = settle(frame)
        assert any(e.get('content_id') == 'EVENT.TRIAL' for e in frame['public']['entities'])
        verb = 'ABANDON_RUN' if abandon else 'CHOOSE_EVENT_OPTION'
        candidate = next(c for c in frame['legal']['candidates'] if c['verb'] == verb)
        frame = settle(engine.send(execution_command(frame, candidate['candidate_ref'])))
        assert native_replay_error(engine) is None
        if abandon:
            assert frame['boundary'] == 'terminal'
            assert frame['public']['outcome']['victory'] is False
            assert not frame['legal']['candidates']
        else:
            assert frame['boundary'] == 'decision'
            assert frame['public']['phase'] in {'event', 'map'}
