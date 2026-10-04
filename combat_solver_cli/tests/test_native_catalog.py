"""Native semantic regressions; opt in with COMBAT_SOLVER_CONFIG."""
import json
import os
import time
from pathlib import Path

import pytest

from combat_solver_cli.client import SolverEngine
from combat_solver_cli.search_support import collapse_noncombat_cancels, resolve
from model.config import ModelConfig
from model.protocol import execution_command, validate_frame
from model.representation import Vocabulary
from combat_solver_cli.trajectory import native_replay_error

SOLVER = {'budget_ms': 250, 'boss_budget_ms': 5000, 'potions': True, 'reuse_turn_plan': True}


@pytest.fixture
def native_config():
    config = os.environ.get('COMBAT_SOLVER_CONFIG')
    if not config:
        pytest.skip('Set COMBAT_SOLVER_CONFIG to a pinned native configuration')
    return config


@pytest.mark.engine
def test_catalog_preserves_outside_choice_ids(native_config):
    with SolverEngine(native_config) as engine:
        initial = engine.reset('Ironclad', 'catalog-regression', 0)
        catalog = engine.send({'cmd': 'public_catalog'})
    vocabulary = Vocabulary.from_frames([initial, catalog], ModelConfig(vocabulary_size=8192))
    for symbol in ['content_id=HEAL', 'content_id=SMITH', 'content_id=HATCH',
                   'content_id=COLORFUL_PHILOSOPHERS.pages.INITIAL.options.DEFECT',
                   'content_id=NEOW.pages.INITIAL.options.CURSED_PEARL',
                   'content_id=DARV.pages.INITIAL.options.SNECKO_EYE']:
        assert vocabulary.encode(symbol) != 1, symbol


@pytest.mark.engine
def test_rolling_boulder_advances_and_applies_increasing_damage(native_config):
    path = Path('combat_solver_cli/tests/native/fixtures/rolling-boulder-prefix.json')
    if not path.exists():
        pytest.skip('Captured RollingBoulder prefix is not available')
    data = json.loads(path.read_text())
    with SolverEngine(native_config) as engine:
        def settle(frame):
            deadline = time.monotonic() + 5
            while frame['boundary'] == 'waiting' and time.monotonic() < deadline:
                time.sleep(.01)
                frame = engine.send({'cmd': 'advance_to_boundary'})
            assert frame['boundary'] != 'waiting', native_replay_error(engine)
            validate_frame(frame)
            return frame

        def state(frame):
            entities = frame['public']['entities']
            hp = next(e['hp'] for e in entities if e.get('entity_type') == 'enemy')
            amount = next(e['stacks'] for e in entities
                          if e.get('content_id') == 'POWER.ROLLING_BOULDER_POWER')
            return hp, amount

        frame = engine.reset(data['character'], data['seed'], 0)
        for record in collapse_noncombat_cancels(data['records'][:-1]):
            frame = settle(frame)
            candidate = resolve(frame, record['action'])
            frame = engine.send(execution_command(frame, candidate['candidate_ref']))
        frame = settle(frame)
        assert state(frame) == (211, 5)
        for expected in [(206, 10), (196, 15)]:
            end = next(c for c in frame['legal']['candidates'] if c['verb'] == 'END_TURN')
            frame = settle(engine.send(execution_command(frame, end['candidate_ref'])))
            assert frame['public']['phase'] == 'combat'
            assert state(frame) == expected
        assert native_replay_error(engine) is None


@pytest.mark.engine
def test_implicit_duplicate_is_consumed_after_real_copy(native_config):
    path = Path('combat_solver_cli/tests/native/fixtures/regent-choice-prefix.json')
    if not path.exists():
        pytest.skip('Captured Regent prefix is not available')
    data = json.loads(path.read_text())
    with SolverEngine(native_config) as engine:
        frame = engine.reset(data['character'], data['seed'], 0)
        for record in collapse_noncombat_cancels(data['records'][:-1]):
            while frame['boundary'] == 'waiting':
                frame = engine.send({'cmd': 'advance_to_boundary'})
            candidate = resolve(frame, record['action'])
            frame = engine.send(execution_command(frame, candidate['candidate_ref']))
        for _ in range(150):
            while frame['boundary'] == 'waiting':
                frame = engine.send({'cmd': 'advance_to_boundary'})
            validate_frame(frame)
            if frame['boundary'] == 'terminal' or frame['public']['phase'] not in {'combat', 'card_select', 'card_reward'}:
                break
            # Narrow beam exercises the recorded hammer choice reliably;
            # production keeps its existing beam portfolio and time budgets.
            reply = engine.step(frame, **(SOLVER | {'beam_width': 1, 'beam_portfolio': False}))
            assert reply['type'] == 'solver_step', reply
            frame = reply['frame']
        else:
            pytest.fail('Combat did not finish within the regression step limit')
        engine.stderr.flush()
        engine.stderr.seek(0)
        assert 'status=selected mode=automatic cards=EXTERMINATE' in engine.stderr.read()


@pytest.mark.engine
def test_necrobinder_boss_portfolio_deadline_returns_legal_step(native_config):
    data = json.loads(Path('combat_solver_cli/tests/native/fixtures/solver-cancel-prefix.json').read_text())
    with SolverEngine(native_config) as engine:
        def settle(frame):
            deadline = time.monotonic() + 5
            while frame['boundary'] == 'waiting' and time.monotonic() < deadline:
                time.sleep(.01)
                frame = engine.send({'cmd': 'advance_to_boundary'})
            assert frame['boundary'] != 'waiting'
            return frame

        frame = engine.reset(data['character'], data['seed'], 0)
        for record in data['records']:
            frame = settle(frame)
            candidate = resolve(frame, record['action'])
            frame = engine.send(execution_command(frame, candidate['candidate_ref']))
        frame = settle(frame)
        reply = engine.step(frame, **SOLVER)
        assert reply['type'] == 'solver_step', reply
        assert reply['candidate_ref'] in {c['candidate_ref'] for c in frame['legal']['candidates']}
        validate_frame(settle(reply['frame']))
        assert native_replay_error(engine) is None
