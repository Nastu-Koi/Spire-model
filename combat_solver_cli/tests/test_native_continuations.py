"""Native work that continues after the action: no boundary may be published in its gaps.

Both prefixes come from Regent failures in the M0 generation run:
- Lizard Tail: the enemy turn takes the player to 0 HP; the relic revives them and
  Toasty Mittens then asks for a turn-start exhaust. Zero HP is not a death until
  the native Died event, so no terminal frame may appear before the selection.
  The prefix follows that run into a fight with both relics and ends turns until
  the next enemy attack is lethal; the fixture records the HP before and after.
- Lord's Parasol: entering the shop buys everything, opening Kifuda's enchant
  selection and then the card removal. The shop must not be published between them.
"""
import json
import os
import time
from pathlib import Path

import pytest

from combat_solver_cli.client import SolverEngine
from combat_solver_cli.search_support import resolve
from model.protocol import execution_command, validate_frame

FIXTURES = Path('combat_solver_cli/tests/native/fixtures')


def config():
    value = os.environ.get('COMBAT_SOLVER_CONFIG')
    if not value:
        pytest.skip('Pinned native configuration required')
    return value


def poll(engine, frame):
    """Advance to the next published boundary, returning every frame seen."""
    seen = [frame]
    deadline = time.monotonic() + 10
    while frame.get('boundary') == 'waiting' and time.monotonic() < deadline:
        time.sleep(.01)
        frame = engine.send({'cmd': 'advance_to_boundary'})
        seen.append(frame)
    validate_frame(frame)
    assert frame['boundary'] != 'waiting'
    return seen


def replay(engine, name):
    data = json.loads((FIXTURES / name).read_text())
    frame = engine.reset(data['character'], data['seed'], 0)
    for record in data['records']:
        frame = poll(engine, frame)[-1]
        candidate = resolve(frame, record['action'])
        frame = engine.send(execution_command(frame, candidate['candidate_ref']))
    return poll(engine, frame)[-1]


def player(frame):
    return next(e for e in frame['public']['entities'] if e.get('entity_type') == 'player')


@pytest.mark.engine
def test_death_prevention_is_not_published_as_terminal():
    with SolverEngine(config()) as engine:
        expected = json.loads((FIXTURES / 'lizard-tail-turn-start-prefix.json').read_text())
        frame = replay(engine, 'lizard-tail-turn-start-prefix.json')
        assert frame['public']['phase'] == 'combat' and player(frame)['hp'] == expected['hp_before_death']
        end = next(c for c in frame['legal']['candidates'] if c['verb'] == 'END_TURN')
        seen = poll(engine, engine.send(execution_command(frame, end['candidate_ref'])))
        assert [f['boundary'] for f in seen[:-1]] == ['waiting'] * (len(seen) - 1)
        frame = seen[-1]
        assert frame['public']['phase'] == 'card_select'
        assert frame['public']['selection_context']['operation'] == 'exhaust'
        assert player(frame)['hp'] == expected['hp_after_revive']


@pytest.mark.engine
def test_solver_turn_start_choice_survives_death_prevention():
    with SolverEngine(config()) as engine:
        frame = replay(engine, 'lizard-tail-turn-start-prefix.json')
        result = engine.step(frame, budget_ms=250, boss_budget_ms=5000, potions=True, reuse_turn_plan=True)
        assert result.get('type') == 'solver_step', result.get('message')
        assert result['frame']['boundary'] != 'terminal'


@pytest.mark.engine
def test_lords_parasol_selections_are_not_interleaved_with_the_shop():
    with SolverEngine(config()) as engine:
        # The prefix ends by finishing Kifuda's enchant selection inside the
        # Parasol purchase chain; the card removal must follow directly.
        frame = replay(engine, 'lords-parasol-prefix.json')
        context = frame['public']['selection_context']
        assert frame['public']['phase'] == 'card_select'
        assert (context['operation'], context['min_total'], context['max_total']) == ('remove', 1, 1)
        assert context['can_cancel'] is False
        pick = next(c for c in frame['legal']['candidates'] if c['verb'] == 'SELECT_ONE')
        frame = poll(engine, engine.send(execution_command(frame, pick['candidate_ref'])))[-1]
        done = next(c for c in frame['legal']['candidates'] if c['verb'] == 'FINISH_SELECTION')
        frame = poll(engine, engine.send(execution_command(frame, done['candidate_ref'])))[-1]
        assert frame['public']['phase'] == 'shop'
        # Every published shop candidate is executable once the chain is over.
        for candidate in frame['legal']['candidates']:
            assert candidate['verb'] in {'BUY_ITEM', 'DISCARD_POTION', 'LEAVE_ROOM'}
        buy = next(c for c in frame['legal']['candidates'] if c['verb'] == 'BUY_ITEM')
        frame = poll(engine, engine.send(execution_command(frame, buy['candidate_ref'])))[-1]
        assert frame['boundary'] == 'decision'
