"""Choices of the first turn's setup (Gambling Chip, Toasty Mittens, Toolbox at the start of a
fight) come from a solver search of the setup, as the solver mod's own turn-setup takeover
makes them in the game, not from a rule outside the solver. The fight enters from an anchor
save, the way demonstration battles do; the worker leaves null fields out of its replies."""
from contextlib import contextmanager
import os

import pytest


@contextmanager
def entered(config, relics, character='Ironclad'):
    """A worker at the first boundary of a fight entered holding `relics`."""
    from combat_solver_cli.client import SolverEngine
    from spire_codex_data import anchor
    from spire_codex_data.anchor import settle

    with SolverEngine(config) as engine:
        settle(engine, engine.send(dict(cmd='start_run', character=character, seed='turn-setup-check',
                                        ascension=0, decision_protocol=True)))
        base = engine.send(dict(cmd='anchor_state'))['save']
    player = base['players'][0]
    owned = player['relics'] + [dict(id='RELIC.' + r, floor_added_to_deck=1) for r in relics]

    def node(kind, room_type):
        room = dict(room_type=room_type, turns_taken=3)
        if kind == 'monster':
            room.update(model_id='ENCOUNTER.TOADPOLES_WEAK', monster_ids=[])
        return dict(map_point_type=kind, rooms=[room], player_stats=[dict(current_hp=70, max_hp=80, current_gold=99)])

    run = dict(seed='turn-setup-check', ascension=0, players=[dict(deck=player['deck'], relics=owned)],
               map_point_history=[[node('ancient', 'event'), node('monster', 'monster')]])
    state = dict(deck=player['deck'], relics=owned, potions=[], hp=70, max_hp=player['max_hp'], gold=99)
    save = anchor.build_save(base, state, run, 1, 2, run['map_point_history'][0][1], 'turn-setup-check')
    spot = dict(floor=2, map_point_type='monster', second_boss=False, route=['ancient', 'monster'],
                room=dict(type='combat', encounter='TOADPOLES_WEAK'))
    with SolverEngine(config) as engine:
        frame = anchor.enter(engine, save, spot)
        assert frame['public'].get('selection_context'), 'the relic asks for a choice at the start of the fight'
        yield engine, frame


def play_out(engine, frame, budget_ms=500):
    """The reply of every solver step from `frame` to the end of the fight, in order, and
    whether the fight ended."""
    from spire_codex_data.anchor import in_combat, settle

    replies = []
    while in_combat(engine, frame) and len(replies) < 200:
        reply = engine.step(frame, budget_ms=budget_ms, potions=True, reuse_turn_plan=True)
        replies.append(reply)
        if reply['type'] != 'solver_step':
            break
        frame = settle(engine, reply['frame'])
    return replies, not in_combat(engine, frame)


def fight(config, relics, budget_ms=500):
    """`play_out` of one fight entered holding `relics`."""
    with entered(config, relics) as (engine, frame):
        return play_out(engine, frame, budget_ms)


def config_or_skip():
    config = os.environ.get('COMBAT_SOLVER_CONFIG')
    if not config:
        pytest.skip('Pinned native config required')
    return config


@pytest.mark.engine
@pytest.mark.parametrize('relics', [['GAMBLING_CHIP'], ['TOASTY_MITTENS'], ['TOOLBOX'], ['GAMBLING_CHIP', 'TOASTY_MITTENS']])
def test_choices_of_the_first_turn_setup_are_searched(relics):
    replies, ended = fight(config_or_skip(), relics)
    assert all(r['type'] == 'solver_step' for r in replies), [(r['type'], r.get('message')) for r in replies]
    assert ended
    first = replies[0]['search']
    assert first['turn_setup'] is True and first['expanded_nodes'] > 0
    # The play phase searches again from the state the setup left; no later search is a setup search.
    assert sum(1 for r in replies if (r.get('search') or {}).get('turn_setup')) == 1


@pytest.mark.engine
def test_a_setup_choice_comes_after_the_turn_start_effects():
    # The game pauses a turn's setup at its first selection and runs the side's turn-start
    # effects first: Orange Dough's cards are in hand when Toasty Mittens asks for a card.
    with entered(config_or_skip(), ['TOASTY_MITTENS', 'ORANGE_DOUGH']) as (_, frame):
        cards = [e for e in frame['public']['entities'] if e.get('entity_type') == 'card']
        hand = {e['content_id'] for e in cards if e.get('zone') == 'hand'}
        assert hand - {e['content_id'] for e in cards if e.get('zone') == 'deck'}, hand


@pytest.mark.engine
def test_a_first_turn_vakuu_plays_is_not_taken_over():
    # Whispering Earring: the game plays the first turn with choices of its own.
    replies, _ = fight(config_or_skip(), ['GAMBLING_CHIP', 'WHISPERING_EARRING'])
    assert replies[0]['type'] == 'solver_selection_required', replies[0]


@pytest.mark.engine
def test_a_failed_setup_search_leaves_the_choice_outside_the_solver():
    with entered(config_or_skip(), ['GAMBLING_CHIP']) as (engine, frame):
        # A beam width out of range fails the setup search before the solver runs.
        failed = engine.step(frame, budget_ms=500, potions=True, beam_width=0)
        assert failed['type'] == 'solver_selection_required', failed
        # The worker stays usable and does not search that setup again.
        again = engine.step(frame, budget_ms=500, potions=True)
        assert again['type'] == 'solver_selection_required', again


@pytest.mark.engine
# Silent starts with Neutralize, which a run begun by the command patches until PrepareEngine.
@pytest.mark.parametrize('character', ['Ironclad', 'Silent'])
def test_a_fight_resumed_as_the_first_command_of_a_worker_is_taken_over(character):
    from combat_solver_cli.client import SolverEngine
    from main.env.saves import continue_save, fight_resume_command
    from spire_codex_data.anchor import settle

    config = config_or_skip()
    with entered(config, ['GAMBLING_CHIP'], character) as (engine, frame):
        save = continue_save(engine)
    row = dict(save=save, encounter='ENCOUNTER.TOADPOLES_WEAK', floor=2, map_point_type='monster', second_boss=False)
    # Group sampling resumes a fight from its entry save as a worker's first command, and the
    # fight begins inside that command: the worker's patches must already be in place.
    with SolverEngine(config) as engine:
        frame = settle(engine, engine.send(fight_resume_command(row)))
        assert frame['public'].get('selection_context')
        replies, ended = play_out(engine, frame)
    assert all(r['type'] == 'solver_step' for r in replies), [(r['type'], r.get('message')) for r in replies]
    assert ended and replies[0]['search']['turn_setup'] is True
