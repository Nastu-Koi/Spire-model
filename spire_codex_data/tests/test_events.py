"""Event anchors: what a summary records of an event, held against what the game generates."""
import os
from copy import deepcopy

import pytest

from spire_codex_data import anchor, recorded


def number(value):
    return dict(type='BaseDynamic', decimal_value=value, bool_value=False, string_value=None)


def text(value):
    return dict(type='DynamicString', decimal_value=0, bool_value=False, string_value=value)


def test_event_variables_are_held_against_the_record():
    wrote = dict(HpLoss=number(7), RandomCard=text('Ball Lightning'))
    # The engine names content by its key; the record by its name in the player's language.
    assert recorded.same_variables(wrote, dict(HpLoss=number(7), RandomCard=text('BALL_LIGHTNING.title'))) == 'equal'
    assert recorded.same_variables(dict(wrote, RandomCard=text('球状闪电')),
                                   dict(HpLoss=number(7), RandomCard=text('BALL_LIGHTNING.title'))) == 'equal'
    # Another number, another variable, or another card in a language at hand: not this event state.
    assert recorded.same_variables(wrote, dict(HpLoss=number(3), RandomCard=text('BALL_LIGHTNING.title'))) is None
    assert recorded.same_variables(wrote, dict(HpLoss=number(7))) is None
    assert recorded.same_variables(wrote, dict(HpLoss=number(7), RandomCard=text('VOLTAIC.title'))) is None
    # A name in a language that is not installed cannot be held against the key.
    assert recorded.same_variables(dict(wrote, RandomCard=text('ボールライトニング')),
                                   dict(HpLoss=number(7), RandomCard=text('BALL_LIGHTNING.title'))) == 'text'
    assert recorded.same_variables({}, {}) == 'equal'


def test_the_record_names_the_cards_a_selection_takes():
    stats = dict(cards_removed=[dict(id='CARD.STRIKE_IRONCLAD', floor_added_to_deck=1)], upgraded_cards=['CARD.BASH'],
                 cards_enchanted=[dict(card=dict(id='CARD.ANGER', enchantment=dict(id='ENCHANTMENT.SHARP', amount=3)))],
                 cards_transformed=[dict(original_card=dict(id='CARD.DEFEND_IRONCLAD'), final_card=dict(id='CARD.CLASH'))])
    pending = recorded.changes(stats)
    # A card is named as it was before the change.
    assert pending['enchanted'] == [dict(id='CARD.ANGER')] and pending['transformed'] == [dict(id='CARD.DEFEND_IRONCLAD')]

    def frame(operation, selected=0, **limits):
        cards = [dict(entity_type='card', ref=f'card:{i}', content_id=c, upgrade_level=level)
                 for i, (c, level) in enumerate((('CARD.BASH', 1), ('CARD.BASH', 0), ('CARD.ANGER', 0)))]
        candidates = [dict(verb='SELECT_ONE', source_refs=[c['ref']]) for c in cards]
        candidates.append(dict(verb='FINISH_SELECTION', source_refs=[]))
        return dict(legal=dict(candidates=candidates), public=dict(phase='card_select', entities=cards,
                    selection_context=dict(operation=operation, selected_count=selected, **limits)))
    # The upgrade takes the copy that is not upgraded yet, then the selection ends.
    assert recorded.recorded_selection(frame('upgrade', max_total=1), pending)['source_refs'] == ['card:1']
    assert recorded.recorded_selection(frame('upgrade', selected=1, max_total=1), pending)['verb'] == 'FINISH_SELECTION'
    assert recorded.recorded_selection(frame('enchant', max_total=3), pending)['source_refs'] == ['card:2']
    # The record names no card for this selection, or names one the selection does not offer.
    with pytest.raises(recorded.Illegal):
        recorded.recorded_selection(frame('remove', min_total=1, max_total=1), pending)
    with pytest.raises(recorded.Illegal):
        recorded.recorded_selection(frame('upgrade', min_total=1, max_total=1), recorded.changes(stats | dict(upgraded_cards=[])))
    with pytest.raises(ValueError, match='choice_not_recorded:card_select'):
        recorded.recorded_selection(frame('duplicate', max_total=1), pending)


def planned(nodes):
    start = [dict(id='CARD.' + name) for name in ('STRIKE_IRONCLAD', 'DEFEND_IRONCLAD', 'BASH')]
    run = dict(seed='SEED', ascension=0, map_point_history=[nodes], players=[dict(deck=start, relics=[])])
    states, info = anchor.node_states(run, dict(players=[dict(deck=start, relics=[], current_hp=64, max_hp=80, gold=99)]))
    return {a['id'].split('-', 1)[1]: a for a, _ in anchor.plan(run, 'h', states, info, {'event'})}


def node(kind, rooms, **stats):
    return dict(map_point_type=kind, rooms=[dict(room, turns_taken=0) for room in rooms],
                player_stats=[dict(dict(current_hp=60, max_hp=80, current_gold=110), **stats)])


def test_events_are_planned_from_the_state_before_the_node():
    event = lambda name: dict(room_type='event', model_id='EVENT.' + name)
    step = dict(title=dict(key='SUNKEN_TREASURY.pages.INITIAL.options.SMALL.title', table='events'))
    found = planned([
        node('ancient', [event('NEOW')]),
        node('unknown', [event('SUNKEN_TREASURY')], event_choices=[step], current_gold=164),
        node('unknown', [event('FAKE_MERCHANT')]),
        node('unknown', [event('BATTLEWORN_DUMMY'), dict(room_type='monster', model_id='ENCOUNTER.X')], event_choices=[step])])
    # An ancient is not an ordinary event.
    assert set(found) == {'a01f02-event', 'a01f03-event', 'a01f04-event'}
    treasury = found['a01f02-event']
    assert treasury['room'] == dict(type='event', event='SUNKEN_TREASURY') and treasury['choices'] == [step]
    assert (treasury['ledger']['gold'], treasury['expected']['gold']) == (110, 164)
    assert found['a01f03-event']['error'] == 'event_without_recorded_choice'
    # A fight inside the event shares its node: there is no boundary between them.
    assert found['a01f04-event']['error'] == 'event_or_multiple_room_boundary'


def test_rewards_left_on_the_screen_are_planned_from_the_state_after_the_node():
    fight = lambda **stats: node('monster', [dict(room_type='monster', model_id='ENCOUNTER.X')], **stats)
    left = lambda name, picked=False: dict(choice=name, was_picked=picked)
    cards = [dict(card=dict(id='CARD.' + name), was_picked=False) for name in ('ANGER', 'HAVOC', 'CLASH')]
    run_nodes = [
        node('ancient', [dict(room_type='event', model_id='EVENT.NEOW')]),
        # A potion left behind and no card taken: both are on the screen that was left.
        fight(potion_choices=[left('POTION.FIRE_POTION')], card_choices=cards, gold_gained=14, current_gold=124),
        # Everything was taken: no decision to leave anything.
        fight(potion_choices=[left('POTION.FIRE_POTION', True)], gold_gained=10),
        fight(relic_choices=[left('RELIC.ANCHOR')], gold_gained=0),
        fight(potion_choices=[left('POTION.FIRE_POTION'), left('POTION.FIRE_POTION')], gold_gained=9)]
    start = [dict(id='CARD.' + name) for name in ('STRIKE_IRONCLAD', 'DEFEND_IRONCLAD', 'BASH')]
    run = dict(seed='SEED', ascension=0, map_point_history=[run_nodes], players=[dict(deck=start, relics=[])])
    states, info = anchor.node_states(run, dict(players=[dict(deck=start, relics=[], current_hp=64, max_hp=80, gold=99)]))
    found = {a['id'].split('-', 1)[1]: a for a, _ in anchor.plan(run, 'h', states, info, {'unclaimed'})}
    assert set(found) == {'a01f02-unclaimed', 'a01f04-unclaimed', 'a01f05-unclaimed'}
    first = found['a01f02-unclaimed']
    assert first['room'] == dict(type='rewards', potions=['POTION.FIRE_POTION'], relics=[],
                                 cards=[dict(id='CARD.' + name) for name in ('ANGER', 'HAVOC', 'CLASH')])
    # The screen is left after the gold was taken: the state is the node's end.
    assert first['ledger']['gold'] == 124
    # Gold that may still lie on the screen, and a reward named twice, are not settled by the record.
    assert found['a01f04-unclaimed']['error'] == 'gold_reward_unsettled'
    assert found['a01f05-unclaimed']['error'] == 'unclaimed_reward_repeated'


def test_native_reward_screen_holds_what_was_left():
    if os.environ.get('SPIRE_CODEX_DATA_NATIVE_TESTS') != '1':
        pytest.skip('Set SPIRE_CODEX_DATA_NATIVE_TESTS=1 with the local solver configured')
    from combat_solver_cli.client import DEFAULT_CONFIG, SolverEngine
    with SolverEngine() as engine:
        anchor.settle(engine, engine.send(dict(cmd='start_run', character='Ironclad', seed='unclaimed-check',
                                               ascension=0, decision_protocol=True)))
        base = engine.send(dict(cmd='anchor_state'))['save']
    player = base['players'][0]
    fight = node('monster', [dict(room_type='monster', model_id='ENCOUNTER.TOADPOLES_WEAK')])
    run = dict(seed='unclaimed-check', ascension=0, players=[dict(deck=player['deck'], relics=player['relics'])],
               map_point_history=[[node('ancient', [dict(room_type='event')]), fight]])
    spot = dict(floor=2, map_point_type='monster', second_boss=False, route=['ancient', 'monster'], kind='unclaimed')
    offer = [dict(id='CARD.ANGER'), dict(id='CARD.HAVOC'), dict(id='CARD.CLASH')]

    def saved(potions, relics=()):
        state = dict(deck=player['deck'], relics=player['relics'] + list(relics), potions=potions, hp=50,
                     max_hp=player['max_hp'], gold=120)
        return anchor.build_save(base, state, run, 1, 2, fight, 'unclaimed-check')

    with anchor.RunProcess(DEFAULT_CONFIG) as shared:
        # A full belt: the potion on the screen can only be taken after one is thrown away.
        full = ['POTION.BLOCK_POTION'] * player['max_potion_slot_count']
        room = dict(type='rewards', potions=['POTION.FIRE_POTION'], relics=['RELIC.ANCHOR'], cards=offer)
        result = anchor.play_unclaimed(shared, saved(full), dict(spot, room=room))
        (sample,) = result['records']
        verbs = [o['verb'] for o in sample['options']]
        assert sample['label']['verb'] == 'LEAVE_REWARDS' and result['outcome'] == 'label_legal'
        assert verbs.count('TAKE_CARD_REWARD') == 3 and 'DISCARD_POTION' in verbs
        rewards = [e['content_id'] for e in sample['observation']['entities'] if e['entity_type'] == 'reward']
        assert rewards == ['POTION.FIRE_POTION', 'RELIC.ANCHOR', 'CardReward']
        # The relic can be taken, the potion cannot: one take, for the relic.
        takes = [o for o in sample['options'] if o['verb'] == 'TAKE_REWARD']
        assert [t['source_refs'] for t in takes] == [['reward:1']]
        # With room on the belt the potion can be taken too, and leaving it is a choice against that.
        result = anchor.play_unclaimed(shared, saved([]), dict(spot, room=dict(room, relics=[], cards=[])))
        (sample,) = result['records']
        assert [o['verb'] for o in sample['options']] == ['TAKE_REWARD', 'LEAVE_REWARDS']
        # Sozu takes no potion: the screen can only be left, and that is no decision to learn.
        sozu = saved([], [dict(id='RELIC.SOZU', floor_added_to_deck=1)])
        result = anchor.play_unclaimed(shared, sozu, dict(spot, room=dict(room, relics=[], cards=[])))
        assert (result['records'], result['outcome']) == ([], 'single_action')


def test_native_event_options_show_the_numbers_and_names_the_player_reads():
    if os.environ.get('SPIRE_CODEX_DATA_NATIVE_TESTS') != '1':
        pytest.skip('Set SPIRE_CODEX_DATA_NATIVE_TESTS=1 with the local solver configured')
    from combat_solver_cli.client import DEFAULT_CONFIG, SolverEngine
    with SolverEngine() as engine:
        anchor.settle(engine, engine.send(dict(cmd='start_run', character='Ironclad', seed='event-values-check',
                                               ascension=0, decision_protocol=True)))
        base = engine.send(dict(cmd='anchor_state'))['save']
    player = base['players'][0]
    state = dict(deck=player['deck'], relics=player['relics'], potions=[], hp=60, max_hp=player['max_hp'], gold=99)
    run = dict(seed='event-values-check', ascension=0, players=[dict(deck=player['deck'], relics=player['relics'])],
               map_point_history=[[node('ancient', [dict(room_type='event')]),
                                   node('unknown', [dict(room_type='event', model_id='EVENT.BYRDONIS_NEST')])]])
    save = anchor.build_save(base, state, run, 1, 2, run['map_point_history'][0][1], 'event-values')
    spot = dict(floor=2, map_point_type='unknown', second_boss=False, route=['ancient', 'unknown'],
                room=dict(type='event', event='BYRDONIS_NEST'))
    with anchor.RunProcess(DEFAULT_CONFIG) as shared:
        for wanted in ('EAT', 'TAKE'):
            engine, frame = shared.enter(save, dict(spot))
            entities = frame['public']['entities']
            options = {e['content_id'].rsplit('.', 1)[-1]: e['ref'] for e in entities if e['entity_type'] == 'event_option'}
            shown = {(e['owner_ref'], e['content_id']): e for e in entities if e['entity_type'] == 'displayed_variable'}
            # Eating shows how much max HP it gives; taking shows which card it adds.
            gain = shown[options['EAT'], 'MaxHp']['amount']
            assert shown[options['TAKE'], 'Card']['names'] == 'CARD.BYRDONIS_EGG' and gain > 0
            assert 'amount' not in shown[options['TAKE'], 'Card']
            choice = next(c for c in frame['legal']['candidates'] if c['source_refs'] == [options[wanted]])
            anchor.advance(engine, frame, choice)
            after = anchor.installed_state(engine)
            # What the option showed is what it does.
            if wanted == 'EAT':
                assert after['max_hp'] == player['max_hp'] + gain
            else:
                assert sum(c['id'] == 'CARD.BYRDONIS_EGG' for c in after['deck']) == 1
        assert frame['contract']['observation_schema'] == 'public-state-v6'


def test_native_event_is_generated_again_and_takes_the_recorded_choice():
    if os.environ.get('SPIRE_CODEX_DATA_NATIVE_TESTS') != '1':
        pytest.skip('Set SPIRE_CODEX_DATA_NATIVE_TESTS=1 with the local solver configured')
    from combat_solver_cli.client import DEFAULT_CONFIG, SolverEngine
    with SolverEngine() as engine:
        anchor.settle(engine, engine.send(dict(cmd='start_run', character='Ironclad', seed='event-anchor-check',
                                               ascension=0, decision_protocol=True)))
        base = engine.send(dict(cmd='anchor_state'))['save']
    player = base['players'][0]
    state = dict(deck=player['deck'], relics=player['relics'], potions=[], hp=60, max_hp=player['max_hp'], gold=99)
    run = dict(seed='event-anchor-check', ascension=0, players=[dict(deck=player['deck'], relics=player['relics'])],
               map_point_history=[[node('ancient', [dict(room_type='event')]),
                                   node('unknown', [dict(room_type='event', model_id='EVENT.TABLET_OF_TRUTH')])]])
    save = anchor.build_save(base, state, run, 1, 2, run['map_point_history'][0][1], 'event-check')
    spot = dict(floor=2, map_point_type='unknown', second_boss=False, route=['ancient', 'unknown'],
                kind='event', id='event-check', room=dict(type='event', event='TABLET_OF_TRUTH'))
    with anchor.RunProcess(DEFAULT_CONFIG) as shared:
        # Play the event once by hand: decipher twice, then give up.
        engine, frame = shared.enter(save, dict(spot))
        chosen = []
        for wanted in ('DECIPHER', 'DECIPHER', 'GIVE_UP'):
            page = engine.send(dict(cmd='anchor_event'))
            index, option = next((i, o) for i, o in enumerate(page['options'])
                                 if not o['locked'] and wanted in o['text_key'].rsplit('.', 1)[-1])
            chosen.append(option['history']['title'])
            candidate = next(c for c in frame['legal']['candidates'] if c['source_refs'] == [f'option:{index}'])
            frame = anchor.advance(engine, frame, candidate)
        while frame['public']['phase'] == 'event':
            frame = anchor.advance(engine, frame, frame['legal']['candidates'][0])
        ended = anchor.installed_state(engine)
        variables = engine.send(dict(cmd='anchor_event'))['variables']
        assert ended['max_hp'] < player['max_hp']
        # The game writes every choice with the variables as the event left them.
        choices = [dict(title=title, variables=deepcopy(variables)) for title in chosen]
        end = dict(hp=ended['current_hp'], max_hp=ended['max_hp'], gold=ended['gold'], deck=ended['deck'],
                   relics=ended['relics'], potions=[p['id'] for p in ended.get('potions', [])])
        record = dict(spot, choices=choices, changes=recorded.changes({}), expected=recorded.expected_end(end))
        result = anchor.play_event(shared, save, dict(record))
        assert result['outcome'] == 'matched_record' and result['variables'] == {'equal': 3}
        labels = [r['label'] for r in result['records']]
        assert sum(label['verb'] == 'CHOOSE_EVENT_OPTION' for label in labels) >= 3
        assert all(r['label'] in r['options'] and r['coverage']['label'] == 'historical_choice' for r in result['records'])
        # The first page shows the loss before any of it was taken: the frame is the one the choice was made in.
        first = next(e for e in result['records'][0]['observation']['entities'] if e['entity_type'] == 'player')
        assert (first['hp'], first['max_hp']) == (60, player['max_hp'])

        # A record of another event state, of another outcome, or of a choice the page does not offer is refused.
        other = deepcopy(choices)
        other[0]['variables']['DecipherMaxHpLoss']['decimal_value'] += 1
        with pytest.raises(ValueError, match='event_variables_differ'):
            anchor.play_event(shared, save, dict(record, choices=other))
        with pytest.raises(ValueError, match='outcome_differs_from_record:max_hp'):
            anchor.play_event(shared, save, dict(record, expected=recorded.expected_end(dict(end, max_hp=1))))
        with pytest.raises(ValueError, match='event_option_not_offered'):
            anchor.play_event(shared, save, dict(record, choices=choices[2:] + choices[:2]))
        with pytest.raises(ValueError, match='event_choice_not_recorded'):
            anchor.play_event(shared, save, dict(record, choices=choices[:1]))
