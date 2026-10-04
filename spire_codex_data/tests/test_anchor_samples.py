import gzip
import json
import os

import pytest

from spire_codex_data import anchor
from spire_codex_data.map_ambiguity import act_report, analyze

STRIKE, DEFEND, BASH = (dict(id='CARD.' + x) for x in ('STRIKE_IRONCLAD', 'DEFEND_IRONCLAD', 'BASH'))


def stats(**extra):
    return dict(dict(current_hp=60, max_hp=80, current_gold=110), **extra)


def node(kind, room_type=None, **extra):
    room = dict(room_type=room_type or kind, turns_taken=3)
    if room['room_type'] in anchor.COMBAT:
        room.update(model_id='ENCOUNTER.TOADPOLES_WEAK', monster_ids=[])
    return dict(map_point_type=kind, rooms=[room], player_stats=[stats(**extra)])


def choice(card, picked=False):
    return dict(card=dict(card, **({'floor_added_to_deck': 2} if picked else {})), was_picked=picked)


def summary(nodes):
    return dict(seed='SEED', ascension=0, map_point_history=[nodes],
                players=[dict(deck=[STRIKE, DEFEND, dict(BASH, current_upgrade_level=1),
                                    dict(id='CARD.ANGER', floor_added_to_deck=2)], relics=[])])


def template():
    return dict(players=[dict(deck=[STRIKE, DEFEND, BASH], relics=[], current_hp=64, max_hp=80, gold=99)])


def anchors(run, kinds=anchor.KINDS):
    states, info = anchor.node_states(run, template())
    return {a['id'].split('-', 1)[1]: a for a, _ in anchor.plan(run, 'h', states, info, set(kinds))}


def test_every_kind_starts_from_the_right_side_of_its_node():
    run = summary([
        node('ancient', 'event'),
        node('monster', card_choices=[choice(dict(id='CARD.ANGER'), True), choice(dict(id='CARD.HAVOC')),
                                      choice(dict(id='CARD.CLASH'))], cards_gained=[dict(id='CARD.ANGER')]),
        node('rest_site', rest_site_choices=['SMITH'], upgraded_cards=['CARD.BASH']),
        node('boss')])
    found = anchors(run)
    assert set(found) == {'a01f01-map', 'a01f02r0', 'a01f02-reward', 'a01f02-map', 'a01f03-rest', 'a01f03-map',
                          'a01f04r0'}
    deck = lambda a: sorted(c['id'] for c in a['ledger']['deck'])
    # A battle starts before its node; its reward after the fight but before the pick.
    assert 'CARD.ANGER' not in deck(found['a01f02r0']) and 'CARD.ANGER' not in deck(found['a01f02-reward'])
    assert sum('ANGER' in s for s in found['a01f02-reward']['expected']['deck']) == 1
    # The travel decision after a node sees that node's result.
    assert 'CARD.ANGER' in deck(found['a01f02-map']) and found['a01f02-map']['ledger']['hp'] == 60
    assert found['a01f03-rest']['expected']['deck'] != sorted(map(anchor.signature, found['a01f03-rest']['ledger']['deck']))
    assert found['a01f02r0']['route'] == ['ancient', 'monster', 'rest_site', 'boss']
    assert 'a01f04-map' not in found  # nothing to travel to after the act boss


@pytest.mark.parametrize('extra,reason', [
    (dict(card_choices=[choice(dict(id='CARD.' + str(i))) for i in range(6)]), 'multiple_card_rewards'),
    (dict(card_choices=[choice(dict(id='CARD.ANGER'), True), choice(dict(id='CARD.HAVOC'), True),
                        choice(dict(id='CARD.CLASH'))]), 'multiple_card_rewards'),
])
def test_offers_the_summary_cannot_separate_are_isolated(extra, reason):
    found = anchors(summary([node('monster', **extra)]), ('reward',))
    assert found['a01f01-reward']['error'] == reason and 'ledger' not in found['a01f01-reward']


def test_ledger_failure_keeps_earlier_anchors_and_drops_later_ones():
    run = summary([node('monster'), node('monster', cards_removed=[dict(id='CARD.MISSING')]), node('monster')])
    found = anchors(run, ('battle',))
    assert 'ledger' in found['a01f01r0'] and 'ledger' in found['a01f02r0']
    assert found['a01f03r0']['error'].startswith('ledger:floor_2')


def frame(phase, candidates, entities):
    return dict(public=dict(phase=phase, entities=entities), legal=dict(candidates=candidates))


def candidate(verb, ref=None):
    return dict(verb=verb, source_refs=[ref] if ref else [], target_refs=[])


def card(ref, content_id, level=0, enchantment=None):
    return dict(ref=ref, entity_type='card', content_id=content_id, upgrade_level=level, enchantment=enchantment)


def test_smith_target_must_be_one_kind_of_copy():
    plan = dict(choices=['SMITH'], upgraded=['CARD.BASH'])
    rest = frame('rest_site', [candidate('CHOOSE_REST_OPTION', 'rest:0'), candidate('CHOOSE_REST_OPTION', 'rest:1')],
                 [dict(ref='rest:0', content_id='HEAL'), dict(ref='rest:1', content_id='SMITH')])
    choose = anchor.rest_choice(plan)
    assert choose(rest)['source_refs'] == ['rest:1']
    same = frame('card_select', [candidate('SELECT_ONE', 'card:0'), candidate('SELECT_ONE', 'card:1'),
                                 candidate('SELECT_ONE', 'card:2')],
                 [card('card:0', 'CARD.BASH'), card('card:1', 'CARD.BASH'), card('card:2', 'CARD.ANGER')])
    assert choose(same)['source_refs'] == ['card:0']
    assert choose(frame('card_select', [candidate('FINISH_SELECTION')], []))['verb'] == 'FINISH_SELECTION'
    assert choose(frame('rest_site', [candidate('LEAVE_ROOM')], [])) is None
    choose = anchor.rest_choice(plan)
    choose(rest)
    differing = frame('card_select', [candidate('SELECT_ONE', 'card:0'), candidate('SELECT_ONE', 'card:1')],
                      [card('card:0', 'CARD.BASH'), card('card:1', 'CARD.BASH', enchantment='ENCHANTMENT.SHARP')])
    with pytest.raises(ValueError, match='ambiguous_upgrade_target'):
        choose(differing)
    # The ledger's settled copy decides between them.
    choose = anchor.rest_choice(dict(plan, upgraded_enchantments=['ENCHANTMENT.SHARP']))
    choose(rest)
    assert choose(differing)['source_refs'] == ['card:1']


def reward_frame(cards, alternative=False):
    candidates = [candidate('TAKE_CARD_REWARD', c['ref']) for c in cards] + [candidate('LEAVE_REWARDS')]
    if alternative:
        candidates.append(candidate('CHOOSE_REWARD_ALTERNATIVE', 'alt'))
    return frame('rewards', candidates, cards)


def test_reward_label_follows_the_record_or_is_refused():
    offer = [dict(id='CARD.ANGER', current_upgrade_level=1), dict(id='CARD.HAVOC'),
             dict(id='CARD.CLASH', enchantment=dict(id='ENCHANTMENT.GLAM', amount=1))]
    shown = [card('card:1', 'CARD.ANGER', 1), card('card:2', 'CARD.HAVOC'),
             card('card:3', 'CARD.CLASH', enchantment='ENCHANTMENT.GLAM')]
    picked = dict(room=dict(cards=offer), picked=[offer[2]])
    assert anchor.reward_choice(picked)(reward_frame(shown))['source_refs'] == ['card:3']
    skipped = dict(room=dict(cards=offer), picked=[])
    assert anchor.reward_choice(skipped)(reward_frame(shown))['verb'] == 'LEAVE_REWARDS'
    with pytest.raises(ValueError, match='skip_or_alternative_unknown'):
        anchor.reward_choice(skipped)(reward_frame(shown, alternative=True))
    # A relic that changed the injected cards means the frame is not what the player saw.
    with pytest.raises(ValueError, match='native_offer_differs_from_record'):
        anchor.reward_choice(picked)(reward_frame([card('card:1', 'CARD.ANGER', 1), card('card:2', 'CARD.HAVOC', 1),
                                                   shown[2]]))


def test_export_keeps_winning_actions_and_every_fight_outcome(tmp_path):
    hero = dict(entity_type='player', ref='player', hp=50, max_hp=80, round=1)
    potion = dict(entity_type='potion', ref='potion:0', content_id='POTION.FIRE')
    entities = [hero, potion, dict(entity_type='card', zone='deck', content_id='CARD.BASH'),
                dict(entity_type='card', zone='hand', content_id='CARD.BASH'),
                dict(entity_type='enemy', ref='creature:1', hp=30)]

    def item(group, kind, actor, option=None, **seen):
        option = option or dict(verb='END_TURN')
        public = [dict(e, **seen) if e is hero else e for e in entities]
        return dict(schema=anchor.SCHEMA, observation=dict(entities=public), options=[option], label=option,
                    actor=actor, coverage={}, metadata=dict(sample_group=group, anchor_kind=kind, run_hash='h',
                    split_group='seed-group', character='Ironclad', ascension=7, state_sources=['derived'],
                    source_integrity='unverified_source'))
    (tmp_path / 'runs').mkdir()
    (tmp_path / 'samples').mkdir()
    fight = dict(kind='battle', status='verified', act=2, floor=5, encounter='ENCOUNTER.TOADPOLES_WEAK', entry_hp=50)
    battles = dict(run_hash='h', kind='battle', identity=dict(anchor.BUDGETS), anchors=[
        dict(fight, id='won', room_type='monster', outcome='win', exit_hp=56),
        dict(fight, id='lost', room_type='boss', outcome='loss', exit_hp=None),
        dict(fight, id='bad', room_type='elite', status='execution_failed', error='x')])
    rests = dict(run_hash='h', kind='rest', identity={}, anchors=[
        dict(id='rest', kind='rest', status='verified', outcome='matched_record')])
    use = dict(verb='USE_POTION', source_refs=['potion:0'])
    for name, report, rows in (
            ('h.battle', battles, [item('won', 'battle', 'combat_solver', use), item('won', 'battle', 'combat_solver', round=2),
                                   item('lost', 'battle', 'combat_solver'), item('bad', 'battle', 'combat_solver')]),
            ('h.rest', rests, [item('rest', 'rest', 'historical_noncombat')])):
        (tmp_path / 'runs' / f'{name}.json').write_text(json.dumps(report))
        with gzip.open(tmp_path / 'samples' / f'{name}.jsonl.gz', 'wt') as stream:
            stream.write(''.join(json.dumps(row) + '\n' for row in rows))
    (tmp_path / 'h.run.json').write_text(json.dumps(dict(map_point_history=[])))
    (tmp_path / 'manifest.json').write_text(json.dumps(dict(selected=[dict(run_hash='h', path=str(tmp_path / 'h.run.json'))])))
    summary = anchor.export(tmp_path, tmp_path / 'manifest.json')
    assert summary['counts'] == {'battle/combat_solver': 2, 'rest/historical_noncombat': 1}
    with gzip.open(tmp_path / 'independent-training.jsonl.gz', 'rt') as stream:
        assert [json.loads(line)['metadata']['sample_group'] for line in stream] == ['won', 'won', 'rest']
    # A lost fight teaches no actions, but it is an outcome.
    assert summary['combat_outcomes']['counts'] == {'boss/loss': 1, 'regular/win': 1}
    with gzip.open(tmp_path / 'combat-outcomes.jsonl.gz', 'rt') as stream:
        won, lost = map(json.loads, stream)
    assert (won['start_hp'], won['end_hp'], won['hp_lost'], won['result'], won['turns']) == (50, 56, -6, 'win', 2)
    assert (lost['start_hp'], lost['end_hp'], lost['hp_lost'], lost['result'], lost['kind']) == (50, 0, 50, 'loss', 'boss')
    assert (won['seed'], won['encounter'], won['act'], won['max_hp']) == ('seed-group', 'TOADPOLES_WEAK', 2, 80)
    assert [e['content_id'] for e in won['potions_used']] == ['POTION.FIRE']
    # The input is the state the fight was entered with: no hand, no enemies.
    assert [(e['entity_type'], e.get('zone')) for e in won['entities']] == [('player', None), ('potion', None), ('card', 'deck')]


def grid(edges, kinds):
    return {c: dict(coord=c, kind=k, children=edges.get(c, [])) for c, k in kinds.items()}


def test_export_names_the_bosses_of_the_current_act_only():
    boss = lambda *ids: dict(map_point_type='boss', rooms=[dict(room_type='boss', model_id=i) for i in ids])
    fight = dict(map_point_type='monster', rooms=[dict(room_type='monster', model_id='ENCOUNTER.X')])
    bosses = anchor.act_bosses(dict(map_point_history=[[fight, boss('ENCOUNTER.A_BOSS')], [fight, boss('ENCOUNTER.B_BOSS')],
                                                       [boss('ENCOUNTER.C_BOSS'), boss('ENCOUNTER.D_BOSS')]]))
    node = lambda floor, kind='Boss': dict(entity_type='map_node', content_id=kind, floor=floor, col=3)
    row = lambda act, *nodes: dict(observation=dict(entities=[dict(entity_type='player', act=act), *nodes]))
    first = row(1, node(15, 'RestSite'), node(16))
    assert anchor.name_bosses(first, bosses)
    assert [e.get('encounter') for e in first['observation']['entities'][1:]] == [None, 'ENCOUNTER.A_BOSS']
    # Two bosses close the last act of the highest ascension, in the order they are fought.
    last = row(3, node(17), node(16))
    assert anchor.name_bosses(last, bosses)
    assert [e['encounter'] for e in last['observation']['entities'][1:]] == ['ENCOUNTER.D_BOSS', 'ENCOUNTER.C_BOSS']
    # Nodes the record cannot be paired with are refused, not guessed.
    assert not anchor.name_bosses(row(2, node(15), node(16)), bosses)
    assert anchor.name_bosses(row(2), bosses)


def test_type_sequence_determines_a_route_only_when_one_path_spells_it():
    kinds = {'0,0': 'ancient', '0,1': 'monster', '1,1': 'monster', '0,2': 'shop', '1,2': 'elite', '0,3': 'boss'}
    edges = {'0,0': ['0,1', '1,1'], '0,1': ['0,2'], '1,1': ['1,2'], '0,2': ['0,3'], '1,2': ['0,3']}
    unique = act_report(grid(edges, kinds), ['0,0', '0,1', '0,2', '0,3'])
    assert unique == dict(status='ok', unique_route=True, choices=1, determined=1, position_known=1, widest=1)
    twin = act_report(grid(edges, dict(kinds, **{'1,2': 'shop'})), ['0,0', '0,1', '0,2', '0,3'])
    assert twin['unique_route'] is False and twin['determined'] == 0 and twin['widest'] == 2
    assert act_report(grid(edges, kinds), ['0,0', '1,2'])['status'] == 'step_not_on_an_edge'
    journal = [dict(t='map', act=1, nodes=list(grid(edges, kinds).values()))] + [
        dict(t='room', act=1, coord=c) for c in ('0,0', '0,0', '1,1', '1,2', '0,3')]
    assert analyze(journal)[1]['unique_route'] is True


@pytest.mark.engine
def test_native_anchor_entry_is_a_named_training_contract():
    if os.environ.get('SPIRE_CODEX_DATA_NATIVE_TESTS') != '1':
        pytest.skip('Set SPIRE_CODEX_DATA_NATIVE_TESTS=1 with the local solver configured')
    from combat_solver_cli.client import SolverEngine
    from spire_codex_data.anchor import settle
    with SolverEngine() as engine:
        settle(engine, engine.send(dict(cmd='start_run', character='Ironclad', seed='anchor-entry-check',
                                        ascension=3, decision_protocol=True)))
        base = engine.send(dict(cmd='anchor_state'))['save']
    player = base['players'][0]
    run = dict(seed='anchor-entry-check', ascension=3, players=[dict(deck=player['deck'], relics=player['relics'])],
               map_point_history=[[node('ancient', 'event'), node('monster')]])
    state = dict(deck=player['deck'], relics=player['relics'], potions=[], hp=41, max_hp=player['max_hp'], gold=77)
    save = anchor.build_save(base, state, run, 1, 2, run['map_point_history'][0][1], 'anchor-entry-check')
    offer = [dict(id='CARD.ANGER', current_upgrade_level=1), dict(id='CARD.HAVOC'), dict(id='CARD.CLASH')]
    spot = dict(floor=2, map_point_type='monster', second_boss=False, route=['ancient', 'monster'],
                room=dict(type='card_reward', cards=offer))
    with SolverEngine() as engine:
        frame = anchor.enter(engine, save, spot)
        assert frame['contract']['training_ready'] is True
        assert frame['contract']['initialization'] == 'summary-anchor-v1' and frame['contract']['fixed_ascension'] == 3
        hero = next(e for e in frame['public']['entities'] if e.get('entity_type') == 'player')
        assert (hero['hp'], hero['gold'], hero['floor']) == (41, 77, 2)
        takes = [c for c in frame['legal']['candidates'] if c['verb'] == 'TAKE_CARD_REWARD']
        assert sorted(anchor.card_key(anchor.entity(frame, c)) for c in takes) == sorted(map(anchor.summary_key, offer))
        # A second room, or any debug mutation, ends the anchor's training contract.
        assert engine.send(dict(cmd='anchor_room', room=dict(type='rest_site')))['type'] == 'error'
        again = engine.send(dict(cmd='enter_room', type='rest_site', decision_protocol=True))
        assert again['contract']['training_ready'] is False


def test_native_enchantment_amount_and_ascension_reach_the_model_input():
    if os.environ.get('SPIRE_CODEX_DATA_NATIVE_TESTS') != '1':
        pytest.skip('Set SPIRE_CODEX_DATA_NATIVE_TESTS=1 with the local solver configured')
    from combat_solver_cli.client import SolverEngine
    from model.representation import observation
    from spire_codex_data.anchor import settle
    digests = {}
    for ascension, amount in ((3, 3), (3, 7), (8, 3)):
        with SolverEngine() as engine:
            settle(engine, engine.send(dict(cmd='start_run', character='Ironclad', seed='enchantment-check',
                                            ascension=ascension, decision_protocol=True)))
            base = engine.send(dict(cmd='anchor_state'))['save']
        player = base['players'][0]
        deck = [dict(c) for c in player['deck']]
        strike = next(c for c in deck if c['id'] == 'CARD.STRIKE_IRONCLAD')
        strike['enchantment'] = dict(id='ENCHANTMENT.ADROIT', amount=amount)
        run = dict(seed='enchantment-check', ascension=ascension, players=[dict(deck=deck, relics=player['relics'])],
                   map_point_history=[[node('ancient', 'event'), node('rest_site')]])
        state = dict(deck=deck, relics=player['relics'], potions=[], hp=41, max_hp=player['max_hp'], gold=77)
        save = anchor.build_save(base, state, run, 1, 2, run['map_point_history'][0][1], 'enchantment-check')
        spot = dict(floor=2, map_point_type='rest_site', second_boss=False, route=['ancient', 'rest_site'],
                    room=dict(type='rest_site'))
        with SolverEngine() as engine:
            frame = anchor.enter(engine, save, spot)
        cards = [e for e in frame['public']['entities'] if e.get('enchantment') == 'ENCHANTMENT.ADROIT']
        assert [c['enchantment_amount'] for c in cards] == [amount]
        hero = next(e for e in frame['public']['entities'] if e.get('entity_type') == 'player')
        assert hero['ascension'] == ascension == frame['contract']['fixed_ascension']
        assert frame['contract']['observation_schema'] == 'public-state-v3'
        digests[ascension, amount] = observation(frame).digest
    # What the game shows differently, the model is given differently.
    assert len(set(digests.values())) == 3


@pytest.mark.engine
def test_native_ancient_shows_the_recorded_offer_in_recorded_order():
    if os.environ.get('SPIRE_CODEX_DATA_NATIVE_TESTS') != '1':
        pytest.skip('Set SPIRE_CODEX_DATA_NATIVE_TESTS=1 with the local solver configured')
    from combat_solver_cli.client import SolverEngine
    from spire_codex_data.anchor import settle
    with SolverEngine() as engine:
        settle(engine, engine.send(dict(cmd='start_run', character='Ironclad', seed='anchor-ancient-check',
                                        ascension=0, decision_protocol=True)))
        base = engine.send(dict(cmd='anchor_state'))['save']
    player = base['players'][0]
    start = node('ancient', 'event')
    start['rooms'][0]['model_id'] = 'EVENT.NEOW'
    run = dict(seed='anchor-ancient-check', ascension=0, map_point_history=[[start, node('monster')]],
               players=[dict(deck=player['deck'], relics=player['relics'])])
    state = dict(deck=player['deck'], relics=player['relics'], potions=[], hp=player['current_hp'],
                 max_hp=player['max_hp'], gold=player['gold'])
    save = anchor.build_save(base, state, run, 1, 1, start, 'anchor-ancient-check')
    wanted = ['LEAFY_POULTICE', 'POMANDER', 'LEAD_PAPERWEIGHT']
    spot = dict(floor=1, map_point_type='ancient', second_boss=False, route=['ancient'],
                room=dict(type='ancient', event='NEOW', options=wanted))
    with SolverEngine() as engine:
        frame = anchor.enter(engine, save, spot)
        shown = [anchor.entity(frame, c)['content_id'] for c in frame['legal']['candidates']
                 if c['verb'] == 'CHOOSE_EVENT_OPTION']
        assert [s.rsplit('.', 1)[-1] for s in shown] == wanted and frame['public']['phase'] == 'event'
        assert frame['contract']['training_ready'] is True
    spot['room'] = dict(type='ancient', event='NEOW', options=['NOT_A_NEOW_OPTION'])
    with SolverEngine() as engine:
        with pytest.raises(ValueError):
            anchor.enter(engine, save, spot)


def test_replay_audit_reads_old_and_new_replays_alike():
    from spire_codex_data.anchor_audit import compare, timeline
    offer = [dict(option_index=i, option_id=name, up=0) for i, name in enumerate(['ANGER', 'HAVOC', 'CLASH'])]
    journal = [
        dict(t='deck', s=1, cards=[dict(id='BASH')]),
        dict(t='room', s=2, floor=2, act=1, coord='0,1'),
        dict(t='hp', s=3, floor=2, dst='player', d=6, hp=50),
        dict(t='decision', s=4, floor=2, decision_id=7, decision_type='card_reward', source='reward', options=offer),
        dict(t='gold', s=5, floor=2, gold=120),
        # Old replays can name the wrong card here; the deck snapshot decides.
        dict(t='acquire', s=6, floor=2, decision_id=7, source='reward', id='HAVOC', option_index=1),
        dict(t='deck', s=7, cards=[dict(id='BASH'), dict(id='ANGER')]),
        dict(t='buy', s=8, floor=2, gold_on_hand=70),
        dict(t='room', s=9, floor=3, act=1, coord='0,2'),
    ]
    facts = timeline(journal)
    cards = [dict(ref=f'card:{i}', entity_type='card', content_id='CARD.' + o['option_id'], upgrade_level=0)
             for i, o in enumerate(offer)]
    options = [candidate('TAKE_CARD_REWARD', c['ref']) for c in cards] + [candidate('LEAVE_REWARDS')]
    row = dict(options=options, label=options[0], observation=dict(entities=cards + [
        dict(entity_type='player', hp=50, max_hp=80, gold=70),
        dict(entity_type='card', zone='deck', content_id='CARD.BASH', upgrade_level=0)]))
    checks = compare([row], dict(kind='reward', global_floor=2, act=1, floor=2), facts)
    assert checks == dict(offer_cards=True, offer_upgrades=True, label_vs_acquire_record=False,
                          label_vs_deck_snapshots=True, deck=True, hp=True, gold=True, relics=True, potions=True)


def enchanted_run(nodes, final):
    spiral = dict(STRIKE, enchantment=dict(id='ENCHANTMENT.SPIRAL', amount=1))
    base = dict(players=[dict(deck=[spiral, STRIKE, DEFEND], relics=[], current_hp=64, max_hp=80, gold=99)])
    return dict(seed='SEED', ascension=0, map_point_history=[nodes], players=[dict(deck=final(spiral), relics=[])]), base


def test_final_deck_settles_which_copy_an_upgrade_meant():
    smith = dict(rest_site_choices=['SMITH'], upgraded_cards=['CARD.STRIKE_IRONCLAD'])
    up = lambda card: dict(card, current_upgrade_level=1)
    # The summary only says "a Strike was upgraded"; the final deck shows it was the plain one.
    run, base = enchanted_run([node('monster'), node('rest_site', **smith), node('monster')],
                              lambda spiral: [spiral, up(STRIKE), DEFEND])
    states, info = anchor.node_states(run, base)
    assert info == dict(error=None, suspect_card_ids=[], resolved_by_final_deck=True)
    upgraded = [c for c in states[1, 3]['start']['deck'] if c.get('current_upgrade_level')]
    assert len(upgraded) == 1 and 'enchantment' not in upgraded[0]
    # Both copies upgraded at different sites: the end is certain, the order between is not.
    run, base = enchanted_run([node('rest_site', **smith), node('monster'), node('rest_site', **smith), node('boss')],
                              lambda spiral: [up(spiral), up(STRIKE), DEFEND])
    states, info = anchor.node_states(run, base)
    assert states[1, 1]['start'] is not None and states[1, 2]['start'] is None and states[1, 4]['start'] is not None
    assert 'ambiguous_card_instance' in states[1, 2]['error']
    # No choice reproduces the final deck: stop at the open card as before.
    run, base = enchanted_run([node('rest_site', **smith), node('monster')], lambda spiral: [spiral, STRIKE, DEFEND])
    states, info = anchor.node_states(run, base)
    assert info['error'] == 'floor_1:ambiguous_card_instance:CARD.STRIKE_IRONCLAD' and states[1, 2]['start'] is None

