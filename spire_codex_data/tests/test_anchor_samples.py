import gzip
import json
import os

import pytest

from combat_outcome.data import fights
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
    assert set(found) == {'a01f01-map', 'a01f02r0', 'a01f02-reward', 'a01f02-claim', 'a01f02-map', 'a01f03-rest',
                          'a01f03-map', 'a01f04r0', 'a01f04-claim'}
    # A fight whose record shows no gold gained leaves the claims of its reward screen open.
    assert found['a01f02-claim']['error'] == 'gold_reward_unsettled'
    deck = lambda a: sorted(c['id'] for c in a['ledger']['deck'])
    # A battle starts before its node; its reward after the fight but before the pick.
    assert 'CARD.ANGER' not in deck(found['a01f02r0']) and 'CARD.ANGER' not in deck(found['a01f02-reward'])
    assert sum('ANGER' in s for s in found['a01f02-reward']['expected']['deck']) == 1
    # The travel decision after a node sees that node's result.
    assert 'CARD.ANGER' in deck(found['a01f02-map']) and found['a01f02-map']['ledger']['hp'] == 60
    assert found['a01f03-rest']['expected']['deck'] != sorted(map(anchor.signature, found['a01f03-rest']['ledger']['deck']))
    assert found['a01f02r0']['route'] == ['ancient', 'monster', 'rest_site', 'boss']
    assert 'a01f04-map' not in found  # nothing to travel to after the act boss


def test_one_fight_in_two_is_fought_again_with_potions_it_must_use(monkeypatch):
    belt = ['POTION.FIRE_POTION', 'POTION.BLOCK_POTION', 'POTION.FAIRY_IN_A_BOTTLE']
    drawn = [anchor.forced_potions(f'fight-{i}', dict(potions=belt)) for i in range(2000)]
    assert drawn == [anchor.forced_potions(f'fight-{i}', dict(potions=belt)) for i in range(2000)]
    assert 900 < sum(map(bool, drawn)) < 1100
    # At least one, any number of those held, and never the potion the game uses by itself.
    assert {len(d) for d in drawn} == {0, 1, 2} and all(set(d) <= set(belt[:2]) for d in drawn)
    assert 400 < sum(len(d) == 2 for d in drawn) < 600
    assert not any(anchor.forced_potions(f'fight-{i}', dict(potions=belt[2:])) for i in range(50))
    assert anchor.forced_potions('fight', None) == []
    # The potion anchor starts where the fight does and repeats it.
    monkeypatch.setattr(anchor, 'forced_potions', lambda fight, state: list(state['potions']))
    nodes = [node('ancient', 'event'),
             node('monster', potion_choices=[dict(choice='POTION.FIRE_POTION', was_picked=True)]), node('elite')]
    run = dict(seed='SEED', ascension=0, map_point_history=[nodes], players=[dict(deck=[STRIKE, DEFEND, BASH], relics=[])])
    found = anchors(run, ('battle', 'potion'))
    assert set(found) == {'a01f02r0', 'a01f03r0', 'a01f03r0-potion'}
    fight, again = found['a01f03r0'], found['a01f03r0-potion']
    assert again['forced'] == ['POTION.FIRE_POTION'] and again['ledger'] == fight['ledger']
    assert (again['room'], again['room_type'], again['state']) == (fight['room'], 'elite', 'start')


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


@pytest.mark.parametrize("controlled,verbs", [
    ("treasure", ["OPEN_CHEST", "LEAVE_ROOM"]),
    ("event", ["CHOOSE_EVENT_OPTION", "DISCARD_POTION"]),
])
def test_export_omits_controller_only_groups_but_retains_followup_evidence(tmp_path, controlled, verbs):
    report = dict(kind='noncombat', anchors=[dict(id=group, status='verified', outcome='matched_record')
                                          for group in ('ordinary', 'selection')])
    report_path, source, sample_path = (tmp_path / name for name in ('report.json', 'source.json', 'rows.jsonl.gz'))
    report_path.write_text(json.dumps(report))
    source.write_text(json.dumps(dict(map_point_history=[])))
    rows = []
    for group, phases in (('ordinary', [controlled, controlled]),
                          ('selection', [controlled, controlled, 'card_select'])):
        for phase in phases:
            options = [dict(verb=verb) for verb in verbs] if phase == controlled else [
                dict(verb='SELECT_ONE', source_refs=['card:0']), dict(verb='SELECT_ONE', source_refs=['card:1'])]
            rows.append(dict(schema=anchor.SCHEMA, observation=dict(phase=phase, entities=[]),
                options=options, label=options[0], actor='historical_noncombat',
                metadata=dict(sample_group=group, anchor_kind=controlled)))
    with gzip.open(sample_path, 'wt') as out:
        out.writelines(json.dumps(row) + '\n' for row in rows)
    member, outcomes, counts, unnamed, automatic = anchor.export_report((report_path, source, sample_path, 'source'))
    exported = [json.loads(line) for line in gzip.decompress(member).splitlines()]
    assert [row['metadata']['sample_group'] for row in exported] == ['selection'] * 3
    assert [row['observation']['phase'] for row in exported] == [controlled, controlled, 'card_select']
    assert automatic == {controlled: 2} and not outcomes and not unnamed
    assert counts == {(controlled, 'historical_noncombat'): 3}


def test_export_keeps_winning_actions_and_every_fight_outcome(tmp_path):
    hero = dict(entity_type='player', ref='player', hp=50, max_hp=80, round=1)
    potion = dict(entity_type='potion', ref='potion:0', content_id='POTION.FIRE')
    entities = [hero, potion, dict(entity_type='card', zone='deck', content_id='CARD.BASH'),
                dict(entity_type='card', zone='hand', content_id='CARD.BASH'),
                dict(entity_type='enemy', ref='creature:1', hp=30)]

    def item(group, kind, actor, option=None, forced=False, **seen):
        option = option or dict(verb='END_TURN')
        public = [dict(e, **seen) if e is hero else e for e in entities]
        options = [option] if forced else [option, dict(verb='PLAY_CARD', source_refs=['hand:0'])]
        return dict(schema=anchor.SCHEMA, observation=dict(entities=public), options=options, label=option,
                    actor=actor, coverage={}, metadata=dict(sample_group=group, anchor_kind=kind, run_hash='h',
                    split_group='seed-group', character='Ironclad', ascension=7, state_sources=['derived'],
                    source_integrity='unverified_source'))
    (tmp_path / 'runs').mkdir()
    (tmp_path / 'samples').mkdir()
    fight = dict(kind='battle', status='verified', act=2, floor=5, encounter='ENCOUNTER.TOADPOLES_WEAK', entry_hp=50)
    battles = dict(run_hash='h', kind='battle', identity=dict(anchor.BUDGETS), anchors=[
        dict(fight, id='won', room_type='monster', outcome='win', exit_hp=56),
        dict(fight, id='lost', room_type='boss', outcome='loss', exit_hp=None),
        dict(fight, id='bad', room_type='elite', status='execution_failed', error='x'),
        dict(fight, id='idle', room_type='elite', outcome='win', exit_hp=50)])
    rests = dict(run_hash='h', kind='rest', identity={}, anchors=[
        dict(id='rest', kind='rest', status='verified', outcome='matched_record')])
    use = dict(verb='USE_POTION', source_refs=['potion:0'])

    def write(name, report, rows):
        (tmp_path / 'runs' / f'{name}.json').write_text(json.dumps(report))
        with gzip.open(tmp_path / 'samples' / f'{name}.jsonl.gz', 'wt') as stream:
            stream.write(''.join(json.dumps(row) + '\n' for row in rows))
    write('h.battle', battles, [item('won', 'battle', 'combat_solver', use), item('won', 'battle', 'combat_solver', round=2),
                                item('lost', 'battle', 'combat_solver'), item('bad', 'battle', 'combat_solver'),
                                item('idle', 'battle', 'combat_solver', forced=True)])
    write('h.rest', rests, [item('rest', 'rest', 'historical_noncombat')])
    (tmp_path / 'h.run.json').write_text(json.dumps(dict(map_point_history=[])))
    (tmp_path / 'manifest.json').write_text(json.dumps(dict(selected=[dict(run_hash='h', path=str(tmp_path / 'h.run.json'))])))
    summary = anchor.export(tmp_path, tmp_path / 'manifest.json')
    assert summary['counts'] == {'battle/combat_solver': 2, 'rest/historical_noncombat': 1}
    with gzip.open(tmp_path / 'independent-training.jsonl.gz', 'rt') as stream:
        assert [json.loads(line)['metadata']['sample_group'] for line in stream] == ['won', 'won', 'rest']
    # A lost fight teaches no actions, and neither does one won without a choice; both are outcomes.
    assert summary['rows_without_decision'] == {'battle': 1}
    assert summary['combat_outcomes']['counts'] == {'boss/loss': 1, 'elite/win': 1, 'regular/win': 1}
    with gzip.open(tmp_path / 'combat-outcomes.jsonl.gz', 'rt') as stream:
        won, lost, idle = map(json.loads, stream)
    assert (idle['result'], idle['frames']['count']) == ('win', 1)
    assert (won['start_hp'], won['end_hp'], won['hp_lost'], won['result'], won['turns']) == (50, 56, -6, 'win', 2)
    assert (lost['start_hp'], lost['end_hp'], lost['hp_lost'], lost['result'], lost['kind']) == (50, 0, 50, 'loss', 'boss')
    assert (won['seed'], won['encounter'], won['act'], won['max_hp']) == ('seed-group', 'TOADPOLES_WEAK', 2, 80)
    assert [e['content_id'] for e in won['potions_used']] == ['POTION.FIRE']
    # Solver fights, and the decision frames of every one of them: a lost fight keeps its frames.
    assert (won['actor'], lost['actor'], lost['frames']) == (
        'combat_solver', 'combat_solver', dict(file='samples/h.battle.jsonl.gz', group='lost', count=1))
    labelled = {label['anchor']: frames for label, frames in fights(tmp_path / 'combat-outcomes.jsonl.gz')}
    assert [f['action']['verb'] for f in labelled['won']] == ['USE_POTION', 'END_TURN']
    assert [next(e for e in f['public']['entities'] if e['entity_type'] == 'player')['hp']
            for f in labelled['lost']] == [50]
    # The input is the state the fight was entered with: no hand, no enemies.
    assert [(e['entity_type'], e.get('zone')) for e in won['entities']] == [('player', None), ('potion', None), ('card', 'deck')]
    # Several processes write the rows and the outcomes that one writes, in the same order.
    again = anchor.export(tmp_path, tmp_path / 'manifest.json', tmp_path / 'again.jsonl.gz',
                          tmp_path / 'again-outcomes.jsonl.gz', workers=2)
    assert (again['counts'], again['combat_outcomes']['counts']) == (summary['counts'], summary['combat_outcomes']['counts'])
    for one, several in (('independent-training', 'again'), ('combat-outcomes', 'again-outcomes')):
        assert gzip.open(tmp_path / f'{one}.jsonl.gz').read() == gzip.open(tmp_path / f'{several}.jsonl.gz').read()
    # A fight won again with potions forced gives its actions in place of the fight it
    # repeats; one lost again does not, and the outcomes are those of the first fights.
    potions = dict(run_hash='h', kind='potion', identity=dict(anchor.BUDGETS), anchors=[
        dict(fight, kind='potion', id='won-potion', room_type='monster', outcome='win', exit_hp=60),
        dict(fight, kind='potion', id='idle-potion', room_type='elite', outcome='loss', exit_hp=None)])
    write('h.potion', potions, [item('won-potion', 'potion', 'combat_solver', use),
                                item('idle-potion', 'potion', 'combat_solver', use)])
    forced = anchor.export(tmp_path, tmp_path / 'manifest.json')
    assert forced['counts'] == {'potion/combat_solver': 1, 'rest/historical_noncombat': 1}
    assert forced['combat_outcomes'] == summary['combat_outcomes']


def grid(edges, kinds):
    return {c: dict(coord=c, kind=k, children=edges.get(c, [])) for c, k in kinds.items()}


def test_export_keeps_rows_whose_map_names_the_recorded_bosses():
    boss = lambda *ids: dict(map_point_type='boss', rooms=[dict(room_type='boss', model_id=i) for i in ids])
    fight = dict(map_point_type='monster', rooms=[dict(room_type='monster', model_id='ENCOUNTER.X')])
    bosses = anchor.act_bosses(dict(map_point_history=[[fight, boss('ENCOUNTER.A_BOSS')], [fight, boss('ENCOUNTER.B_BOSS')],
                                                       [boss('ENCOUNTER.C_BOSS'), boss('ENCOUNTER.D_BOSS')]]))
    node = lambda floor, boss=None: dict(dict(entity_type='map_node', content_id='Boss' if boss else 'RestSite',
                                              floor=floor, col=3), **({'encounter': 'ENCOUNTER.' + boss} if boss else {}))
    row = lambda act, *nodes: dict(observation=dict(entities=[dict(entity_type='player', act=act), *nodes]))
    assert anchor.shows_bosses(row(1, node(15), node(16, 'A_BOSS')), bosses)
    # Two bosses close the last act of the highest ascension, in the order they are fought.
    assert anchor.shows_bosses(row(3, node(17, 'D_BOSS'), node(16, 'C_BOSS')), bosses)
    assert not anchor.shows_bosses(row(3, node(17, 'C_BOSS'), node(16, 'D_BOSS')), bosses)
    # A map that names another boss, or more bosses than the record holds, is refused.
    assert not anchor.shows_bosses(row(1, node(16, 'B_BOSS')), bosses)
    assert not anchor.shows_bosses(row(2, node(15, 'B_BOSS'), node(16, 'X_BOSS')), bosses)
    unnamed = row(1, dict(node(16, 'A_BOSS')))
    del unnamed['observation']['entities'][1]['encounter']
    assert not anchor.shows_bosses(unnamed, bosses)
    # The export changes no row.
    assert anchor.shows_bosses(row(2), bosses) and 'encounter' not in unnamed['observation']['entities'][1]


def test_an_anchor_save_holds_the_recorded_bosses():
    boss = lambda name: dict(node('boss'), rooms=[dict(room_type='boss', model_id=name, turns_taken=5)])
    run = summary([node('monster'), boss('ENCOUNTER.A_BOSS')])
    run['map_point_history'] += [[node('monster'), boss('ENCOUNTER.B_BOSS')],
                                 [node('monster'), boss('ENCOUNTER.C_BOSS'), boss('ENCOUNTER.D_BOSS')]]
    drawn = lambda first, second=None: dict(rooms=dict(boss_id=first, second_boss_id=second))
    base = dict(players=[dict(deck=[], relics=[], max_potion_slot_count=3, net_id=1)], extra_fields={},
                rng=dict(rngs={}), acts=[drawn('ENCOUNTER.SEED_1'), drawn('ENCOUNTER.SEED_2', 'ENCOUNTER.SEED_3'),
                                         drawn('ENCOUNTER.SEED_4', 'ENCOUNTER.SEED_5')])
    state = dict(deck=[], relics=[], potions=[], hp=50, max_hp=80, gold=10)
    save = anchor.build_save(base, state, run, 1, 1, run['map_point_history'][0][0], 'bosses')
    # The seed draws bosses for a player who has unlocked everything; the record says whom this
    # player met, in order. An act whose record does not fill its boss nodes keeps what it has.
    assert [(a['rooms']['boss_id'], a['rooms']['second_boss_id']) for a in save['acts']] == [
        ('ENCOUNTER.A_BOSS', None), ('ENCOUNTER.SEED_2', 'ENCOUNTER.SEED_3'), ('ENCOUNTER.C_BOSS', 'ENCOUNTER.D_BOSS')]
    assert base['acts'][0]['rooms']['boss_id'] == 'ENCOUNTER.SEED_1'


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
        assert frame['contract']['observation_schema'] == 'public-state-v6'
        digests[ascension, amount] = observation(frame).digest
    # What the game shows differently, the model is given differently.
    assert len(set(digests.values())) == 3


@pytest.mark.engine
def test_native_anchor_map_names_the_recorded_boss():
    if os.environ.get('SPIRE_CODEX_DATA_NATIVE_TESTS') != '1':
        pytest.skip('Set SPIRE_CODEX_DATA_NATIVE_TESTS=1 with the local solver configured')
    from combat_solver_cli.client import SolverEngine
    from spire_codex_data.anchor import settle
    with SolverEngine() as engine:
        first = settle(engine, engine.send(dict(cmd='start_run', character='Ironclad', seed='anchor-boss-check',
                                                ascension=3, decision_protocol=True)))
        base = engine.send(dict(cmd='anchor_state'))['save']
        pool = engine.send(dict(cmd='get_map'))['encounter_pools'][0]['bosses']
    named = lambda frame: [e.get('encounter') for e in frame['public']['entities']
                           if e.get('entity_type') == 'map_node' and e['content_id'] == 'Boss']
    drawn = base['acts'][0]['rooms']['boss_id']
    # A run names the boss its seed drew on the map from its first decision.
    assert named(first) == [drawn]
    met = next('ENCOUNTER.' + name for name in pool if 'ENCOUNTER.' + name != drawn)
    player = base['players'][0]
    nodes = [node('ancient', 'event'), node('rest_site'),
             dict(node('boss'), rooms=[dict(room_type='boss', model_id=met, turns_taken=5)])]
    run = dict(seed='anchor-boss-check', ascension=3, map_point_history=[nodes],
               players=[dict(deck=player['deck'], relics=player['relics'])])
    state = dict(deck=player['deck'], relics=player['relics'], potions=[], hp=41, max_hp=player['max_hp'], gold=77)
    save = anchor.build_save(base, state, run, 1, 2, nodes[1], 'anchor-boss-check')
    spot = dict(floor=2, map_point_type='rest_site', second_boss=False, route=['ancient', 'rest_site'],
                room=dict(type='rest_site'))
    with SolverEngine() as engine:
        frame = anchor.enter(engine, save, spot)
    # The anchor shows the boss this player met, which is what the export asks of a row.
    assert named(frame) == [met]
    assert anchor.shows_bosses(dict(observation=frame['public']), anchor.act_bosses(run))


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


def test_a_used_process_that_cannot_begin_the_next_run_is_replaced(monkeypatch):
    started = []

    class Engine:
        closed = False

        def __init__(self, config):
            started.append(self)

        def close(self):
            self.closed = True

    monkeypatch.setattr(anchor, 'SolverEngine', Engine)
    outcomes = iter([1, ValueError('enter_anchor:The run in progress cannot be ended'), 3, 4,
                     ValueError('native_frame_error'), ValueError('native_frame_error')])

    def start(engine):
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome
    with anchor.RunProcess('config') as shared:
        assert shared.begin(start) == (started[0], 1)
        # Refused on the used process: begun again on a new one, which is then used on.
        assert shared.begin(start) == (started[1], 3) and started[0].closed
        assert shared.begin(start) == (started[1], 4)
        # What also fails on a new process is the anchor's own failure.
        with pytest.raises(ValueError, match='native_frame_error'):
            shared.begin(start)
        assert started[1].closed and len(started) == 3
    assert started[2].closed


def test_a_finished_report_stands_until_the_summary_or_the_version_changes(tmp_path, monkeypatch):
    class NoEngine:
        def __init__(self, config):
            raise RuntimeError('generating')

    monkeypatch.setattr(anchor, 'SolverEngine', NoEngine)
    raw = tmp_path / 'h.run.json'
    raw.write_text(json.dumps(dict(summary([node('ancient', 'event'), node('monster')]), acts=[])))
    digest = anchor.hashlib.sha256(raw.read_bytes()).hexdigest()
    source = dict(run_hash='h', character='IRONCLAD', path=str(raw), sha256=digest)
    (tmp_path / 'runs').mkdir()
    # Whatever else a report notes of how it was made does not bind it.
    report = dict(identity=dict(version=anchor.VERSION, source_sha256=digest, kind='map', worker_sha256='a build'),
                  anchors=[dict(id='h-a01f01-map', kind='map', status='verified')])
    (tmp_path / 'runs/h.map.json').write_text(json.dumps(report))
    assert anchor.process(source, tmp_path, 'config', {'map'})['anchors'] == report['anchors']
    report['identity']['version'] = 'another'
    (tmp_path / 'runs/h.map.json').write_text(json.dumps(report))
    with pytest.raises(RuntimeError, match='generating'):
        anchor.process(source, tmp_path, 'config', {'map'})
    # A battle is also bound to the solver budgets it was fought with.
    fought = dict(identity=dict(anchor.BUDGETS, version=anchor.KIND_VERSIONS['battle'], source_sha256=digest,
                               kind='battle'), anchors=[])
    (tmp_path / 'runs/h.battle.json').write_text(json.dumps(fought))
    assert anchor.process(source, tmp_path, 'config', {'battle'})['anchors'] == []
    with pytest.raises(RuntimeError, match='generating'):
        anchor.process(source, tmp_path, 'config', {'battle'}, dict(anchor.BUDGETS, boss_budget_ms=1))


@pytest.mark.engine
def test_native_anchors_sharing_a_process_equal_those_of_new_processes():
    if os.environ.get('SPIRE_CODEX_DATA_NATIVE_TESTS') != '1':
        pytest.skip('Set SPIRE_CODEX_DATA_NATIVE_TESTS=1 with the local solver configured')
    from combat_solver_cli.client import DEFAULT_CONFIG, SolverEngine
    from combat_solver_cli.search_support import state_key
    from spire_codex_data.anchor import settle
    with SolverEngine() as engine:
        settle(engine, engine.send(dict(cmd='start_run', character='Ironclad', seed='anchor-process-check',
                                        ascension=3, decision_protocol=True)))
        base = engine.send(dict(cmd='anchor_state'))['save']
    player = base['players'][0]
    nodes = [node('ancient', 'event'), node('monster'), node('rest_site')]
    run = dict(seed='anchor-process-check', ascension=3, map_point_history=[nodes],
               players=[dict(deck=player['deck'], relics=player['relics'])])
    state = dict(deck=player['deck'], relics=player['relics'], potions=[], hp=41, max_hp=player['max_hp'], gold=77)
    offer = [dict(id='CARD.ANGER'), dict(id='CARD.HAVOC'), dict(id='CARD.CLASH')]
    reward = (anchor.build_save(base, state, run, 1, 2, nodes[1], 'process-check-reward'),
              dict(floor=2, map_point_type='monster', second_boss=False, route=['ancient', 'monster'],
                   room=dict(type='card_reward', cards=offer)))
    rest = (anchor.build_save(base, state, run, 1, 3, nodes[2], 'process-check-rest'),
            dict(floor=3, map_point_type='rest_site', second_boss=False, route=['ancient', 'monster', 'rest_site'],
                 room=dict(type='rest_site')))
    spots = [reward, rest, reward]
    alone = []
    for save, spot in spots:
        with SolverEngine() as engine:
            alone.append(anchor.enter(engine, save, dict(spot)))
    with anchor.RunProcess(DEFAULT_CONFIG) as shared:
        engines = []
        for (save, spot), expected in zip(spots, alone):
            engine, frame = shared.enter(save, dict(spot))
            engines.append(engine)
            # The anchor before is left where it stopped, its reward menu open: the run is ended there.
            assert state_key(frame) == state_key(expected) and frame['contract'] == expected['contract']
            assert frame['routing']['state_version'] == expected['routing']['state_version']
        assert engines[0] is engines[1] is engines[2]
        # A run in combat is not ended: the process is replaced.
        fight = (reward[0], dict(reward[1], room=dict(type='combat', encounter='TOADPOLES_WEAK')))
        shared.enter(fight[0], dict(fight[1]))
        engine, frame = shared.enter(*rest[:1], dict(rest[1]))
        assert engine is not engines[0] and state_key(frame) == state_key(alone[1])


@pytest.mark.engine
def test_native_battles_of_a_run_share_the_solver_and_the_replay_process():
    if os.environ.get('SPIRE_CODEX_DATA_NATIVE_TESTS') != '1':
        pytest.skip('Set SPIRE_CODEX_DATA_NATIVE_TESTS=1 with the local solver configured')
    from combat_solver_cli.client import DEFAULT_CONFIG, SolverEngine
    from spire_codex_data.anchor import settle
    with SolverEngine() as engine:
        settle(engine, engine.send(dict(cmd='start_run', character='Ironclad', seed='anchor-battle-check',
                                        ascension=0, decision_protocol=True)))
        base = engine.send(dict(cmd='anchor_state'))['save']
    player = base['players'][0]
    nodes = [node('ancient', 'event'), node('monster')]
    run = dict(seed='anchor-battle-check', ascension=0, map_point_history=[nodes],
               players=[dict(deck=player['deck'], relics=player['relics'])])
    state = dict(deck=player['deck'], relics=player['relics'], potions=[], hp=player['max_hp'],
                 max_hp=player['max_hp'], gold=99)
    save = anchor.build_save(base, state, run, 1, 2, nodes[1], 'battle-check')
    spot = dict(floor=2, map_point_type='monster', second_boss=False, route=['ancient', 'monster'],
                room_type='monster', room=dict(type='combat', encounter='TOADPOLES_WEAK'))
    budgets = dict(budget_ms=200, elite_budget_ms=200, boss_budget_ms=200)
    with anchor.RunProcess(DEFAULT_CONFIG) as shared, anchor.RunProcess(DEFAULT_CONFIG) as second:
        first = anchor.play_battle(shared, second, save, dict(spot), budgets)
        engines = shared.engine, second.engine
        again = anchor.play_battle(shared, second, save, dict(spot), budgets)
        # A finished fight leaves both processes free for the next one.
        assert (shared.engine, second.engine) == engines and engines[0] is not engines[1]
    for fight in (first, again):
        assert fight['outcome'] == 'win' and fight['exit_hp'] > 0 and len(fight['records']) > 2
    # The same entry state gives the same first decision, whatever the process did before.
    assert first['records'][0]['before_hash'] == again['records'][0]['before_hash']


def test_native_potion_anchor_uses_the_named_potions_and_no_other():
    if os.environ.get('SPIRE_CODEX_DATA_NATIVE_TESTS') != '1':
        pytest.skip('Set SPIRE_CODEX_DATA_NATIVE_TESTS=1 with the local solver configured')
    from pathlib import Path
    from combat_solver_cli.client import DEFAULT_CONFIG, SolverEngine
    config = Path(os.environ.get('COMBAT_SOLVER_CONFIG', DEFAULT_CONFIG))
    with SolverEngine(config) as engine:
        anchor.settle(engine, engine.send(dict(cmd='start_run', character='Ironclad', seed='anchor-potion-check',
                                               ascension=0, decision_protocol=True)))
        base = engine.send(dict(cmd='anchor_state'))['save']
    player = base['players'][0]
    nodes = [node('ancient', 'event'), node('monster')]
    run = dict(seed='anchor-potion-check', ascension=0, map_point_history=[nodes],
               players=[dict(deck=player['deck'], relics=player['relics'])])
    belt = ['POTION.BLOCK_POTION', 'POTION.FIRE_POTION', 'POTION.FIRE_POTION']
    state = dict(deck=player['deck'], relics=player['relics'], potions=belt, hp=player['max_hp'],
                 max_hp=player['max_hp'], gold=99)
    save = anchor.build_save(base, state, run, 1, 2, nodes[1], 'potion-check')
    spot = dict(floor=2, map_point_type='monster', second_boss=False, route=['ancient', 'monster'],
                room_type='monster', room=dict(type='combat', encounter='TOADPOLES_WEAK'))
    budgets = dict(budget_ms=200, elite_budget_ms=200, boss_budget_ms=200)

    def used(fight):
        return sorted(next(e['content_id'] for e in r['observation']['entities'] if e.get('ref') == r['label']['source_refs'][0])
                      for r in fight['records'] if r['label']['verb'] == 'USE_POTION')
    with anchor.RunProcess(config) as shared, anchor.RunProcess(config) as second:
        # A fight this easy is not worth a potion to the solver.
        assert used(anchor.play_battle(shared, second, save, dict(spot), budgets)) == []
        for forced in (['POTION.FIRE_POTION'], ['POTION.BLOCK_POTION', 'POTION.FIRE_POTION', 'POTION.FIRE_POTION']):
            fight = anchor.play_battle(shared, second, save, dict(spot, forced=forced), budgets)
            assert fight['outcome'] == 'win' and used(fight) == forced


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

