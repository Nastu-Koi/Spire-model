"""Summary rewards must prove both the offer and the order before yielding labels."""
from copy import deepcopy
import os
from pathlib import Path

import pytest

from spire_codex_data import anchor
from spire_codex_data import recorded
from spire_codex_data.tests.test_anchor_samples import anchors, choice, node, template


def summary(nodes):
    return dict(seed='SEED', ascension=0, map_point_history=[nodes],
                players=[dict(deck=deepcopy(template()['players'][0]['deck']), relics=[])])


def test_treasure_starts_before_opening_and_requires_a_recorded_offer():
    run = summary([node('treasure', current_gold=146, gold_gained=47,
                        relic_choices=[dict(choice='RELIC.ANCHOR', was_picked=True)])])
    found = anchors(run, ('treasure',))
    chest = found['a01f01-treasure']
    assert chest['ledger']['gold'] == 99
    assert all(r['id'] != 'RELIC.ANCHOR' for r in chest['ledger']['relics'])
    assert chest['expected']['gold'] == 146
    assert chest['room'] == dict(type='treasure', relic='RELIC.ANCHOR')
    assert chest['picked'] is True
    # An empty history does not say what was in an unopened or skipped chest.
    missing = anchors(summary([node('treasure')]), ('treasure',))['a01f01-treasure']
    assert missing['error'] == 'treasure_offer_unknown'


def test_skipped_card_with_unclaimed_rewards_does_not_make_an_incomplete_duplicate():
    stats = dict(card_choices=[choice(dict(id='CARD.' + c)) for c in ('ANGER', 'HAVOC', 'CLASH')],
                 potion_choices=[dict(choice='POTION.FIRE_POTION', was_picked=False)], gold_gained=15)
    found = anchors(summary([node('monster', **stats)]), ('reward', 'unclaimed'))
    assert found['a01f01-reward']['error'] == 'covered_by_unclaimed_reward_group'
    assert 'ledger' in found['a01f01-unclaimed']
    # Requesting only the legacy kind must not resurrect the incomplete observation.
    assert anchors(summary([node('monster', **stats)]), ('reward',))['a01f01-reward']['error'] \
        == 'covered_by_unclaimed_reward_group'


def reward_frame(contents, full=False):
    entities = [dict(entity_type='reward', ref=f'reward:{i}', content_id=cid)
                for i, cid in enumerate(contents)]
    candidates = [dict(verb='TAKE_REWARD', source_refs=[e['ref']], candidate_ref=f'take:{i}')
                  for i, e in enumerate(entities) if not full or not e['content_id'].startswith('POTION.')]
    candidates.append(dict(verb='LEAVE_REWARDS', candidate_ref='leave'))
    return dict(public=dict(phase='rewards', entities=entities), legal=dict(candidates=candidates))


def test_single_native_reward_has_a_proven_take_or_skip_and_never_guesses_discard():
    from spire_codex_data.rewards import RecordedReward
    record = dict(potion_choices=[dict(choice='POTION.FIRE_POTION', was_picked=True)])
    offered = reward_frame(['POTION.FIRE_POTION'])
    picker = RecordedReward(record)
    assert picker.choose(offered)['verb'] == 'TAKE_REWARD'
    assert picker.choose(reward_frame([]))['verb'] == 'LEAVE_REWARDS'
    assert picker.used
    with pytest.raises(ValueError, match='reward_offer_differs_from_record'):
        RecordedReward(record).choose(reward_frame(['POTION.BLOCK_POTION']))
    with pytest.raises(ValueError, match='recorded_reward_not_legal'):
        RecordedReward(record).choose(reward_frame(['POTION.FIRE_POTION'], full=True))
    with pytest.raises(ValueError, match='reward_potion_timing_unknown'):
        RecordedReward(dict(record, potion_discarded=['POTION.BLOCK_POTION'])).choose(offered)
    skipped = deepcopy(record)
    skipped['potion_choices'][0]['was_picked'] = False
    assert RecordedReward(skipped).choose(reward_frame(['POTION.FIRE_POTION'], full=True))['verb'] == 'LEAVE_REWARDS'


def test_multiple_rewards_and_unrecorded_gold_are_quarantined():
    from spire_codex_data.rewards import RecordedReward
    record = dict(relic_choices=[dict(choice='RELIC.ANCHOR', was_picked=True)],
                  potion_choices=[dict(choice='POTION.FIRE_POTION', was_picked=True)])
    with pytest.raises(ValueError, match='reward_order_unknown'):
        RecordedReward(record).choose(reward_frame(['POTION.FIRE_POTION', 'RELIC.ANCHOR']))
    with pytest.raises(ValueError, match='reward_offer_unrecorded'):
        RecordedReward({}).choose(reward_frame(['GoldReward']))


@pytest.mark.parametrize('relic', ['MAW_BANK', 'AMETHYST_AUBERGINE'])
def test_positive_gold_does_not_prove_that_all_gold_rewards_were_taken(relic):
    run = summary([
        node('ancient', 'event', relic_choices=[dict(choice='RELIC.' + relic, was_picked=True)]),
        node('monster', gold_gained=12, potion_choices=[dict(choice='POTION.FIRE_POTION', was_picked=False)])])
    found = anchors(run, ('unclaimed',))['a01f02-unclaimed']
    assert found.get('error') == 'gold_reward_multiple_sources'


def fight(room='elite', kind='claim', **extra):
    cards = [choice(dict(id='CARD.ANGER'), True), choice(dict(id='CARD.HAVOC')), choice(dict(id='CARD.CLASH'))]
    record = dict(card_choices=cards, cards_gained=[dict(id='CARD.ANGER')], current_gold=129, gold_gained=30,
                  relic_choices=[dict(choice='RELIC.ANCHOR', was_picked=True)],
                  potion_choices=[dict(choice='POTION.FIRE_POTION', was_picked=True)])
    record.update(extra)
    run = summary([node(room, **{k: v for k, v in record.items() if v is not None})])
    if any(c['was_picked'] for c in record['card_choices']):
        run['players'][0]['deck'].append(dict(id='CARD.ANGER', floor_added_to_deck=2))
    return anchors(run, (kind,)).get(f'a01f01-{kind}')


def test_claims_start_with_nothing_taken_and_end_where_the_card_choice_starts():
    found = fight()
    assert found['claims'] == dict(relics=['RELIC.ANCHOR'], gold=30, potions=['POTION.FIRE_POTION'])
    assert {k: found['room'][k] for k in ('type', 'gold', 'relics', 'potions')} == dict(
        type='rewards', gold=30, relics=['RELIC.ANCHOR'], potions=['POTION.FIRE_POTION'])
    assert [c['id'] for c in found['room']['cards']] == ['CARD.ANGER', 'CARD.HAVOC', 'CARD.CLASH']
    before, after = found['ledger'], found['expected']
    assert (before['gold'], before['potions']) == (99, []) and not any(r['id'] == 'RELIC.ANCHOR' for r in before['relics'])
    assert (after['gold'], after['potions']) == (129, ['POTION.FIRE_POTION']) and 'RELIC.ANCHOR' in after['relics']
    # The card is taken afterwards, from the state the reward anchor starts in.
    assert not any(c['id'] == 'CARD.ANGER' for c in before['deck']) and not any('ANGER' in c for c in after['deck'])
    # A potion left behind stays on the screen and is not claimed.
    left = fight(potion_choices=[dict(choice='POTION.FIRE_POTION', was_picked=False)])
    assert left['claims']['potions'] == [] and left['room']['potions'] == ['POTION.FIRE_POTION']


def test_pick_shows_the_card_offer_with_the_potion_and_what_was_left():
    found = fight(kind='pick')
    # The relic and the gold are taken already; the potion is taken after the card.
    assert {k: found['room'][k] for k in ('type', 'relics', 'potions')} == dict(
        type='rewards', relics=[], potions=['POTION.FIRE_POTION'])
    assert 'gold' not in found['room'] and found['later'] == ['POTION.FIRE_POTION']
    assert [c['id'] for c in found['room']['cards']] == ['CARD.ANGER', 'CARD.HAVOC', 'CARD.CLASH']
    before, after = found['ledger'], found['expected']
    assert (before['gold'], before['potions']) == (129, []) and any(r['id'] == 'RELIC.ANCHOR' for r in before['relics'])
    assert not any(c['id'] == 'CARD.ANGER' for c in before['deck'])
    assert after['potions'] == ['POTION.FIRE_POTION'] and sum('ANGER' in c for c in after['deck']) == 1
    # What was left stays on the screen, and nothing is taken after the card.
    left = fight(kind='pick', potion_choices=[dict(choice='POTION.FIRE_POTION', was_picked=False)],
                 relic_choices=[dict(choice='RELIC.ANCHOR', was_picked=False)])
    assert (left['room']['relics'], left['room']['potions'], left['later']) == (
        ['RELIC.ANCHOR'], ['POTION.FIRE_POTION'], [])
    assert left['ledger']['potions'] == [] and left['expected']['potions'] == []
    # A screen with the card offer alone is the reward anchor's; no card taken is a leave.
    assert fight(kind='pick', potion_choices=None) is None
    skipped = [choice(dict(id='CARD.' + c)) for c in ('ANGER', 'HAVOC', 'CLASH')]
    assert fight(kind='pick', card_choices=skipped, cards_gained=None) is None
    assert fight(kind='pick', potion_discarded=['POTION.FIRE_POTION'])['error'] == 'reward_potion_timing_unknown'


@pytest.mark.parametrize('extra,reason', [
    (dict(potion_discarded=['POTION.FIRE_POTION']), 'reward_potion_timing_unknown'),
    (dict(potion_choices=[dict(choice='POTION.FIRE_POTION', was_picked=True),
                          dict(choice='POTION.BLOCK_POTION', was_picked=True)]), 'reward_potion_source_unknown'),
    (dict(room='monster'), 'reward_relic_source_unknown'),
    (dict(gold_gained=None), 'gold_reward_unsettled'),
    (dict(gold_stolen=5), 'gold_reward_multiple_sources'),
    (dict(relic_choices=[dict(choice='RELIC.CAULDRON', was_picked=True)]), 'reward_potion_source_unknown'),
])
def test_claims_the_record_cannot_place_on_the_reward_screen_are_isolated(extra, reason):
    assert fight(**extra).get('error', '').split(':')[0] == reason


@pytest.fixture
def native():
    if os.environ.get('SPIRE_CODEX_DATA_NATIVE_TESTS') != '1':
        pytest.skip('Set SPIRE_CODEX_DATA_NATIVE_TESTS=1 with the local solver configured')
    from combat_solver_cli.client import DEFAULT_CONFIG, SolverEngine
    config = Path(os.environ.get('COMBAT_SOLVER_CONFIG', DEFAULT_CONFIG))
    with SolverEngine(config) as engine:
        anchor.settle(engine, engine.send(dict(cmd='start_run', character='Ironclad', seed='summary-rewards-check',
                                               ascension=0, decision_protocol=True)))
        base = engine.send(dict(cmd='anchor_state'))['save']
    return config, base


def spot_and_save(base, kind, state=None, **room):
    p = base['players'][0]
    state = state or dict(deck=p['deck'], relics=p['relics'], potions=[], hp=50, max_hp=p['max_hp'], gold=99)
    point = node(kind, 'event' if kind == 'unknown' else kind)
    run = dict(seed='summary-rewards-check', ascension=base['ascension'], players=[dict(deck=p['deck'], relics=p['relics'])],
               map_point_history=[[node('ancient', 'event'), point]])
    save = anchor.build_save(base, state, run, 1, 2, point, 'reward-native-check')
    spot = dict(floor=2, map_point_type=kind, second_boss=False, route=['ancient', kind],
                kind='treasure' if kind == 'treasure' else 'event', room=room)
    return spot, save


@pytest.mark.parametrize('relic,picked', [('ANCHOR', True), ('OLD_COIN', True), ('STRAWBERRY', True),
                                         ('POTION_BELT', True), ('ANCHOR', False)])
def test_native_chest_open_pick_effects_and_independent_replay(native, relic, picked):
    config, base = native
    spot, save = spot_and_save(base, 'treasure', type='treasure', relic='RELIC.' + relic, gold_roll=47)
    with anchor.RunProcess(config) as shared, anchor.RunProcess(config) as second:
        engine, frame = shared.enter(save, deepcopy(spot))
        assert not any(e.get('zone') == 'treasure' for e in frame['public']['entities'])
        choose = lambda verb: next(c for c in frame['legal']['candidates'] if c['verb'] == verb)
        frame = anchor.advance(engine, frame, choose('OPEN_CHEST'))
        if frame['public']['phase'] == 'rewards':
            frame = anchor.advance(engine, frame, choose('LEAVE_REWARDS'))
        assert anchor.player(frame)['gold'] == 146
        frame = anchor.advance(engine, frame, choose('TAKE_TREASURE_RELIC' if picked else 'LEAVE_ROOM'))
        assert frame['public']['phase'] == 'map'
        final = anchor.installed_state(engine)
        if picked and relic == 'OLD_COIN':
            assert final['gold'] == 446
        if picked and relic == 'STRAWBERRY':
            assert (final['current_hp'], final['max_hp']) == (57, 87)
        if picked and relic == 'POTION_BELT':
            assert final['max_potion_slot_count'] == save['players'][0]['max_potion_slot_count'] + 2
        result = anchor.play_treasure(shared, second, save, dict(spot, picked=picked,
            changes=recorded.changes({}), expected=recorded.end_state(final)))
        assert result['recovery']['matching_gold_rolls'] == [47]
        assert result['recovery']['replay'] == 'independent_native_process'
        assert [r['label']['verb'] for r in result['records']] == [
            'OPEN_CHEST', 'TAKE_TREASURE_RELIC' if picked else 'LEAVE_ROOM']
        before_pick = result['records'][-1]['observation']['entities']
        assert not any(e.get('entity_type') == 'relic' and e.get('zone') != 'treasure'
                       and e.get('content_id') == 'RELIC.' + relic for e in before_pick)
        assert next(e['gold'] for e in before_pick if e.get('entity_type') == 'player') == 146
        # The supplied outcome cannot be made true by setting player.gold.
        wrong = dict(recorded.end_state(final), gold=50000)
        with pytest.raises(ValueError, match='treasure_gold_unresolved'):
            anchor.play_treasure(shared, second, save, dict(spot, picked=picked,
                changes=recorded.changes({}), expected=wrong))


@pytest.mark.parametrize('take_potion', [True, False])
def test_native_event_single_reward_continues_the_original_event(native, take_potion):
    config, base = native
    spot, save = spot_and_save(base, 'unknown', type='event', event='WELLSPRING')
    with anchor.RunProcess(config) as shared:
        engine, frame = shared.enter(save, deepcopy(spot))
        page = engine.send(dict(cmd='anchor_event'))
        index, option = next((i, o) for i, o in enumerate(page['options']) if o['text_key'].endswith('.BOTTLE'))
        frame = anchor.advance(engine, frame, next(c for c in frame['legal']['candidates']
                                                   if c.get('source_refs') == [f'option:{index}']))
        assert frame['public']['phase'] == 'rewards'
        offered = [e for e in frame['public']['entities'] if e.get('entity_type') == 'reward']
        assert len(offered) == 1 and offered[0]['content_id'].startswith('POTION.')
        verb = 'TAKE_REWARD' if take_potion else 'LEAVE_REWARDS'
        frame = anchor.advance(engine, frame, next(c for c in frame['legal']['candidates'] if c['verb'] == verb))
        while frame['public']['phase'] != 'map':
            assert len(frame['legal']['candidates']) == 1
            frame = anchor.advance(engine, frame, frame['legal']['candidates'][0])
        variables = engine.send(dict(cmd='anchor_event'))['variables']
        record = dict(spot, choices=[dict(title=option['history']['title'], variables=variables)],
            changes=recorded.changes({}), expected=recorded.end_state(anchor.installed_state(engine)),
            reward_record=dict(potion_choices=[dict(choice=offered[0]['content_id'], was_picked=take_potion)]))
        result = anchor.play_event(shared, save, record)
        assert result['reward_decisions'] >= 1 and 'unique_reward_choice' in result['state_sources']
        choices = [r for r in result['records'] if r['observation']['phase'] == 'rewards']
        assert choices[0]['label']['verb'] == verb
        assert not any(e.get('entity_type') == 'potion' for e in choices[0]['observation']['entities'])


def test_native_chest_a10_rounding_has_one_public_result(native):
    config, base = native
    base = deepcopy(base)
    base['ascension'] = 10
    spot, save = spot_and_save(base, 'treasure', type='treasure', relic='RELIC.ANCHOR', gold_roll=48)
    with anchor.RunProcess(config) as shared, anchor.RunProcess(config) as second:
        engine, frame = shared.enter(save, deepcopy(spot))
        frame = anchor.advance(engine, frame, next(c for c in frame['legal']['candidates'] if c['verb'] == 'OPEN_CHEST'))
        frame = anchor.advance(engine, frame, next(c for c in frame['legal']['candidates'] if c['verb'] == 'TAKE_TREASURE_RELIC'))
        result = anchor.play_treasure(shared, second, save, dict(spot, picked=True,
            changes=recorded.changes({}), expected=recorded.end_state(anchor.installed_state(engine))))
        rolls = result['recovery']['matching_gold_rolls']
        assert rolls == [48, 49]  # floor(raw * 0.75) == 36 for both native rolls.
        assert all(r['label'] in r['options'] for r in result['records'])


def claim_spot(base, state, expected, **claims):
    spot, save = spot_and_save(base, 'elite', state=state, type='rewards', gold=claims['gold'],
                               relics=claims['relics'], potions=claims['potions'],
                               cards=[dict(id='CARD.' + c) for c in ('ANGER', 'HAVOC', 'CLASH')])
    spot.update(kind='claim', expected=expected,
                claims=dict(relics=claims['relics'], gold=claims['gold'], potions=claims['potions']))
    return spot, save


def test_native_claims_take_relic_gold_and_potion_with_the_whole_screen_shown(native):
    config, base = native
    p = base['players'][0]
    state = dict(deck=p['deck'], relics=p['relics'], potions=[], hp=50, max_hp=p['max_hp'], gold=99)
    expected = recorded.expected_end(dict(state, gold=129, relics=p['relics'] + [dict(id='RELIC.ANCHOR')],
                                          potions=['POTION.FIRE_POTION']))
    spot, save = claim_spot(base, state, expected, relics=['RELIC.ANCHOR'], gold=30, potions=['POTION.FIRE_POTION'])
    with anchor.RunProcess(config) as shared:
        result = anchor.play_claim(shared, save, deepcopy(spot))
        rows = result['records']
        content = lambda r: next(e['content_id'] for e in r['observation']['entities']
                                 if e.get('ref') == r['label']['source_refs'][0])
        assert [content(r) for r in rows] == ['RELIC.ANCHOR', 'GoldReward', 'POTION.FIRE_POTION']
        assert all(r['label']['verb'] == 'TAKE_REWARD' and r['coverage']['label'] == 'reconstructed_order' for r in rows)
        # Every claim is chosen with the card offer and the way out still on the screen.
        for r in rows:
            verbs = [o['verb'] for o in r['options']]
            assert verbs.count('TAKE_CARD_REWARD') == 3 and 'LEAVE_REWARDS' in verbs
        shown = [sorted(e['content_id'] for e in r['observation']['entities'] if e.get('entity_type') == 'reward')
                 for r in rows]
        assert shown[0] == ['CardReward', 'GoldReward', 'POTION.FIRE_POTION', 'RELIC.ANCHOR']
        assert shown[2] == ['CardReward', 'POTION.FIRE_POTION']
        assert result['recovery'] == dict(order='relic_gold_potion', claims=spot['claims'])
        # A relic whose pickup changes the state again does not reproduce the record.
        berry = dict(state, hp=57, max_hp=p['max_hp'] + 7)
        spot2, save2 = claim_spot(base, berry, recorded.expected_end(dict(berry, gold=129,
                                  relics=p['relics'] + [dict(id='RELIC.STRAWBERRY')])),
                                  relics=['RELIC.STRAWBERRY'], gold=30, potions=[])
        with pytest.raises(ValueError, match='outcome_differs_from_record'):
            anchor.play_claim(shared, save2, deepcopy(spot2))
        # A full belt cannot take the potion: the record of taking it is not this screen's.
        full = dict(state, potions=['POTION.BLOCK_POTION'] * p['max_potion_slot_count'])
        spot3, save3 = claim_spot(base, full, expected, relics=[], gold=30, potions=['POTION.FIRE_POTION'])
        with pytest.raises(ValueError, match='recorded_reward_not_legal'):
            anchor.play_claim(shared, save3, deepcopy(spot3))


def test_native_claim_pays_the_shown_gold_at_ascension_ten(native):
    # Ascension lowers what a fight offers, not what a claim pays: shown and paid agree.
    config, base = native
    base = deepcopy(base)
    base['ascension'] = 10
    p = base['players'][0]
    state = dict(deck=p['deck'], relics=p['relics'], potions=[], hp=50, max_hp=p['max_hp'], gold=99)
    spot, save = claim_spot(base, state, recorded.expected_end(dict(state, gold=135)), relics=[], gold=36, potions=[])
    with anchor.RunProcess(config) as shared:
        result = anchor.play_claim(shared, save, deepcopy(spot))
        gold = next(e for e in result['records'][0]['observation']['entities'] if e.get('content_id') == 'GoldReward')
        assert gold['gold'] == 36 and len(result['records']) == 1


def pick_spot(base, state, expected, later, **shown):
    cards = [dict(id='CARD.' + c) for c in ('ANGER', 'HAVOC', 'CLASH')]
    spot, save = spot_and_save(base, 'elite', state=state, type='rewards', cards=cards, **shown)
    spot.update(kind='pick', expected=expected, picked=cards[:1], later=later)
    return spot, save


def test_native_pick_takes_the_card_with_the_potion_on_the_screen_then_the_potion(native):
    config, base = native
    p = base['players'][0]
    state = dict(deck=p['deck'], relics=p['relics'], potions=[], hp=50, max_hp=p['max_hp'], gold=99)
    deck = p['deck'] + [dict(id='CARD.ANGER', floor_added_to_deck=2)]
    content = lambda r: next(e['content_id'] for e in r['observation']['entities']
                             if e.get('ref') == r['label']['source_refs'][0])
    shown = lambda r: sorted(e['content_id'] for e in r['observation']['entities'] if e.get('entity_type') == 'reward')
    with anchor.RunProcess(config) as shared:
        spot, save = pick_spot(base, state, recorded.expected_end(dict(state, deck=deck, potions=['POTION.FIRE_POTION'])),
                               ['POTION.FIRE_POTION'], potions=['POTION.FIRE_POTION'], relics=[])
        result = anchor.play_choice(shared, save, deepcopy(spot))
        rows = result['records']
        assert [(r['label']['verb'], content(r)) for r in rows] == [
            ('TAKE_CARD_REWARD', 'CARD.ANGER'), ('TAKE_REWARD', 'POTION.FIRE_POTION')]
        assert shown(rows[0]) == ['CardReward', 'POTION.FIRE_POTION'] and shown(rows[1]) == ['POTION.FIRE_POTION']
        assert all(r['coverage']['label'] == 'reconstructed_order' for r in rows)
        assert result['recovery'] == dict(order='card_potion', claims=dict(potions=['POTION.FIRE_POTION']))
        # A potion and a relic that were left: the card is the only step, with both in view
        # and, the belt being full, the potion out of reach.
        full = dict(state, potions=['POTION.BLOCK_POTION'] * p['max_potion_slot_count'])
        spot2, save2 = pick_spot(base, full, recorded.expected_end(dict(full, deck=deck)), [],
                                 potions=['POTION.FIRE_POTION'], relics=['RELIC.ANCHOR'])
        result = anchor.play_choice(shared, save2, deepcopy(spot2))
        (row,) = result['records']
        assert (row['label']['verb'], content(row)) == ('TAKE_CARD_REWARD', 'CARD.ANGER')
        assert shown(row) == ['CardReward', 'POTION.FIRE_POTION', 'RELIC.ANCHOR']
        assert row['coverage']['label'] == 'historical_choice'
        takes = [content(dict(row, label=o)) for o in row['options'] if o['verb'] == 'TAKE_REWARD']
        assert takes == ['RELIC.ANCHOR'] and any(o['verb'] == 'LEAVE_REWARDS' for o in row['options'])
        # A potion the record has taken into a full belt is not this screen's.
        spot3, save3 = pick_spot(base, full, recorded.expected_end(dict(full, deck=deck)),
                                 ['POTION.FIRE_POTION'], potions=['POTION.FIRE_POTION'], relics=[])
        with pytest.raises(ValueError, match='recorded_reward_not_legal'):
            anchor.play_choice(shared, save3, deepcopy(spot3))
