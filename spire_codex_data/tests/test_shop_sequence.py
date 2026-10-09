"""Shop anchors: the stock a summary records, the shop stream, and the order of the purchases."""
import os
from copy import deepcopy

import pytest

from spire_codex_data import anchor, recorded, shop_sequence
from spire_codex_data.shop_sequence import (advance, locked_sets, next_order, purchases, ranked, recorded_stock,
                                            search, seatings, shop_history, stock_candidates)

INFO = {
    **{'CARD.' + name: dict(card_type=kind, rarity=rarity, colorless=False) for name, kind, rarity in (
        ('ANGER', 'Attack', 'Common'), ('BLUDGEON', 'Attack', 'Rare'), ('TREMBLE', 'Skill', 'Common'),
        ('ARMAMENTS', 'Skill', 'Common'), ('INFLAME', 'Power', 'Uncommon'))},
    'CARD.DISCOVERY': dict(card_type='Skill', rarity='Uncommon', colorless=True),
    'CARD.SECRET_WEAPON': dict(card_type='Skill', rarity='Rare', colorless=True),
    'RELIC.ANCHOR': dict(rarity='Common'), 'RELIC.VAJRA': dict(rarity='Uncommon'),
    'RELIC.MEMBERSHIP_CARD': dict(rarity='Shop'),
}


def choice(name, picked=False):
    return dict(choice=name, was_picked=picked)


def node(kind, room_type=None, **stats):
    return dict(map_point_type=kind, rooms=[dict(room_type=room_type or kind, turns_taken=0)],
                player_stats=[dict(dict(current_hp=60, max_hp=80, current_gold=110), **stats)])


def planned(nodes, final_deck):
    start = [dict(id='CARD.' + name) for name in ('STRIKE_IRONCLAD', 'DEFEND_IRONCLAD', 'BASH')]
    run = dict(seed='SEED', ascension=0, map_point_history=[nodes],
               players=[dict(deck=[dict(id='CARD.' + name) for name in final_deck], relics=[])])
    states, info = anchor.node_states(run, dict(players=[dict(deck=start, relics=[], current_hp=64, max_hp=80, gold=99)]))
    return {a['id'].split('-', 1)[1]: a for a, _ in anchor.plan(run, 'h', states, info, {'shop'})}


def shop_stats(**changes):
    """ANGER and the membership card bought, the first strike removed; the rest left."""
    stats = dict(
        card_choices=[dict(card=dict(id='CARD.' + n), was_picked=False)
                      for n in ('BLUDGEON', 'TREMBLE', 'ARMAMENTS', 'INFLAME', 'DISCOVERY', 'SECRET_WEAPON')],
        cards_gained=[dict(id='CARD.ANGER')], cards_removed=[dict(id='CARD.STRIKE_IRONCLAD', floor_added_to_deck=1)],
        relic_choices=[choice('RELIC.ANCHOR'), choice('RELIC.VAJRA'), choice('RELIC.MEMBERSHIP_CARD', True)],
        bought_relics=['RELIC.MEMBERSHIP_CARD'],
        potion_choices=[choice('POTION.FIRE_POTION'), choice('POTION.BLOCK_POTION'), choice('POTION.SWIFT_POTION')])
    stats.update(changes)
    return stats


def test_the_stream_advances_as_the_game_draws():
    # The shop stream of a native Ironclad run before and after its first merchant.
    before = dict(counter=0, s0=18434571650920319132, s1=274641435432117288, s2=906975981272057174,
                  s3=8933162181968937524)
    after = dict(counter=28, s0=207785125236311147, s1=14646820044184551964, s2=2953686200228430957,
                 s3=11960315200458538518)
    assert advance(before, shop_sequence.SHOP_DRAWS) == after
    assert advance(advance(before, 11), 17) == after and advance(before, 0) == before


def test_earlier_merchants_set_the_stream_and_the_removal_price():
    def node(room, **stats):
        return dict(rooms=[room], player_stats=[stats])
    shop, fake = dict(room_type='shop'), dict(room_type='event', model_id='EVENT.FAKE_MERCHANT')
    run = dict(map_point_history=[[
        node(shop, cards_removed=[dict(id='CARD.A')]),
        node(fake),
        node(shop, bought_relics=['RELIC.THE_COURIER'], relic_choices=[choice('RELIC.THE_COURIER', True)]),
        # With the courier every purchase draws a replacement: a card two, a relic one, a potion three.
        node(shop, cards_gained=[dict(id='CARD.B')], bought_relics=['RELIC.X'], bought_potions=['POTION.Y']),
        node(shop)]])
    assert shop_history(run, 1) == dict(draws=0, removals=0, courier=False)
    assert shop_history(run, 3) == dict(draws=28 + 6, removals=1, courier=False)
    assert shop_history(run, 4) == dict(draws=28 + 6 + 28, removals=1, courier=True)
    assert shop_history(run, 5) == dict(draws=28 + 6 + 28 + 28 + 2 + 1 + 3, removals=1, courier=True)


def test_a_plain_shop_record_is_read_and_others_are_refused():
    stock = recorded_stock(shop_stats())
    assert [c['id'] for c in stock['left_cards']][:2] == ['CARD.BLUDGEON', 'CARD.TREMBLE']
    assert (stock['bought_relics'], stock['left_relics']) == (['RELIC.MEMBERSHIP_CARD'], ['RELIC.ANCHOR', 'RELIC.VAJRA'])
    assert purchases(stock) == [('card', 'CARD.ANGER'), ('relic', 'RELIC.MEMBERSHIP_CARD'), ('removal', None)]
    for changes, reason in (
            # A purchase opened a card reward: its cards share the list with the shop's.
            (dict(card_choices=shop_stats()['card_choices'] + [dict(card=dict(id='CARD.X'), was_picked=True)]),
             'shop_mixed_card_sources'),
            # A relic gained without being bought.
            (dict(bought_relics=[]), 'shop_gains_not_purchases'),
            # More stock than a shop holds: restocked, or gained otherwise.
            (dict(cards_gained=[dict(id='CARD.ANGER'), dict(id='CARD.CLASH')]), 'shop_stock_size:8/3/3'),
            (dict(cards_removed=[dict(id='CARD.A'), dict(id='CARD.B')]), 'shop_several_removals')):
        with pytest.raises(ValueError, match=reason):
            recorded_stock(shop_stats(**changes))
    drunk = recorded_stock(shop_stats(potion_used=['POTION.FOUL_POTION'], potion_discarded=['POTION.FIRE_POTION']))
    assert purchases(drunk)[-2:] == [('use', 'POTION.FOUL_POTION'), ('discard', 'POTION.FIRE_POTION')]


def test_the_recorded_stock_fills_the_typed_slots():
    kind = lambda card: INFO[card]['card_type']
    # What was left keeps its order; a bought card takes the free slot of its type.
    assert seatings(['CARD.BLUDGEON', 'CARD.TREMBLE', 'CARD.ARMAMENTS', 'CARD.INFLAME'], ['CARD.ANGER'],
                    shop_sequence.CARD_SLOTS, kind) == [
        ['CARD.BLUDGEON', 'CARD.ANGER', 'CARD.TREMBLE', 'CARD.ARMAMENTS', 'CARD.INFLAME'],
        ['CARD.ANGER', 'CARD.BLUDGEON', 'CARD.TREMBLE', 'CARD.ARMAMENTS', 'CARD.INFLAME']]
    assert seatings(['CARD.TREMBLE', 'CARD.TREMBLE'], [], ('Skill', 'Attack'), kind) == []
    cards, colorless, relics, potions = stock_candidates(recorded_stock(shop_stats()), INFO)
    assert len(cards) == 2 and colorless == ['CARD.DISCOVERY', 'CARD.SECRET_WEAPON']
    # Two relics of a rolled rarity in their recorded order, then the shop's own.
    assert relics == [['RELIC.ANCHOR', 'RELIC.VAJRA', 'RELIC.MEMBERSHIP_CARD']]
    assert potions == [['POTION.FIRE_POTION', 'POTION.BLOCK_POTION', 'POTION.SWIFT_POTION']]
    # One of the rolled relics bought: it stood in either slot.
    stats = shop_stats(relic_choices=[choice('RELIC.VAJRA'), choice('RELIC.MEMBERSHIP_CARD'), choice('RELIC.ANCHOR', True)],
                       bought_relics=['RELIC.ANCHOR'])
    assert stock_candidates(recorded_stock(stats), INFO)[2] == [
        ['RELIC.VAJRA', 'RELIC.ANCHOR', 'RELIC.MEMBERSHIP_CARD'], ['RELIC.ANCHOR', 'RELIC.VAJRA', 'RELIC.MEMBERSHIP_CARD']]
    with pytest.raises(ValueError, match='shop_cards_do_not_fill_the_slots'):
        stock_candidates(recorded_stock(shop_stats(cards_gained=[dict(id='CARD.INFLAME')])), INFO)


def test_unlock_states_are_tried_from_the_fullest():
    options = locked_sets(['A', 'B'])
    assert options == [set(), {'A'}, {'B'}, {'A', 'B'}]
    assert len(locked_sets(shop_sequence.PARTS['colorless'][1])) == 32


def test_the_first_order_tried_puts_price_changers_first():
    actions = [('removal', None), ('card', 'CARD.B'), ('potion', 'POTION.P'), ('relic', 'RELIC.R'), ('discard', 'POTION.Q')]
    first = next_order(ranked(actions), set())
    assert [kind for kind, _ in first] == ['discard', 'relic', 'potion', 'card', 'removal']
    # Equal actions are one: two copies of a card and a relic have three orders, not six.
    twins, refused, found = ranked([('card', 'CARD.A'), ('card', 'CARD.A'), ('relic', 'RELIC.R')]), set(), []
    while (order := next_order(twins, refused)) is not None:
        found.append(order)
        refused.add(order)
    assert len(found) == len(set(found)) == 3
    # A refused beginning rules out every order that starts with it.
    refused = {(('relic', 'RELIC.R'),)}
    assert next_order(twins, refused) == (('card', 'CARD.A'), ('relic', 'RELIC.R'), ('card', 'CARD.A'))


class Shop:
    """A merchant with prices, a membership discount, two potion slots and a removal service."""

    def __init__(self, gold, potions=()):
        self.start = dict(gold=gold, potions=list(potions))
        self.entries = 0

    def enter(self):
        self.entries += 1
        self.gold, self.potions = self.start['gold'], list(self.start['potions'])
        self.stock = {'CARD.ANGER': 50, 'RELIC.MEMBERSHIP_CARD': 100, 'POTION.FIRE_POTION': 60}
        self.deck = ['CARD.STRIKE_IRONCLAD', 'CARD.BASH']
        self.relics, self.removal, self.selecting = [], 100, False
        return self.frame()

    def price(self, base):
        return base // 2 if 'RELIC.MEMBERSHIP_CARD' in self.relics else base

    def frame(self):
        entities = [dict(entity_type='player', ref='player', gold=self.gold)]
        entities += [dict(entity_type='potion', ref=f'potion:{i}', content_id=p) for i, p in enumerate(self.potions)]
        if self.selecting:
            entities += [dict(entity_type='card', ref=f'card:{i}', content_id=c) for i, c in enumerate(self.deck)]
            candidates = [dict(verb='SELECT_ONE', source_refs=[f'card:{i}']) for i in range(len(self.deck))]
            return dict(boundary='decision', legal=dict(candidates=candidates), public=dict(
                phase='card_select', entities=entities, selection_context=dict(operation='remove', min_total=1, max_total=1)))
        candidates = []
        items = list(self.stock.items()) + ([('MerchantCardRemovalEntry', self.removal)] if self.removal else [])
        for i, (name, base) in enumerate(items):
            entities.append(dict(entity_type='shop_item', ref=f'merchant:{i}', content_id=name, price=self.price(base)))
            if self.price(base) <= self.gold and (not name.startswith('POTION') or len(self.potions) < 2):
                candidates.append(dict(verb='BUY_ITEM', source_refs=[f'merchant:{i}']))
        candidates += [dict(verb='DISCARD_POTION', source_refs=[f'potion:{i}']) for i in range(len(self.potions))]
        candidates.append(dict(verb='LEAVE_ROOM', source_refs=[]))
        return dict(boundary='decision', legal=dict(candidates=candidates), public=dict(phase='shop', entities=entities))

    def step(self, frame, candidate):
        entities = {e['ref']: e for e in frame['public']['entities']}
        target = entities.get((candidate['source_refs'] or [None])[0], {})
        if candidate['verb'] == 'SELECT_ONE':
            self.deck.remove(target['content_id'])
            self.selecting = False
        elif candidate['verb'] == 'DISCARD_POTION':
            self.potions.remove(target['content_id'])
        elif target['content_id'] == 'MerchantCardRemovalEntry':
            self.gold -= target['price']
            self.removal, self.selecting = None, True
        else:
            self.gold -= target['price']
            del self.stock[target['content_id']]
            (self.relics if target['content_id'].startswith('RELIC') else self.potions if
             target['content_id'].startswith('POTION') else self.deck).append(target['content_id'])
        return self.frame()

    def state(self):
        return dict(gold=self.gold, deck=sorted(self.deck), relics=self.relics, potions=sorted(self.potions))


def record(**changes):
    stock = dict(bought_cards=[], bought_relics=[], bought_potions=[], used_potions=[], discarded_potions=[],
                 removed=[], enchanted=[], upgraded=[], transformed=[])
    stock.update(changes)
    return stock


def rows_of(shop, stock, expected, **options):
    row = lambda frame, candidate, leaving=False: (candidate['verb'], leaving)
    return search(shop.enter, shop.step, shop.state, stock, expected, row, **options)


def test_an_order_is_found_that_the_shop_accepts_and_that_ends_as_recorded():
    # 190 gold buys all three only at the member's price: the card comes after the relic.
    shop = Shop(190)
    stock = record(bought_cards=[dict(id='CARD.ANGER')], bought_relics=['RELIC.MEMBERSHIP_CARD'],
                   removed=[dict(id='CARD.STRIKE_IRONCLAD')])
    expected = dict(gold=190 - 100 - 25 - 50, deck=['CARD.ANGER', 'CARD.BASH'], relics=['RELIC.MEMBERSHIP_CARD'], potions=[])
    rows, tried = rows_of(shop, stock, expected)
    assert tried == 1
    # The removal opens a selection, answered with the recorded card; leaving is the last decision.
    assert rows == [('BUY_ITEM', False), ('BUY_ITEM', False), ('BUY_ITEM', False), ('SELECT_ONE', False), ('LEAVE_ROOM', True)]

    # The record says the card cost its full price: it was bought before the membership.
    shop = Shop(250)
    expected = dict(gold=250 - 50 - 100 - 50, deck=['CARD.ANGER', 'CARD.BASH'], relics=['RELIC.MEMBERSHIP_CARD'], potions=[])
    rows, tried = rows_of(shop, stock, expected)
    assert tried > 1 and rows[-1] == ('LEAVE_ROOM', True)

    # A full potion belt takes a new potion only after one is thrown away.
    shop = Shop(100, potions=['POTION.A', 'POTION.B'])
    stock = record(bought_potions=['POTION.FIRE_POTION'], discarded_potions=['POTION.A'])
    expected = dict(gold=40, deck=['CARD.BASH', 'CARD.STRIKE_IRONCLAD'], relics=[], potions=['POTION.B', 'POTION.FIRE_POTION'])
    assert [verb for verb, _ in rows_of(shop, stock, expected)[0]] == ['DISCARD_POTION', 'BUY_ITEM', 'LEAVE_ROOM']

    # Nothing bought: leaving is the one decision.
    assert rows_of(Shop(30), record(), dict(gold=30, deck=['CARD.BASH', 'CARD.STRIKE_IRONCLAD'], relics=[], potions=[])) == (
        [('LEAVE_ROOM', True)], 1)


def test_a_shop_no_order_explains_is_given_up():
    stock = record(bought_cards=[dict(id='CARD.ANGER')], bought_relics=['RELIC.MEMBERSHIP_CARD'])
    wrong = dict(gold=1, deck=['CARD.ANGER', 'CARD.BASH', 'CARD.STRIKE_IRONCLAD'], relics=['RELIC.MEMBERSHIP_CARD'], potions=[])
    shop = Shop(250)
    assert rows_of(shop, stock, wrong) == (None, 2)
    # A purchase the gold never covers is refused once, not once for every order after it.
    shop = Shop(40)
    assert rows_of(shop, stock, wrong) == (None, 2) and shop.entries == 2
    assert rows_of(Shop(250), stock, wrong, budget=1) == (None, 1)
    # The removed card must be one the selection offers.
    with_removal = record(removed=[dict(id='CARD.CLASH')])
    assert rows_of(Shop(250), with_removal, wrong)[0] is None


def test_a_shop_with_many_purchases_is_searched_within_the_budget():
    # Thirteen recorded actions have billions of orders. They are never listed: each
    # execution either ends an order or rules out everything that begins as it did.
    many = record(bought_cards=[dict(id=f'CARD.C{i}') for i in range(9)],
                  bought_relics=['RELIC.MEMBERSHIP_CARD', 'RELIC.R1', 'RELIC.R2'], removed=[dict(id='CARD.BASH')])
    assert len(purchases(many)) == 13
    wrong = dict(gold=-1, deck=[], relics=[], potions=[])

    class Poor(Shop):
        """Sells what the record names, to a player who can pay for one relic and nothing after it."""

        def enter(self):
            frame = super().enter()
            self.stock = {'RELIC.MEMBERSHIP_CARD': 100, 'RELIC.R1': 100, 'RELIC.R2': 100,
                          **{f'CARD.C{i}': 500 for i in range(9)}}
            self.removal = 500
            return self.frame()

    shop = Poor(100)
    rows, tried = rows_of(shop, many, wrong)
    # Three relics can come first; after each, nothing else is affordable. Then no order is left.
    assert rows is None and tried <= shop_sequence.ORDER_BUDGET and shop.entries == tried

    class Rich(Poor):
        def price(self, base):
            return 0

    # Every order completes and none ends as recorded: the budget ends the search.
    shop = Rich(100)
    assert rows_of(shop, many, wrong) == (None, shop_sequence.ORDER_BUDGET)
    assert shop.entries == shop_sequence.ORDER_BUDGET


def test_shops_are_planned_from_the_state_before_the_node():
    found = planned([node('ancient', 'event'), node('shop', **shop_stats(current_gold=20)),
                     node('unknown', 'shop', **shop_stats(relic_choices=[], bought_relics=[], cards_removed=[]))],
                    ['DEFEND_IRONCLAD', 'BASH', 'ANGER', 'ANGER'])
    shop = found['a01f02-shop']
    assert shop['history'] == dict(draws=0, removals=0, courier=False) and shop['ledger']['gold'] == 110
    assert (shop['expected']['gold'], shop['expected']['relics']) == (20, ['RELIC.MEMBERSHIP_CARD'])
    assert 'CARD.ANGER' not in [c['id'] for c in shop['ledger']['deck']]
    # A shop behind a question mark is a shop; a record that is not one plain shop is refused.
    assert found['a01f03-shop']['error'] == 'shop_stock_size:7/0/3'

    courier = shop_stats(relic_choices=[choice('RELIC.ANCHOR'), choice('RELIC.VAJRA'), choice('RELIC.THE_COURIER', True)],
                         bought_relics=['RELIC.THE_COURIER'])
    found = planned([node('ancient', 'event'), node('shop', **courier)], ['DEFEND_IRONCLAD', 'BASH', 'ANGER'])
    assert found['a01f02-shop']['error'] == 'shop_restocks'
    # A shop that shares its node with another room has no boundary of its own.
    two = node('unknown', 'event', **shop_stats())
    two['rooms'].append(dict(room_type='shop', turns_taken=0))
    assert planned([two], ['DEFEND_IRONCLAD', 'BASH', 'ANGER'])['a01f01-shop']['error'] == 'event_or_multiple_room_boundary'


def test_native_shop_shows_the_recorded_stock_and_takes_the_recorded_purchases():
    if os.environ.get('SPIRE_CODEX_DATA_NATIVE_TESTS') != '1':
        pytest.skip('Set SPIRE_CODEX_DATA_NATIVE_TESTS=1 with the local solver configured')
    from combat_solver_cli.client import DEFAULT_CONFIG, SolverEngine
    with SolverEngine() as engine:
        anchor.settle(engine, engine.send(dict(cmd='start_run', character='Ironclad', seed='shop-anchor-check',
                                               ascension=0, decision_protocol=True)))
        base = engine.send(dict(cmd='anchor_state'))['save']
    player = base['players'][0]
    state = dict(deck=player['deck'], relics=player['relics'], potions=[], hp=50, max_hp=player['max_hp'], gold=400)
    spot = dict(floor=2, map_point_type='shop', second_boss=False, route=['ancient', 'shop'])
    relics = ['RELIC.ANCHOR', 'RELIC.VAJRA', 'RELIC.MEMBERSHIP_CARD']

    def saved(history, gold=400):
        run = dict(seed='shop-anchor-check', ascension=0, players=[dict(deck=player['deck'], relics=player['relics'])],
                   map_point_history=[[node('ancient', 'event'), node('shop')]])
        save = anchor.build_save(base, dict(state, gold=gold), run, 1, 2, run['map_point_history'][0][1], 'shop-check')
        return shop_sequence.install(save, base, history)

    with anchor.RunProcess(DEFAULT_CONFIG) as shared:
        # The rarities of the character cards are the anchor's; the cards are the game's draw.
        rarities = ['CARD.ANGER', 'CARD.BLUDGEON', 'CARD.SHRUG_IT_OFF', 'CARD.SHRUG_IT_OFF', 'CARD.INFLAME']
        first = dict(draws=0, removals=0, courier=False)
        _, frame = shared.enter(saved(first), dict(spot, room=dict(type='shop', cards=rarities, relics=relics)))
        shown = shop_sequence.shown(frame)
        content, prices = [name for name, _ in shown], dict(shown)
        info = anchor.content_info(shared, content[:13] + rarities)
        assert [info[c]['rarity'] for c in content[:5]] == [info[c]['rarity'] for c in rarities]
        assert [info[c]['card_type'] for c in content[:5]] == list(shop_sequence.CARD_SLOTS)
        assert content[7:10] == relics and content[13] == 'MerchantCardRemovalEntry'
        # The stock is drawn again as often as the shop is entered: same cards, same prices.
        _, again = shared.enter(saved(first), dict(spot, room=dict(type='shop', cards=content[:5], relics=relics)))
        assert shop_sequence.shown(again) == shown
        # A later position of the stream is another shop, and a used removal service costs more.
        later = dict(draws=shop_sequence.SHOP_DRAWS, removals=1, courier=False)
        _, other = shared.enter(saved(later), dict(spot, room=dict(type='shop', cards=content[:5], relics=relics)))
        assert [n for n, _ in shop_sequence.shown(other)][:5] != content[:5]
        assert dict(shop_sequence.shown(other))['MerchantCardRemovalEntry'] > prices['MerchantCardRemovalEntry']
        # A player who has not unlocked the last character cards draws from a smaller pool.
        fewer = saved(first)
        fewer['players'][0]['unlock_state']['unlocked_epochs'].remove('IRONCLAD7_EPOCH')
        _, locked = shared.enter(fewer, dict(spot, room=dict(type='shop', cards=content[:5], relics=relics)))
        seen = [n for n, _ in shop_sequence.shown(locked)]
        assert seen[:5] != content[:5] and seen[5:13] == content[5:13]

        # A record of this shop: the member's card, then a card and the removal at half price.
        card, strike = content[0], dict(id='CARD.STRIKE_IRONCLAD', floor_added_to_deck=1)
        spent = prices['RELIC.MEMBERSHIP_CARD'] + prices[card] // 2 + prices['MerchantCardRemovalEntry'] // 2
        stats = dict(
            card_choices=[dict(card=dict(id=c), was_picked=False) for c in content[1:7]], cards_gained=[dict(id=card)],
            cards_removed=[strike], bought_relics=['RELIC.MEMBERSHIP_CARD'],
            relic_choices=[choice(relics[0]), choice(relics[1]), choice(relics[2], True)],
            potion_choices=[choice(p) for p in content[10:13]])
        deck = deepcopy(player['deck'])
        deck.remove(next(c for c in deck if c['id'] == strike['id']))
        end = dict(state, gold=400 - spent, deck=deck + [dict(id=card)],
                   relics=player['relics'] + [dict(id='RELIC.MEMBERSHIP_CARD')])
        shop = dict(spot, kind='shop', id='shop-check', stock=recorded_stock(stats), history=first,
                    expected=recorded.expected_end(end))
        result = anchor.play_shop(shared, saved(first), shop)
        assert result['outcome'] == 'matched_record' and result['stock']['locked_epochs'] == []
        assert result['stock']['prices'] == [price for _, price in shown]
        verbs = [r['label']['verb'] for r in result['records']]
        assert verbs[0] == 'BUY_ITEM' and verbs[-1] == 'LEAVE_ROOM' and 'SELECT_ONE' in verbs
        assert [r['coverage']['label'] for r in result['records']][-2:] == ['reconstructed_order', 'historical_choice']
        # Every row offers the complete legal set of its frame.
        assert all(r['label'] in r['options'] for r in result['records'])

        # The same record from a player without the last character cards: the unlock
        # state under which the game draws the recorded stock is found.
        locked_stock = [n for n, _ in shop_sequence.shown(locked)]
        stats['card_choices'] = [dict(card=dict(id=c), was_picked=False) for c in locked_stock[:7]]
        stats.update(cards_gained=[], cards_removed=[], bought_relics=[],
                     relic_choices=[choice(r) for r in relics])
        shop = dict(spot, kind='shop', id='shop-check', stock=recorded_stock(stats), history=first,
                    expected=recorded.expected_end(state))
        shared.memo.clear()
        result = anchor.play_shop(shared, saved(first), shop)
        assert result['stock']['locked_epochs'] == ['IRONCLAD7_EPOCH'] and len(result['records']) == 1
        assert 'inferred_unlock_state' in result['state_sources']
