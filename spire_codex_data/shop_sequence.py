"""Shop anchors: the merchant's stock drawn again from the run's own shop stream,
and the recorded purchases in an order the game accepts.

A summary names what a shop held (what was bought, and what was left when the
player went on) but no price and no order of purchase. Prices are not lost: the
game prices every item from the player's shop stream, a stream seeded by the run
and consumed by merchants alone, a fixed number of draws at a time. Installed at
the position it had on entry, it prices the stock again. The game drew the
rarity of each character card and the three relics from other sources (the
reward stream, the relic grab bag), so the anchor supplies those from the
record; which card of that rarity, which potions and what everything costs is
then drawn by the game. A shop is used only if what the game shows equals the
record, slot by slot.

Which cards and potions the game can draw depends on what the player had
unlocked, which no summary states. The pools are gated by a handful of epochs,
so the unlock state is found the same way: the one under which the game draws
the recorded stock. A wrong state does not reproduce seven cards and three
potions by chance.

The purchases are executed in an order found by trying: every step must be a
legal action of the frame it is taken in, and the state after the last one must
be the node's recorded end. That order is a legal reconstruction, not the
order the player bought in; the samples say so.
"""
from collections import Counter
from copy import deepcopy
from itertools import combinations, permutations
import json

from .ledger import all_nodes
from .recorded import Illegal, changes, recorded_selection

MASK = (1 << 64) - 1
# Shop-stream draws: a merchant's stock on entry, the fake merchant's six relic
# prices, and what the courier draws to restock a card, a relic or a potion.
SHOP_DRAWS, FAKE_MERCHANT_DRAWS = 28, 6
RESTOCK_DRAWS = dict(card=2, relic=1, potion=3)
FAKE_MERCHANT, COURIER = 'EVENT.FAKE_MERCHANT', 'RELIC.THE_COURIER'
CARD_SLOTS = ('Attack', 'Attack', 'Skill', 'Skill', 'Power')
COLORLESS_SLOTS = ('Uncommon', 'Rare')
# The slots of a shop frame that each part of the stock fills, and the epochs
# that gate the pool the part is drawn from ({} is the character).
PARTS = dict(cards=(slice(0, 5), ('{}2_EPOCH', '{}5_EPOCH', '{}7_EPOCH')),
             colorless=(slice(5, 7), tuple(f'COLORLESS{n}_EPOCH' for n in range(1, 6))),
             potions=(slice(10, 13), ('{}4_EPOCH', 'POTION1_EPOCH', 'POTION2_EPOCH')))
# Complete executions of one shop before it is given up.
ORDER_BUDGET = 24
# Shops the record or the game's own draws do not settle, as opposed to failures of execution.
UNRECOVERED = {'shop_stock_not_reproduced', 'shop_no_order_matches_record', 'shop_relic_slots_ambiguous',
               'choice_not_recorded', 'shop_entry_opens_another_choice', 'shop_restocks',
               'shop_mixed_card_sources', 'shop_gains_not_purchases', 'shop_stock_size', 'shop_several_removals',
               'shop_cards_do_not_fill_the_slots', 'shop_relics_do_not_fill_the_slots', 'shop_stock_unnamed'}


def advance(rng, draws):
    """The serialized stream `draws` values later: every draw is one step of xoshiro256**."""
    s0, s1, s2, s3 = (rng[f's{i}'] for i in range(4))
    for _ in range(draws):
        shifted = (s1 << 17) & MASK
        s2 ^= s0
        s3 ^= s1
        s1 ^= s2
        s0 ^= s3
        s2 ^= shifted
        s3 = ((s3 << 45) | (s3 >> 19)) & MASK
    return dict(counter=rng['counter'] + draws, s0=s0, s1=s1, s2=s2, s3=s3)


def shop_history(run, before):
    """What earlier merchants did to the state a shop is entered with: the draws they
    took from the shop stream, and how often the removal service was bought."""
    draws, removals, owned = 0, 0, set()
    for _, _, global_floor, node in all_nodes(run):
        if global_floor >= before:
            break
        stats = node['player_stats'][0]
        for room in node['rooms']:
            if room['room_type'] == 'shop':
                draws += SHOP_DRAWS
                if COURIER in owned:
                    draws += (RESTOCK_DRAWS['card'] * len(stats.get('cards_gained', []))
                              + RESTOCK_DRAWS['relic'] * len(stats.get('bought_relics', []))
                              + RESTOCK_DRAWS['potion'] * len(stats.get('bought_potions', [])))
                removals += bool(stats.get('cards_removed'))
            if room.get('model_id') == FAKE_MERCHANT:
                draws += FAKE_MERCHANT_DRAWS
        owned.update(x['choice'] for x in stats.get('relic_choices', []) if x.get('was_picked'))
        owned.update(stats.get('bought_relics', []))
    return dict(draws=draws, removals=removals, courier=COURIER in owned)


def recorded_stock(stats):
    """The stock a summary records for a shop: what was left on leaving, in slot
    order, and what was bought. Raises where the record is not one plain shop."""
    cards = stats.get('card_choices', [])
    if any(c.get('was_picked') for c in cards):
        # A picked card comes from a reward a purchase opened, mixed into the same list.
        raise ValueError('shop_mixed_card_sources')
    relics, potions = stats.get('relic_choices', []), stats.get('potion_choices', [])
    stock = dict(
        left_cards=[c['card'] for c in cards], bought_cards=deepcopy(stats.get('cards_gained', [])),
        left_relics=[x['choice'] for x in relics if not x.get('was_picked')],
        bought_relics=list(stats.get('bought_relics', [])),
        left_potions=[x['choice'] for x in potions if not x.get('was_picked')],
        bought_potions=list(stats.get('bought_potions', [])),
        used_potions=list(stats.get('potion_used', [])), discarded_potions=list(stats.get('potion_discarded', [])),
        # Cards the removal service or a purchase asked the player to pick.
        **changes(stats))
    if sorted(x['choice'] for x in relics if x.get('was_picked')) != sorted(stock['bought_relics']) \
            or sorted(x['choice'] for x in potions if x.get('was_picked')) != sorted(stock['bought_potions']):
        raise ValueError('shop_gains_not_purchases')
    sizes = (len(stock['left_cards']) + len(stock['bought_cards']), len(stock['left_relics']) + len(stock['bought_relics']),
             len(stock['left_potions']) + len(stock['bought_potions']))
    if sizes != (len(CARD_SLOTS) + len(COLORLESS_SLOTS), 3, 3):
        # Restocked entries, or gains that were not purchases.
        raise ValueError('shop_stock_size:' + '/'.join(map(str, sizes)))
    if len(stock['removed']) > 1:
        raise ValueError('shop_several_removals')
    if any(name.startswith('NONE.') for name in stock['left_relics'] + stock['left_potions']
           + stock['bought_relics'] + stock['bought_potions']):
        # An emptied slot the game wrote down without a content ID.
        raise ValueError('shop_stock_unnamed')
    return stock


def seatings(left, bought, slots, kind):
    """Every way the recorded items fill typed slots: what was left keeps its recorded
    order, what was bought takes the remaining slots."""
    found = []
    for places in combinations(range(len(slots)), len(left)):
        if any(kind(item) != slots[place] for item, place in zip(left, places)):
            continue
        free = [i for i in range(len(slots)) if i not in places]
        for order in sorted(set(permutations(bought))):
            if len(order) == len(free) and all(kind(item) == slots[i] for item, i in zip(order, free)):
                seats = [None] * len(slots)
                for item, place in zip(list(left) + list(order), list(places) + free):
                    seats[place] = item
                if seats not in found:
                    found.append(seats)
    return found


def stock_candidates(stock, info):
    """Slot assignments of the recorded stock that the game's shop layout allows:
    (character cards, colorless cards, relics, sets of potions by slot)."""
    ids = lambda cards: [c['id'] for c in cards]
    colored = lambda cards: [c for c in ids(cards) if not info[c]['colorless']]
    colorless = lambda cards: [c for c in ids(cards) if info[c]['colorless']]
    cards = seatings(colored(stock['left_cards']), colored(stock['bought_cards']), CARD_SLOTS,
                     lambda c: info[c]['card_type'])
    spare = seatings(colorless(stock['left_cards']), colorless(stock['bought_cards']), COLORLESS_SLOTS,
                     lambda c: info[c]['rarity'])
    if not cards or len(spare) != 1:
        raise ValueError('shop_cards_do_not_fill_the_slots')
    # Two relics of a rolled rarity, then the shop's own.
    rarity = lambda r: 'Shop' if info[r]['rarity'] == 'Shop' else 'Rolled'
    relics = seatings(stock['left_relics'], stock['bought_relics'], ('Rolled', 'Rolled', 'Shop'), rarity)
    if not relics:
        raise ValueError('shop_relics_do_not_fill_the_slots')
    potions = seatings(stock['left_potions'], stock['bought_potions'], ('Potion',) * 3, lambda p: 'Potion')
    return cards, spare[0], relics, potions


def shown(frame):
    """Content and price of every merchant entry of a shop frame, by slot."""
    if frame['boundary'] != 'decision' or frame['public']['phase'] != 'shop':
        raise ValueError('shop_entry_opens_another_choice')
    items = sorted((e for e in frame['public']['entities'] if e.get('entity_type') == 'shop_item'),
                   key=lambda e: int(e['ref'].split(':')[1]))
    return [(e['content_id'], e['price']) for e in items]


def locked_sets(epochs):
    """Every set of these epochs a player may not have unlocked, fewest first."""
    return [set(locked) for n in range(len(epochs) + 1) for locked in combinations(epochs, n)]


def purchases(stock):
    """What the record says was done in the shop, as actions; the order among them
    is not recorded. A potion drunk or thrown away is one of them: it frees a slot."""
    return ([('card', c['id']) for c in stock['bought_cards']] + [('relic', r) for r in stock['bought_relics']]
            + [('potion', p) for p in stock['bought_potions']] + [('removal', None)] * len(stock['removed'])
            + [('use', p) for p in stock['used_potions']] + [('discard', p) for p in stock['discarded_potions']])


def ranked(actions):
    """The actions in the order tried first: what changes later prices or what can be
    bought comes first; a removal comes last, so that it can take a card bought here."""
    rank = dict(discard=0, use=0, relic=1, potion=2, card=3, removal=4)
    return tuple(sorted(actions, key=lambda a: (rank[a[0]], str(a[1]))))


def next_order(actions, refused):
    """The first distinct order of `actions`, as ranked, that begins with nothing in
    `refused`; None when no order is left. A beginning under which nothing is left
    joins `refused`.

    The orders are never listed: a shop with a dozen purchases has hundreds of
    millions of them, and a refused beginning rules out all that follow it at once.
    """
    def descend(prefix, rest):
        if not rest:
            return prefix
        seen = set()
        for index, action in enumerate(rest):
            head = prefix + (action,)
            if action in seen or head in refused:
                continue
            seen.add(action)
            found = descend(head, rest[:index] + rest[index + 1:])
            if found is not None:
                return found
        refused.add(prefix)
        return None
    return None if () in refused else descend((), tuple(actions))


def execute(frame, step, action, pending, row):
    """Take one recorded action from `frame`, and answer what it asks for from the
    record; returns the frame after it and its rows."""
    kind, content = action
    entities = {e.get('ref'): e for e in frame['public']['entities']}
    if kind in ('use', 'discard'):
        verb = 'USE_POTION' if kind == 'use' else 'DISCARD_POTION'
        buys = [c for c in frame['legal']['candidates'] if c['verb'] == verb
                and entities[c['source_refs'][0]].get('content_id') == content]
        # Equal potions are one choice; a potion that asks for a target is not recorded.
        if not buys or len({json.dumps(c.get('target_refs', [])) for c in buys}) != 1:
            raise Illegal(kind)
        buys = buys[:1]
    else:
        wanted = 'MerchantCardRemovalEntry' if kind == 'removal' else content
        buys = [c for c in frame['legal']['candidates'] if c['verb'] == 'BUY_ITEM'
                and entities[c['source_refs'][0]]['content_id'] == wanted]
        if len(buys) != 1:
            raise Illegal(kind)
    rows = [row(frame, buys[0])]
    frame = step(frame, buys[0])
    while frame is not None and frame['boundary'] == 'decision' and frame['public']['phase'] != 'shop':
        candidate = recorded_selection(frame, pending)
        rows.append(row(frame, candidate))
        frame = step(frame, candidate)
        if len(rows) > 24:
            raise ValueError('shop_selection_did_not_finish')
    if frame is None or frame['boundary'] != 'decision':
        raise ValueError('shop_purchase_left_the_shop')
    return frame, rows


def search(enter, step, state, stock, expected, row, budget=ORDER_BUDGET):
    """The first order of the recorded purchases that the game accepts step by step and
    that ends in the recorded state: (rows, executions tried). None when no order
    within the budget does."""
    actions, refused, tried = ranked(purchases(stock)), set(), 0
    while tried < budget:
        order = next_order(actions, refused)
        if order is None:
            break
        tried += 1
        frame, rows = enter(), []
        pending = {name: deepcopy(stock[name]) for name in ('removed', 'enchanted', 'upgraded', 'transformed')}
        try:
            for index, action in enumerate(order):
                frame, taken = execute(frame, step, action, pending, row)
                rows += taken
        except Illegal:
            refused.add(order[:index + 1])
            continue
        leave = [c for c in frame['legal']['candidates'] if c['verb'] == 'LEAVE_ROOM']
        if len(leave) != 1:
            raise ValueError('shop_cannot_be_left')
        if state() == expected:
            return rows + [row(frame, leave[0], leaving=True)], tried
        # A complete order that ends elsewhere is not tried again.
        refused.add(order)
    return None, tried


def install(save, template, history):
    """Put the shop stream and the removal count where earlier merchants left them."""
    player = save['players'][0]
    player['rng']['rngs']['shops'] = advance(template['players'][0]['rng']['rngs']['shops'], history['draws'])
    if history['removals']:
        player['extra_fields'] = dict(player.get('extra_fields') or {}, card_shop_removals_used=history['removals'])
    return save
