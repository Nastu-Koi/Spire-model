"""Conservative node ledger built from a run summary alone.

Missing chronology is reported, never repaired from a solver's later state.
"""
from collections import Counter
from copy import deepcopy
import json


# Enchantments whose amount grows as the card is played. A summary shows the amount
# only at the moment it happens to mention the card, so it never identifies a copy.
GROWING = {'ENCHANTMENT.GOOPY'}


def enchantment(card):
    value = card.get('enchantment')
    return {'id': value['id']} if value and value.get('id') in GROWING else value


def signature(card, *, floor=False):
    keys = ['id', 'current_upgrade_level']
    if floor:
        keys.append('floor_added_to_deck')
    normalized = {k: card.get(k) for k in keys}
    normalized['current_upgrade_level'] = normalized.get('current_upgrade_level') or 0
    normalized['enchantment'] = enchantment(card)
    return json.dumps(normalized, sort_keys=True)


def all_nodes(run):
    global_floor = 0
    for act, nodes in enumerate(run['map_point_history'], 1):
        for floor, node in enumerate(nodes, 1):
            global_floor += 1
            yield act, floor, global_floor, node


def find_card(deck, card, *, upgrading=False, choose=None):
    """The deck card a summary entry refers to.

    A summary names an upgraded card by id alone. When the copies it could mean
    differ (one is enchanted), `choose(n)` picks among the n kinds of copy; the
    caller tries each and keeps what the final deck confirms. Without `choose`
    the entry is ambiguous.
    """
    matches = [c for c in deck if c['id'] == card['id']]
    if upgrading:
        matches = [c for c in matches if not c.get('current_upgrade_level', 0)]
    else:
        # SerializableCard omits default values. An absent upgrade/enchantment
        # in a full card snapshot means zero/none, not an unconstrained match.
        matches = [c for c in matches if c.get('current_upgrade_level', 0) == card.get('current_upgrade_level', 0)
                   and enchantment(c) == enchantment(card)]
        # Copies with the same upgrade and enchantment are the same card. Which of them
        # an earlier upgrade or removal touched was an arbitrary pick, so the recorded
        # floor narrows the match when it can and is otherwise not an error.
        if 'floor_added_to_deck' in card:
            matches = [c for c in matches if c.get('floor_added_to_deck', 0) == card['floor_added_to_deck']] or matches
    if not matches:
        raise ValueError('card_not_found:' + card['id'])
    kinds = list(dict.fromkeys(signature(c) for c in matches))
    if len(kinds) > 1:
        if choose is None:
            raise ValueError('ambiguous_card_instance:' + card['id'])
        wanted = kinds[choose(len(kinds))]
        matches = [c for c in matches if signature(c) == wanted]
    return matches[0]


def apply_node(state, node, global_floor, choose=None):
    stats = node['player_stats'][0]
    deck = state['deck']
    # Growth happens before post-battle rewards and follows the upgrade at play.
    battles = sum(r['room_type'] in ('monster', 'elite', 'boss') for r in node['rooms'])
    for card in deck:
        if card['id'] in ('CARD.GENETIC_ALGORITHM', 'CARD.THE_SCYTHE'):
            card['_growth'] = card.get('_growth', 0) + battles * (
                (3 + bool(card.get('current_upgrade_level'))) if card['id'] == 'CARD.GENETIC_ALGORITHM'
                else (5 + 2 * bool(card.get('current_upgrade_level'))))
    for card in stats.get('cards_removed', []):
        deck.remove(find_card(deck, card))
    def transform(change):
        deck.remove(find_card(deck, change['original_card']))
        card = deepcopy(change['final_card']); card.setdefault('floor_added_to_deck', global_floor)
        deck.append(card)
    # A card can be gained and transformed in the same node (an event curse turned
    # into another): such a transform waits until the gains are in the deck.
    gained_here = {c['id'] for c in stats.get('cards_gained', [])}
    late = []
    for change in stats.get('cards_transformed', []):
        try:
            transform(change)
        except ValueError:
            if change['original_card']['id'] not in gained_here:
                raise
            late.append(change)
    for card in stats.get('cards_gained', []):
        card = deepcopy(card); card.setdefault('floor_added_to_deck', global_floor)
        if card['id'] in ('CARD.GENETIC_ALGORITHM', 'CARD.THE_SCYTHE'):
            name = 'IncreasedBlock' if card['id'] == 'CARD.GENETIC_ALGORITHM' else 'IncreasedDamage'
            card['_growth'] = next((p['value'] for p in card.get('props', {}).get('ints', []) if p['name'] == name), 0)
        deck.append(card)
    for change in late:
        transform(change)
    for cid in stats.get('upgraded_cards', []):
        card = find_card(deck, dict(id=cid), upgrading=True, choose=choose)
        card['current_upgrade_level'] = card.get('current_upgrade_level', 0) + 1
    for cid in stats.get('downgraded_cards', []):
        choices = [c for c in deck if c['id'] == cid and c.get('current_upgrade_level', 0) > 0]
        if not choices or len({signature(c) for c in choices}) > 1:
            raise ValueError('ambiguous_downgrade:' + cid)
        choices[0]['current_upgrade_level'] -= 1
    for change in stats.get('cards_enchanted', []):
        snapshot = change['card']; prior = {k: v for k, v in snapshot.items() if k != 'enchantment'}
        card = find_card(deck, prior)
        if not snapshot.get('enchantment'):
            raise ValueError('missing_enchantment_snapshot')
        card['enchantment'] = deepcopy(snapshot['enchantment'])

    # Potion choices already include shop purchases. Do not count bought twice.
    acquired = Counter(x['choice'] for x in stats.get('potion_choices', []) if x.get('was_picked'))
    bought = Counter(stats.get('bought_potions', []))
    acquired |= bought
    inventory = Counter(state['potions']) + acquired
    consumed = Counter(stats.get('potion_used', [])) + Counter(stats.get('potion_discarded', []))
    if consumed - inventory:
        raise ValueError('unresolved_potion_generation_or_order:' + ','.join((consumed - inventory).elements()))
    state['potions'] = list((inventory - consumed).elements())

    gained = list(dict.fromkeys([x['choice'] for x in stats.get('relic_choices', []) if x.get('was_picked')]
                              + stats.get('bought_relics', [])))
    for cid in gained:
        if cid not in [r['id'] for r in state['relics']]:
            state['relics'].append(dict(id=cid, floor_added_to_deck=global_floor))
    for cid in stats.get('relics_removed', []):
        state['relics'] = [r for r in state['relics'] if r['id'] != cid]
    state.update(hp=stats['current_hp'], max_hp=stats['max_hp'], gold=stats['current_gold'])
