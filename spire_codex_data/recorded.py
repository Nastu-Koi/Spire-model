"""What a summary records about a node, as the engine sees it: the cards a
selection is answered with, and the state a node ends in."""
from copy import deepcopy
from functools import lru_cache
import json
from pathlib import Path

from .ledger import signature


class Illegal(Exception):
    """A recorded action that is no legal action of the frame it was tried in."""


def changes(stats):
    """The cards a node's record names for each card operation, as they were before it."""
    return dict(
        removed=deepcopy(stats.get('cards_removed', [])),
        enchanted=[{k: v for k, v in x['card'].items() if k != 'enchantment'} for x in stats.get('cards_enchanted', [])],
        upgraded=[dict(id=x, current_upgrade_level=0, any_enchantment=True) for x in stats.get('upgraded_cards', [])],
        transformed=[deepcopy(x['original_card']) for x in stats.get('cards_transformed', [])])


def card_matches(entity, card):
    return (entity.get('content_id') == card['id']
            and (entity.get('upgrade_level') or 0) == card.get('current_upgrade_level', 0)
            and (card.get('any_enchantment') or entity.get('enchantment') == (card.get('enchantment') or {}).get('id')))


# The record of each card operation a selection can ask for.
SELECTIONS = dict(remove='removed', enchant='enchanted', upgrade='upgraded', transform='transformed')


def recorded_selection(frame, pending):
    """The next step of a card selection an action opened: the cards the record names
    for that operation, in recorded order, then the end of the selection. `pending`
    holds what `changes` returned and is used up as selections take their cards."""
    candidates = frame['legal']['candidates']
    if len(candidates) == 1:
        return candidates[0]
    context = frame['public'].get('selection_context') or {}
    cards = pending.get(SELECTIONS.get(context.get('operation')))
    if frame['public']['phase'] != 'card_select' or cards is None:
        # A choice the summary does not record.
        raise ValueError('choice_not_recorded:' + str(frame['public']['phase']))
    if not context.get('selected_count'):
        take = min(context.get('max_total') or len(cards), len(cards))
        if take < (context.get('min_total') or 0):
            raise Illegal('selection')
        pending['selecting'] = [cards.pop(0) for _ in range(take)]
    if not pending.get('selecting'):
        finish = [c for c in candidates if c['verb'] == 'FINISH_SELECTION']
        if len(finish) != 1:
            raise Illegal('selection')
        return finish[0]
    card = pending['selecting'].pop(0)
    entities = {e.get('ref'): e for e in frame['public']['entities']}
    targets = [c for c in candidates if c['verb'] == 'SELECT_ONE' and card_matches(entities.get(c['source_refs'][0], {}), card)]
    # Copies alike in everything the player sees are one choice.
    if not targets or len({json.dumps({k: v for k, v in entities[c['source_refs'][0]].items() if k != 'ref'},
                                      sort_keys=True) for c in targets}) != 1:
        raise Illegal('selection')
    return targets[0]


def end_state(player):
    """What a node outside combat can change, as the game serializes it."""
    return dict(hp=player['current_hp'], max_hp=player['max_hp'], gold=player['gold'],
                deck=sorted(signature(c) for c in player['deck']),
                relics=sorted(r['id'] for r in player['relics']),
                potions=sorted(p['id'] for p in player.get('potions', [])))


def expected_end(state):
    return dict(hp=state['hp'], max_hp=state['max_hp'], gold=state['gold'],
                deck=sorted(signature(c) for c in state['deck']),
                relics=sorted(r['id'] for r in state['relics']), potions=sorted(state['potions']))


@lru_cache(maxsize=None)
def renderings():
    """Every text a localization key stands for, in the languages installed with the engine."""
    texts = {}
    for path in (Path(__file__).resolve().parents[1] / 'sts2-cli').glob('localization_*/*.json'):
        table = json.loads(path.read_text())
        if isinstance(table, dict):
            for key, text in table.items():
                if isinstance(text, str):
                    texts.setdefault(key, set()).add(text)
    return texts


def same_variables(recorded, native):
    """How the variables the game would write for an event compare with those it wrote.

    'equal'; 'text' when they differ only in text that cannot be held against the
    record (the game writes names in the player's language, the engine gives the
    key of the name: a key is settled where the record reads as one of its
    installed renderings); None when a number, a flag or a name differs.
    """
    if set(recorded) != set(native):
        return None
    result = 'equal'
    for name, value in recorded.items():
        other = native[name]
        if any(value.get(k) != other.get(k) for k in set(value) | set(other) if k != 'string_value'):
            return None
        wrote, key = value.get('string_value'), other.get('string_value')
        if wrote == key or wrote in renderings().get(key, ()):
            continue
        # A name in English that is not this key's English name is another name.
        if isinstance(wrote, str) and isinstance(key, str) and key.endswith('.title') and wrote.isascii() \
                and key in renderings():
            return None
        result = 'text'
    return result
