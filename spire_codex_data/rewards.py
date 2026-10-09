"""Only reward choices whose complete offer and chronology the summary settles."""
from collections import Counter
from copy import deepcopy

from .recorded import card_matches


FIELDS = ('card_choices', 'relic_choices', 'potion_choices', 'potion_used', 'potion_discarded')
UNRECOVERED = {'treasure_offer_unknown', 'treasure_mixed_rewards', 'treasure_gold_unresolved',
              'treasure_observation_ambiguous', 'treasure_potion_timing_unknown',
              'reward_order_unknown', 'reward_offer_unrecorded', 'reward_offer_differs_from_record',
              'reward_potion_timing_unknown', 'recorded_reward_not_legal',
              'covered_by_unclaimed_reward_group', 'gold_reward_multiple_sources'}

# Native 0.111 gold sources besides the base combat reward. Some are deliberately
# conservative (an inactive Maw Bank is also rejected); the summary is not a
# timestamped account of which gold source paid or which reward was left behind.
GOLD_RELICS = {'RELIC.MAW_BANK', 'RELIC.AMETHYST_AUBERGINE', 'RELIC.LUCKY_FYSH'}
GOLD_PICKUPS = {'RELIC.OLD_COIN', 'RELIC.GOLDEN_PEARL', 'RELIC.CURSED_PEARL', 'RELIC.SIGNET_RING'}
GOLD_CARDS = {'CARD.HAND_OF_GREED', 'CARD.ROYALTIES', 'CARD.SPOILS_MAP'}


def settled_gold(node, before, after):
    """Prove the combat's only gold source was claimed, without splitting a total."""
    stats = node['player_stats'][0]
    if not stats.get('gold_gained'):
        raise ValueError('gold_reward_unsettled')
    if before is None:
        raise ValueError('gold_reward_multiple_sources')
    relics = {r['id'] for state in (before, after) for r in state['relics']}
    gained = {x['choice'] for x in stats.get('relic_choices', []) if x.get('was_picked')}
    cards = {c['id'] for state in (before, after) for c in state['deck']}
    if (relics & GOLD_RELICS or gained & GOLD_PICKUPS or cards & GOLD_CARDS
            or stats.get('gold_stolen') or stats.get('stolen_loot')
            or any(r.get('model_id') == 'ENCOUNTER.GREMLIN_MERC_NORMAL' for r in node['rooms'])):
        raise ValueError('gold_reward_multiple_sources')


def evidence(stats):
    return {key: deepcopy(stats.get(key, [])) for key in FIELDS}


def treasure(stats):
    """A single-player chest's one recorded relic. No record is not an empty offer.

    Potion use/discard has no timestamp within the node. A gain from a chest
    followed by another source also leaves the offered relic ambiguous.
    """
    relics = stats.get('relic_choices', [])
    if len(relics) != 1:
        raise ValueError('treasure_offer_unknown')
    if stats.get('card_choices') or stats.get('potion_choices') or stats.get('bought_relics'):
        raise ValueError('treasure_mixed_rewards')
    if stats.get('potion_used') or stats.get('potion_discarded'):
        raise ValueError('treasure_potion_timing_unknown')
    return dict(type='treasure', relic=relics[0]['choice']), bool(relics[0].get('was_picked'))


class RecordedReward:
    """One native event reward screen, fully checked before choosing anything.

    The first supported subset is a single reward (a card offer counts as one).
    A summary does not separate several screens or order different reward types;
    never infer that order from a final state or from the list of legal actions.
    """

    def __init__(self, stats):
        self.record = evidence(stats)
        self.used = self.taken = self.left = False

    @staticmethod
    def only(candidates, verb, ref=None):
        choices = [c for c in candidates if c['verb'] == verb
                   and (ref is None or ref in c.get('source_refs', []))]
        if len(choices) != 1:
            raise ValueError('recorded_reward_not_legal')
        return choices[0]

    def choose(self, frame):
        candidates = frame['legal']['candidates']
        entities = frame['public']['entities']
        offered = [e for e in entities if e.get('entity_type') == 'reward']
        if self.left:
            raise ValueError('reward_order_unknown')
        if self.used:
            if not self.taken or offered:
                raise ValueError('reward_order_unknown')
            self.left = True
            return self.only(candidates, 'LEAVE_REWARDS')
        if len(offered) != 1:
            raise ValueError('reward_order_unknown')
        if self.record['potion_used'] or self.record['potion_discarded']:
            raise ValueError('reward_potion_timing_unknown')
        reward = offered[0]
        content = reward['content_id']
        category = ('potion_choices' if content.startswith('POTION.') else
                    'relic_choices' if content.startswith('RELIC.') else
                    'card_choices' if content == 'CardReward' else None)
        if category is None:
            raise ValueError('reward_offer_unrecorded')
        if any(self.record[key] for key in FIELDS[:3] if key != category):
            raise ValueError('reward_order_unknown')
        choices = self.record[category]
        picked = [c for c in choices if c.get('was_picked')]
        if category == 'card_choices':
            refs = {e['ref']: e for e in entities if 'ref' in e}
            cards = [refs[c['source_refs'][0]] for c in candidates if c['verb'] == 'TAKE_CARD_REWARD']
            key = lambda e: (e.get('content_id'), e.get('upgrade_level') or 0, e.get('enchantment'))
            expected = [(c['card']['id'], c['card'].get('current_upgrade_level', 0),
                         (c['card'].get('enchantment') or {}).get('id')) for c in choices]
            if len(choices) not in (3, 4) or Counter(map(key, cards)) != Counter(expected):
                raise ValueError('reward_offer_differs_from_record')
            if len(picked) > 1:
                raise ValueError('reward_order_unknown')
            if picked:
                matching = [c for c in candidates if c['verb'] == 'TAKE_CARD_REWARD'
                            and card_matches(refs[c['source_refs'][0]], picked[0]['card'])]
                selected = self.only(matching, 'TAKE_CARD_REWARD')
            elif any(c['verb'] == 'CHOOSE_REWARD_ALTERNATIVE' for c in candidates):
                raise ValueError('skip_or_alternative_unknown')
        else:
            if len(choices) != 1 or choices[0]['choice'] != content:
                raise ValueError('reward_offer_differs_from_record')
            if picked:
                selected = self.only(candidates, 'TAKE_REWARD', reward['ref'])
        self.used = True
        if picked:
            self.taken = True
            return selected
        self.left = True
        return self.only(candidates, 'LEAVE_REWARDS')
