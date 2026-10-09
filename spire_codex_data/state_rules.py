"""v0.111.0 persistent-state rules for summary anchors, separate from source integrity.

Rules run BEFORE native entry hooks. Final mutable props are never copied back.
Unknown non-counter mechanics fail locally instead of installing zero silently.
"""
from copy import deepcopy
import hashlib
import random
from .ledger import all_nodes

VERSION = 'anchor-state-v1'
COMBAT = {'monster', 'elite', 'boss'}
SAMPLED = {'PEN_NIB': ('AttacksPlayed', 10), 'NUNCHAKU': ('AttacksPlayed', 10),
           'TUNING_FORK': ('SkillsPlayed', 10), 'JOSS_PAPER': ('CardsExhausted', 5),
           'IRON_CLUB': ('CardsPlayed', 4), 'GALACTIC_DUST': ('StarsSpent', 10)}
TURNS = {'HAPPY_FLOWER': 3, 'FAKE_HAPPY_FLOWER': 5, 'PENDULUM': 3, 'POLLINOUS_CORE': 4}
STATIC = {'SEA_GLASS', 'DUSTY_TOME', 'TOUCH_OF_OROBAS', 'ARCHAIC_TOOTH', 'BYRDPIP', 'PAELS_LEGION'}
# These require map/lifecycle evidence that the standalone battle initializer lacks.
UNSUPPORTED = {'FUR_COAT', 'GOLDEN_COMPASS', 'PAELS_TOOTH', 'TOY_BOX'}
BOOK_CARDS, BOOK_HEAL = 5, 20  # Book of Five Rings: every fifth card added heals
SAVED_RELICS = set(SAMPLED) | set(TURNS) | STATIC | UNSUPPORTED | {
    'BONE_TEA','BOOK_OF_FIVE_RINGS','EMBER_TEA','FAKE_VENERABLE_TEA_SET','FISHING_ROD',
    'GIRYA','LASTING_CANDY','LAVA_LAMP','LAVA_ROCK','LIZARD_TAIL','MAW_BANK','PAELS_WING',
    'PUMPKIN_CANDLE','SILKEN_TRESS','SILVER_CRUCIBLE','SWORD_OF_STONE','TEA_OF_DISCOURTESY',
    'VENERABLE_TEA_SET','WINGED_BOOTS','WONGOS_MYSTERY_TICKET'}


def all_nodes_after(run, floor):
    return [(g, n) for _, _, g, n in all_nodes(run) if g > floor]


def props_values(model):
    return {p['name']: p['value'] for rows in model.get('props', {}).values() for p in rows}


def put(model, name, value):
    kind = 'bools' if isinstance(value, bool) else 'ints'
    items = model.setdefault('props', {}).setdefault(kind, [])
    items[:] = [p for p in items if p['name'] != name]
    items.append(dict(name=name, value=value))


def complete(state, run, before_global_floor, sample_key, combat=True, travel=False, pending=0):
    """Fill what a summary does not record. `combat` says whether the state enters a
    battle and `travel` whether it is a route choice: a value that decides the fight,
    or which moves are legal, is never guessed for those. `pending` is how many cards
    the anchor's own pick is still to add: the last node's record already holds them."""
    state = deepcopy(state); evidence = []; rng = random.Random(hashlib.sha256(
        (VERSION + ':' + sample_key).encode()).hexdigest())
    final_relics = {r['id']: r for r in run['players'][0]['relics']}
    past = [(g,n) for _,_,g,n in all_nodes(run) if g < before_global_floor]
    joint = {}
    for relic in state['relics']:
        name = relic['id'].split('.')[-1]; obtained = relic.get('floor_added_to_deck', 1)
        history = [n for g,n in past if g > obtained]
        rooms = [r for n in history for r in n['rooms']]
        fights = [r for r in rooms if r['room_type'] in COMBAT]
        final = final_relics.get(relic['id'], {})
        if name in UNSUPPORTED or props_values(final).get('IsWax'):
            raise ValueError('unsupported_relic_lifecycle:' + name)
        values = {}; source = 'derived'
        if name in STATIC:
            # These model choices/cosmetics are assigned once at acquisition.
            if not final or final.get('floor_added_to_deck', 1) != obtained:
                raise ValueError('missing_immutable_relic_anchor:' + name)
            relic['props'] = deepcopy(final.get('props', {})); source = 'observed_immutable'
        elif name in SAMPLED:
            field, modulus = SAMPLED[name]
            if fights:
                group = ('attack' if name in {'PEN_NIB','NUNCHAKU'} else name, obtained)
                if group not in joint: joint[group] = rng.randrange(modulus)
                value = joint[group]; source = 'sampled_counter'
            else: value = 0
            values[field] = value
        elif name in TURNS:
            # TurnsTaken includes a terminal turn that can stop between hooks.
            # A shared sampled prefix preserves BeforeHandDraw -> draw modifier
            # -> AfterSideTurnStart order, across all relic acquisition dates.
            count = 0
            for g,n in past:
                if g <= obtained: continue
                for index,room in enumerate(n['rooms']):
                    if room['room_type'] not in COMBAT: continue
                    turns = room['turns_taken']
                    if not turns: continue
                    turn_rng = random.Random(hashlib.sha256(
                        f'turn-hook-prefix-v2:{sample_key}:{g}:{index}'.encode()).hexdigest())
                    stage = turn_rng.randrange(4)
                    if name == 'POLLINOUS_CORE':
                        # Unlike Pendulum, reset happens in a later hook. A
                        # cancelled terminal turn can leave a value >= 4.
                        for _ in range(turns-1):
                            count += 1
                            if count >= 4: count = 0
                        if stage >= 1: count += 1
                        if stage >= 2 and count >= 4: count = 0
                    else:
                        threshold = 3 if name in {'HAPPY_FLOWER','FAKE_HAPPY_FLOWER'} else 1
                        count += turns-1 + (stage >= threshold)
            values['TurnsSeen'] = count if name=='POLLINOUS_CORE' else count % TURNS[name]
            source = 'sampled_counter' if fights else 'derived'
        elif name in {'VENERABLE_TEA_SET','FAKE_VENERABLE_TEA_SET'}:
            relevant = [r for r in rooms if r['room_type'] in COMBAT | {'rest_site'}]
            values['GainEnergyInNextCombat'] = bool(relevant and relevant[-1]['room_type']=='rest_site')
        elif name in {'BONE_TEA','EMBER_TEA','TEA_OF_DISCOURTESY'}:
            values['CombatsLeft'] = max(0, (5 if name=='EMBER_TEA' else 1)-len(fights))
        elif name == 'FISHING_ROD': values['CombatsSeen'] = sum(r['room_type']=='monster' for r in fights)
        elif name == 'SWORD_OF_STONE': values['ElitesDefeated'] = sum(r['room_type']=='elite' for r in fights)
        elif name == 'GIRYA':
            values['TimesLifted'] = sum(n['player_stats'][0].get('rest_site_choices',[]).count('LIFT') for n in history)
            if values['TimesLifted'] > 3: raise ValueError('invalid_girya_count')
        elif name == 'PUMPKIN_CANDLE':
            count = 5
            for n in history:
                count = max(0,count-sum(r['room_type'] in COMBAT for r in n['rooms']))
                count += 5*n['player_stats'][0].get('rest_site_choices',[]).count('KINDLE')
            values['KindleCount'] = count
        elif name == 'LIZARD_TAIL':
            values['WasUsed'] = False
            if fights and (not final or props_values(final).get('WasUsed', False)):
                # The tail was spent in some fight, the summary does not say which. In a
                # battle that decides whether the player survives, so it stays unresolved.
                if combat or not final:
                    raise ValueError('unresolved_lizard_tail_use_time')
                # Outside combat it only shows as spent or not: put the use in one of the
                # fights it was held for, the same one for every anchor of the run.
                held = sum(r['room_type'] in COMBAT for g, n in all_nodes_after(run, obtained) for r in n['rooms'])
                spent = random.Random(hashlib.sha256(
                    f'{VERSION}:lizard-tail:{run.get("seed")}:{obtained}'.encode()).hexdigest()).randrange(held)
                values['WasUsed'] = len(fights) > spent
                source = 'sampled_use_time'
        elif name == 'LAVA_LAMP': values['TookDamageThisCombat'] = False  # reset on entry
        elif name == 'WONGOS_MYSTERY_TICKET':
            values.update(CombatsFinished=len(fights), GaveRelic=len(fights)>=5)
        elif name == 'MAW_BANK':
            # Any paid purchase from a merchant, a card or the removal service included.
            values['HasItemBeenBought'] = any(n['player_stats'][0].get('gold_spent', 0) > 0 and any(
                r['room_type'] == 'shop' or r.get('model_id') == 'EVENT.FAKE_MERCHANT' for r in n['rooms'])
                for n in history)
        elif name == 'BOOK_OF_FIVE_RINGS':
            acquisition = next((n for g,n in past if g == obtained), None)
            def adds(n):
                s = n['player_stats'][0]
                return len(s.get('cards_gained',[])) + len(s.get('cards_transformed',[]))
            # The summary does not order what the pickup node added around the pickup; the
            # final count does, where it fits the additions recorded since.
            same = adds(acquisition) if acquisition else 0
            after = props_values(final).get('CardsAdded', -1) - sum(adds(n) for _, n in all_nodes_after(run, obtained))
            # The pick comes after everything else its node added, the book included.
            own = pending if obtained >= before_global_floor - 1 else 0
            known = sum(adds(n) for n in history) - (pending - own)
            fits = 0 <= after <= same
            offsets = [max(0, after - own)] if fits else list(range(max(0, same - own) + 1))
            # A final count the record does not add up to leaves the count open at every node.
            broken = bool(final) and not fits
            if broken or len(offsets) > 1: source = 'sampled_counter'
            values['CardsAdded'] = known + (rng.choice(offsets) if len(offsets) > 1 else offsets[0])
            # A pick that completes a set of five heals, and the node-end HP holds that heal.
            heals = {(known + o) % BOOK_CARDS + pending >= BOOK_CARDS for o in offsets}
            if pending and (broken or True in heals):
                if broken or False in heals or not BOOK_HEAL < state['hp'] < state['max_hp']:
                    raise ValueError('unresolved_pre_pick_hp')
                state['hp'] -= BOOK_HEAL
                evidence.append(dict(id='player', source='derived_pre_pick_hp', values=dict(hp=state['hp'])))
        elif name in {'LASTING_CANDY', 'SILKEN_TRESS', 'SILVER_CRUCIBLE'}:
            # These count the card rewards of fights. The rule reproduces the final value
            # of every run it was checked on.
            rewards = sum(1 for n in history if n['player_stats'][0].get('card_choices')
                          and any(r['room_type'] in COMBAT for r in n['rooms']))
            if name == 'LASTING_CANDY':
                values['CombatRewardsSeen'] = rewards
            elif name == 'SILKEN_TRESS':
                values['IsUsed'] = rewards > 0
            else:
                values.update(TimesUsed=min(3, rewards),
                              TreasureRoomsEntered=sum(r['room_type'] == 'treasure' for r in rooms))
        elif name == 'WINGED_BOOTS':
            # Each free move spends a charge; the summary does not say on which floors.
            if travel and props_values(final).get('TimesUsed', 0):
                raise ValueError('unresolved_winged_boots_use_time')
            source = 'noncombat_only_default'
        elif name in {'BOOK_OF_FIVE_RINGS','LAVA_ROCK','PAELS_WING'}:
            # Only future deck/reward/map operations use these fields. They have
            # no combat-entry/play/death/heal hooks. Do not claim historical values.
            # Book healing on permanent mid-combat deck gain is not covered.
            source = 'noncombat_only_default'
        elif name in SAVED_RELICS:
            raise ValueError('unhandled_saved_relic:' + name)
        if name not in STATIC:
            relic['props'] = {}
            for field,value in values.items(): put(relic,field,value)
        if name in SAVED_RELICS:
            evidence.append(dict(id=relic['id'],source=source,values=values,obtained_floor=obtained))
    for card in state['deck']:
        name = card['id'].split('.')[-1]
        if name == 'DOWSING':
            raise ValueError('unsupported_persistent_card:' + name)
        if name == 'SPOILS_MAP':
            # AfterCreated fixes this to act index 1. Its map quest hooks are
            # irrelevant inside the independently selected combat room.
            put(card,'SpoilsActIndex',1)
        if name == 'GUILTY':
            acquired = card.get('floor_added_to_deck',1)
            count = sum(r['room_type'] in COMBAT for g,n in past if g>acquired for r in n['rooms'])
            if count>=5:raise ValueError('guilty_removal_missing')
            put(card,'CombatsSeen',count)
        enchantment = card.get('enchantment', {}).get('id')
        if enchantment == 'ENCHANTMENT.GOOPY':
            raise ValueError('unresolved_growing_enchantment:GOOPY')
        if name in {'GENETIC_ALGORITHM','THE_SCYTHE'}:
            field = 'IncreasedBlock' if name=='GENETIC_ALGORITHM' else 'IncreasedDamage'
            current = 'CurrentBlock' if name=='GENETIC_ALGORITHM' else 'CurrentDamage'
            growth = card.pop('_growth',0)
            put(card,field,growth);put(card,current,(1 if name=='GENETIC_ALGORITHM' else 13)+growth)
            evidence.append(dict(id=card['id'],source='estimated_growth',value=growth))
        if name=='MAD_SCIENCE' and not card.get('props'):
            raise ValueError('missing_mad_science_properties')
    return state, evidence
