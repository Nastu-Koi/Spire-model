from copy import deepcopy
from spire_codex_data.ledger import apply_node, find_card
from spire_codex_data.state_rules import complete, props_values


def node(rooms=(), **extra):
    return dict(rooms=list(rooms),player_stats=[dict(current_hp=70,max_hp=80,current_gold=100,**extra)])


def test_omitted_upgrade_is_zero():
    upgraded=dict(id='CARD.STRIKE_IRONCLAD',current_upgrade_level=1)
    plain=dict(id='CARD.STRIKE_IRONCLAD')
    assert find_card([upgraded,plain],dict(id=plain['id'])) is plain


def test_growth_historical_upgrade_and_acquisition():
    state=dict(deck=[],relics=[],potions=[])
    battle=dict(room_type='monster',turns_taken=2)
    apply_node(state,node([battle],cards_gained=[dict(id='CARD.GENETIC_ALGORITHM')]),2)
    assert state['deck'][0]['_growth']==0
    apply_node(state,node([battle]),3)
    apply_node(state,node([battle],upgraded_cards=['CARD.GENETIC_ALGORITHM']),4)
    apply_node(state,node([battle]),5)
    assert state['deck'][0]['_growth']==10  # 3 + 3 + 4


def test_counters_joint_initial_and_turns():
    rel=[dict(id='RELIC.'+x,floor_added_to_deck=1) for x in
         ['PEN_NIB','NUNCHAKU','HAPPY_FLOWER','FAKE_HAPPY_FLOWER','BONE_TEA']]
    state=dict(deck=[],relics=rel,potions=[],hp=80,max_hp=80,gold=100)
    run=dict(players=[dict(relics=deepcopy(rel))],map_point_history=[[node(),node([dict(room_type='monster',turns_taken=4)])]])
    first,e=complete(state,run,2,'first')
    assert props_values(first['relics'][0])['AttacksPlayed']==0
    assert props_values(first['relics'][-1])['CombatsLeft']==1
    later,e=complete(state,run,3,'later')
    assert props_values(later['relics'][0])['AttacksPlayed']==props_values(later['relics'][1])['AttacksPlayed']
    h=props_values(later['relics'][2])['TurnsSeen']
    f=props_values(later['relics'][3])['TurnsSeen']
    assert (h,f) in {(0,3),(1,4)}  # One terminal-turn hook may not have run.
    assert props_values(later['relics'][4])['CombatsLeft']==0
    assert complete(state,run,3,'later')==(later,e)


def test_potion_purchase_not_duplicated():
    state=dict(deck=[],relics=[],potions=[])
    apply_node(state,node(potion_choices=[dict(choice='POTION.FIRE_POTION',was_picked=True)],
        bought_potions=['POTION.FIRE_POTION']),1)
    assert state['potions']==['POTION.FIRE_POTION']


def test_lizard_tail_spent_time_is_not_randomized():
    rel=dict(id='RELIC.LIZARD_TAIL',floor_added_to_deck=1)
    final=deepcopy(rel);final['props']=dict(bools=[dict(name='WasUsed',value=True)])
    run=dict(players=[dict(relics=[final])],map_point_history=[[node(),node([dict(room_type='monster',turns_taken=2)])]])
    state=dict(deck=[],relics=[rel],potions=[])
    complete(state,run,2,'first')  # Newly obtained, before any possible death.
    try:complete(state,run,3,'second')
    except ValueError as e:assert str(e)=='unresolved_lizard_tail_use_time'
    else:raise AssertionError('Used tail was silently reset')


def test_pantograph_and_entry_healing_are_not_injected_from_end_hp():
    from spire_codex_data.anchor import node_states
    t=dict(players=[dict(deck=[],relics=[],current_hp=40,max_hp=80,gold=100)])
    r=dict(map_point_history=[[node(current_dummy=0),node([dict(room_type='boss',turns_taken=4)])]],
        players=[dict(deck=[],relics=[],potions=[])])
    r['map_point_history'][0][0]['player_stats'][0]['current_hp']=30
    r['map_point_history'][0][1]['player_stats'][0]['current_hp']=65
    states,_=node_states(r,t)
    assert states[1,2]['start']['hp']==30


def test_turn_hook_order_and_pollinous_late_reset():
    rel=[dict(id='RELIC.'+x,floor_added_to_deck=1) for x in
         ['HAPPY_FLOWER','FAKE_HAPPY_FLOWER','PENDULUM','POLLINOUS_CORE']]
    run=dict(players=[dict(relics=deepcopy(rel))],map_point_history=[[node(),node([dict(room_type='monster',turns_taken=4)])]])
    state=dict(deck=[],relics=rel,potions=[])
    seen=set()
    for variant in range(50):
        restored,_=complete(state,run,3,str(variant))
        values=tuple(props_values(r)['TurnsSeen'] for r in restored['relics'])
        seen.add(values)
    # Before draw; after draw before modifier; after modifier; after side-start.
    assert seen == {(0,3,0,3),(0,3,1,4),(0,3,1,0),(1,4,1,0)}


def test_spent_lizard_tail_is_placed_once_per_run_outside_combat():
    rel=dict(id='RELIC.LIZARD_TAIL',floor_added_to_deck=1)
    final=deepcopy(rel);final['props']=dict(bools=[dict(name='WasUsed',value=True)])
    fight=[dict(room_type='monster',turns_taken=2)]
    run=dict(seed='S',players=[dict(relics=[final])],map_point_history=[[node()]+[node(fight) for _ in range(6)]])
    state=dict(deck=[],relics=[rel],potions=[])
    used=[props_values(complete(state,run,g,f'anchor-{g}',combat=False)[0]['relics'][0])['WasUsed'] for g in range(2,9)]
    # Unspent, then spent from one fight on, whatever anchor asks; never spent before any fight.
    assert used[0] is False and used[-1] is True and used==sorted(used)
    assert used==[props_values(complete(state,run,g,'other',combat=False)[0]['relics'][0])['WasUsed'] for g in range(2,9)]
    assert complete(state,run,5,'x',combat=False)[1][0]['source']=='sampled_use_time'
    try:complete(state,run,5,'x')
    except ValueError as e:assert str(e)=='unresolved_lizard_tail_use_time'
    else:raise AssertionError('A battle must not guess whether the tail is spent')


def test_reward_counting_relics_follow_the_fights_since_pickup():
    rel=[dict(id='RELIC.'+x,floor_added_to_deck=1) for x in ['LASTING_CANDY','SILKEN_TRESS','SILVER_CRUCIBLE','WINGED_BOOTS']]
    final=deepcopy(rel);final[3]['props']=dict(ints=[dict(name='TimesUsed',value=2)])
    offer=dict(card_choices=[dict(card=dict(id='CARD.A'),was_picked=False)])
    fight=[dict(room_type='monster',turns_taken=2)]
    run=dict(seed='S',players=[dict(relics=final)],map_point_history=[[node(),node(fight,**offer),
        node([dict(room_type='treasure',turns_taken=0)]),node(fight),node(fight,**offer)]])
    state=dict(deck=[],relics=rel,potions=[])
    values=lambda g,**k:[props_values(r) for r in complete(state,run,g,'k',combat=False,**k)[0]['relics']]
    assert values(2)[:3]==[dict(CombatRewardsSeen=0),dict(IsUsed=False),dict(TimesUsed=0,TreasureRoomsEntered=0)]
    # A fight without a card reward does not count; the treasure room does for the crucible.
    assert values(6)[:3]==[dict(CombatRewardsSeen=2),dict(IsUsed=True),dict(TimesUsed=2,TreasureRoomsEntered=1)]
    # Spent boot charges change which moves are legal: a route choice is not built on a guess.
    try:values(3,travel=True)
    except ValueError as e:assert str(e)=='unresolved_winged_boots_use_time'
    else:raise AssertionError('Route choice built on unknown boot charges')


def test_book_of_five_rings_counts_up_to_the_pick_and_takes_its_heal_back():
    rel=dict(id='RELIC.BOOK_OF_FIVE_RINGS',floor_added_to_deck=1)
    gain=lambda n:dict(cards_gained=[dict(id='CARD.A')]*n)
    fight=[dict(room_type='monster',turns_taken=2)]
    run=dict(seed='S',players=[dict(relics=[dict(rel,props=dict(ints=[dict(name='CardsAdded',value=5)]))])],
             map_point_history=[[node(),node(fight,**gain(4)),node(fight,**gain(1))]])
    state=dict(deck=[],relics=[rel],potions=[],hp=50,max_hp=80)
    book=lambda *a,**k:complete(state,run,*a,combat=False,**k)
    # After the node, as a route choice sees it: all five are in the deck.
    restored,evidence=book(4,'map')
    assert props_values(restored['relics'][0])==dict(CardsAdded=5) and restored['hp']==50
    # Before the fifth card is picked: four counted, and the node-end HP still holds the heal.
    restored,evidence=book(4,'reward',pending=1)
    assert props_values(restored['relics'][0])==dict(CardsAdded=4) and restored['hp']==30
    assert 'derived_pre_pick_hp' in {e['source'] for e in evidence}
    # A pick that completes no set changes nothing but the count.
    restored,_=book(3,'reward',pending=1)
    assert props_values(restored['relics'][0])==dict(CardsAdded=3) and restored['hp']==50
    # At full HP the heal may have been cut short: the HP before the pick is unknown.
    try:complete(dict(state,hp=80),run,4,'reward',combat=False,pending=1)
    except ValueError as e:assert str(e)=='unresolved_pre_pick_hp'
    else:raise AssertionError('Pre-pick HP guessed from a clamped heal')
    # Picked up on a node that also added a card: the final count says which came first.
    late=dict(rel,floor_added_to_deck=2)
    for final,counted in ((5,4),(4,3)):
        run['players'][0]['relics']=[dict(late,props=dict(ints=[dict(name='CardsAdded',value=final)]))]
        restored,evidence=complete(dict(state,relics=[late]),run,4,'reward',combat=False,pending=1)
        assert props_values(restored['relics'][0])==dict(CardsAdded=counted)
        assert evidence[-1]['source']=='derived' and restored['hp']==(30 if counted==4 else 50)
    # A final count the recorded additions cannot reach: no pick is built on it.
    run['players'][0]['relics']=[dict(rel,props=dict(ints=[dict(name='CardsAdded',value=9)]))]
    assert book(4,'map')[1][-1]['source']=='sampled_counter'
    try:book(3,'reward',pending=1)
    except ValueError as e:assert str(e)=='unresolved_pre_pick_hp'
    else:raise AssertionError('Pick built on a count that contradicts the record')


def test_ledger_matches_copies_and_growing_enchantments_by_what_the_card_is():
    from spire_codex_data.ledger import signature
    goopy=lambda n:dict(id='CARD.DEFEND_IRONCLAD',enchantment=dict(id='ENCHANTMENT.GOOPY',amount=n))
    # The amount of a growing enchantment is not part of a card's identity.
    assert signature(goopy(1))==signature(goopy(12))
    assert find_card([goopy(1)],goopy(12)) is not None
    # The recorded floor narrows a match but is not required: equal copies are interchangeable.
    a,b=dict(id='CARD.CRUELTY',floor_added_to_deck=1,current_upgrade_level=1),dict(id='CARD.CRUELTY',floor_added_to_deck=2)
    assert find_card([a,b],dict(id='CARD.CRUELTY',floor_added_to_deck=2,current_upgrade_level=1)) is a
    assert find_card([a,b],dict(id='CARD.CRUELTY',floor_added_to_deck=2)) is b
    # A curse gained and transformed in one node.
    state=dict(deck=[],relics=[],potions=[])
    apply_node(state,node(cards_gained=[dict(id='CARD.DOUBT')],cards_transformed=[dict(
        original_card=dict(id='CARD.DOUBT',floor_added_to_deck=3),final_card=dict(id='CARD.INJURY'))]),3)
    assert [c['id'] for c in state['deck']]==['CARD.INJURY']

