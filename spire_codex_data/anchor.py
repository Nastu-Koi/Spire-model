"""Independent supervised samples from run summaries (summary anchors).

Every sample starts from a node boundary rebuilt by the summary ledger and
installed in the native engine: a battle fought again by CombatSolver, or a
rest-site, card-reward, ancient, map or shop decision labelled with the recorded
human choice. Nothing carries over between anchors, and no anchor proves a
continuous winning run. The anchors of a run share native processes: one for its choice
anchors, one for the solver's battles and one that repeats those battles.

python -m spire_codex_data.anchor run --manifest data/spire-codex/sources/manifest.json --output data/anchors
python -m spire_codex_data.anchor export --manifest data/spire-codex/sources/manifest.json --output data/anchors
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from copy import deepcopy
import gzip
import hashlib
import json
import os
from pathlib import Path
import time

from combat_outcome.data import SCHEMA as OUTCOME_SCHEMA
from combat_outcome.frames import input_entities
from combat_solver_cli.client import DEFAULT_CONFIG, SolverEngine
from combat_solver_cli.search_support import player, preference, resolve, state_key
from combat_solver_cli.trajectory import native_replay_error
from model.control import controller_for
from model.protocol import CHARACTERS, action_semantics, execution_command, validate_frame
from . import recorded, rewards, shop_sequence
from .fetch import save_json
from .ledger import all_nodes, apply_node, find_card, signature
from .sources import split_group
from .state_rules import COMBAT, complete, props_values

SCHEMA = 'spire-independent-decisions-v1'
VERSION = 'summary-anchor-samples-v3'
KIND_VERSIONS = {'battle': 'summary-battle-refresh-v1'}
INITIALIZATION = 'summary-anchor-v1'
KINDS = ('battle', 'rest', 'reward', 'map', 'ancient', 'shop', 'event', 'unclaimed', 'treasure')
# A recorded offer may carry fields written by recorder mods; only these describe the card.
CARD_KEYS = ('id', 'current_upgrade_level', 'enchantment', 'props')
POTION_SLOTS = {'RELIC.POTION_BELT': 2, 'RELIC.ALCHEMICAL_COFFER': 4, 'RELIC.PHIAL_HOLSTER': 1}


def read(path):
    return json.loads(Path(path).read_text())


def settle(engine, frame):
    """Pump the engine to its next decision or terminal frame; only training frames pass."""
    deadline = time.monotonic() + 30
    while frame.get('boundary') == 'waiting':
        if time.monotonic() >= deadline:
            raise TimeoutError('native_boundary_timeout')
        time.sleep(.01)
        frame = engine.send(dict(cmd='advance_to_boundary'))
    if frame.get('type') != 'decision_frame' or frame.get('boundary') == 'error':
        raise ValueError(f'native_frame_error:{frame.get("error") or frame.get("message")}')
    validate_frame(frame)
    return frame


def in_combat(engine, frame):
    if frame['boundary'] == 'terminal':
        return False
    if frame['public']['phase'] == 'combat':
        return True
    if frame['public']['phase'] in ('card_select', 'card_reward'):
        info = engine.send(dict(cmd='solver_info'))
        if info.get('type') != 'solver_info':
            raise ValueError('solver_info_failed')
        return info['combat_in_progress']
    return False


def row(frame, candidate, actor):
    """One sample: the public observation, every legal action, and the one taken."""
    return dict(schema=SCHEMA, observation=deepcopy(frame['public']),
        options=[action_semantics(c) for c in frame['legal']['candidates']],
        label=action_semantics(candidate), actor=actor,
        coverage=dict(state='summary_anchor', options='complete_native', label='executed'),
        routing=deepcopy(frame['routing']), before_hash=state_key(frame))


def template(source, run, directory, shared):
    """The run as the game serializes it at its start, and its public encounter pools."""
    path = Path(directory) / 'templates' / (source['run_hash'] + '.json')
    if path.exists():
        return read(path)
    engine, _ = shared.begin(lambda engine: settle(engine, engine.send(dict(cmd='start_run',
        character=source['character'], seed=run['seed'], ascension=run['ascension'], acts=run['acts'],
        decision_protocol=True))))
    state = engine.send(dict(cmd='anchor_state'))
    pools = engine.send(dict(cmd='get_map')).get('encounter_pools')
    if state.get('type') != 'anchor_state' or not pools:
        raise ValueError('template_failed:' + str(state.get('message')))
    result = dict(save=state['save'], pools=pools)
    save_json(path, result)
    return result


class Undecided(Exception):
    """A ledger pass reached a card the summary leaves open, with no pick planned for it."""


def ledger_pass(run, save, plan):
    """One ledger pass that takes the planned pick at every ambiguous card.

    Returns (states, error, final state). Raises Undecided(n) at the first of n
    kinds of copy that the plan does not cover.
    """
    picks = iter(plan or [])

    def choose(n):
        pick = next(picks, None)
        if pick is None:
            raise Undecided(n)
        return pick
    p = save['players'][0]
    state = dict(deck=deepcopy(p['deck']), relics=deepcopy(p['relics']), potions=[],
                 hp=p['current_hp'], max_hp=p['max_hp'], gold=p['gold'])
    states, error = {}, None
    for act, floor, global_floor, node in all_nodes(run):
        start = None if error else deepcopy(state)
        if not error:
            try:
                apply_node(state, node, global_floor, choose if plan is not None else None)
            except ValueError as exc:
                error = f'floor_{global_floor}:{exc}'
        states[act, floor] = dict(start=start, end=None if error else deepcopy(state), error=error)
    return states, error, state


def node_states(run, save, limit=256):
    """Ledger state before and after every node; None where it is not known.

    An upgrade is recorded by card id only. When copies of that card differ, every
    choice is tried and those that reproduce the final deck are kept; a node's
    state stands only if all of them agree on it.
    """
    final = Counter(signature(c) for c in run['players'][0]['deck'])
    plans, passes = [[]], []
    while plans and len(passes) + len(plans) <= limit:
        plan = plans.pop()
        try:
            passes.append(ledger_pass(run, save, plan))
        except Undecided as open_choice:
            plans.extend(plan + [i] for i in range(open_choice.args[0]))
    matching = [p for p in passes if not p[1] and Counter(signature(c) for c in p[2]['deck']) == final]
    if plans or (len(passes) > 1 and not matching):
        # Too many combinations, or none reproduces the final deck: stop at the first open card.
        passes = matching = [ledger_pass(run, save, None)]
    elif len(passes) == 1:
        matching = passes
    states, error, state = matching[0]
    resolved = len(passes) > 1
    if len(matching) > 1:
        canon = lambda value: None if value is None else json.dumps(value, sort_keys=True)
        for key, entry in states.items():
            for side in ('start', 'end'):
                if len({canon(p[0][key][side]) for p in matching}) > 1:
                    entry[side] = None
                    entry['error'] = entry['error'] or 'ambiguous_card_instance:order_of_upgrades'
    # A final mismatch names the affected card families; it is no licence to
    # copy final upgrades or enchantments back into earlier nodes.
    suspect = set()
    if not error:
        ledger = Counter(signature(c) for c in state['deck'])
        suspect = {json.loads(x)['id'] for x in (final - ledger) + (ledger - final)}
    return states, dict(error=error, suspect_card_ids=sorted(suspect), resolved_by_final_deck=resolved)


def build_save(base, state, run, act, floor, node, key):
    save = deepcopy(base)
    p = save['players'][0]
    p.update(current_hp=state['hp'], max_hp=state['max_hp'], gold=state['gold'], deck=state['deck'],
             relics=state['relics'], potions=[dict(id=x, slot_index=i) for i, x in enumerate(state['potions'])])
    # Pickup capacity increases persist even if the relic is later removed.
    acquired = {r['id'] for r in p['relics']}
    for a, f, _, n in all_nodes(run):
        if (a, f) >= (act, floor):
            break
        stats = n['player_stats'][0]
        acquired.update(x['choice'] for x in stats.get('relic_choices', []) if x.get('was_picked'))
        acquired.update(stats.get('bought_relics', []))
    p['max_potion_slot_count'] += sum(v for k, v in POTION_SLOTS.items() if k in acquired)
    if len(p['potions']) > p['max_potion_slot_count']:
        raise ValueError('potion_capacity_mismatch')
    save['current_act_index'] = act - 1
    save['extra_fields']['started_with_neow'] = False
    save['pre_finished_room'] = None
    save['visited_map_coords'] = []
    history = deepcopy(run['map_point_history'][:act])
    history[-1] = history[-1][:floor - 1] + [dict(
        map_point_type=node['map_point_type'], player_stats=[dict(player_id=p['net_id'])],
        rooms=[dict({k: node['rooms'][0][k] for k in ('model_id',) if k in node['rooms'][0]},
                    room_type=node['rooms'][0]['room_type'], turns_taken=0)])]
    save['map_point_history'] = history
    # The map names the bosses of the act. The seed draws them for a player who has
    # unlocked everything; the summary records the ones this player met, in order.
    for rooms, met in zip((a['rooms'] for a in save['acts']), act_bosses(run)):
        slots = ['boss_id', 'second_boss_id'][:1 + (rooms.get('second_boss_id') is not None)]
        if len(met) == len(slots):
            rooms.update(zip(slots, met))
    # New, deterministic RNG streams: the human's were never recorded.
    for name, value in save['rng']['rngs'].items():
        digest = hashlib.sha256(('anchor-combat-rng-v1:' + key + ':' + name).encode()).digest()
        value.update(counter=0, **{'s' + str(i): int.from_bytes(digest[i * 8:i * 8 + 8], 'little') for i in range(4)})
    return save


def offered(choice):
    return {k: deepcopy(choice['card'][k]) for k in CARD_KEYS if k in choice['card']}


def plan(run, key, states, info, kinds):
    """Every anchor of a run with the ledger state it starts from, or why it has none."""
    suspect = set(info['suspect_card_ids'])
    bosses = Counter()
    for act, floor, global_floor, node in all_nodes(run):
        rooms, stats, s = node['rooms'], node['player_stats'][0], states[act, floor]
        second_boss = bosses[act] > 0
        bosses[act] += any(r['room_type'] == 'boss' for r in rooms)
        act_nodes = run['map_point_history'][act - 1]
        base = dict(run_hash=key, act=act, floor=floor, global_floor=global_floor,
                    map_point_type=node['map_point_type'], second_boss=second_boss,
                    route=[n['map_point_type'] for n in act_nodes])
        found = []
        for index, room in enumerate(rooms):
            if room['room_type'] in COMBAT:
                found.append(dict(base, kind='battle', id=f'{key}-a{act:02d}f{floor:02d}r{index}', state='start',
                    before=global_floor, room=dict(type='combat', encounter=room['model_id'].split('.')[-1]),
                    encounter=room['model_id'], room_type=room['room_type'],
                    source_damage=stats.get('damage_taken'), source_turns=room.get('turns_taken')))
        single = len(rooms) == 1
        if rooms[0]['room_type'] == 'rest_site' and stats.get('rest_site_choices'):
            found.append(dict(base, kind='rest', id=f'{key}-a{act:02d}f{floor:02d}-rest', state='start',
                before=global_floor, room=dict(type='rest_site'), choices=stats['rest_site_choices'],
                upgraded=stats.get('upgraded_cards', [])))
        if any(r['room_type'] in COMBAT for r in rooms) and stats.get('card_choices'):
            choices = stats['card_choices']
            found.append(dict(base, kind='reward', id=f'{key}-a{act:02d}f{floor:02d}-reward', state='end',
                before=global_floor + 1, room=dict(type='card_reward', cards=[offered(c) for c in choices]),
                picked=[offered(c) for c in choices if c.get('was_picked')], offer_size=len(choices)))
        left = dict(potions=[x['choice'] for x in stats.get('potion_choices', []) if not x.get('was_picked')],
                    relics=[x['choice'] for x in stats.get('relic_choices', []) if not x.get('was_picked')])
        if any(r['room_type'] in COMBAT for r in rooms) and (left['potions'] or left['relics']):
            # Leaving the reward screen of a fight with a potion or a relic still on it. Only
            # a reward left behind is written down as not picked, so the screen at that moment
            # is known: what was left, and the card offer if no card was taken from it.
            choices = stats.get('card_choices', [])
            found.append(dict(base, kind='unclaimed', id=f'{key}-a{act:02d}f{floor:02d}-unclaimed', state='end',
                before=global_floor + 1, offer_size=len(choices), gold_gained=stats.get('gold_gained', 0),
                room=dict(type='rewards', **left, cards=[] if any(c.get('was_picked') for c in choices)
                          else [offered(c) for c in choices])))
        offer = stats.get('ancient_choice') or []
        if node['map_point_type'] == 'ancient' and offer and str(rooms[0].get('model_id', '')).startswith('EVENT.'):
            # An ancient is the act's start point whatever route follows it.
            found.append(dict(base, kind='ancient', id=f'{key}-a{act:02d}f{floor:02d}-ancient', state='start',
                route=['ancient'], before=global_floor, chosen=[i for i, c in enumerate(offer) if c.get('was_chosen')],
                room=dict(type='ancient', event=rooms[0]['model_id'].split('.')[-1],
                          options=[c['TextKey'] for c in offer])))
        if node['map_point_type'] != 'ancient' and rooms[0]['room_type'] == 'event' \
                and str(rooms[0].get('model_id', '')).startswith('EVENT.'):
            found.append(dict(base, kind='event', id=f'{key}-a{act:02d}f{floor:02d}-event', state='start',
                before=global_floor, event=rooms[0]['model_id'], choices=stats.get('event_choices', []),
                room=dict(type='event', event=rooms[0]['model_id'].split('.')[-1])))
        if any(r['room_type'] == 'shop' for r in rooms):
            found.append(dict(base, kind='shop', id=f'{key}-a{act:02d}f{floor:02d}-shop', state='start',
                before=global_floor, room=dict(type='shop')))
        if any(r['room_type'] == 'treasure' for r in rooms):
            found.append(dict(base, kind='treasure', id=f'{key}-a{act:02d}f{floor:02d}-treasure', state='start',
                before=global_floor, room=dict(type='treasure')))
        if floor < len(act_nodes):
            # The travel decision taken after this node, towards the next recorded one.
            found.append(dict(base, kind='map', id=f'{key}-a{act:02d}f{floor:02d}-map', state='end',
                before=global_floor + 1, room=dict(type='map')))
        for anchor in found:
            if anchor['kind'] not in kinds:
                continue
            try:
                if not single and anchor['kind'] != 'map':
                    raise ValueError('event_or_multiple_room_boundary')
                state = deepcopy(s[anchor['state']])
                if state is None:
                    raise ValueError('ledger:' + str(s['error']))
                if anchor['kind'] == 'rest' and len(anchor['choices']) != 1:
                    raise ValueError('multiple_rest_choices')
                if anchor['kind'] == 'ancient' and len(anchor['chosen']) != 1:
                    raise ValueError('ancient_choice_unclear')
                if anchor['kind'] == 'unclaimed':
                    if anchor['offer_size'] not in (0, 3, 4):
                        # Several card offers: which of them were still open is not recorded.
                        raise ValueError('multiple_card_rewards')
                    rewards.settled_gold(node, s['start'], state)
                    named = anchor['room']['potions'] + anchor['room']['relics']
                    if len(set(named)) != len(named):
                        # One reward written down twice, or two alike: the record does not say.
                        raise ValueError('unclaimed_reward_repeated')
                if anchor['kind'] == 'event':
                    if not anchor['choices']:
                        # A merchant in disguise, a minigame: nothing the record calls a choice.
                        raise ValueError('event_without_recorded_choice')
                    if s['end'] is None:
                        raise ValueError('ledger:' + str(s['error']))
                    anchor['changes'] = recorded.changes(stats)
                    anchor['reward_record'] = rewards.evidence(stats)
                    anchor['expected'] = recorded.expected_end(s['end'])
                if anchor['kind'] == 'treasure':
                    anchor['room'], anchor['picked'] = rewards.treasure(stats)
                    if s['end'] is None:
                        raise ValueError('ledger:' + str(s['error']))
                    anchor['changes'] = recorded.changes(stats)
                    anchor['expected'] = recorded.expected_end(s['end'])
                if anchor['kind'] == 'shop':
                    anchor['stock'] = shop_sequence.recorded_stock(stats)
                    anchor['history'] = shop_sequence.shop_history(run, global_floor)
                    if anchor['history']['courier'] or shop_sequence.COURIER in anchor['stock']['bought_relics']:
                        # Every purchase restocks from the shop stream: what is drawn depends
                        # on the order, and the restocked cards' rarities are not recorded.
                        raise ValueError('shop_restocks')
                    if s['end'] is None:
                        raise ValueError('ledger:' + str(s['error']))
                    anchor['expected'] = recorded.expected_end(s['end'])
                if anchor['kind'] == 'rest':
                    anchor['expected'] = expected_end(s['end'], s['error'])
                    # The ledger knows which copy the smith upgraded once the final deck has
                    # settled it: the cards the node added to the deck, as upgraded.
                    after = Counter(signature(c) for c in s['end']['deck']) - Counter(signature(c) for c in state['deck'])
                    anchor['upgraded_enchantments'] = [
                        (json.loads(x)['enchantment'] or {}).get('id') for x in after
                        if anchor['upgraded'] and json.loads(x)['id'] == anchor['upgraded'][0]]
                if anchor['kind'] == 'reward':
                    # One native offer holds three cards, four with an extra-choice relic;
                    # a longer list is several offers that the summary does not separate.
                    if anchor['offer_size'] not in (3, 4) or len(anchor['picked']) > 1:
                        raise ValueError('multiple_card_rewards')
                    if not anchor['picked'] and (left['potions'] or left['relics']):
                        # The same leave decision has a complete group in unclaimed.
                        # Do not teach it again with the other visible rewards erased.
                        raise ValueError('covered_by_unclaimed_reward_group')
                    anchor['expected'] = expected_end(state, None)
                    for card in anchor['picked']:
                        # A relic can add a second copy of the picked card: the summary
                        # lists both as gained, and neither was in the deck before the pick.
                        copies = sum(c['id'] == card['id'] for c in stats.get('cards_gained', []))
                        for _ in range(max(1, copies)):
                            state['deck'].remove(find_card(state['deck'], card))
                    anchor['pending'] = len(s[anchor['state']]['deck']) - len(state['deck'])
                if any(c['id'] in suspect for c in state['deck']):
                    raise ValueError('unresolved_final_card_validation:' + ','.join(sorted(suspect)))
                anchor['ledger'] = state
            except ValueError as exc:
                anchor.update(status='reconstruction_failed', error=str(exc))
            yield anchor, node


def expected_end(state, error):
    if state is None:
        raise ValueError('ledger:' + str(error))
    return dict(hp=state['hp'], max_hp=state['max_hp'], gold=state['gold'],
                deck=sorted(signature(c) for c in state['deck']))


def check_installed(actual, expected):
    """The native round trip of an installed anchor must reproduce the supplied state."""
    for name in ('current_hp', 'max_hp', 'gold', 'max_energy', 'max_potion_slot_count', 'base_orb_slot_count'):
        if actual[name] != expected[name]:
            raise ValueError('anchor_roundtrip:' + name)
    if Counter(signature(c) for c in actual['deck']) != Counter(signature(c) for c in expected['deck']):
        raise ValueError('anchor_roundtrip:deck')
    if [r['id'] for r in actual['relics']] != [r['id'] for r in expected['relics']]:
        raise ValueError('anchor_roundtrip:relic_order')
    if actual.get('potions', []) != expected.get('potions', []):
        raise ValueError('anchor_roundtrip:potions')
    for name in ('deck', 'relics'):
        for a, b in zip(actual[name], expected[name], strict=True):
            have = props_values(a)
            for prop, value in props_values(b).items():
                if have.get(prop, False if isinstance(value, bool) else 0) != value:
                    raise ValueError('anchor_roundtrip:props:' + b['id'] + ':' + prop)


def installed_state(engine):
    reply = engine.send(dict(cmd='anchor_state'))
    if reply.get('type') != 'anchor_state':
        raise ValueError('anchor_state:' + str(reply.get('message')))
    return reply['save']['players'][0]


def enter(engine, save, anchor):
    reply = engine.send(dict(cmd='enter_anchor', save=save, act_floor=anchor['floor'],
        map_point_type=anchor['map_point_type'], second_boss=anchor['second_boss'], route=anchor['route']))
    if reply.get('type') != 'anchor_installed':
        raise ValueError('enter_anchor:' + str(reply.get('message')))
    check_installed(installed_state(engine), save['players'][0])
    # Where the engine placed the run: on the recorded route, or on a stand-in point.
    anchor['map_position'], anchor['native_route'] = reply['map_position'], reply.get('route')
    anchor['route_exits'] = reply.get('route_exits')
    frame = settle(engine, engine.send(dict(cmd='anchor_room', room=anchor['room'])))
    if frame['contract'].get('initialization') != INITIALIZATION:
        raise ValueError('anchor_contract_missing')
    return frame


class RunProcess:
    """A native process that anchors of one run share.

    Entering an anchor ends the run of the anchor before it, the way the game ends
    a run it leaves for the menu, so the game is loaded once for the run instead of
    once for every anchor, and a solver search runs on code the process has already
    compiled. The engine refuses that while game code of the old run may still
    execute or after the run failed inside it. Whatever goes wrong when a used
    process begins a run, the run is begun again in a new process, and only that
    outcome stands.
    """

    def __init__(self, config):
        self.config, self.engine, self.logged = config, None, 0
        self.build_identity = {}
        if Path(config).is_file():
            settings = read(config)
            for name in ('worker_dll', 'game_dll', 'solver_dll'):
                path = Path(settings[name])
                self.build_identity[name + '_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            adapter = Path(settings['worker_dll']).with_name('Sts2Headless.dll')
            self.build_identity['adapter_dll_sha256'] = hashlib.sha256(adapter.read_bytes()).hexdigest()
        # What one anchor learns about the run that later anchors of the run start from.
        self.memo = {}

    def begin(self, start):
        """The process and what `start(engine)` returned on it."""
        if self.engine is not None:
            try:
                return self.engine, start(self.engine)
            except Exception:
                self.close()
        self.engine = SolverEngine(self.config)
        return self.engine, start(self.engine)

    def enter(self, save, anchor):
        def start(engine):
            # What the process logged for earlier anchors is not this anchor's.
            self.logged = os.fstat(engine.stderr.fileno()).st_size
            return enter(engine, save, anchor)
        return self.begin(start)

    def native_error(self):
        return native_replay_error(self.engine, self.logged)

    def close(self):
        if self.engine is not None:
            self.engine.close()
        self.engine = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def advance(engine, frame, candidate):
    """The next frame, or None when the node leads nowhere: an act boss anchor has
    no next act to travel to, which the engine reports as an empty legal set."""
    try:
        return settle(engine, engine.send(execution_command(frame, candidate['candidate_ref'])))
    except ValueError as exc:
        if 'empty_legal_set' in str(exc):
            return None
        raise


def end_key(frame):
    return state_key(frame) if frame else 'node_end_without_travel'


def replay(second, save, anchor, records, final=None):
    """Another process repeats the executed actions; nothing is searched or chosen again."""
    engine, frame = second.enter(save, anchor)
    for index, record in enumerate(records):
        if frame is None or state_key(frame) != record['before_hash']:
            raise ValueError(f'replay_diverged:{index}')
        frame = advance(engine, frame, resolve(frame, record['label']))
    if final is not None and end_key(frame) != final:
        raise ValueError('replay_final_diverged')
    error = second.native_error()
    if error:
        raise ValueError(error)
    return frame


# Solver time per new search, by the kind of fight. The worker tells a boss fight by
# its room; an elite is told apart here, from the summary.
BUDGETS = dict(budget_ms=1000, elite_budget_ms=1000, boss_budget_ms=5000)
# Runs in progress at once. A search is one thread cut off by the clock: as many
# battles as the machine has cores give the most fights per hour, and each search
# then expands about a quarter fewer nodes than with half as many. Choice anchors
# never search.
WORKERS = dict(battle=16, choice=16)


def play_battle(shared, second, save, anchor, budgets):
    """The solver fights the battle on one process of the run; the other repeats it."""
    records = []
    budget_ms = budgets['elite_budget_ms' if anchor['room_type'] == 'elite' else 'budget_ms']
    engine, frame = shared.enter(save, anchor)
    contract, entry_hp = deepcopy(frame['contract']), player(frame)['hp']
    while in_combat(engine, frame):
        if len(records) >= 1000:
            raise TimeoutError('solver_step_limit')
        before = frame
        reply = engine.step(frame, budget_ms=budget_ms, boss_budget_ms=budgets['boss_budget_ms'], potions=True,
                            reuse_turn_plan=True)
        if reply.get('type') == 'solver_step':
            chosen = next(c for c in before['legal']['candidates'] if c['candidate_ref'] == reply['candidate_ref'])
            frame, actor = settle(engine, reply['frame']), 'combat_solver'
        elif reply.get('type') == 'solver_selection_required':
            chosen = max(before['legal']['candidates'], key=lambda c: preference(before, c))
            frame = settle(engine, engine.send(execution_command(before, chosen['candidate_ref'])))
            actor = 'combat_selection_rule'
        else:
            raise ValueError('solver_failed:' + str(reply.get('message') or reply.get('type')))
        records.append(row(before, chosen, actor))
    error = shared.native_error()
    if error:
        raise ValueError(error)
    final = state_key(frame)
    lost = frame['boundary'] == 'terminal' and not frame['public'].get('outcome', {}).get('victory')
    if frame['boundary'] != 'terminal':
        exit_hp = player(frame)['hp']
    else:
        # Winning the last fight ends the run: no frame follows, the run state still holds the HP.
        exit_hp = None if lost else installed_state(engine)['current_hp']
    replay(second, save, anchor, records, final)
    return dict(records=records, contract=contract, outcome='loss' if lost else 'win', entry_hp=entry_hp,
                exit_hp=exit_hp)


def entity(frame, candidate):
    refs = candidate.get('source_refs', []) + candidate.get('target_refs', [])
    found = [e for e in frame['public']['entities'] if e.get('ref') in refs]
    return found[0] if len(found) == 1 else {}


def card_key(e):
    return e.get('content_id'), e.get('upgrade_level') or 0, e.get('enchantment')


def summary_key(card):
    return card['id'], card.get('current_upgrade_level', 0), (card.get('enchantment') or {}).get('id')


def rest_choice(anchor):
    """The recorded option, then the recorded upgrade target of a smith."""
    done = dict(option=False, card=False)

    def choose(frame):
        candidates, phase = frame['legal']['candidates'], frame['public']['phase']
        if phase == 'rest_site' and not done['option']:
            matches = [c for c in candidates if c['verb'] == 'CHOOSE_REST_OPTION'
                       and entity(frame, c).get('content_id') == anchor['choices'][0]]
            if len(matches) != 1:
                raise ValueError('rest_option_not_offered:' + anchor['choices'][0])
            done['option'] = True
            return matches[0]
        if phase == 'card_select' and done['option']:
            if len(candidates) == 1:
                return candidates[0]
            if done['card'] or len(anchor['upgraded']) != 1:
                raise ValueError('unsupported_rest_followup')
            matches = [c for c in candidates if entity(frame, c).get('content_id') == anchor['upgraded'][0]]
            # Copies that differ (enchantment, earlier upgrade) are different choices: take
            # the kind the ledger settled, or none.
            wanted = anchor.get('upgraded_enchantments') or []
            if len({card_key(entity(frame, c)) for c in matches}) > 1 and len(wanted) == 1:
                matches = [c for c in matches if entity(frame, c).get('enchantment') == wanted[0]]
            if not matches or len({card_key(entity(frame, c)) for c in matches}) != 1:
                raise ValueError('ambiguous_upgrade_target:' + anchor['upgraded'][0])
            done['card'] = True
            return matches[0]
        return None
    return choose


def reward_choice(anchor):
    done = [False]

    def choose(frame):
        if done[0] or frame['public']['phase'] != 'rewards':
            return None
        candidates = frame['legal']['candidates']
        takes = [c for c in candidates if c['verb'] == 'TAKE_CARD_REWARD']
        shown = Counter(card_key(entity(frame, c)) for c in takes)
        if shown != Counter(summary_key(c) for c in anchor['room']['cards']):
            raise ValueError('native_offer_differs_from_record')
        done[0] = True
        if anchor['picked']:
            return next(c for c in takes if card_key(entity(frame, c)) == summary_key(anchor['picked'][0]))
        # Without a pick the summary cannot tell a skip from an alternative.
        if any(c['verb'] == 'CHOOSE_REWARD_ALTERNATIVE' for c in candidates):
            raise ValueError('skip_or_alternative_unknown')
        return next(c for c in candidates if c['verb'] == 'LEAVE_REWARDS')
    return choose


def play_map(shared, save, anchor):
    """The recorded next node as a travel label. The move is not executed: what lies
    in the next room is drawn afresh and proves nothing about the label."""
    _, frame = shared.enter(save, anchor)
    if anchor['map_position'] != 'recorded_route':
        raise ValueError('route_not_determined')
    moves = [c for c in frame['legal']['candidates'] if c['verb'] == 'MOVE_TO_NODE']
    if len(moves) < 2:
        return dict(records=[], contract=deepcopy(frame['contract']), outcome='single_move')
    col, floor = anchor['native_route'][anchor['floor']]
    chosen = [c for c in moves if (entity(frame, c).get('col'), entity(frame, c).get('floor')) == (col, floor)]
    if len(chosen) != 1:
        raise ValueError('recorded_node_not_reachable')
    records = [dict(row(frame, chosen[0], 'historical_noncombat'),
        coverage=dict(state='summary_anchor', options='complete_native', label='historical_route'))]
    return dict(records=records, contract=deepcopy(frame['contract']), outcome='on_recorded_route')


def play_ancient(shared, save, anchor):
    """The recorded pick among the recorded offer. It is not executed: what the relic
    then asks for (a card to upgrade, to transform) is not in a summary."""
    _, frame = shared.enter(save, anchor)
    options = [c for c in frame['legal']['candidates'] if c['verb'] == 'CHOOSE_EVENT_OPTION']
    shown = [str(entity(frame, c).get('content_id')) for c in options]
    wanted = anchor['room']['options']
    if len(shown) != len(wanted) or any(s != w and not s.endswith('.' + w) for s, w in zip(shown, wanted)):
        raise ValueError('native_offer_differs_from_record')
    records = [dict(row(frame, options[anchor['chosen'][0]], 'historical_noncombat'),
        coverage=dict(state='summary_anchor', options='complete_native', label='historical_choice'))]
    return dict(records=records, contract=deepcopy(frame['contract']), outcome='label_legal')


# What the shop rules read from a card, relic or potion: static content, asked once.
CONTENT = {}


def content_info(shared, ids):
    missing = sorted(set(ids) - set(CONTENT))
    if missing:
        engine, reply = shared.begin(lambda engine: engine.send(dict(cmd='content_info', ids=missing)))
        if reply.get('type') != 'content_info':
            raise ValueError('content_info:' + str(reply.get('message')))
        CONTENT.update(reply['items'])
    return CONTENT


def play_shop(shared, save, anchor, second=None):
    """The recorded purchases of a shop whose stock the game draws again as recorded.

    The stock is evidence of its own: the run's shop stream at the position earlier
    merchants left it, with the recorded rarities and relics, must show the recorded
    cards and potions slot by slot. The unlock state the player's pools had is the
    one under which it does; a run keeps it for all its shops. The order of the
    purchases is found by trying.
    """
    stock = anchor['stock']
    names = [c['id'] for c in stock['left_cards'] + stock['bought_cards']] + stock['left_relics'] \
        + stock['bought_relics'] + stock['left_potions'] + stock['bought_potions']
    cards, colorless, relics, potions = shop_sequence.stock_candidates(stock, content_info(shared, names))
    player = save['players'][0]
    unlocked = list(player['unlock_state']['unlocked_epochs'])
    character = player['character_id'].split('.')[-1]
    locked = shared.memo.setdefault('locked_epochs', {part: set() for part in shop_sequence.PARTS})
    contract = None

    def enter_shop(seats, relic_seats, trial=None):
        nonlocal contract
        gone = set().union(*(trial or locked).values())
        player['unlock_state']['unlocked_epochs'] = [e for e in unlocked if e not in gone]
        anchor['room'] = dict(type='shop', cards=seats, relics=relic_seats)
        _, frame = shared.enter(save, anchor)
        contract = deepcopy(frame['contract'])
        return frame

    def reproduced(part, seats, trial):
        content = [name for name, _ in shop_sequence.shown(enter_shop(seats, relics[0], trial))]
        drawn = content[shop_sequence.PARTS[part][0]]
        return drawn in potions if part == 'potions' else drawn == (seats if part == 'cards' else colorless)

    # Each part is drawn from its own pool, and the relics are supplied: settle the
    # pools one at a time, the state that served the run's earlier shops first.
    for part, (_, epochs) in shop_sequence.PARTS.items():
        options = shop_sequence.locked_sets([e.format(character) for e in epochs])
        options.sort(key=lambda option: option != locked[part])
        for option in options:
            trial = dict(locked, **{part: option})
            found = [seats for seats in (cards if part == 'cards' else cards[:1]) if reproduced(part, seats, trial)]
            if found:
                locked[part] = option
                if part == 'cards':
                    cards = found
                break
        else:
            raise ValueError('shop_stock_not_reproduced:' + part)

    def choice(frame, candidate, leaving=False):
        return dict(row(frame, candidate, 'historical_noncombat'), coverage=dict(state='summary_anchor',
            options='complete_native', label='historical_choice' if leaving else 'reconstructed_order'))

    solutions = []
    for relic_seats in relics:
        shown = shop_sequence.shown(enter_shop(cards[0], relic_seats))
        rows, tried = shop_sequence.search(
            lambda: enter_shop(cards[0], relic_seats), lambda frame, candidate: advance(shared.engine, frame, candidate),
            lambda: recorded.end_state(installed_state(shared.engine)), stock, anchor['expected'], choice)
        if rows is not None:
            solutions.append(dict(rows=rows, shown=shown, orders_tried=tried, room=deepcopy(anchor['room'])))
    if not solutions:
        raise ValueError('shop_no_order_matches_record')
    if len({json.dumps(s['shown']) for s in solutions}) > 1:
        # Two relics changed hands and either could have stood in either slot at
        # these prices: what the player saw is not settled.
        raise ValueError('shop_relic_slots_ambiguous')
    best = solutions[0]
    error = shared.native_error()
    if error:
        raise ValueError(error)
    # Freeze the chosen slot assignment and replay the exact labels, including
    # leaving, in another process. Do not search or infer unlocks on that process.
    anchor['room'] = best['room']
    def verify(process):
        frame = replay(process, save, anchor, best['rows'])
        if frame is None or frame['public']['phase'] != 'map' \
                or recorded.end_state(installed_state(process.engine)) != anchor['expected']:
            raise ValueError('shop_replay_outcome_differs_from_record')
    if second is None:
        with RunProcess(shared.config) as verifier:
            verify(verifier)
    else:
        verify(second)
    missing = sorted(set().union(*locked.values()))
    # A shop where leaving is all the gold allows holds no decision.
    decided = any(len(r['options']) > 1 for r in best['rows'])
    return dict(records=best['rows'] if decided else [], contract=contract,
                outcome='matched_record' if decided else 'single_action',
                state_sources=['seed_shop_stream'] + (['inferred_unlock_state'] if missing else []),
                stock=dict(source='seed_shop_stream', draws=anchor['history']['draws'], removals=anchor['history']['removals'],
                           locked_epochs=missing, prices=[price for _, price in best['shown']]),
                order=dict(source='reconstructed_order', actions=len(shop_sequence.purchases(stock)),
                           orders_tried=best['orders_tried']),
                recovery=dict(offer='seed_shop_stream', order='reconstructed_order',
                    draws=anchor['history']['draws'], removals=anchor['history']['removals'],
                    locked_epochs=missing, prices=[price for _, price in best['shown']],
                    replay='independent_native_process'))


def play_treasure(shared, second, save, anchor):
    """Open and take/skip the recorded chest offer, then repeat independently.

    A chest has one automatic gold roll, before its relic decision. Try the
    complete native domain (42..52), without assigning the final gold or HP.
    Keep a result only when every matching roll yields the same public sequence.
    Ascension can round two raw rolls to the same amount; those are one observed
    outcome, not two competing reconstructions.
    """
    best, matched = None, []
    for gold_roll in range(42, 53):
        anchor['room']['gold_roll'] = gold_roll
        engine, frame = shared.enter(save, anchor)
        contract, rows = deepcopy(frame['contract']), []
        pending = deepcopy(anchor['changes'])

        def take(candidate):
            nonlocal frame
            rows.append(dict(row(frame, candidate, 'historical_noncombat'), coverage=dict(
                state='summary_anchor', options='complete_native', label='historical_choice')))
            frame = advance(engine, frame, candidate)

        if frame['public']['phase'] != 'treasure':
            raise ValueError('treasure_mixed_rewards')
        take(rewards.RecordedReward.only(frame['legal']['candidates'], 'OPEN_CHEST'))
        # The game may expose an empty extra-reward screen before showing the relic.
        if frame is not None and frame['public']['phase'] == 'rewards':
            if any(e.get('entity_type') == 'reward' for e in frame['public']['entities']):
                raise ValueError('treasure_mixed_rewards')
            take(rewards.RecordedReward.only(frame['legal']['candidates'], 'LEAVE_REWARDS'))
        if frame is None or frame['public']['phase'] != 'treasure':
            raise ValueError('treasure_offer_unknown')
        shown = [e for e in frame['public']['entities'] if e.get('zone') == 'treasure']
        if [e['content_id'] for e in shown] != [anchor['room']['relic']]:
            raise ValueError('treasure_offer_unknown')
        take(rewards.RecordedReward.only(frame['legal']['candidates'],
             'TAKE_TREASURE_RELIC' if anchor['picked'] else 'LEAVE_ROOM'))
        while frame is not None and frame['boundary'] == 'decision' and frame['public']['phase'] != 'map':
            if len(rows) > 20:
                raise TimeoutError('choice_step_limit')
            if frame['public']['phase'] != 'card_select':
                raise ValueError('treasure_mixed_rewards')
            try:
                take(recorded.recorded_selection(frame, pending))
            except recorded.Illegal as exc:
                raise ValueError('recorded_cards_not_selectable:' + str(exc)) from None
        if frame is None or frame['boundary'] != 'decision':
            raise ValueError('treasure_did_not_return_to_the_map')
        actual = recorded.end_state(installed_state(engine))
        wrong = sorted(k for k in anchor['expected'] if actual[k] != anchor['expected'][k])
        if wrong == ['gold']:
            continue
        if wrong:
            raise ValueError('outcome_differs_from_record:' + ','.join(wrong))
        error = shared.native_error()
        if error:
            raise ValueError(error)
        hashes = [r['before_hash'] for r in rows]
        if best is not None and best['hashes'] != hashes:
            raise ValueError('treasure_observation_ambiguous')
        matched.append(gold_roll)
        if best is None:
            best = dict(records=rows, hashes=hashes, final=end_key(frame), contract=contract)
    if best is None:
        raise ValueError('treasure_gold_unresolved')
    anchor['room']['gold_roll'] = matched[0]
    replay(second, save, anchor, best['records'], best['final'])
    if recorded.end_state(installed_state(second.engine)) != anchor['expected']:
        raise ValueError('replay_outcome_differs_from_record')
    return dict(records=best['records'], contract=best['contract'], outcome='matched_record',
                state_sources=['recorded_treasure_offer', 'derived_chest_gold'],
                recovery=dict(offer='recorded_single_chest_relic', order='native_chest_sequence',
                              matching_gold_rolls=matched, replay='independent_native_process'))


def play_unclaimed(shared, save, anchor):
    """Leaving a fight's reward screen with what the record says was left on it. The
    leave is not executed: it ends the node, and nothing the node records follows it."""
    _, frame = shared.enter(save, anchor)
    if frame['public']['phase'] != 'rewards':
        raise ValueError('reward_screen_not_shown')
    room, candidates = anchor['room'], frame['legal']['candidates']
    shown = Counter(e['content_id'] for e in frame['public']['entities'] if e.get('entity_type') == 'reward'
                    and e['content_id'] != 'CardReward')
    cards = Counter(card_key(entity(frame, c)) for c in candidates if c['verb'] == 'TAKE_CARD_REWARD')
    if shown != Counter(room['potions'] + room['relics']) or cards != Counter(summary_key(c) for c in room['cards']):
        raise ValueError('native_offer_differs_from_record')
    if any(c['verb'] == 'CHOOSE_REWARD_ALTERNATIVE' for c in candidates):
        # Without a pick the summary cannot tell a skipped card offer from an alternative taken.
        raise ValueError('skip_or_alternative_unknown')
    leave = [c for c in candidates if c['verb'] == 'LEAVE_REWARDS']
    if len(leave) != 1:
        raise ValueError('reward_screen_cannot_be_left')
    # A screen that can only be left holds no decision: what is on it cannot be taken.
    decided = len(candidates) > 1
    records = [dict(row(frame, leave[0], 'historical_noncombat'),
        coverage=dict(state='summary_anchor', options='complete_native', label='historical_choice'))]
    return dict(records=records if decided else [], contract=deepcopy(frame['contract']),
                outcome='label_legal' if decided else 'single_action',
                state_sources=['recorded_unclaimed_group', 'single_gold_source'])


def play_event(shared, save, anchor):
    """The recorded choices of an event the game generates again for this run.

    An event draws from a stream seeded by the run and the event, so the game shows
    what it showed, where the state it reads is the recorded one. Each recorded
    choice must be an option on screen; a selection a choice opens is answered with
    the cards the record names; the event's variables must end as the record shows
    them (a summary holds their value after the event); and the state the event
    ends in must be the recorded one.
    """
    engine, frame = shared.enter(save, anchor)
    contract = deepcopy(frame['contract'])
    steps, pending = list(anchor['choices']), deepcopy(anchor['changes'])
    reward = rewards.RecordedReward(anchor.get('reward_record', {}))
    rows, written = [], []

    def page():
        reply = engine.send(dict(cmd='anchor_event'))
        if reply.get('type') != 'anchor_event':
            raise ValueError('anchor_event:' + str(reply.get('message')))
        return reply

    def take(candidate):
        nonlocal frame
        rows.append(dict(row(frame, candidate, 'historical_noncombat'), coverage=dict(
            state='summary_anchor', options='complete_native', label='historical_choice')))
        if len(rows) > 60:
            raise TimeoutError('choice_step_limit')
        frame = advance(engine, frame, candidate)

    while frame is not None and frame['boundary'] == 'decision' and frame['public']['phase'] != 'map':
        if frame['public']['phase'] == 'rewards':
            take(reward.choose(frame))
            continue
        if frame['public']['phase'] != 'event':
            try:
                take(recorded.recorded_selection(frame, pending))
            except recorded.Illegal as exc:
                raise ValueError('recorded_cards_not_selectable:' + str(exc)) from None
            continue
        offered = [(i, o) for i, o in enumerate(page()['options']) if not o['locked']]
        option = lambda index: next(c for c in frame['legal']['candidates'] if c['verb'] == 'CHOOSE_EVENT_OPTION'
                                    and c['source_refs'] == [f'option:{index}'])
        named = [(i, o) for i, o in offered if steps and o['history'] and o['history']['title'] == steps[0].get('title')]
        if len(named) == 1:
            written.append((steps.pop(0), named[0][1]['history'].get('variables') or {}))
            take(option(named[0][0]))
        elif len(offered) == 1 and offered[0][1]['history'] is None:
            # The one way on from a page, which the game does not write down.
            take(option(offered[0][0]))
        else:
            raise ValueError('event_option_not_offered' if steps else 'event_choice_not_recorded')
    if steps:
        raise ValueError('event_ended_before_record')
    if frame is None or frame['boundary'] != 'decision':
        raise ValueError('event_did_not_return_to_the_map')
    # The record of every choice shows the event's variables as the event left them.
    final = page().get('variables') or {}
    compared = Counter(recorded.same_variables(step.get('variables') or {}, dict(shown, **{
        name: final[name] for name in shown if name in final})) for step, shown in written)
    if None in compared:
        raise ValueError('event_variables_differ')
    actual = recorded.end_state(installed_state(engine))
    wrong = sorted(k for k in anchor['expected'] if actual[k] != anchor['expected'][k])
    if wrong:
        raise ValueError('outcome_differs_from_record:' + ','.join(wrong))
    if not any(len(r['options']) > 1 for r in rows):
        raise ValueError('no_branching_decision')
    error = shared.native_error()
    if error:
        raise ValueError(error)
    return dict(records=rows, contract=contract, outcome='matched_record', state_sources=['seed_event_stream'] +
                (['recorded_reward_offer', 'unique_reward_choice'] if reward.used else []),
                variables=dict(compared), reward_decisions=sum(r['observation']['phase'] == 'rewards' for r in rows))


def play_choice(shared, save, anchor):
    choose = (rest_choice if anchor['kind'] == 'rest' else reward_choice)(anchor)
    records = []
    engine, frame = shared.enter(save, anchor)
    contract = deepcopy(frame['contract'])
    while frame is not None and frame['boundary'] == 'decision':
        chosen = choose(frame)
        if chosen is None:
            break
        if len(records) >= 20:
            raise TimeoutError('choice_step_limit')
        records.append(dict(row(frame, chosen, 'historical_noncombat'),
            coverage=dict(state='summary_anchor', options='complete_native', label='historical_choice')))
        frame = advance(engine, frame, chosen)
    if not any(len(r['options']) > 1 for r in records):
        raise ValueError('no_branching_decision')
    # The recorded node result checks the label against history, not only its legality.
    after, expected = installed_state(engine), anchor['expected']
    actual = dict(hp=after['current_hp'], max_hp=after['max_hp'], gold=after['gold'],
                  deck=sorted(signature(c) for c in after['deck']))
    wrong = sorted(k for k in expected if actual[k] != expected[k])
    if wrong == ['gold'] and anchor['kind'] == 'reward' and 'gold_from_pick' not in anchor:
        # Taking the card paid gold (a relic that pays for every card added). The
        # node-end gold already holds that payment, so the gold before the pick is
        # lower by exactly what the engine just paid; try once from there.
        anchor['gold_from_pick'] = actual['gold'] - expected['gold']
        adjusted = deepcopy(save)
        adjusted['players'][0]['gold'] -= anchor['gold_from_pick']
        if adjusted['players'][0]['gold'] < 0:
            raise ValueError('outcome_differs_from_record:gold')
        return play_choice(shared, adjusted, anchor)
    if wrong:
        raise ValueError('outcome_differs_from_record:' + ','.join(wrong))
    result = dict(records=records, contract=contract, outcome='matched_record')
    if anchor.get('gold_from_pick'):
        result['gold_from_pick'] = anchor['gold_from_pick']
    return result


# How a sample was checked, by anchor kind.
VERIFIED_BY = dict(battle='native_replay', rest='recorded_outcome', reward='recorded_outcome',
                   map='legal_label', ancient='legal_label', shop='recorded_outcome', event='recorded_outcome',
                   unclaimed='legal_label', treasure='recorded_outcome')


def work(task):
    """One run, in a worker process: (run hash, its anchors, None) or (run hash, None, why not).

    Runs are spread over processes, not threads: reading, comparing and writing the rows
    of a run takes a few seconds of Python, and the threads of one interpreter would take
    turns at it while their engines wait.
    """
    source, arguments = task[0], task[1:]
    try:
        return source['run_hash'], process(source, *arguments), None
    except Exception as exc:
        return source['run_hash'], None, f'{type(exc).__name__}:{exc}'


def process(source, directory, config, kinds, budgets=BUDGETS, reuse_battles=None):
    """The requested anchor kinds of one run. Each kind has its own report and row
    file, so another kind can be added later without touching what exists."""
    with RunProcess(config) as shared, RunProcess(config) as second:
        reports = [process_kind(source, directory, shared, second, kind, budgets, reuse_battles)
                   for kind in KINDS if kind in kinds]
    return dict(run_hash=source['run_hash'], anchors=[a for r in reports for a in r['anchors']])


def process_kind(source, directory, shared, second, kind, budgets=BUDGETS, reuse_battles=None):
    """All anchors of one kind for one run; rows go to a gzip file, the report stays small.

    A finished report stands while the source, kind version, native build and
    solver budgets agree. Choice rule changes do not invalidate battle caches.
    """
    directory = Path(directory)
    key = source['run_hash']
    raw = Path(source['path'])
    digest = hashlib.sha256(raw.read_bytes()).hexdigest()
    if digest != source['sha256']:
        raise ValueError('source_checksum_changed')
    # Budgets only shape solver battles; the other kinds never search.
    identity = dict(version=KIND_VERSIONS.get(kind, VERSION), source_sha256=digest, kind=kind,
                    **shared.build_identity, **(budgets if kind == 'battle' else {}))
    target = directory / 'runs' / f'{key}.{kind}.json'
    if target.exists():
        done = read(target)
        if all(done['identity'].get(k) == v for k, v in identity.items()):
            return done
    run = read(raw)
    reusable = {}
    if kind == 'battle' and reuse_battles is not None:
        from .battle_refresh import load_battles
        reusable = load_battles(reuse_battles, key, digest, budgets)
    base = template(source, run, directory, shared)
    states, info = node_states(run, base['save'])
    character = {c.upper(): c for c in CHARACTERS}[source['character'].upper()]
    group = split_group(run['seed'])
    reports = []
    samples = directory / 'samples' / f'{key}.{kind}.jsonl.gz'
    samples.parent.mkdir(parents=True, exist_ok=True)
    temp = samples.with_suffix('.tmp')
    exits = {}  # act -> drawn exits of every node on its recorded route
    with gzip.open(temp, 'wt', encoding='utf-8') as stream:
        for anchor, node in plan(run, key, states, info, {kind}):
            started = time.monotonic()
            report = {k: v for k, v in anchor.items()
                      if k not in ('ledger', 'room', 'expected', 'route', 'upgraded_enchantments', 'stock',
                                   'choices', 'changes', 'reward_record')}
            ledger = anchor.pop('ledger', None)
            known = exits.get(anchor['act'])
            if ledger is not None and kind == 'map' and known and known[anchor['floor'] - 1] < 2:
                # The route is already located: this node has one way on, so no engine is needed.
                report.update(status='verified', outcome='single_move', rows=0, map_position='recorded_route')
            elif ledger is not None:
                try:
                    state, evidence = complete(ledger, run, anchor['before'], anchor['id'], combat=kind == 'battle',
                                               travel=kind == 'map', pending=anchor.get('pending', 0))
                    save = build_save(base['save'], state, run, anchor['act'], anchor['floor'], node, anchor['id'])
                    if anchor['kind'] == 'shop':
                        shop_sequence.install(save, base['save'], anchor['history'])
                    report.update(state_evidence=evidence, save_sha256=hashlib.sha256(
                        json.dumps(save, sort_keys=True).encode()).hexdigest())
                    if anchor['kind'] == 'battle':
                        previous = reusable.get(anchor['id'])
                        if previous:
                            from .battle_refresh import refresh_battle
                            try:
                                result = refresh_battle(shared, second, save, anchor, previous)
                            except Exception as exc:
                                report['refresh_error'] = f'{type(exc).__name__}:{exc}'
                                result = play_battle(shared, second, save, anchor, budgets)
                                result['recovery'] = dict(mode='solver_after_incompatible_replay',
                                                          previous_version=previous['version'])
                        else:
                            result = play_battle(shared, second, save, anchor, budgets)
                            result['recovery'] = dict(mode='solver_missing_previous_battle')
                    elif anchor['kind'] == 'map':
                        result = play_map(shared, save, anchor)
                    elif anchor['kind'] == 'ancient':
                        result = play_ancient(shared, save, anchor)
                    elif anchor['kind'] == 'shop':
                        result = play_shop(shared, save, anchor, second)
                    elif anchor['kind'] == 'event':
                        result = play_event(shared, save, anchor)
                    elif anchor['kind'] == 'unclaimed':
                        result = play_unclaimed(shared, save, anchor)
                    elif anchor['kind'] == 'treasure':
                        result = play_treasure(shared, second, save, anchor)
                    else:
                        result = play_choice(shared, save, anchor)
                    records = result.pop('records')
                    contract = result.pop('contract')
                    report.update(result, status='verified', rows=len(records), map_position=anchor['map_position'])
                    if anchor.get('route_exits'):
                        exits[anchor['act']] = anchor['route_exits']
                    sources = sorted({e['source'] for e in evidence} | set(result.pop('state_sources', [])) | (
                        {'derived_pre_pick_gold'} if result.get('gold_from_pick') else set()))
                    for record in records:
                        solver = record['actor'] != 'historical_noncombat'
                        item = {k: record[k] for k in ('schema', 'observation', 'options', 'label', 'coverage', 'actor')}
                        item['metadata'] = dict(run_hash=key, character=character, ascension=run['ascension'],
                            game_version='v0.111.0', independent=True, bc_only=True,
                            native_replay_verified=kind in ('battle', 'treasure', 'shop'), verified_by=VERIFIED_BY[kind],
                            teacher_visibility='privileged' if solver else 'recorded_public',
                            label_source='combat_solver' if solver else record['coverage']['label'],
                            routing=record['routing'], contract=contract, sample_group=anchor['id'],
                            split_group=group, initialization=INITIALIZATION, anchor_kind=anchor['kind'],
                            outcome=report['outcome'], state_sources=sources, map_position=anchor['map_position'],
                            source_integrity=source.get('integrity'), **(
                                dict(recovery=result['recovery']) if 'recovery' in result else {}))
                        stream.write(json.dumps(item, ensure_ascii=False) + '\n')
                except Exception as exc:
                    report.update(status='reconstruction_failed' if isinstance(exc, ValueError)
                        and str(exc).split(':')[0] in RECONSTRUCTION | shop_sequence.UNRECOVERED | UNRECORDED | rewards.UNRECOVERED
                        else 'execution_failed',
                        error=f'{type(exc).__name__}:{exc}')
            report['seconds'] = round(time.monotonic() - started, 2)
            reports.append(report)
    temp.replace(samples)
    result = dict(identity=identity, run_hash=key, kind=kind, character=character, ascension=run['ascension'],
        split_group=group, source_integrity=source.get('integrity'), ledger=info, anchors=reports)
    save_json(target, result)
    return result


# Events and selections the record does not settle or the game does not generate again.
UNRECORDED = {'event_without_recorded_choice', 'event_choice_not_recorded',
              'event_variables_differ', 'event_option_not_offered', 'event_ended_before_record',
              'recorded_cards_not_selectable', 'gold_reward_unsettled', 'unclaimed_reward_repeated'}
# Failures of the summary state itself, as opposed to failures of native execution.
RECONSTRUCTION = {'unsupported_relic_lifecycle', 'missing_immutable_relic_anchor', 'invalid_girya_count',
    'unresolved_lizard_tail_use_time', 'unhandled_saved_relic', 'unsupported_persistent_card',
    'guilty_removal_missing', 'unresolved_growing_enchantment', 'missing_mad_science_properties',
    'potion_capacity_mismatch'}


def summarize(directory):
    counts, errors, seconds = Counter(), Counter(), Counter()
    runs = sorted((Path(directory) / 'runs').glob('*.json'))
    for path in runs:
        for anchor in read(path)['anchors']:
            kind = anchor['kind']
            counts[kind, anchor['status']] += 1
            seconds[kind] += anchor.get('seconds', 0)
            if anchor['status'] == 'verified':
                counts[kind, 'rows'] += anchor['rows']
                counts[kind, 'outcome:' + anchor['outcome']] += 1
            else:
                errors[kind, anchor['error'].split(':')[1 if anchor['error'].startswith(('ValueError', 'Timeout')) else 0]] += 1
    result = dict(version=VERSION, runs=len({read(p)['run_hash'] for p in runs}),
        anchors={k: {s: n for (kind, s), n in sorted(counts.items()) if kind == k} for k in KINDS},
        errors={k: dict(Counter({e: n for (kind, e), n in errors.items() if kind == k}).most_common()) for k in KINDS},
        worker_seconds={k: round(v, 1) for k, v in seconds.items()})
    save_json(Path(directory) / 'summary.json', result)
    return result


def outcome_row(report, anchor, rows, samples):
    """One solver fight as a combat-outcome label: the state it was entered with and how
    it ended. Its decision frames are the fight's rows in `samples`, won or lost."""
    first = rows[0]
    hero, meta = player(dict(public=first['observation'])), first['metadata']
    refs = {e.get('ref'): e for row in rows if row['label']['verb'] == 'USE_POTION'
            for e in row['observation']['entities']}
    lost = anchor['outcome'] == 'loss'
    end_hp = 0 if lost else anchor['exit_hp']
    return dict(schema=OUTCOME_SCHEMA, source='summary_anchor', actor='combat_solver', seed=meta['split_group'],
        run_hash=meta['run_hash'], contract=meta.get('contract'),
        anchor=anchor['id'], character=meta['character'], ascension=meta['ascension'], act=anchor['act'],
        floor=anchor['floor'], encounter=anchor['encounter'].split('.')[-1],
        kind={'monster': 'regular'}.get(anchor['room_type'], anchor['room_type']),
        entities=input_entities(dict(public=first['observation'])),
        solver=dict({k: report['identity'][k] for k in BUDGETS}, potions=True),
        start_hp=anchor['entry_hp'], max_hp=hero['max_hp'], end_hp=end_hp, hp_lost=anchor['entry_hp'] - end_hp,
        result='loss' if lost else 'win',
        turns=len({player(dict(public=row['observation'])).get('round') for row in rows}),
        potions_used=[refs[ref] for row in rows if row['label']['verb'] == 'USE_POTION'
                      for ref in row['label'].get('source_refs', []) if ref in refs],
        frames=dict(file=samples, group=anchor['id'], count=len(rows)),
        state_sources=meta['state_sources'], source_integrity=meta['source_integrity'])


def act_bosses(run):
    """The boss encounters of every act, in the order they were fought."""
    return [[room['model_id'] for node in act if node['map_point_type'] == 'boss'
             for room in node['rooms'] if room['room_type'] == 'boss'] for act in run['map_point_history']]


def shows_bosses(item, bosses):
    """Whether the boss nodes of a row's map name the act's recorded bosses, in order.

    The game shows the act's bosses on the map from the start of the act, and an
    anchor is entered with the recorded ones. A map differs from the record where
    the record does not fill the act's boss nodes: nothing says then who stood there.
    """
    entities = item['observation']['entities']
    nodes = sorted((e for e in entities if e['entity_type'] == 'map_node' and e['content_id'] == 'Boss'),
                   key=lambda e: e['floor'])
    if not nodes:
        return True
    act = next(e for e in entities if e['entity_type'] == 'player').get('act')
    known = bosses[act - 1] if type(act) is int and 0 < act <= len(bosses) else []
    return [e.get('encounter') for e in nodes] == known


# Level 6 packs sample rows within a few percent of level 9 in a third of the time.
COMPRESSION = 6


def export_report(task):
    """What one report gives an export: its exported rows as a gzip member of their own,
    its fights as outcome rows, and the counts of the rows kept, of those left out for
    a map without the recorded bosses, and of those of groups in which nothing was chosen."""
    path, source, samples, relative = task
    report = read(path)
    verified = {a['id']: a for a in report['anchors'] if a['status'] == 'verified'}
    bosses = act_bosses(read(source))
    battles, kept, unnamed = {}, [], Counter()
    with gzip.open(samples, 'rt', encoding='utf-8') as rows:
        for line in rows:
            item = json.loads(line)
            if item['schema'] != SCHEMA or item['label'] not in item['options']:
                raise ValueError('Export label is not a legal option')
            anchor = verified.get(item['metadata']['sample_group'])
            if anchor is None:
                continue
            if report['kind'] == 'battle':
                battles.setdefault(anchor['id'], []).append(item)
            if anchor['outcome'] != 'loss':
                if not shows_bosses(item, bosses):
                    unnamed[item['metadata']['anchor_kind']] += 1
                    continue
                chosen = (len(item['options']) > 1
                          and controller_for(item['observation'].get('phase'), item['options']) is None)
                kept.append((anchor['id'], item['metadata']['anchor_kind'], item['actor'], chosen,
                             json.dumps(item, ensure_ascii=False) + '\n'))
    # Forced steps and controller actions teach no policy
    # choices. Keep the group's evidence only if it also has a policy decision,
    # such as a relic's card selection. Battle outcomes are kept independently.
    decided = {group for group, _, _, chosen, _ in kept if chosen}
    forced = Counter(kind for group, kind, _, _, _ in kept if group not in decided)
    kept = [entry for entry in kept if entry[0] in decided]
    counts = Counter((kind, actor) for _, kind, actor, _, _ in kept)
    labels = [outcome_row(report, verified[key], items, relative) for key, items in battles.items()]
    member = gzip.compress(''.join(entry[-1] for entry in kept).encode('utf-8'), COMPRESSION)
    return member, labels, counts, unnamed, forced


def export(directory, manifest, target=None, outcomes=None, workers=1):
    """Verified anchors as independent supervised samples, and every solver fight as a
    combat-outcome label. Lost battles teach no actions and are left out of the samples,
    and so are groups in which nothing was chosen; as outcomes the fights are kept.
    `manifest` locates the run summaries, against whose recorded bosses the map of every
    row is checked.

    Reports are read by `workers` processes and written in the order of their names:
    the sample file is the gzip members of the reports one after another, which reads
    as one stream."""
    directory = Path(directory)
    sources = {s['run_hash']: s['path'] for s in read(manifest)['selected']}
    target = Path(target or directory / 'independent-training.jsonl.gz')
    outcomes = Path(outcomes or directory / 'combat-outcomes.jsonl.gz')
    counts, fights, unnamed, forced = Counter(), Counter(), Counter(), Counter()
    temp, pending = target.with_suffix('.tmp'), outcomes.with_suffix('.tmp')
    tasks = []
    for path in sorted((directory / 'runs').glob('*.json')):
        samples = directory / 'samples' / (path.stem + '.jsonl.gz')
        # A report is named after its run: <run hash>.<kind>.json.
        tasks.append((path, sources[path.name.split('.')[0]], samples, os.path.relpath(samples, outcomes.parent)))
    with temp.open('wb') as stream, gzip.open(pending, 'wt', encoding='utf-8', compresslevel=COMPRESSION) as labels, \
            ProcessPoolExecutor(max_workers=workers) if workers > 1 else nullcontext() as pool:
        results = pool.map(export_report, tasks, chunksize=8) if pool else map(export_report, tasks)
        for member, rows, kept, without_boss, without_decision in results:
            stream.write(member)
            counts.update(kept)
            unnamed.update(without_boss)
            forced.update(without_decision)
            for row in rows:
                labels.write(json.dumps(row, ensure_ascii=False) + '\n')
                fights[row['kind'], row['result']] += 1
    temp.replace(target)
    pending.replace(outcomes)
    summary = dict(path=str(target), counts={f'{k}/{a}': n for (k, a), n in sorted(counts.items())},
                   rows_without_recorded_boss=dict(sorted(unnamed.items())),
                   rows_without_decision=dict(sorted(forced.items())),
                   combat_outcomes=dict(path=str(outcomes), schema=OUTCOME_SCHEMA,
                                        counts={f'{k}/{r}': n for (k, r), n in sorted(fights.items())}),
                   continuous_winning_trajectory=False, initialization=INITIALIZATION)
    save_json(directory / 'independent-export-summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('mode', choices=['run', 'export'])
    parser.add_argument('--output', type=Path, required=True, help='Anchor data directory')
    parser.add_argument('--manifest', type=Path, help='Manifest of selected runs (spire_codex_data.sources)')
    parser.add_argument('--kinds', nargs='+', choices=KINDS, default=list(KINDS))
    parser.add_argument('--max-runs', type=int, default=0, help='0 means all')
    parser.add_argument('--workers', type=int,
        help=f'Runs in progress at once; default {WORKERS["battle"]} for battles and {WORKERS["choice"]} for the other kinds '
             'and for the reports read at once by export')
    parser.add_argument('--budget-ms', type=int, default=BUDGETS['budget_ms'], help='Solver time per search in a normal fight')
    parser.add_argument('--elite-budget-ms', type=int, default=BUDGETS['elite_budget_ms'])
    parser.add_argument('--boss-budget-ms', type=int, default=BUDGETS['boss_budget_ms'])
    parser.add_argument('--solver-config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--reuse-battles', type=Path,
        help='Previous anchor directory: revalidate and refresh stored battle actions before searching again')
    args = parser.parse_args()
    if args.mode == 'export':
        if not args.manifest:
            parser.error('export needs --manifest')
        result = export(args.output, args.manifest, workers=args.workers or WORKERS['choice'])
    else:
        if not args.manifest:
            parser.error('run needs --manifest')
        selected = read(args.manifest)['selected']
        if args.max_runs:
            selected = selected[:args.max_runs]
        failures = {}
        budgets = dict(budget_ms=args.budget_ms, elite_budget_ms=args.elite_budget_ms,
                       boss_budget_ms=args.boss_budget_ms)
        # Choice kinds first, then battles, each at its own concurrency.
        passes = [([k for k in args.kinds if k != 'battle'], args.workers or WORKERS['choice']),
                  ([k for k in args.kinds if k == 'battle'], args.workers or WORKERS['battle'])]
        for kinds, workers in passes:
            if not kinds:
                continue
            tasks = [(source, args.output, args.solver_config, set(kinds), budgets, args.reuse_battles)
                     for source in selected]
            with ProcessPoolExecutor(max_workers=workers) as pool:
                for key, report, error in pool.map(work, tasks):
                    if error:
                        failures[key] = error
                    done = Counter(a['status'] for a in report['anchors']) if report else {}
                    print(json.dumps(dict(run=key, kinds=kinds, anchors=dict(done), error=error)), flush=True)
        result = dict(summarize(args.output), run_failures=failures)
        save_json(args.output / 'last-run.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result.get('run_failures'):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
