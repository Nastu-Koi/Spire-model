"""Check summary-anchor samples against replays of the same runs.

Anchors are built from run summaries alone. Where a replay of the run exists, its
recorded offers, choices, coordinates and resources are an independent record of
what the player saw and did, and every anchor sample can be compared with it.

python -m spire_codex_data.anchor_audit data/anchors --replays data/spire-codex/raw
"""
import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path

from .fetch import save_json


PICK_COUNTERS = {'BOOK_OF_FIVE_RINGS'}


def timeline(journal):
    """Recorded facts by global floor, and the resources known before each record.

    A replay gives absolute HP only on heals; damage in between is subtracted, and
    `exact` says whether the value comes straight from an absolute record.
    """
    facts = dict(reward=defaultdict(list), rest={}, upgrade={}, ancient={}, path=defaultdict(list),
                 state={}, entered={}, decks=[], deck_kinds=[], battle={}, closed={}, acquired={}, rested={})
    hp = gold = max_hp = None
    exact = False
    header = journal[0] if journal and journal[0].get('t') == 'header' else {}
    relics, potions, counters, attempts = Counter(header.get('starting_relics') or []), Counter(), {}, {}
    for row in journal:
        kind, floor = row.get('t'), row.get('floor')
        facts['state'][row.get('s')] = dict(hp=hp if exact else None, max_hp=max_hp, gold=gold,
            relics=+relics, potions=+potions, counters=dict(counters))
        if kind == 'relic':
            relics[row['id']] += 1
        elif kind == 'relic_lost':
            relics[row['id']] -= 1
        elif kind == 'relic_counter':
            counters[row['id']] = row.get('n')
        elif kind == 'potion_got':
            potions[row['id']] += 1
        elif kind in ('potion_used', 'potion_dropped'):
            potions[row['id']] -= 1
        if row.get('decision_id') is not None:
            facts['closed'][row['decision_id']] = row['s']  # last record of the decision
        if kind == 'acquire' and row.get('source') == 'reward':
            facts['acquired'].setdefault(row.get('decision_id'), row.get('id'))  # a reused id keeps its first card
        if kind in ('hp', 'max_hp') and row.get('dst') == 'player':
            hp, max_hp, exact = row['hp'], row.get('max_hp', max_hp), True
        elif kind in ('hit', 'hp_loss') and (kind == 'hp_loss' or row.get('dst') == 'player'):
            exact = False
        elif kind == 'gold':
            gold = row['gold']
        elif kind == 'buy' and type(row.get('gold_on_hand')) is int:
            gold = row['gold_on_hand']  # purchases are not followed by a gold record
        elif kind == 'deck':
            facts['decks'].append((row['s'], Counter(c['id'] for c in row['cards'])))
            facts['deck_kinds'].append((row['s'], Counter((c['id'], c.get('up', 0)) for c in row['cards'])))
        elif kind == 'combat_start':
            facts['battle'].setdefault(floor, row['s'])
            # A restarted fight starts again with the potions of its first attempt.
            if row.get('attempt_id') and row.get('combat_id') in attempts:
                potions = Counter(attempts[row['combat_id']])
            attempts.setdefault(row.get('combat_id'), Counter(potions))
        elif kind == 'room':
            facts['entered'].setdefault(floor, row['s'])
            if row.get('coord'):
                path = facts['path'][row['act']]
                if not path or path[-1] != row['coord']:
                    path.append(row['coord'])
        elif kind == 'decision' and row.get('decision_type') == 'card_reward' and row.get('source') == 'reward':
            if row.get('offer_generation', 0) == 0:
                facts['reward'][floor].append(row)
        elif kind == 'rest':
            facts['rest'].setdefault(floor, []).append(row['option'].upper())
            facts['rested'][floor] = row['s']
        elif kind == 'upgrade':
            facts['upgrade'].setdefault(floor, []).append(row['id'])
        elif kind == 'decision' and row.get('decision_type') == 'event':
            facts['ancient'].setdefault(floor, row)
        elif kind == 'outcome' and row.get('decision_type') == 'event':
            # Older replays name the chosen option without its index.
            facts.setdefault('chosen', {}).setdefault(row.get('decision_id'), row.get('option_id'))
    facts.setdefault('chosen', {})
    facts['final'] = dict(hp=hp if exact else None, max_hp=max_hp, gold=gold, relics=+relics, potions=+potions,
                          counters=dict(counters))
    return facts


def gained(facts, decision):
    """Cards the deck holds after a decision that it did not hold before, by the
    deck snapshots on either side of it."""
    before = [deck for at, deck in facts['decks'] if at < decision['s']]
    after = [deck for at, deck in facts['decks'] if at > facts['closed'][decision['decision_id']]]
    return None if not before or not after else after[0] - before[-1]


def deck_before(facts, s):
    """The last deck snapshot (card, upgrade level) the replay took before record s."""
    known = [deck for at, deck in facts['deck_kinds'] if at < s]
    return known[-1] if known else None


def shown_deck(row):
    return Counter((short(e['content_id']), e.get('upgrade_level') or 0) for e in row['observation']['entities']
                   if e.get('entity_type') == 'card' and e.get('zone') == 'deck')


def leaving(facts, floor):
    """Resources when the player left a floor: just before the next floor's room record."""
    later = [s for f, s in facts['entered'].items() if f is not None and f > floor]
    return facts['state'][min(later)] if later else facts['final']


def entity(row, action):
    refs = action.get('source_refs', []) + action.get('target_refs', [])
    return next((e for e in row['observation']['entities'] if e.get('ref') in refs), {})


def hero(row):
    return next(e for e in row['observation']['entities'] if e.get('entity_type') == 'player')


def short(content_id):
    return str(content_id).split('.')[-1]


def compare(rows, anchor, facts):
    """Named agreements (True/False) of one anchor's rows with the replay."""
    first, floor = rows[0], anchor['global_floor']
    result = {}
    if anchor['kind'] == 'battle':
        at = facts['battle'].get(floor)
        if at is None:
            return dict(replay_comparable=False)
        # A battle anchor starts at the first decision of the fight: entry effects have
        # run, so only what they cannot change is compared with the replay.
        deck = deck_before(facts, at)
        if deck is not None:
            cards = Counter((short(e['content_id']), e.get('upgrade_level') or 0) for e in first['observation']['entities']
                            if e.get('entity_type') == 'card' and e.get('zone') in ('deck', 'hand', 'draw_pile', 'discard_pile', 'exhaust_pile'))
            result['deck'] = shown_deck(first) == deck if shown_deck(first) else cards == deck
        known = dict(facts['state'][at], hp=None)
        player = hero(first)
        for name in ('max_hp', 'gold'):
            if known.get(name) is not None:
                result[name] = player[name] == known[name]
        result.update(belongings(first, known))
        return result
    if anchor['kind'] == 'reward':
        offers = facts['reward'].get(floor, [])
        if len(offers) != 1:
            return dict(replay_comparable=False)
        decision = offers[0]
        shown = [entity(first, o) for o in first['options'] if o['verb'] == 'TAKE_CARD_REWARD']
        result['offer_cards'] = Counter(short(e['content_id']) for e in shown) == \
            Counter(o['option_id'] for o in decision['options'])
        # The replay logs the offer when the screen opens; a relic taken on that
        # screen can still upgrade it before the pick, which the summary reflects.
        result['offer_upgrades'] = Counter((short(e['content_id']), e.get('upgrade_level') or 0) for e in shown) == \
            Counter((o['option_id'], o.get('up', 0)) for o in decision['options'])
        mine = short(entity(first, first['label']).get('content_id')) if first['label']['verb'] == 'TAKE_CARD_REWARD' else None
        result['label_vs_acquire_record'] = facts['acquired'].get(decision['decision_id']) == mine
        cards = gained(facts, decision)
        if cards is not None:
            # Independent of the acquire record, which older replays can mislabel.
            # A snapshot some floors later can also hold an offered card gained elsewhere.
            taken = {o['option_id'] for o in decision['options'] if cards[o['option_id']] > 0}
            result['label_vs_deck_snapshots'] = mine in taken if mine else not taken
        # HP as it stood at the pick (the next act's ancient heals before its room record).
        # Everything else as it stands once the floor's other rewards are claimed, which
        # the fixed rule does before the pick; players claimed in any order.
        known = dict(leaving(facts, floor), hp=facts['state'][facts['closed'][decision['decision_id']]]['hp'])
        # A relic that counts cards added counts the pick: its counter is compared as it
        # stood when the offer opened.
        opened = facts['state'][decision['s']]['counters']
        known['counters'] = {k: opened.get(k) if k in PICK_COUNTERS else v for k, v in known['counters'].items()}
        later = [s for f, s in facts['entered'].items() if f is not None and f > floor]
        after = deck_before(facts, min(later)) if later else None
        if after is not None:
            gain = after - shown_deck(first)
            result['deck'] = not (shown_deck(first) - after) and {name for name, _ in gain} <= (
                {mine} if mine else set())
        deck = None
    elif anchor['kind'] == 'rest':
        options = facts['rest'].get(floor)
        if not options or len(options) != 1:
            return dict(replay_comparable=False)
        result['label'] = short(entity(first, first['label'])['content_id']) == options[0]
        picks = [r for r in rows if r['label']['verb'].startswith('SELECT')]
        if picks:
            # Later upgrade records on the same floor number belong to the next fight.
            result['upgrade_target'] = short(entity(picks[0], picks[0]['label'])['content_id']) in \
                facts['upgrade'].get(floor, [])[:1]
        # Entering a rest site can heal (relics) before the choice, and the replay logs
        # the option after its own heal: HP is comparable only for options that do not heal.
        known = dict(facts['state'][facts['rested'][floor]])
        if options[0] == 'HEAL':
            known['hp'] = None
        deck = deck_before(facts, facts['rested'][floor])
    elif anchor['kind'] == 'ancient':
        decision = facts['ancient'].get(floor)
        if decision is None:
            return dict(replay_comparable=False)
        shown = [entity(first, o)['content_id'] for o in first['options'] if o['verb'] == 'CHOOSE_EVENT_OPTION']
        result['offer_in_order'] = shown == [o['option_id'] for o in decision['options']]
        result['label'] = entity(first, first['label'])['content_id'] == facts['chosen'].get(decision['decision_id'])
        known = facts['state'][decision['s']]
        deck = deck_before(facts, decision['s'])
    else:
        path = facts['path'].get(anchor['act'], [])
        if anchor['floor'] >= len(path):
            return dict(replay_comparable=False)
        target = entity(first, first['label'])
        result['label'] = f"{target['col']},{target['floor']}" == path[anchor['floor']]
        current = next((e for e in first['observation']['entities']
                        if e.get('entity_type') == 'map_node' and e.get('current')), {})
        result['position'] = f"{current.get('col')},{current.get('floor')}" == path[anchor['floor'] - 1]
        known = leaving(facts, floor)
        later = [s for f, s in facts['entered'].items() if f is not None and f > floor]
        deck = deck_before(facts, min(later)) if later else None
    if deck is not None:
        result['deck'] = shown_deck(first) == deck
    player = hero(first)
    for name in ('hp', 'max_hp', 'gold'):
        if known.get(name) is not None:
            result[name] = player[name] == known[name]
    result.update(belongings(first, known))
    return result


def belongings(row, known):
    """Relics, potions and displayed relic counters against what the replay had tracked."""
    if 'relics' not in known:
        return {}
    held = [e for e in row['observation']['entities'] if e.get('entity_type') == 'relic']
    # Presence only: replays log some pickups twice and do not log every loss.
    result = dict(relics={short(e['content_id']) for e in held} >= set(known['relics']),
                  potions=Counter(short(e['content_id']) for e in row['observation']['entities']
                                  if e.get('entity_type') == 'potion' and e.get('content_id')) == known['potions'])
    for e in held:
        name = short(e['content_id'])
        if e.get('counter') is not None and known['counters'].get(name) is not None:
            result['counter:' + name] = e['counter'] == known['counters'][name]
    return result


def audit(directory, replays):
    directory = Path(directory)
    journals = {p.name.removesuffix('.replay.jsonl'): p for root in replays for p in Path(root).glob('*.replay.jsonl')}
    totals, misses, runs = defaultdict(Counter), [], set()
    for report_path in sorted((directory / 'runs').glob('*.json')):
        report = json.loads(report_path.read_text())
        key, kind = report['run_hash'], report['kind']
        if key not in journals:
            continue
        journal = [json.loads(line) for line in journals[key].read_text().splitlines() if line.strip()]
        # A resumed journal mixes save branches; its floors are not one history.
        if any(row.get('t') == 'resume' for row in journal):
            totals['runs']['resumed_replay_skipped'] += kind == 'rest'
            continue
        runs.add(key)
        facts = timeline(journal)
        rows = defaultdict(list)
        with gzip.open(directory / 'samples' / (report_path.stem + '.jsonl.gz'), 'rt', encoding='utf-8') as stream:
            for line in stream:
                item = json.loads(line)
                rows[item['metadata']['sample_group']].append(item)
        for anchor in report['anchors']:
            if anchor['status'] != 'verified' or not anchor.get('rows'):
                continue
            checks = compare(rows[anchor['id']], anchor, facts)
            for name, agrees in checks.items():
                totals[kind][name + (':agree' if agrees else ':differ')] += 1
                if agrees is False and name != 'replay_comparable':
                    misses.append(dict(anchor=anchor['id'], check=name))
    result = dict(scope='verified anchors of runs whose replay has no resume', runs=sorted(runs),
                  checks={k: dict(sorted(v.items())) for k, v in totals.items()}, differences=misses)
    save_json(directory / 'replay-audit.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--replays', nargs='+', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.directory, args.replays)
    print(json.dumps({k: result[k] for k in ('scope', 'checks')}, ensure_ascii=False, indent=2))
    print(json.dumps(dict(runs=len(result['runs']), differences=len(result['differences']))))


if __name__ == '__main__':
    main()
