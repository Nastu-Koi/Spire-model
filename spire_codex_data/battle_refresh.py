"""Refresh verified solver actions under the current public observation contract."""

from collections import defaultdict
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path

from model.public_history import HISTORY_KINDS


def without_history(public):
    """Only the explicitly versioned public history extension may differ."""
    result = deepcopy(public)
    result['memory'] = [fact for fact in result.get('memory', [])
                        if fact.get('entity_type') not in HISTORY_KINDS]
    return result


def load_battles(directory, key, source_sha256, budgets):
    from .anchor import SCHEMA, read

    directory = Path(directory)
    path = directory / 'runs' / f'{key}.battle.json'
    if not path.exists():
        return {}
    report = read(path)
    identity = report['identity']
    if identity.get('source_sha256') != source_sha256:
        raise ValueError('previous_battle_source_checksum_changed')
    if any(identity.get(k) != value for k, value in budgets.items()):
        raise ValueError('previous_battle_solver_budgets_changed')
    accepted = {a['id']: a for a in report['anchors'] if a['status'] == 'verified'}
    groups = defaultdict(list)
    samples = directory / 'samples' / f'{key}.battle.jsonl.gz'
    with gzip.open(samples, 'rt', encoding='utf-8') as stream:
        for line in stream:
            item = json.loads(line)
            meta = item['metadata']
            if meta['sample_group'] not in accepted:
                continue
            if (item['schema'] != SCHEMA or item['label'] not in item['options']
                    or meta.get('native_replay_verified') is not True
                    or meta.get('anchor_kind') != 'battle' or meta['run_hash'] != key):
                raise ValueError('previous_battle_not_verified')
            groups[meta['sample_group']].append(item)
    for key, rows in groups.items():
        if len(rows) != accepted[key]['rows']:
            raise ValueError('previous_battle_rows_incomplete')
    return {key: dict(anchor=accepted[key], rows=rows, version=identity['version'])
            for key, rows in groups.items()}


def refresh_battle(shared, second, save, anchor, previous):
    from .anchor import (advance, in_combat, installed_state, player, replay,
                         resolve, row, state_key, action_semantics)

    digest = hashlib.sha256(json.dumps(save, sort_keys=True).encode()).hexdigest()
    old = previous['anchor']
    if digest != old['save_sha256']:
        raise ValueError('previous_battle_initialization_changed')
    engine, frame = shared.enter(save, anchor)
    contract = deepcopy(frame['contract'])
    entry_hp = player(frame)['hp']
    records = []
    for index, item in enumerate(previous['rows']):
        if frame is None or not in_combat(engine, frame):
            raise ValueError(f'previous_battle_ended_early:{index}')
        if item['metadata']['contract']['game_assembly_sha256'] != contract['game_assembly_sha256']:
            raise ValueError('previous_battle_game_changed')
        if without_history(frame['public']) != without_history(item['observation']):
            raise ValueError(f'previous_battle_observation_changed:{index}')
        if [action_semantics(c) for c in frame['legal']['candidates']] != item['options']:
            raise ValueError(f'previous_battle_options_changed:{index}')
        chosen = resolve(frame, item['label'])
        records.append(row(frame, chosen, item['actor']))
        frame = advance(engine, frame, chosen)
    if frame is None or in_combat(engine, frame):
        raise ValueError('previous_battle_incomplete')
    error = shared.native_error()
    if error:
        raise ValueError(error)
    lost = frame['boundary'] == 'terminal' and not frame['public'].get('outcome', {}).get('victory')
    outcome = 'loss' if lost else 'win'
    exit_hp = (None if lost else installed_state(engine)['current_hp']) if frame['boundary'] == 'terminal' \
        else player(frame)['hp']
    if (outcome, entry_hp, exit_hp) != (old['outcome'], old['entry_hp'], old['exit_hp']):
        raise ValueError('previous_battle_outcome_changed')
    replay(second, save, anchor, records, state_key(frame))
    return dict(records=records, contract=contract, outcome=outcome, entry_hp=entry_hp, exit_hp=exit_hp,
                recovery=dict(mode='refreshed_native_replay', previous_version=previous['version'],
                              observation_comparison='exact_except_public_history',
                              solver_searches=0, independent_replay=True))
