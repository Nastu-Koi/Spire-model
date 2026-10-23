import gzip
import json

import pytest

from spire_codex_data import entry_battles
from spire_codex_data.anchor import SCHEMA


def entry(kind='normal', act=2, floor=4, encounter='ENCOUNTER.TUNNELER_WEAK', **fight):
    return dict(at='fight', save={'players': [{}]}, character='Defect', seed='C5D6', ascension=3,
                fight=dict(dict(fight=10, encounter=encounter, kind=kind, act=act, floor=floor, second_boss=False), **fight))


def test_a_selection_groups_its_fights_by_run_in_floor_order(tmp_path):
    saves = tmp_path / 'assets' / 'saves'
    saves.mkdir(parents=True)
    for name in ('a', 'b', 'c'):
        (saves / f'{name}.json.gz').write_bytes(gzip.compress(b'{}'))
    selection = tmp_path / 'assets' / 'pilot' / 'selection.json'
    selection.parent.mkdir()
    fight = dict(character='Defect', seed='S1', kind='normal')
    selection.write_text(json.dumps(dict(
        fights=[dict(fight, id='late', path='assets/saves/a.json.gz', act=2, floor=3),
                dict(fight, id='other', path='assets/saves/b.json.gz', act=1, floor=1, seed='S2')],
        extra=[dict(fight, id='early', path='assets/saves/c.json.gz', act=1, floor=8)])))
    runs = entry_battles.runs_of(selection)
    assert [r['run_id'] for r in runs] == ['Defect-S1', 'Defect-S2']
    assert [(f['id'], f['role']) for f in runs[0]['fights']] == [('early', 'extra'), ('late', 'fights')]
    # Save paths are relative to a parent of the selection that holds them.
    assert runs[0]['fights'][1]['path'] == str(saves / 'a.json.gz')
    with pytest.raises(FileNotFoundError):
        entry_battles.resolve(selection, 'assets/saves/missing.json.gz')


def test_an_entry_save_is_fought_at_its_floor_point_and_encounter():
    spot = entry_battles.spot_for(entry('boss', act=3, floor=15, encounter='ENCOUNTER.DOORMAKER_BOSS',
                                        map_point_type='boss', second_boss=True), 'run-021')
    assert spot == dict(id='run-021', floor=15, map_point_type='boss', second_boss=True, route=None,
                        map_position='entry_save', room_type='boss',
                        room=dict(type='combat', encounter='DOORMAKER_BOSS'), act=3,
                        encounter='ENCOUNTER.DOORMAKER_BOSS', kind='boss')
    # An entry without its map point takes it from the kind of fight.
    assert entry_battles.spot_for(entry('elite'), 'x')['map_point_type'] == 'elite'


def record(verb, selection=None, actor='combat_solver', options=2):
    observation = dict(phase='card_select' if selection else 'combat', **({'selection_context': selection} if selection else {}))
    label = dict(verb=verb, source_refs=[], target_refs=[])
    return dict(schema=SCHEMA, observation=observation, label=label, actor=actor, coverage={}, routing={},
                options=[label] + [dict(verb='OTHER', source_refs=[str(i)], target_refs=[]) for i in range(options - 1)])


def test_the_first_selection_of_a_fight_tells_how_many_cards_it_took():
    chip = dict(min_total=0, max_total=999999999)
    records = [record('SELECT_ONE', chip), record('SELECT_ONE', chip), record('FINISH_SELECTION', chip), record('PLAY_CARD')]
    assert entry_battles.setup_choice(records) == dict(picks=2, actor='combat_solver', max_total=999999999)
    assert entry_battles.setup_choice([record('PLAY_CARD')]) is None
    assert entry_battles.setup_choice([record('FINISH_SELECTION', chip, actor='combat_selection_rule')])['picks'] == 0


class Idle:
    """Stands in for a native process; nothing starts until a fight is played."""
    build_identity = {}

    def __init__(self, config):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


def fight_save(path):
    # Group sampling writes the save and the fight, without the run it belongs to.
    path.write_bytes(gzip.compress(json.dumps(dict(at='fight', save={'players': [{}]}, fight=dict(
        fight=3, encounter='ENCOUNTER.TUNNELER_WEAK', kind='normal', act=2, floor=4, second_boss=False))).encode()))
    return path


def test_an_entry_save_without_its_run_takes_the_run_from_the_selection(tmp_path, monkeypatch):
    monkeypatch.setattr(entry_battles, 'EntryProcess', Idle)
    monkeypatch.setattr(entry_battles, 'play_battle', lambda *_: dict(
        records=[record('PLAY_CARD')], contract={}, outcome='win', entry_hp=50, exit_hp=45))
    fight = dict(id='Defect:S1:act2-0-003', path=str(fight_save(tmp_path / 'fight.json.gz')), role='fights',
                 character='Defect', seed='S1', ascension=4, kind='normal', act=2, floor=4)
    report = entry_battles.process_run(dict(run_id='Defect-S1', fights=[fight]), tmp_path / 'out', 'no-config.json')
    assert [(f['status'], f['ascension']) for f in report['fights']] == [('verified', 4)]
    with gzip.open(tmp_path / 'out' / 'samples' / 'Defect-S1.jsonl.gz', 'rt') as stream:
        metadata = json.loads(stream.readline())['metadata']
    assert (metadata['run_hash'], metadata['ascension'], metadata['game_seed']) == ('Defect-S1', 4, 'S1')


def test_the_entry_save_of_a_failed_fight_is_kept(tmp_path, monkeypatch):
    def fail(*_):
        raise ValueError('solver_failed:choice_transaction_failed transaction=41')
    monkeypatch.setattr(entry_battles, 'EntryProcess', Idle)
    monkeypatch.setattr(entry_battles, 'play_battle', fail)
    save = fight_save(tmp_path / 'Regent-S1-003.json.gz')
    fight = dict(id='Regent:S1:003', path=str(save), role='fights', character='Regent', seed='S1', ascension=5,
                 kind='normal', act=2, floor=4)
    report = entry_battles.process_run(dict(run_id='Regent-S1', fights=[fight]), tmp_path / 'out', 'no-config.json')
    kept = tmp_path / 'out' / 'failed' / 'Regent-S1-003.json.gz'
    # Group sampling deletes its saves after the round: the failure can be fought again from the copy.
    assert report['fights'][0]['kept'] == str(kept) and kept.read_bytes() == save.read_bytes()


def test_a_run_that_fails_as_a_whole_is_counted_in_the_summary(tmp_path):
    save = fight_save(tmp_path / 'fight.json.gz')
    selection = tmp_path / 'selection.json'
    selection.write_text(json.dumps(dict(fights=[dict(id='a', path=str(save), character='Defect', seed='S1',
                                                      ascension=4, kind='normal', act=2, floor=4)])))
    output = tmp_path / 'out'
    output.mkdir()
    # A file where the run's samples go: the run fails before any fight.
    (output / 'samples').write_text('')
    result = entry_battles.run(selection, output, workers=1, config=str(tmp_path / 'no-config.json'), log=lambda _: None)
    assert list(result['run_failures']) == ['Defect-S1']
    assert result['summary']['normal']['fights'] == 1 and result['summary']['normal']['failures'] == {'FileExistsError': 1}
    # Nothing of the run is exported, and running the selection again fights it again.
    assert entry_battles.export(output)['counts'] == {}
    (output / 'samples').unlink()
    assert entry_battles.run(selection, output, workers=1, config=str(tmp_path / 'no-config.json'),
                             log=lambda _: None)['run_failures'] == {}


def test_only_the_won_fights_of_the_selection_are_exported(tmp_path):
    (tmp_path / 'runs').mkdir()
    (tmp_path / 'samples').mkdir()
    fights = [dict(id='won', role='fights', status='verified', outcome='win', rows=2),
              dict(id='lost', role='fights', status='verified', outcome='loss', rows=1),
              dict(id='extra', role='extra', status='verified', outcome='win', rows=1),
              dict(id='forced', role='fights', status='verified', outcome='win', rows=1),
              dict(id='failed', role='fights', status='execution_failed', error='ValueError:replay_diverged:3')]
    (tmp_path / 'runs' / 'Defect-S1.json').write_text(json.dumps(dict(fights=fights)))
    (tmp_path / 'selection.json').write_text(json.dumps(dict(runs=['Defect-S1'])))
    rows = [('won', record('PLAY_CARD')), ('won', record('END_TURN')), ('lost', record('PLAY_CARD')),
            ('extra', record('PLAY_CARD')), ('forced', record('END_TURN', options=1))]
    with gzip.open(tmp_path / 'samples' / 'Defect-S1.jsonl.gz', 'wt') as stream:
        for group, item in rows:
            stream.write(json.dumps(dict(item, metadata=dict(sample_group=group))) + '\n')
    summary = entry_battles.export(tmp_path)
    assert summary['counts'] == dict(extra_fights=1, fights=1, fights_without_decision=1, lost_fights=1, rows=2,
                                     rows_combat_solver=2)
    with gzip.open(tmp_path / 'independent-training.jsonl.gz', 'rt') as stream:
        assert [json.loads(line)['metadata']['sample_group'] for line in stream] == ['won', 'won']
    assert entry_battles.error_class('ValueError:replay_diverged:3') == 'replay_diverged'
