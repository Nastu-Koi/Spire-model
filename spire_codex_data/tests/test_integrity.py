from copy import deepcopy

import pytest

from spire_codex_data.integrity import console_kind, inspect_journal
from spire_codex_data.sources import audit, source_reasons, split_group
from spire_codex_data.tests.test_fetch import journal, run

ACTS = ['ACT.OVERGROWTH', 'ACT.HIVE', 'ACT.GLORY']
CATALOG = {'CHARACTER.IRONCLAD', 'CARD.A', 'CARD.FUTURE_CARD', *ACTS}


def summary():
    return dict(run(), acts=list(ACTS))


def resumed():
    j = journal()
    header = deepcopy(j[0])
    header.update(s=5, starting_max_hp=80, starting_relics=[], relics=[], potions=[])
    j += [header, dict(t='resume', s=6, act=1, floor=1, hp=50, gold=99,
        deck_size=0, rng_state={'shuffle': 1}, relics=[], potions=[]),
        dict(t='remap', s=7, cards=[], exact=True), dict(t='deck', s=8, cards=[]),
        dict(t='end', s=9, capture_status='complete')]
    return j


def test_console_queries_do_not_block_but_mutations_and_unknown_commands_do():
    for cmd, args in [('help', []), ('help', ['ancient']), ('ancient', [])]:
        j = journal()
        j[-1:] = [dict(t='console', s=4, cmd=cmd, args=args), dict(t='end', s=5)]
        assert not source_reasons(run(), j)
    assert console_kind(dict(cmd='ancient', args=['NEOW'])) == 'state_mutation_command'
    assert console_kind(dict(cmd='win')) == 'state_mutation_command'
    assert console_kind(dict(cmd='mprint', args=['debug'])) == 'unknown_command'


@pytest.mark.parametrize('mutation,reason', [
    (lambda j: j[0].update(mods=[dict(id='Cheat', affects_gameplay=True)]), 'gameplay_mods'),
    (lambda j: j[0].update(mods=[dict(id='UnknownHelper', affects_gameplay=False)]), 'unapproved_mods'),
    (lambda j: j[0].update(mods=[dict(id='BaseLib')]), 'mod_gameplay_effect_unknown'),
    (lambda j: j[0].update(harmony_owners=['UnknownPatcher']), 'unapproved_harmony_patches'),
    (lambda j: j.insert(-1, dict(t='console', s=3.5)), 'console_effect_unknown'),
])
def test_a_replay_that_shows_tampering_rejects_the_run(mutation, reason):
    j = journal()
    mutation(j)
    assert reason in audit(summary(), j, CATALOG)[0]


def test_a_summary_alone_is_usable_but_never_audited():
    assert audit(summary(), None, CATALOG) == ([], 'unverified_source')
    assert audit(summary(), journal(), CATALOG) == ([], 'audited_recording')
    # What a replay cannot show stays open rather than counting against the run.
    j = journal()
    del j[0]['mods']
    assert audit(summary(), j, CATALOG) == ([], 'unverified_source')


def test_content_outside_the_native_game_is_rejected_without_any_replay():
    r = summary()
    r['players'][0]['deck'] = [dict(id='CARD.MODDED')]
    assert 'unknown_native_content:CARD.MODDED' in audit(r, None, CATALOG)[0]
    assert 'unsupported_acts' in audit(dict(summary(), acts=['ACT.GLORY'] * 3), None, CATALOG)[0]
    assert 'invalid_ascension' in audit(dict(summary(), ascension=11), None, CATALOG)[0]


def test_resume_is_not_cheating_but_snapshots_do_not_prove_native_continuity():
    j = resumed()
    assert source_reasons(run(), j) == ['resume_requires_native_reconciliation']
    assert audit(summary(), j, CATALOG) == ([], 'unverified_source')
    # Missing evidence blocks the run when an advertised journal exists.
    del j[6]['potions']
    del j[5]['potions']
    assert 'resume_snapshot_incomplete' in audit(summary(), j, CATALOG)[0]


def test_each_session_manifest_is_checked():
    j = resumed()
    j[5]['mods'] = [dict(id='InjectedLater', affects_gameplay=True)]
    j[5]['harmony_owners'] = ['InjectedLater']
    assert {'gameplay_mods', 'unapproved_mods', 'unapproved_harmony_patches'} <= set(source_reasons(run(), j))
    assert inspect_journal(j)['manifests_changed']


def test_inexact_card_remap_and_rollback_are_visible():
    j = resumed()
    j[4]['hp'] = 1
    j[7]['exact'] = False
    boundary = inspect_journal(j)['resumes'][0]
    assert boundary['hp_changed'] and boundary['previous_hp'] == 1 and boundary['resumed_hp'] == 50
    assert 'exact_card_remap' in boundary['missing']
    assert not boundary['native_state_verified']


def test_lookalike_seeds_share_a_split_group():
    assert split_group('O1I') == split_group('0ll'.replace('l', '1')) != split_group('011X')


def test_a_manifest_rebuilt_over_a_growing_cache_only_gains_runs(tmp_path):
    import json
    from spire_codex_data.sources import collect
    raw = tmp_path / 'data' / 'raw'
    raw.mkdir(parents=True)
    catalog = tmp_path / 'catalog.json'
    catalog.write_text(json.dumps(dict(public=dict(entities=[dict(content_id=c) for c in CATALOG]))))

    def cache(name, seed):
        (raw / f'{name}.run.json').write_text(json.dumps(dict(summary(), seed=seed)))
    cache('m' * 16, 'S1')
    first = collect(tmp_path / 'sources', tmp_path / 'data', catalog, per_character=0)
    assert [s['run_hash'] for s in first['selected']] == ['m' * 16]
    # A later download that sorts first and repeats the seed does not displace the run already used.
    cache('a' * 16, 'S1')
    cache('z' * 16, 'S2')
    second = collect(tmp_path / 'sources', tmp_path / 'data', catalog, per_character=0)
    assert [s['run_hash'] for s in second['selected']] == ['m' * 16, 'z' * 16]
    assert second['rejected'] == {'a' * 16: ['duplicate_character_seed']}

