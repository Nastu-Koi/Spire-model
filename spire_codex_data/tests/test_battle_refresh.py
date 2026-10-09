from copy import deepcopy
import gzip
import json

import pytest

from spire_codex_data import anchor
from spire_codex_data.battle_refresh import load_battles, without_history


def test_comparison_allows_only_explicit_public_history():
    old = dict(phase='combat', entities=[dict(entity_type='player', hp=40)],
               memory=[dict(entity_type='other_fact', value=3)])
    current = deepcopy(old)
    current['memory'].append(dict(entity_type='previous_intent', known=True, owner_ref='enemy:0'))
    assert without_history(old) == without_history(current)
    current['entities'][0]['hp'] = 39
    assert without_history(old) != without_history(current)
    current = deepcopy(old)
    current['memory'][0]['value'] = 4
    assert without_history(old) != without_history(current)


def previous(tmp_path, rows=1, verified=True):
    (tmp_path / 'runs').mkdir()
    (tmp_path / 'samples').mkdir()
    (tmp_path / 'runs/h.battle.json').write_text(json.dumps(dict(
        identity=dict(version='legacy', source_sha256='source', **anchor.BUDGETS),
        anchors=[dict(id='fight', status='verified', rows=rows)])))
    item = dict(schema=anchor.SCHEMA, options=[dict(verb='END_TURN')], label=dict(verb='END_TURN'),
                metadata=dict(sample_group='fight', run_hash='h', anchor_kind='battle',
                              native_replay_verified=verified))
    with gzip.open(tmp_path / 'samples/h.battle.jsonl.gz', 'wt') as stream:
        stream.write(json.dumps(item) + '\n')


def test_reuse_requires_complete_previously_verified_rows(tmp_path):
    previous(tmp_path)
    assert load_battles(tmp_path, 'h', 'source', anchor.BUDGETS)['fight']['version'] == 'legacy'
    with pytest.raises(ValueError, match='source_checksum'):
        load_battles(tmp_path, 'h', 'different', anchor.BUDGETS)
    with pytest.raises(ValueError, match='budgets_changed'):
        load_battles(tmp_path, 'h', 'source', dict(anchor.BUDGETS, budget_ms=2))


@pytest.mark.parametrize('rows,verified,error', [(2, True, 'incomplete'), (1, False, 'not_verified')])
def test_reuse_rejects_incomplete_or_unverified_data(tmp_path, rows, verified, error):
    previous(tmp_path, rows, verified)
    with pytest.raises(ValueError, match=error):
        load_battles(tmp_path, 'h', 'source', anchor.BUDGETS)
