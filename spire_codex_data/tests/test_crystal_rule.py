import random
import pytest
from model.crystal_rule import collection_candidate
from model.protocol import ProtocolError


def frame(phase, entities, choices):
    return dict(public=dict(phase=phase, entities=entities), legal=dict(candidates=choices))


@pytest.mark.parametrize('field', ['source_refs', 'source_ref'])
def test_small_cells_only_in_both_protocols(field):
    entities = [dict(entity_type='tool', ref='tool:'+t, content_id=t) for t in ('Small', 'Big')]
    choices = [dict(verb='DIVINE_CELL', candidate_ref=str(i), **{field: [ref] if field.endswith('refs') else ref})
               for i, ref in enumerate(('tool:Small', 'tool:Big', 'tool:Small'))]
    f = frame('crystal_sphere', entities, choices)
    assert {collection_candidate(f, random.Random(i))['candidate_ref'] for i in range(30)} == {'0','2'}


def test_native_first_option_not_candidate_order_or_potion():
    choices = [dict(verb='DISCARD_POTION', source_refs=['potion:0']),
               dict(verb='CHOOSE_EVENT_OPTION', source_refs=['option:1']),
               dict(verb='CHOOSE_EVENT_OPTION', source_refs=['option:0'])]
    f = frame('event', [dict(entity_type='event', content_id='EVENT.CRYSTAL_SPHERE')], choices)
    assert collection_candidate(f) is choices[2]
    f['public']['entities'][0]['content_id'] = 'EVENT.NEOW'
    assert collection_candidate(f) is None


def test_missing_small_tool_fails_instead_of_using_model():
    with pytest.raises(ProtocolError):
        collection_candidate(frame('crystal_sphere', [], []))


def test_payment_plan_is_not_silently_taken_when_first_option_missing():
    f = frame('event', [dict(entity_type='event', content_id='EVENT.CRYSTAL_SPHERE')],
              [dict(verb='CHOOSE_EVENT_OPTION', source_refs=['option:1'])])
    with pytest.raises(ProtocolError):
        collection_candidate(f)
