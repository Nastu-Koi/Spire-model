import pytest
from model.crystal_rule import collection_candidate
from model.protocol import ProtocolError
from model.tests.test_crystal_planner import board_frame, corners


def frame(phase, entities, choices):
    return dict(public=dict(phase=phase, entities=entities), legal=dict(candidates=choices))


@pytest.mark.parametrize('field', ['source_refs', 'source_ref'])
def test_same_public_geometry_in_both_protocols(field):
    f = board_frame(revealed=corners())
    expected = collection_candidate(f)['candidate_ref']
    if field == 'source_ref':
        for choice in f['legal']['candidates']:
            choice['source_ref'] = choice.pop('source_refs')[0]
            choice['target_ref'] = choice.pop('target_refs')[0]
    assert collection_candidate(f)['candidate_ref'] == expected


def test_payment_and_potions_stay_with_policy():
    choices = [dict(verb='DISCARD_POTION', source_refs=['potion:0']),
               dict(verb='CHOOSE_EVENT_OPTION', source_refs=['option:1']),
               dict(verb='CHOOSE_EVENT_OPTION', source_refs=['option:0'])]
    f = frame('event', [dict(entity_type='event', content_id='EVENT.CRYSTAL_SPHERE')], choices)
    assert collection_candidate(f) is None
    f['public']['entities'][0]['content_id'] = 'EVENT.NEOW'
    assert collection_candidate(f) is None


def test_incomplete_public_board_fails_instead_of_using_model():
    with pytest.raises(ProtocolError):
        collection_candidate(frame('crystal_sphere', [], []))


def test_payment_plan_does_not_require_a_specific_option():
    f = frame('event', [dict(entity_type='event', content_id='EVENT.CRYSTAL_SPHERE')],
              [dict(verb='CHOOSE_EVENT_OPTION', source_refs=['option:1'])])
    assert collection_candidate(f) is None
