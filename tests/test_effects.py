from copy import deepcopy

import torch

from model.representation import Vocabulary, _program, clean_entity, effect_tree, observation, symbols_from_frames


def test_random_result_identity_stays_unknown():
    a = {"kind": "random", "generator": {"object_kind": "card", "result_identity": {"known": False, "content_id": "SECRET_A"}}}
    b = deepcopy(a)
    b["generator"]["result_identity"]["content_id"] = "SECRET_B"
    assert effect_tree(_program(a)) == effect_tree(_program(b))


def test_alpha_renaming_preserves_variable_binding_and_zone_names():
    a = {"kind": "random", "bind": "item", "body": {"kind": "effect", "op": "ACQUIRE", "bindings": {
        "subject": {"kind": "variable", "name": "item"}, "destination": {"kind": "zone", "name": "hand"}}}}
    b = deepcopy(a)
    b["bind"] = "renamed"
    b["body"]["bindings"]["subject"]["name"] = "renamed"
    assert effect_tree(a) == effect_tree(b)
    assert any(role == "binds_variable" for _, _, role in effect_tree(a).edges)
    b["body"]["bindings"]["destination"]["name"] = "exhaust_pile"
    assert effect_tree(a) != effect_tree(b)


def test_effect_order_and_recipient_change_encoding(setup, demo):
    model, _ = setup
    frame = deepcopy(demo["macros"][0]["steps"][0]["frame"])
    program = {"kind": "sequence", "steps": [
        {"kind": "effect", "op": "DAMAGE", "args": {"amount": {"kind": "literal", "value": 6}},
         "bindings": {"recipient": {"kind": "entity", "ref": "card:0"}}},
        {"kind": "effect", "op": "GAIN_BLOCK", "args": {"amount": {"kind": "literal", "value": 3}},
         "bindings": {"recipient": {"kind": "entity", "ref": "player"}}}]}
    frame["public"]["entities"][1]["semantic_program"] = program
    reverse = deepcopy(frame)
    reverse["public"]["entities"][1]["semantic_program"]["steps"].reverse()
    rebound = deepcopy(frame)
    rebound["public"]["entities"][1]["semantic_program"]["steps"][0]["bindings"]["recipient"]["ref"] = "player"
    vocab = Vocabulary(symbols_from_frames([frame, reverse, rebound]), model.config.vocabulary_size)
    with torch.no_grad():
        a, b, c = [model.encoder(observation(f), vocab)[0] for f in (frame, reverse, rebound)]
    assert not torch.allclose(a, b)
    assert not torch.allclose(a, c)


def test_unrevealed_entity_and_internal_pile_order_are_removed():
    card = {"entity_type": "card", "ref": "c1", "zone": "draw_pile", "content_id": "STRIKE", "position": 77, "slot": 4}
    assert "position" not in clean_entity(card) and "slot" not in clean_entity(card)
    hidden = dict(card, revealed=False, cost=9, stats={"damage": 999})
    public = clean_entity(hidden)
    assert "content_id" not in public and "cost" not in public and "stats" not in public


def test_known_numeric_payload_is_also_whitelisted():
    a = clean_entity({"entity_type": "card", "stats": {"damage": {"value": 6, "known": True, "rng": 123}}})
    assert "rng" not in a["stats"]["damage"]
