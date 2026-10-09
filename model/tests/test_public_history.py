import copy
import json

import pytest
import torch

from model.config import ModelConfig
from model.checkpoint import save_checkpoint, load_model
from model.model import PolicyValue
from model.protocol import ProtocolError
from model.public_history import HISTORY_VERSION, recorded_memory, draw_relations, unknown_memory
from model.recorder import PublicSnapshot
from model.representation import Vocabulary, clean_public, fields_of, observation
from model.steam import live_frame
from model.tests.test_encoding import action, build, frame, logits
from model.tests.test_steam import message, card


def combat_message():
    result = message("combat_play", [{"command": "end_turn", "args": {}},
        {"command": "play_card", "args": {"card_instance_id": 103, "target_instance_id": 200}}], {})
    player = result["state"]["players"][0]
    player["combat"] = {"energy": 3, "hand": [card("ANGER", 103)], "draw": [card("STRIKE_IRONCLAD", 100), card("STRIKE_IRONCLAD", 101),
                                              card("DEFEND_IRONCLAD", 102)]}
    result["state"]["combat"] = {"in_progress": True, "round": 2,
        "enemies": [{"instance_id": 200, "id": "CULTIST", "hp": 20, "max_hp": 50}]}
    return result


def test_legacy_recording_and_summary_history_remain_unknown():
    state = combat_message()["state"]
    old = PublicSnapshot(state)
    assert len(old.memory) == 16
    assert all(e["probability"]["value"] is None and not e["probability"]["known"]
               for e in old.memory if e["entity_type"] == "history_probability")
    assert old.memory[-1]["known"] is False
    assert not draw_relations(old.memory)
    legacy = frame(old.entities, [action("end", "END_TURN")])
    explicit = copy.deepcopy(legacy)
    explicit["public"]["memory"] = old.memory
    assert observation(legacy).digest == observation(explicit).digest


def test_live_and_recorded_history_bind_only_public_owners():
    msg = combat_message()
    state = msg["state"]
    memory = unknown_memory([]) + [
        {"entity_type": "previous_intent", "owner_ref": "200", "known": True,
         "applicable": True, "intent": "Attack", "damage": 6, "hits": 2},
        {"entity_type": "known_draw_position", "owner_ref": "100", "known": True, "position": 0},
        {"entity_type": "known_draw_position", "owner_ref": "102", "known": True, "position": 2},
    ]
    state["public_history"] = {"version": HISTORY_VERSION, "memory": memory}
    recorded = PublicSnapshot(state)
    live, _ = live_frame(msg)
    assert live["public"]["memory"] == clean_public({"memory": recorded.memory})["memory"]
    assert live["contract"]["public_history_version"] == HISTORY_VERSION
    order = [r for r in recorded.relations if r["role"] == "known_draw_before"]
    assert order == [{"source": recorded.refs[100], "target": recorded.refs[102], "role": "known_draw_before"}]
    state["public_history"]["memory"][-1]["owner_ref"] = "99999"
    with pytest.raises(ProtocolError, match="absent public entity"):
        PublicSnapshot(state)


def test_hidden_order_and_duplicate_cards_do_not_change_policy_outputs():
    msg = combat_message()
    msg["state"]["public_history"] = {"version": HISTORY_VERSION, "memory": unknown_memory([]) + [
        {"entity_type": "known_draw_position", "owner_ref": "100", "known": True, "position": 0}]}
    before, _ = live_frame(msg)
    swapped = copy.deepcopy(msg)
    swapped["state"]["players"][0]["combat"]["draw"].reverse()
    after, _ = live_frame(swapped)
    assert sum(e.get("zone") == "draw_pile" and e["entity_type"] == "card"
               for e in after["public"]["entities"]) == 3
    assert not any("position" in e for e in after["public"]["entities"] if e.get("zone") == "draw_pile")
    model, vocabulary = build([before, after])
    a, b = logits(model, vocabulary, [before, after])
    assert torch.allclose(a, b, atol=1e-5)


def test_unknown_is_not_zero_and_previous_intent_keeps_enemy_binding():
    unknown = unknown_memory([{"entity_type": "enemy", "ref": "enemy"}])
    cleaned = clean_public({"memory": unknown})["memory"]
    assert cleaned[-1]["owner_ref"] == "enemy"
    probability = next(f for f in fields_of(cleaned[1]) if f.name == "probability")
    assert probability.number[1] == 0
    zero = copy.deepcopy(cleaned[1])
    zero["probability"].update(value=0, known=True)
    assert next(f for f in fields_of(zero) if f.name == "probability").number[1] == 1
    hidden = clean_public({"entities": [{"entity_type": "board_cell", "revealed": False,
                                         "owner_ref": "hidden-reward"}]})
    assert "owner_ref" not in hidden["entities"][0]


def test_guessed_or_incomplete_recorded_probabilities_are_rejected():
    state = combat_message()["state"]
    state["public_history"] = {"version": HISTORY_VERSION, "memory": unknown_memory([])}
    state["public_history"]["memory"][1]["probability"]["value"] = .4
    with pytest.raises(ProtocolError, match="guessed value"):
        PublicSnapshot(state)
    state["public_history"]["memory"] = unknown_memory([])[:-1]
    with pytest.raises(ProtocolError, match="Incomplete"):
        PublicSnapshot(state)


def test_history_version_prevents_silent_checkpoint_resume(tmp_path):
    config = ModelConfig.tiny()
    f = frame([], [action("end", "END_TURN")])
    model = PolicyValue(config)
    vocabulary = Vocabulary.from_frames([f], config)
    save_checkpoint(tmp_path / "model", model, vocabulary)
    manifest_file = tmp_path / "model/manifest.json"
    manifest = json.loads(manifest_file.read_text())
    assert manifest["public_history_version"] == HISTORY_VERSION
    load_model(tmp_path / "model")
    del manifest["public_history_version"]
    manifest_file.write_text(json.dumps(manifest))
    with pytest.raises(ProtocolError, match="Public history input version"):
        load_model(tmp_path / "model")
