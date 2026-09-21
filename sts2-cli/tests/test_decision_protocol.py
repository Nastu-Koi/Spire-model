"""Real-engine checks for versioned decisions and native selection semantics."""
import json

import pytest


def prepare(game, deck=None):
    state = game.start(seed="protocol_a10", ascension=10)
    game.skip_neow(state)
    game.set_player(hp=70, max_hp=80, gold=999, relics=[], potions=[],
                    deck=deck or ["STRIKE_IRONCLAD"] * 5 + ["DEFEND_IRONCLAD"] * 4 + ["BASH"])


def advance(game):
    return game.send({"cmd": "advance_to_boundary"})


def execute(game, frame, candidate=None, **overrides):
    routing = frame["routing"]
    command = {"cmd": "execute_candidate", "decision_id": routing["decision_id"],
               "state_version": routing["state_version"],
               "candidate_ref": (candidate or frame["legal"]["candidates"][0])["candidate_ref"]}
    if "selection_revision" in routing:
        command["selection_revision"] = routing["selection_revision"]
    return game.send(command | overrides)


def smith(game):
    prepare(game)
    state = game.enter_room("rest_site")
    choice = next(o for o in state["options"] if o["option_id"] == "SMITH")
    state = game.act("choose_option", option_index=choice["index"])
    assert state["decision"] == "card_select"
    return advance(game)


def test_query_stable_and_rejected_commands_do_not_edit_prefix(game):
    frame = smith(game)
    assert frame["boundary"] == "decision", frame
    assert frame["contract"]["training_ready"] is False
    assert len(frame["legal"]["candidates"]) == 11  # Native Smith also permits canceling.
    assert frame == advance(game)
    for overrides, code in [({"state_version": -1}, "stale_decision"),
                            ({"decision_id": "old"}, "stale_decision"),
                            ({"selection_revision": 99}, "stale_selection"),
                            ({"selection_revision": None}, "stale_selection"),
                            ({"candidate_ref": "invented"}, "unknown_candidate")]:
        result = execute(game, frame, **overrides)
        assert result["error"]["code"] == code
        assert advance(game) == frame
    result = execute(game, frame)
    assert result["public"]["selection_context"]["selected_count"] == 1
    assert result["routing"]["selection_revision"] == 1
    assert result["routing"]["base_public_version"] == frame["routing"]["base_public_version"]
    assert result["routing"]["action_bank_version"] == frame["routing"]["action_bank_version"]
    assert result["public"]["decoder_bank"] == frame["public"]["decoder_bank"]
    assert result["events"][0]["type"] == "candidate_executed"
    assert advance(game)["events"] == []
    assert execute(game, frame)["error"]["code"] == "stale_decision"
    assert [c["verb"] for c in result["legal"]["candidates"]] == ["FINISH_SELECTION", "CANCEL"]
    finished = execute(game, result)
    assert finished["boundary"] == "decision"
    assert finished["public"]["phase"] == "map"
    state = game.set_player()
    assert sum(card["upgraded"] for card in state["player"]["deck"]) == 1


@pytest.mark.parametrize("indices", ["-1", "500", "0,0", "0,1", "0,x", "", "0,"])
def test_legacy_selection_rejects_invalid_array_without_completing(game, indices):
    smith(game)
    assert game.act("select_cards", indices=indices)["type"] == "error"
    frame = advance(game)
    assert frame["boundary"] == "decision"
    assert frame["public"]["selection_context"]["selected_count"] == 0
    assert len(frame["legal"]["candidates"]) == 11
    assert game.act("skip_select")["type"] == "error"
    state = game.act("select_cards", indices="0")
    assert sum(card["upgraded"] for card in state["player"]["deck"]) == 1


@pytest.mark.parametrize("count", [0, 1, 2, 3])
def test_purity_stop_exhausts_only_selected_cards(game, count):
    deck = ["PURITY", "BASH", "STRIKE_IRONCLAD", "DEFEND_IRONCLAD", "ANGER"]
    prepare(game, deck)
    game.set_draw_order(deck)
    state = game.enter_room("combat", encounter="SHRINKER_BEETLE_WEAK")
    purity = next(card for card in state["hand"] if card["id"] == "CARD.PURITY")
    energy = state["energy"]
    state = game.act("play_card", card_index=purity["index"])
    assert state["decision"] == "card_select", state
    frame = advance(game)
    assert frame["public"]["selection_context"]["min_total"] == 0
    assert frame["public"]["selection_context"]["max_total"] == 3
    selected = []
    for _ in range(count):
        candidate = next(c for c in frame["legal"]["candidates"] if c["verb"] == "SELECT_ONE")
        ref = candidate["source_refs"][0]
        selected.append(next(c["content_id"] for c in frame["public"]["entities"] if c.get("ref") == ref))
        old = frame
        frame = execute(game, old, candidate)
        assert ref not in [c["source_refs"][0] for c in frame["legal"]["candidates"] if c["verb"] == "SELECT_ONE"]
    stop = next(c for c in frame["legal"]["candidates"] if c["verb"] == "FINISH_SELECTION")
    result = execute(game, frame, stop)
    assert result["public"]["phase"] == "combat"
    # Legacy proceed in an active combat only queries the settled public state.
    state = game.act("proceed")
    assert state["decision"] == "combat_play", state
    assert len(state["hand"]) == 4 - count
    assert state["energy"] == energy - purity["cost"]
    assert not set(selected) & {card["id"] for card in state["hand"]}


def test_debug_mutation_invalidates_handles(game):
    frame = smith(game)
    game.set_player(gold=50)
    assert execute(game, frame)["error"]["code"] == "stale_decision"


def test_rest_boundary_does_not_auto_choose(game):
    prepare(game)
    state = game.enter_room("rest_site")
    frame = advance(game)
    assert frame["boundary"] == "decision"
    assert frame["public"]["phase"] == "rest_site"
    assert {c["verb"] for c in frame["legal"]["candidates"]} == {"CHOOSE_REST_OPTION"}
    assert advance(game) == frame
    assert game.set_player()["player"] == state["player"]


def test_public_packet_has_no_internal_routing_or_rng(game):
    frame = smith(game)
    public = json.dumps(frame["public"]).lower()
    for forbidden in ["seed", "rng", "instance_id", "state_version", "decision_id", "engine_command"]:
        assert forbidden not in public
    assert frame["public"]["selection_context"]["known_masks"]["operation"] is True
    assert frame["public"]["selection_context"]["operation"] == "upgrade"


def test_requires_verified_a10_run(game):
    game.start(ascension=0)
    assert advance(game)["error"]["code"] == "unverified_run_contract"


def test_public_selection_bank_ignores_deck_order(game):
    from conftest import Game
    deck = ["BASH", "STRIKE_IRONCLAD", "DEFEND_IRONCLAD", "ANGER"]
    def selection(client, cards):
        prepare(client, cards)
        state = client.enter_room("rest_site")
        smith = next(o for o in state["options"] if o["option_id"] == "SMITH")
        client.act("choose_option", option_index=smith["index"])
        return advance(client)
    other = Game()
    try:
        first = selection(game, deck)
        second = selection(other, list(reversed(deck)))
        assert first["public"] == second["public"]
        assert first["legal"] == second["legal"]
    finally:
        other.close()


@pytest.mark.parametrize("field,value", [("state_version", "bad"), ("selection_revision", {}),
                                        ("candidate_ref", 0), ("decision_id", None)])
def test_malformed_protocol_request_does_not_consume_frame(game, field, value):
    frame = smith(game)
    result = execute(game, frame, **{field: value})
    assert result["code"] == "invalid_candidate_request"
    assert advance(game) == frame


def test_bundle_selection_and_invalid_legacy_index(game):
    state = game.start(seed="run_5", ascension=10)
    option = next(o for o in state["options"] if o["title"] == "Scroll Boxes")
    state = game.act("choose_option", option_index=option["index"])
    assert state["decision"] == "bundle_select"
    before = state["player"]["deck_size"]
    assert game.act("select_bundle", bundle_index=999)["type"] == "error"
    frame = advance(game)
    assert frame["public"]["phase"] == "bundle_select"
    assert len(frame["legal"]["candidates"]) == len(state["bundles"])
    candidate = frame["legal"]["candidates"][-1]
    bundle = next(e for e in frame["public"]["entities"] if e.get("ref") == candidate["source_refs"][0])
    result = execute(game, frame, candidate)
    assert result["public"]["phase"] == "map"
    assert game.set_player()["player"]["deck_size"] == before + len(bundle["cards"])


def test_optional_single_card_keeps_stop_candidate(game):
    prepare(game, ["PURITY", "BASH"])
    state = game.enter_room("combat", encounter="SHRINKER_BEETLE_WEAK")
    purity = next(c for c in state["hand"] if c["id"] == "CARD.PURITY")
    game.act("play_card", card_index=purity["index"])
    frame = advance(game)
    assert [c["verb"] for c in frame["legal"]["candidates"]] == ["SELECT_ONE", "FINISH_SELECTION"]
