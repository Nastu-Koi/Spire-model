"""Regression checks for effects that resume after a headless selection."""

import pytest


def setup_player(game, seed):
    state = game.start(seed=seed)
    game.skip_neow(state)
    game.set_player(hp=30, max_hp=80, gold=999, relics=[],
                    deck=["STRIKE_IRONCLAD"] * 5 + ["DEFEND_IRONCLAD"] * 4 + ["BASH"])


@pytest.mark.parametrize("attempt", range(5))
def test_smith_result_is_complete_before_map(game, attempt):
    setup_player(game, f"wait_smith_{attempt}")
    state = game.enter_room("rest_site")
    smith = next(o for o in state["options"] if o["option_id"] == "SMITH")
    state = game.act("choose_option", option_index=smith["index"])
    assert state["decision"] == "card_select"
    state = game.act("select_cards", indices="0")
    assert state["decision"] == "map_select"
    assert sum(c["upgraded"] for c in state["player"]["deck"]) == 1


@pytest.mark.parametrize("attempt", range(5))
def test_shop_removal_finishes_and_charges_once(game, attempt):
    setup_player(game, f"wait_remove_{attempt}")
    state = game.enter_room("shop")
    before = state["player"]
    cost = state["card_removal_cost"]
    state = game.act("remove_card")
    assert state["decision"] == "card_select"
    state = game.act("select_cards", indices="0")
    assert state["decision"] == "shop"
    assert state["player"]["deck_size"] == before["deck_size"] - 1
    assert state["player"]["gold"] == before["gold"] - cost
    again = game.act("select_cards", indices="0")
    assert again["type"] == "error"
    state = game.act("leave_room")
    assert state["player"]["gold"] == before["gold"] - cost


def test_event_continuation_finishes_after_selection(game):
    setup_player(game, "wait_sapphire")
    state = game.enter_room("event", event="SAPPHIRE_SEED")
    assert state["decision"] == "event_choice"
    state = game.act("choose_option", option_index=0)
    assert state["decision"] == "card_select"
    healed_hp = state["player"]["hp"]
    assert healed_hp > 30
    state = game.act("select_cards", indices="0")
    assert state["decision"] == "map_select"
    assert state["player"]["hp"] == healed_hp
    assert sum(c["upgraded"] for c in state["player"]["deck"]) == 1


@pytest.mark.parametrize("skip_middle", [False, True])
def test_three_event_rewards_do_not_lose_the_next_prompt(game, skip_middle):
    setup_player(game, f"wait_rewards_{skip_middle}")
    state = game.enter_room("event", event="COLORFUL_PHILOSOPHERS")
    assert state["decision"] == "event_choice"
    state = game.act("choose_option", option_index=0)
    before = state["player"]["deck_size"]
    for i in range(3):
        assert state["decision"] == "card_reward", state
        if i == 1 and skip_middle:
            state = game.act("skip_card_reward")
        else:
            state = game.act("select_card_reward", card_index=0)
    assert state["decision"] == "map_select", state
    assert state["player"]["deck_size"] == before + (2 if skip_middle else 3)
