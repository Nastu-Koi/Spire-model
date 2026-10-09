"""Steam frames follow the engine protocol: a card reward is published with its cards."""

import json
from pathlib import Path

import pytest

from model import steam
from model.protocol import ProtocolError
from model.recorder import card_fields
from model.representation import clean_public
from model.steam import PROTOCOL, SteamBridge, choose_action, live_frame

FIXTURES = Path(__file__).parent / "fixtures"


def card(name, key):
    return dict(id=name, instance_id=key, card_type="Attack", rarity="Common", upgraded=False)


def message(scope, actions, context, ascension=7):
    player = dict(creature=dict(hp=50, max_hp=80, block=0, instance_id="hero"), character="IRONCLAD", gold=99,
                  deck=[card("STRIKE_IRONCLAD", "d1")], relics=[], potions=[])
    return dict(episode="e", token="t", state=dict(
        run=dict(ascension=ascension, act=1, floor=3), players=[player],
        legal=dict(schema=1, status="complete", scope=scope, actions=actions, context=context)))


class Policy:
    """Takes the candidate with the wanted verb; records what it was shown."""

    def __init__(self, verb):
        self.verb, self.frames = verb, []

    def choose(self, frame, sample=False):
        self.frames.append(frame)
        return type("Choice", (), dict(candidate_ref=next(
            c["candidate_ref"] for c in frame["legal"]["candidates"] if c["verb"] == self.verb)))


GOLD = dict(command="take_reward", args=dict(reward_instance_id="r0"))
LEAVE = dict(command="skip_rewards", args=dict(reward_set_id=0))


def take(key):
    return dict(command="take_card_reward", args=dict(reward_instance_id="r1", card_instance_id=key))


def rewards(*names, gold=True, alternatives=(), expanded=True):
    """The rewards screen after a fight: gold, and a card reward with its cards."""
    cards = [card(n, f"c{i}") for i, n in enumerate(names)]
    options = [dict(command="take_reward_alternative", args=dict(reward_instance_id="r1", option_id=a))
               for a in alternatives]
    offered = [take(c["instance_id"]) for c in cards] + options if expanded else [
        dict(command="take_reward", args=dict(reward_instance_id="r1"))]
    return message("reward_choice", ([GOLD] if gold else []) + offered + [LEAVE], dict(rewards=[
        dict(reward_type="Gold", instance_id="r0", value=12, selected=not gold),
        dict(reward_type="Card", instance_id="r1", selected=False, cards=cards, expanded_card_reward=expanded,
             alternatives=[dict(OptionId=a) for a in alternatives])]))


def offer(*names):
    cards = [card(n, f"c{i}") for i, n in enumerate(names)]
    picks = [dict(command="select_reward_option", args=dict(index=i)) for i in range(len(cards))]
    return message("card_reward", picks + [dict(command="select_reward_option", args=dict(index=None))],
                   dict(cards=cards, alternatives=[]))


def test_taking_a_card_is_one_action_on_the_rewards_screen():
    policy = Policy("TAKE_CARD_REWARD")
    # The policy can take a card before gold; collection order is a real choice.
    assert choose_action(policy, rewards("FEED", "ANGER", "HAVOC")) == take("c0")
    shown = policy.frames[0]["public"]
    assert shown["phase"] == "rewards"
    assert sorted(c["verb"] for c in policy.frames[0]["legal"]["candidates"]) == [
        "LEAVE_REWARDS"] + ["TAKE_CARD_REWARD"] * 3 + ["TAKE_REWARD"]
    offered = {e["ref"]: e["content_id"] for e in shown["entities"]
               if e["entity_type"] == "card" and e.get("zone") == "reward"}
    assert list(offered.values()) == ["CARD.FEED", "CARD.ANGER", "CARD.HAVOC"]
    # Both uncollected rewards remain visible; an offer is not a pile.
    assert [e["content_id"] for e in shown["entities"] if e["entity_type"] == "reward"] == ["GoldReward", "CardReward"]
    assert [(r["source"], r["target"]) for r in shown["relations"] if r["role"] == "offers"] == [
        ("reward:1", ref) for ref in offered]
    assert not any(e["entity_type"] == "pile_summary" and e["zone"] == "reward" for e in shown["entities"])


@pytest.mark.parametrize("verb,expected", [("TAKE_REWARD", GOLD), ("LEAVE_REWARDS", LEAVE)])
def test_gold_and_leaving_are_policy_choices(verb, expected):
    policy = Policy(verb)
    assert choose_action(policy, rewards("FEED", "ANGER")) == expected
    assert len(policy.frames) == 1


def test_declining_a_card_reward_is_leaving_and_an_alternative_is_its_own_action():
    shown = rewards("FEED", "ANGER", gold=False, alternatives=["REROLL"])
    assert choose_action(Policy("LEAVE_REWARDS"), shown) == LEAVE
    policy = Policy("CHOOSE_REWARD_ALTERNATIVE")
    assert choose_action(policy, shown)["args"] == dict(reward_instance_id="r1", option_id="REROLL")
    chosen = next(c for c in policy.frames[0]["legal"]["candidates"] if c["verb"] == "CHOOSE_REWARD_ALTERNATIVE")
    entity = next(e for e in policy.frames[0]["public"]["entities"] if e["entity_type"] == "reward_alternative")
    assert chosen["source_refs"] == [entity["ref"]] and entity["content_id"] == "REROLL"


def test_a_card_reward_without_its_cards_is_not_a_decision():
    # Opening the offer would be the only way to see it: that is the bridge's work.
    with pytest.raises(ProtocolError, match="without its cards"):
        live_frame(rewards("FEED", "ANGER", gold=False, expanded=False))


def test_the_installed_bridge_rewards_screen_is_a_frame():
    shown = json.loads((FIXTURES / "steam-rewards.json").read_text())
    frame, commands = live_frame(shown)
    verbs = [c["verb"] for c in frame["legal"]["candidates"]]
    assert verbs == ["TAKE_REWARD"] + ["TAKE_CARD_REWARD"] * 3 + ["LEAVE_REWARDS"]
    assert len(commands) == len(shown["state"]["legal"]["actions"])
    taken = [commands[c["candidate_ref"]] for c in frame["legal"]["candidates"] if c["verb"] == "TAKE_CARD_REWARD"]
    assert all(a["command"] == "take_card_reward" for a in taken)
    cards = {e["ref"]: e["content_id"] for e in frame["public"]["entities"] if e.get("zone") == "reward"}
    assert list(cards.values()) == ["CARD.STOMP", "CARD.FLAME_BARRIER", "CARD.FIGHT_ME"]
    assert [c["source_refs"] for c in frame["legal"]["candidates"] if c["verb"] == "TAKE_CARD_REWARD"] == [
        [ref] for ref in cards]


def test_a_recorded_card_has_the_fields_of_an_engine_card():
    shown = json.loads((FIXTURES / "steam-rewards.json").read_text())
    stomp = shown["state"]["legal"]["context"]["rewards"][1]["cards"][0]
    # RunSimulator.PublicSelectionCard for the same card.
    engine = dict(
        entity_type="card", content_id="CARD.STOMP", cost=3, card_type="Attack", rarity="Uncommon", upgraded=False,
        upgrade_level=0, x_cost=False, star_cost=-1, base_star_cost=-1, stars_x=False, retain=False,
        exhaust_on_next_play=False, rider_effect=None, keywords=[], stats=dict(Damage=12), enchantment=None,
        enchantment_amount=None, affliction=None, affliction_amount=None, target_type="AllEnemies",
        effect_coverage="opaque",
        semantic_program=dict(kind="effect", op="OPAQUE_RULE", content_id="CARD.STOMP", coverage="opaque"))

    def cleaned(entity):
        return clean_public(dict(phase="rewards", entities=[entity], relations=[], memory=[],
                                 decoder_bank=[]))["entities"][0]
    assert cleaned(card_fields(stomp)) == cleaned(engine)
    nimble = card_fields(dict(stomp, enchantment=dict(id="NIMBLE", amount=2)))
    assert (nimble["enchantment"], nimble["enchantment_amount"]) == ("ENCHANTMENT.NIMBLE", 2)


def test_the_offered_cards_and_the_ascension_change_the_frame():
    feed, strike = (live_frame(offer(name, "ANGER"))[0] for name in ("FEED", "STRIKE_IRONCLAD"))
    assert feed["public"] != strike["public"]
    hero = next(e for e in feed["public"]["entities"] if e["entity_type"] == "player")
    assert hero["ascension"] == 7 == feed["contract"]["fixed_ascension"]


def test_an_event_shows_the_numbers_of_its_options_and_a_closed_chest_only_its_lid():
    option = dict(instance_id="o1", text_key="WOOD_CARVINGS.pages.INITIAL.options.BIRD", locked=False,
                  displayed_variables=dict(Gold=50), displayed_names=dict(Card="CARD.PECK"))
    event = message("event_choice", [dict(command="choose_event_option", args=dict(option_instance_id="o1"))],
                    dict(event_id="WOOD_CARVINGS", options=[option]))
    # Left over from the last fight; shown, and published by the engine, only during one.
    event["state"]["players"][0]["combat"] = dict(energy=2, max_energy=3, stars=0)
    shown = live_frame(event)[0]["public"]["entities"]
    assert dict(entity_type="displayed_variable", owner_ref="option:0", content_id="Gold", amount=50,
                known=True) in shown
    # A name the option shows is published as the content it names, as the headless engine does.
    assert dict(entity_type="displayed_variable", owner_ref="option:0", content_id="Card", names="CARD.PECK",
                known=True) in shown
    hero = next(e for e in shown if e["entity_type"] == "player")
    assert (hero["energy"], hero["max_energy"], hero["stars"], hero["round"]) == (None,) * 4
    chest, commands = live_frame(message("treasure", [dict(command="open_chest", args={})], []))
    assert [c["verb"] for c in chest["legal"]["candidates"]] == ["OPEN_CHEST"]
    assert not any(e.get("zone") == "treasure" for e in chest["public"]["entities"])
    opened = message("treasure", [dict(command="pick_relic", args=dict(index=0)),
                                  dict(command="pick_relic", args=dict(index=None))],
                     [dict(index=0, relic=dict(id="ANCHOR"))])
    frame = live_frame(opened)[0]
    assert [c["verb"] for c in frame["legal"]["candidates"]] == ["TAKE_TREASURE_RELIC", "LEAVE_ROOM"]
    relic = next(e for e in frame["public"]["entities"] if e.get("zone") == "treasure")
    assert relic["content_id"] == "RELIC.ANCHOR" and "enabled" not in relic


def test_a_shop_entry_is_an_item_that_offers_its_card():
    def entry(key, kind, cost, stocked=True, **value):
        return dict(instance_id=key, value=dict(type="MegaCrit.Sts2.Core.Entities.Merchant." + kind, Cost=cost,
                                                IsStocked=stocked, **value))
    shop = message("shop", [dict(command="purchase", args=dict(entry_instance_id="e0")),
                            dict(command="purchase", args=dict(entry_instance_id="e2")),
                            dict(command="leave_shop", args={})],
                   [entry("e0", "MerchantCardEntry", 50, CreationResult=dict(Card=card("FEED", "s0"))),
                    entry("e1", "MerchantRelicEntry", 150, Model=dict(id="ANCHOR")),
                    entry("e2", "MerchantCardRemovalEntry", 75),
                    entry("e3", "MerchantCardEntry", 60, stocked=False)])
    frame = live_frame(shop)[0]
    items = [e for e in frame["public"]["entities"] if e["entity_type"] == "shop_item"]
    assert [(e["content_id"], e["price"], e["sold_out"]) for e in items] == [
        ("CARD.FEED", 50, False), ("RELIC.ANCHOR", 150, False), ("MerchantCardRemovalEntry", 75, False),
        ("MerchantCardEntry", 60, True)]
    assert not any("zone" in e for e in items)
    sold = next(e for e in frame["public"]["entities"] if e.get("zone") == "shop")
    assert (sold["entity_type"], sold["content_id"]) == ("card", "CARD.FEED")
    assert dict(source=items[0]["ref"], target=sold["ref"], role="offers") in frame["public"]["relations"]
    assert [(c["verb"], c["source_refs"]) for c in frame["legal"]["candidates"]] == [
        ("BUY_ITEM", [items[0]["ref"]]), ("BUY_ITEM", [items[2]["ref"]]), ("LEAVE_ROOM", [])]


def reply(tmp_path, monkeypatch, **payload):
    """One bridge answer, as the mod writes it."""
    monkeypatch.setattr(steam.uuid, "uuid4", lambda: type("Id", (), dict(hex="request"))())
    (tmp_path / "response-request.json").write_text(json.dumps(dict(payload, id="request")))
    bridge = SteamBridge(tmp_path, timeout=1)
    try:
        return bridge.request("observe")
    finally:
        bridge.close()


def test_the_bridge_must_speak_the_controllers_protocol(tmp_path, monkeypatch):
    assert reply(tmp_path, monkeypatch, status="waiting", protocol=PROTOCOL)["status"] == "waiting"
    with pytest.raises(ProtocolError, match="reinstall"):
        reply(tmp_path, monkeypatch, status="waiting")
    with pytest.raises(ProtocolError, match="switched off"):
        reply(tmp_path, monkeypatch, status="paused", protocol=PROTOCOL)


def test_the_map_names_the_bosses_of_the_act():
    def named(**run):
        shown = offer("FEED", "ANGER")
        shown["state"]["run"].update(run)
        shown["state"]["map"] = [dict(col=3, row=17, type="Boss", children=[]),
                                 dict(col=3, row=15, type="RestSite", children=[dict(col=3, row=16)]),
                                 dict(col=3, row=16, type="Boss", children=[dict(col=3, row=17)])]
        frame = live_frame(shown)[0]
        return {e["floor"]: e.get("encounter") for e in frame["public"]["entities"] if e["entity_type"] == "map_node"}
    # In the order they are fought, whatever order the nodes arrive in.
    assert named(boss="QUEEN_BOSS", second_boss="AEONGLASS_BOSS") == {
        15: None, 16: "ENCOUNTER.QUEEN_BOSS", 17: "ENCOUNTER.AEONGLASS_BOSS"}
    # A recording that does not say who a boss is leaves it unknown.
    assert named(boss="QUEEN_BOSS") == {15: None, 16: "ENCOUNTER.QUEEN_BOSS", 17: None}
    assert named() == {15: None, 16: None, 17: None}
