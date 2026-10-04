"""Rule text: description templates rendered with public values into local programs."""

import torch

from model import rules
from model.config import ModelConfig
from model.model import PolicyValue
from model.representation import Vocabulary, clean_entity, observation


def text(entity):
    words = []
    for row in rules.program(entity).nodes[1:]:
        fields = {f.name: f for f in row}
        if "variable" in fields:
            value = fields["value"]
            words.append("{%s=%s}" % (fields["variable"].symbol.split("=")[1], value.number[0] if value.number[3] else "?"))
        elif "number" in fields:
            words.append(str(fields["number"].number[0]))
        else:
            words.append(fields["word"].symbol.split("=", 1)[1])
    return " ".join(words)


def test_a_masked_stat_is_not_shown():
    known = {"content_id": "CARD.BASH", "stats": {"Damage": {"value": 8}, "VulnerablePower": 2}}
    assert text(known).startswith("deal {Damage=8} damage")
    for mask in ({"known": False}, {"applicable": False}):
        hidden = {"content_id": "CARD.BASH", "stats": {"Damage": dict(value=999, **mask), "VulnerablePower": 2}}
        assert text(hidden) == "deal {Damage=?} damage . apply {VulnerablePower=2} vulnerable ."


def test_variables_carry_the_entity_values():
    card = {"content_id": "CARD.BASH", "stats": {"Damage": 8, "VulnerablePower": 2}}
    assert text(card) == "deal {Damage=8} damage . apply {VulnerablePower=2} vulnerable ."
    assert text({"content_id": "CARD.BASH"}) == "deal {Damage=?} damage . apply {VulnerablePower=?} vulnerable ."


def test_conditional_text_follows_public_state():
    armaments = {"content_id": "CARD.ARMAMENTS", "stats": {"Block": 5}}
    assert text(dict(armaments, upgraded=False)).endswith("upgrade a card in your hand .")
    assert text(dict(armaments, upgraded=True)).endswith("upgrade all cards in your hand .")
    assert "next time you" in text({"content_id": "CARD.BUFFER", "stats": {"BufferPower": 1}})
    assert "next {BufferPower=2} times you" in text({"content_id": "CARD.BUFFER", "stats": {"BufferPower": 2}})
    strength = {"content_id": "POWER.STRENGTH_POWER", "entity_type": "power", "owner_ref": "player"}
    assert text(dict(strength, stacks=3)) == "increases attack damage by {Amount=3} ."
    assert text(dict(strength, stacks=-2)) == "decreases attack damage by {Amount=2} ."
    poison = {"content_id": "POWER.POISON_POWER", "entity_type": "power", "stacks": 5}
    assert text(dict(poison, owner_ref="player")).startswith("at the start of your turn , lose")
    assert text(dict(poison, owner_ref="creature:1")).startswith("at the start of its turn , loses")
    science = {"content_id": "CARD.MAD_SCIENCE", "stats": {"Damage": 8, "Block": 8}}
    assert text(dict(science, card_type="Attack")).startswith("deal {Damage=8} damage")
    assert text(dict(science, card_type="Skill")).startswith("gain {Block=8} block")


def test_options_and_enchantments_have_text():
    assert text({"content_id": "NEOW.pages.INITIAL.options.SCROLL_BOXES"}) == \
        "choose 1 of 2 packs of cards to add to your deck ."
    assert text({"content_id": "SMITH"}).startswith("upgrade")
    strike = {"content_id": "CARD.STRIKE_IRONCLAD", "stats": {"Damage": 6},
              "enchantment": "ENCHANTMENT.ADROIT", "enchantment_amount": 3}
    assert text(strike) == "deal {Damage=6} damage . gain {Block=3} block ."
    assert rules.program({"content_id": "MONSTER.TOADPOLE"}) is None


def test_every_description_renders():
    tables = rules._tables()
    for category, (table, keys) in rules.CATEGORIES.items():
        for key in tables[table]:
            entry, _, kind = key.rpartition(".")
            if kind in keys:
                rules.program({"content_id": f"{category}.{entry}"})
    for table in ("events", "ancients"):
        for key in tables[table]:
            if ".options." in key and key.endswith(".description"):
                rules.program({"content_id": key[: -len(".description")]})


def test_displayed_values_survive_cleaning():
    relic = clean_entity({"entity_type": "relic", "content_id": "RELIC.WINGED_BOOTS",
                          "stats": {"Rooms": 2, "Internal": 7}, "counter": 2})
    assert relic["stats"] == {"Rooms": 2} and relic["counter"] == 2


def frame(card):
    entities = [{"ref": "player", "entity_type": "player", "hp": 70},
                dict(card, ref="card:0", entity_type="card", zone="hand")]
    candidates = [{"verb": verb, "decoder_slot_ref": verb, "candidate_ref": verb,
                   "source_refs": ["card:0"] if verb == "PLAY_CARD" else [], "target_refs": []}
                  for verb in ("PLAY_CARD", "END_TURN")]
    return {"public": {"phase": "combat", "entities": entities, "relations": [], "memory": []},
            "legal": {"candidates": candidates}}


def test_rule_text_replaces_an_opaque_program_and_reaches_the_model():
    opaque = {"kind": "effect", "op": "OPAQUE_RULE", "content_id": "CARD.BASH", "coverage": "opaque"}
    bash = {"content_id": "CARD.BASH", "semantic_program": opaque, "stats": {"Damage": 8, "VulnerablePower": 2}}
    weaker = dict(bash, stats={"Damage": 8, "VulnerablePower": 1})
    obs = observation(frame(bash))
    assert len(obs.effects[obs.refs["card:0"]].nodes) == 9
    structured = dict(bash, semantic_program={"kind": "effect", "op": "DAMAGE"})
    assert len(observation(frame(structured)).effects[1].nodes) == 1  # the engine's own program wins
    config = ModelConfig.tiny()
    vocabulary = Vocabulary.from_frames([frame(bash), frame(weaker)], config)
    assert vocabulary.encode("word=vulnerable") != 1 and not vocabulary.unregistered()
    torch.manual_seed(0)
    model = PolicyValue(config).eval()
    with torch.no_grad():
        a, b = (e.hidden[1] for e in model.encode([observation(frame(bash)), observation(frame(weaker))], vocabulary))
    assert (a - b).abs().max() > 1e-4
