"""Rule text: description templates rendered with public values into local programs."""

from copy import deepcopy

import pytest
import torch

from model import rules
from model.config import ModelConfig
from model.model import PolicyValue
from model.representation import Vocabulary, clean_entity, observation
from model.batch import collate


def text(entity):
    words = []
    for row in rules.program(entity).nodes[1:]:
        fields = {f.name: f for f in row}
        if "variable" in fields and "content_id" in fields:
            words.append("{%s=%s}" % (fields["variable"].symbol.split("=")[1], fields["content_id"].symbol.split("=")[1]))
        elif "variable" in fields:
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


def test_mad_science_rider_changes_text_and_packed_input():
    science = {"content_id": "CARD.MAD_SCIENCE", "card_type": "Attack",
               "stats": {"Damage": 8, "Block": 8, "SappingWeak": 2, "SappingVulnerable": 3,
                         "EnergizedEnergy": 1, "ViolenceHits": 4}}
    sapping, energized = (dict(science, rider_effect=value) for value in ("Sapping", "Energized"))
    assert "weak" in text(sapping) and "vulnerable" in text(sapping) and "energy" not in text(sapping)
    assert "energy" in text(energized) and "weak" not in text(energized)
    assert "{ViolenceHits=4} times" in text(dict(science, rider_effect="Violence"))
    assert rules._values(dict(science, rider_effect="None"))["hasrider"] is False
    assert rules._values(science)["hasrider"] is None
    masked = dict(sapping, known_masks={"rider_effect": False})
    assert rules._values(masked)["hasrider"] is None and "weak" not in text(masked)
    config = ModelConfig.tiny()
    vocabulary = Vocabulary.from_frames([], config)
    for value in ("None", "Sapping", "Violence", "Choking", "Energized", "Wisdom", "Chaos", "Expertise", "Curious", "Improvement"):
        assert vocabulary.encode("rider_effect=" + value) != 1
    a, b = (collate([observation(frame(card))], vocabulary) for card in (sapping, energized))
    assert not torch.equal(a.tokens.ids, b.tokens.ids)


def test_dynamic_relic_titles_keep_distinct_content_and_unknown_numbers():
    names = ["BING_BONG", "DAUGHTER_OF_THE_WIND", "MR_STRUGGLES"]
    entities = [{"entity_type": "event_option", "ref": f"option:{i}", "content_id": name + ".title",
                 "semantic_program": {"kind": "effect", "op": "OPAQUE_RULE", "content_id": name + ".title"}}
                for i, name in enumerate(names)]
    decision = {"public": {"phase": "event", "entities": entities}, "legal": {"candidates": [
        {"verb": "CHOOSE_EVENT_OPTION", "candidate_ref": f"c{i}", "decoder_slot_ref": f"option:{i}",
         "source_refs": [f"option:{i}"]} for i in range(3)]}}
    vocabulary = Vocabulary.from_frames([decision], ModelConfig.tiny())
    obs = observation(decision)
    packed = collate([obs], vocabulary)
    rows = [obs.refs[f"option:{i}"] for i in range(3)]
    assert not torch.equal(packed.tokens.ids[rows[0]], packed.tokens.ids[rows[1]])
    for entity, name in zip(entities, names):
        assert rules.canonical_content_id(name + ".title") == "RELIC." + name
        assert vocabulary.encode("content_id=RELIC." + name) != 1
        assert rules.program(entity) is not None
    assert "{Block=?}" in text(entities[1])
    assert "{Block=2}" in text(dict(entities[1], stats={"Block": 2}))
    for unchanged in ("NONEXISTENT.title", "RELIC.BING_BONG", "DOLL_ROOM.pages.INITIAL.options.EXAMINE"):
        assert rules.canonical_content_id(unchanged) == unchanged


def test_reward_alternatives_and_crystal_items_have_their_own_symbols():
    from model.crystal_rule import SHAPES

    alternatives = rules.reward_alternatives()
    assert {"REROLL", "SACRIFICE"} <= alternatives
    names = alternatives | {shape[0] for shape in SHAPES} | {"empty"}
    for vocabulary in (Vocabulary.from_frames([], ModelConfig.tiny()), Vocabulary(["z=existing"], 8192).expanded()):
        ids = {vocabulary.encode("content_id=" + name) for name in names}
        assert 1 not in ids and len(ids) == len(names)
        assert not vocabulary.unregistered()
    # The reveal program a Crystal Sphere tool carries.
    vocabulary = Vocabulary.from_frames([], ModelConfig.tiny())
    for symbol in ("op=REVEAL", "shape=cell", "shape=square", "radius=<number>", "clip_to_board=True"):
        assert vocabulary.encode(symbol) != 1
    assert {"shape", "radius", "clip_to_board"} <= vocabulary.fields.lookup.keys()


def test_vocabulary_expansion_preserves_ids_and_reports_missing_symbols():
    vocabulary = Vocabulary(["content_id=RELIC.BING_BONG", "z=existing"], 8192)
    expanded = vocabulary.expanded()
    assert all(expanded.encode(name) == index for name, index in vocabulary.lookup.items())
    assert expanded.encode("rider_effect=Sapping") != 1
    restored = Vocabulary.from_state(expanded.state(), ModelConfig(vocabulary_size=8192, field_size=512))
    assert restored.lookup == expanded.lookup
    assert restored.encode("content_id=UNKNOWN_CONTENT") == 1
    assert "symbol:content_id=UNKNOWN_CONTENT" in restored.unregistered()


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


def event_frame(max_hp=5, damage=20):
    option = "ABYSSAL_BATHS.pages.INITIAL.options.IMMERSE"
    entities = [{"entity_type": "event_option", "ref": f"option:{i}", "content_id": option}
                for i in range(2)]
    for i, values in enumerate(((max_hp, damage), (2, 7))):
        entities.extend({"entity_type": "displayed_variable", "owner_ref": f"option:{i}",
                         "content_id": name, "amount": value, "known": True}
                        for name, value in zip(("MaxHp", "Damage"), values))
    return {"public": {"phase": "event", "entities": entities},
            "legal": {"candidates": [{"verb": "CHOOSE_EVENT_OPTION", "candidate_ref": f"c{i}",
                                      "decoder_slot_ref": f"option:{i}", "source_refs": [f"option:{i}"]}
                                     for i in range(2)]}}


def displayed_values(obs, ref):
    nodes = [{f.name: f for f in row} for row in obs.effects[obs.refs[ref]].nodes]
    return {row["variable"].symbol: row["value"].number for row in nodes if "variable" in row}


def naming_frame(enchantment, amount=2):
    option = "SELF_HELP_BOOK.pages.INITIAL.options.READ_THE_BACK"
    entities = [{"entity_type": "event_option", "ref": "option:0", "content_id": option},
                {"entity_type": "displayed_variable", "owner_ref": "option:0", "content_id": "Enchantment1Amount",
                 "amount": amount, "known": True}]
    if enchantment:
        entities.append({"entity_type": "displayed_variable", "owner_ref": "option:0", "content_id": "Enchantment1",
                         "names": enchantment, "known": True})
    return {"public": {"phase": "event", "entities": entities},
            "legal": {"candidates": [{"verb": "CHOOSE_EVENT_OPTION", "candidate_ref": "c0",
                                      "decoder_slot_ref": "option:0", "source_refs": ["option:0"]}]}}


def test_a_name_an_option_displays_is_the_content_it_names():
    original = naming_frame("ENCHANTMENT.SHARP")
    unchanged = deepcopy(original)
    obs = observation(original)
    assert original == unchanged
    nodes = [{f.name: f.symbol for f in row} for row in obs.effects[obs.refs["option:0"]].nodes]
    # The option reads "enchant with {Enchantment1} {Enchantment1Amount}": the name is a content, the amount a number.
    assert {"variable": "variable=Enchantment1", "content_id": "content_id=ENCHANTMENT.SHARP"} in nodes
    assert displayed_values(observation(naming_frame(None)), "option:0")["variable=Enchantment1"][1] == 0
    # The binding is for the text only: the option's own token is the same whatever it names.
    option = lambda o: sorted(f.symbol for f in o.tokens[o.refs["option:0"]])
    assert option(obs) == option(observation(naming_frame("ENCHANTMENT.NIMBLE")))

    catalog = {"public": {"phase": "catalog", "entities": [
        {"entity_type": "enchantment", "content_id": "ENCHANTMENT." + name} for name in ("SHARP", "NIMBLE")]},
        "legal": {"candidates": []}}
    config = ModelConfig.tiny()
    vocabulary = Vocabulary.from_frames([catalog, naming_frame(None)], config)
    # Every content a name can stand for is registered with the static catalog, named in a frame or not.
    assert min(vocabulary.encode("names=ENCHANTMENT." + name) for name in ("SHARP", "NIMBLE")) > 1
    frames = [naming_frame("ENCHANTMENT.SHARP"), naming_frame("ENCHANTMENT.NIMBLE")]
    torch.manual_seed(0)
    model = PolicyValue(config).eval()
    with torch.no_grad():
        a, b = model.encode([observation(state) for state in frames], vocabulary)
    assert (a.actions - b.actions).abs().max() > 1e-4 and not vocabulary.unregistered()


def test_displayed_variables_bind_to_their_option_and_reach_the_model():
    original = event_frame()
    unchanged = deepcopy(original)
    obs = observation(original)
    assert original == unchanged  # Binding must not modify the recorded frame.
    assert displayed_values(obs, "option:0") == {
        "variable=MaxHp": (5, 1, 1, 1), "variable=Damage": (20, 1, 1, 1)}
    assert displayed_values(obs, "option:1") == {
        "variable=MaxHp": (2, 1, 1, 1), "variable=Damage": (7, 1, 1, 1)}

    # Vocabulary initialization sees the option types, not a running event's values.
    catalog = event_frame()
    catalog["public"]["entities"] = catalog["public"]["entities"][:2]
    config = ModelConfig.tiny()
    vocabulary = Vocabulary.from_frames([catalog], config)
    assert vocabulary.encode("content_id=MaxHp") != vocabulary.encode("content_id=Damage")
    assert min(vocabulary.encode("content_id=" + name) for name in ("MaxHp", "Damage")) > 1
    swapped = observation(event_frame(20, 5))
    reordered = deepcopy(original)
    reordered["public"]["entities"].reverse()
    torch.manual_seed(0)
    model = PolicyValue(config).eval()
    with torch.no_grad():
        a, b, c = model.encode([obs, swapped, observation(reordered)], vocabulary)
    assert (a.actions - b.actions).abs().max() > 1e-4
    torch.testing.assert_close(a.actions, c.actions)


@pytest.mark.parametrize("mask", [
    {"amount": {"value": 999, "known": False}},
    {"amount": {"value": 999, "applicable": False}},
    {"amount": 999, "known_masks": {"amount": False}},
    {"amount": 999, "applicable_masks": {"amount": False}},
    {"amount": 999, "known": False},
    {"amount": 999, "revealed": False},
    {"amount": 999, "applicable": False},
])
def test_masked_displayed_variables_stay_unknown_in_option_text(mask):
    state = event_frame()
    state["public"]["entities"][3].update(mask)
    values = displayed_values(observation(state), "option:0")
    assert values["variable=MaxHp"] == (5, 1, 1, 1)
    assert values["variable=Damage"][0] == 0 and values["variable=Damage"][1] == 0


def test_rest_option_text_uses_its_displayed_healing():
    state = {"public": {"phase": "rest_site", "entities": [
        {"entity_type": "rest_option", "ref": "rest:0", "content_id": "HEAL"},
        {"entity_type": "displayed_variable", "owner_ref": "rest:0", "content_id": "Heal", "amount": 24}]},
        "legal": {"candidates": [{"verb": "CHOOSE_REST_OPTION", "decoder_slot_ref": "heal",
                                  "candidate_ref": "c0", "source_refs": ["rest:0"]}]}}
    assert displayed_values(observation(state), "rest:0")["variable=Heal"] == (24, 1, 1, 1)
