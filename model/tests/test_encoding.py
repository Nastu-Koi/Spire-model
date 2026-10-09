"""Observation and encoder properties: field binding, map compression, relations, batching."""

import copy

import torch

from model.config import ModelConfig
from model.model import PolicyValue
from model.batch import Fields
from model.representation import Vocabulary, fields_of, observation


def action(slot, verb="PLAY_CARD", source=(), target=()):
    return {"verb": verb, "decoder_slot_ref": slot, "candidate_ref": slot,
            "source_refs": list(source), "target_refs": list(target)}


def frame(entities, candidates, relations=(), phase="combat"):
    return {"public": {"phase": phase, "entities": entities, "relations": list(relations), "memory": []},
            "legal": {"candidates": candidates}}


def node(col, floor, content="Monster", **flags):
    return dict({"ref": f"map:{col}:{floor}", "entity_type": "map_node", "content_id": content,
                 "floor": floor, "col": col, "current": False, "visited": False}, **flags)


def edge(source, target):
    return {"source": source, "target": target, "role": "map_edge"}


def map_frame(change=None):
    """Floor 0 start (current), two branches which rejoin at the boss, and a dead branch.

        0:0 -> 0:1 -> 0:2 -> boss        1:1 -> 1:2 -> boss        2:1 (not reachable from 0:0)
        0:0 -> 1:1
    """
    nodes = [node(0, 0, "Ancient", current=True, visited=True), node(0, 1), node(1, 1, "Elite"), node(2, 1, "Shop"),
             node(0, 2, "RestSite"), node(1, 2, "Treasure"), node(0, 3, "Boss")]
    for item in nodes:
        if change and item["ref"] == change[0]:
            item["content_id"] = change[1]
    edges = [edge("map:0:0", "map:0:1"), edge("map:0:0", "map:1:1"), edge("map:0:1", "map:0:2"),
             edge("map:1:1", "map:1:2"), edge("map:2:1", "map:1:2"), edge("map:0:2", "map:0:3"),
             edge("map:1:2", "map:0:3")]
    player = {"ref": "player", "entity_type": "player", "hp": 60, "max_hp": 80}
    moves = [action("move:" + ref, "MOVE_TO_NODE", source=[ref]) for ref in ("map:0:1", "map:1:1")]
    return frame([player] + nodes, moves, edges, phase="map")


def build(frames):
    torch.manual_seed(0)
    config = ModelConfig.tiny()
    return PolicyValue(config).eval(), Vocabulary.from_frames(frames, config)


def row_vocabulary(rows):
    return Vocabulary({f.symbol for row in rows for f in row}, 4096,
                      fields={f.name for row in rows for f in row})


def logits(model, vocabulary, frames):
    observations = [observation(f) for f in frames]
    with torch.no_grad():
        encoded = model.encode(observations, vocabulary)
        masks = [torch.ones(len(o.slot_refs), dtype=torch.bool) for o in observations]
        return [out.logits for out in model.decode_batch(encoded, masks)]


def test_numbers_are_bound_to_their_fields():
    torch.manual_seed(0)
    encoder = PolicyValue(ModelConfig.tiny()).eval().encoder
    rows = [fields_of(x) for x in ({"entity_type": "enemy", "hp": 12, "block": 40},
                                   {"entity_type": "enemy", "hp": 40, "block": 12},
                                   {"entity_type": "enemy", "hp": 12, "block": 40})]
    with torch.no_grad():
        a, swapped, same = encoder.fields(Fields.pack(rows, row_vocabulary(rows)))
    assert torch.allclose(a, same, atol=1e-6)
    assert (a - swapped).abs().max() > 1e-3


def test_adjacent_large_numbers_stay_distinct():
    torch.manual_seed(0)
    encoder = PolicyValue(ModelConfig.tiny()).eval().encoder
    rows = [fields_of({"entity_type": "player", "gold": gold}) for gold in (998, 999)]
    with torch.no_grad():
        a, b = encoder.fields(Fields.pack(rows, row_vocabulary(rows)))
    assert (a - b).abs().max() > 1e-3


def test_map_keeps_the_current_node_and_its_frontier():
    obs = observation(map_frame())
    assert {ref for ref in obs.refs if ref.startswith("map:")} == {"map:0:0", "map:0:1", "map:1:1"}
    assert len(obs.tokens) == 1 + 1 + 3 + 15 + 2  # global, player, map, unknown history, moves
    sizes = {ref: len(obs.effects[index].nodes) for ref, index in obs.refs.items() if ref.startswith("map:")}
    # Each frontier node carries itself, its branch and the boss; the current node carries no map.
    assert sizes == {"map:0:0": 1, "map:0:1": 3, "map:1:1": 3}
    roles = {(i, j): role for i, j, role in obs.edges}
    current, frontier = obs.refs["map:0:0"], obs.refs["map:0:1"]
    assert roles[current, frontier] == "map_edge" and roles[frontier, current] == "reverse_map_edge"


def test_frontier_program_describes_the_reachable_map():
    obs = observation(map_frame())
    program = obs.effects[obs.refs["map:1:1"]]
    contents = [next(f.symbol for f in row if f.name == "content_id") for row in program.nodes]
    assert contents[0] == "content_id=Elite" and sorted(contents[1:]) == ["content_id=Boss", "content_id=Treasure"]
    depths = sorted(next(f.number[0] for f in row if f.name == "depth") for row in program.nodes)
    assert depths == [0, 1, 2]
    relation = {(i, j): role for i, j, role in program.edges}
    boss = contents.index("content_id=Boss")
    treasure = contents.index("content_id=Treasure")
    assert relation[0, treasure] == "map_forward_direct" and relation[0, boss] == "map_forward_indirect"
    assert relation[boss, 0] == "map_reverse_indirect" and relation[0, 0] == "map_self"


def test_only_the_reachable_map_reaches_the_model():
    base, dead, ahead = map_frame(), map_frame(("map:2:1", "Elite")), map_frame(("map:1:2", "Elite"))
    assert observation(base).digest == observation(dead).digest
    assert observation(base).digest != observation(ahead).digest
    model, vocabulary = build([base, dead, ahead])
    a, b, c = logits(model, vocabulary, [base, dead, ahead])
    assert torch.equal(a, b)
    # The changed room lies behind map:1:1 only, and that move's score follows it.
    assert (a - c).abs().max() > 1e-4


def test_the_boss_of_the_act_reaches_the_model():
    def named(boss):
        result = map_frame()
        next(e for e in result["public"]["entities"] if e.get("content_id") == "Boss")["encounter"] = boss
        return result
    unnamed, fysh, giant = map_frame(), named("ENCOUNTER.SOUL_FYSH_BOSS"), named("ENCOUNTER.WATERFALL_GIANT_BOSS")
    assert len({observation(f).digest for f in (unnamed, fysh, giant)}) == 3
    # Every branch ends at the boss, so every frontier node carries who it is.
    obs = observation(fysh)
    for ref in ("map:0:1", "map:1:1"):
        symbols = {f.symbol for row in obs.effects[obs.refs[ref]].nodes for f in row}
        assert "encounter=ENCOUNTER.SOUL_FYSH_BOSS" in symbols
    # The static catalog registers every encounter, also the ones no frame has shown yet.
    catalog = frame([{"entity_type": "encounter", "content_id": "ENCOUNTER." + name}
                     for name in ("SOUL_FYSH_BOSS", "WATERFALL_GIANT_BOSS")], [action("a")])
    model, vocabulary = build([catalog, unnamed])
    known = {vocabulary.encode("encounter=ENCOUNTER." + name) for name in ("SOUL_FYSH_BOSS", "WATERFALL_GIANT_BOSS")}
    assert len(known) == 2 and 1 not in known  # 1 is <unknown>
    a, b = logits(model, vocabulary, [fysh, giant])
    assert (a - b).abs().max() > 1e-4 and not vocabulary.unregistered()


def test_a_map_node_an_action_refers_to_stays_an_entity():
    jump = map_frame()
    jump["legal"]["candidates"].append(action("move:map:2:1", "MOVE_TO_NODE", source=["map:2:1"]))
    obs = observation(jump)
    assert "map:2:1" in obs.refs and len(obs.effects[obs.refs["map:2:1"]].nodes) == 3


def test_entity_order_is_not_an_input():
    base = map_frame()
    shuffled = copy.deepcopy(base)
    shuffled["public"]["entities"].reverse()
    shuffled["public"]["relations"].reverse()
    model, vocabulary = build([base])
    a, b = logits(model, vocabulary, [base, shuffled])
    assert torch.allclose(a, b, atol=1e-5)


def duel(owner):
    enemies = [{"ref": ref, "entity_type": "enemy", "content_id": "MONSTER.CULTIST", "hp": 40, "block": 0}
               for ref in ("enemy:0", "enemy:1")]
    power = {"ref": "power:0", "entity_type": "power", "content_id": "POWER.VULNERABLE", "stacks": 2,
             "owner_ref": owner}
    card = {"ref": "card:0", "entity_type": "card", "content_id": "CARD.STRIKE", "zone": "hand", "cost": 1}
    attacks = [action(f"play:{ref}", source=["card:0"], target=[ref]) for ref in ("enemy:0", "enemy:1")]
    return frame(enemies + [power, card], attacks)


def test_identical_entities_differ_through_their_relations():
    """Two enemies with equal fields: only the power's owner tells the attacks apart."""
    first, second = duel("enemy:0"), duel("enemy:1")
    model, vocabulary = build([first, second])
    with torch.no_grad():
        attacks = model.encode([observation(first)], vocabulary)[0].actions
    assert (attacks[0] - attacks[1]).abs().max() > 1e-4
    a, b = logits(model, vocabulary, [first, second])
    # Moving the power to the other enemy mirrors the two scores.
    assert torch.allclose(a, b.flip(0), atol=1e-5)


def test_batched_encoding_matches_single_encoding():
    frames = [map_frame(), duel("enemy:0"), map_frame(("map:1:2", "Elite"))]
    model, vocabulary = build(frames)
    together = logits(model, vocabulary, frames)
    for f, expected in zip(frames, together):
        assert torch.allclose(logits(model, vocabulary, [f])[0], expected, atol=1e-5)


def test_value_head_starts_at_zero():
    model, vocabulary = build([duel("enemy:0")])
    obs = observation(duel("enemy:0"))
    with torch.no_grad():
        encoded = model.encode([obs], vocabulary)
        out = model.decode(encoded[0], torch.ones(2, dtype=torch.bool))
    assert float(out.value) == 0.0


def test_free_travel_reaches_every_node_of_the_next_floor():
    walk, fly = map_frame(), map_frame()
    fly["public"]["entities"][0]["free_travel"] = True
    assert "map:2:1" not in observation(walk).refs
    obs = observation(fly)
    assert {ref for ref in obs.refs if ref.startswith("map:")} == {"map:0:0", "map:0:1", "map:1:1", "map:2:1"}
    # From map:0:1 both rooms of the next floor are one move away, one of them only by flight.
    program = obs.effects[obs.refs["map:0:1"]]
    contents = [next(f.symbol for f in row if f.name == "content_id") for row in program.nodes]
    assert sorted(contents[1:]) == ["content_id=Boss", "content_id=RestSite", "content_id=Treasure"]
    relation = {(i, j): role for i, j, role in program.edges}
    rest, treasure = contents.index("content_id=RestSite"), contents.index("content_id=Treasure")
    assert relation[0, rest] == "map_forward_direct"
    assert relation[0, treasure] == "map_free_direct" and relation[treasure, 0] == "map_free_reverse"
