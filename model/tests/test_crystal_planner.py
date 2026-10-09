"""Public geometry decisions, independent of actual hidden layouts and game RNG."""

import random
from copy import deepcopy

from model.crystal_rule import collection_candidate, plan


def board_frame(items=(), *, revealed=(), remaining=3):
    tiles = {}
    for kind, x, y, width, height in items:
        for dx in range(width):
            for dy in range(height):
                tiles[x + dx, y + dy] = kind
    opened = set(revealed)
    entities = [dict(entity_type="event_state", content_id="CRYSTAL_SPHERE", remaining_steps=remaining)]
    entities += [dict(entity_type="tool", ref=f"tool:{tool}", content_id=tool) for tool in ("Small", "Big")]
    candidates = []
    for x in range(11):
        for y in range(11):
            known = (x, y) in opened
            entity = dict(entity_type="board_cell", ref=f"cell:{x}:{y}", x=x, y=y, revealed=known, known=known)
            if known:
                entity["content_id"] = tiles.get((x, y), "empty")
            entities.append(entity)
            if not known:
                for tool in ("Small", "Big"):
                    candidates.append(dict(verb="DIVINE_CELL", candidate_ref=f"{tool}:{x}:{y}",
                        decoder_slot_ref=f"{tool}:{x}:{y}", source_refs=[f"tool:{tool}"],
                        target_refs=[entity["ref"]]))
    return dict(public=dict(phase="crystal_sphere", entities=entities, memory=[], relations=[]),
                legal=dict(candidates=candidates))


def opened_by(candidate):
    tool, x, y = candidate["candidate_ref"].split(":")
    x, y = int(x), int(y)
    radius = int(tool == "Big")
    return {(i, j) for i in range(max(0, x - radius), min(11, x + radius + 1))
            for j in range(max(0, y - radius), min(11, y + radius + 1))}


def corners():
    return {(x, y) for x in range(11) for y in range(11)
            if min(x, 10 - x) + min(y, 10 - y) <= 2}


def test_payment_and_reward_choices_are_not_planner_actions():
    for phase in ("event", "rewards"):
        frame = dict(public=dict(phase=phase, entities=[dict(entity_type="event", content_id="CRYSTAL_SPHERE")]),
                     legal=dict(candidates=[dict(verb="CHOOSE_EVENT_OPTION", candidate_ref="pay")]))
        assert collection_candidate(frame) is None


def test_hidden_content_seed_and_packing_order_do_not_change_decision():
    frame = board_frame(revealed=corners(), remaining=6)
    first = plan(frame, samples=4, beam_width=3)
    other = deepcopy(frame)
    other["seed"] = "a different game seed"
    other["public"]["private_rng"] = 123456
    for entity in other["public"]["entities"]:
        if entity.get("revealed") is False:
            entity.update(content_id="CrystalSphereCurse", private_item_extent=[0, 0, 11, 11])
    random.Random(1).shuffle(other["public"]["entities"])
    random.Random(2).shuffle(other["legal"]["candidates"])
    second = plan(other, samples=4, beam_width=3)
    assert first.candidate == second.candidate
    assert first.diagnostics() == second.diagnostics()
    assert first.hypotheses == 4 and not first.fallback


def test_small_tool_can_finish_gold_without_completing_adjacent_curse():
    items = [("CrystalSphereGold", 4, 5, 2, 1), ("CrystalSphereCurse", 6, 4, 2, 2)]
    opened = {(x, y) for x in range(11) for y in range(11)} - {(5, 5), (6, 5)}
    result = plan(board_frame(items, revealed=opened, remaining=1), samples=8)
    assert result.candidate["candidate_ref"] == "Small:5:5"
    assert not result.fallback


def test_four_click_plan_can_finish_a_four_by_four_relic():
    item = ("CrystalSphereRelic", 3, 3, 4, 4)
    relic = {(x, y) for x in range(3, 7) for y in range(3, 7)}
    opened = {(x, y) for x in range(11) for y in range(11)} - relic
    for remaining in (4, 3, 2, 1):
        frame = board_frame([item], revealed=opened, remaining=remaining)
        chosen = plan(frame, samples=4).candidate
        assert chosen in frame["legal"]["candidates"]
        opened |= opened_by(chosen)
    assert relic <= opened


def test_bounded_constraint_failure_has_a_reported_public_fallback():
    items = [("CrystalSphereCurse", 3, 3, 2, 2), ("CrystalSphereGold", 7, 7, 1, 1)]
    frame = board_frame(items, revealed=corners() | {(3, 3), (7, 7)}, remaining=2)
    result = plan(frame, samples=1, search_budget=1, beam_width=2)
    assert result.fallback and result.hypotheses == 0
    assert result.candidate in frame["legal"]["candidates"]
