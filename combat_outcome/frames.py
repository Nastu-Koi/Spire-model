"""Synthetic public frames: one (entry state, encounter) as a model input.

The public player / deck card / relic / potion / player power entities of a
battle's first decision, plus an `encounter` entity, with a single
PREDICT_COMBAT action bound to the encounter. The action token is where the
model reads its prediction.
"""

from copy import deepcopy

from .features import deck_summary

COUNTS = {"deck_size", "upgraded_total", "damage_total", "block_total", "relic_count", "max_hp"}


def summary_entity(entities):
    """Deck aggregates as one entity: per-card details stay in the card entities, but
    small distributed changes (one upgrade among 25 cards) are explicit totals here."""
    deck = [e for e in entities if e.get("entity_type") == "card"]
    relics = [e for e in entities if e.get("entity_type") == "relic"]
    player = next((e for e in entities if e.get("entity_type") == "player"), {})
    stats = deck_summary(dict(deck=deck, relics=relics, max_hp=player.get("max_hp") or 0))
    n = max(1, len(deck))
    # Per-card averages become deck totals so the numeric encoder sees counts, not tiny fractions.
    totals = {k: v if k in COUNTS else round(v * n, 3) for k, v in stats.items()}
    return {"entity_type": "deck_summary", "ref": "summary:0", "stats": totals}


def frame_for(entities, encounter, summary=False):
    entities = [dict(e) for e in entities] + [{"entity_type": "encounter", "content_id": encounter,
                                               "ref": "encounter:0"}]
    if summary:
        entities.append(summary_entity(entities))
    action = {"verb": "PREDICT_COMBAT", "decoder_slot_ref": "predict", "source_refs": ["encounter:0"],
              "target_refs": [], "candidate_ref": "c0"}
    return {"public": {"phase": "combat", "entities": entities, "memory": [], "relations": []},
            "legal": {"candidates": [action]}}


def input_entities(frame):
    """What a fight is predicted from: the state it is entered with, never the fight itself."""
    return deepcopy([e for e in frame["public"]["entities"]
                     if e.get("entity_type") in {"player", "relic", "potion"}
                     or (e.get("entity_type") == "card" and e.get("zone") == "deck")
                     or (e.get("entity_type") == "power" and e.get("owner_ref") == "player")])


def summary_info(row, entities):
    deck = [e for e in entities if e.get("entity_type") == "card"]
    relics = [e["content_id"] for e in entities if e.get("entity_type") == "relic"]
    return dict(deck=deck, relics=relics, character=row["character"], max_hp=row["max_hp"], act=row["act"])
