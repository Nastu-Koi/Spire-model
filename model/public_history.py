"""Versioned public facts. A missing prefix never becomes a fresh-run prior."""

import math

from .protocol import ProtocolError

HISTORY_VERSION = "public-history-v1"
HISTORY_KINDS = {"history_probability", "history_rule", "previous_intent", "known_draw_position"}
ROOM_TYPES = ("Monster", "Elite", "Treasure", "Shop", "Event")
PROBABILITY_SCOPES = ("Monster", "Elite", "Boss", "Event") + tuple(
    f"{condition}:{room}" for condition in ("shop_allowed", "shop_blocked") for room in ROOM_TYPES
)


def require_history_version(version):
    if version != HISTORY_VERSION:
        raise ProtocolError(
            "Public history input version mismatch; regenerate on-policy data and "
            "start a new checkpoint from BC (legacy BC keeps missing history unknown)"
        )


def unknown_memory(entities):
    result = [{"entity_type": "history_rule", "content_id": "question_previous_shop",
               "enabled": {"value": None, "known": False, "applicable": True}}]
    for kind, scopes in (
        ("potion_drop", PROBABILITY_SCOPES[:4]),
        ("question_room", PROBABILITY_SCOPES[4:]),
    ):
        result.extend(
            {"entity_type": "history_probability", "content_id": kind, "scope": scope,
             "probability": {"value": None, "known": False, "applicable": True}}
            for scope in scopes
        )
    result.extend(
        {"entity_type": "previous_intent", "owner_ref": e["ref"], "known": False, "applicable": True}
        for e in entities if e.get("entity_type") == "enemy" and e.get("ref")
    )
    return result


def recorded_memory(state, refs, entities):
    """Translate recorder object identities to frame-local, reference-only handles."""
    history = state.get("public_history")
    if history is None:
        return unknown_memory(entities)
    require_history_version(history.get("version"))
    result = []
    for raw in history["memory"]:
        kind = raw.get("entity_type")
        if kind not in HISTORY_KINDS:
            raise ProtocolError("Unrecognized public history fact")
        row = dict(raw)
        if "owner_ref" in row:
            try:
                row["owner_ref"] = refs[int(row["owner_ref"])]
            except (KeyError, ValueError, TypeError):
                raise ProtocolError("Public history refers to an absent public entity") from None
        if kind == "known_draw_position":
            if row.get("known") is not True or type(row.get("position")) is not int or row["position"] < 0:
                raise ProtocolError("Invalid known draw position")
            owner = next(e for e in entities if e.get("ref") == row["owner_ref"])
            if owner.get("zone") != "draw_pile":
                raise ProtocolError("Known position does not belong to the draw pile")
            if row["position"] >= sum(e.get("entity_type") == "card" and e.get("zone") == "draw_pile" for e in entities):
                raise ProtocolError("Known position exceeds the complete draw pile")
        if kind == "history_probability":
            probability = row.get("probability") or {}
            if type(probability.get("known")) is not bool or probability.get("applicable") is not True:
                raise ProtocolError("Probability requires explicit known/applicable masks")
            value = probability.get("value")
            if probability["known"] and (type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 1):
                raise ProtocolError("Known probability must be between zero and one")
            if not probability["known"] and value is not None:
                raise ProtocolError("Unknown probability must not contain a guessed value")
        result.append(row)
    expected = {("potion_drop", s) for s in PROBABILITY_SCOPES[:4]} | {
        ("question_room", s) for s in PROBABILITY_SCOPES[4:]}
    actual = [(e.get("content_id"), e.get("scope")) for e in result if e["entity_type"] == "history_probability"]
    if set(actual) != expected or len(actual) != len(expected):
        raise ProtocolError("Incomplete or repeated public probability facts")
    if sum(e["entity_type"] == "history_rule" and e.get("content_id") == "question_previous_shop" for e in result) != 1:
        raise ProtocolError("Missing previous-room shop eligibility fact")
    return result


def draw_relations(memory):
    known = sorted((e for e in memory if e.get("entity_type") == "known_draw_position"
                    and e.get("known") is True), key=lambda e: e["position"])
    if len({e["position"] for e in known}) != len(known) or len({e["owner_ref"] for e in known}) != len(known):
        raise ProtocolError("Conflicting known draw positions")
    return [{"source": a["owner_ref"], "target": b["owner_ref"], "role": "known_draw_before"}
            for a, b in zip(known, known[1:])]
