"""Validate transport contracts, never infer game legality in Python."""

import hashlib
import json
from copy import deepcopy
from typing import Any

CHARACTERS = ("Ironclad", "Silent", "Defect", "Regent", "Necrobinder")
SCHEMA = "spire-trajectory-v1"


class ProtocolError(RuntimeError):
    pass


def fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def validate_frame(frame: dict, *, allow_prototype: bool = False) -> dict:
    if frame.get("type") != "decision_frame":
        raise ProtocolError("Expected decision_frame")
    contract = frame.get("contract", {})
    ascension = contract.get("fixed_ascension")
    if type(ascension) is not int or not 0 <= ascension <= 10:
        raise ProtocolError("Only verified A0-A10 runs are supported")
    if contract.get("training_ready") is not True and not allow_prototype:
        raise ProtocolError(
            "Engine contract is not training_ready; use inspect for prototype diagnostics"
        )
    for key in ("adapter_version", "observation_schema", "action_schema"):
        if not contract.get(key):
            raise ProtocolError(f"Missing contract field: {key}")
    boundary = frame.get("boundary")
    if boundary not in {"decision", "terminal", "waiting", "error"}:
        raise ProtocolError("Unknown boundary")
    if boundary == "error":
        raise ProtocolError(frame.get("error", {}).get("code", "engine_error"))
    if (
        boundary == "terminal"
        and type((frame.get("public", {}).get("outcome") or {}).get("victory"))
        is not bool
    ):
        raise ProtocolError("Terminal frame needs an explicit victory/death outcome")
    if boundary != "decision":
        return frame
    routing = frame.get("routing", {})
    if not routing.get("decision_id") or type(routing.get("state_version")) is not int:
        raise ProtocolError("Missing decision version")
    if (
        routing.get("selection_id")
        and type(routing.get("selection_revision")) is not int
    ):
        raise ProtocolError("Missing selection revision")
    candidates = frame.get("legal", {}).get("candidates", [])
    if not candidates:
        raise ProtocolError("Decision has no legal candidates")
    refs = [c.get("candidate_ref") for c in candidates]
    slots = [c.get("decoder_slot_ref") for c in candidates]
    if any(not isinstance(x, str) or not x for x in refs + slots):
        raise ProtocolError("Every candidate needs a reference and decoder slot")
    if len(set(refs)) != len(refs) or len(set(slots)) != len(slots):
        raise ProtocolError("Duplicate candidate or ambiguous decoder slot")
    public = frame.get("public", {})
    if not isinstance(public.get("entities"), list) or not public.get("phase"):
        raise ProtocolError("Missing public observation")
    context = public.get("selection_context") or {}
    if context.get("mode") == "buffered" and not routing.get("selection_id"):
        raise ProtocolError("Buffered selection requires a stable selection identity")
    if context.get("mode") == "buffered":
        selected = context.get("selected_refs", [])
        if context.get("selected_count") != len(selected):
            raise ProtocolError("Selection count disagrees with public prefix")
        if context.get("repetition_allowed") is False and len(set(selected)) != len(
            selected
        ):
            raise ProtocolError("Repeated reference in non-repeating selection")
        controls = {
            "can_finish": "FINISH_SELECTION",
            "can_skip": "SKIP",
            "can_cancel": "CANCEL",
        }
        verbs = {c.get("verb") for c in candidates}
        for field, verb in controls.items():
            if type(context.get(field)) is bool and context[field] != (verb in verbs):
                raise ProtocolError(
                    f"Selection control {field} disagrees with engine candidates"
                )
    bank = public.get("decoder_bank") or candidates
    bank_refs = [c.get("decoder_slot_ref") for c in bank]
    if len(set(bank_refs)) != len(bank_refs) or not set(slots) <= set(bank_refs):
        raise ProtocolError(
            "Decoder bank does not uniquely cover every legal candidate"
        )
    by_slot = {c["decoder_slot_ref"]: c for c in bank}
    for candidate in candidates:
        if action_semantics(candidate) != action_semantics(
            by_slot[candidate["decoder_slot_ref"]]
        ):
            raise ProtocolError("Candidate semantics differ from decoder bank")
    return frame


def action_semantics(candidate):
    return {k: v for k, v in candidate.items() if k != "candidate_ref"}


def execution_command(frame: dict, candidate_ref: str) -> dict:
    if candidate_ref not in {c["candidate_ref"] for c in frame["legal"]["candidates"]}:
        raise ProtocolError("Policy selected a non-legal candidate")
    routing = frame["routing"]
    command = {
        "cmd": "execute_candidate",
        "candidate_ref": candidate_ref,
        "decision_id": routing["decision_id"],
        "state_version": routing["state_version"],
    }
    if routing.get("selection_revision") is not None:
        command["selection_revision"] = routing["selection_revision"]
    return command


def segment_key(frame):
    routing = frame["routing"]
    context = frame["public"].get("selection_context") or {}
    # An incremental reveal always starts a fresh decoder, even in the same UI.
    segment = routing.get("selection_id") if context.get("mode") == "buffered" else None
    return (routing.get("episode_id"), segment or routing["decision_id"])


def clean_frame(frame):
    """Retain replay/routing metadata, but only whitelisted model input fields."""
    from .representation import clean_public

    result = deepcopy(frame)
    result["public"] = clean_public(frame["public"])
    # Public candidate semantics are cleaned by the same representation path.
    result["legal"] = {
        "candidates": [
            clean_public({"decoder_bank": [c]})["decoder_bank"][0]
            for c in frame["legal"]["candidates"]
        ]
    }
    return result
