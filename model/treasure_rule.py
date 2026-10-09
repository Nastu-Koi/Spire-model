"""Ordinary chests follow open -> take the relic -> leave, using public offers."""

from .protocol import ProtocolError


RULE_VERSION = "treasure-open-take-leave-v1"


def collection_candidate(frame):
    """Both map chests and chests reached through '?' use the treasure phase.

    The engine still owns legality and reveals the relic only after opening.
    Selections triggered by a relic have their own phase and reach the policy.
    """
    if frame["public"]["phase"] != "treasure":
        return None
    candidates = frame["legal"]["candidates"]
    for verb in ("OPEN_CHEST", "TAKE_TREASURE_RELIC", "LEAVE_ROOM"):
        offered = [candidate for candidate in candidates if candidate["verb"] == verb]
        if len(offered) > 1:
            raise ProtocolError(f"Ordinary treasure requires one {verb} candidate")
        if offered:
            return offered[0]
    raise ProtocolError("Treasure has no open, relic or leave action")
