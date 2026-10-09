"""One policy/controller split for live play, replay, imports and dataset counts."""

from .crystal_rule import RULE_VERSION as CRYSTAL_RULE, plan as crystal_plan
from .treasure_rule import RULE_VERSION as TREASURE_RULE, collection_candidate as treasure_candidate


CONTROL_VERSION = "public-control-v1"
EVENT_RULE = "single-event-without-usable-potion-v1"
REWARD_RULE = "claim-free-rewards-v1"


def controller_for(phase, candidates):
    """Classify public legal candidates without running a controller or planner."""
    if phase == "treasure":
        return "treasure_controller"
    if phase == "crystal_sphere":
        return "crystal_sphere_planner"
    if phase == "event":
        options = [c for c in candidates if c["verb"] != "DISCARD_POTION"]
        if len(options) == 1 and options[0]["verb"] == "CHOOSE_EVENT_OPTION":
            return "event_controller"
    return None


def is_policy_frame(frame):
    candidates = frame["legal"]["candidates"]
    return len(candidates) > 1 and controller_for(frame["public"]["phase"], candidates) is None


def policy_macros(run):
    """Keep historical evidence intact; only macros with policy choices train."""
    return (macro for macro in run.get("macros", [])
            if any(is_policy_frame(step["frame"]) for step in macro["steps"]))


def environment_action(frame):
    """An executable controller action and its provenance, or None for the policy.

    Only the caller executes it. Classification never reveals future options.
    """
    candidates = frame["legal"]["candidates"]
    actor = controller_for(frame["public"]["phase"], candidates)
    if actor == "crystal_sphere_planner":
        result = crystal_plan(frame)
        return dict(actor=actor, candidate_ref=result.candidate["candidate_ref"],
                    rule=CRYSTAL_RULE, planner=result.diagnostics())
    if actor == "treasure_controller":
        candidate = treasure_candidate(frame)
        return dict(actor=actor, candidate_ref=candidate["candidate_ref"], rule=TREASURE_RULE)
    if actor == "event_controller":
        candidate = next(c for c in candidates if c["verb"] == "CHOOSE_EVENT_OPTION")
        return dict(actor=actor, candidate_ref=candidate["candidate_ref"], rule=EVENT_RULE)
    return None


def free_reward(frame):
    """The action that takes gold, a relic, or a potion the belt has room for from a reward
    screen, or None when none is waiting.

    It is not part of the shared split above. The demonstrations were rebuilt as if these
    had been claimed before the card choice, and show such a reward on screen only where a
    person left it; a rollout started with `claim_rewards` claims them the same way, so
    that the policy meets the reward screens it was trained on. Cards, removals, linked
    sets, a potion the belt has no room for, and leaving stay with the policy.
    """
    if frame["public"]["phase"] != "rewards":
        return None
    contents = {entity["ref"]: entity.get("content_id") or "" for entity in frame["public"]["entities"]
                if entity.get("entity_type") == "reward"}
    takes = [c for c in frame["legal"]["candidates"] if c["verb"] == "TAKE_REWARD"]
    for prefix in ("GoldReward", "RELIC.", "POTION."):
        for candidate in takes:
            if contents.get(candidate["source_refs"][0], "").startswith(prefix):
                return dict(actor="reward_controller", candidate_ref=candidate["candidate_ref"], rule=REWARD_RULE)
    return None
