"""Fixed reward collection outside PPO decisions.

On a rewards screen, relics are taken first (they can add potion slots or change
gold), then gold, then potions. The engine offers a potion reward only while a
slot is free; with full slots the policy decides whether to discard a potion
first, after which the reward is collected here. Card rewards, special card
rewards and card-removal rewards remain policy decisions.
"""

RULE_VERSION = "reward-collect-v1"
PER_FLOOR_LIMIT = 8


def _priority(content_id):
    if content_id.startswith("RELIC."):
        return 0
    if content_id == "GoldReward":
        return 1
    if content_id.startswith("POTION."):
        return 2
    return None


def collection_candidate(frame):
    """The rule's next TAKE_REWARD candidate, or None when the policy must act."""
    if frame["public"]["phase"] != "rewards":
        return None
    refs = {e.get("ref"): e for e in frame["public"]["entities"]}
    ranked = []
    for index, candidate in enumerate(frame["legal"]["candidates"]):
        if candidate["verb"] != "TAKE_REWARD":
            continue
        source = (candidate.get("source_refs") or [None])[0]
        rank = _priority((refs.get(source) or {}).get("content_id") or "")
        if rank is not None:
            ranked.append((rank, index, candidate))
    return min(ranked)[2] if ranked else None
