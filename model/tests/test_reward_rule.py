"""Fixed reward collection: relics, then gold, then potions; cards stay policy decisions."""

from model.reward_rule import collection_candidate


def frame(rewards, extra=()):
    entities, candidates = [], []
    for i, content_id in enumerate(rewards):
        entities.append({"entity_type": "reward", "ref": f"reward:{i}", "content_id": content_id})
        candidates.append({"verb": "TAKE_REWARD", "source_refs": [f"reward:{i}"], "candidate_ref": f"c{i}"})
    candidates += [dict(c) for c in extra]
    candidates.append({"verb": "LEAVE_REWARDS", "source_refs": [], "candidate_ref": "leave"})
    return {"public": {"phase": "rewards", "entities": entities}, "legal": {"candidates": candidates}}


def test_relic_before_gold_before_potion():
    f = frame(["POTION.FIRE_POTION", "GoldReward", "RELIC.ANCHOR"])
    assert collection_candidate(f)["candidate_ref"] == "c2"
    f = frame(["POTION.FIRE_POTION", "GoldReward"])
    assert collection_candidate(f)["candidate_ref"] == "c1"
    f = frame(["POTION.FIRE_POTION"])
    assert collection_candidate(f)["candidate_ref"] == "c0"


def test_cards_special_and_removal_rewards_stay_with_the_policy():
    card = {"verb": "TAKE_CARD_REWARD", "source_refs": ["card:1"], "candidate_ref": "card"}
    assert collection_candidate(frame(["SpecialCardReward", "CardRemovalReward"], [card])) is None
    assert collection_candidate(frame([], [card])) is None


def test_only_rewards_screens():
    f = frame(["GoldReward"])
    f["public"]["phase"] = "shop"
    assert collection_candidate(f) is None
