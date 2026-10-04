"""Steam frames follow the engine protocol: card offers are seen before they are decided."""

from model.steam import RewardInspection, choose_action, live_frame


def card(name, key):
    return dict(id=name, instance_id=key, card_type="Attack", rarity="Common", upgraded=False)


def message(scope, actions, context, ascension=7):
    player = dict(creature=dict(hp=50, max_hp=80, block=0, instance_id="hero"), character="IRONCLAD", gold=99,
                  deck=[card("STRIKE_IRONCLAD", "d1")], relics=[], potions=[])
    return dict(episode="e", token="t", state=dict(
        run=dict(ascension=ascension, act=1, floor=3), players=[player],
        legal=dict(schema=1, status="complete", scope=scope, actions=actions, context=context)))


class Policy:
    """Takes the candidate with the wanted verb; records what it was shown."""

    def __init__(self, verb):
        self.verb, self.frames = verb, []

    def choose(self, frame, sample=False):
        self.frames.append(frame)
        return type("Choice", (), dict(candidate_ref=next(
            c["candidate_ref"] for c in frame["legal"]["candidates"] if c["verb"] == self.verb)))


TAKE = dict(command="take_reward", args=dict(reward_instance_id="r1"))
LEAVE = dict(command="skip_rewards", args={})
REWARDS = message("reward_choice", [TAKE, LEAVE], dict(rewards=[dict(reward_type="Card", instance_id="r1")]))


def offer(*names):
    cards = [card(n, f"c{i}") for i, n in enumerate(names)]
    picks = [dict(command="select_reward_option", args=dict(index=i)) for i in range(len(cards))]
    return message("card_reward", picks + [dict(command="select_reward_option", args=dict(index=None))],
                   dict(cards=cards, alternatives=[]))


def test_a_card_reward_is_opened_before_the_policy_decides_and_declining_it_leaves():
    rewards, policy = RewardInspection(), Policy("LEAVE_REWARDS")
    # The policy is not asked whether to look: an unopened offer shows no cards.
    assert choose_action(policy, REWARDS, rewards) == TAKE and not policy.frames
    declined = choose_action(policy, offer("FEED", "ANGER", "HAVOC"), rewards)
    assert declined["args"] == dict(index=None)
    shown = policy.frames[0]
    assert shown["public"]["phase"] == "rewards"
    assert sorted(c["verb"] for c in shown["legal"]["candidates"]) == ["LEAVE_REWARDS"] + ["TAKE_CARD_REWARD"] * 3
    assert {e["content_id"] for e in shown["public"]["entities"]
            if e["entity_type"] == "card" and e.get("zone") == "reward"} == {
        "CARD.FEED", "CARD.ANGER", "CARD.HAVOC"}
    # Back on the rewards screen the declined offer is not reopened: the controller leaves.
    assert choose_action(policy, REWARDS, rewards) == LEAVE and len(policy.frames) == 1


def test_the_offered_cards_and_the_ascension_change_the_frame():
    feed, strike = (live_frame(offer(name, "ANGER"), opened_reward=True)[0] for name in ("FEED", "STRIKE_IRONCLAD"))
    assert feed["public"] != strike["public"]
    hero = next(e for e in feed["public"]["entities"] if e["entity_type"] == "player")
    assert hero["ascension"] == 7 == feed["contract"]["fixed_ascension"]


def test_taking_a_card_ends_the_inspection():
    rewards, policy = RewardInspection(), Policy("TAKE_CARD_REWARD")
    choose_action(policy, REWARDS, rewards)
    taken = choose_action(policy, offer("FEED", "ANGER"), rewards)
    assert taken["args"]["index"] is not None and rewards.opened is None and not rewards.declined
