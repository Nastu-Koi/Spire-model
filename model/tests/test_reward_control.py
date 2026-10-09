"""Policy rewards retain downstream returns across controller actions."""

from copy import deepcopy
from types import SimpleNamespace

from model.config import ModelConfig
from model.data import validate_run
from model.model import PolicyValue
from model.representation import Vocabulary
from model.rollout import RolloutRunner
from model.testing import SyntheticEngine
from model.tests.test_crystal_planner import board_frame, corners


def policy_frame(phase, verbs, index):
    frame = SyntheticEngine(4, 2).reset("Ironclad", "control", ascension=0)
    frame["routing"].update(decision_id=f"control:{index}", state_version=index,
                           base_public_version=index, action_bank_version=str(index))
    frame["routing"].pop("selection_id", None)
    frame["routing"].pop("selection_revision", None)
    frame["public"] = dict(phase=phase, memory=[], relations=[], entities=[
        dict(entity_type="player", ref="player", hp=50, max_hp=80, act=3),
        dict(entity_type="event", ref="sphere", content_id="CRYSTAL_SPHERE"),
        dict(entity_type="reward", ref="gold", content_id="GoldReward", gold=20),
    ])
    frame["legal"] = dict(candidates=[dict(verb=verb, candidate_ref=f"{index}:{i}",
        decoder_slot_ref=f"action:{i}", source_refs=["gold"] if verb == "TAKE_REWARD" else [])
        for i, verb in enumerate(verbs)])
    return frame


class ScriptedEngine:
    def __init__(self, frames):
        self.frames = deepcopy(frames)
        self.actions = []

    def reset(self, character, seed, ascension):
        assert ascension == 0
        return self.frames.pop(0)

    def send(self, command):
        self.actions.append(command)
        return self.frames.pop(0)


class Policy:
    def __init__(self):
        self.phases = []

    def reset(self):
        pass

    def choose(self, frame, *, sample=True):
        self.phases.append(frame["public"]["phase"])
        return SimpleNamespace(candidate_ref=frame["legal"]["candidates"][-1]["candidate_ref"],
                               log_prob=-0.7, value=0.25)


def test_payment_and_leaving_rewards_receive_return_across_grid_environment_steps():
    payment = policy_frame("event", ["CHOOSE_EVENT_OPTION", "CHOOSE_EVENT_OPTION"], 0)
    crystal = policy_frame("crystal_sphere", [], 1)
    crystal.update(board_frame(revealed=corners(), remaining=1))
    rewards = policy_frame("rewards", ["TAKE_REWARD", "LEAVE_REWARDS"], 2)
    terminal = policy_frame("terminal", [], 3)
    terminal.update(boundary="terminal", events=[dict(type="run_completed", victory=True,
                    act=3, final_boss_defeated=True)])
    terminal["public"]["outcome"] = dict(victory=True)
    engine = ScriptedEngine([payment, crystal, rewards, terminal])
    config = ModelConfig.tiny()
    vocabulary = Vocabulary.from_frames([payment, rewards], config)
    policy = Policy()
    trace = RolloutRunner(PolicyValue(config), vocabulary).run(
        engine, "Ironclad", "control", _policy=policy)
    assert trace["status"] == "complete", trace.get("error")
    assert policy.phases == ["event", "rewards"]
    assert [m["phase"] for m in trace["macros"]] == ["event", "rewards"]
    assert [m["return"] for m in trace["macros"]] == [5, 5]
    assert [m["old_log_prob"] for m in trace["macros"]] == [-0.7, -0.7]
    assert trace["macros"][-1]["steps"][0]["candidate_ref"] == "2:1"
    assert trace["automatic_steps"] == 1
    assert trace["environment_actions"][0]["actor"] == "crystal_sphere_planner"
    assert trace["environment_actions"][0]["planner"]["hypotheses"] > 0
    validate_run(trace, on_policy=True)


def reward_screen(index, rewards, extra=("LEAVE_REWARDS",)):
    frame = policy_frame("rewards", [], index)
    frame["public"]["entities"] = [dict(entity_type="player", ref="player", hp=50, max_hp=80, act=1)] + [
        dict(entity_type="reward", ref=f"reward:{i}", content_id=content) for i, content in enumerate(rewards)]
    takes = [dict(verb="TAKE_REWARD", candidate_ref=f"{index}:take:{i}", decoder_slot_ref=f"take:reward:{i}",
                  source_refs=[f"reward:{i}"]) for i, content in enumerate(rewards) if content != "CardReward"]
    others = [dict(verb=verb, candidate_ref=f"{index}:{verb}", decoder_slot_ref=verb, source_refs=[]) for verb in extra]
    frame["legal"] = dict(candidates=takes + others)
    return frame


def test_free_rewards_are_claimed_in_order_and_everything_else_stays_with_the_policy():
    from model.control import REWARD_RULE, controller_for, free_reward

    screen = reward_screen(0, ["POTION.FIRE_POTION", "CardReward", "RELIC.ANCHOR", "GoldReward"])
    assert free_reward(screen) == dict(actor="reward_controller", candidate_ref="0:take:3", rule=REWARD_RULE)
    assert free_reward(reward_screen(0, ["POTION.FIRE_POTION", "RELIC.ANCHOR"]))["candidate_ref"] == "0:take:1"
    assert free_reward(reward_screen(0, ["POTION.FIRE_POTION", "CardReward"]))["candidate_ref"] == "0:take:0"
    # A potion the belt has no room for has no take action; cards, removals and linked sets are choices.
    full_belt = reward_screen(0, ["POTION.FIRE_POTION"], extra=("DISCARD_POTION", "LEAVE_REWARDS"))
    full_belt["legal"]["candidates"] = full_belt["legal"]["candidates"][1:]
    for kept in (full_belt, reward_screen(0, ["CardReward"], extra=("TAKE_CARD_REWARD", "LEAVE_REWARDS")),
                 reward_screen(0, ["LinkedRewardSet", "CardRemovalReward", "SpecialCardReward"])):
        assert free_reward(kept) is None
    assert free_reward(dict(screen, public=dict(screen["public"], phase="event"))) is None
    # The shared split, and with it what demonstrations train on, is unchanged.
    assert controller_for("rewards", screen["legal"]["candidates"]) is None


def test_rollout_claims_free_rewards_only_when_asked():
    first = reward_screen(0, ["GoldReward", "RELIC.ANCHOR", "CardReward"], extra=("TAKE_CARD_REWARD", "LEAVE_REWARDS"))
    second = reward_screen(1, ["RELIC.ANCHOR", "CardReward"], extra=("TAKE_CARD_REWARD", "LEAVE_REWARDS"))
    third = reward_screen(2, ["CardReward"], extra=("TAKE_CARD_REWARD", "LEAVE_REWARDS"))
    terminal = policy_frame("terminal", [], 3)
    terminal.update(boundary="terminal", events=[])
    terminal["public"]["outcome"] = dict(victory=False)
    config = ModelConfig.tiny()
    vocabulary = Vocabulary.from_frames([first, third], config)
    engine, policy = ScriptedEngine([first, second, third, terminal]), Policy()
    trace = RolloutRunner(PolicyValue(config), vocabulary, claim_rewards=True).run(
        engine, "Ironclad", "control", _policy=policy)
    assert trace["status"] == "complete", trace.get("error")
    assert [a["actor"] for a in trace["environment_actions"]] == ["reward_controller"] * 2
    assert [a["candidate_ref"] for a in trace["environment_actions"]] == ["0:take:0", "1:take:0"]
    assert policy.phases == ["rewards"] and trace["macros"][0]["steps"][0]["candidate_ref"] == "2:LEAVE_REWARDS"
    assert trace["reward_claim_rule"] == "claim-free-rewards-v1"
    engine, policy = ScriptedEngine([first, terminal]), Policy()
    trace = RolloutRunner(PolicyValue(config), vocabulary).run(engine, "Ironclad", "control", _policy=policy)
    assert policy.phases == ["rewards"] and not trace["environment_actions"] and trace["reward_claim_rule"] is None
