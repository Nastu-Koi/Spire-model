"""Live Steam inference, using exactly the recorder's public-state representation."""

import fcntl
import json
import time
import uuid
from pathlib import Path

import torch

from .checkpoint import load_model
from .policy import SessionPolicy
from .crystal_rule import collection_candidate as crystal_candidate
from .reward_rule import collection_candidate as reward_candidate
from .protocol import ProtocolError, clean_frame, validate_frame
from .recorder import PHASES, SELECTION_CANCEL, PublicSnapshot, cancel_offered, opaque, require
from .rollout import precision_context


class SteamBridge:
    def __init__(self, directory, timeout=120):
        self.directory = Path(directory).expanduser()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.lock = (self.directory / "controller.lock").open("a")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.lock.close()
            raise ProtocolError(
                "Another model controller is using this bridge"
            ) from None

    def request(self, command, **payload):
        request_id = uuid.uuid4().hex
        request = self.directory / "request.json"
        response = self.directory / "response.json"
        temporary = request.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(dict(id=request_id, command=command, **payload))
        )
        temporary.replace(request)
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            try:
                result = json.loads(response.read_text())
                if result.get("id") == request_id:
                    if result.get("status") == "error":
                        raise ProtocolError(result.get("error", "Steam bridge failed"))
                    return result
            except (FileNotFoundError, json.JSONDecodeError):
                pass
            time.sleep(0.1)
        request.unlink(missing_ok=True)
        raise TimeoutError(
            f"Steam bridge did not respond within {self.timeout}s: {self.directory}"
        )

    def close(self):
        (self.directory / "request.json").unlink(missing_ok=True)
        self.lock.close()


def live_frame(message, selected=(), opened_reward=False):
    """The decision frame of a Steam state, in the engine protocol's terms.

    `opened_reward` marks a card-reward screen the controller opened from the
    rewards screen. The engine shows such an offer on the rewards screen itself,
    where declining it is leaving; the frame is published the same way.
    """
    state, token = message["state"], message["token"]
    ascension = state["run"]["ascension"]
    require(type(ascension) is int and 0 <= ascension <= 10, "Start or load a single-player run")
    raw_legal = state["legal"]
    require(raw_legal["status"] == "complete", "Steam has no complete legal action set")
    scope = raw_legal["scope"]
    require(scope in PHASES, f"Unsupported Steam decision: {scope}")
    snapshot = PublicSnapshot(state)
    offered = raw_legal["actions"]
    selection = raw_legal.get("selection")
    raw_bank = list(offered)
    offer_cancel = False
    if selection:
        raw_bank += [{"command": "finish_selection", "args": {}}]
        # Same rule as the engine protocol: no undo outside combat.
        offer_cancel = cancel_offered(
            state,
            selection["cancelable"],
            len(selected) >= selection["min"],
            len(selected) < selection["max"]
            and any(a["command"] == "select_card" and a["args"]["card_instance_id"] not in selected
                    for a in offered),
        )
        if offer_cancel:
            raw_bank += [{"command": "cancel", "args": {}}]
    bank = [
        snapshot.action(action, index, scope) for index, action in enumerate(raw_bank)
    ]
    phase = PHASES[scope]
    if opened_reward and scope == "card_reward":
        phase = "rewards"
        offered_cards = [e["ref"] for e in snapshot.entities
                         if e.get("entity_type") == "card" and e.get("zone") == "reward"]
        snapshot.entities.append({"entity_type": "reward", "ref": "reward:0", "content_id": "CardReward",
                                  "effect_coverage": "opaque", "semantic_program": opaque("CardReward")})
        snapshot.relations.extend({"source": "reward:0", "target": ref, "role": "offers"} for ref in offered_cards)
        for candidate in bank:
            if candidate["verb"] == "SKIP":
                candidate["verb"] = "LEAVE_REWARDS"
    public = {
        "phase": phase,
        "entities": snapshot.entities,
        "relations": snapshot.relations,
        "memory": [],
        "decoder_bank": bank,
    }
    routing = {
        "episode_id": message["episode"],
        "decision_id": token,
        "state_version": 0,
    }
    candidates = bank
    if selection:
        minimum, maximum = selection["min"], selection["max"]
        require(0 <= minimum <= maximum, "Invalid Steam selection bounds")
        candidates = [
            bank[i]
            for i, action in enumerate(raw_bank)
            if (
                action["command"] == "select_card"
                and len(selected) < maximum
                and action["args"]["card_instance_id"] not in selected
            )
            or (action["command"] == "finish_selection" and len(selected) >= minimum)
            or action["command"] == "cancel"
        ]
        public["selection_context"] = {
            "mode": "buffered",
            "min_total": minimum,
            "max_total": maximum,
            "selected_refs": [snapshot.refs[x] for x in selected],
            "selected_count": len(selected),
            "remaining_required": max(0, minimum - len(selected)),
            "can_finish": len(selected) >= minimum,
            "can_cancel": offer_cancel,
            "can_skip": False,
            "repetition_allowed": False,
            "order_matters": selection.get("ordered", True),
        }
        routing.update(
            selection_id=token,
            selection_revision=len(selected),
            state_version=len(selected),
            base_public_version=0,
            action_bank_version=0,
        )
    frame = clean_frame(
        {
            "type": "decision_frame",
            "boundary": "decision",
            "routing": routing,
            "contract": {
                "fixed_ascension": ascension,
                "training_ready": True,
                "adapter_version": "steam-live-v1",
                "observation_schema": "public-state-v3",
                "action_schema": "candidate-v0",
                "selection_cancel": SELECTION_CANCEL,
            },
            "public": public,
            "legal": {"candidates": candidates},
            "events": [],
        }
    )
    validate_frame(frame)
    return frame, {
        bank[i]["candidate_ref"]: action for i, action in enumerate(raw_bank)
    }


class RewardInspection:
    """Card rewards of the current rewards screen that the controller has opened.

    Looking at a card offer costs nothing, so it is never a policy decision: the
    controller opens every card reward before the policy acts on the screen, and
    leaves on its own once the policy has declined all that is left.
    """

    def __init__(self):
        self.opened, self.declined = None, set()

    def rule(self, message):
        """The controller's own action on a rewards screen, or None when the policy acts."""
        legal = message["state"]["legal"]
        if legal["scope"] != "reward_choice":
            return None
        cards = {r.get("instance_id") for r in legal["context"]["rewards"] if r["reward_type"] == "Card"}
        takes = [a for a in legal["actions"] if a["command"] == "take_reward"]
        for action in takes:
            key = action["args"].get("reward_instance_id")
            if key in cards and key not in self.declined:
                self.opened = key
                return action
        leave = next((a for a in legal["actions"] if a["command"] == "skip_rewards"), None)
        if leave and takes and all(a["args"].get("reward_instance_id") in self.declined for a in takes):
            return leave
        return None


def choose_action(policy, message, rewards=None):
    rewards = rewards or RewardInspection()
    scope = message["state"]["legal"]["scope"]
    if scope not in {"reward_choice", "card_reward"}:
        rewards.opened, rewards.declined = None, set()
    opened = rewards.opened if scope == "card_reward" else None
    selected = []
    while True:
        frame, commands = live_frame(message, selected, opened_reward=opened is not None)
        rule = crystal_candidate(frame) or reward_candidate(frame)
        if rule is not None:
            # Same environment rules as training rollouts.
            return commands[rule["candidate_ref"]]
        inspect = rewards.rule(message)
        if inspect is not None:
            return inspect
        choice = policy.choose(frame, sample=False)
        action = commands[choice.candidate_ref]
        if opened is not None:
            rewards.opened = None
            if action["command"] == "select_reward_option" and action["args"].get("index") is None:
                rewards.declined.add(opened)
        if action["command"] == "select_card" and message["state"]["legal"].get(
            "selection"
        ):
            selected.append(action["args"]["card_instance_id"])
        elif action["command"] in {"finish_selection", "cancel"}:
            return {
                "command": "select_cards",
                "args": {
                    "card_instance_ids": []
                    if action["command"] == "cancel"
                    else selected
                },
            }
        else:
            return action


def play(checkpoint, directory, device="cpu", timeout=120, max_decisions=10000):
    if timeout <= 0 or max_decisions < 1:
        raise ValueError("timeout and max-decisions must be positive")
    model, vocabulary, manifest = load_model(checkpoint, device)
    model.eval()
    policy = SessionPolicy(model, vocabulary)
    rewards = RewardInspection()
    bridge = SteamBridge(directory, timeout)
    count, last_progress = 0, time.monotonic()
    try:
        with (
            torch.inference_mode(),
            precision_context(model, manifest["training"]["precision"]),
        ):
            while count < max_decisions:
                message = bridge.request("observe")
                status = message["status"]
                if status == "terminal":
                    return {
                        "decisions": count,
                        "status": status,
                        "victory": message.get("victory"),
                    }
                if status == "decision":
                    action = choose_action(policy, message, rewards)
                    result = bridge.request(
                        "execute", token=message["token"], action=action
                    )
                    if result["status"] == "stale":
                        policy.reset()
                        continue
                    require(
                        result["status"] == "executed",
                        "Steam did not execute the action",
                    )
                    count += 1
                    last_progress = time.monotonic()
                    print(
                        json.dumps(
                            {"decision": count, "action": action}, ensure_ascii=False
                        ),
                        flush=True,
                    )
                elif status == "waiting":
                    if time.monotonic() - last_progress > timeout:
                        raise TimeoutError(
                            "No playable Steam decision: "
                            + message.get("reason", "waiting")
                        )
                    bridge.request("advance")
                    time.sleep(0.2)
                else:
                    raise ProtocolError("Unexpected Steam bridge status: " + status)
        return {"decisions": count, "status": "decision_limit"}
    finally:
        bridge.close()
