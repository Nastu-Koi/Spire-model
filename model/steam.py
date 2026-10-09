"""Live Steam inference, using exactly the recorder's public-state representation."""

import fcntl
import json
import time
import uuid
from pathlib import Path

import torch

from .checkpoint import load_model
from .policy import SessionPolicy
from .control import environment_action
from .protocol import ProtocolError, clean_frame, validate_frame
from .recorder import PHASES, SELECTION_CANCEL, PublicSnapshot, cancel_offered, require
from .rollout import precision_context
from .public_history import HISTORY_VERSION

# What the bridge must speak: card rewards arrive with their cards, a chest is opened by an action.
PROTOCOL = "steam-live-v2"


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
        # An answer of its own: the game cannot replace a file this side still has open.
        response = self.directory / f"response-{request_id}.json"
        temporary = request.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(dict(id=request_id, command=command, unique_response=True, **payload))
        )
        temporary.replace(request)
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            try:
                result = json.loads(response.read_text())
                if result.get("id") == request_id:
                    response.unlink(missing_ok=True)
                    if result.get("protocol") != PROTOCOL:
                        raise ProtocolError(
                            f"The installed RunRecorder speaks {result.get('protocol') or 'an older protocol'}, "
                            f"not {PROTOCOL}: reinstall it with steam_recorder/install.py"
                        )
                    if result.get("status") == "error":
                        raise ProtocolError(result.get("error", "Steam bridge failed"))
                    if result.get("status") == "paused":
                        raise ProtocolError("Model control is switched off in the game's RunRecorder panel")
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


def live_frame(message, selected=()):
    """The decision frame of a Steam state, in the engine protocol's terms."""
    state, token = message["state"], message["token"]
    ascension = state["run"]["ascension"]
    require(type(ascension) is int and 0 <= ascension <= 10, "Start or load a single-player run")
    raw_legal = state["legal"]
    require(raw_legal["status"] == "complete", "Steam has no complete legal action set")
    scope = raw_legal["scope"]
    require(scope in PHASES, f"Unsupported Steam decision: {scope}")
    offered = raw_legal["actions"]
    if scope == "reward_choice":
        # Opening a card reward is never a decision: its cards come with the screen.
        unseen = {r.get("instance_id") for r in raw_legal["context"]["rewards"]
                  if r["reward_type"] == "Card" and not r.get("expanded_card_reward")}
        require(not any(a["command"] == "take_reward" and a["args"].get("reward_instance_id") in unseen
                        for a in offered), "Steam offers a card reward without its cards")
    snapshot = PublicSnapshot(state)
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
    public = {
        "phase": PHASES[scope],
        "entities": snapshot.entities,
        "relations": snapshot.relations,
        "memory": snapshot.memory,
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
        described = selection.get("metadata") or {}
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
            # What the selection does, in the engine's words; unknown where the game does not say.
            **{key: described.get(key, "unknown") for key in ("operation", "source", "destination")},
            "known_masks": described.get("known_masks") or dict.fromkeys(
                ("operation", "source", "destination", "order_matters"), False),
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
                "observation_schema": "public-state-v6",
                "public_history_version": HISTORY_VERSION,
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


def choose_action(policy, message):
    selected = []
    while True:
        frame, commands = live_frame(message, selected)
        rule = environment_action(frame)
        if rule is not None:
            # Use the same controller decision and reveal order as rollouts.
            return commands[rule["candidate_ref"]]
        choice = policy.choose(frame, sample=False)
        action = commands[choice.candidate_ref]
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
    trained = manifest["training"]["ascension"]
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
                    require(message["state"]["run"]["ascension"] == trained,
                            f"The checkpoint was trained on ascension {trained}")
                    action = choose_action(policy, message)
                    result = bridge.request(
                        "execute", token=message["token"], action=action
                    )
                    if result["status"] == "stale":
                        # The game moved on between observing and acting: look again,
                        # but not forever.
                        if time.monotonic() - last_progress > timeout:
                            raise TimeoutError("Steam state kept changing before an action could be executed")
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
