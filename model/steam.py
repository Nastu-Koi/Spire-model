"""Live Steam inference, using exactly the recorder's public-state representation."""

import fcntl
import json
import time
import uuid
from pathlib import Path

import torch

from .checkpoint import load_model
from .policy import SessionPolicy
from .protocol import ProtocolError, clean_frame, validate_frame
from .recorder import PHASES, PublicSnapshot, require
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


def live_frame(message, selected=()):
    state, token = message["state"], message["token"]
    require(state["run"]["ascension"] == 10, "Start or load a single-player A10 run")
    raw_legal = state["legal"]
    require(raw_legal["status"] == "complete", "Steam has no complete legal action set")
    scope = raw_legal["scope"]
    require(scope in PHASES, f"Unsupported Steam decision: {scope}")
    snapshot = PublicSnapshot(state)
    offered = raw_legal["actions"]
    selection = raw_legal.get("selection")
    raw_bank = list(offered)
    if selection:
        raw_bank += [{"command": "finish_selection", "args": {}}]
        if selection["cancelable"]:
            raw_bank += [{"command": "cancel", "args": {}}]
    bank = [
        snapshot.action(action, index, scope) for index, action in enumerate(raw_bank)
    ]
    public = {
        "phase": PHASES[scope],
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
            "can_cancel": selection["cancelable"],
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
                "fixed_ascension": 10,
                "training_ready": True,
                "adapter_version": "steam-live-v1",
                "observation_schema": "public-state-v1",
                "action_schema": "candidate-v0",
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
                    action = choose_action(policy, message)
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
