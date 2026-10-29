"""Live Steam inference through the SpireTakeover mod's bridge.

The controller (`Controller`, run by `play`) stays beside the game for as long as it
is asked to: it observes through the mod's bridge, decides, executes, and reports
each decision to the mod's panel. The mod holds the switches (which phases the model
plays, how long to wait between steps) and sends them with every observation. A
phase the model does not play is `manual`; a run's end is `terminal` and the next run
goes on; an error the controller meets is written down, reported, and halts the
controller until the panel's reset.
"""

import fcntl
import json
import re
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import torch

from main.network.checkpoint import load_model
from main.network.policy import SessionPolicy
from main.env.control import environment_action
from main.env.protocol import ProtocolError, validate_frame
from main.encoding.frames import clean_frame
from main.steam.snapshot import PublicSnapshot, require
from main.play.rollout import precision_context
from main.env.public_history import HISTORY_VERSION

PHASES = {
    "combat_play": "combat",
    "map_select": "map",
    "event_choice": "event",
    "rest_site": "rest_site",
    "shop": "shop",
    "treasure": "treasure",
    "reward_choice": "rewards",
    "card_select": "card_select",
    "card_reward": "card_reward",
    "bundle_select": "bundle_select",
    "crystal_sphere": "crystal_sphere",
}
SELECTION_CANCEL = "combat-or-dead-end-v1"


def cancel_offered(state, cancelable, can_finish, can_select):
    """Outside combat a native cancel only restores the state before the
    action that opened the selection, so it is offered only when nothing else is."""
    in_combat = bool((state.get("combat") or {}).get("in_progress"))
    return bool(cancelable) and (in_combat or (not can_finish and not can_select))


# What the bridge must speak: settings and the reset count with every answer, `manual`
# and `halted` statuses, the `report` command, `leave_rest_site` at a rest site that
# keeps options after a choice.
PROTOCOL = "steam-live-v5"
DECISIONS, ERRORS, RUNS = "decisions.jsonl", "errors.jsonl", "runs.jsonl"
# The whole message of each selection the bridge offers no card of, to find where the
# mod lost its cards (issue 17 of .scratch/training-plan).
EMPTY_SELECTIONS = "empty-selections.jsonl"
RANKED = 5  # candidates reported beside the chosen one
# What halts the controller until the panel's reset: anything the bridge, the frame or
# the model can get wrong, short of a bug in this module's own control flow.
HALTING = (ProtocolError, TimeoutError, RuntimeError, KeyError, ValueError, TypeError, OSError)
IDLE = ("waiting", "manual", "paused", "halted")


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
        """One request to the game and its answer. An answer of status `error` is
        raised; `paused` (control switched off in the panel) is returned like the other
        idle statuses, for the caller to wait out."""
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
                            f"The installed SpireTakeover mod speaks {result.get('protocol') or 'an older protocol'}, "
                            f"not {PROTOCOL}: reinstall it with steam_takeover/install.py"
                        )
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


def parse_duration(text):
    """Seconds of a duration such as `8h`, `45m`, `1h30m`, `90s` or `90`."""
    text = str(text).strip().lower()
    if re.fullmatch(r"\d+(\.\d+)?", text):
        return float(text)
    parts = re.findall(r"(\d+(?:\.\d+)?)([hms])", text)
    if not parts or "".join(number + unit for number, unit in parts) != text:
        raise ValueError(f"Not a duration (such as 8h, 45m, 1h30m or 90s): {text}")
    return sum(float(number) * {"h": 3600, "m": 60, "s": 1}[unit] for number, unit in parts)


def live_frame(message, selected=()):
    """The decision frame of a Steam state, in the engine protocol's terms."""
    state, token = message["state"], message["token"]
    ascension = state["run"]["ascension"]
    require(type(ascension) is int and ascension >= 0, "Start or load a single-player run")
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
                "observation_schema": "public-state-v7",
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


def summary_of(frame, candidate_ref):
    """A candidate in a few words: its verb, what it acts with and what it acts on,
    named by the content of the entities it refers to."""
    candidate = next(c for c in frame["legal"]["candidates"] if c["candidate_ref"] == candidate_ref)
    names = {e["ref"]: e.get("content_id") or e.get("entity_type") for e in frame["public"]["entities"] if "ref" in e}

    def named(refs):
        return " ".join(str(names.get(ref, ref)).split(".")[-1] for ref in refs)
    text = candidate["verb"]
    if candidate.get("source_refs"):
        text += " " + named(candidate["source_refs"])
    if candidate.get("target_refs"):
        text += " → " + named(candidate["target_refs"])
    return text


def decide(policy, message):
    """The action to execute for a Steam decision, and what to tell the panel about it:
    a selection is walked step by step as the model makes it. Returns (action, detail);
    `detail` has the phase, the summary of the (last) choice, whether a controller
    rule chose it, and the model's ranking of the alternatives."""
    selected = []
    while True:
        frame, commands = live_frame(message, selected)
        phase = frame["public"]["phase"]
        rule = environment_action(frame)
        if rule is not None:
            # Use the same controller decision and reveal order as rollouts.
            return commands[rule["candidate_ref"]], dict(
                phase=phase, summary=summary_of(frame, rule["candidate_ref"]), controller=True, ranking=[])
        choice = policy.choose(frame, sample=False, top_k=RANKED)
        detail = dict(phase=phase, summary=summary_of(frame, choice.candidate_ref), controller=False,
                      ranking=[dict(summary=summary_of(frame, r["candidate_ref"]), probability=round(r["probability"], 4))
                               for r in (getattr(choice, "ranking", None) or [])])
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
            }, detail
        else:
            return action, detail


def empty_selection(message):
    """Whether a decision is a card selection whose first frame offers the policy no
    card: the bridge sent none, or bounds that leave none selectable."""
    if not (message["state"].get("legal") or {}).get("selection"):
        return False
    frame, _ = live_frame(message)
    return not any(c["verb"] == "SELECT_ONE" for c in frame["legal"]["candidates"])


def choose_action(policy, message):
    return decide(policy, message)[0]


def _now():
    return datetime.now(UTC).isoformat(timespec="seconds")


class Controller:
    """The loop beside the game. `bridge` answers `request(command, **payload)`
    (`SteamBridge`); `policy` chooses (`SessionPolicy`). The files the controller
    writes under `directory` are the decisions, the errors and the runs' ends, one
    JSON line each. `clock` and `sleep` are the monotonic clock and the pause, so
    that a test can drive time."""

    def __init__(self, bridge, policy, *, directory, run_for=None, max_decisions=None,
                 poll=0.2, clock=time.monotonic, sleep=time.sleep, say=None):
        if run_for is not None and run_for <= 0 or max_decisions is not None and max_decisions < 1:
            raise ValueError("run_for and max_decisions must be positive")
        self.bridge, self.policy, self.directory = bridge, policy, Path(directory)
        self.run_for, self.max_decisions = run_for, max_decisions
        self.poll, self.clock, self.sleep = poll, clock, sleep
        self.say = say or (lambda record: print(json.dumps(record, ensure_ascii=False), flush=True))
        self.stats = dict(runs=0, wins=0, decisions=0, errors=0)
        self.deadline = None
        self.last_executed = None
        self.previous = None  # the last status observed
        self.reset_count = None  # the panel's reset count, as the bridge last told it

    def write(self, name, record):
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / name).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    def due(self):
        """Why the controller should stop now, or None."""
        if self.deadline is not None and self.clock() >= self.deadline:
            return "run_for"
        if self.max_decisions is not None and self.stats["decisions"] >= self.max_decisions:
            return "decision_limit"
        return None

    def run(self):
        if self.run_for is not None:
            self.deadline = self.clock() + self.run_for
        status = None
        try:
            while (status := self.due()) is None:
                try:
                    self.step()
                except HALTING as exc:
                    status = self.halt(exc)
                    if status:
                        break
        except KeyboardInterrupt:
            status = "interrupted"
        return dict(self.stats, status=status)

    def note_reset(self, message):
        """The panel's reset count: a change means the user reset the game side, so
        the session starts over from the state as it stands."""
        count = message.get("reset_count")
        if self.reset_count is None:
            self.reset_count = count
        elif count != self.reset_count:
            self.reset_count = count
            self.policy.reset()

    def step(self):
        message = self.bridge.request("observe")
        self.note_reset(message)
        status = message["status"]
        if status == "decision":
            self.act(message, message.get("settings") or {})
        elif status == "terminal":
            if self.previous != "terminal":
                record = dict(time=_now(), victory=message.get("victory"), decisions=self.stats["decisions"])
                self.stats["runs"] += 1
                self.stats["wins"] += bool(message.get("victory"))
                self.write(RUNS, record)
                self.say(dict(run=self.stats["runs"], **record))
            self.sleep(self.poll)
        elif status == "waiting":
            # A screen with nothing to decide: the mod advances it when it is one it
            # may, and answers `waiting` otherwise (the menu, an animation, a manual phase).
            self.bridge.request("advance")
            self.sleep(self.poll)
        elif status in IDLE:
            self.sleep(self.poll)
        else:
            raise ProtocolError("Unexpected Steam bridge status: " + str(status))
        self.previous = status

    def act(self, message, settings):
        episode = message.get("episode")
        if empty_selection(message):
            self.write(EMPTY_SELECTIONS, dict(time=_now(), episode=episode, token=message.get("token"), message=message))
        action, detail = decide(self.policy, message)
        interval = float(settings.get("step_interval_seconds") or 0)
        if self.last_executed is not None and interval > 0:
            wait = self.last_executed + interval - self.clock()
            if wait > 0:
                self.sleep(wait)
        result = self.bridge.request("execute", token=message["token"], action=action)
        if result["status"] == "stale" or result["status"] in IDLE:
            # The game moved on, or the panel changed its mind, between observing and
            # acting: look again.
            self.policy.reset()
            return
        require(result["status"] == "executed", "Steam did not execute the action")
        self.stats["decisions"] += 1
        self.last_executed = self.clock()
        run = message["state"]["run"]
        record = dict(time=_now(), episode=episode, decision=self.stats["decisions"], act=run.get("act"),
                      floor=run.get("floor"), action=action, **detail)
        self.write(DECISIONS, record)
        self.say(record)
        self.bridge.request("report", kind="decision", **record)

    def halt(self, exc):
        """An error: write it down, report it to the panel, and wait for the panel's
        reset before going on. Returns the reason to stop instead, if one comes due
        while waiting."""
        record = dict(time=_now(), error=str(exc), exception=type(exc).__name__, after=self.previous)
        self.stats["errors"] += 1
        self.write(ERRORS, record)
        self.say(dict(halted=record))
        self.policy.reset()
        waited, acknowledged = self.reset_count, False
        while True:
            if (status := self.due()) is not None:
                return status
            try:
                if not acknowledged:
                    self.bridge.request("report", kind="error", **record)
                    acknowledged = True
                message = self.bridge.request("observe")
            except HALTING:
                # The game may be gone for a while: keep trying, without a new record each time.
                self.sleep(max(self.poll, 1.0))
                continue
            count = message.get("reset_count")
            if count is None:
                return None  # a bridge that cannot halt: go on as before
            if count != waited:
                self.reset_count = count
                self.policy.reset()
                self.previous = None
                return None
            self.sleep(self.poll)


def play(checkpoint, directory, device="cpu", *, request_timeout=120, run_for=None, max_decisions=None):
    """Control the game through the bridge in `directory` until `run_for` seconds
    pass, `max_decisions` are made, or the process is interrupted; returns the counts."""
    if request_timeout <= 0:
        raise ValueError("request_timeout must be positive")
    model, vocabulary, manifest = load_model(checkpoint, device)
    model.eval()
    policy = SessionPolicy(model, vocabulary)
    bridge = SteamBridge(directory, request_timeout)
    try:
        with (
            torch.inference_mode(),
            precision_context(model, manifest["training"]["precision"]),
        ):
            return Controller(bridge, policy, directory=bridge.directory,
                              run_for=run_for, max_decisions=max_decisions).run()
    finally:
        bridge.close()
