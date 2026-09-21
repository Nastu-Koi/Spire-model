"""Decision-protocol driver. The engine alone owns legality and continuations.

This prototype records executions and public events; it is not a PPO dataset or
reward implementation. A policy is never called for a forced action, including
the final commit of a buffered selection.
"""

from dataclasses import dataclass, field
from time import monotonic, sleep
from typing import Callable, Protocol


class Engine(Protocol):
    def send(self, command: dict) -> dict: ...


class EnvironmentError(RuntimeError):
    def __init__(self, code: str, frame: dict):
        super().__init__(code)
        self.code = code
        self.frame = frame


@dataclass
class Execution:
    frame: dict
    candidate_ref: str
    origin: str


@dataclass
class RunTrace:
    executions: list[Execution] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    policy_calls: int = 0
    terminal: dict | None = None


class EnvironmentRunner:
    def __init__(self, engine: Engine, choose: Callable[[dict, list[dict]], str], *,
                 allow_prototype: bool = False, max_steps: int = 10000,
                 waiting_timeout: float = 30, poll_interval: float = 0.01):
        if max_steps < 1 or waiting_timeout <= 0 or poll_interval < 0:
            raise ValueError("Invalid runner limits")
        self.engine = engine
        self.choose = choose
        self.allow_prototype = allow_prototype
        self.max_steps = max_steps
        self.waiting_timeout = waiting_timeout
        self.poll_interval = poll_interval
        self.trace = RunTrace()

    def run(self) -> RunTrace:
        self.trace = RunTrace()
        frame = self.engine.send({"cmd": "advance_to_boundary"})
        waiting_since = None
        steps = 0
        while True:
            if frame.get("type") != "decision_frame":
                raise EnvironmentError("invalid_protocol_response", frame)
            contract = frame.get("contract", {})
            if contract.get("fixed_ascension") != 10:
                raise EnvironmentError("invalid_ascension_contract", frame)
            if not contract.get("training_ready", False) and not self.allow_prototype:
                raise EnvironmentError("prototype_contract_requires_opt_in", frame)
            self.trace.events.extend(frame.get("events", []))
            boundary = frame.get("boundary")
            if boundary == "terminal":
                self.trace.terminal = frame
                return self.trace
            if boundary == "error":
                raise EnvironmentError(frame.get("error", {}).get("code", "engine_error"), frame)
            if steps >= self.max_steps:
                raise EnvironmentError("unresolved_step_limit", frame)
            if boundary == "waiting":
                if waiting_since is None:
                    waiting_since = monotonic()
                if monotonic() - waiting_since >= self.waiting_timeout:
                    raise EnvironmentError("unresolved_wait_timeout", frame)
                sleep(self.poll_interval)
                steps += 1
                frame = self.engine.send({"cmd": "advance_to_boundary"})
                continue
            waiting_since = None
            if boundary != "decision":
                raise EnvironmentError("unknown_boundary", frame)
            candidates = frame.get("legal", {}).get("candidates", [])
            if not candidates:
                raise EnvironmentError("empty_legal_set", frame)
            refs = [candidate["candidate_ref"] for candidate in candidates]
            if len(set(refs)) != len(refs):
                raise EnvironmentError("duplicate_candidate_ref", frame)
            if len(candidates) == 1:
                chosen, origin = refs[0], "forced"
            else:
                # Routing, binary hashes and private engine command bindings are
                # intentionally excluded from the policy interface.
                self.trace.policy_calls += 1
                chosen = self.choose(frame["public"], candidates)
                origin = "policy"
                if chosen not in refs:
                    raise EnvironmentError("policy_selected_unknown_candidate", frame)
            routing = frame["routing"]
            command = {"cmd": "execute_candidate", "decision_id": routing["decision_id"],
                       "state_version": routing["state_version"], "candidate_ref": chosen}
            if "selection_revision" in routing:
                command["selection_revision"] = routing["selection_revision"]
            self.trace.executions.append(Execution(frame, chosen, origin))
            steps += 1
            frame = self.engine.send(command)
