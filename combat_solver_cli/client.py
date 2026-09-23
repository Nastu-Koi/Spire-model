"""Isolated, serial engine workers with pinned dependencies and bounded I/O."""
import hashlib
import json
import os
from pathlib import Path
import sys

from model.engine import CliEngine
from model.protocol import ProtocolError

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "combat_solver_cli/artifacts/config.json"


class InfrastructureError(RuntimeError):
    """Stop the search and retain the frontier; this is not a bad game branch."""


def configuration(path=DEFAULT_CONFIG):
    try:
        config = json.loads(Path(path).read_text())
        for key in ("solver_dll", "game_dll"):
            if hashlib.sha256(Path(config[key]).read_bytes()).hexdigest() != config[key + "_sha256"]:
                raise InfrastructureError(f"{key} changed; configure and validate again")
        if not Path(config["worker_dll"]).is_file(): raise InfrastructureError("Build the worker first")
        return config
    except (OSError, ValueError, KeyError) as exc:
        raise InfrastructureError(f"Invalid worker configuration: {exc}") from exc


def environment(config):
    return dict(os.environ, STS2_LIB=str(Path(config["game_dll"]).parent),
                COMBAT_SOLVER_DLL=config["solver_dll"],
                COMBAT_SOLVER_DEPENDENCIES=os.pathsep.join(config["dependency_dirs"]))


class SolverEngine(CliEngine):
    def __init__(self, config=DEFAULT_CONFIG, timeout=150):
        config = Path(config).resolve()
        configuration(config)
        launcher = ROOT / "combat_solver_cli/launch.py"
        if not launcher.is_file(): raise InfrastructureError(f"Worker launcher disappeared: {launcher}")
        try:
            super().__init__([sys.executable, str(launcher), str(config)], root=ROOT / "sts2-cli", timeout=timeout)
        except (OSError, ProtocolError, TimeoutError) as exc:
            raise InfrastructureError(f"Worker could not start: {exc}") from exc

    def send(self, command):
        try:
            return super().send(command)
        except (OSError, ProtocolError, TimeoutError) as exc:
            raise InfrastructureError(f"Worker transport failed: {exc}") from exc

    def solver(self, command, frame, *, budget_ms=1000, potions=False, potion_policy=None, reuse_turn_plan=False, beam_width=60, beam_portfolio=True):
        routing = frame["routing"]
        return self.send(dict(cmd=command, budget_ms=budget_ms, potions=potions, reuse_turn_plan=reuse_turn_plan, beam_width=beam_width, beam_portfolio=beam_portfolio,
            **({"potion_policy": potion_policy} if potion_policy else {}),
            **{k: routing[k] for k in ("decision_id", "state_version", "selection_revision") if k in routing}))

    def solve(self, frame, **options): return self.solver("solver_solve", frame, **options)
    def step(self, frame, **options): return self.solver("solver_step", frame, **options)
