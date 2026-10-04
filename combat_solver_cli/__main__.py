"""CombatSolver worker: configure pinned dependencies, verify trajectory prefixes.

python -m combat_solver_cli configure --solver <CombatSolver.dll> \
    --dependency-dir <RitsuLib/compat/0.111.0> --dependency-dir <RitsuLib/shared>
python -m combat_solver_cli verify --prefix <prefix.json> --output <dir>

configure copies the solver into combat_solver_cli/lib and writes lib/config.json;
omit --solver to pin the copy that is already there.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from .client import DEFAULT_CONFIG, ROOT, SOLVER_DIR


def pin(solver):
    """Copy the solver and its manifest into SOLVER_DIR, so a Steam update cannot change them."""
    target = SOLVER_DIR / solver.name
    if solver != target.resolve():
        SOLVER_DIR.mkdir(exist_ok=True)
        for source in (solver, solver.with_suffix(".json")):
            shutil.copy2(source, SOLVER_DIR / source.name)
    return target


def configure(solver, lib, dependency_dir, config):
    solver, lib = solver.resolve(), lib.resolve()
    if not solver.is_file() or not (lib / "sts2.dll").is_file():
        raise FileNotFoundError("Solver or game DLL missing")
    manifest = json.loads(solver.with_suffix(".json").read_text())
    if str(manifest.get("version")) != "0.44.0":
        raise ValueError("Adapter currently supports CombatSolver 0.44.0")
    dependencies = [p.resolve() for p in dependency_dir]
    if any(not p.is_dir() for p in dependencies):
        raise FileNotFoundError("Dependency directory missing")
    solver = pin(solver)
    from model.dotnet_runtime import find_sdk

    sdk = find_sdk()
    if sdk is None:
        raise RuntimeError("A .NET 9 or newer SDK is required to build the worker")
    subprocess.run([sdk, "build", str(ROOT / "combat_solver_cli/CombatSolverCli.csproj"), "--nologo", "-v:q", "-m:1"],
                   check=True)
    result = {
        "solver_dll": str(solver),
        "game_dll": str(lib / "sts2.dll"),
        "dependency_dirs": list(map(str, dependencies)),
        "worker_dll": str(ROOT / "combat_solver_cli/bin/Debug/net9.0/CombatSolverCli.dll"),
    }
    for key in ("solver_dll", "game_dll"):
        result[key + "_sha256"] = hashlib.sha256(Path(result[key]).read_bytes()).hexdigest()
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(json.dumps(result, indent=2) + "\n")
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    x = sub.add_parser("configure", help="Build the worker and pin solver / game DLL hashes")
    x.add_argument("--solver", type=Path, default=SOLVER_DIR / "CombatSolver.dll")
    x.add_argument("--lib", type=Path, default=ROOT / "sts2-cli/lib")
    x.add_argument("--dependency-dir", action="append", type=Path, default=[])
    x.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    x = sub.add_parser("verify", help="Independently replay a prefix in a fresh engine and export it")
    x.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    x.add_argument("--prefix", type=Path, required=True)
    x.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "configure":
        print(configure(args.solver, args.lib, args.dependency_dir, args.config))
        return
    from .trajectory import verify_and_export

    result = verify_and_export(args.config, args.prefix, args.output)
    print(json.dumps({"run_id": result["run_id"], "provenance": result["provenance"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
