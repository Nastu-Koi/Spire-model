import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from combat_solver_cli.client import ROOT, configuration, environment
from model.dotnet_runtime import runtime_command

if __name__ == "__main__":
    config = configuration(sys.argv[1])
    os.chdir(ROOT / "sts2-cli")
    command = runtime_command(config["worker_dll"])
    os.execvpe(command[0], command, environment(config))
