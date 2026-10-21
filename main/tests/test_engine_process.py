"""The engine process a `CliEngine` starts, with a stand-in for the engine."""

import json
import sys

from main.env.engine import RUNTIME_SETTINGS, CliEngine

# Says it is ready, then answers every command with the runtime settings it was given.
STAND_IN = """
import json, os, sys
print(json.dumps({"type": "ready"}), flush=True)
for line in sys.stdin:
    print(json.dumps({name: os.environ.get(name) for name in %r}), flush=True)
""" % (sorted(RUNTIME_SETTINGS),)


def test_the_engine_runs_without_the_runtimes_extra_work_unless_told_otherwise(tmp_path, monkeypatch):
    monkeypatch.delenv("DOTNET_TieredPGO", raising=False)
    monkeypatch.delenv("DOTNET_gcConcurrent", raising=False)
    engine = CliEngine([sys.executable, "-c", STAND_IN], root=tmp_path)
    try:
        assert engine.send({"cmd": "settings"}) == RUNTIME_SETTINGS == {"DOTNET_TieredPGO": "0", "DOTNET_gcConcurrent": "0"}
    finally:
        engine.close()
    # A setting of the process's own environment stands.
    monkeypatch.setenv("DOTNET_TieredPGO", "1")
    engine = CliEngine([sys.executable, "-c", STAND_IN], root=tmp_path)
    try:
        assert engine.send({"cmd": "settings"}) == dict(RUNTIME_SETTINGS, DOTNET_TieredPGO="1")
    finally:
        engine.close()
