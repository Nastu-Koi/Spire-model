"""Locate a .NET host with the runtime required by the native game adapter."""

import os
import re
import shutil
import subprocess
from pathlib import Path

_PROBE_TIMEOUT_SECONDS = 5
_RUNTIME_LINE = re.compile(r"^Microsoft\.NETCore\.App 9\.\d+\.\d+ \[")
_SDK_LINE = re.compile(r"^(\d+)\.\d+\.\d+ \[")


def _candidates(*, include_override: bool):
    if include_override and (override := os.environ.get("STS2_DOTNET")):
        yield str(Path(override).expanduser().resolve())
    home = Path.home()
    for path in (
        home / ".dotnet-spire/dotnet",
        home / ".dotnet-arm64/dotnet",
        home / ".dotnet/dotnet",
    ):
        yield str(path)
    if system_dotnet := shutil.which("dotnet"):
        yield str(Path(system_dotnet).resolve())


def _probe(binary: str, argument: str) -> str | None:
    if not Path(binary).is_file() or not os.access(binary, os.X_OK):
        return None
    try:
        result = subprocess.run(
            [binary, argument],
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout if result.returncode == 0 else None


def find_runtime() -> str | None:
    """Return a host with a stable Microsoft.NETCore.App 9 runtime.

    An explicit STS2_DOTNET is authoritative: an invalid selection fails early
    instead of silently running the game through another installation.
    """
    override = os.environ.get("STS2_DOTNET")
    if override is not None and not override.strip():
        raise RuntimeError("STS2_DOTNET is set but empty")
    seen = set()
    for binary in _candidates(include_override=True):
        if binary in seen:
            continue
        seen.add(binary)
        installed = _probe(binary, "--list-runtimes")
        if installed and any(_RUNTIME_LINE.match(line) for line in installed.splitlines()):
            return binary
        if override:
            raise RuntimeError(
                f"STS2_DOTNET={override!r} does not provide a stable "
                "Microsoft.NETCore.App 9 runtime"
            )
    return None


def find_sdk() -> str | None:
    """Return a host with a stable .NET 9 or newer SDK to build net9.0."""
    seen = set()
    for binary in _candidates(include_override=True):
        if binary in seen:
            continue
        seen.add(binary)
        installed = _probe(binary, "--list-sdks")
        if installed and any(
            (match := _SDK_LINE.match(line)) and int(match.group(1)) >= 9
            for line in installed.splitlines()
        ):
            return binary
    return None


def runtime_command(dll: str | Path, *, runtime: str | None = None) -> list[str]:
    """Run a framework-dependent assembly without rolling into .NET 10."""
    runtime = runtime or find_runtime()
    if runtime is None:
        raise RuntimeError(
            "Microsoft.NETCore.App 9 runtime is required; install .NET 9 "
            "or set STS2_DOTNET to a compatible dotnet executable"
        )
    return [runtime, "--roll-forward", "Minor", str(dll)]
