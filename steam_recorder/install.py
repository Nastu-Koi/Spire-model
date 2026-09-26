#!/usr/bin/env python3
"""Build against Steam's original assemblies; install only this mod, with backup."""

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--game-dir", type=Path, required=True, help="Steam Slay the Spire 2 directory"
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="Install the built DLL and manifest, keeping a backup",
    )
    args = parser.parse_args()
    game = args.game_dir.resolve()
    candidates = [
        game / "data_sts2_linuxbsd_x86_64",
        game / "data_sts2_linux_x86_64",
        game / "data_sts2_windows_x86_64",
        game,
        *sorted(game.glob("data_*/")),
    ]
    data = next(
        (
            p
            for p in candidates
            if all(
                (p / n).is_file()
                for n in ("sts2.dll", "GodotSharp.dll", "0Harmony.dll")
            )
        ),
        None,
    )
    if data is None:
        parser.error(f"No original game assemblies found below {game}")
    for name in ("sts2.dll", "GodotSharp.dll", "0Harmony.dll"):
        if not (data / name).is_file():
            parser.error(f"Missing original game assembly: {data / name}")
    subprocess.run(
        [
            "dotnet",
            "build",
            str(ROOT / "RunRecorder.csproj"),
            "-c",
            "Release",
            "--nologo",
            f"-p:GameDataDir={data}",
        ],
        check=True,
    )
    files = {
        "RunRecorder.dll": ROOT / "bin/Release/net9.0/RunRecorder.dll",
        "RunRecorder.json": ROOT / "RunRecorder.json",
    }
    target = game / "mods/RunRecorder"
    result = {
        "version": json.loads(files["RunRecorder.json"].read_text())["version"],
        "target": str(target),
        "installed": False,
        "files": {
            name: {"sha256": sha256(path), "bytes": path.stat().st_size}
            for name, path in files.items()
        },
    }
    if args.install:
        backup = (
            ROOT.parent
            / "build/recorder-backups"
            / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        )
        backup.mkdir(parents=True, exist_ok=False)
        target.mkdir(parents=True, exist_ok=True)
        originals = {}
        for name in files:
            if (target / name).exists():
                shutil.copy2(target / name, backup / name)
                originals[name] = sha256(backup / name)
        (backup / "backup.json").write_text(
            json.dumps({"target": str(target), "original_sha256": originals}, indent=2)
        )
        installed = []
        try:
            for name, source in files.items():
                temporary = target / (name + ".installing")
                shutil.copy2(source, temporary)
                temporary.replace(target / name)
                installed.append(name)
            for name, source in files.items():
                if sha256(target / name) != sha256(source):
                    raise RuntimeError("Installed file checksum mismatch: " + name)
        except BaseException:
            for name in installed:
                if name in originals:
                    shutil.copy2(backup / name, target / name)
                else:
                    (target / name).unlink()
            raise
        finally:
            for name in files:
                (target / (name + ".installing")).unlink(missing_ok=True)
        result.update(installed=True, backup=str(backup))
        (backup / "installation.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
