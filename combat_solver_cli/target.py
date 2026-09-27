"""Serially produce an exact, balanced target of replay-verified trajectories."""

import json
import math
import secrets
import shutil
import sys
import threading
import time
from pathlib import Path

from model.data import load_runs
from model.protocol import CHARACTERS, ProtocolError, fingerprint

from .client import InfrastructureError
from .mcts import search
from .search_support import write_json


def target_quotas(target, characters):
    characters = tuple(characters)
    if type(target) is not int or target < 1:
        raise ValueError("target_trajectories must be a positive integer")
    if not characters or len(set(characters)) != len(characters):
        raise ValueError("characters must be nonempty and unique")
    if any(character not in CHARACTERS for character in characters):
        raise ValueError("Unsupported character")
    base, remainder = divmod(target, len(characters))
    return {
        character: base + (index < remainder)
        for index, character in enumerate(characters)
    }


def target_schedule(quotas):
    return [
        character
        for index in range(max(quotas.values(), default=0))
        for character, quota in quotas.items()
        if index < quota
    ]


def _new_seed(used):
    while True:
        seed = secrets.token_hex(8).upper()
        if seed not in used:
            return seed


def _write_aggregate(parts, destination):
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as output:
        for part in parts:
            with part.open("rb") as stream:
                shutil.copyfileobj(stream, output)
    temporary.replace(destination)


def _completed_state(manifest, parts):
    completed = {character: 0 for character in manifest["quotas"]}
    results = []
    for part in parts:
        index = int(part.stem)
        game = next(game for game in manifest["games"] if game["index"] == index)
        completed[game["character"]] += 1
        results.append(
            {
                "character": game["character"],
                "seed": game["seed"],
                "status": "verified_victory",
                "segments": len(game["segments"]),
            }
        )
    return completed, results


def _publish_verified(segment, part, character, ascension):
    runs = load_runs(segment / "accepted.jsonl")
    expected = f"A{ascension}_final_boss_victory"
    if (
        len(runs) != 1
        or runs[0]["character"] != character
        or runs[0]["provenance"].get("verified_outcome") != expected
    ):
        raise ValueError("Winning segment did not contain one matching verified run")
    temporary = part.with_suffix(".jsonl.tmp")
    shutil.copy2(segment / "accepted.jsonl", temporary)
    temporary.replace(part)


def _recover_segments(game_record, segments, manifest_path, manifest):
    """Reconcile segment directories created before a manifest write completed."""
    while True:
        index = len(game_record["segments"])
        segment = segments / f"{index:04d}"
        if not segment.is_dir():
            break
        summary = segment / "summary.json"
        try:
            result = (
                json.loads(summary.read_text())
                if summary.is_file()
                else {"status": "interrupted"}
            )
        except (OSError, ValueError):
            result = {"status": "interrupted"}
        game_record["segments"].append(
            {"index": index, "output": str(segment), "result": result}
        )
        write_json(manifest_path, manifest)


def generate_target(
    config,
    output,
    target_trajectories,
    *,
    characters=("Ironclad",),
    workers=1,
    search_fn=search,
    resume_existing=False,
    max_attempts=1000,
    max_total_seconds=3600,
    progress=True,
    progress_interval=10,
    **options,
):
    if workers != 1:
        raise ValueError("Only one game may be searched at a time")
    if options.get("search_lanes", 1) not in (1, 2, 4):
        raise ValueError("search_lanes must be 1, 2 or 4")
    if type(max_attempts) is not int or max_attempts < 1:
        raise ValueError("max_attempts must be a positive integer")
    if any(
        type(v) not in (int, float) or not math.isfinite(v) or v <= 0
        for v in (max_total_seconds, progress_interval)
    ):
        raise ValueError("Time budgets must be positive and finite")
    started = time.monotonic()
    deadline = started + max_total_seconds
    options.setdefault("ascension", 0)
    ascension = options["ascension"]
    if type(ascension) is not int or not 0 <= ascension <= 10:
        raise ValueError("ascension must be an integer from 0 to 10")
    quotas = target_quotas(target_trajectories, characters)
    schedule = target_schedule(quotas)
    output = Path(output)
    manifest_path = output / "manifest.json"
    if resume_existing:
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest.get("mode") != "mcts_target_v1"
            or manifest.get("algorithm") != "mcts"
            or manifest.get("target_trajectories") != target_trajectories
            or manifest.get("quotas") != quotas
            or manifest.get("config") != str(config)
            or manifest.get("options") != options
        ):
            raise ValueError(
                "Target manifest is incompatible with MCTS or current options"
            )
        (output / "STOP").unlink(missing_ok=True)
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output / "accepted-parts").mkdir()
        (output / "games").mkdir()
        manifest = {
            "mode": "mcts_target_v1",
            "algorithm": "mcts",
            "config": str(config),
            "target_trajectories": target_trajectories,
            "characters": list(characters),
            "quotas": quotas,
            "workers": 1,
            "seed_source": "system_random_per_trajectory",
            "options": options,
            "max_attempts": max_attempts,
            "max_total_seconds": max_total_seconds,
            "progress": progress,
            "progress_interval": progress_interval,
            "games": [],
            "invocations": [],
        }
        write_json(manifest_path, manifest)
        (output / "accepted.jsonl").touch()
    parts_dir, games = output / "accepted-parts", output / "games"
    part_paths = sorted(parts_dir.glob("*.jsonl"))
    if resume_existing:
        _write_aggregate(part_paths, output / "accepted.jsonl")
    completed, results = _completed_state(manifest, part_paths)
    used = {game["seed"] for game in manifest["games"]}
    attempts = 0
    budget_stopped = False

    def report(event, *, seed=None, status=None):
        if not progress:
            return
        message = (
            f"[{time.monotonic() - started:.1f}s] {event} "
            f"attempts={attempts}/{max_attempts} "
            f"wins={len(part_paths)}/{target_trajectories}"
        )
        if seed is not None:
            message += f" seed={seed}"
        if status is not None:
            message += f" status={status}"
        print(message, file=sys.stderr, flush=True)

    def finish(status, error=None, active_game=None):
        elapsed = time.monotonic() - started
        invocation = {"wall_seconds": elapsed, "attempts": attempts}
        manifest.setdefault("invocations", []).append(invocation)
        write_json(manifest_path, manifest)
        all_segments = [
            entry["result"] for game in manifest["games"] for entry in game["segments"]
        ]
        counts = {
            state: sum(result.get("status") == state for result in all_segments)
            for state in sorted({result.get("status") for result in all_segments})
        }
        counts.update(
            {
                category: sum(
                    result.get("error_category") == category for result in all_segments
                )
                for category in (
                    "native_error",
                    "protocol_error",
                    "replay_rejected",
                    "other_error",
                )
            }
        )
        total_wall = sum(item["wall_seconds"] for item in manifest["invocations"])
        summary = {
            "status": status,
            "target_trajectories": target_trajectories,
            "verified_trajectories": len(part_paths),
            "quotas": quotas,
            "completed": completed,
            "workers": 1,
            "search_lanes": options.get("search_lanes", 1),
            "attempts": sum(item["attempts"] for item in manifest["invocations"]),
            "outcome_counts": counts,
            "wall_seconds": total_wall,
            "invocation_wall_seconds": elapsed,
            "budget_overrun_seconds": max(0.0, elapsed - max_total_seconds),
            "wall_seconds_per_verified_victory": total_wall / len(part_paths)
            if part_paths
            else None,
            "unfinished_attempt_seconds": sum(
                result.get("segment_wall_seconds", result.get("seconds", 0))
                for result in all_segments
                if result.get("status") != "verified_victory"
            ),
            "results": results,
        }
        if error:
            summary["error"] = error
        if active_game:
            summary["active_game"] = active_game
        write_json(output / "summary.json", summary)
        report("stopped", status=status)
        return summary

    report("started")

    for trajectory_index in range(len(part_paths), len(schedule)):
        if (output / "STOP").exists():
            break
        if attempts >= max_attempts or time.monotonic() >= deadline:
            budget_stopped = True
            break
        character = schedule[trajectory_index]
        existing = next(
            (g for g in manifest["games"] if g["index"] == trajectory_index), None
        )
        if existing:
            game_record = existing
            if existing["character"] != character:
                raise ValueError("Saved target schedule differs")
            seed = existing["seed"]
        else:
            seed = _new_seed(used)
            used.add(seed)
            game_record = {
                "index": trajectory_index,
                "character": character,
                "seed": seed,
                "target_for_character": quotas[character],
                "segments": [],
            }
            manifest["games"].append(game_record)
            write_json(manifest_path, manifest)
        game = games / f"{trajectory_index:05d}-{character}-{fingerprint(seed)[:12]}"
        segments = game / "segments"
        segments.mkdir(parents=True, exist_ok=True)
        _recover_segments(game_record, segments, manifest_path, manifest)
        part = parts_dir / f"{trajectory_index:05d}.jsonl"
        recorded_win = next(
            (
                entry
                for entry in reversed(game_record["segments"])
                if entry["result"].get("status") == "verified_victory"
            ),
            None,
        )
        if recorded_win:
            _publish_verified(Path(recorded_win["output"]), part, character, ascension)
            part_paths.append(part)
            _write_aggregate(part_paths, output / "accepted.jsonl")
            completed[character] += 1
            results.append(
                {
                    "character": character,
                    "seed": seed,
                    "status": "verified_victory",
                    "segments": len(game_record["segments"]),
                }
            )
            continue
        prior = next(
            (
                Path(entry["output"]) / "tree.json"
                for entry in reversed(game_record["segments"])
                if (Path(entry["output"]) / "tree.json").is_file()
            ),
            None,
        )
        if (
            game_record["segments"]
            and prior is None
            and any(
                entry["result"].get("status") != "interrupted"
                for entry in game_record["segments"]
            )
        ):
            return finish(
                "blocked",
                "Active seed ended without a resumable MCTS tree",
                game_record,
            )
        resume = prior
        segment_index = len(game_record["segments"])
        while True:
            if attempts >= max_attempts or time.monotonic() >= deadline:
                budget_stopped = True
                break
            segment = segments / f"{segment_index:04d}"
            stop_monitor = threading.Event()

            def propagate_stop(stop_monitor=stop_monitor, segment=segment):
                while not stop_monitor.wait(0.25):
                    if (output / "STOP").exists() or time.monotonic() >= deadline:
                        if segment.is_dir():
                            (segment / "STOP").touch()
                        return

            monitor = threading.Thread(target=propagate_stop, daemon=True)
            monitor.start()
            attempts += 1
            segment_started = time.monotonic()
            if progress:

                def report_waiting(stop_monitor=stop_monitor, seed=seed):
                    while not stop_monitor.wait(progress_interval):
                        report("waiting", seed=seed)

                progress_thread = threading.Thread(target=report_waiting, daemon=True)
                progress_thread.start()
            else:
                progress_thread = None
            try:
                result = search_fn(
                    config, character, seed, segment, resume=resume, **options
                )
            except Exception as exc:  # noqa: BLE001 - persist cost and checkpoint before stopping
                result = {
                    "status": "job_error",
                    "error": str(exc),
                    "error_category": (
                        "native_error"
                        if isinstance(exc, (InfrastructureError, OSError))
                        else "protocol_error"
                        if isinstance(exc, ProtocolError)
                        else "other_error"
                    ),
                }
            finally:
                stop_monitor.set()
                monitor.join(timeout=1)
                if progress_thread:
                    progress_thread.join(timeout=1)
            if result.get("status") == "infrastructure_error":
                result.setdefault("error_category", "native_error")
            elif result.get("status") == "replay_failed":
                result.setdefault("error_category", "replay_rejected")
            elif result.get("status") in {"lane_error", "job_error"}:
                result.setdefault("error_category", "other_error")
            result_record = {k: v for k, v in result.items() if k != "lanes"}
            result_record["segment_wall_seconds"] = time.monotonic() - segment_started
            game_record["segments"].append(
                {
                    "index": segment_index,
                    "output": str(segment),
                    "result": result_record,
                }
            )
            write_json(manifest_path, manifest)
            report("finished", seed=seed, status=result.get("status"))
            if result.get("status") in {
                "job_error",
                "infrastructure_error",
                "lane_error",
                "replay_failed",
                "tree_exhausted",
                "tree_exhausted_with_unresolved",
            }:
                return finish(
                    "blocked",
                    f"Search segment ended with {result['status']}",
                    game_record,
                )
            if result.get("status") == "verified_victory":
                _publish_verified(segment, part, character, ascension)
                part_paths.append(part)
                _write_aggregate(part_paths, output / "accepted.jsonl")
                completed[character] += 1
                results.append(
                    {
                        "character": character,
                        "seed": seed,
                        "status": "verified_victory",
                        "segments": segment_index + 1,
                    }
                )
                break
            frontier = segment / "tree.json"
            if result.get("status") in {"stopped", "interrupted"}:
                if not frontier.is_file():
                    return finish(
                        "blocked",
                        "Stopped search segment has no resumable MCTS tree",
                        game_record,
                    )
                return finish(
                    "budget_exhausted" if time.monotonic() >= deadline else "stopped"
                )
            if (output / "STOP").exists():
                break
            if not frontier.is_file():
                return finish(
                    "blocked",
                    "Active seed ended without a resumable MCTS tree",
                    game_record,
                )
            resume = frontier
            segment_index += 1
        if (output / "STOP").exists() or budget_stopped:
            break

    complete = len(part_paths) == target_trajectories
    return finish(
        "complete"
        if complete
        else ("budget_exhausted" if budget_stopped else "stopped")
    )


def resume_target(output, search_fn=search):
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest.get("mode") != "mcts_target_v1" or manifest.get("algorithm") != "mcts":
        raise ValueError("Not a compatible MCTS target trajectory output")
    return generate_target(
        manifest["config"],
        output,
        manifest["target_trajectories"],
        characters=manifest["characters"],
        workers=1,
        search_fn=search_fn,
        resume_existing=True,
        max_attempts=manifest.get("max_attempts", 1000),
        max_total_seconds=manifest.get("max_total_seconds", 3600),
        progress=manifest.get("progress", True),
        progress_interval=manifest.get("progress_interval", 10),
        **manifest["options"],
    )
