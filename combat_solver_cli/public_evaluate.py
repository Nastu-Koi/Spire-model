"""Evaluate frozen public-policy configurations on explicit game seeds.

Example: python3 -m combat_solver_cli.public_evaluate --manifest FILE \
    --split dev --output DIRECTORY
"""

import argparse
import hashlib
import json
import math
import time
from itertools import pairwise
from pathlib import Path

from .public_generate import generate


def _manifest(manifest):
    if isinstance(manifest, (str, Path)):
        manifest = json.loads(Path(manifest).read_text())
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != "public-evaluation-v1"
    ):
        raise ValueError("Expected public-evaluation-v1 manifest")
    seeds = manifest.get("seeds")
    if not isinstance(seeds, dict) or set(seeds) != {"dev", "holdout"}:
        raise ValueError("Manifest needs dev and holdout seeds")
    for split, values in seeds.items():
        if (
            not isinstance(values, list)
            or not values
            or any(not isinstance(value, str) or not value for value in values)
            or len(set(values)) != len(values)
        ):
            raise ValueError(f"Invalid {split} seed list")
    if set(seeds["dev"]) & set(seeds["holdout"]):
        raise ValueError("Development and holdout seeds overlap")
    profiles = manifest.get("profiles")
    if not isinstance(profiles, list) or not profiles:
        raise ValueError("Manifest needs profiles")
    from .public_effects import CombatModel

    for profile in profiles:
        if not isinstance(profile, dict) or set(profile) != {
            "name",
            "depth",
            "beam_width",
            "effect_profile",
        }:
            raise ValueError(
                "Each profile needs name, depth, beam_width, effect_profile"
            )
        if (
            type(profile["depth"]) is not int
            or not 1 <= profile["depth"] <= 12
            or type(profile["beam_width"]) is not int
            or not 1 <= profile["beam_width"] <= 256
            or profile["effect_profile"] not in CombatModel.profiles
        ):
            raise ValueError("Invalid public planner profile")
    names = [profile.get("name") for profile in profiles]
    if any(not isinstance(name, str) or not name for name in names) or len(
        set(names)
    ) != len(names):
        raise ValueError("Profile names must be unique strings")
    if type(manifest.get("repetitions")) is not int or manifest["repetitions"] < 1:
        raise ValueError("repetitions must be positive")
    if type(manifest.get("sampling_seed")) is not int:
        raise ValueError("sampling_seed must be an integer")
    budget = manifest.get("budget") or {}
    if (
        type(budget.get("max_seconds_per_run")) not in (int, float)
        or not math.isfinite(budget["max_seconds_per_run"])
        or budget["max_seconds_per_run"] <= 0
        or type(budget.get("max_steps")) is not int
        or budget["max_steps"] < 1
    ):
        raise ValueError("Invalid evaluation budget")
    stop = manifest.get("stop") or {}
    if (
        stop.get("complete_all_cases") is not True
        or type(stop.get("abandon")) is not bool
    ):
        raise ValueError("Evaluation must complete all scheduled cases and set abandon")
    if (
        not isinstance(manifest.get("game_version"), str)
        or not manifest["game_version"]
    ):
        raise ValueError("game_version is required")
    return manifest


def _sampling_seed(master, split, seed_index):
    """Use schedule position, never the actual game seed, to seed model sampling."""
    payload = json.dumps([master, split, seed_index]).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _decisions(run_dir):
    path = run_dir / "attempts/000001/decisions.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def _final_observation(run_dir):
    from model.representation import clean_public

    path = run_dir / "attempts/000001/result.json"
    if not path.is_file():
        return None
    frame = json.loads(path.read_text()).get("prefix", {}).get("last_frame")
    if not frame:
        return None
    return {"boundary": frame["boundary"], "public": clean_public(frame["public"])}


def _first_difference(left, right):
    for step, (a, b) in enumerate(zip(left, right), 1):
        fields = [
            name
            for name, key in (("public", "frame"), ("action", "record"))
            if (a[key]["public"] if name == "public" else a[key]["action"])
            != (b[key]["public"] if name == "public" else b[key]["action"])
        ]
        if fields:
            return {"step": step, "fields": fields}
    if len(left) != len(right):
        return {"step": min(len(left), len(right)) + 1, "fields": ["length"]}
    return None


def _combat_snapshot(public):
    entities = public.get("entities", [])
    player = next(
        (entity for entity in entities if entity.get("entity_type") == "player"), {}
    )
    enemies = {
        entity["ref"]: entity
        for entity in entities
        if entity.get("entity_type") == "enemy" and entity.get("ref")
    }
    return {
        "phase": public.get("phase"),
        "act": player.get("act"),
        "floor": player.get("floor"),
        "round": player.get("round"),
        "player_hp": player.get("hp"),
        "player_block": player.get("block"),
        "player_energy": player.get("energy"),
        "enemy_hp": {ref: entity.get("hp") for ref, entity in enemies.items()},
        "enemy_block": {ref: entity.get("block") for ref, entity in enemies.items()},
        "enemy_content": {
            ref: entity.get("content_id") for ref, entity in enemies.items()
        },
    }


def _delta(original, observed):
    delta = {}
    for key in ("player_hp", "player_block", "player_energy"):
        left, right = original[key], observed[key]
        delta[key] = (
            right - left
            if type(left) in (int, float) and type(right) in (int, float)
            else None
        )
    before_content = original["enemy_content"]
    after_content = observed["enemy_content"]
    comparable_enemies = (
        before_content == after_content
        and all(
            isinstance(content, str) and content for content in before_content.values()
        )
        and len(set(before_content.values())) == len(before_content)
    )
    for key in ("enemy_hp", "enemy_block"):
        if not comparable_enemies:
            delta[key] = None
            continue
        delta[key] = {
            ref: observed[key].get(ref, 0) - amount
            for ref, amount in original[key].items()
            if type(amount) in (int, float)
            and type(observed[key].get(ref, 0)) in (int, float)
        }
    return delta


def _prediction(before, profile, original):
    from .public_effects import CombatModel

    action = before["record"]["action"]
    model = CombatModel(before["frame"], effect_profile=profile)
    prediction = model.predict(action).to_dict()
    state = prediction["state"]
    if state is None:
        return prediction["coverage"], prediction["reason"], None
    predicted = {
        **{key: original[key] for key in ("phase", "act", "floor", "round")},
        "player_hp": state["hp"],
        "player_block": state["block"],
        "player_energy": state["energy"],
        "enemy_hp": {ref: values[0] for ref, values in state["enemies"].items()},
        "enemy_block": {ref: values[1] for ref, values in state["enemies"].items()},
        "enemy_content": original["enemy_content"],
    }
    return prediction["coverage"], prediction["reason"], _delta(original, predicted)


def _prediction_error(observed, predicted):
    error = {}
    for key in ("player_hp", "player_block", "player_energy"):
        left, right = predicted[key], observed[key]
        error[key] = left - right if left is not None and right is not None else None
    for key in ("enemy_hp", "enemy_block"):
        if not isinstance(predicted[key], dict) or not isinstance(observed[key], dict):
            error[key] = None
            continue
        error[key] = {
            ref: amount - observed[key][ref]
            for ref, amount in predicted[key].items()
            if ref in observed[key]
        }
    return error


def _effect_rows(decisions, profile, terminal=None):
    """Net public observations across adjacent boundaries, not causal engine traces."""
    rows = []
    pairs = list(pairwise(decisions))
    if decisions and terminal and terminal["boundary"] == "terminal":
        pairs.append((decisions[-1], {"frame": terminal}))
    for index, (before, after) in enumerate(pairs, 1):
        original = _combat_snapshot(before["frame"]["public"])
        observed = _combat_snapshot(after["frame"]["public"])
        if (
            original["phase"] != "combat"
            or observed["phase"] != "combat"
            or before["record"]["action"]["verb"] == "END_TURN"
            or any(original[key] != observed[key] for key in ("act", "floor", "round"))
        ):
            continue
        delta = _delta(original, observed)
        try:
            coverage, reason, predicted_delta = _prediction(before, profile, original)
            prediction_exception = None
        except Exception as exc:  # noqa: BLE001 - preserve native evidence if model diagnostics fail
            coverage, reason, predicted_delta = "unknown", "prediction_exception", None
            prediction_exception = f"{type(exc).__name__}: {exc}"
        rows.append(
            {
                "step": index,
                "scope": "next_public_boundary_same_combat_round",
                "attribution": "net_observed_delta_includes_automatic_effects",
                "action": before["record"]["action"],
                "before": original,
                "after": observed,
                "observed_delta": delta,
                "enemy_alignment": (
                    "reference_and_unique_content_stable"
                    if delta["enemy_hp"] is not None
                    else "not_comparable"
                ),
                "prediction_coverage": coverage,
                "prediction_reason": reason,
                "predicted_delta": predicted_delta,
                "prediction_error": _prediction_error(delta, predicted_delta)
                if predicted_delta is not None
                else None,
                "prediction_exception": prediction_exception,
            }
        )
    return rows


def _effect_comparison(rows):
    comparison = {
        coverage: {"zero_error": 0, "nonzero_error": 0, "not_comparable": 0}
        for coverage in ("modeled", "approximate", "unknown")
    }
    for row in rows:
        coverage = row["prediction_coverage"]
        group = comparison.setdefault(
            coverage, {"zero_error": 0, "nonzero_error": 0, "not_comparable": 0}
        )
        error = row["prediction_error"]
        if coverage == "unknown" or row["prediction_exception"] or error is None:
            group["not_comparable"] += 1
            continue
        values = [
            number
            for value in error.values()
            for number in (value.values() if isinstance(value, dict) else [value])
            if type(number) in (int, float) and math.isfinite(number)
        ]
        if not values or (
            row["observed_delta"]["enemy_hp"] is None
            and not any(abs(value) > 1e-9 for value in values)
        ):
            group["not_comparable"] += 1
        elif any(abs(value) > 1e-9 for value in values):
            group["nonzero_error"] += 1
        else:
            group["zero_error"] += 1
    return comparison


def _player(public):
    return next(
        (
            entity
            for entity in public.get("entities", [])
            if entity.get("entity_type") == "player"
        ),
        {},
    )


def _trajectory_metrics(decisions, terminal, result):
    observations = [row["frame"]["public"] for row in decisions]
    if terminal and terminal["boundary"] == "terminal":
        observations.append(terminal["public"])
    hp_loss = 0.0
    observed_edges = 0
    missing_edges = 0
    for before, after in pairwise(observations):
        if before.get("phase") != "combat":
            continue
        prior, following = _player(before), _player(after)
        if (
            prior.get("act") != following.get("act")
            or prior.get("floor") != following.get("floor")
            or type(prior.get("hp")) not in (int, float)
            or type(following.get("hp")) not in (int, float)
        ):
            missing_edges += 1
            continue
        hp_loss += prior["hp"] - following["hp"]
        observed_edges += 1
    if decisions and (not terminal or terminal["boundary"] != "terminal"):
        missing_edges += decisions[-1]["frame"]["public"].get("phase") == "combat"
    prefix = result.get("prefix", {})
    ledger = prefix.get("last_ledger") or result.get("outcome", {}).get("ledger") or {}
    final_public = (terminal or {}).get("public") or {}
    final = _player(final_public) or _player(
        (prefix.get("last_decision_frame") or {}).get("public", {})
    )
    return {
        "potions_used": sum(
            row["record"]["action"]["verb"] == "USE_POTION" for row in decisions
        ),
        "combat_hp_net_loss_observed": hp_loss,
        "combat_hp_observed_edges": observed_edges,
        "combat_hp_missing_edges": missing_edges,
        "bosses": sorted(ledger.get("bosses", [])),
        "final_act": final.get("act"),
        "final_floor": final.get("floor"),
    }


def evaluate(manifest, output, *, split, engine_factory=None, planner_factory=None):
    """Run every seed/profile/repetition, regardless of early victories."""
    manifest = _manifest(manifest)
    if split not in {"dev", "holdout"}:
        raise ValueError("split must be dev or holdout")
    started = time.monotonic()
    evaluator_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "runs").mkdir()
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    )
    cases = []
    source_hashes = None
    source_hash_mismatch = False
    settings = {}
    if engine_factory is not None:
        settings["engine_factory"] = engine_factory
    if planner_factory is not None:
        settings["planner_factory"] = planner_factory
    for profile in manifest["profiles"]:
        for seed_index, seed in enumerate(manifest["seeds"][split]):
            for repetition in range(manifest["repetitions"]):
                index = len(cases) + 1
                run_dir = output / "runs" / f"{index:06d}"
                sampling_seed = _sampling_seed(
                    manifest["sampling_seed"], split, seed_index
                )
                summary = generate(
                    run_dir,
                    target_trajectories=1,
                    workers=1,
                    max_attempts=1,
                    max_seconds=manifest["budget"]["max_seconds_per_run"],
                    max_steps=manifest["budget"]["max_steps"],
                    depth=profile["depth"],
                    beam_width=profile["beam_width"],
                    planner_options={
                        "sampling_seed": sampling_seed,
                        "effect_profile": profile["effect_profile"],
                    },
                    abandon=manifest["stop"]["abandon"],
                    seed_factory=lambda current=seed: current,
                    **settings,
                )
                if source_hashes is None:
                    source_hashes = {
                        **summary["source_hashes"],
                        "evaluator_sha256": evaluator_hash,
                    }
                elif {
                    key: value
                    for key, value in source_hashes.items()
                    if key != "evaluator_sha256"
                } != summary["source_hashes"]:
                    source_hash_mismatch = True
                decisions = _decisions(run_dir)
                terminal = _final_observation(run_dir)
                if summary["attempts"] == 0:
                    attempt = {"status": "limit", "reason": "deadline_before_attempt"}
                    result = {}
                else:
                    if not summary["results"]:
                        raise RuntimeError(
                            "Submitted evaluation attempt has no outcome"
                        )
                    attempt = summary["results"][0]
                    result = json.loads(
                        (run_dir / "attempts/000001/result.json").read_text()
                    )
                effect_rows = _effect_rows(
                    decisions,
                    profile["effect_profile"],
                    terminal,
                )
                with (run_dir / "effects.jsonl").open("w") as stream:
                    for row in effect_rows:
                        stream.write(
                            json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                        )
                cases.append(
                    {
                        "profile": profile["name"],
                        "seed": seed,
                        "sampling_seed": sampling_seed,
                        "repetition": repetition + 1,
                        "run_dir": f"runs/{index:06d}",
                        "attempts": summary["attempts"],
                        "status": attempt["status"],
                        "reason": attempt.get("reason"),
                        "error_category": attempt.get("error_category"),
                        "effect_rows": len(effect_rows),
                        "effect_comparison": _effect_comparison(effect_rows),
                        **_trajectory_metrics(decisions, terminal, result),
                        "prediction_exceptions": sum(
                            row["prediction_exception"] is not None
                            for row in effect_rows
                        ),
                        "wall_seconds": summary["wall_seconds"],
                    }
                )
    counts = {
        key: 0
        for key in (
            "verified_victory",
            "defeat",
            "abandoned",
            "limit",
            "error",
            "budget_stop",
            "native_error",
            "protocol_error",
            "replay_rejected",
            "other_error",
        )
    }
    for case in cases:
        counts[case["status"]] += 1
        if case["status"] == "limit":
            counts["budget_stop"] += 1
        if case["error_category"]:
            counts[case["error_category"]] += 1
    boss_pass_counts = {
        str(act): sum(act in case["bosses"] for case in cases) for act in (1, 2, 3)
    }
    successes = counts.get("verified_victory", 0)
    source_hash_mismatch |= (
        hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != evaluator_hash
    )
    repeatability = {}
    for profile in manifest["profiles"]:
        divergences = []
        incomplete = False
        for seed in manifest["seeds"][split]:
            group = [
                case
                for case in cases
                if case["seed"] == seed and case["profile"] == profile["name"]
            ]
            baseline = _decisions(output / group[0]["run_dir"])
            for other in group[1:]:
                other_decisions = _decisions(output / other["run_dir"])
                difference = _first_difference(baseline, other_decisions)
                if not difference:
                    first_end = _final_observation(output / group[0]["run_dir"])
                    other_end = _final_observation(output / other["run_dir"])
                    if first_end != other_end:
                        difference = {
                            "step": len(baseline) + 1,
                            "fields": [
                                "terminal_public"
                                if first_end
                                and other_end
                                and first_end["boundary"] == other_end["boundary"]
                                else "final_boundary"
                            ],
                        }
                if group[0]["status"] != other["status"]:
                    difference = difference or {
                        "step": min(len(baseline), len(other_decisions)) + 1,
                        "fields": ["outcome"],
                    }
                if difference:
                    divergences.append(
                        {"seed": seed, "repetition": other["repetition"], **difference}
                    )
            incomplete |= source_hash_mismatch or any(
                case["status"] in {"limit", "error"} for case in group
            )
        repeatability[profile["name"]] = {
            "status": "diverged"
            if divergences
            else "inconclusive"
            if incomplete
            else "untested"
            if manifest["repetitions"] < 2
            else "stable",
            "first_difference": divergences[0] if divergences else None,
            "divergences": divergences,
        }
    elapsed = time.monotonic() - started
    report = {
        "schema": "public-evaluation-report-v1",
        "split": split,
        "game_version": manifest["game_version"],
        "profiles": manifest["profiles"],
        "budget": manifest["budget"],
        "stop": manifest["stop"],
        "workers_per_run": 1,
        "source_hashes": source_hashes,
        "source_hash_mismatch": source_hash_mismatch,
        "manifest_sha256": hashlib.sha256(
            (output / "manifest.json").read_bytes()
        ).hexdigest(),
        "cases": len(cases),
        "counts": counts,
        "boss_pass_counts": boss_pass_counts,
        "combat_hp_scope": (
            "Net player HP loss across adjacent public observations from combat "
            "on the same act/floor, including END_TURN; positive means loss. "
            "Unobserved edges are counted separately."
        ),
        "effect_comparison_scope": (
            "Predicted minus observed net public deltas at the next same-round "
            "boundary; automatic effects can contribute. Unknown predictions "
            "are not comparable."
        ),
        "repeatability": repeatability,
        "wall_seconds": elapsed,
        "wall_seconds_per_verified_victory": elapsed / successes if successes else None,
        "runs": cases,
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=("dev", "holdout"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate(args.manifest, args.output, split=args.split), ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()
