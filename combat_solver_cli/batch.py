"""Bounded independent seed jobs; publish only verified final-boss victories."""

import json
from pathlib import Path

from model.data import load_runs
from model.protocol import CHARACTERS, fingerprint

from .astar import search, write_json


def generate(config, jobs, output, *, workers=1, **options):
    ascension = options.get("ascension", 10)
    if type(ascension) is not int or not 0 <= ascension <= 10:
        raise ValueError("ascension must be an integer from 0 to 10")
    if workers != 1:
        raise ValueError("Only one game may be searched at a time")
    identities = [(j["character"], str(j["seed"])) for j in jobs]
    if not identities or len(set(identities)) != len(identities):
        raise ValueError("Jobs must be nonempty and unique")
    if any(c not in CHARACTERS for c, s in identities):
        raise ValueError("Unsupported character")
    if any(j.get("resume") and j.get("prefix_path") for j in jobs):
        raise ValueError("A job cannot use both resume and prefix_path")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "manifest.json",
        {"config": str(config), "jobs": jobs, "workers": workers, "options": options},
    )
    results, accepted = [], []
    for job, (character, seed) in zip(jobs, identities):
        directory = output / "jobs" / (character + "-" + fingerprint(seed)[:16])
        job_options = dict(options)
        for key in ("resume", "prefix_path"):
            if job.get(key):
                job_options[key] = Path(job[key])
        try:
            result = search(config, character, seed, directory, **job_options)
            if result["status"] == "verified_victory":
                runs = load_runs(directory / "accepted.jsonl")
                if (
                    len(runs) != 1
                    or runs[0]["provenance"].get("verified_outcome")
                    != f"A{ascension}_final_boss_victory"
                ):
                    raise ValueError("Missing verified outcome")
                accepted.extend(runs)
        except Exception as exc:
            result = {
                "character": character,
                "seed": seed,
                "status": "job_error",
                "error": str(exc),
            }
        results.append(result)
        write_json(output / "progress.json", results)
    accepted.sort(key=lambda r: (r["character"], r["seed"]))
    temporary = output / "accepted.jsonl.tmp"
    with temporary.open("x") as stream:
        for run in accepted:
            stream.write(json.dumps(run, allow_nan=False) + "\n")
    temporary.replace(output / "accepted.jsonl")
    summary = {
        "ascension": ascension,
        "verified_victories": len(accepted),
        "unresolved_or_defeated": len(jobs) - len(accepted),
        "jobs": results,
    }
    write_json(output / "summary.json", summary)
    return summary
