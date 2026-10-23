"""A round's combat demonstrations: the solver's fights of the fights the round picked.

The fights a round of act groups picked for the solver (`main.play.demo_picks`, a
`selection.json` under the round's `demos/`) are fought by the solver
(`python -m spire_codex_data.entry_battles run`), each from its entry save and
replayed step by step on a second native process; the won fights are exported as
independent supervised samples (`entry_battles export`) and imported as a data
directory (`independent.import_samples`) that the update reads its combat
demonstrations from.

The solver runs in processes of its own, a run (character and seed) of the selection
at a time on each of `workers`; it times its searches in milliseconds, so nothing
else should run beside it. A step cut off goes on where it stopped: the solver keeps
the runs it finished (by the identity of its build and settings), and an import that
did not finish is made again.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

from main.demos.independent import import_samples

ROOT = Path(__file__).resolve().parents[2]
SOLVED = "solved"  # the solver's reports, samples and export
IMPORTED = "imported"  # the data directory of the round's demonstrations
EXPORT = "independent-training.jsonl.gz"
WORKERS = 16


def solver(mode, selection, output, workers, log):
    command = [sys.executable, "-m", "spire_codex_data.entry_battles", mode, "--output", str(output)]
    if mode == "run":
        command += ["--selection", str(selection), "--workers", str(workers)]
    log(" ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def imported_summary(directory):
    path = Path(directory) / "summary.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def solve_round(directory, *, workers=WORKERS, import_workers=8, log=print, run=None):
    """Have the solver fight the selection under `directory` (a round's `demos/`) and
    import its won fights; the solver's summary and the import's, with the data
    directory's path. An import already made stands."""
    directory, run = Path(directory), run or solver
    selection, solved, data = directory / "selection.json", directory / SOLVED, directory / IMPORTED
    if not selection.is_file():
        raise FileNotFoundError(f"No selection of fights for the solver in {directory}")
    summary = imported_summary(data)
    if summary is None:
        run("run", selection, solved, workers, log)
        run("export", selection, solved, workers, log)
        if data.exists():
            shutil.rmtree(data)  # an import cut off is made again
        summary = import_samples(solved / EXPORT, data, workers=import_workers)
    path = solved / "summary.json"
    fought = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    return dict(data=str(data), solver=fought, imported=dict(
        accepted_runs=summary["accepted_runs"], samples=summary["samples"], macros=summary["macros"],
        quarantined=sum(summary.get("quarantine_reasons", {}).values())))
