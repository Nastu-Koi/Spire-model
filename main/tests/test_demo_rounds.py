"""A round's combat demonstrations (`main.demos.rounds`): the solver fights the round's
selection, its won fights are exported and imported, and a step cut off goes on."""

import gzip
import json
from pathlib import Path

import pytest

from main.demos.rounds import EXPORT, IMPORTED, SOLVED, solve_round
from main.tests.test_independent import samples
from main.tests.test_iterate import keep_uniform, world  # noqa: F401  (fixtures)


class Solver:
    """The solver's two commands as the round calls them: `run` reports, `export`
    writes the won fights' samples."""

    def __init__(self, groups=2):
        self.calls, self.groups = [], groups

    def __call__(self, mode, selection, output, workers, log):
        self.calls.append((mode, workers))
        output.mkdir(parents=True, exist_ok=True)
        if mode == "run":
            (output / "summary.json").write_text(json.dumps(dict(boss=dict(fights=1, wins=1))))
        else:
            rows = [r for n in range(self.groups) for r in samples(f"fight-{n}", seed=f"s{n}")]
            with gzip.open(output / EXPORT, "wt", encoding="utf-8") as stream:
                stream.writelines(json.dumps(r) + "\n" for r in rows)


def test_the_solvers_won_fights_become_the_rounds_data_directory(tmp_path):
    (tmp_path / "selection.json").write_text(json.dumps(dict(fights=[], extra=[])))
    solver = Solver()
    found = solve_round(tmp_path, workers=16, log=lambda line: None, run=solver)
    assert solver.calls == [("run", 16), ("export", 16)]
    assert found["data"] == str(tmp_path / IMPORTED) and (tmp_path / IMPORTED / "index.jsonl.gz").is_file()
    assert found["imported"]["accepted_runs"] == 2 and found["imported"]["quarantined"] == 0
    assert found["solver"] == dict(boss=dict(fights=1, wins=1))
    # Done once: the import stands and the solver is not asked again.
    again = Solver()
    assert solve_round(tmp_path, log=lambda line: None, run=again)["imported"] == found["imported"]
    assert again.calls == []


def test_an_import_cut_off_is_made_again(tmp_path):
    (tmp_path / "selection.json").write_text(json.dumps(dict(fights=[], extra=[])))
    (tmp_path / IMPORTED).mkdir()  # no summary: the import did not finish
    (tmp_path / IMPORTED / "stray").write_text("")
    found = solve_round(tmp_path, log=lambda line: None, run=Solver(groups=1))
    assert found["imported"]["accepted_runs"] == 1 and not (tmp_path / IMPORTED / "stray").exists()
    assert (tmp_path / SOLVED / EXPORT).is_file()


def test_a_round_without_a_selection(tmp_path):
    with pytest.raises(FileNotFoundError, match="selection"):
        solve_round(tmp_path, log=lambda line: None, run=Solver())


def test_an_iteration_hands_fights_to_the_solver_and_learns_the_last_rounds_demonstrations(world, keep_uniform):
    """The BC phase: after its advantages every round picks fights for the solver, and
    its update learns the imports of this round and the one before (`demo_buffer` 2);
    an import no later update takes is cleaned up."""
    from main.learn.mixed import ACT, DEMOS
    from main.tests.test_iterate import driver
    from main.tests.test_mixed_batches import demo_directory

    solved = []

    def solve(directory, *, workers, log):
        selection = json.loads((Path(directory) / "selection.json").read_text())
        solved.append((len(selection["fights"]), workers))
        data = demo_directory(Path(directory) / IMPORTED, first=10 * len(solved))
        return dict(data=str(data), solver=None, imported=dict(accepted_runs=6, samples=18, macros=18, quarantined=0))

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("main.iterate.demo_rounds.solve_round", solve)
        state = driver(world, eval_seeds=None, round_demos=4, demo_buffer=2,
                       demo_workers=3, source_shares={ACT: 1, DEMOS: 1}).run(3)
    # The synthetic rounds hold fewer kinds of fight than the quotas ask for: what they have is picked.
    picked = [state["iterations"][str(n)]["demos"]["account"] for n in range(3)]
    assert solved == [(account["picked"], 3) for account in picked] and all(account["picked"] for account in picked)
    assert all(sum(kind["quota"] for kind in account["kinds"].values()) == 4 for account in picked)
    assert all(account["picked"] + sum(kind["short"] for kind in account["kinds"].values()) == 4 for account in picked)
    rounds = world.output / "groups"
    for iteration in range(3):
        entry = state["iterations"][str(iteration)]
        assert entry["demos"]["selection"] == str(rounds / f"round-{iteration}/demos/selection.json")
        taken = [str(rounds / f"round-{number}/demos/imported") for number in range(max(0, iteration - 1), iteration + 1)]
        assert entry["update"]["demos"]["directories"] == taken
    # Read by the updates of its round and the next: round 0's import is gone, the picks stay.
    assert not (rounds / "round-0/demos/imported").exists() and (rounds / "round-2/demos/imported").exists()
    assert all((rounds / f"round-{n}/demos/selection.json").is_file() for n in range(3))
    history = [json.loads(line) for line in (world.output / "history.jsonl").read_text().splitlines()]
    assert [line["metrics"]["demo_fights"] for line in history] == [account["picked"] for account in picked]
    for wrong in (dict(round_demos=-1), dict(round_demos=4, demo_buffer=0),
                  dict(round_demos=4, demo_opening=1.5)):
        with pytest.raises(ValueError):
            driver(world, eval_seeds=None, **wrong)


def test_the_entropy_guard_is_on_the_policys_own_decisions():
    """Demonstrations are not the policy's choices: with them in the batch the guard reads
    the act groups' entropy."""
    from main.iterate import guarded_entropy
    from main.learn.mixed import ACT, DEMOS

    assert guarded_entropy(dict(entropy=0.8, mix={ACT: dict(entropy=0.4), DEMOS: dict(entropy=1.1)})) == 0.4
    assert guarded_entropy(dict(entropy=0.6)) == 0.6
