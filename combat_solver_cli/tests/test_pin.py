"""configure keeps its own copy of the solver inside the package's lib directory."""

import combat_solver_cli.__main__ as cli


def test_pin_copies_the_solver_and_manifest_once(tmp_path, monkeypatch):
    source, lib = tmp_path / "steam", tmp_path / "lib"
    source.mkdir()
    (source / "CombatSolver.dll").write_bytes(b"solver")
    (source / "CombatSolver.json").write_text('{"version": "0.44.0"}')
    monkeypatch.setattr(cli, "SOLVER_DIR", lib)

    pinned = cli.pin(source / "CombatSolver.dll")
    assert pinned == lib / "CombatSolver.dll" and pinned.read_bytes() == b"solver"
    assert (lib / "CombatSolver.json").read_text() == '{"version": "0.44.0"}'

    # Pinning the copy that is already in place leaves it alone.
    (source / "CombatSolver.dll").write_bytes(b"updated by steam")
    assert cli.pin(pinned.resolve()) == pinned and pinned.read_bytes() == b"solver"
