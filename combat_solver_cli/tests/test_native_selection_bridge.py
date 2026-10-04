"""Exercise native automatic and asynchronous branches without card-specific patches."""
import os
import shutil
import subprocess
from pathlib import Path
import pytest

@pytest.mark.engine
def test_native_selection_entry_matrix():
    config = os.environ.get('COMBAT_SOLVER_CONFIG')
    if not config or not shutil.which('dotnet'):
        pytest.skip('Pinned native config and .NET SDK required')
    root = Path(__file__).resolve().parents[2]
    project = Path(__file__).with_name('native_selection') / 'check.csproj'
    build = subprocess.run(['dotnet', 'build', str(project), '--nologo', '-v:q', '-m:1'],
                           capture_output=True, text=True, timeout=60)
    assert build.returncode == 0, build.stdout + build.stderr
    assembly = project.parent / 'bin/Debug/net9.0/CombatSolverCli.SelectionTests.dll'
    result = subprocess.run(['dotnet', str(assembly), str(root), str(Path(config).resolve())],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'PASS 25 native selection branches' in result.stdout
