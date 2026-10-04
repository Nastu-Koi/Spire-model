"""Hold a real native combat save across boundary polls to expose teardown races."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.engine
@pytest.mark.parametrize('mode', ['final', 'normal'])
def test_post_combat_waits_for_native_combat_won(mode):
    config = os.environ.get('COMBAT_SOLVER_CONFIG')
    if not config or not shutil.which('dotnet'):
        pytest.skip('Pinned native config and .NET SDK required')
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run([
        'dotnet', 'run', '--project',
        str(root / 'combat_solver_cli/tests/native/harness/combat-boundary-harness/check.csproj'),
        '--', str(root), str(Path(config).resolve()), mode,
    ], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    expected = ('real CombatWon confirms victory once' if mode == 'final'
                else 'normal rewards follow CombatWon')
    assert 'PASS: pending native save waits; ' + expected in result.stdout
    assert '[ERROR]' not in result.stderr
