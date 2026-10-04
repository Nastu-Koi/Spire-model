"""Headbutt's real automatic selection survives Unceasing Top drawing its target."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.engine
@pytest.mark.parametrize('mode', ['match', 'mismatch', 'wrong_pile', 'wrong_effect', 'wrong_occurrence'])
def test_headbutt_automatic_discard_choice(mode):
    config = os.environ.get('COMBAT_SOLVER_CONFIG')
    if not config or not shutil.which('dotnet'):
        pytest.skip('Pinned native config and .NET SDK required')
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run([
        'dotnet', 'run', '--project',
        str(root / 'combat_solver_cli/tests/native/harness/headbutt-replay-harness/check.csproj'),
        '--', str(root), str(Path(config).resolve()), mode,
    ], capture_output=True, text=True, timeout=60)
    if mode != 'match':
        assert result.returncode != 0
        assert 'choice_transaction_mismatch' in result.stderr
        assert 'choice_action_verified transaction=1' not in result.stderr
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        assert 'entry=FromCombatPile status=selected mode=automatic cards=TAUNT' in result.stderr
        assert 'PASS: captured Headbutt choice consumed' in result.stdout
        assert '[ERROR]' not in result.stderr
