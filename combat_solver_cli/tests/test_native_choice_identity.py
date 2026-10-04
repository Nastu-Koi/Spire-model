"""Random candidate order must not change the source instance selected by the plan."""
import os
import shutil
import subprocess
from pathlib import Path
import pytest

@pytest.mark.engine
def test_source_identity_with_reordered_random_candidates():
    config = os.environ.get('COMBAT_SOLVER_CONFIG')
    if not config or not shutil.which('dotnet'):
        pytest.skip('Pinned native config and .NET SDK required')
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(['dotnet', 'run', '--project',
        str(root / 'combat_solver_cli/tests/replay_choice/check.csproj'),
        '--', str(root), str(Path(config).resolve()), 'seeker'], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'PASS: reordered candidate source identity preserved' in result.stdout
    assert 'entry=FromCombatPile status=selected mode=explicit cards=DEFEND_DEFECT' in result.stderr

@pytest.mark.engine
@pytest.mark.parametrize('scenario', ['photon-end', 'dagger-end'])
def test_lethal_action_cancels_only_unrequested_choices(scenario):
    config = os.environ.get('COMBAT_SOLVER_CONFIG')
    if not config or not shutil.which('dotnet'):
        pytest.skip('Pinned native config and .NET SDK required')
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(['dotnet', 'run', '--project',
        str(root / 'combat_solver_cli/tests/replay_choice/check.csproj'),
        '--', str(root), str(Path(config).resolve()), scenario], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'PASS: native combat end cancels unrequested selection' in result.stdout
    assert 'choice_action_cancelled_by_combat_end' in result.stderr
    assert 'status=not_opened' in result.stderr

@pytest.mark.engine
def test_empty_native_upgrade_captured_training_action():
    config = os.environ.get('COMBAT_SOLVER_CONFIG')
    if not config or not shutil.which('dotnet'):
        pytest.skip('Pinned native config and .NET SDK required')
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(['dotnet', 'run', '--project',
        str(root / 'combat_solver_cli/tests/replay_choice/check.csproj'),
        '--', str(root), str(Path(config).resolve()), 'empty-upgrade'], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'PASS: empty native upgrade completes captured action' in result.stdout
    assert 'entry=FromHandForUpgrade status=no_candidates mode=automatic cards=' in result.stderr
    assert 'choice_action_verified' in result.stderr
    assert 'choice_transaction_failed' not in result.stderr
