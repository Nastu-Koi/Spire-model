"""Execute the captured solver plan against the real game, bypassing search timing."""
import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.engine
@pytest.mark.parametrize('mismatch', [False, True])
def test_implicit_decisions_selection_checks_actual_native_card(mismatch):
    config = os.environ.get('COMBAT_SOLVER_CONFIG')
    if not config or not shutil.which('dotnet'):
        pytest.skip('Pinned native config and .NET SDK required')
    root = Path(__file__).resolve().parents[2]
    project = root / 'combat_solver_cli/tests/native/harness/decisions-harness/check.csproj'
    result = subprocess.run(['dotnet', 'run', '--project', str(project), '--', str(root),
                             str(Path(config).resolve()), 'mismatch' if mismatch else 'match'],
                            capture_output=True, text=True, timeout=60)
    if mismatch:
        assert result.returncode != 0
        assert 'choice_transaction_mismatch' in result.stderr
        assert 'mode=automatic cards=' not in result.stderr
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        assert 'mode=automatic cards=VENERATE' in result.stderr
        assert 'PASS: captured Decisions Decisions choice consumed' in result.stdout


@pytest.mark.engine
@pytest.mark.parametrize('card,explicit', [('NIGHTMARE', False), ('NIGHTMARE', True),
    ('SURVIVOR', False), ('ARMAMENTS', False), ('BURNING_PACT', False)])
def test_automatic_single_hand_choices_across_effects(card, explicit):
    config = os.environ.get('COMBAT_SOLVER_CONFIG')
    if not config or not shutil.which('dotnet'):
        pytest.skip('Pinned native config and .NET SDK required')
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(['dotnet', 'run', '--project',
                             str(root / 'combat_solver_cli/tests/native/harness/choice-matrix/check.csproj'),
                             '--', str(root), str(Path(config).resolve()), card, 'explicit' if explicit else 'automatic'],
                            capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'PASS ' + card in result.stdout
    if explicit:
        assert 'Card selection pending: 2 options' in result.stderr
        assert 'mode=automatic cards=' not in result.stderr
    elif card == 'NIGHTMARE':
        assert 'mode=automatic cards=DEFEND_SILENT' in result.stderr


@pytest.mark.engine
@pytest.mark.parametrize('mismatch', [False, True])
def test_repeated_hammer_automatic_then_explicit_choice(mismatch):
    config = os.environ.get('COMBAT_SOLVER_CONFIG')
    if not config or not shutil.which('dotnet'):
        pytest.skip('Pinned native config and .NET SDK required')
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(['dotnet', 'run', '--project',
        str(root / 'combat_solver_cli/tests/native/harness/hammer-replay-harness/check.csproj'),
        '--', str(root), str(Path(config).resolve()), 'mismatch' if mismatch else 'match'],
        capture_output=True, text=True, timeout=60)
    if mismatch:
        assert result.returncode != 0
        assert 'choice_transaction_mismatch' in result.stderr
        assert 'mode=automatic cards=' not in result.stderr
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        assert 'mode=automatic cards=LUMINESCE' in result.stderr
        assert 'PASS: repeated hammer implicit and explicit choices consumed' in result.stdout


@pytest.mark.engine
def test_repeated_hammer_preserves_second_instance_selection():
    config = os.environ.get('COMBAT_SOLVER_CONFIG')
    if not config or not shutil.which('dotnet'):
        pytest.skip('Pinned native config and .NET SDK required')
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(['dotnet', 'run', '--project',
        str(root / 'combat_solver_cli/tests/native/harness/hammer-replay-harness/check.csproj'),
        '--', str(root), str(Path(config).resolve()), 'second_instance'],
        capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'PASS: repeated hammer implicit and explicit choices consumed' in result.stdout
    assert 'mode=automatic cards=LUMINESCE' in result.stderr


@pytest.mark.engine
@pytest.mark.parametrize('card,mode', [('NIGHTMARE', 'repeat_auto'),
    ('NIGHTMARE', 'repeat_explicit'), ('SURVIVOR', 'repeat_explicit')])
def test_repeated_choices_automatic_and_explicit_order(card, mode):
    config = os.environ.get('COMBAT_SOLVER_CONFIG')
    if not config or not shutil.which('dotnet'):
        pytest.skip('Pinned native config and .NET SDK required')
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(['dotnet', 'run', '--project',
        str(root / 'combat_solver_cli/tests/native/harness/choice-matrix/check.csproj'),
        '--', str(root), str(Path(config).resolve()), card, mode],
        capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'PASS ' + card in result.stdout
    expected_auto = 2 if mode == 'repeat_auto' else (1 if card == 'SURVIVOR' else 0)
    assert result.stderr.count('mode=automatic cards=') == expected_auto
