"""The execution cursor rejects identity, order and lifecycle corruption."""
import shutil
import subprocess
from pathlib import Path
import pytest


def test_choice_transaction_contract():
    if not shutil.which('dotnet'):
        pytest.skip('.NET SDK required')
    project = Path(__file__).with_name('choice_transaction') / 'check.csproj'
    result = subprocess.run(['dotnet', 'run', '--project', str(project), '-v:q'],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'PASS 33 transaction cases' in result.stdout
