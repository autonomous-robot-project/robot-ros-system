"""Lint and format check of the package with Ruff (configuration in the repository ruff.toml)."""

from pathlib import Path
import shutil
import subprocess

import pytest

PACKAGE_DIR = Path(__file__).resolve().parents[1]


def run_ruff(*arguments):
    ruff = shutil.which('ruff')
    # A missing Ruff fails instead of skipping: a skipped lint would go unnoticed.
    assert ruff is not None, 'ruff not found in PATH (pipx install ruff==0.17.2)'
    return subprocess.run([ruff, *arguments, str(PACKAGE_DIR)], capture_output=True, text=True)


@pytest.mark.linter
def test_ruff_check():
    result = run_ruff('check')
    assert result.returncode == 0, result.stdout


@pytest.mark.linter
def test_ruff_format():
    result = run_ruff('format', '--check', '--diff')
    assert result.returncode == 0, result.stdout
