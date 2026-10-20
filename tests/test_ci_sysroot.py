"""Daily CI receipts retain the SDK update path and propagate failures."""
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("exit_code", [0, 17])
def test_daily_receipt_is_forwarded_unchanged(tmp_path, exit_code):
    receipt = "3.0.0~git202610140145.e5ece50-2121"
    invocation = tmp_path / "invocation"
    updater = tmp_path / "sysroot"
    updater.write_text(f'#!/bin/sh\nprintf "%s\\n" "$@" > "{invocation}"\nexit {exit_code}\n')
    updater.chmod(0o755)
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/ci/prepare_sysroot.sh"), receipt],
        env={**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}"},
        capture_output=True, text=True,
    )
    assert result.returncode == exit_code, result.stderr
    assert invocation.read_text().splitlines() == ["update", receipt]
