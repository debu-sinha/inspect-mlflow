"""Exercise entry-point registration, evaluation, artifacts, traces, and Scout together."""

import os
import subprocess
import sys
from pathlib import Path


def test_installed_hooks_end_to_end(tmp_path):
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("MLFLOW_", "INSPECT_MLFLOW_", "OPENAI_", "ANTHROPIC_"))
    }
    env["PYTHONIOENCODING"] = "utf-8"
    script = Path(__file__).resolve().parents[1] / "scripts" / "verify_local.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        env=env,
        text=True,
        encoding="utf-8",
        capture_output=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
