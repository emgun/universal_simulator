import os
import subprocess
import sys
from pathlib import Path


def test_evaluate_lightning_help():
    script = Path("scripts/evaluate_lightning.py")
    env = {"PYTHONPATH": str(Path("src").resolve())}
    result = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True, env={**os.environ, **env})
    assert result.returncode == 0
    assert "Lightning-native evaluation" in result.stdout
