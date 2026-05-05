from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path


def load_vast_launch_module():
    spec = importlib.util.spec_from_file_location("vast_launch", Path("scripts/vast_launch.py"))
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_vast_launch_dry_run_redacts_secret_values():
    proc = subprocess.run(
        [
            "python",
            "scripts/vast_launch.py",
            "launch",
            "--dry-run",
            "--repo-url",
            "https://example.invalid/repo.git",
            "--remote-script",
            "scripts/run_remote_smoke_pipeline.sh",
            "--git-ref",
            "codex/test",
            "--b2-key-id",
            "secret-key-id",
            "--b2-app-key",
            "secret-app-key",
            "--b2-bucket",
            "bucket",
            "--wandb-api-key",
            "secret-wandb-key",
            "--order",
            "dph_total",
            "--limit",
            "5",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "secret-key-id" not in proc.stdout
    assert "secret-app-key" not in proc.stdout
    assert "secret-wandb-key" not in proc.stdout
    assert "B2_KEY_ID=<redacted>" in proc.stdout
    assert "B2_APP_KEY=<redacted>" in proc.stdout
    assert "WANDB_API_KEY=<redacted>" in proc.stdout
    assert "-o dph_total --limit 5" in proc.stdout


def test_vast_launch_offer_id_uses_create_instance():
    proc = subprocess.run(
        [
            "python",
            "scripts/vast_launch.py",
            "launch",
            "--dry-run",
            "--repo-url",
            "https://example.invalid/repo.git",
            "--remote-script",
            "scripts/run_remote_smoke_pipeline.sh",
            "--git-ref",
            "codex/test",
            "--offer-id",
            "123456",
            "--disk",
            "32",
            "--order",
            "dph_total",
            "--limit",
            "5",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "vastai create instance 123456" in proc.stdout
    assert "--disk 32" in proc.stdout
    assert "vastai launch instance" not in proc.stdout
    assert "-o dph_total" not in proc.stdout
    assert "--limit 5" not in proc.stdout


def test_run_can_display_redacted_command_without_changing_executed_command(monkeypatch, capsys):
    vast_launch = load_vast_launch_module()
    executed = []

    def fake_run(cmd, **kwargs):
        executed.append(cmd)

        class Result:
            returncode = 0
            stdout = "Started. {'instance_api_key': 'secret-instance-key'}\n"
            stderr = ""

        return Result()

    command = ["vastai", "create", "instance", "1", "--env", "B2_APP_KEY=secret"]
    display = ["vastai", "create", "instance", "1", "--env", "B2_APP_KEY=<redacted>"]

    monkeypatch.setattr(vast_launch.subprocess, "run", fake_run)
    vast_launch.run(command, display_cmd=display)

    assert executed == [command]
    captured = capsys.readouterr()
    assert "B2_APP_KEY=<redacted>" in captured.out
    assert "instance_api_key': '<redacted>'" in captured.out
    assert "secret" not in captured.out
