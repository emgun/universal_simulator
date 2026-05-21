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
    assert "apt-get" not in proc.stdout
    assert "rclone-current-linux-amd64.zip" in proc.stdout
    assert "codeload.github.com" in proc.stdout


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


def test_vast_launch_experiment_mode_installs_wandb():
    proc = subprocess.run(
        [
            "python",
            "scripts/vast_launch.py",
            "launch",
            "--dry-run",
            "--repo-url",
            "https://example.invalid/repo.git",
            "--remote-script",
            "scripts/run_remote_light_promotion.sh",
            "--git-ref",
            "codex/test",
            "--install-mode",
            "experiment",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "pip install h5py numpy PyYAML matplotlib wandb" in proc.stdout


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


def test_run_redacts_api_key_in_error_url(monkeypatch, capsys):
    vast_launch = load_vast_launch_module()

    def fake_run(cmd, **kwargs):
        class Result:
            returncode = 1
            stdout = ""
            stderr = "Bad Request for url: https://console.vast.ai/api/v0/launch_instance/?api_key=secret-api-key\n"

        return Result()

    monkeypatch.setattr(vast_launch.subprocess, "run", fake_run)
    try:
        vast_launch.run(["vastai", "launch", "instance"])
    except SystemExit:
        pass

    captured = capsys.readouterr()
    assert "api_key=<redacted>" in captured.err
    assert "secret-api-key" not in captured.err


def test_run_retries_transient_vast_cli_dns_failure(monkeypatch, capsys):
    vast_launch = load_vast_launch_module()
    attempts = []

    def fake_run(cmd, **kwargs):
        attempts.append(cmd)

        class Result:
            returncode = 1 if len(attempts) == 1 else 0
            stdout = "created instance\n" if len(attempts) == 2 else ""
            stderr = (
                "NameResolutionError: Failed to resolve 'console.vast.ai'\n"
                if len(attempts) == 1
                else ""
            )

        return Result()

    monkeypatch.setattr(vast_launch.subprocess, "run", fake_run)
    monkeypatch.setattr(vast_launch.time, "sleep", lambda _seconds: None)

    result = vast_launch.run(["vastai", "create", "instance", "1"], retries=1, retry_backoff=0)

    assert result == 0
    assert attempts == [
        ["vastai", "create", "instance", "1"],
        ["vastai", "create", "instance", "1"],
    ]
    captured = capsys.readouterr()
    assert "Retrying command after transient Vast CLI failure" in captured.err


def test_launch_passes_retry_knobs_to_create_command(monkeypatch):
    vast_launch = load_vast_launch_module()
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return 0

    monkeypatch.setattr(vast_launch, "git_remote_url", lambda: "https://example.invalid/repo.git")
    monkeypatch.setattr(vast_launch, "run", fake_run)
    parser = vast_launch.build_parser()
    args = parser.parse_args(
        [
            "launch",
            "--offer-id",
            "123456",
            "--disk",
            "32",
            "--remote-script",
            "scripts/run_remote_official_hydration.sh",
            "--git-ref",
            "codex/test",
            "--launch-retries",
            "3",
            "--launch-retry-backoff",
            "2.5",
        ]
    )

    args.func(args)

    assert calls
    cmd, kwargs = calls[-1]
    assert cmd[:4] == ["vastai", "create", "instance", "123456"]
    assert kwargs["retries"] == 3
    assert kwargs["retry_backoff"] == 2.5
