from __future__ import annotations

import subprocess


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
