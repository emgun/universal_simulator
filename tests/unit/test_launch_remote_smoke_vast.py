from __future__ import annotations

import os
import subprocess


def test_launch_remote_smoke_vast_dry_run_redacts_env_file_secrets(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "B2_KEY_ID=secret-key-id",
                "B2_APP_KEY=secret-app-key",
                "B2_BUCKET=bucket",
            ]
        ),
        encoding="utf-8",
    )

    env = os.environ.copy()
    env.update(
        {
            "ENV_FILE": str(env_file),
            "DRY_RUN": "1",
            "GIT_REF": "codex/test",
        }
    )

    proc = subprocess.run(
        [
            "bash",
            "scripts/launch_remote_smoke_vast.sh",
        ],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    assert "secret-key-id" not in proc.stdout
    assert "secret-app-key" not in proc.stdout
    assert "scripts/run_remote_smoke_pipeline.sh DRY_RUN=0 ENV_FILE=.env" in proc.stdout
    assert "B2_KEY_ID=<redacted>" in proc.stdout
    assert "-o dph_total --limit 10" in proc.stdout


def test_launch_remote_smoke_vast_offer_id_skips_ordered_search(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "B2_KEY_ID=secret-key-id",
                "B2_APP_KEY=secret-app-key",
                "B2_BUCKET=bucket",
            ]
        ),
        encoding="utf-8",
    )

    env = os.environ.copy()
    env.update(
        {
            "ENV_FILE": str(env_file),
            "DRY_RUN": "1",
            "GIT_REF": "codex/test",
            "OFFER_ID": "123456",
        }
    )

    proc = subprocess.run(
        [
            "bash",
            "scripts/launch_remote_smoke_vast.sh",
        ],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    assert "secret-key-id" not in proc.stdout
    assert "secret-app-key" not in proc.stdout
    assert "vastai create instance 123456" in proc.stdout
    assert "-o dph_total" not in proc.stdout
    assert "--limit 10" not in proc.stdout
