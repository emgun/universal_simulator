from __future__ import annotations

import os
import subprocess


def test_launch_remote_medium_vast_dry_run_targets_medium_pipeline(tmp_path):
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
            "SSH": "0",
            "ARGS_MODE": "1",
        }
    )

    proc = subprocess.run(
        [
            "bash",
            "scripts/launch_remote_medium_vast.sh",
        ],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    assert "secret-key-id" not in proc.stdout
    assert "secret-app-key" not in proc.stdout
    assert "vastai create instance 123456" in proc.stdout
    assert "--disk 160" in proc.stdout
    assert "--entrypoint bash --args -lc" in proc.stdout
    assert "scripts/run_remote_medium_confirmation.sh" in proc.stdout
    assert "DRY_RUN=0 ENV_FILE=.env PIPELINE_ROOT=reports/demo/remote_medium_pipeline" in (
        proc.stdout
    )
    assert "pip install h5py numpy PyYAML matplotlib wandb" in proc.stdout
