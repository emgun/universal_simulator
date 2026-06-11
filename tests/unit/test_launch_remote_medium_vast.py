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


def test_launch_remote_medium_vast_refuses_real_launch_without_b2(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n" 'echo vast_launch_invoked "$@" >&2\n' "exit 0\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "DRY_RUN": "0",
            "ENV_FILE": str(tmp_path / "missing.env"),
            "REMOTE_SCRIPT": "scripts/run_remote_recipe_sweep.sh",
            "EXTRA_PIPELINE_ARGS": "RUN_SWEEP=1 PUBLISH_SWEEP_ARTIFACTS=1",
            "PATH": f"{fake_bin}:{env['PATH']}",
        }
    )
    for key in ("B2_KEY_ID", "B2_APP_KEY", "B2_BUCKET"):
        env.pop(key, None)

    proc = subprocess.run(
        [
            "bash",
            "scripts/launch_remote_medium_vast.sh",
        ],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )

    assert proc.returncode == 2
    assert "Refusing DRY_RUN=0" in proc.stderr
    assert "B2_KEY_ID" in proc.stderr
    assert "B2_APP_KEY" in proc.stderr
    assert "B2_BUCKET" in proc.stderr
    assert "vast_launch_invoked" not in proc.stderr
