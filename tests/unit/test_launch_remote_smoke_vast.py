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
    assert "pip install -e . --no-deps" in proc.stdout
    assert "pip install h5py numpy PyYAML" in proc.stdout
    assert "pip install -e .[dev]" not in proc.stdout


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


def test_launch_remote_smoke_vast_can_skip_ssh_runtime(tmp_path):
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

    command_line = proc.stdout.splitlines()[0]
    assert "vastai create instance 123456" in command_line
    assert "--ssh" not in command_line
    assert "--onstart" in command_line


def test_launch_remote_smoke_vast_args_mode_skips_onstart_runtime(tmp_path):
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
            "ARGS_MODE": "1",
            "SSH": "0",
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

    command_line = proc.stdout.splitlines()[0]
    assert "vastai create instance 123456" in command_line
    assert "--entrypoint bash --args -lc" in command_line
    assert "--onstart" not in command_line
    assert "--ssh" not in command_line
    assert "secret-key-id" not in proc.stdout
    assert "secret-app-key" not in proc.stdout


def test_launch_remote_smoke_vast_can_pass_experiment_pipeline_args(tmp_path):
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
            "ARGS_MODE": "1",
            "SSH": "0",
            "INSTALL_MODE": "experiment",
            "EXTRA_PIPELINE_ARGS": (
                "PREP_SHARDS=0 RUN_EXPERIMENTS=1 QUEUE_DRY_RUN=0 QUEUE_VARIANTS=current_best"
            ),
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

    assert "pip install h5py numpy PyYAML matplotlib" in proc.stdout
    assert "PREP_SHARDS=0 RUN_EXPERIMENTS=1 QUEUE_DRY_RUN=0" in proc.stdout
    assert "QUEUE_VARIANTS=current_best" in proc.stdout
    assert "pip install -e .[dev]" not in proc.stdout


def test_launch_remote_smoke_vast_can_use_tracked_bootstrap(tmp_path):
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
            "GIT_REF": "main",
            "OFFER_ID": "123456",
            "BOOTSTRAP_MODE": "tracked-script",
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

    assert "--bootstrap-mode tracked-script" not in proc.stdout
    assert "scripts/vast_remote_bootstrap.sh" in proc.stdout
    assert "UPS_SCRIPT_ARGS_B64=" in proc.stdout
    assert "DRY_RUN=0 ENV_FILE=.env PIPELINE_ROOT=" not in proc.stdout
