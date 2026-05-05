from __future__ import annotations

import subprocess


def test_remote_shard_prep_dry_run_supports_source_key_overrides():
    proc = subprocess.run(
        [
            "bash",
            "scripts/run_remote_shard_prep_b2.sh",
        ],
        check=True,
        capture_output=True,
        env={
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "DRY_RUN": "1",
            "TASKS": "burgers1d",
            "BURGERS1D_SOURCE_SPLITS": "train",
            "BURGERS1D_TRAIN_SOURCE_KEYS": "burgers1d/burgers1d_train_000.h5",
        },
        text=True,
    )

    assert "source_splits=train" in proc.stdout
    assert "fetch full/burgers1d/burgers1d_train_000.h5" in proc.stdout
    assert "burgers1d_val.h5" not in proc.stdout


def test_smoke_shard_prep_wrapper_uses_smoke_defaults():
    proc = subprocess.run(
        [
            "bash",
            "scripts/run_smoke_shard_prep_b2.sh",
        ],
        check=True,
        capture_output=True,
        env={"PATH": "/usr/bin:/bin:/usr/sbin:/sbin"},
        text=True,
    )

    assert "would build smoke-v1 shards" in proc.stdout
    assert "fetch full/burgers1d/burgers1d_train_000.h5" in proc.stdout
    assert "fetch full/advection1d/advection1d_train.h5" in proc.stdout
    assert "fetch full/darcy2d/darcy2d_train.h5" in proc.stdout
    assert "publish data/pdebench_light/*.h5 and docs/demo_smoke_data_manifest.yaml to prefix smoke-v1" in proc.stdout
