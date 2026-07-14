from __future__ import annotations

import os
import subprocess


def test_compatibility_fetcher_rejects_unlocked_dataset_names(tmp_path):
    env = os.environ.copy()
    env.pop("DATA_LOCK", None)
    env["DATA_ROOT"] = str(tmp_path / "data")

    proc = subprocess.run(
        ["bash", "scripts/fetch_datasets_b2.sh", "guessed-dataset-name"],
        capture_output=True,
        env=env,
        text=True,
    )

    assert proc.returncode == 2
    assert "DATA_LOCK is required" in proc.stderr
    assert "fuzzy B2 dataset-name hydration has been retired" in proc.stderr
    assert not (tmp_path / "data").exists()
