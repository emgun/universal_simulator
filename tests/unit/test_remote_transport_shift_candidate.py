from __future__ import annotations

import os
import subprocess


def test_remote_transport_shift_candidate_dry_run_is_local_safe():
    env = os.environ.copy()
    env["DRY_RUN"] = "1"
    proc = subprocess.run(
        ["bash", "scripts/run_remote_transport_shift_candidate.sh"],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    assert "hydrate full advection1d train/val/test" in proc.stdout
    assert "target_shift=40" in proc.stdout
    assert "run train/val gate" in proc.stdout


def test_remote_transport_shift_candidate_dry_run_supports_all_split_scan():
    env = os.environ.copy()
    env["DRY_RUN"] = "1"
    env["SCAN_ALL_SPLITS"] = "1"
    env["REQUIRE_TEST_COMPATIBLE"] = "1"
    proc = subprocess.run(
        ["bash", "scripts/run_remote_transport_shift_candidate.sh"],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    assert "scan advection1d_{train,val,test}.h5" in proc.stdout
    assert "selected train start and held-out val/test starts" in proc.stdout
