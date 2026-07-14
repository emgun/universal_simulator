from __future__ import annotations

import os
import subprocess


def test_remote_transport_shift_candidate_is_archived():
    env = os.environ.copy()
    env["DRY_RUN"] = "1"
    proc = subprocess.run(
        ["bash", "scripts/run_remote_transport_shift_candidate.sh"],
        capture_output=True,
        env=env,
        text=True,
    )

    assert proc.returncode == 2
    assert "Archived legacy workflow" in proc.stderr


def test_remote_transport_shift_candidate_cannot_enable_all_split_scan():
    env = os.environ.copy()
    env["DRY_RUN"] = "1"
    env["SCAN_ALL_SPLITS"] = "1"
    env["REQUIRE_TEST_COMPATIBLE"] = "1"
    proc = subprocess.run(
        ["bash", "scripts/run_remote_transport_shift_candidate.sh"],
        capture_output=True,
        env=env,
        text=True,
    )

    assert proc.returncode == 2
    assert "Archived legacy workflow" in proc.stderr


def test_remote_transport_shift_candidate_cannot_request_test_ready_audit():
    env = os.environ.copy()
    env["DRY_RUN"] = "1"
    env["SCAN_ALL_SPLITS"] = "1"
    env["AUDIT_REQUIRE_STATUS"] = "test-ready"
    proc = subprocess.run(
        ["bash", "scripts/run_remote_transport_shift_candidate.sh"],
        capture_output=True,
        env=env,
        text=True,
    )

    assert proc.returncode == 2
    assert "Archived legacy workflow" in proc.stderr
