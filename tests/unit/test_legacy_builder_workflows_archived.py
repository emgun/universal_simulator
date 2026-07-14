from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REMOVED_OPTIONS = (
    "--split-source",
    "--split-start-index",
    "--split-block-size",
    "--split-block-offset",
    "--fallback-source-split",
)


def test_removed_builder_options_exist_only_in_explicitly_archived_workflows():
    offenders = []
    for path in sorted(Path("scripts").iterdir()):
        if path.suffix not in {".py", ".sh"}:
            continue
        text = path.read_text(encoding="utf-8")
        if "make_light_hdf5_shards.py" not in text:
            continue
        if (
            any(option in text for option in REMOVED_OPTIONS)
            and "ARCHIVED_LEGACY_WORKFLOW" not in text
        ):
            offenders.append(str(path))

    assert offenders == []


@pytest.mark.parametrize(
    ("command", "message"),
    [
        ([sys.executable, "scripts/fetch_datasets.py"], "Archived legacy workflow"),
        ([sys.executable, "scripts/build_full_artifacts.py"], "Archived legacy workflow"),
        (
            [sys.executable, "scripts/plan_transport_official_hydration.py"],
            "Archived legacy workflow",
        ),
        (
            [sys.executable, "scripts/run_official_hydrated_post_validation_test.py"],
            "Archived legacy workflow",
        ),
        (
            ["bash", "scripts/run_remote_model_side_beta_head_pretest.sh"],
            "Archived legacy workflow",
        ),
        (
            ["bash", "scripts/run_remote_model_side_transport_head_real_shard.sh"],
            "Archived legacy workflow",
        ),
        (["bash", "scripts/run_remote_medium_confirmation.sh"], "Archived legacy workflow"),
        (["bash", "scripts/run_remote_official_hydration.sh"], "Archived legacy workflow"),
        (["bash", "scripts/run_remote_transport_shift_candidate.sh"], "Archived legacy workflow"),
        (
            ["bash", "scripts/launch_remote_model_side_beta_head_pretest_vast.sh"],
            "Archived legacy workflow",
        ),
        (
            ["bash", "scripts/launch_remote_model_side_transport_head_vast.sh"],
            "Archived legacy workflow",
        ),
        (["bash", "scripts/launch_remote_medium_vast.sh"], "Archived legacy workflow"),
        (
            ["bash", "scripts/launch_remote_transport_shift_candidate_vast.sh"],
            "Archived legacy workflow",
        ),
    ],
)
def test_archived_legacy_workflow_entrypoints_fail_before_work(command, message):
    proc = subprocess.run(command, capture_output=True, text=True)

    assert proc.returncode != 0
    assert message in proc.stderr


@pytest.mark.parametrize(
    "command",
    [
        ["bash", "scripts/run_smoke_shard_prep_b2.sh"],
        ["bash", "scripts/run_remote_smoke_pipeline.sh"],
        ["bash", "scripts/launch_remote_smoke_vast.sh"],
        ["bash", "scripts/run_remote_smoke_persistence_baseline.sh"],
    ],
)
def test_future_smoke_entrypoints_block_before_provider_or_download(command):
    proc = subprocess.run(command, capture_output=True, text=True)

    assert proc.returncode == 2
    assert "Blocked:" in proc.stderr
