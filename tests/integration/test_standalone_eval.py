"""Integration tests for standalone evaluation."""

from pathlib import Path

import pytest


def test_evaluate_script_exists():
    """Test that evaluate.py script exists and is accessible."""
    eval_script = Path("scripts/evaluate.py")
    assert eval_script.exists(), "evaluate.py script should exist"


def test_run_fast_to_sota_script_exists():
    """Test that run_fast_to_sota.py script exists."""
    script = Path("scripts/run_fast_to_sota.py")
    assert script.exists(), "run_fast_to_sota.py script should exist"


def test_show_training_status_script_exists():
    """Test that show_training_status.py script exists and is executable."""
    script = Path("scripts/show_training_status.py")
    assert script.exists(), "show_training_status.py should exist"

    # Check if executable
    import stat
    mode = script.stat().st_mode
    is_executable = bool(mode & stat.S_IXUSR)
    assert is_executable, "show_training_status.py should be executable"


def test_stage_status_display(tmp_path):
    """Test that show_training_status can display stage status."""
    import sys
    import json

    # Add scripts to path
    sys.path.insert(0, str(Path("scripts")))

    from show_training_status import show_status  # noqa: E402

    # Create test checkpoint directory with stage_status.json
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()

    stage_status = {
        "schema_version": 1,
        "created_at": "2025-10-28T12:00:00Z",
        "stages": {
            "operator": {
                "status": "completed",
                "checkpoint": "operator_ema.pt",
                "started_at": "2025-10-28T12:00:00Z",
                "completed_at": "2025-10-28T12:15:00Z",
                "total_epochs": 25
            },
            "diff_residual": {
                "status": "in_progress",
                "started_at": "2025-10-28T12:16:00Z",
                "epoch": 3,
                "total_epochs": 8
            },
            "consistency_distill": {
                "status": "not_started"
            },
            "steady_prior": {
                "status": "not_started"
            }
        }
    }

    status_file = checkpoint_dir / "stage_status.json"
    with open(status_file, "w") as f:
        json.dump(stage_status, f)

    # Create a dummy checkpoint file
    (checkpoint_dir / "operator_ema.pt").write_bytes(b"dummy checkpoint")

    # Capture output by calling show_status
    # This should not raise an exception
    try:
        show_status(checkpoint_dir)
        success = True
    except Exception as e:
        success = False
        print(f"Error: {e}")

    assert success, "show_status should execute without errors"


def test_metadata_display(tmp_path):
    """Test that show_training_status can display metadata."""
    import sys
    import json

    # Add scripts to path
    sys.path.insert(0, str(Path("scripts")))

    from show_training_status import show_status  # noqa: E402

    # Create test checkpoint directory with metadata.json
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()

    metadata = {
        "trained": True,
        "trained_at": "2025-10-28T13:00:00Z",
        "training_wandb": {
            "id": "test-run-123",
            "name": "test-run",
            "project": "universal-simulator",
            "entity": "test-entity",
            "url": "https://wandb.ai/test-entity/universal-simulator/runs/test-run-123"
        }
    }

    metadata_file = checkpoint_dir / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f)

    # This should not raise an exception
    try:
        show_status(checkpoint_dir)
        success = True
    except Exception:
        success = False

    assert success, "show_status should handle metadata correctly"


def test_missing_checkpoint_dir_handling(tmp_path):
    """Test that show_training_status handles missing checkpoint directory gracefully."""
    import sys

    # Add scripts to path
    sys.path.insert(0, str(Path("scripts")))

    from show_training_status import show_status  # noqa: E402

    nonexistent_dir = tmp_path / "nonexistent"

    # Should exit gracefully
    with pytest.raises(SystemExit):
        show_status(nonexistent_dir)
