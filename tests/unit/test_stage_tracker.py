"""Unit tests for StageTracker."""

import json

import pytest

from ups.utils.stage_tracker import StageTracker


@pytest.fixture
def temp_checkpoint_dir(tmp_path):
    """Create temporary checkpoint directory."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    return checkpoint_dir


def test_stage_tracker_initialization(temp_checkpoint_dir):
    """Test that stage tracker initializes status file."""
    _tracker = StageTracker(temp_checkpoint_dir)  # noqa: F841 (side effect: creates status file)

    # Check that status file was created
    status_file = temp_checkpoint_dir / "stage_status.json"
    assert status_file.exists()

    # Check that all stages are initialized to not_started
    with open(status_file) as f:
        data = json.load(f)

    assert data["schema_version"] == 1
    assert "created_at" in data
    assert "stages" in data

    for stage in StageTracker.STAGES:
        assert stage in data["stages"]
        assert data["stages"][stage]["status"] == "not_started"


def test_mark_stage_started(temp_checkpoint_dir):
    """Test marking a stage as started."""
    tracker = StageTracker(temp_checkpoint_dir)

    tracker.mark_stage_started("operator", total_epochs=25)

    status = tracker.get_stage_status("operator")
    assert status.status == "in_progress"
    assert status.total_epochs == 25
    assert status.epoch == 0
    assert status.started_at is not None


def test_mark_stage_completed(temp_checkpoint_dir):
    """Test marking a stage as completed."""
    tracker = StageTracker(temp_checkpoint_dir)

    tracker.mark_stage_started("operator", total_epochs=25)
    tracker.mark_stage_completed("operator", checkpoint="operator_ema.pt")

    status = tracker.get_stage_status("operator")
    assert status.status == "completed"
    assert status.checkpoint == "operator_ema.pt"
    assert status.completed_at is not None


def test_mark_stage_failed(temp_checkpoint_dir):
    """Test marking a stage as failed."""
    tracker = StageTracker(temp_checkpoint_dir)

    tracker.mark_stage_started("operator", total_epochs=25)
    tracker.mark_stage_failed("operator", error_message="OOM error")

    status = tracker.get_stage_status("operator")
    assert status.status == "failed"
    assert status.error_message == "OOM error"


def test_get_next_stage_to_run(temp_checkpoint_dir):
    """Test determining next stage to run."""
    tracker = StageTracker(temp_checkpoint_dir)

    # Config with all stages enabled
    cfg = {
        "stages": {
            "operator": {"epochs": 25},
            "diff_residual": {"epochs": 8},
            "consistency_distill": {"epochs": 8},
            "steady_prior": {"epochs": 0},  # Disabled
        }
    }

    # Initially, should start with operator
    assert tracker.get_next_stage_to_run(cfg) == "operator"

    # After operator completes, should move to diff_residual
    tracker.mark_stage_completed("operator", "operator_ema.pt")
    assert tracker.get_next_stage_to_run(cfg) == "diff_residual"

    # After diff_residual completes, should move to consistency_distill
    tracker.mark_stage_completed("diff_residual", "diffusion_residual_ema.pt")
    assert tracker.get_next_stage_to_run(cfg) == "consistency_distill"

    # After consistency_distill completes, steady_prior is disabled, so None
    tracker.mark_stage_completed("consistency_distill", "diffusion_residual_ema.pt")
    assert tracker.get_next_stage_to_run(cfg) is None


def test_get_completed_stages(temp_checkpoint_dir):
    """Test getting list of completed stages."""
    tracker = StageTracker(temp_checkpoint_dir)

    assert tracker.get_completed_stages() == []

    tracker.mark_stage_completed("operator", "operator_ema.pt")
    assert tracker.get_completed_stages() == ["operator"]

    tracker.mark_stage_completed("diff_residual", "diffusion_residual_ema.pt")
    completed = tracker.get_completed_stages()
    assert "operator" in completed
    assert "diff_residual" in completed


def test_reset(temp_checkpoint_dir):
    """Test resetting all stages."""
    tracker = StageTracker(temp_checkpoint_dir)

    tracker.mark_stage_completed("operator", "operator_ema.pt")
    tracker.mark_stage_completed("diff_residual", "diffusion_residual_ema.pt")

    tracker.reset()

    # All stages should be back to not_started
    all_statuses = tracker.get_all_statuses()
    for stage in StageTracker.STAGES:
        assert all_statuses[stage].status == "not_started"


def test_invalid_stage_name(temp_checkpoint_dir):
    """Test that invalid stage names raise ValueError."""
    tracker = StageTracker(temp_checkpoint_dir)

    with pytest.raises(ValueError, match="Unknown stage"):
        tracker.get_stage_status("invalid_stage")

    with pytest.raises(ValueError, match="Unknown stage"):
        tracker.mark_stage_started("invalid_stage", total_epochs=10)
