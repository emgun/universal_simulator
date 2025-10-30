"""Integration tests for auto-resume functionality."""

import json
from pathlib import Path

import pytest
import yaml

from ups.utils.stage_tracker import StageTracker


@pytest.fixture
def test_config(tmp_path):
    """Create minimal test config."""
    config = {
        "data": {"root": "data/pdebench", "split": "train"},
        "latent": {"dim": 16, "tokens": 32},
        "operator": {
            "pdet": {
                "input_dim": 16,
                "hidden_dim": 48,
                "depths": [1, 1],
                "num_heads": [4, 4],
            }
        },
        "diffusion": {"latent_dim": 16, "hidden_dim": 64, "depth": 2},
        "checkpoint": {"dir": str(tmp_path / "checkpoints")},
        "stages": {
            "operator": {"epochs": 2},
            "diff_residual": {"epochs": 1},
            "consistency_distill": {"epochs": 0},
            "steady_prior": {"epochs": 0},
        },
        "logging": {"wandb": {"enabled": False}},
    }

    config_path = tmp_path / "test_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    return config_path


def test_stage_tracker_integration(tmp_path):
    """Test that stage tracker integrates correctly with config structure."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()

    tracker = StageTracker(checkpoint_dir)

    # Simulate completing operator stage
    tracker.mark_stage_started("operator", total_epochs=25)
    tracker.mark_stage_completed("operator", "operator_ema.pt")

    # Verify status file was created and updated
    status_file = checkpoint_dir / "stage_status.json"
    assert status_file.exists()

    with open(status_file) as f:
        status_data = json.load(f)

    operator_status = status_data["stages"]["operator"]
    assert operator_status["status"] == "completed"
    assert operator_status["checkpoint"] == "operator_ema.pt"
    assert operator_status["completed_at"] is not None


def test_auto_resume_config_modification(tmp_path):
    """Test that auto-resume logic correctly modifies config."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()

    # Create stage status with operator completed
    tracker = StageTracker(checkpoint_dir)
    tracker.mark_stage_completed("operator", "operator_ema.pt")

    # Simulate auto-resume logic
    cfg = {
        "stages": {
            "operator": {"epochs": 25},
            "diff_residual": {"epochs": 8},
            "consistency_distill": {"epochs": 8},
            "steady_prior": {"epochs": 0},
        }
    }

    completed_stages = tracker.get_completed_stages()
    assert "operator" in completed_stages

    # Override config to skip completed stages
    for stage_name in completed_stages:
        cfg["stages"][stage_name]["epochs"] = 0

    # Verify operator was set to 0 epochs
    assert cfg["stages"]["operator"]["epochs"] == 0
    # Verify other stages unchanged
    assert cfg["stages"]["diff_residual"]["epochs"] == 8


def test_stage_progression(tmp_path):
    """Test that stage tracker correctly determines next stage to run."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()

    tracker = StageTracker(checkpoint_dir)

    cfg = {
        "stages": {
            "operator": {"epochs": 25},
            "diff_residual": {"epochs": 8},
            "consistency_distill": {"epochs": 8},
            "steady_prior": {"epochs": 0},
        }
    }

    # Initially should start with operator
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


def test_failed_stage_handling(tmp_path):
    """Test that failed stages are handled correctly."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()

    tracker = StageTracker(checkpoint_dir)

    # Start operator
    tracker.mark_stage_started("operator", total_epochs=25)

    # Simulate failure
    tracker.mark_stage_failed("operator", error_message="OOM error")

    # Verify status
    status = tracker.get_stage_status("operator")
    assert status.status == "failed"
    assert status.error_message == "OOM error"

    # Verify completed_stages doesn't include failed stage
    assert "operator" not in tracker.get_completed_stages()


def test_checkpoint_dir_creation(tmp_path):
    """Test that checkpoint directory is created if it doesn't exist."""
    checkpoint_dir = tmp_path / "new_checkpoints"
    assert not checkpoint_dir.exists()

    tracker = StageTracker(checkpoint_dir)

    # Should create directory and status file
    assert checkpoint_dir.exists()
    assert (checkpoint_dir / "stage_status.json").exists()
