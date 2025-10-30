"""Integration tests for VastAI resume functionality."""

import sys
from pathlib import Path

# Add scripts directory to path to import vast_launch
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from vast_launch import generate_onstart_script  # noqa: E402


def test_generate_onstart_fresh_start():
    """Test onstart script generation for fresh start."""
    script = generate_onstart_script(
        config="configs/train_burgers_32dim.yaml",
        stage="all",
        repo_url="https://github.com/example/universal_simulator.git",
        branch="main",
        workdir="/workspace",
        auto_shutdown=False,
        run_args=[],
        precompute=True,
        resume_from_wandb=None,  # Fresh start
        resume_mode="allow",
    )

    # Should clear checkpoints
    assert "rm -rf checkpoints" in script
    assert "FRESH START" in script

    # Should NOT download from WandB
    assert "CheckpointManager" not in script
    assert "download_checkpoints_from_run" not in script

    # Should NOT have auto-resume flag
    assert "--auto-resume" not in script


def test_generate_onstart_resume_from_wandb():
    """Test onstart script generation for WandB resume."""
    script = generate_onstart_script(
        config="configs/train_burgers_32dim.yaml",
        stage="all",
        repo_url="https://github.com/example/universal_simulator.git",
        branch="main",
        workdir="/workspace",
        auto_shutdown=True,
        run_args=[],
        precompute=True,
        resume_from_wandb="train-20251028_120000",
        resume_mode="must",
    )

    # Should NOT clear checkpoints
    assert "rm -rf checkpoints" not in script
    assert "RESUME MODE" in script

    # Should download from WandB
    assert "CheckpointManager" in script
    assert "download_checkpoints_from_run" in script
    assert "train-20251028_120000" in script
    assert "resume_mode = 'must'" in script

    # Should include auto-resume flag
    assert "--auto-resume" in script

    # Should include auto-shutdown
    assert "vastai stop instance" in script or "shutdown" in script.lower()


def test_onstart_script_has_required_components():
    """Test that generated onstart script has all required components."""
    script = generate_onstart_script(
        config="configs/train_burgers_32dim.yaml",
        stage="all",
        workdir="/workspace",
    )

    # Should have bash shebang
    assert script.startswith("#!/bin/bash")

    # Should have git clone/checkout
    assert "git clone" in script
    assert "git checkout" in script

    # Should install dependencies
    assert "pip install" in script

    # Should download data
    assert "rclone" in script or "data" in script.lower()

    # Should run training
    assert "python scripts/run_fast_to_sota.py" in script


def test_resume_mode_variations():
    """Test different resume mode configurations."""
    for resume_mode in ["allow", "must", "never"]:
        script = generate_onstart_script(
            config="configs/train_burgers_32dim.yaml",
            stage="all",
            resume_from_wandb="train-test-123",
            resume_mode=resume_mode,
        )

        # Should include the resume mode
        assert f"resume_mode = '{resume_mode}'" in script
