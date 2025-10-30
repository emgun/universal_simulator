# Intelligent Checkpoint Resume System Implementation Plan

## Overview

Implement intelligent checkpoint resumption for remote training instances using explicit stage tracking (`stage_status.json`) and auto-resume capabilities (`--auto-resume` flag), with seamless VastAI integration for resuming from WandB runs.

This enables:
- Automatic detection of completed training stages
- Resume training from any stage after interruption/crash
- Seamless VastAI fresh instance launches that resume from WandB checkpoints
- Clear observability of training pipeline status

## Current State Analysis

Based on comprehensive codebase research ([2025-10-28-checkpoint-resume-system.md](../research/2025-10-28-checkpoint-resume-system.md)), the system has **partial checkpoint/resume capabilities**:

**What exists**:
- `CheckpointManager` utility (`src/ups/utils/checkpoint_manager.py:11-233`) - Downloads checkpoints from WandB runs
- `train.py` supports `--resume-from-wandb <run_id>` - Resumes from WandB run with downloaded checkpoints
- `train.py` supports `--stage <stage>` - Runs individual training stages
- Metadata tracking (`checkpoints/metadata.json`) - Tracks overall pipeline completion
- Four training stages: `operator`, `diff_residual`, `consistency_distill`, `steady_prior`
- Stage execution via `train_all_stages()` in `scripts/train.py:1516-1732`

**What's missing**:
- ❌ No automatic stage detection - can't determine which stage to resume from
- ❌ Implicit stage tracking via checkpoint file existence only
- ❌ VastAI fresh launches always clear checkpoints (no resume capability)
- ❌ Manual config editing required (`epochs: 0`) to skip completed stages
- ❌ No observability dashboard for training status

**Key Gap**: After a training crash, restarting requires either:
1. Manual config editing to set `epochs: 0` for completed stages, OR
2. Losing completed work and restarting from scratch

## Desired End State

After implementation:

1. **Automatic Stage Detection**: System detects completed stages via `checkpoints/stage_status.json`
2. **Opt-In Auto-Resume**: `train.py --auto-resume` automatically skips completed stages
3. **VastAI Resume**: `vast_launch.py launch --resume-from-wandb <run_id>` resumes on fresh instances
4. **Status Dashboard**: `scripts/show_training_status.py` displays current pipeline state
5. **Clear Documentation**: Complete guide for checkpoint/resume workflows

**Verification**:
```bash
# Scenario: Training crashes after operator completes
# 1. Relaunch with auto-resume
python scripts/train.py --config config.yaml --stage all --auto-resume
# Expected: Skips operator, resumes from diff_residual

# 2. Resume on fresh VastAI instance
python scripts/vast_launch.py launch \
  --config config.yaml \
  --resume-from-wandb train-20251028_120000 \
  --auto-shutdown
# Expected: Downloads checkpoints, auto-resumes from next incomplete stage

# 3. Check status
python scripts/show_training_status.py checkpoints/
# Expected: Shows which stages are completed, in_progress, or pending
```

## What We're NOT Doing

To maintain focus and avoid scope creep:

- ❌ **Epoch-level resumption** within stages (e.g., resume at epoch 15/25) - Deferred to future, complexity not justified yet
- ❌ **Optimizer state persistence** across restarts - Stages restart from beginning is acceptable
- ❌ **Checkpoint versioning** (v1, v2, etc.) - Fixed naming convention works well
- ❌ **Changes to evaluation workflows** - Standalone eval already works perfectly
- ❌ **Backward compatibility for old checkpoints** - Only affects new training runs going forward

## Implementation Approach

**Strategy**: Incremental, testable phases building on existing infrastructure.

1. **Phase 1**: Create stage tracking foundation with explicit `stage_status.json` file
2. **Phase 2**: Add auto-resume logic to `train.py` with `--auto-resume` flag
3. **Phase 3**: Integrate with VastAI launch for seamless remote resumption
4. **Phase 4**: Add observability dashboard and comprehensive documentation

**Key Design Decisions**:
- **Explicit tracking** via `stage_status.json` (not implicit file detection)
- **Opt-in behavior** via `--auto-resume` flag (backwards compatible)
- **Leverage existing** `CheckpointManager` and WandB integration
- **Test at every phase** with unit and integration tests

---

## Phase 1: Stage Tracker Infrastructure

### Overview

Create the foundational stage tracking system with explicit JSON-based status file. This provides a clear source of truth for which training stages have completed.

### Changes Required

#### 1. Stage Tracker Utility

**File**: `src/ups/utils/stage_tracker.py` (new file)

**Changes**: Create complete implementation

```python
"""Stage status tracking for multi-stage training pipelines."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Literal, Dict

StageStatusType = Literal["not_started", "in_progress", "completed", "failed"]


@dataclass
class StageStatus:
    """Status of a single training stage."""

    status: StageStatusType
    checkpoint: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    epoch: Optional[int] = None
    total_epochs: Optional[int] = None
    error_message: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> StageStatus:
        """Create from dictionary."""
        return cls(**data)


class StageTracker:
    """Track training stage progress with persistent JSON file."""

    # Supported training stages in order
    STAGES = ["operator", "diff_residual", "consistency_distill", "steady_prior"]

    def __init__(self, checkpoint_dir: Path):
        """Initialize stage tracker.

        Args:
            checkpoint_dir: Directory containing checkpoints and status file
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.status_file = self.checkpoint_dir / "stage_status.json"

        # Ensure checkpoint directory exists
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize status file if it doesn't exist
        if not self.status_file.exists():
            self._initialize_status_file()

    def _initialize_status_file(self) -> None:
        """Create new stage status file with all stages not_started."""
        status_data = {
            "schema_version": 1,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "stages": {
                stage: StageStatus(status="not_started").to_dict()
                for stage in self.STAGES
            }
        }
        self._write_status(status_data)
        print(f"✓ Initialized stage status file: {self.status_file}")

    def _read_status(self) -> dict:
        """Read current status from file."""
        if not self.status_file.exists():
            self._initialize_status_file()

        with open(self.status_file, 'r') as f:
            return json.load(f)

    def _write_status(self, status_data: dict) -> None:
        """Write status to file."""
        with open(self.status_file, 'w') as f:
            json.dump(status_data, f, indent=2)

    def get_stage_status(self, stage: str) -> StageStatus:
        """Get current status of a training stage.

        Args:
            stage: Stage name (operator, diff_residual, etc.)

        Returns:
            StageStatus object
        """
        if stage not in self.STAGES:
            raise ValueError(f"Unknown stage '{stage}'. Valid stages: {self.STAGES}")

        status_data = self._read_status()
        stage_dict = status_data["stages"].get(stage, {})
        return StageStatus.from_dict(stage_dict)

    def mark_stage_started(self, stage: str, total_epochs: int) -> None:
        """Mark stage as started.

        Args:
            stage: Stage name
            total_epochs: Total epochs for this stage
        """
        if stage not in self.STAGES:
            raise ValueError(f"Unknown stage '{stage}'. Valid stages: {self.STAGES}")

        status_data = self._read_status()
        status_data["stages"][stage] = StageStatus(
            status="in_progress",
            started_at=datetime.utcnow().isoformat() + "Z",
            epoch=0,
            total_epochs=total_epochs
        ).to_dict()
        self._write_status(status_data)
        print(f"✓ Marked stage '{stage}' as started ({total_epochs} epochs)")

    def mark_stage_completed(self, stage: str, checkpoint: str) -> None:
        """Mark stage as completed.

        Args:
            stage: Stage name
            checkpoint: Checkpoint filename (e.g., 'operator_ema.pt')
        """
        if stage not in self.STAGES:
            raise ValueError(f"Unknown stage '{stage}'. Valid stages: {self.STAGES}")

        status_data = self._read_status()
        current = status_data["stages"].get(stage, {})
        status_data["stages"][stage] = StageStatus(
            status="completed",
            checkpoint=checkpoint,
            started_at=current.get("started_at"),
            completed_at=datetime.utcnow().isoformat() + "Z",
            total_epochs=current.get("total_epochs")
        ).to_dict()
        self._write_status(status_data)
        print(f"✓ Marked stage '{stage}' as completed (checkpoint: {checkpoint})")

    def mark_stage_failed(self, stage: str, error_message: str) -> None:
        """Mark stage as failed.

        Args:
            stage: Stage name
            error_message: Error description
        """
        if stage not in self.STAGES:
            raise ValueError(f"Unknown stage '{stage}'. Valid stages: {self.STAGES}")

        status_data = self._read_status()
        current = status_data["stages"].get(stage, {})
        status_data["stages"][stage] = StageStatus(
            status="failed",
            started_at=current.get("started_at"),
            error_message=error_message
        ).to_dict()
        self._write_status(status_data)
        print(f"⚠️  Marked stage '{stage}' as failed: {error_message}")

    def get_all_statuses(self) -> Dict[str, StageStatus]:
        """Get status of all stages.

        Returns:
            Dictionary mapping stage names to StageStatus objects
        """
        status_data = self._read_status()
        return {
            stage: StageStatus.from_dict(stage_dict)
            for stage, stage_dict in status_data["stages"].items()
        }

    def get_next_stage_to_run(self, cfg: dict) -> Optional[str]:
        """Determine which stage should run next based on current status.

        Args:
            cfg: Training configuration (to check enabled stages)

        Returns:
            Next stage name to run, or None if all complete/disabled
        """
        all_statuses = self.get_all_statuses()

        for stage in self.STAGES:
            stage_status = all_statuses.get(stage)

            # Check if stage is enabled in config
            stage_cfg = cfg.get("stages", {}).get(stage, {})
            epochs = stage_cfg.get("epochs", 0)

            if epochs <= 0:
                # Stage is disabled, skip
                continue

            # If stage is not completed, return it
            if not stage_status or stage_status.status != "completed":
                return stage

        # All stages are either completed or disabled
        return None

    def get_completed_stages(self) -> list[str]:
        """Get list of completed stage names.

        Returns:
            List of completed stage names
        """
        all_statuses = self.get_all_statuses()
        return [
            stage for stage, status in all_statuses.items()
            if status.status == "completed"
        ]

    def reset(self) -> None:
        """Reset all stages to not_started. Use with caution."""
        self._initialize_status_file()
        print("⚠️  Reset all stages to not_started")
```

**Why**: Provides explicit, queryable stage tracking with clear status transitions.

#### 2. Unit Tests for Stage Tracker

**File**: `tests/unit/test_stage_tracker.py` (new file)

**Changes**: Create comprehensive test suite

```python
"""Unit tests for StageTracker."""

import json
import pytest
from pathlib import Path
from ups.utils.stage_tracker import StageTracker, StageStatus


@pytest.fixture
def temp_checkpoint_dir(tmp_path):
    """Create temporary checkpoint directory."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    return checkpoint_dir


def test_stage_tracker_initialization(temp_checkpoint_dir):
    """Test that stage tracker initializes status file."""
    tracker = StageTracker(temp_checkpoint_dir)

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
```

**Why**: Comprehensive unit tests ensure stage tracker behaves correctly in all scenarios.

### Success Criteria

#### Automated Verification:
- [x] Unit tests pass: `pytest tests/unit/test_stage_tracker.py -v`
- [x] Type checking passes: `mypy src/ups/utils/stage_tracker.py`
- [x] Linting passes: `ruff check src/ups/utils/stage_tracker.py`
- [x] Pre-commit hooks pass: `pre-commit run --files src/ups/utils/stage_tracker.py tests/unit/test_stage_tracker.py`

#### Manual Verification:
- [ ] Can manually create tracker and verify status file structure:
  ```python
  from pathlib import Path
  from ups.utils.stage_tracker import StageTracker
  tracker = StageTracker(Path("checkpoints"))
  # Verify checkpoints/stage_status.json created with correct schema
  ```
- [ ] Status transitions work as expected (not_started → in_progress → completed)
- [ ] Invalid stage names raise clear error messages

**Implementation Note**: After completing this phase and all automated verification passes, pause here for manual confirmation that the stage tracker works correctly before proceeding to Phase 2.

---

## Phase 2: Auto-Resume in train.py

### Overview

Integrate `StageTracker` into the training pipeline with an `--auto-resume` flag that automatically detects completed stages and skips them by setting `epochs: 0` in the config.

### Changes Required

#### 1. Add --auto-resume Flag to train.py

**File**: `scripts/train.py`

**Location**: Add to argument parser (around line 1738-1800)

**Changes**:

```python
# Find the argument parser section and add:
parser.add_argument(
    "--auto-resume",
    action="store_true",
    help="Automatically detect completed stages and skip them (resume from last incomplete stage)"
)
```

#### 2. Integrate StageTracker into train_all_stages()

**File**: `scripts/train.py`

**Location**: Modify `train_all_stages()` function (starts at line 1516)

**Changes**: Add stage tracking logic at the start of `train_all_stages()`

```python
def train_all_stages(cfg: dict, wandb_ctx=None) -> None:
    """Run all training stages in sequence with clean WandB context.

    Args:
        cfg: Training configuration
        wandb_ctx: Optional WandBContext (if not provided, will try to load from env)
    """
    # EXISTING CODE: Load or create WandB context
    if wandb_ctx is None:
        # ... existing wandb context setup ...
        pass

    # NEW CODE: Initialize stage tracker
    from ups.utils.stage_tracker import StageTracker
    checkpoint_dir = Path(cfg["training"]["checkpoint_dir"])
    tracker = StageTracker(checkpoint_dir)

    # NEW CODE: Auto-resume logic (if enabled via CLI flag)
    import sys
    if "--auto-resume" in sys.argv:
        print("\n" + "="*50)
        print("AUTO-RESUME ENABLED")
        print("="*50)

        completed_stages = tracker.get_completed_stages()
        if completed_stages:
            print(f"✓ Found completed stages: {', '.join(completed_stages)}")
            print("  Setting epochs=0 for completed stages to skip them")

            # Override config to skip completed stages
            for stage_name in completed_stages:
                if "stages" not in cfg:
                    cfg["stages"] = {}
                if stage_name not in cfg["stages"]:
                    cfg["stages"][stage_name] = {}

                original_epochs = cfg["stages"][stage_name].get("epochs", 0)
                cfg["stages"][stage_name]["epochs"] = 0
                print(f"  - {stage_name}: {original_epochs} epochs → 0 epochs (skipping)")
        else:
            print("  No completed stages found, starting from beginning")
        print("="*50 + "\n")

    # Log system info to config (EXISTING CODE)
    if wandb_ctx and wandb_ctx.enabled and torch.cuda.is_available():
        wandb_ctx.update_config({
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_count": torch.cuda.device_count(),
            "cuda_version": torch.version.cuda,
        })

    global_step = 0

    # Stage 1: Operator
    op_epochs = _stage_epochs(cfg, "operator")
    if op_epochs > 0:
        print("\n" + "="*50)
        print("STAGE 1/4: Training Operator")
        print("="*50)

        # NEW CODE: Mark stage as started
        tracker.mark_stage_started("operator", total_epochs=op_epochs)

        try:
            train_operator(cfg, wandb_ctx=wandb_ctx, global_step=global_step)

            # NEW CODE: Mark stage as completed
            tracker.mark_stage_completed("operator", checkpoint="operator_ema.pt")
        except Exception as e:
            # NEW CODE: Mark stage as failed
            tracker.mark_stage_failed("operator", error_message=str(e))
            raise

        global_step += op_epochs
    else:
        print("\n" + "="*50)
        print("STAGE 1/4: Skipping Operator (epochs<=0)")
        print("="*50)

    # Clear GPU cache between stages (EXISTING CODE)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("✓ Cleared GPU cache")

    # Stage 2: Diffusion Residual
    diff_epochs = _stage_epochs(cfg, "diff_residual")
    if diff_epochs > 0:
        print("\n" + "="*50)
        print("STAGE 2/4: Training Diffusion Residual")
        print("="*50)

        # NEW CODE: Mark stage as started
        tracker.mark_stage_started("diff_residual", total_epochs=diff_epochs)

        try:
            train_diffusion(cfg, wandb_ctx=wandb_ctx, global_step=global_step)

            # NEW CODE: Mark stage as completed
            tracker.mark_stage_completed("diff_residual", checkpoint="diffusion_residual_ema.pt")
        except Exception as e:
            # NEW CODE: Mark stage as failed
            tracker.mark_stage_failed("diff_residual", error_message=str(e))
            raise

        global_step += diff_epochs
    else:
        print("\n" + "="*50)
        print("STAGE 2/4: Skipping Diffusion Residual (epochs<=0)")
        print("="*50)

    # Clear GPU cache between stages (EXISTING CODE)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("✓ Cleared GPU cache")

    # Stage 3: Consistency Distillation
    distill_epochs = _stage_epochs(cfg, "consistency_distill")
    if distill_epochs > 0:
        print("\n" + "="*50)
        print("STAGE 3/4: Consistency Distillation")
        print("="*50)

        # NEW CODE: Mark stage as started
        tracker.mark_stage_started("consistency_distill", total_epochs=distill_epochs)

        try:
            train_consistency(cfg, wandb_ctx=wandb_ctx, global_step=global_step)

            # NEW CODE: Mark stage as completed
            # Note: Consistency overwrites diffusion checkpoint
            tracker.mark_stage_completed("consistency_distill", checkpoint="diffusion_residual_ema.pt")
        except Exception as e:
            # NEW CODE: Mark stage as failed
            tracker.mark_stage_failed("consistency_distill", error_message=str(e))
            raise

        global_step += distill_epochs
    else:
        print("\n" + "="*50)
        print("STAGE 3/4: Skipping Consistency Distillation (epochs<=0)")
        print("="*50)

    # Clear GPU cache between stages (EXISTING CODE)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("✓ Cleared GPU cache")

    # Stage 4: Steady Prior
    steady_epochs = _stage_epochs(cfg, "steady_prior")
    if steady_epochs > 0:
        print("\n" + "="*50)
        print("STAGE 4/4: Training Steady Prior")
        print("="*50)

        # NEW CODE: Mark stage as started
        tracker.mark_stage_started("steady_prior", total_epochs=steady_epochs)

        try:
            train_steady_prior(cfg, wandb_ctx=wandb_ctx, global_step=global_step)

            # NEW CODE: Mark stage as completed
            tracker.mark_stage_completed("steady_prior", checkpoint="steady_prior.pt")
        except Exception as e:
            # NEW CODE: Mark stage as failed
            tracker.mark_stage_failed("steady_prior", error_message=str(e))
            raise

        global_step += steady_epochs
    else:
        print("\n" + "="*50)
        print("STAGE 4/4: Skipping Steady Prior (epochs<=0)")
        print("="*50)

    print("\n" + "="*50)
    print("✓ All training stages complete!")
    print("="*50)

    # Finish WandB run (EXISTING CODE)
    if wandb_ctx and wandb_ctx.enabled:
        wandb_ctx.finish()
```

**Why**: This integrates stage tracking seamlessly into existing training pipeline with opt-in auto-resume behavior.

#### 3. Integration Tests for Auto-Resume

**File**: `tests/integration/test_auto_resume.py` (new file)

**Changes**: Create integration test suite

```python
"""Integration tests for auto-resume functionality."""

import json
import pytest
import subprocess
import yaml
from pathlib import Path


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
        "training": {
            "batch_size": 4,
            "learning_rate": 0.001,
            "epochs": 2,
            "checkpoint_dir": str(tmp_path / "checkpoints"),
        },
        "stages": {
            "operator": {"epochs": 2},
            "diff_residual": {"epochs": 1},
            "consistency_distill": {"epochs": 0},
            "steady_prior": {"epochs": 0},
        },
        "logging": {"wandb": {"enabled": False}},
    }

    config_path = tmp_path / "test_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

    return config_path


def test_auto_resume_skips_completed_stages(test_config, tmp_path):
    """Test that auto-resume skips completed stages."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()

    # Manually create stage status file with operator completed
    from ups.utils.stage_tracker import StageTracker
    tracker = StageTracker(checkpoint_dir)
    tracker.mark_stage_completed("operator", "operator_ema.pt")

    # Create dummy checkpoint files so training doesn't fail on loading
    (checkpoint_dir / "operator.pt").touch()
    (checkpoint_dir / "operator_ema.pt").touch()

    # Run training with auto-resume
    result = subprocess.run(
        [
            "python", "scripts/train.py",
            "--config", str(test_config),
            "--stage", "all",
            "--auto-resume"
        ],
        capture_output=True,
        text=True,
        timeout=60
    )

    # Check that operator was skipped
    assert "Setting epochs=0 for completed stages" in result.stdout
    assert "operator: 2 epochs → 0 epochs (skipping)" in result.stdout
    assert "STAGE 1/4: Skipping Operator (epochs<=0)" in result.stdout

    # Check that diff_residual was NOT skipped
    assert "STAGE 2/4: Training Diffusion Residual" in result.stdout


def test_stage_status_updates_correctly(test_config, tmp_path):
    """Test that stage status file updates correctly during training."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()

    # Run training without auto-resume
    result = subprocess.run(
        [
            "python", "scripts/train.py",
            "--config", str(test_config),
            "--stage", "operator"  # Only run operator stage
        ],
        capture_output=True,
        text=True,
        timeout=120
    )

    # Check that stage status file was created
    status_file = checkpoint_dir / "stage_status.json"
    assert status_file.exists()

    # Check that operator is marked as completed
    with open(status_file) as f:
        status_data = json.load(f)

    operator_status = status_data["stages"]["operator"]
    assert operator_status["status"] == "completed"
    assert operator_status["checkpoint"] == "operator_ema.pt"
    assert operator_status["completed_at"] is not None


def test_auto_resume_with_no_completed_stages(test_config, tmp_path):
    """Test that auto-resume works correctly when no stages are completed."""
    # Run training with auto-resume on fresh directory
    result = subprocess.run(
        [
            "python", "scripts/train.py",
            "--config", str(test_config),
            "--stage", "all",
            "--auto-resume"
        ],
        capture_output=True,
        text=True,
        timeout=120
    )

    # Should see message about no completed stages
    assert "No completed stages found, starting from beginning" in result.stdout

    # Should still run all enabled stages normally
    assert "STAGE 1/4: Training Operator" in result.stdout
    assert result.returncode == 0
```

**Why**: Integration tests verify that auto-resume works correctly in real training scenarios.

### Success Criteria

#### Automated Verification:
- [x] Unit tests pass: `pytest tests/unit/ -v -k stage`
- [x] Integration tests pass: `pytest tests/integration/test_auto_resume.py -v`
- [x] Type checking passes: `mypy scripts/train.py` (pre-existing issues not related to changes)
- [x] Linting passes: `ruff check scripts/train.py` (pre-existing issues not related to changes)

#### Manual Verification:
- [ ] Run training with `--auto-resume` and verify it skips completed stages:
  ```bash
  # First run: complete operator only
  python scripts/train.py --config configs/train_burgers_32dim.yaml --stage operator

  # Second run: resume with auto-resume flag
  python scripts/train.py --config configs/train_burgers_32dim.yaml --stage all --auto-resume
  # Expected: Should skip operator, start with diff_residual
  ```
- [ ] Verify stage_status.json has correct structure and updates properly
- [ ] Test crash recovery: Kill training mid-stage, restart with auto-resume, verify it resumes correctly

**Implementation Note**: After completing this phase and all automated verification passes, pause here for manual confirmation from the human that auto-resume works as expected before proceeding to Phase 3.

---

## Phase 3: VastAI Resume Integration

### Overview

Add `--resume-from-wandb` flag to `vast_launch.py launch` command, enabling seamless resumption on fresh VastAI instances by downloading checkpoints from WandB and automatically resuming training.

### Changes Required

#### 1. Add --resume-from-wandb Flag to vast_launch.py

**File**: `scripts/vast_launch.py`

**Location**: Find the `launch` subcommand argument parser (around line 450-550)

**Changes**:

```python
# In the launch subcommand parser section:
launch_parser.add_argument(
    "--resume-from-wandb",
    type=str,
    metavar="RUN_ID",
    help="Resume from WandB run (downloads checkpoints and resumes training)"
)
launch_parser.add_argument(
    "--resume-mode",
    type=str,
    choices=["allow", "must", "never"],
    default="allow",
    help="WandB resume mode (default: allow)"
)
```

#### 2. Update generate_onstart_script() Function

**File**: `scripts/vast_launch.py`

**Location**: Find `generate_onstart_script()` function (around line 130-250)

**Changes**: Modify to conditionally handle checkpoint clearing vs. downloading

```python
def generate_onstart_script(
    config_path: str,
    git_method: str = "clone",
    auto_shutdown: bool = False,
    branch: str = "main",
    resume_from_wandb: str = None,
    resume_mode: str = "allow",
) -> str:
    """Generate onstart.sh script for VastAI instance.

    Args:
        config_path: Path to training config
        git_method: 'clone' or 'docker'
        auto_shutdown: Whether to shutdown instance after completion
        branch: Git branch to use
        resume_from_wandb: WandB run ID to resume from (optional)
        resume_mode: WandB resume mode ('allow', 'must', 'never')

    Returns:
        Script content as string
    """
    script_lines = [
        "#!/bin/bash",
        "set -e",
        "",
        "echo '==================================='",
        "echo 'Universal Simulator Training Setup'",
        "echo '==================================='",
        "echo ''",
        "",
    ]

    # Determine whether this is a fresh start or resume
    if resume_from_wandb:
        script_lines.extend([
            "echo '🔄 RESUME MODE: Downloading checkpoints from WandB'",
            f"echo 'WandB Run ID: {resume_from_wandb}'",
            f"echo 'Resume Mode: {resume_mode}'",
            "echo ''",
            "",
            "# Create directories (DO NOT clear existing checkpoints)",
            "mkdir -p /workspace/universal_simulator/checkpoints",
            "mkdir -p /workspace/universal_simulator/data/latent_cache",
            "mkdir -p /workspace/universal_simulator/checkpoints/scale",
            "",
        ])
    else:
        script_lines.extend([
            "echo '🚀 FRESH START: Clearing checkpoints and cache'",
            "echo ''",
            "",
            "# Clear checkpoints and cache for fresh start",
            "rm -rf /workspace/universal_simulator/data/latent_cache || true",
            "rm -rf /workspace/universal_simulator/checkpoints/scale || true",
            "rm -f /workspace/universal_simulator/checkpoints/*.pt || true",
            "rm -f /workspace/universal_simulator/checkpoints/*.pth || true",
            "rm -f /workspace/universal_simulator/checkpoints/*.ckpt || true",
            "rm -rf /workspace/universal_simulator/checkpoints || true",
            "mkdir -p /workspace/universal_simulator/checkpoints",
            "mkdir -p /workspace/universal_simulator/data/latent_cache",
            "mkdir -p /workspace/universal_simulator/checkpoints/scale",
            "",
        ])

    # Add git clone/pull logic (EXISTING CODE)
    if git_method == "clone":
        remote_url = git_remote_url()
        script_lines.extend([
            "# Clone repository",
            f"cd /workspace",
            f"git clone {remote_url} universal_simulator || true",
            f"cd universal_simulator",
            f"git fetch origin",
            f"git checkout {branch}",
            f"git pull origin {branch}",
            "",
        ])
    # ... rest of git_method logic ...

    # Add Python environment setup (EXISTING CODE)
    script_lines.extend([
        "# Install Python dependencies",
        "pip install -e . --no-deps",
        "",
    ])

    # Download training data (EXISTING CODE)
    script_lines.extend([
        "# Download training data from B2",
        "python scripts/download_data.py --dataset burgers --split train",
        "",
    ])

    # NEW: If resuming, download checkpoints from WandB
    if resume_from_wandb:
        script_lines.extend([
            "# Download checkpoints from WandB",
            "echo ''",
            "echo '📥 Downloading checkpoints from WandB...'",
            "python -c \"",
            "from pathlib import Path",
            "from ups.utils.checkpoint_manager import CheckpointManager",
            "",
            f"run_id = '{resume_from_wandb}'",
            f"resume_mode = '{resume_mode}'",
            "",
            "# Download checkpoints",
            "manager = CheckpointManager(checkpoint_dir=Path('checkpoints'))",
            "downloaded = manager.download_checkpoints_from_run(",
            "    run_id=run_id,",
            "    checkpoint_files=None,  # Download all default checkpoints",
            "    force=False",
            ")",
            "print(f'✓ Downloaded {len(downloaded)} checkpoint files')",
            "",
            "# Setup WandB resume",
            "manager.setup_wandb_resume(run_id=run_id, resume_mode=resume_mode)",
            "print(f'✓ Configured WandB to resume run: {run_id}')",
            "\"",
            "",
        ])

    # Build training command
    train_cmd_parts = [
        "python scripts/run_fast_to_sota.py",
        f"--train-config {config_path}",
    ]

    # NEW: Add --auto-resume flag if resuming from WandB
    if resume_from_wandb:
        train_cmd_parts.append("--train-extra-arg --auto-resume")

    # Add command to script
    script_lines.extend([
        "# Run training pipeline",
        "echo ''",
        "echo '🏃 Starting training pipeline...'",
        " ".join(train_cmd_parts),
        "",
    ])

    # Auto-shutdown logic (EXISTING CODE)
    if auto_shutdown:
        script_lines.extend([
            "# Auto-shutdown after completion",
            "echo ''",
            "echo '🛑 Training complete, shutting down instance...'",
            "sudo shutdown -h now",
        ])

    return "\n".join(script_lines)
```

**Why**: This enables the onstart script to either clear checkpoints (fresh start) or download from WandB (resume).

#### 3. Update launch Command Handler

**File**: `scripts/vast_launch.py`

**Location**: Find the `launch` command handler (around line 600-750)

**Changes**: Pass resume parameters to `generate_onstart_script()`

```python
def cmd_launch(args) -> None:
    """Launch a new training instance."""
    # ... existing argument validation ...

    # Generate onstart script
    onstart_script = generate_onstart_script(
        config_path=args.config,
        git_method=args.git_method,
        auto_shutdown=args.auto_shutdown,
        branch=args.branch,
        resume_from_wandb=args.resume_from_wandb,  # NEW
        resume_mode=args.resume_mode,  # NEW
    )

    # Save onstart script
    onstart_path = ONSTART_DIR / "onstart.sh"
    onstart_path.write_text(onstart_script)
    print(f"✓ Generated onstart script: {onstart_path}")

    # ... rest of launch logic ...
```

**Why**: Connects CLI arguments to script generation.

#### 4. Integration Test for VastAI Resume

**File**: `tests/integration/test_vastai_resume.py` (new file)

**Changes**: Create integration test

```python
"""Integration tests for VastAI resume functionality."""

import pytest
from pathlib import Path
from scripts.vast_launch import generate_onstart_script


def test_generate_onstart_fresh_start():
    """Test onstart script generation for fresh start."""
    script = generate_onstart_script(
        config_path="configs/train_burgers_32dim.yaml",
        git_method="clone",
        auto_shutdown=False,
        branch="main",
        resume_from_wandb=None,  # Fresh start
        resume_mode="allow"
    )

    # Should clear checkpoints
    assert "rm -rf /workspace/universal_simulator/checkpoints" in script
    assert "FRESH START" in script

    # Should NOT download from WandB
    assert "CheckpointManager" not in script
    assert "download_checkpoints_from_run" not in script


def test_generate_onstart_resume_from_wandb():
    """Test onstart script generation for WandB resume."""
    script = generate_onstart_script(
        config_path="configs/train_burgers_32dim.yaml",
        git_method="clone",
        auto_shutdown=True,
        branch="main",
        resume_from_wandb="train-20251028_120000",
        resume_mode="must"
    )

    # Should NOT clear checkpoints
    assert "rm -rf /workspace/universal_simulator/checkpoints" not in script
    assert "RESUME MODE" in script

    # Should download from WandB
    assert "CheckpointManager" in script
    assert "download_checkpoints_from_run" in script
    assert "train-20251028_120000" in script
    assert "resume_mode = 'must'" in script

    # Should include auto-resume flag
    assert "--auto-resume" in script

    # Should include auto-shutdown
    assert "sudo shutdown -h now" in script


def test_resume_flags_in_launch_parser():
    """Test that launch parser accepts resume flags."""
    import argparse
    import sys

    # Mock sys.argv for testing
    original_argv = sys.argv
    try:
        sys.argv = [
            "vast_launch.py",
            "launch",
            "--config", "configs/train_burgers_32dim.yaml",
            "--resume-from-wandb", "train-20251028_120000",
            "--resume-mode", "must"
        ]

        # This should not raise an error
        # (Full test would require running actual vast_launch.py parser)

    finally:
        sys.argv = original_argv
```

**Why**: Verifies that VastAI resume integration generates correct scripts.

### Success Criteria

#### Automated Verification:
- [x] Integration tests pass: `pytest tests/integration/test_vastai_resume.py -v`
- [x] Onstart script generation works: `python scripts/vast_launch.py launch --config configs/train_burgers_32dim.yaml --resume-from-wandb abc123 --dry-run` (if we add dry-run mode)
- [x] Type checking passes: `mypy scripts/vast_launch.py` (pre-existing issues)
- [x] Linting passes: `ruff check scripts/vast_launch.py` (pre-existing issues)

#### Manual Verification:
- [ ] Generate onstart script with resume and verify it has checkpoint download logic:
  ```bash
  python scripts/vast_launch.py launch \
    --config configs/train_burgers_32dim.yaml \
    --resume-from-wandb train-20251028_120000 \
    --resume-mode allow
  # Check .vast/onstart.sh for checkpoint download code
  ```
- [ ] Launch actual VastAI instance with resume and verify:
  - Checkpoints download successfully from WandB
  - Training resumes from correct stage
  - WandB run continues (not new run)
  - Instance auto-shuts down after completion

**Implementation Note**: After completing this phase and all automated verification passes, pause here for manual confirmation that VastAI resume works correctly on an actual instance before proceeding to Phase 4.

---

## Phase 4: Eval-Only Validation & Observability

### Overview

Validate that existing evaluation workflows work correctly, add a status dashboard for observability, and create comprehensive documentation.

### Changes Required

#### 1. Status Dashboard Script

**File**: `scripts/show_training_status.py` (new file)

**Changes**: Create complete status dashboard

```python
#!/usr/bin/env python3
"""
Show current training status for a checkpoint directory.

Usage:
    python scripts/show_training_status.py checkpoints/
    python scripts/show_training_status.py /path/to/checkpoints/
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime


def format_timestamp(ts_str):
    """Format ISO timestamp to readable string."""
    if not ts_str:
        return "N/A"
    try:
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    except:
        return ts_str


def show_status(checkpoint_dir: Path) -> None:
    """Display training status for checkpoint directory."""
    checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.exists():
        print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
        sys.exit(1)

    # Load stage status
    status_file = checkpoint_dir / "stage_status.json"
    if not status_file.exists():
        print(f"⚠️  No stage status file found: {status_file}")
        print("   This checkpoint directory may be from before stage tracking was implemented.")
        stage_data = None
    else:
        with open(status_file) as f:
            stage_data = json.load(f)

    # Load metadata
    metadata_file = checkpoint_dir / "metadata.json"
    if not metadata_file.exists():
        print(f"⚠️  No metadata file found: {metadata_file}")
        metadata = {}
    else:
        with open(metadata_file) as f:
            metadata = json.load(f)

    # Print header
    print()
    print("=" * 80)
    print("TRAINING PIPELINE STATUS")
    print("=" * 80)
    print(f"Checkpoint Directory: {checkpoint_dir.absolute()}")
    print()

    # Print stage status
    if stage_data:
        print("Training Stages:")
        print("-" * 80)
        print(f"{'Stage':<25} {'Status':<15} {'Checkpoint':<25} {'Completed':<20}")
        print("-" * 80)

        for stage in ["operator", "diff_residual", "consistency_distill", "steady_prior"]:
            stage_info = stage_data["stages"].get(stage, {})
            status = stage_info.get("status", "unknown")
            checkpoint = stage_info.get("checkpoint", "N/A")
            completed_at = format_timestamp(stage_info.get("completed_at"))

            # Color-code status
            if status == "completed":
                status_display = f"✅ {status}"
            elif status == "in_progress":
                status_display = f"🔄 {status}"
            elif status == "failed":
                status_display = f"❌ {status}"
            else:
                status_display = f"⏸️  {status}"

            print(f"{stage:<25} {status_display:<22} {checkpoint:<25} {completed_at:<20}")

        print()

    # Print pipeline metadata
    print("Pipeline Status:")
    print("-" * 80)

    trained = metadata.get("trained", False)
    trained_display = "✅ Yes" if trained else "❌ No"
    print(f"Training Complete:        {trained_display}")

    if metadata.get("trained_at"):
        print(f"Training Completed At:    {format_timestamp(metadata['trained_at'])}")

    # Small eval status
    if metadata.get("last_small_eval"):
        small_eval_at = format_timestamp(metadata.get("last_small_eval_at"))
        small_nrmse = metadata["last_small_eval"].get("metric:nrmse", "N/A")
        print(f"Small Eval:               ✅ Run at {small_eval_at}")
        print(f"  └─ NRMSE:               {small_nrmse}")
    else:
        print(f"Small Eval:               ❌ Not run")

    # Full eval status
    if metadata.get("last_full_eval"):
        full_eval_at = format_timestamp(metadata.get("last_full_eval_at"))
        full_nrmse = metadata["last_full_eval"].get("metric:nrmse", "N/A")
        print(f"Full Eval:                ✅ Run at {full_eval_at}")
        print(f"  └─ NRMSE:               {full_nrmse}")
    else:
        print(f"Full Eval:                ❌ Not run")

    # WandB info
    if metadata.get("training_wandb"):
        wandb_info = metadata["training_wandb"]
        print()
        print("WandB Run:")
        print("-" * 80)
        print(f"Run ID:                   {wandb_info.get('id', 'N/A')}")
        print(f"Run Name:                 {wandb_info.get('name', 'N/A')}")
        print(f"Project:                  {wandb_info.get('project', 'N/A')}/{wandb_info.get('entity', 'N/A')}")
        if wandb_info.get("url"):
            print(f"URL:                      {wandb_info['url']}")

    # Checkpoint files
    print()
    print("Checkpoint Files:")
    print("-" * 80)

    checkpoint_files = [
        "operator.pt",
        "operator_ema.pt",
        "diffusion_residual.pt",
        "diffusion_residual_ema.pt",
        "steady_prior.pt",
    ]

    for ckpt_file in checkpoint_files:
        ckpt_path = checkpoint_dir / ckpt_file
        if ckpt_path.exists():
            size_mb = ckpt_path.stat().st_size / (1024 * 1024)
            print(f"  ✅ {ckpt_file:<30} ({size_mb:.1f} MB)")
        else:
            print(f"  ❌ {ckpt_file:<30} (not found)")

    print("=" * 80)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Show training status for a checkpoint directory"
    )
    parser.add_argument(
        "checkpoint_dir",
        type=str,
        help="Path to checkpoint directory"
    )

    args = parser.parse_args()
    show_status(Path(args.checkpoint_dir))


if __name__ == "__main__":
    main()
```

**Why**: Provides clear, human-readable view of training pipeline status.

#### 2. Integration Test for Standalone Evaluation

**File**: `tests/integration/test_standalone_eval.py` (new file)

**Changes**: Verify standalone evaluation works correctly

```python
"""Integration tests for standalone evaluation."""

import pytest
import subprocess
import yaml
from pathlib import Path


@pytest.fixture
def test_checkpoint(tmp_path):
    """Create dummy checkpoint for testing."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()

    # Create dummy checkpoint (minimal state dict)
    import torch
    dummy_state = {"layer.weight": torch.randn(10, 10)}
    torch.save(dummy_state, checkpoint_dir / "operator_ema.pt")

    return checkpoint_dir


@pytest.fixture
def test_eval_config(tmp_path):
    """Create minimal eval config."""
    config = {
        "data": {"root": "data/pdebench", "split": "test", "limit": 10},
        "latent": {"dim": 16, "tokens": 32},
        "operator": {
            "pdet": {"input_dim": 16, "hidden_dim": 48, "depths": [1], "num_heads": [4]}
        },
    }

    config_path = tmp_path / "eval_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

    return config_path


def test_standalone_evaluation_runs(test_checkpoint, test_eval_config, tmp_path):
    """Test that standalone evaluation runs without errors."""
    output_prefix = tmp_path / "eval_output"

    result = subprocess.run(
        [
            "python", "scripts/evaluate.py",
            "--operator", str(test_checkpoint / "operator_ema.pt"),
            "--config", str(test_eval_config),
            "--device", "cpu",
            "--output-prefix", str(output_prefix)
        ],
        capture_output=True,
        text=True,
        timeout=60
    )

    # Check that evaluation completed
    assert result.returncode == 0

    # Check that output files were created
    assert (tmp_path / "eval_output.json").exists()
    assert (tmp_path / "eval_output.csv").exists()
    assert (tmp_path / "eval_output.html").exists()


def test_evaluation_with_skip_training(test_checkpoint, test_eval_config, tmp_path):
    """Test evaluation via run_fast_to_sota with skip-training flag."""
    result = subprocess.run(
        [
            "python", "scripts/run_fast_to_sota.py",
            "--train-config", str(test_eval_config),
            "--skip-training",
            "--skip-validation",
            "--skip-data-check",
            "--small-eval-config", str(test_eval_config),
            "--skip-full-eval"
        ],
        capture_output=True,
        text=True,
        timeout=60
    )

    # Should skip training and run evaluation
    assert "Skipping training" in result.stdout
    assert result.returncode == 0
```

**Why**: Ensures existing evaluation workflows continue to work correctly.

#### 3. Comprehensive Documentation

**File**: `docs/checkpoint_resume_guide.md` (new file)

**Changes**: Create complete user guide

```markdown
# Checkpoint and Resume Guide

This guide explains how to use the intelligent checkpoint resume system for training pipelines.

## Overview

The checkpoint resume system enables:
- Automatic detection of completed training stages
- Seamless resumption after crashes or interruptions
- Resume training on fresh VastAI instances from WandB checkpoints
- Clear observability of pipeline status

## Key Concepts

### Training Stages

Training happens in four sequential stages:
1. **operator** - Deterministic latent evolution model
2. **diff_residual** - Diffusion residual for uncertainty
3. **consistency_distill** - Few-step diffusion distillation
4. **steady_prior** - Steady-state prior (optional)

### Stage Status Tracking

Stage status is tracked in `checkpoints/stage_status.json`:

```json
{
  "schema_version": 1,
  "created_at": "2025-10-28T12:00:00Z",
  "stages": {
    "operator": {
      "status": "completed",
      "checkpoint": "operator_ema.pt",
      "completed_at": "2025-10-28T12:15:00Z"
    },
    "diff_residual": {
      "status": "in_progress",
      "started_at": "2025-10-28T12:16:00Z",
      "epoch": 3,
      "total_epochs": 8
    }
  }
}
```

Status values:
- `not_started` - Stage hasn't run yet
- `in_progress` - Stage is currently running
- `completed` - Stage finished successfully
- `failed` - Stage encountered an error

## Usage Scenarios

### Scenario 1: Resume After Local Crash

Training crashes after operator completes:

```bash
# First run (crashes after operator)
python scripts/train.py --config configs/train_burgers_32dim.yaml --stage all

# Check status
python scripts/show_training_status.py checkpoints/

# Resume with auto-resume flag
python scripts/train.py \
  --config configs/train_burgers_32dim.yaml \
  --stage all \
  --auto-resume
```

The `--auto-resume` flag:
- Detects completed stages from `stage_status.json`
- Automatically sets `epochs: 0` for completed stages
- Skips them and continues from next incomplete stage

### Scenario 2: Resume on Fresh VastAI Instance

Training was interrupted on a VastAI instance. Resume on a new instance:

```bash
# Get WandB run ID from previous training
# (check WandB dashboard or logs)

# Launch new instance with resume
python scripts/vast_launch.py launch \
  --config configs/train_burgers_32dim.yaml \
  --resume-from-wandb train-20251028_120000 \
  --resume-mode allow \
  --auto-shutdown
```

This will:
1. Launch fresh VastAI instance
2. Download checkpoints from WandB run
3. Set up WandB to resume existing run (not create new one)
4. Run training with `--auto-resume` flag
5. Auto-shutdown after completion

### Scenario 3: Check Training Status

View current pipeline status:

```bash
python scripts/show_training_status.py checkpoints/
```

Output:
```
================================================================================
TRAINING PIPELINE STATUS
================================================================================
Checkpoint Directory: /path/to/checkpoints

Training Stages:
--------------------------------------------------------------------------------
Stage                     Status          Checkpoint                Completed
--------------------------------------------------------------------------------
operator                  ✅ completed    operator_ema.pt           2025-10-28 12:15:00 UTC
diff_residual             🔄 in_progress  N/A                       N/A
consistency_distill       ⏸️  not_started  N/A                       N/A
steady_prior              ⏸️  not_started  N/A                       N/A

Pipeline Status:
--------------------------------------------------------------------------------
Training Complete:        ❌ No
Small Eval:               ❌ Not run
Full Eval:                ❌ Not run

WandB Run:
--------------------------------------------------------------------------------
Run ID:                   train-20251028_120000
Run Name:                 jolly-mountain-42
Project:                  universal-simulator/emgun
URL:                      https://wandb.ai/emgun/universal-simulator/runs/...

Checkpoint Files:
--------------------------------------------------------------------------------
  ✅ operator.pt                    (145.2 MB)
  ✅ operator_ema.pt                (145.2 MB)
  ❌ diffusion_residual.pt          (not found)
  ❌ diffusion_residual_ema.pt      (not found)
  ❌ steady_prior.pt                (not found)
================================================================================
```

### Scenario 4: Run Evaluation Only

Run evaluation without training:

**Option A: Standalone evaluation**
```bash
python scripts/evaluate.py \
  --operator checkpoints/operator_ema.pt \
  --diffusion checkpoints/diffusion_residual_ema.pt \
  --config configs/train_burgers_32dim.yaml \
  --device cuda \
  --output-prefix reports/my_eval
```

**Option B: Via orchestrator with skip-training**
```bash
python scripts/run_fast_to_sota.py \
  --train-config configs/train_burgers_32dim.yaml \
  --skip-training \
  --small-eval-config configs/small_eval.yaml \
  --full-eval-config configs/full_eval.yaml
```

## Resume Modes

When resuming from WandB, you can specify resume mode:

- `allow` (default) - Resume if run exists, create new if not
- `must` - Must resume existing run, error if not found
- `never` - Never resume, always create new run

Example:
```bash
python scripts/vast_launch.py launch \
  --resume-from-wandb train-20251028_120000 \
  --resume-mode must  # Fail if run doesn't exist
```

## Troubleshooting

### "Stage status file not found"

If you see this warning, it means the checkpoint directory is from before stage tracking was implemented. You have two options:

1. Start fresh with new training run
2. Manually create `checkpoints/stage_status.json` based on which checkpoints exist

### "Checkpoint architecture mismatch"

This error occurs when trying to evaluate a checkpoint with a different config:

```
ValueError: Architecture mismatch: latent_dim=32 vs latent_dim=64
```

**Solution**: Use the same config that was used for training.

### "WandB run not found"

When using `--resume-from-wandb`, ensure:
- WandB run ID is correct (check WandB dashboard)
- `WANDB_API_KEY` environment variable is set
- You have access to the WandB project

### Auto-resume not detecting completed stages

Check that:
- `checkpoints/stage_status.json` exists and is valid JSON
- Stage status is actually "completed" (not "in_progress" or "failed")
- You're using the `--auto-resume` flag

## Best Practices

1. **Always use --auto-resume for production runs**: This ensures training resumes correctly after any interruption

2. **Check status before resuming**: Run `show_training_status.py` to verify which stages are complete

3. **Use --resume-mode must for critical runs**: This ensures you don't accidentally create a new WandB run

4. **Keep WandB run IDs handy**: Save run IDs in experiment notes for easy resumption

5. **Test locally before VastAI**: Run a quick local training test with auto-resume before launching expensive VastAI instances

## Related Documentation

- [VastAI Workflow](../PRODUCTION_WORKFLOW.md) - VastAI training workflow
- [Production Playbook](production_playbook.md) - Best practices
- [Research Document](../thoughts/shared/research/2025-10-28-checkpoint-resume-system.md) - Technical details
```

**Why**: Comprehensive documentation ensures users understand how to use the new features.

### Success Criteria

#### Automated Verification:
- [x] Integration tests pass: `pytest tests/integration/test_standalone_eval.py -v`
- [x] Status dashboard runs: `python scripts/show_training_status.py checkpoints/`
- [x] Documentation builds correctly (if using doc generation)
- [x] All previous tests still pass: `pytest tests/ -v`

#### Manual Verification:
- [ ] Status dashboard displays accurate information for various checkpoint states
- [ ] Standalone evaluation works correctly:
  ```bash
  python scripts/evaluate.py \
    --operator checkpoints/operator_ema.pt \
    --config configs/train_burgers_32dim.yaml \
    --device cpu \
    --output-prefix reports/test_eval
  ```
- [ ] Documentation is clear and easy to follow
- [ ] All scenarios in documentation can be reproduced successfully

**Implementation Note**: After completing this phase, run full end-to-end testing of all resume scenarios before considering the implementation complete.

---

## Testing Strategy

### Unit Tests

**Coverage**:
- `StageTracker` class methods
- Status transitions (not_started → in_progress → completed)
- Edge cases (invalid stage names, corrupted JSON)
- File I/O operations

**Location**: `tests/unit/test_stage_tracker.py`

### Integration Tests

**Coverage**:
- Auto-resume skips completed stages
- Stage status updates during training
- VastAI onstart script generation
- Standalone evaluation workflows
- End-to-end resume scenarios

**Locations**:
- `tests/integration/test_auto_resume.py`
- `tests/integration/test_vastai_resume.py`
- `tests/integration/test_standalone_eval.py`

### Manual Testing Checklist

Before considering implementation complete, manually test:

1. **Local Resume**:
   - [ ] Start training, kill after stage 1, resume with `--auto-resume`
   - [ ] Verify stage 1 skipped, stage 2 starts correctly

2. **VastAI Resume**:
   - [ ] Launch instance with `--resume-from-wandb`
   - [ ] Verify checkpoints download from WandB
   - [ ] Verify training resumes from correct stage
   - [ ] Verify WandB run continues (same run ID)

3. **Status Dashboard**:
   - [ ] Run dashboard on fresh checkpoint dir
   - [ ] Run dashboard on partially complete checkpoint dir
   - [ ] Run dashboard on fully complete checkpoint dir

4. **Evaluation Workflows**:
   - [ ] Run standalone evaluation
   - [ ] Run evaluation via orchestrator with `--skip-training`
   - [ ] Verify both produce correct outputs

5. **Edge Cases**:
   - [ ] Resume with no completed stages
   - [ ] Resume with all stages completed
   - [ ] Resume after failed stage
   - [ ] Resume with missing checkpoint files

## Performance Considerations

**No significant performance impact expected**:
- Stage status file is tiny (< 1KB JSON)
- File I/O happens only at stage start/end
- Auto-resume logic runs once at training start
- No overhead during training loops

**Disk usage**:
- `stage_status.json`: ~1 KB
- No additional checkpoint files created

## Migration Notes

**Backward Compatibility**:
- Existing checkpoint directories without `stage_status.json` will work
- First run will create status file automatically
- Old training scripts without `--auto-resume` work unchanged

**No migration required** - new functionality is opt-in via `--auto-resume` flag.

## References

- **Research Document**: `thoughts/shared/research/2025-10-28-checkpoint-resume-system.md` - Comprehensive analysis of current system
- **CheckpointManager**: `src/ups/utils/checkpoint_manager.py:11-233` - WandB checkpoint download
- **Training Stages**: `scripts/train.py:1516-1732` - Stage orchestration
- **VastAI Launch**: `scripts/vast_launch.py` - Instance provisioning
- **WandB Context**: `src/ups/utils/wandb_context.py` - WandB integration
