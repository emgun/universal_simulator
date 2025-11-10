"""Stage status tracking for multi-stage training pipelines."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

try:  # Python 3.11+
    from datetime import UTC
except ImportError:  # Python 3.10 fallback
    UTC = timezone.utc
from pathlib import Path
from typing import Literal

StageStatusType = Literal["not_started", "in_progress", "completed", "failed"]


@dataclass
class StageStatus:
    """Status of a single training stage."""

    status: StageStatusType
    checkpoint: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    epoch: int | None = None
    total_epochs: int | None = None
    error_message: str | None = None

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
            "created_at": datetime.now(UTC).isoformat(),
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

        with open(self.status_file) as f:
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
            started_at=datetime.now(UTC).isoformat(),
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
            completed_at=datetime.now(UTC).isoformat(),
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

    def get_all_statuses(self) -> dict[str, StageStatus]:
        """Get status of all stages.

        Returns:
            Dictionary mapping stage names to StageStatus objects
        """
        status_data = self._read_status()
        return {
            stage: StageStatus.from_dict(stage_dict)
            for stage, stage_dict in status_data["stages"].items()
        }

    def get_next_stage_to_run(self, cfg: dict) -> str | None:
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
