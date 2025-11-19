"""Checkpoint management utility for downloading and resuming from WandB runs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import wandb


class CheckpointManager:
    """Manager for downloading and managing checkpoints from WandB runs."""

    def __init__(
        self,
        checkpoint_dir: Path,
        wandb_entity: Optional[str] = None,
        wandb_project: Optional[str] = None,
    ):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Local directory to store checkpoints
            wandb_entity: WandB entity (defaults to WANDB_ENTITY env var)
            wandb_project: WandB project (defaults to WANDB_PROJECT env var)
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.wandb_entity = wandb_entity or os.environ.get("WANDB_ENTITY", "")
        self.wandb_project = wandb_project or os.environ.get("WANDB_PROJECT", "universal-simulator")

        if not self.wandb_entity:
            raise ValueError("WandB entity must be provided or set in WANDB_ENTITY")

    def download_checkpoints_from_run(
        self,
        run_id: str,
        checkpoint_files: Optional[List[str]] = None,
        force: bool = False,
    ) -> List[Path]:
        """Download checkpoints from a WandB run.

        Args:
            run_id: WandB run ID (e.g., 'train-20251027_193043')
            checkpoint_files: List of checkpoint file paths to download.
                If None, downloads all common checkpoint files.
            force: If True, re-download even if files exist locally

        Returns:
            List of paths to downloaded checkpoint files
        """
        # Default checkpoint files to download
        if checkpoint_files is None:
            checkpoint_files = [
                "checkpoints/operator.pt",
                "checkpoints/operator_ema.pt",
                "checkpoints/diffusion_residual.pt",
                "checkpoints/diffusion_residual_ema.pt",
                "checkpoints/scale/input_stats.pt",
                "checkpoints/scale/latent_stats.pt",
            ]

        # Authenticate with WandB
        api_key = os.environ.get("WANDB_API_KEY")
        if not api_key:
            raise ValueError("WANDB_API_KEY environment variable must be set")

        wandb.login(key=api_key)

        # Get run
        api = wandb.Api()
        run_path = f"{self.wandb_entity}/{self.wandb_project}/{run_id}"
        print(f"Accessing WandB run: {run_path}")

        try:
            run = api.run(run_path)
        except Exception as e:
            raise ValueError(f"Failed to access WandB run '{run_path}': {e}")

        # Download each checkpoint file
        downloaded_files = []
        for file_path in checkpoint_files:
            local_path = self.checkpoint_dir / Path(file_path).name

            # Skip if file exists and not forcing re-download
            if local_path.exists() and not force:
                print(f"✓ Checkpoint already exists: {local_path}")
                downloaded_files.append(local_path)
                continue

            try:
                print(f"Downloading {file_path}...")
                file_obj = run.file(file_path)

                # Create parent directory for scale stats
                if "scale/" in file_path:
                    scale_dir = self.checkpoint_dir / "scale"
                    scale_dir.mkdir(exist_ok=True)
                    download_root = str(self.checkpoint_dir / "scale")
                    # Extract just the filename for scale stats
                    file_obj.download(root=download_root, replace=True)
                    local_path = scale_dir / Path(file_path).name
                else:
                    file_obj.download(root=str(self.checkpoint_dir), replace=True)

                print(f"✓ Downloaded: {local_path}")
                downloaded_files.append(local_path)

            except Exception as e:
                print(f"⚠️  Could not download {file_path}: {e}")
                # Continue with other files even if one fails
                continue

        if not downloaded_files:
            raise ValueError(f"No checkpoints were successfully downloaded from run '{run_id}'")

        print(f"\n✓ Downloaded {len(downloaded_files)} checkpoint files")

        # Generate stage_status.json based on downloaded checkpoints
        self._generate_stage_status_from_checkpoints(downloaded_files)

        return downloaded_files

    def _generate_stage_status_from_checkpoints(self, downloaded_files: list) -> None:
        """Generate stage_status.json based on which checkpoint files exist.

        This allows auto-resume to work even when resuming from runs that don't
        have stage_status.json (e.g., crashed runs or runs before tracking was added).
        """
        import json
        from datetime import datetime, timezone

        # Map checkpoint files to stages
        stage_map = {
            "operator": ["operator.pt", "operator_ema.pt"],
            "diff_residual": ["diffusion_residual.pt", "diffusion_residual_ema.pt"],
            "consistency_distill": ["consistency_distill.pt"],
            "steady_prior": ["steady_prior.pt"],
        }

        # Determine which stages are complete based on checkpoint files
        completed_stages = {}
        for stage, checkpoint_files in stage_map.items():
            # Check if any of the expected checkpoints exist
            stage_complete = any(
                any(str(f).endswith(ckpt) for f in downloaded_files) for ckpt in checkpoint_files
            )

            if stage_complete:
                # Mark as completed with current timestamp
                completed_stages[stage] = {
                    "status": "completed",
                    "checkpoint": checkpoint_files[0],  # Use first expected checkpoint
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                }
            else:
                completed_stages[stage] = {"status": "not_started"}

        # Create stage_status.json
        stage_status = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "stages": completed_stages,
        }

        status_file = self.checkpoint_dir / "stage_status.json"
        status_file.write_text(json.dumps(stage_status, indent=2))

        # Report which stages were marked as complete
        complete_count = sum(1 for s in completed_stages.values() if s["status"] == "completed")
        if complete_count > 0:
            stage_names = [
                name for name, info in completed_stages.items() if info["status"] == "completed"
            ]
            print(
                f"✓ Generated stage_status.json: {complete_count} stages marked as completed ({', '.join(stage_names)})"
            )
        else:
            print(f"✓ Generated stage_status.json: no completed stages detected")

    def setup_wandb_resume(
        self,
        run_id: str,
        resume_mode: str = "allow",
    ) -> None:
        """Configure environment variables for WandB run resumption.

        Args:
            run_id: WandB run ID to resume
            resume_mode: WandB resume mode ('allow', 'must', 'never')
        """
        os.environ["WANDB_RUN_ID"] = run_id
        os.environ["WANDB_RESUME"] = resume_mode
        print(f"✓ Configured WandB to resume run: {run_id}")

    def verify_checkpoints(
        self,
        required_files: Optional[List[str]] = None,
    ) -> bool:
        """Verify that required checkpoint files exist.

        Args:
            required_files: List of required checkpoint filenames.
                If None, checks for operator checkpoints only.

        Returns:
            True if all required files exist
        """
        if required_files is None:
            required_files = [
                "operator.pt",
                "operator_ema.pt",
            ]

        missing_files = []
        for filename in required_files:
            file_path = self.checkpoint_dir / filename
            if not file_path.exists():
                missing_files.append(filename)

        if missing_files:
            print(f"⚠️  Missing checkpoints: {', '.join(missing_files)}")
            return False

        print(f"✓ All required checkpoints exist: {', '.join(required_files)}")
        return True

    def list_available_checkpoints(self) -> List[Path]:
        """List all checkpoint files in the checkpoint directory.

        Returns:
            List of paths to checkpoint files
        """
        checkpoint_files = []

        # Check main checkpoint directory
        for ext in [".pt", ".pth", ".ckpt"]:
            checkpoint_files.extend(self.checkpoint_dir.glob(f"*{ext}"))

        # Check scale subdirectory
        scale_dir = self.checkpoint_dir / "scale"
        if scale_dir.exists():
            for ext in [".pt", ".pth", ".ckpt"]:
                checkpoint_files.extend(scale_dir.glob(f"*{ext}"))

        return sorted(checkpoint_files)

    @staticmethod
    def get_run_checkpoints(
        run_id: str,
        entity: Optional[str] = None,
        project: Optional[str] = None,
    ) -> List[str]:
        """List all checkpoint files available in a WandB run.

        Args:
            run_id: WandB run ID
            entity: WandB entity (defaults to WANDB_ENTITY env var)
            project: WandB project (defaults to WANDB_PROJECT env var)

        Returns:
            List of checkpoint file paths in the run
        """
        entity = entity or os.environ.get("WANDB_ENTITY", "")
        project = project or os.environ.get("WANDB_PROJECT", "universal-simulator")

        if not entity:
            raise ValueError("WandB entity must be provided or set in WANDB_ENTITY")

        api_key = os.environ.get("WANDB_API_KEY")
        if not api_key:
            raise ValueError("WANDB_API_KEY environment variable must be set")

        wandb.login(key=api_key)

        api = wandb.Api()
        run_path = f"{entity}/{project}/{run_id}"

        try:
            run = api.run(run_path)
        except Exception as e:
            raise ValueError(f"Failed to access WandB run '{run_path}': {e}")

        # List all files in the run
        checkpoint_files = []
        for file in run.files():
            if file.name.startswith("checkpoints/") and any(
                file.name.endswith(ext) for ext in [".pt", ".pth", ".ckpt"]
            ):
                checkpoint_files.append(file.name)

        return checkpoint_files


# DDP-aware checkpoint save/load functions (Phase 5: Production Hardening)


def save_checkpoint(
    model: "torch.nn.Module",  # type: ignore
    optimizer: "torch.optim.Optimizer",  # type: ignore
    epoch: int,
    path: str | Path,
    is_distributed: bool = False,
    rank: int = 0,
) -> None:
    """Save checkpoint (rank 0 only in distributed mode).

    Args:
        model: Model to save (may be DDP-wrapped)
        optimizer: Optimizer to save
        epoch: Current epoch
        path: Path to save checkpoint
        is_distributed: Whether running in distributed mode
        rank: Current process rank
    """
    import torch

    if is_distributed and rank != 0:
        return  # Only rank 0 saves

    # Unwrap DDP if needed
    model_state = model.module.state_dict() if is_distributed else model.state_dict()

    checkpoint = {
        "model": model_state,
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }

    torch.save(checkpoint, path)
    if rank == 0:
        print(f"Checkpoint saved to {path}")


def load_checkpoint(
    model: "torch.nn.Module",  # type: ignore
    optimizer: "torch.optim.Optimizer",  # type: ignore
    path: str | Path,
    device: "torch.device",  # type: ignore
    is_distributed: bool = False,
) -> int:
    """Load checkpoint (all ranks in distributed mode).

    Args:
        model: Model to load into (may be DDP-wrapped)
        optimizer: Optimizer to load into
        path: Path to load checkpoint from
        device: Device to map checkpoint to
        is_distributed: Whether running in distributed mode

    Returns:
        Epoch number from checkpoint
    """
    import torch

    checkpoint = torch.load(path, map_location=device)

    # Load into DDP-wrapped or unwrapped model
    if is_distributed:
        model.module.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint["model"])

    optimizer.load_state_dict(checkpoint["optimizer"])
    epoch = checkpoint["epoch"]

    print(f"Checkpoint loaded from {path} (epoch {epoch})")
    return epoch
