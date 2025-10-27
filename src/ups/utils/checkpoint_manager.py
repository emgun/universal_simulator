"""Checkpoint management utility for downloading and resuming from WandB runs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, List
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
        return downloaded_files

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
