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
    except Exception:
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
        print("Small Eval:               ❌ Not run")

    # Full eval status
    if metadata.get("last_full_eval"):
        full_eval_at = format_timestamp(metadata.get("last_full_eval_at"))
        full_nrmse = metadata["last_full_eval"].get("metric:nrmse", "N/A")
        print(f"Full Eval:                ✅ Run at {full_eval_at}")
        print(f"  └─ NRMSE:               {full_nrmse}")
    else:
        print("Full Eval:                ❌ Not run")

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
