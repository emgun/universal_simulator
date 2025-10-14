#!/usr/bin/env python
"""Download checkpoints from W&B artifacts or run files."""

import argparse
import os
import sys
from pathlib import Path

try:
    import wandb
except ImportError:
    print("Error: wandb not installed", file=sys.stderr)
    sys.exit(1)


def download_from_artifact(artifact_path: str, dest_dir: Path, filename: str) -> bool:
    """Download a checkpoint from a W&B artifact."""
    try:
        print(f"Downloading artifact: {artifact_path}", file=sys.stderr)
        api = wandb.Api()
        artifact = api.artifact(artifact_path)

        # Download the artifact
        artifact_dir = artifact.download(root=str(dest_dir / "tmp"))

        # Find the checkpoint file in the artifact
        artifact_path_obj = Path(artifact_dir)
        pt_files = list(artifact_path_obj.rglob("*.pt"))

        if not pt_files:
            print(f"  No .pt files found in artifact", file=sys.stderr)
            return False

        # Copy the first .pt file to the destination
        src_file = pt_files[0]
        dest_file = dest_dir / filename

        print(f"  Found: {src_file.name}", file=sys.stderr)
        print(f"  Copying to: {dest_file}", file=sys.stderr)

        import shutil
        shutil.copy2(src_file, dest_file)

        # Cleanup temp dir
        shutil.rmtree(artifact_path_obj)

        print(f"  ✓ Downloaded {filename}", file=sys.stderr)
        return True

    except Exception as e:
        print(f"  Error downloading artifact {artifact_path}: {e}", file=sys.stderr)
        return False


def download_from_run_files(run_path: str, dest_dir: Path, pattern: str, filename: str) -> bool:
    """Download checkpoint files directly from a run."""
    try:
        print(f"Checking run: {run_path}", file=sys.stderr)
        api = wandb.Api()
        run = api.run(run_path)

        # Find matching files
        matching_files = []
        for f in run.files():
            if pattern in f.name and f.name.endswith('.pt'):
                matching_files.append(f)

        if not matching_files:
            print(f"  No files matching '{pattern}' found", file=sys.stderr)
            return False

        # Download the first matching file
        file_obj = matching_files[0]
        print(f"  Found: {file_obj.name}", file=sys.stderr)

        dest_file = dest_dir / filename
        file_obj.download(root=str(dest_dir), replace=True)

        # Handle nested paths - move to root if needed
        downloaded_path = dest_dir / file_obj.name
        if downloaded_path != dest_file and downloaded_path.exists():
            downloaded_path.rename(dest_file)
            # Clean up any parent directories
            try:
                downloaded_path.parent.rmdir()
            except:
                pass

        print(f"  ✓ Downloaded {filename}", file=sys.stderr)
        return True

    except Exception as e:
        print(f"  Error downloading from run {run_path}: {e}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description="Download checkpoints from W&B")
    parser.add_argument("--dest", default="checkpoints/scale", help="Destination directory")
    parser.add_argument("--entity", default="emgun-morpheus-space", help="W&B entity")
    parser.add_argument("--project", default="universal-simulator", help="W&B project")
    parser.add_argument("--operator-artifact", help="Operator artifact path (e.g., run-xxx-history:v0)")
    parser.add_argument("--diffusion-artifact", help="Diffusion artifact path (e.g., run-xxx-history:v0)")
    parser.add_argument("--consistency-artifact", help="Consistency/distill artifact path (e.g., run-xxx-history:v0)")
    parser.add_argument("--operator-run", help="Operator run ID (alternative to artifact)")
    parser.add_argument("--diffusion-run", help="Diffusion run ID (alternative to artifact)")

    args = parser.parse_args()

    dest_dir = Path(args.dest)
    dest_dir.mkdir(parents=True, exist_ok=True)

    success = True

    # Download operator
    if args.operator_artifact:
        artifact_path = f"{args.entity}/{args.project}/{args.operator_artifact}"
        if not download_from_artifact(artifact_path, dest_dir, "operator.pt"):
            success = False
    elif args.operator_run:
        run_path = f"{args.entity}/{args.project}/{args.operator_run}"
        if not download_from_run_files(run_path, dest_dir, "operator", "operator.pt"):
            success = False

    # Download diffusion (try consistency artifact first if provided)
    diffusion_downloaded = False
    if args.consistency_artifact:
        artifact_path = f"{args.entity}/{args.project}/{args.consistency_artifact}"
        if download_from_artifact(artifact_path, dest_dir, "diffusion_residual.pt"):
            diffusion_downloaded = True

    if not diffusion_downloaded and args.diffusion_artifact:
        artifact_path = f"{args.entity}/{args.project}/{args.diffusion_artifact}"
        if not download_from_artifact(artifact_path, dest_dir, "diffusion_residual.pt"):
            success = False
    elif not diffusion_downloaded and args.diffusion_run:
        run_path = f"{args.entity}/{args.project}/{args.diffusion_run}"
        if not download_from_run_files(run_path, dest_dir, "diffusion", "diffusion_residual.pt"):
            success = False

    if success:
        print("\n✓ All checkpoints downloaded successfully", file=sys.stderr)
        print(f"  Operator: {dest_dir / 'operator.pt'}")
        print(f"  Diffusion: {dest_dir / 'diffusion_residual.pt'}")
        sys.exit(0)
    else:
        print("\n✗ Failed to download some checkpoints", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
