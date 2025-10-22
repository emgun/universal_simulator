#!/usr/bin/env python3
"""Archive completed or failed experimental configs and their artifacts.

This script implements automated experiment lifecycle management to prevent
experimental configs, markdowns, and artifacts from piling up.

Experiment Classification:
- active: Recent training runs (< 7 days old) or in-progress
- success: Completed runs with passing gates
- failed: Runs with errors or failing gates
- stale: No recent activity (> 30 days)

Directory Structure:
experiments/
  YYYY-MM-DD-experiment-name/
    config.yaml
    notes.md
    metadata.json (optional)

experiments-archive/
  YYYY-MM-DD-experiment-name/
    config.yaml
    metadata.json (generated during archiving)
    notes.md
    reports/ (optional)
    checkpoints/ (optional)
"""

import argparse
import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Literal, Optional

# Experiment status types
ExperimentStatus = Literal["active", "success", "failed", "stale"]


def find_experiment_dirs(experiments_dir: Path) -> List[Path]:
    """Find all experiment directories.

    Args:
        experiments_dir: Path to experiments/ directory

    Returns:
        List of experiment directory paths
    """
    if not experiments_dir.exists():
        return []

    # Find all subdirectories that contain a config.yaml
    experiments = []
    for item in experiments_dir.iterdir():
        if item.is_dir() and (item / "config.yaml").exists():
            experiments.append(item)

    return sorted(experiments)


def classify_experiment(
    exp_dir: Path,
    artifacts_dir: Path,
    reports_dir: Path,
    max_active_days: int = 7,
    max_stale_days: int = 30,
) -> ExperimentStatus:
    """Classify experiment status based on artifacts and timestamps.

    Args:
        exp_dir: Path to experiment directory
        artifacts_dir: Path to artifacts/runs directory
        reports_dir: Path to reports directory
        max_active_days: Days before considering experiment inactive
        max_stale_days: Days before considering experiment stale

    Returns:
        Experiment status classification
    """
    now = datetime.now()

    # Check directory modification time
    dir_mtime = datetime.fromtimestamp(exp_dir.stat().st_mtime)
    dir_age = (now - dir_mtime).days

    # Check config.yaml modification time
    config_file = exp_dir / "config.yaml"
    config_mtime = datetime.fromtimestamp(config_file.stat().st_mtime)
    config_age = (now - config_mtime).days

    # Check for metadata.json with explicit status
    metadata_file = exp_dir / "metadata.json"
    if metadata_file.exists():
        try:
            with open(metadata_file) as f:
                metadata = json.load(f)
                if "status" in metadata:
                    # Trust explicit status in metadata
                    return metadata["status"]
        except Exception:
            pass

    # Check for recent reports
    recent_reports = []
    if reports_dir.exists():
        for report in reports_dir.glob("*.json"):
            report_mtime = datetime.fromtimestamp(report.stat().st_mtime)
            report_age = (now - report_mtime).days
            if report_age < max_active_days:
                recent_reports.append(report)

                # Check report for success/failure indicators
                try:
                    with open(report) as f:
                        data = json.load(f)
                        if "gates_passed" in data:
                            return "success" if data["gates_passed"] else "failed"
                        if "error" in data or "exception" in data:
                            return "failed"
                except Exception:
                    pass

    # Check artifacts/runs directory for experiment-specific run
    exp_name = exp_dir.name
    if artifacts_dir.exists():
        for run_dir in artifacts_dir.iterdir():
            if not run_dir.is_dir():
                continue

            run_mtime = datetime.fromtimestamp(run_dir.stat().st_mtime)
            run_age = (now - run_mtime).days

            if run_age < max_active_days:
                # Check for completion indicators
                metrics_file = run_dir / "metrics.json"
                if metrics_file.exists():
                    try:
                        with open(metrics_file) as f:
                            metrics = json.load(f)
                            if "final_nrmse" in metrics:
                                # Has final metrics
                                if "gates_passed" in metrics:
                                    return "success" if metrics["gates_passed"] else "failed"
                                return "success"
                    except Exception:
                        pass

                # Has recent activity but no completion
                return "active"

    # Check for recent checkpoints in experiment directory
    exp_checkpoints = exp_dir / "checkpoints"
    if exp_checkpoints.exists():
        for ckpt in exp_checkpoints.glob("*.ckpt"):
            ckpt_mtime = datetime.fromtimestamp(ckpt.stat().st_mtime)
            ckpt_age = (now - ckpt_mtime).days
            if ckpt_age < max_active_days:
                return "active"

    # Check notes.md for status indicators
    notes_file = exp_dir / "notes.md"
    if notes_file.exists():
        notes_content = notes_file.read_text().lower()
        if "✓" in notes_content or "success" in notes_content or "passed" in notes_content:
            return "success"
        if "✗" in notes_content or "failed" in notes_content or "error" in notes_content:
            return "failed"

    # Age-based classification
    age = min(dir_age, config_age)
    if age > max_stale_days:
        return "stale"
    elif age < max_active_days:
        return "active"
    else:
        # Moderate age, no clear indicators
        return "failed"


def create_experiment_metadata(
    exp_dir: Path,
    status: ExperimentStatus,
    artifacts_dir: Path,
    reports_dir: Path,
) -> Dict:
    """Create metadata document for archived experiment."""
    config_file = exp_dir / "config.yaml"
    config_mtime = datetime.fromtimestamp(config_file.stat().st_mtime)

    metadata = {
        "experiment_name": exp_dir.name,
        "status": status,
        "created_at": config_mtime.isoformat(),
        "archived_at": datetime.now().isoformat(),
        "artifacts": [],
        "reports": [],
        "checkpoints": [],
    }

    # Collect artifact info
    if artifacts_dir.exists():
        for run_dir in artifacts_dir.iterdir():
            if run_dir.is_dir():
                try:
                    size = sum(f.stat().st_size for f in run_dir.rglob("*") if f.is_file())
                    metadata["artifacts"].append({
                        "path": str(run_dir.relative_to(artifacts_dir.parent)),
                        "size_mb": size / (1024 * 1024),
                    })
                except Exception:
                    pass

    # Collect report info
    if reports_dir.exists():
        for report in reports_dir.glob("*.json"):
            metadata["reports"].append({
                "path": str(report.relative_to(reports_dir.parent)),
                "modified": datetime.fromtimestamp(report.stat().st_mtime).isoformat(),
            })

    # Collect checkpoint info from experiment directory
    exp_checkpoints = exp_dir / "checkpoints"
    if exp_checkpoints.exists():
        for ckpt in exp_checkpoints.glob("*.ckpt"):
            metadata["checkpoints"].append({
                "path": str(ckpt.relative_to(exp_dir)),
                "size_mb": ckpt.stat().st_size / (1024 * 1024),
                "modified": datetime.fromtimestamp(ckpt.stat().st_mtime).isoformat(),
            })

    return metadata


def archive_experiment(
    exp_dir: Path,
    status: ExperimentStatus,
    archive_root: Path,
    artifacts_dir: Path,
    reports_dir: Path,
    dry_run: bool = False,
) -> None:
    """Archive an experiment with all its artifacts.

    Args:
        exp_dir: Path to experiment directory
        status: Experiment status classification
        archive_root: Root directory for archives (experiments-archive/)
        artifacts_dir: Path to artifacts/runs
        reports_dir: Path to reports
        dry_run: If True, only print actions without executing
    """
    # Archive directory has same name as experiment
    archive_dir = archive_root / exp_dir.name

    # Create metadata
    metadata = create_experiment_metadata(exp_dir, status, artifacts_dir, reports_dir)

    if dry_run:
        print(f"\n[DRY RUN] Would archive: {exp_dir.name}")
        print(f"  Status: {status}")
        print(f"  Archive to: {archive_dir}")
        print(f"  Files to copy:")
        for file in exp_dir.iterdir():
            if file.is_file():
                print(f"    - {file.name}")
        if metadata['artifacts']:
            print(f"  Artifacts: {len(metadata['artifacts'])} run directories")
        if metadata['reports']:
            print(f"  Reports: {len(metadata['reports'])} files")
        if metadata['checkpoints']:
            print(f"  Checkpoints: {len(metadata['checkpoints'])} files")
        return

    # Create archive directory
    archive_dir.mkdir(parents=True, exist_ok=True)

    # Copy all files from experiment directory
    for item in exp_dir.iterdir():
        if item.is_file():
            shutil.copy2(item, archive_dir / item.name)
        elif item.is_dir() and item.name not in ["checkpoints", "artifacts"]:
            # Copy non-large directories (e.g., plots/)
            shutil.copytree(item, archive_dir / item.name, dirs_exist_ok=True)

    # Write/update metadata
    with open(archive_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Optionally copy checkpoints (can be large)
    if metadata["checkpoints"]:
        checkpoints_archive = archive_dir / "checkpoints"
        checkpoints_archive.mkdir(exist_ok=True)
        for ckpt_info in metadata["checkpoints"]:
            src = exp_dir / ckpt_info["path"]
            if src.exists():
                shutil.copy2(src, checkpoints_archive / src.name)

    print(f"✓ Archived: {exp_dir.name} → {archive_dir}")

    # Remove original experiment directory
    shutil.rmtree(exp_dir)
    print(f"  Removed original: {exp_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Archive experimental configs and artifacts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=Path("experiments"),
        help="Directory containing active experiments (default: experiments/)",
    )
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=Path("experiments-archive"),
        help="Root directory for archives (default: experiments-archive/)",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts/runs"),
        help="Directory containing run artifacts (default: artifacts/runs/)",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("reports"),
        help="Directory containing reports (default: reports/)",
    )
    parser.add_argument(
        "--status",
        type=str,
        choices=["success", "failed", "stale", "all"],
        default="all",
        help="Archive only experiments with this status (default: all)",
    )
    parser.add_argument(
        "--max-active-days",
        type=int,
        default=7,
        help="Days before considering experiment inactive (default: 7)",
    )
    parser.add_argument(
        "--max-stale-days",
        type=int,
        default=30,
        help="Days before considering experiment stale (default: 30)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without executing them",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="List experiments and their status without archiving",
    )

    args = parser.parse_args()

    # Find experiment directories
    experiments = find_experiment_dirs(args.experiments_dir)

    if not experiments:
        print(f"No experiments found in {args.experiments_dir}")
        return

    print(f"Found {len(experiments)} experiment(s) in {args.experiments_dir}\n")

    # Classify and process each experiment
    experiments_by_status = {"active": [], "success": [], "failed": [], "stale": []}

    for exp_dir in experiments:
        status = classify_experiment(
            exp_dir,
            args.artifacts_dir,
            args.reports_dir,
            args.max_active_days,
            args.max_stale_days,
        )
        experiments_by_status[status].append(exp_dir)

    # Print summary
    print("Experiment Summary:")
    print(f"  Active: {len(experiments_by_status['active'])}")
    print(f"  Success: {len(experiments_by_status['success'])}")
    print(f"  Failed: {len(experiments_by_status['failed'])}")
    print(f"  Stale: {len(experiments_by_status['stale'])}\n")

    if args.list_only:
        for status, exps in experiments_by_status.items():
            if exps:
                print(f"\n{status.upper()}:")
                for exp in exps:
                    print(f"  - {exp.name}")
        return

    # Archive experiments
    to_archive = []
    if args.status == "all":
        # Archive everything except active
        to_archive = (
            experiments_by_status["success"]
            + experiments_by_status["failed"]
            + experiments_by_status["stale"]
        )
    else:
        to_archive = experiments_by_status[args.status]

    if not to_archive:
        print(f"No experiments to archive with status: {args.status}")
        return

    print(f"Archiving {len(to_archive)} experiment(s)...\n")

    for exp_dir in to_archive:
        status = classify_experiment(
            exp_dir,
            args.artifacts_dir,
            args.reports_dir,
            args.max_active_days,
            args.max_stale_days,
        )
        archive_experiment(
            exp_dir,
            status,
            args.archive_root,
            args.artifacts_dir,
            args.reports_dir,
            args.dry_run,
        )

    if not args.dry_run:
        print(f"\n✓ Archived {len(to_archive)} experiment(s) to {args.archive_root}")
    else:
        print(f"\n[DRY RUN] Would have archived {len(to_archive)} experiment(s)")


if __name__ == "__main__":
    main()
