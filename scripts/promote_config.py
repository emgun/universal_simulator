#!/usr/bin/env python3
"""Promote successful experimental configs to production status.

This script validates and promotes experimental configs to production after
successful validation runs. It ensures configs meet quality gates and updates
documentation.

Promotion Process:
1. Validate experiment passed all gates (NRMSE, gates_passed, etc.)
2. Check for required metadata (WandB run ID, checkpoint paths)
3. Copy config to production location (configs/)
4. Update leaderboard (optional)
5. Update experiment metadata with promotion info

Usage Examples:
    # Promote specific experiment after manual review
    python scripts/promote_config.py experiments/2025-01-22-64dim-latent

    # Promote with leaderboard update
    python scripts/promote_config.py experiments/2025-01-22-64dim-latent \\
        --update-leaderboard --leaderboard-csv reports/leaderboard.csv

    # Auto-promote if gates pass
    python scripts/promote_config.py experiments/2025-01-22-64dim-latent \\
        --auto-promote --nrmse-threshold 0.10

    # Promote to custom location
    python scripts/promote_config.py experiments/2025-01-22-64dim-latent \\
        --production-dir configs/ \\
        --config-name train_burgers_64dim.yaml
"""

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import yaml


def load_yaml(path: Path) -> Dict:
    """Load YAML config file."""
    with open(path) as f:
        return yaml.safe_load(f)


def validate_experiment_results(
    exp_dir: Path,
    reports_dir: Path,
    nrmse_threshold: Optional[float] = None,
) -> tuple[bool, Dict]:
    """Validate that experiment has successful training results.

    Args:
        exp_dir: Path to experiment directory
        reports_dir: Path to reports directory
        nrmse_threshold: Optional NRMSE threshold for auto-promotion

    Returns:
        (is_valid, metrics_dict)
    """
    exp_name = exp_dir.name

    # Check experiment metadata first
    metadata_file = exp_dir / "metadata.json"
    if metadata_file.exists():
        try:
            with open(metadata_file) as f:
                metadata = json.load(f)
                if "status" in metadata and metadata["status"] == "failed":
                    return False, {"error": "Experiment marked as failed in metadata"}
        except Exception as e:
            return False, {"error": f"Failed to load experiment metadata: {e}"}

    # Look for corresponding evaluation reports
    report_patterns = [
        f"*{exp_name}*.json",
        "full_eval.json",
        "eval_results.json",
    ]

    reports = []
    if reports_dir.exists():
        for pattern in report_patterns:
            reports.extend(reports_dir.glob(pattern))

    if not reports:
        return False, {"error": "No evaluation reports found"}

    # Load most recent report
    reports.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    latest_report = reports[0]

    try:
        with open(latest_report) as f:
            metrics = json.load(f)
    except Exception as e:
        return False, {"error": f"Failed to load report: {e}"}

    # Check for required metrics
    required_fields = ["final_nrmse", "gates_passed"]
    missing_fields = [f for f in required_fields if f not in metrics]
    if missing_fields:
        return False, {"error": f"Missing required fields: {missing_fields}", "metrics": metrics}

    # Check gates passed
    if not metrics["gates_passed"]:
        return False, {"error": "Gates failed", "metrics": metrics}

    # Check NRMSE threshold if provided
    if nrmse_threshold is not None:
        final_nrmse = metrics["final_nrmse"]
        if final_nrmse > nrmse_threshold:
            return False, {
                "error": f"NRMSE {final_nrmse:.4f} exceeds threshold {nrmse_threshold:.4f}",
                "metrics": metrics,
            }

    return True, metrics


def create_promotion_metadata(
    exp_dir: Path,
    metrics: Dict,
    production_config_path: Path,
) -> Dict:
    """Create metadata for promoted config."""
    return {
        "experiment_dir": str(exp_dir),
        "experiment_name": exp_dir.name,
        "promoted_at": datetime.now().isoformat(),
        "promoted_to": str(production_config_path),
        "metrics": metrics,
        "promotion_reason": "Passed validation gates and NRMSE threshold",
    }


def promote_experiment(
    exp_dir: Path,
    production_dir: Path,
    config_name: Optional[str] = None,
    update_leaderboard: bool = False,
    leaderboard_csv: Optional[Path] = None,
    metrics: Optional[Dict] = None,
    dry_run: bool = False,
) -> Optional[Path]:
    """Promote experimental config to production.

    Args:
        exp_dir: Path to experiment directory
        production_dir: Production configs directory
        config_name: Name for promoted config (default: based on experiment name)
        update_leaderboard: Whether to update leaderboard
        leaderboard_csv: Path to leaderboard CSV
        metrics: Metrics from validation
        dry_run: If True, only print actions

    Returns:
        Path to promoted config (None if dry_run)
    """
    # Determine config name
    if not config_name:
        # Use experiment name as base
        exp_name = exp_dir.name
        # Remove date prefix if present (YYYY-MM-DD-)
        parts = exp_name.split("-", 3)
        if len(parts) >= 4 and parts[0].isdigit() and len(parts[0]) == 4:
            # Has date prefix
            base_name = parts[3]
        else:
            base_name = exp_name
        config_name = f"train_{base_name}.yaml"

    production_dir.mkdir(parents=True, exist_ok=True)
    promoted_path = production_dir / config_name

    # Source config from experiment directory
    source_config = exp_dir / "config.yaml"
    if not source_config.exists():
        print(f"❌ Error: No config.yaml found in {exp_dir}")
        return None

    if dry_run:
        print(f"\n[DRY RUN] Would promote:")
        print(f"  From: {exp_dir}")
        print(f"  Config: {source_config}")
        print(f"  To:   {promoted_path}")
        if metrics:
            print(f"  Metrics:")
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"    {key}: {value}")
        return None

    # Copy config to promoted location
    shutil.copy2(source_config, promoted_path)
    print(f"✓ Promoted config: {exp_dir.name} → {promoted_path.name}")

    # Write promotion metadata to experiment directory
    metadata = create_promotion_metadata(exp_dir, metrics or {}, promoted_path)
    metadata_path = exp_dir / "promotion.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Wrote promotion metadata: {metadata_path}")

    # Update experiment metadata status
    exp_metadata_path = exp_dir / "metadata.json"
    if exp_metadata_path.exists():
        with open(exp_metadata_path) as f:
            exp_metadata = json.load(f)
    else:
        exp_metadata = {}

    exp_metadata["status"] = "success"
    exp_metadata["promoted_at"] = datetime.now().isoformat()
    exp_metadata["promoted_to"] = str(promoted_path)

    with open(exp_metadata_path, "w") as f:
        json.dump(exp_metadata, f, indent=2)
    print(f"  Updated experiment metadata: {exp_metadata_path}")

    # Update leaderboard if requested
    if update_leaderboard and leaderboard_csv:
        update_leaderboard_entry(promoted_path, metadata, leaderboard_csv)

    return promoted_path


def update_leaderboard_entry(
    config_path: Path,
    metadata: Dict,
    leaderboard_csv: Path,
) -> None:
    """Update leaderboard with promoted config entry.

    Args:
        config_path: Path to promoted config
        metadata: Promotion metadata with metrics
        leaderboard_csv: Path to leaderboard CSV
    """
    if not leaderboard_csv.exists():
        print(f"  ⚠ Leaderboard not found: {leaderboard_csv}")
        return

    # Try to import leaderboard module
    try:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from ups.utils.leaderboard import add_entry, load_leaderboard

        # Load existing leaderboard
        leaderboard = load_leaderboard(leaderboard_csv)

        # Create entry from metadata
        metrics = metadata.get("metrics", {})
        entry = {
            "config_name": config_path.name,
            "experiment_name": metadata.get("experiment_name", ""),
            "promoted_at": metadata.get("promoted_at", ""),
            "final_nrmse": metrics.get("final_nrmse", None),
            "gates_passed": metrics.get("gates_passed", None),
        }

        # Add to leaderboard
        add_entry(leaderboard, entry)
        leaderboard.to_csv(leaderboard_csv, index=False)
        print(f"  ✓ Updated leaderboard: {leaderboard_csv}")

    except Exception as e:
        print(f"  ⚠ Failed to update leaderboard: {e}")
        print(f"    Manually add config to leaderboard if needed")


def main():
    parser = argparse.ArgumentParser(
        description="Promote experimental configs to production",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "experiment",
        type=Path,
        help="Path to experiment directory (e.g., experiments/2025-01-22-my-experiment/)",
    )
    parser.add_argument(
        "--production-dir",
        type=Path,
        default=Path("configs"),
        help="Production configs directory (default: configs/)",
    )
    parser.add_argument(
        "--config-name",
        type=str,
        help="Name for promoted config (default: derived from experiment name)",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("reports"),
        help="Directory containing evaluation reports (default: reports/)",
    )
    parser.add_argument(
        "--auto-promote",
        action="store_true",
        help="Automatically promote if validation passes",
    )
    parser.add_argument(
        "--nrmse-threshold",
        type=float,
        help="Maximum NRMSE for auto-promotion (e.g., 0.10)",
    )
    parser.add_argument(
        "--update-leaderboard",
        action="store_true",
        help="Update leaderboard with promoted config",
    )
    parser.add_argument(
        "--leaderboard-csv",
        type=Path,
        default=Path("reports/leaderboard.csv"),
        help="Path to leaderboard CSV (default: reports/leaderboard.csv)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without executing them",
    )

    args = parser.parse_args()

    # Validate experiment directory
    if not args.experiment.exists():
        print(f"❌ Error: Experiment directory not found: {args.experiment}")
        return 1

    if not args.experiment.is_dir():
        print(f"❌ Error: Path is not a directory: {args.experiment}")
        return 1

    if not (args.experiment / "config.yaml").exists():
        print(f"❌ Error: No config.yaml found in {args.experiment}")
        return 1

    # Validate results
    print(f"Validating results for: {args.experiment.name}")
    is_valid, result = validate_experiment_results(
        args.experiment,
        args.reports_dir,
        args.nrmse_threshold,
    )

    if not is_valid:
        print(f"\n✗ Validation failed: {result.get('error', 'Unknown error')}")
        if "metrics" in result:
            print("\nAvailable metrics:")
            for key, value in result["metrics"].items():
                print(f"  {key}: {value}")
        if not args.auto_promote:
            print("\nTo promote anyway, omit --auto-promote and --nrmse-threshold")
        return 1

    metrics = result
    print("✓ Validation passed")
    if "final_nrmse" in metrics:
        print(f"  NRMSE: {metrics['final_nrmse']:.4f}")
    print(f"  Gates: {'PASS' if metrics.get('gates_passed') else 'FAIL'}")

    # Promote config
    if args.auto_promote or not args.dry_run or args.dry_run:
        promoted_path = promote_experiment(
            args.experiment,
            args.production_dir,
            args.config_name,
            args.update_leaderboard,
            args.leaderboard_csv,
            metrics,
            args.dry_run,
        )

        if promoted_path and not args.dry_run:
            print(f"\n✓ Successfully promoted experiment to: {promoted_path}")
            print(f"\nNext steps:")
            print(f"  1. Review {promoted_path}")
            print(f"  2. Test with: python scripts/train.py --config {promoted_path} --stage operator --epochs 1")
            print(f"  3. Archive experiment: python scripts/archive_experiments.py --status success")
    else:
        print("\nTo promote, remove --dry-run")

    return 0


if __name__ == "__main__":
    exit(main())
