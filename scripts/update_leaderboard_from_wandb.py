#!/usr/bin/env python3
"""
Extract metrics from WandB runs and update the leaderboard.

Usage:
    python scripts/update_leaderboard_from_wandb.py run_id1 run_id2 run_id3
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import tempfile

try:
    import wandb
except ImportError:
    print("❌ Required package not installed:")
    print("   pip install wandb")
    sys.exit(1)

# Add src to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ups.utils.leaderboard import update_leaderboard


def fetch_run_metrics(run_id: str, entity: str = "emgun-morpheus-space", project: str = "universal-simulator") -> Optional[Dict[str, Any]]:
    """Fetch metrics from a WandB run."""
    api = wandb.Api()

    try:
        run = api.run(f"{entity}/{project}/{run_id}")
    except Exception as e:
        print(f"❌ Failed to fetch run {run_id}: {e}")
        return None

    print(f"✅ Fetched: {run.name} ({run_id})")

    # Extract summary metrics
    summary = run.summary._json_dict
    config = run.config

    # Build metrics dict
    metrics = {}

    # Add evaluation metrics if available - handle nested structure
    for key, value in summary.items():
        # Skip non-numeric values (tables, images, etc.)
        if not isinstance(value, (int, float)):
            continue

        if key.startswith("eval/"):
            # Extract metric name and type
            # eval/baseline_nrmse -> metric:nrmse (baseline)
            # eval/ttc_nrmse -> metric:nrmse (ttc)
            # eval/nrmse -> metric:nrmse (general)
            # eval/physics/conservation_gap -> metric:conservation_gap (physics)

            key_parts = key.replace("eval/", "").split("/")

            if len(key_parts) == 2:  # eval/physics/conservation_gap
                if key_parts[0] == "physics":
                    metrics[f"metric:{key_parts[1]}"] = value
            elif len(key_parts) == 1:  # eval/baseline_nrmse or eval/nrmse
                metric_name = key_parts[0]
                if metric_name.startswith("baseline_"):
                    # baseline metric
                    clean_name = metric_name.replace("baseline_", "")
                    metrics[f"metric:{clean_name}_baseline"] = value
                elif metric_name.startswith("ttc_"):
                    # ttc metric
                    clean_name = metric_name.replace("ttc_", "")
                    metrics[f"metric:{clean_name}_ttc"] = value
                else:
                    # general metric
                    metrics[f"metric:{metric_name}"] = value
        elif key.startswith("training/"):
            # Clean up key name
            clean_key = key.replace("training/", "training:")
            metrics[clean_key] = value

    # Add some metadata
    metrics["wandb_run_id"] = run_id
    metrics["wandb_run_name"] = run.name
    metrics["wandb_url"] = run.url
    metrics["created_at"] = run.created_at
    metrics["state"] = run.state

    # Add config info
    if "latent" in config and "dim" in config["latent"]:
        metrics["latent_dim"] = config["latent"]["dim"]
    if "latent" in config and "tokens" in config["latent"]:
        metrics["latent_tokens"] = config["latent"]["tokens"]

    print(f"   Extracted {len([k for k in metrics.keys() if k.startswith('metric:')])} evaluation metrics")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Update leaderboard from WandB runs")
    parser.add_argument("run_ids", nargs="+", help="WandB run IDs to add to leaderboard")
    parser.add_argument("--entity", default="emgun-morpheus-space", help="WandB entity")
    parser.add_argument("--project", default="universal-simulator", help="WandB project")
    parser.add_argument("--leaderboard", default="reports/leaderboard.csv", help="Leaderboard CSV path")
    parser.add_argument("--html", default="reports/leaderboard.html", help="Leaderboard HTML path")
    parser.add_argument("--label", default="wandb_import", help="Label for these runs")

    args = parser.parse_args()

    print(f"📊 Fetching {len(args.run_ids)} runs from WandB...")

    for run_id in args.run_ids:
        metrics = fetch_run_metrics(run_id, args.entity, args.project)

        if metrics is None:
            print(f"⚠️  Skipping {run_id}")
            continue

        # Write metrics to temporary JSON file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(metrics, f, indent=2)
            temp_path = Path(f.name)

        try:
            # Update leaderboard
            update_leaderboard(
                metrics_path=temp_path,
                run_id=run_id,
                leaderboard_csv=Path(args.leaderboard),
                leaderboard_html=Path(args.html),
                label=args.label,
                config=metrics.get("wandb_run_name", "unknown"),
                notes=f"Imported from WandB: {metrics.get('wandb_url', '')}",
            )
            print(f"✅ Added {run_id} to leaderboard")
        except Exception as e:
            print(f"❌ Failed to update leaderboard for {run_id}: {e}")
        finally:
            # Clean up temp file
            temp_path.unlink(missing_ok=True)

    print(f"\n✅ Leaderboard updated: {args.leaderboard}")
    print(f"📄 HTML report: {args.html}")


if __name__ == "__main__":
    main()
