#!/usr/bin/env python
from __future__ import annotations

"""Aggregate evaluation metrics into a local leaderboard (with optional W&B sync)."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ups.utils.leaderboard import update_leaderboard


def _parse_tags(values):
    tags = {}
    for tag in values or []:
        if "=" not in tag:
            raise ValueError(f"Tag '{tag}' must be formatted as key=value")
        key, value = tag.split("=", 1)
        tags[key] = value
    return tags


def main() -> None:
    parser = argparse.ArgumentParser(description="Update leaderboard CSV/HTML from evaluation metrics")
    parser.add_argument("--metrics", required=True, help="Path to metrics JSON produced by scripts/evaluate.py")
    parser.add_argument("--run-id", required=True, help="Unique identifier for this evaluation run")
    parser.add_argument("--config", help="Config file used (recorded in leaderboard)")
    parser.add_argument("--leaderboard", default="reports/leaderboard.csv", help="Leaderboard CSV path")
    parser.add_argument("--html", default="reports/leaderboard.html", help="Leaderboard HTML path")
    parser.add_argument("--label", help="Label for the evaluation (e.g., small_eval, full_eval)")
    parser.add_argument("--notes", help="Optional free-form notes to attach")
    parser.add_argument(
        "--tag",
        action="append",
        default=[],
        help="Additional key=value pairs to record (may be repeated)",
    )
    parser.add_argument("--wandb", action="store_true", help="Also log metrics row to Weights & Biases")
    parser.add_argument("--wandb-project", help="W&B project name (defaults to env)")
    parser.add_argument("--wandb-entity", help="W&B entity name (defaults to env)")
    parser.add_argument("--wandb-run-name", help="Override W&B run name if a new run is created")

    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    update_leaderboard(
        metrics_path=metrics_path,
        run_id=args.run_id,
        leaderboard_csv=Path(args.leaderboard),
        leaderboard_html=Path(args.html),
        label=args.label,
        config=args.config,
        notes=args.notes,
        tags=_parse_tags(args.tag),
        wandb_log=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name or args.run_id,
    )


if __name__ == "__main__":
    main()
