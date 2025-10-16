#!/usr/bin/env python
from __future__ import annotations

"""Download W&B run metrics via the public API for offline analysis."""

import argparse
import sys
from pathlib import Path

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover - optional dependency
    raise SystemExit("pandas is required. Install with `pip install pandas`." ) from exc

try:
    import wandb
except ImportError as exc:  # pragma: no cover
    raise SystemExit("wandb is required. Install with `pip install wandb`." ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch W&B run history and save as CSV")
    parser.add_argument("run_id", help="W&B run ID (e.g. jz11ge11)")
    parser.add_argument("--entity", default=None, help="W&B entity (defaults to current login)")
    parser.add_argument("--project", required=True, help="W&B project name")
    parser.add_argument("--metrics", nargs="*", help="Optional subset of metric columns to keep")
    parser.add_argument("--out", default=None, help="Output CSV path (default: reports/wandb_<run>.csv)")
    parser.add_argument("--max_rows", type=int, default=None, help="Optional limit on history rows")
    return parser.parse_args()


def fetch_history(run_path: str, metrics: list[str] | None, max_rows: int | None) -> pd.DataFrame:
    api = wandb.Api()
    try:
        run = api.run(run_path)
    except wandb.CommError as exc:
        raise SystemExit(f"Failed to load run {run_path}: {exc}") from exc
    history = run.history(keys=metrics, pandas=True, samples=max_rows)
    return history


def main() -> None:
    args = parse_args()
    project = args.project
    entity = args.entity or wandb.env.get_entity() or getattr(wandb.env, "get_username", lambda: None)()
    if entity is None:
        raise SystemExit(
            "Could not determine W&B entity. Pass --entity explicitly, set WANDB_ENTITY, or run `wandb login`."
        )
    run_path = f"{entity}/{project}/{args.run_id}"
    history = fetch_history(run_path, args.metrics, args.max_rows)
    if history.empty:
        print(f"No history rows returned for {run_path}")
        return
    out_path = Path(args.out) if args.out else Path("reports") / f"wandb_{args.run_id}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(out_path, index=False)
    print(f"Saved history with {len(history)} rows to {out_path}")


if __name__ == "__main__":
    main()
