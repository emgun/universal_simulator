#!/usr/bin/env python3
"""
Generate a catalog of training/evaluation configs to keep the zoo tidy.

Usage:
    python scripts/config_catalog.py --output configs/catalog.csv
    python scripts/config_catalog.py --include-archive
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, List, Dict, Any

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ups.utils.config_loader import load_config_with_includes  # type: ignore


CONFIG_DIR = ROOT / "configs"
ARCHIVE_CONFIG_DIR = ROOT / "archive" / "configs"


def _iter_configs(include_archive: bool) -> Iterable[Path]:
    yield from sorted(CONFIG_DIR.glob("*.yaml"))
    if include_archive and ARCHIVE_CONFIG_DIR.exists():
        yield from sorted(ARCHIVE_CONFIG_DIR.glob("*.yaml"))


def _flatten_config(path: Path) -> Dict[str, Any]:
    cfg = load_config_with_includes(path)
    latent = cfg.get("latent", {})
    stages = cfg.get("stages", {})
    logging_cfg = cfg.get("logging", {}).get("wandb", {})
    ttc = cfg.get("ttc", {})

    operator_cfg = stages.get("operator", {}) if isinstance(stages, dict) else {}
    diff_cfg = stages.get("diff_residual", {}) if isinstance(stages, dict) else {}
    distill_cfg = stages.get("consistency_distill", {}) if isinstance(stages, dict) else {}

    record: Dict[str, Any] = {
        "name": path.stem,
        "path": path.relative_to(ROOT),
        "include": cfg.get("include"),
        "latent_dim": latent.get("dim"),
        "latent_tokens": latent.get("tokens"),
        "operator_hidden": operator_cfg.get("pdet", {}).get("hidden_dim")
        if isinstance(operator_cfg.get("pdet"), dict)
        else None,
        "operator_epochs": operator_cfg.get("epochs"),
        "operator_lr": operator_cfg.get("optimizer", {}).get("lr")
        if isinstance(operator_cfg.get("optimizer"), dict)
        else None,
        "operator_scheduler": operator_cfg.get("scheduler", {}).get("name")
        if isinstance(operator_cfg.get("scheduler"), dict)
        else None,
        "diff_epochs": diff_cfg.get("epochs"),
        "distill_epochs": distill_cfg.get("epochs"),
        "ttc_enabled": bool(ttc.get("enabled")),
        "wandb_tags": ";".join(logging_cfg.get("tags", []) or []),
        "wandb_run_name": logging_cfg.get("run_name"),
    }
    return record


def build_catalog(include_archive: bool) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for cfg_path in _iter_configs(include_archive):
        try:
            records.append(_flatten_config(cfg_path))
        except Exception as exc:
            records.append(
                {
                    "name": cfg_path.stem,
                    "path": cfg_path.relative_to(ROOT),
                    "error": str(exc),
                }
            )
    return records


def write_csv(records: List[Dict[str, Any]], destination: Path) -> None:
    if not records:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in records for key in row.keys()})
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a catalog of configuration files.")
    parser.add_argument(
        "--output",
        default="configs/catalog.csv",
        help="Path to write catalog CSV (default: configs/catalog.csv)",
    )
    parser.add_argument(
        "--include-archive",
        action="store_true",
        help="Also include archived configs in the catalog",
    )
    args = parser.parse_args()

    records = build_catalog(include_archive=args.include_archive)
    write_csv(records, Path(args.output))
    print(f"Catalog written to {args.output} ({len(records)} configs)")


if __name__ == "__main__":
    main()
