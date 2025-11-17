#!/usr/bin/env python
from __future__ import annotations

"""Sequential multi-stage runner for Lightning training."""

import argparse
import subprocess
from typing import Optional

import yaml


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def run_stage(config_path: str, stage: str, devices: Optional[int]) -> None:
    cmd = [
        "python",
        "scripts/train_lightning.py",
        "--config",
        config_path,
        "--stage",
        stage,
    ]
    if devices is not None:
        cmd.extend(["--devices", str(devices)])
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-config", required=True, help="Path to training config YAML")
    parser.add_argument(
        "--devices",
        type=int,
        default=None,
        help="Override device count for all stages (default: use training.num_gpus)",
    )
    args = parser.parse_args()

    cfg = load_config(args.train_config)
    stages_cfg = cfg.get("stages", {}) if isinstance(cfg.get("stages"), dict) else {}

    if stages_cfg.get("operator", {}).get("epochs", 0) > 0:
        print("=" * 60)
        print("Stage 1: Operator Training (Lightning)")
        print("=" * 60)
        run_stage(args.train_config, "operator", args.devices)

    # Placeholder for future Lightning stage support
    if stages_cfg.get("diff_residual", {}).get("epochs", 0) > 0:
        print("=" * 60)
        print("Stage 2: Diffusion Residual Training (Lightning)")
        print("=" * 60)
        raise NotImplementedError("Diffusion residual Lightning stage not yet implemented")


if __name__ == "__main__":
    main()
