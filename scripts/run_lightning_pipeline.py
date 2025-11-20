#!/usr/bin/env python
from __future__ import annotations

"""Sequential multi-stage runner for Lightning training."""

import argparse
import os
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
    env = os.environ.copy()
    subprocess.run(cmd, check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-config", required=True, help="Path to training config YAML")
    parser.add_argument(
        "--devices",
        type=int,
        default=None,
        help="Override device count for all stages (default: use training.num_gpus)",
    )
    parser.add_argument(
        "--operator-ckpt",
        help="Optional operator checkpoint to use as teacher for downstream stages",
    )
    args = parser.parse_args()

    cfg = load_config(args.train_config)
    stages_cfg = cfg.get("stages", {}) if isinstance(cfg.get("stages"), dict) else {}

    if stages_cfg.get("operator", {}).get("epochs", 0) > 0:
        print("=" * 60)
        print("Stage 1: Operator Training (Lightning)")
        print("=" * 60)
        run_stage(args.train_config, "operator", args.devices)

    if stages_cfg.get("diff_residual", {}).get("epochs", 0) > 0:
        print("=" * 60)
        print("Stage 2: Diffusion Residual Training (Lightning)")
        print("=" * 60)
        if args.operator_ckpt:
            os.environ["OPERATOR_CKPT"] = args.operator_ckpt
        run_stage(args.train_config, "diff_residual", args.devices)

    if stages_cfg.get("consistency_distill", {}).get("epochs", 0) > 0:
        print("=" * 60)
        print("Stage 3: Consistency Distillation (Lightning)")
        print("=" * 60)
        if args.operator_ckpt:
            os.environ["OPERATOR_CKPT"] = args.operator_ckpt
        run_stage(args.train_config, "consistency_distill", args.devices)


if __name__ == "__main__":
    main()
