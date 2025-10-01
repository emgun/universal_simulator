#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(repo_root, "src"))

import torch

from ups.core.blocks_pdet import PDETransformerConfig
from ups.core.latent_state import LatentState
from ups.inference.rollout_transient import RolloutConfig, rollout_transient
from ups.models.diffusion_residual import DiffusionResidual, DiffusionResidualConfig
from ups.models.latent_operator import LatentOperator, LatentOperatorConfig


def load_config(path: str):
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/inference_transient.yaml")
    parser.add_argument("--mode", choices=["transient", "steady"], default="transient")
    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.mode == "steady":
        print(f"Running steady-state inference with config {args.config} (placeholder)")
        return

    operator_cfg = LatentOperatorConfig(
        latent_dim=cfg["operator"]["latent_dim"],
        pdet=PDETransformerConfig(**cfg["operator"]["pdet"]),
    )
    operator = LatentOperator(operator_cfg)
    corrector = DiffusionResidual(DiffusionResidualConfig(latent_dim=operator_cfg.latent_dim))

    initial = LatentState(z=torch.zeros(1, 16, operator_cfg.latent_dim))
    run_cfg = dict(cfg["run"])
    run_cfg.pop("mode", None)
    rollout_cfg = RolloutConfig(device="cpu", **run_cfg)
    log = rollout_transient(initial_state=initial, operator=operator, corrector=corrector, config=rollout_cfg)
    print(f"mode={args.mode} rollout steps: {len(log.states)} | corrections applied: {sum(log.corrections)}")


if __name__ == "__main__":
    main()
