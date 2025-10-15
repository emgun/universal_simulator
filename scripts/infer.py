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
from ups.inference.rollout_ttc import TTCConfig, build_reward_model_from_config, ttc_rollout
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
    ttc_cfg = cfg.get("ttc")
    if ttc_cfg and ttc_cfg.get("enabled"):
        device = torch.device(rollout_cfg.device)
        reward_model = build_reward_model_from_config(ttc_cfg, operator_cfg.latent_dim, device).to(device)
        sampler_cfg = ttc_cfg.get("sampler", {})
        tau_range = sampler_cfg.get("tau_range", [0.3, 0.7])
        noise_schedule = sampler_cfg.get("noise_schedule")
        if noise_schedule is not None:
            noise_schedule = [float(value) for value in noise_schedule]
        ttc_runtime = TTCConfig(
            steps=rollout_cfg.steps,
            dt=ttc_cfg.get("dt", rollout_cfg.dt),
            candidates=ttc_cfg.get("candidates", 4),
            tau_range=(float(tau_range[0]), float(tau_range[1])),
            noise_std=float(sampler_cfg.get("noise_std", 0.0)),
            noise_schedule=noise_schedule,
            residual_threshold=ttc_cfg.get("residual_threshold"),
            device=device,
        )
        use_corrector = ttc_cfg.get("use_corrector", True)
        rollout_log, step_logs = ttc_rollout(
            initial_state=initial,
            operator=operator,
            reward_model=reward_model,
            config=ttc_runtime,
            corrector=corrector if use_corrector else None,
        )
        print(
            f"mode={args.mode} TTC rollout steps: {len(rollout_log.states)} | best rewards per step: {[max(log.rewards) for log in step_logs]}"
        )
    else:
        rollout_log = rollout_transient(initial_state=initial, operator=operator, corrector=corrector, config=rollout_cfg)
        print(
            f"mode={args.mode} rollout steps: {len(rollout_log.states)} | corrections applied: {sum(rollout_log.corrections)}"
        )


if __name__ == "__main__":
    main()
