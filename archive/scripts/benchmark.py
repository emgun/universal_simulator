#!/usr/bin/env python
from __future__ import annotations

"""Benchmark UPS latent operator against baseline models."""

import argparse
import json
from pathlib import Path
from typing import Dict

import sys

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ups.baselines.models import BaselineConfig, build_baseline
from ups.eval.pdebench_runner import evaluate_latent_model, evaluate_latent_operator
from ups.models.diffusion_residual import DiffusionResidual, DiffusionResidualConfig
from ups.models.latent_operator import LatentOperator, LatentOperatorConfig
from ups.core.blocks_pdet import PDETransformerConfig
from ups.utils.monitoring import init_monitoring_session


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_operator(cfg: Dict, path: Path) -> LatentOperator:
    latent_cfg = cfg.get("latent", {})
    dim = latent_cfg.get("dim", 32)
    pdet_cfg = cfg.get("operator", {}).get("pdet", {})
    if not pdet_cfg:
        pdet_cfg = {
            "input_dim": dim,
            "hidden_dim": dim * 2,
            "depths": [1, 1, 1],
            "group_size": max(dim // 2, 4),
            "num_heads": 4,
        }
    operator = LatentOperator(
        LatentOperatorConfig(
            latent_dim=dim,
            pdet=PDETransformerConfig(**pdet_cfg),
            time_embed_dim=dim,
        )
    )
    operator.load_state_dict(torch.load(path, map_location="cpu"))
    return operator


def load_diffusion(cfg: Dict, path: Path | None) -> DiffusionResidual | None:
    if path is None:
        return None
    latent_dim = cfg.get("latent", {}).get("dim", 32)
    model = DiffusionResidual(DiffusionResidualConfig(latent_dim=latent_dim, hidden_dim=latent_dim * 2))
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model


def load_baseline(cfg: Dict, name: str, path: Path | None) -> torch.nn.Module:
    latent_cfg = cfg.get("latent", {})
    baseline_cfg = BaselineConfig(latent_dim=latent_cfg.get("dim", 32), tokens=latent_cfg.get("tokens", 64))
    model = build_baseline(name, baseline_cfg)
    if path and path.exists():
        model.load_state_dict(torch.load(path, map_location="cpu"))
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark UPS latent operator against baselines")
    parser.add_argument("--config", default="configs/train_multi_pde.yaml")
    parser.add_argument("--operator", default="checkpoints/operator.pt")
    parser.add_argument("--diffusion", default=None)
    parser.add_argument("--baseline", choices=["identity", "linear", "mlp"], default="linear")
    parser.add_argument("--baseline-checkpoint", default=None)
    parser.add_argument("--output", default="reports/benchmark.json")
    parser.add_argument("--log-path", default="reports/benchmark_log.jsonl")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tau", type=float, default=0.5)
    args = parser.parse_args()

    cfg = load_config(args.config)

    operator = load_operator(cfg, Path(args.operator))
    diffusion = load_diffusion(cfg, Path(args.diffusion)) if args.diffusion else None

    baseline_ckpt = Path(args.baseline_checkpoint) if args.baseline_checkpoint else None
    baseline_model = load_baseline(cfg, args.baseline, baseline_ckpt)

    operator_report = evaluate_latent_operator(cfg, operator, diffusion=diffusion, tau=args.tau, device=args.device)
    baseline_report = evaluate_latent_model(cfg, baseline_model, device=args.device)

    results = {
        "operator": {
            "metrics": operator_report.metrics,
            "extra": operator_report.extra,
        },
        "baseline": {
            "name": args.baseline,
            "metrics": baseline_report.metrics,
            "extra": baseline_report.extra,
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Benchmark results written to {output_path}")

    session = init_monitoring_session(cfg, component="benchmark", file_path=args.log_path)
    session.log({"stage": "benchmark", "event": "config", "config": cfg})
    session.log({"stage": "benchmark", "results": results, "output": str(output_path)})
    session.finish()


if __name__ == "__main__":
    main()
