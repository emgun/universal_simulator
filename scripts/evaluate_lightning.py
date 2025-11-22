#!/usr/bin/env python
from __future__ import annotations

"""Lightning-native evaluation wrapper for latent operator checkpoints."""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger

from ups.data.lightning_datamodule import UPSDataModule
from ups.training.lightning_modules import OperatorLightningModule

from ups.core.blocks_pdet import PDETransformerConfig
from ups.eval.pdebench_runner import evaluate_latent_operator
from ups.eval.reports import MetricReport
from ups.inference.rollout_ttc import TTCConfig, build_reward_model_from_config
from ups.models.diffusion_residual import DiffusionResidual, DiffusionResidualConfig
from ups.models.latent_operator import LatentOperator, LatentOperatorConfig
from ups.models.pure_transformer import PureTransformerConfig
from ups.utils.config_loader import load_config_with_includes
from ups.utils.leaderboard import update_leaderboard
from ups.utils.monitoring import init_monitoring_session
from ups.utils.wandb_context import load_wandb_context_from_file

torch.set_float32_matmul_precision("high")


def _load_config(path: str) -> Dict[str, Any]:
    return load_config_with_includes(path)


def _make_operator(cfg: Dict[str, Any]) -> LatentOperator:
    latent_cfg = cfg.get("latent", {})
    dim = latent_cfg.get("dim", 32)
    operator_cfg = cfg.get("operator", {})
    pdet_cfg = operator_cfg.get("pdet", {})
    if not pdet_cfg:
        pdet_cfg = {
            "input_dim": dim,
            "hidden_dim": dim * 2,
            "depths": [1, 1, 1],
            "group_size": max(dim // 2, 4),
            "num_heads": 4,
        }
    architecture_type = operator_cfg.get("architecture_type", "pdet_unet")
    if architecture_type == "pdet_stack":
        pdet_config = PureTransformerConfig(**pdet_cfg)
    else:
        pdet_config = PDETransformerConfig(**pdet_cfg)
    config = LatentOperatorConfig(
        latent_dim=dim,
        pdet=pdet_config,
        architecture_type=architecture_type,
        time_embed_dim=dim,
    )
    return LatentOperator(config)


def _load_state_dict(model: torch.nn.Module, path: Path, strip_prefixes: Optional[list[str]] = None) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    fixed = {}
    for k, v in state.items():
        new_key = k
        for prefix in strip_prefixes or []:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
        fixed[new_key] = v
    model.load_state_dict(fixed, strict=False)


def _make_diffusion(cfg: Dict[str, Any]) -> DiffusionResidual:
    latent_dim = cfg.get("latent", {}).get("dim", 32)
    hidden_dim = cfg.get("diffusion", {}).get("hidden_dim", latent_dim * 2)
    return DiffusionResidual(DiffusionResidualConfig(latent_dim=latent_dim, hidden_dim=hidden_dim))


def _is_rank_0() -> bool:
    return int(os.environ.get("RANK", 0)) == 0


def _write_json(report: MetricReport, extra: Dict[str, Any], destination: Path) -> None:
    if not _is_rank_0():
        return
    payload = {"metrics": report.metrics, "extra": report.extra}
    if extra:
        payload["details"] = extra
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Lightning-native evaluation of latent operator checkpoints.")
    parser.add_argument("--config", required=True, help="Config YAML")
    parser.add_argument("--operator", required=True, help="Operator checkpoint (Lightning or native)")
    parser.add_argument("--diffusion", help="Optional diffusion residual checkpoint")
    parser.add_argument("--tau", type=float, default=0.5, help="Tau for diffusion correction")
    parser.add_argument("--device", default=None, help="Device override (cpu/cuda)")
    parser.add_argument("--output-prefix", default="reports/evaluation_lightning", help="Prefix for output files")
    parser.add_argument("--app-config", help="Optional monitoring config")
    parser.add_argument("--print-json", action="store_true", help="Print metrics JSON to stdout")
    parser.add_argument(
        "--disable-ttc",
        action="store_true",
        help="Disable TTC during eval (enabled by default per config)",
    )
    parser.add_argument(
        "--no-trainer-eval",
        action="store_true",
        help="Skip Lightning Trainer-based eval (use manual TTC path instead)",
    )
    parser.add_argument("--leaderboard-run-id", help="Append metrics to leaderboard under this run id")
    parser.add_argument("--leaderboard-path", default="reports/leaderboard.csv")
    parser.add_argument("--leaderboard-html", default="reports/leaderboard.html")
    parser.add_argument("--leaderboard-label", help="Label for leaderboard entry")
    parser.add_argument("--leaderboard-tag", action="append", default=[])
    parser.add_argument("--leaderboard-notes", help="Notes for leaderboard")
    parser.add_argument("--leaderboard-wandb", action="store_true")
    parser.add_argument("--leaderboard-wandb-project")
    parser.add_argument("--leaderboard-wandb-entity")
    parser.add_argument("--leaderboard-wandb-run-name")
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # If user passes "cuda" without index, default to cuda:0 to satisfy set_device
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda:0")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    cfg = _load_config(args.config)
    cfg_path = Path(args.config).resolve()
    # Force eval to use test split only to avoid preloading train caches in memory
    if isinstance(cfg.get("data"), dict):
        cfg["data"]["split"] = "test"
    # Avoid preloading entire caches during eval to reduce memory pressure
    if isinstance(cfg.get("training"), dict):
        cfg["training"]["use_preloaded_cache"] = False
    if args.disable_ttc and "ttc" in cfg and isinstance(cfg["ttc"], dict):
        cfg["ttc"]["enabled"] = False

    # Clear any lingering allocations before constructing models/dataloaders
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    monitoring = None
    if args.app_config:
        monitoring = init_monitoring_session(Path(args.app_config))

    if not args.no_trainer_eval:
        training_cfg = cfg.get("training", {}) if isinstance(cfg.get("training"), dict) else {}
        num_gpus = int(training_cfg.get("num_gpus", 1))
        devices = num_gpus if num_gpus > 0 else None
        accelerator = "gpu" if devices else "cpu"
        strategy: str | None = None
        if accelerator == "gpu" and devices and devices > 1:
            strategy = "fsdp" if training_cfg.get("use_fsdp2", False) else "ddp"

        precision = "32-true"
        if training_cfg.get("amp", False):
            amp_dtype = str(training_cfg.get("amp_dtype", "bfloat16")).lower()
            precision = "bf16-mixed" if amp_dtype != "float16" else "16-mixed"

        model = OperatorLightningModule(cfg)
        # Load operator weights into the LightningModule
        _load_state_dict(
            model.operator,
            Path(args.operator),
            strip_prefixes=["module.", "_orig_mod.", "operator."],
        )
        datamodule = UPSDataModule(cfg)
        trainer = Trainer(
            accelerator=accelerator,
            devices=devices,
            strategy=strategy,
            precision=precision,
            logger=CSVLogger("logs", name="eval", version=None),
            enable_checkpointing=False,
            enable_progress_bar=True,
            enable_model_summary=False,
        )
        try:
            results = trainer.test(model, datamodule=datamodule)
        except Exception as e:
            print(f"❌ Evaluation failed on Rank {os.environ.get('RANK', '?')}: {e}")
            import traceback
            traceback.print_exc()
            import sys
            sys.exit(1)

        report = MetricReport(metrics=results[0] if results else {}, extra={"eval_mode": "trainer"})
        dest = Path(args.output_prefix).with_suffix(".json")
        _write_json(report, {}, dest)
        if args.print_json:
            if _is_rank_0():
                print(json.dumps(report.metrics, indent=2))
        return

    # Manual TTC path
    operator = _make_operator(cfg).to(device)
    _load_state_dict(operator, Path(args.operator), strip_prefixes=["module.", "_orig_mod.", "operator."])
    operator.eval()

    diffusion = None
    if args.diffusion:
        diffusion = _make_diffusion(cfg).to(device)
        _load_state_dict(diffusion, Path(args.diffusion), strip_prefixes=["module.", "_orig_mod.", "diffusion_residual.", "diffusion."])
        diffusion.eval()

    reward_model = None
    ttc_runtime_cfg = None
    ttc_cfg = cfg.get("ttc")
    if ttc_cfg and ttc_cfg.get("enabled"):
        reward_model = build_reward_model_from_config(ttc_cfg, cfg.get("latent", {}).get("dim", 32), device).to(device)
        sampler_cfg = ttc_cfg.get("sampler", {})
        tau_range = sampler_cfg.get("tau_range", [0.3, 0.7])
        noise_schedule = sampler_cfg.get("noise_schedule")
        if noise_schedule is not None:
            noise_schedule = [float(value) for value in noise_schedule]
        ttc_runtime_cfg = TTCConfig(
            steps=ttc_cfg.get("steps", 1),
            dt=ttc_cfg.get("dt", cfg.get("training", {}).get("dt", 0.1)),
            candidates=ttc_cfg.get("candidates", 4),
            beam_width=ttc_cfg.get("beam_width", ttc_cfg.get("beam", 1)),
            horizon=ttc_cfg.get("horizon", 1),
            tau_range=(float(tau_range[0]), float(tau_range[1])),
            noise_std=float(sampler_cfg.get("noise_std", 0.0)),
            noise_schedule=noise_schedule,
            residual_threshold=ttc_cfg.get("residual_threshold"),
            max_evaluations=ttc_cfg.get("max_evaluations"),
            early_stop_margin=ttc_cfg.get("early_stop_margin"),
            gamma=float(ttc_cfg.get("gamma", 1.0)),
            device=device,
        )

    report, details = evaluate_latent_operator(
        cfg,
        operator,
        diffusion=diffusion,
        tau=args.tau,
        device=device,
        return_details=True,
        ttc_config=ttc_runtime_cfg,
        reward_model=reward_model,
    )

    output_prefix = Path(args.output_prefix).resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = output_prefix.with_suffix(".json")
    _write_json(report, details or {}, metrics_path)

    if monitoring:
        monitoring.log({"metrics": report.metrics, "extra": report.extra})

    # Try loading wandb context from env if present
    wandb_ctx = None
    context_file = os.environ.get("WANDB_CONTEXT_FILE")
    if context_file:
        wandb_ctx = load_wandb_context_from_file(Path(context_file))
    if wandb_ctx:
        wandb_ctx.log_eval_summary(report.metrics, prefix="eval")
        if report.extra:
            wandb_ctx.update_config(
                {
                    "eval_samples": report.extra.get("samples"),
                    "eval_tau": report.extra.get("tau"),
                    "eval_ttc_enabled": report.extra.get("ttc", False),
                }
            )
        wandb_ctx.save_file(metrics_path)

    # Leaderboard update
    if args.leaderboard_run_id:
        tags = {}
        for tag in args.leaderboard_tag:
            if "=" not in tag:
                raise ValueError(f"Leaderboard tag '{tag}' must be key=value")
            key, value = tag.split("=", 1)
            tags[key] = value
        update_leaderboard(
            metrics_path=metrics_path,
            run_id=args.leaderboard_run_id,
            leaderboard_csv=Path(args.leaderboard_path),
            leaderboard_html=Path(args.leaderboard_html),
            label=args.leaderboard_label,
            config=args.config,
            notes=args.leaderboard_notes,
            tags=tags,
            wandb_log=args.leaderboard_wandb,
            wandb_project=args.leaderboard_wandb_project,
            wandb_entity=args.leaderboard_wandb_entity,
            wandb_run_name=args.leaderboard_wandb_run_name or args.leaderboard_run_id,
        )

    if args.print_json:
        if _is_rank_0():
            print(json.dumps({"metrics": report.metrics, "extra": report.extra, "outputs": {"json": str(metrics_path)}}, indent=2))
    else:
        if _is_rank_0():
            print("Evaluation metrics:")
            for key, value in report.metrics.items():
                print(f"  {key}: {value:.6f}")
            if report.extra:
                print("Extra metadata:")
                for key, value in report.extra.items():
                    print(f"  {key}: {value}")
            print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        import sys
        print(f"[evaluate_lightning] Global exception caught: {e}", file=sys.stderr)
        traceback.print_exc()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        sys.exit(1)
