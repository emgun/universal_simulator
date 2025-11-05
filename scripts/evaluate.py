#!/usr/bin/env python
from __future__ import annotations

"""Evaluate latent operator checkpoints on PDEBench datasets."""

import argparse
import json
import os
import hashlib
from pathlib import Path
from typing import Any, Dict

import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ups.core.blocks_pdet import PDETransformerConfig
from ups.eval.pdebench_runner import evaluate_latent_operator
from ups.eval.reports import MetricReport
from ups.inference.rollout_ttc import TTCConfig, build_reward_model_from_config
from ups.models.diffusion_residual import DiffusionResidual, DiffusionResidualConfig
from ups.models.latent_operator import LatentOperator, LatentOperatorConfig
from ups.utils.monitoring import MonitoringSession, init_monitoring_session
from ups.utils.leaderboard import update_leaderboard

# Use spawn to allow CUDA in DataLoader workers during evaluation
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass


def _load_state_dict_compat(model: torch.nn.Module, ckpt_path: str, *, prefix_to_strip: str = "_orig_mod.") -> None:
    """Load a checkpoint while stripping an optional prefix from keys (e.g., from torch.compile()).

    This makes loading robust across compiled/non-compiled training runs.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        raise RuntimeError(f"Unsupported checkpoint format at {ckpt_path}: {type(ckpt)}")

    if prefix_to_strip:
        fixed = {}
        for k, v in state_dict.items():
            if k.startswith(prefix_to_strip):
                fixed[k[len(prefix_to_strip) :]] = v
            else:
                fixed[k] = v
        state_dict = fixed

    model.load_state_dict(state_dict)


def _extract_arch_fingerprint(cfg: Dict[str, Any]) -> Dict[str, Any]:
    latent = cfg.get("latent", {}) if isinstance(cfg.get("latent"), dict) else {}
    operator_pdet = (
        cfg.get("operator", {}).get("pdet", {})
        if isinstance(cfg.get("operator"), dict)
        else {}
    )
    diffusion_cfg = cfg.get("diffusion", {}) if isinstance(cfg.get("diffusion"), dict) else {}
    return {
        "latent_dim": latent.get("dim"),
        "latent_tokens": latent.get("tokens"),
        "operator_hidden_dim": operator_pdet.get("hidden_dim"),
        "operator_num_heads": operator_pdet.get("num_heads"),
        "operator_depths": operator_pdet.get("depths"),
        "diffusion_hidden_dim": diffusion_cfg.get("hidden_dim"),
    }


def _verify_checkpoint_metadata(metadata_path: Path, cfg: Dict[str, Any], cfg_path: Path) -> None:
    if not metadata_path.exists():
        print(f"⚠️  Warning: checkpoint metadata not found at {metadata_path}; skipping architecture check")
        return
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Failed to parse checkpoint metadata ({metadata_path}): {exc}")

    arch_expected = metadata.get("arch")
    arch_actual = _extract_arch_fingerprint(cfg)
    if arch_expected:
        mismatches = {
            key: (arch_expected.get(key), arch_actual.get(key))
            for key in arch_actual.keys()
            if arch_expected.get(key) != arch_actual.get(key)
        }
        if mismatches:
            details = ", ".join(
                f"{k}: checkpoint={v[0]!r}, config={v[1]!r}"
                for k, v in mismatches.items()
            )
            raise RuntimeError(
                f"Checkpoint architecture mismatch detected ({details}). "
                "Retrain with matching architecture before evaluation."
            )

    if not metadata.get("trained", True):
        raise RuntimeError(
            "Checkpoint metadata indicates training did not finish successfully. Retrain before evaluation."
        )

    expected_hash = metadata.get("config_hash")
    expected_config_path = metadata.get("config_path")
    if expected_hash and expected_config_path:
        expected_name = Path(expected_config_path).name
        if cfg_path.name == expected_name:
            actual_hash = hashlib.sha256(cfg_path.read_bytes()).hexdigest()
            if actual_hash != expected_hash:
                raise RuntimeError(
                    "Evaluation config does not match the training config recorded in checkpoint metadata. "
                    "Retrain or regenerate checkpoints with the current config."
                )


def load_config(path: str) -> Dict[str, Any]:
    """Load config with support for include directives."""
    try:
        from ups.utils.config_loader import load_config_with_includes
        return load_config_with_includes(path)
    except ImportError:
        # Fallback to basic loading if config_loader not available
        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}


def make_operator(cfg: Dict[str, Any]) -> LatentOperator:
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

    # Detect architecture type and use appropriate config class
    architecture_type = operator_cfg.get("architecture_type", "pdet_unet")

    if architecture_type == "pdet_stack":
        # Pure stacked transformer
        from ups.models.pure_transformer import PureTransformerConfig
        pdet_config = PureTransformerConfig(**pdet_cfg)
    else:
        # Default: U-shaped transformer
        pdet_config = PDETransformerConfig(**pdet_cfg)

    config = LatentOperatorConfig(
        latent_dim=dim,
        pdet=pdet_config,
        time_embed_dim=dim,
    )
    return LatentOperator(config)


def make_diffusion(cfg: Dict[str, Any]) -> DiffusionResidual:
    latent_dim = cfg.get("latent", {}).get("dim", 32)
    hidden_dim = cfg.get("diffusion", {}).get("hidden_dim", latent_dim * 2)
    return DiffusionResidual(DiffusionResidualConfig(latent_dim=latent_dim, hidden_dim=hidden_dim))
def _write_outputs(report: MetricReport, prefix: Path, cfg: Dict[str, Any], details: Dict[str, Any]) -> Dict[str, Path]:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}

    config_path = prefix.parent / f"{prefix.name}.config.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    paths["config"] = config_path

    json_path = prefix.with_suffix(".json")
    MetricReport(metrics=report.metrics, extra=report.extra).to_json(json_path)
    paths["json"] = json_path

    csv_path = prefix.with_suffix(".csv")
    headers = list(report.metrics.keys())
    values = [report.metrics[k] for k in headers]
    extras_headers = []
    extras_values = []
    if report.extra:
        extras_headers = [f"extra_{k}" for k in report.extra.keys()]
        extras_values = [report.extra[k] for k in report.extra.keys()]
    with csv_path.open("w", encoding="utf-8") as fh:
        fh.write(",".join(headers + extras_headers) + "\n")
        row = ",".join(str(v) for v in values + extras_values)
        fh.write(row + "\n")
    paths["csv"] = csv_path

    ttc_logs = details.get("ttc_step_logs") if details else None
    if ttc_logs:
        logs_path = prefix.parent / f"{prefix.name}_ttc_step_logs.json"
        logs_path.write_text(json.dumps(ttc_logs, indent=2), encoding="utf-8")
        paths["ttc_logs"] = logs_path
        steps = list(range(len(ttc_logs)))
        best_totals = [max(entry["totals"]) if entry["totals"] else None for entry in ttc_logs]
        chosen_totals = [entry["totals"][entry["chosen"]] if entry["totals"] else None for entry in ttc_logs]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(steps, best_totals, label="Best total reward", marker="o")
        ax.plot(steps, chosen_totals, label="Chosen total reward", marker="x")
        ax.set_xlabel("TTC step")
        ax.set_ylabel("Reward")
        ax.set_title("TTC reward trajectory")
        ax.legend()
        fig.tight_layout()
        reward_plot_path = prefix.parent / f"{prefix.name}_ttc_rewards.png"
        fig.savefig(reward_plot_path, dpi=150)
        plt.close(fig)
        paths["plot_ttc_rewards"] = reward_plot_path

    html_path = prefix.with_suffix(".html")
    metrics_rows = "\n".join(
        f"        <tr><td>{key}</td><td>{value:.6g}</td></tr>" for key, value in report.metrics.items()
    )
    extras_rows = ""
    if report.extra:
        extras_rows = "\n".join(
            f"        <tr><td>{key}</td><td>{value}</td></tr>" for key, value in report.extra.items()
        )
    top_rows = ""
    if details.get("per_sample_mse"):
        per_sample = list(enumerate(details["per_sample_mse"]))
        top = sorted(per_sample, key=lambda x: x[1], reverse=True)[:5]
        top_rows = "\n".join(
            f"        <tr><td>{idx}</td><td>{value:.6g}</td></tr>" for idx, value in top
        )

    ttc_rows = ""
    if ttc_logs:
        ttc_rows = "\n".join(
            f"        <tr><td>{entry['step']}</td><td>{entry['chosen']}</td><td>{max(entry['totals']):.6g}</td><td>{entry['totals'][entry['chosen']]:.6g}</td></tr>"
            for entry in ttc_logs
        )

    html = f"""
<html>
  <head>
    <title>Latent Operator Evaluation</title>
    <style>
      body {{ font-family: Arial, sans-serif; margin: 1.5rem; }}
      table {{ border-collapse: collapse; min-width: 300px; }}
      th, td {{ border: 1px solid #ccc; padding: 0.5rem 0.75rem; text-align: left; }}
      th {{ background-color: #f0f0f0; }}
      h2 {{ margin-top: 2rem; }}
    </style>
  </head>
  <body>
    <h1>Latent Operator Evaluation</h1>
    <h2>Metrics</h2>
    <table>
      <thead><tr><th>Metric</th><th>Value</th></tr></thead>
      <tbody>
{metrics_rows}
      </tbody>
    </table>
    <h2>Extra Metadata</h2>
    <table>
      <thead><tr><th>Key</th><th>Value</th></tr></thead>
      <tbody>
{extras_rows or '        <tr><td colspan="2">(none)</td></tr>'}
      </tbody>
    </table>
    <h2>Top Per-sample MSE</h2>
    <table>
      <thead><tr><th>Sample Index</th><th>MSE</th></tr></thead>
      <tbody>
{top_rows or '        <tr><td colspan="2">(not collected)</td></tr>'}
      </tbody>
    </table>
    <h2>TTC Step Summary</h2>
    <table>
      <thead><tr><th>Step</th><th>Chosen idx</th><th>Best total</th><th>Chosen total</th></tr></thead>
      <tbody>
{ttc_rows or '        <tr><td colspan="4">(not enabled)</td></tr>'}
      </tbody>
    </table>
  </body>
</html>
"""
    html_path.write_text(html, encoding="utf-8")
    paths["html"] = html_path

    mse_hist_path = None
    mae_hist_path = None
    if details.get("per_sample_mse"):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(details["per_sample_mse"], bins=30, color="#0EA5E9", edgecolor="black")
        ax.set_xlabel("Per-sample MSE")
        ax.set_ylabel("Count")
        ax.set_title("Per-sample MSE distribution")
        fig.tight_layout()
        mse_hist_path = prefix.parent / f"{prefix.name}_mse_hist.png"
        fig.savefig(mse_hist_path, dpi=150)
        plt.close(fig)
        paths["plot_mse"] = mse_hist_path

    if details.get("per_sample_mae"):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(details["per_sample_mae"], bins=30, color="#34D399", edgecolor="black")
        ax.set_xlabel("Per-sample MAE")
        ax.set_ylabel("Count")
        ax.set_title("Per-sample MAE distribution")
        fig.tight_layout()
        mae_hist_path = prefix.parent / f"{prefix.name}_mae_hist.png"
        fig.savefig(mae_hist_path, dpi=150)
        plt.close(fig)
        paths["plot_mae"] = mae_hist_path

    fig, ax = plt.subplots(figsize=(6, 4))
    metric_items = list(report.metrics.items())
    if metric_items:
        labels, values = zip(*metric_items)
        bars = ax.bar(labels, values, color="#4F46E5")
        ax.set_ylabel("Value")
        ax.set_title("Evaluation Metrics")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3g}", ha="center", va="bottom")
        fig.tight_layout()
        plot_path = prefix.parent / f"{prefix.name}_metrics.png"
        fig.savefig(plot_path, dpi=150)
        paths["plot_metrics"] = plot_path
    plt.close(fig)

    if "preview_predicted" in details and "preview_target" in details:
        pred = np.asarray(details["preview_predicted"], dtype=np.float32)
        target = np.asarray(details["preview_target"], dtype=np.float32)
        preview_path = prefix.parent / f"{prefix.name}_preview.npz"
        np.savez_compressed(preview_path, predicted=pred, target=target)
        paths["preview_npz"] = preview_path

        diff = pred - target
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
        for ax, data, title in zip(
            axes,
            (target, pred, diff),
            ("Target Latent", "Predicted Latent", "Prediction Error"),
        ):
            im = ax.imshow(data.T, aspect="auto", origin="lower", cmap="coolwarm")
            ax.set_title(title)
            ax.set_xlabel("Token")
            ax.set_ylabel("Dim")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        heatmap_path = prefix.parent / f"{prefix.name}_latent_heatmap.png"
        fig.savefig(heatmap_path, dpi=150)
        plt.close(fig)
        paths["plot_latent_heatmap"] = heatmap_path

        def _spectrum(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            fft = np.fft.rfft(arr, axis=0)
            power = np.abs(fft).mean(axis=1)
            freqs = np.fft.rfftfreq(arr.shape[0])
            return freqs, power

        freq_target, spec_target = _spectrum(target)
        _, spec_pred = _spectrum(pred)
        _, spec_err = _spectrum(diff)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(freq_target, spec_target, label="Target", linewidth=2)
        ax.plot(freq_target, spec_pred, label="Predicted", linewidth=2)
        ax.plot(freq_target, spec_err, label="Error", linewidth=2, linestyle="--")
        ax.set_xlabel("Frequency (normalized)")
        ax.set_ylabel("Mean power")
        ax.set_title("Latent Spectrum Comparison")
        ax.legend()
        fig.tight_layout()
        spectrum_path = prefix.parent / f"{prefix.name}_latent_spectrum.png"
        fig.savefig(spectrum_path, dpi=150)
        plt.close(fig)
        paths["plot_latent_spectrum"] = spectrum_path

    return paths


def _print_report(report: MetricReport, paths: Dict[str, Path], as_json: bool) -> None:
    if as_json:
        print(json.dumps({"metrics": report.metrics, "extra": report.extra, "outputs": {k: str(v) for k, v in paths.items()}}, indent=2))
        return

    print("Evaluation metrics:")
    for key, value in report.metrics.items():
        print(f"  {key}: {value:.6f}")
    if report.extra:
        print("Extra metadata:")
        for key, value in report.extra.items():
            print(f"  {key}: {value}")
    print("Saved outputs:")
    for kind, path in paths.items():
        print(f"  {kind}: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate latent operator checkpoints on PDEBench data")
    parser.add_argument("--config", default="configs/train_multi_pde.yaml", help="Config file describing data/latent setup")
    parser.add_argument("--operator", required=True, help="Path to operator checkpoint")
    parser.add_argument("--diffusion", help="Optional diffusion residual checkpoint")
    parser.add_argument("--tau", type=float, default=0.5, help="Tau value used when applying diffusion residual")
    parser.add_argument("--device", default="cpu", help="Device for evaluation")
    parser.add_argument("--output-prefix", default="reports/evaluation", help="Prefix (without extension) for saved reports")
    parser.add_argument("--log-path", default="reports/eval_log.jsonl", help="Where to append evaluation logs")
    parser.add_argument("--print-json", action="store_true", help="Print metrics and file paths as JSON")
    parser.add_argument("--leaderboard-run-id", help="If provided, append metrics to leaderboard")
    parser.add_argument("--leaderboard-path", default="reports/leaderboard.csv", help="Leaderboard CSV path")
    parser.add_argument("--leaderboard-html", default="reports/leaderboard.html", help="Leaderboard HTML path")
    parser.add_argument("--leaderboard-label", help="Label to record in leaderboard (e.g., small_eval)")
    parser.add_argument("--leaderboard-tag", action="append", default=[], help="Additional key=value pairs to record (may repeat)")
    parser.add_argument("--leaderboard-notes", help="Optional notes to attach to leaderboard row")
    parser.add_argument("--leaderboard-wandb", action="store_true", help="Also log leaderboard row to Weights & Biases")
    parser.add_argument("--leaderboard-wandb-project", help="W&B project for leaderboard logging")
    parser.add_argument("--leaderboard-wandb-entity", help="W&B entity for leaderboard logging")
    parser.add_argument("--leaderboard-wandb-run-name", help="Optional W&B run name override")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg_path = Path(args.config).resolve()

    metadata_path = Path(args.operator).resolve().parent / "metadata.json"
    _verify_checkpoint_metadata(metadata_path, cfg, cfg_path)

    operator = make_operator(cfg)
    _load_state_dict_compat(operator, args.operator)

    diffusion_model = None
    if args.diffusion:
        diffusion_model = make_diffusion(cfg)
        _load_state_dict_compat(diffusion_model, args.diffusion)

    reward_model = None
    ttc_runtime_cfg = None
    ttc_cfg = cfg.get("ttc")
    if ttc_cfg and ttc_cfg.get("enabled"):
        device = torch.device(args.device)
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

    result = evaluate_latent_operator(
        cfg,
        operator,
        diffusion=diffusion_model,
        tau=args.tau,
        device=args.device,
        return_details=True,
        ttc_config=ttc_runtime_cfg,
        reward_model=reward_model,
    )
    report, details = result  # type: ignore[misc]

    output_prefix = Path(args.output_prefix)
    outputs = _write_outputs(report, output_prefix, cfg, details)
    metrics_json = outputs.get("json")
    if args.leaderboard_run_id and metrics_json is not None:
        tags = {}
        for tag in args.leaderboard_tag:
            if "=" not in tag:
                raise ValueError(f"Leaderboard tag '{tag}' must be formatted as key=value")
            key, value = tag.split("=", 1)
            tags[key] = value
        update_leaderboard(
            metrics_path=metrics_json,
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

    # Load or create WandB context (clean way!)
    wandb_ctx = None

    # Try to load from environment (subprocess mode - orchestrator passed context)
    from ups.utils.wandb_context import load_wandb_context_from_file

    context_file = os.environ.get("WANDB_CONTEXT_FILE")
    if context_file:
        wandb_ctx = load_wandb_context_from_file(Path(context_file))
        if wandb_ctx:
            print(f"✓ Loaded WandB context from {context_file}")

    # If no context from orchestrator, evaluation runs standalone - don't create new run
    # (The orchestrator will handle WandB, or user can run eval separately)

    # Log evaluation metrics to WandB summary (SCALARS, not time series!)
    if wandb_ctx:
        # Separate metrics into categories for better organization
        basic_metrics = {}
        physics_metrics = {}

        for key, value in report.metrics.items():
            if key in ["conservation_gap", "bc_violation", "negativity_penalty"]:
                physics_metrics[key] = value
            else:
                basic_metrics[key] = value

        # 1. Log basic metrics to SUMMARY (single values - no line charts!)
        wandb_ctx.log_eval_summary(basic_metrics, prefix="eval")
        print(f"✓ Logged {len(basic_metrics)} evaluation metrics to WandB summary")

        # 2. Log physics metrics to SUMMARY with separate prefix
        if physics_metrics:
            wandb_ctx.log_eval_summary(physics_metrics, prefix="eval/physics")
            print(f"✓ Logged {len(physics_metrics)} physics metrics to WandB summary")

        # 3. Log metadata to CONFIG (not as metrics!)
        if report.extra:
            wandb_ctx.update_config({
                "eval_samples": report.extra.get("samples"),
                "eval_tau": report.extra.get("tau"),
                "eval_ttc_enabled": report.extra.get("ttc", False),
            })
            print("✓ Logged evaluation metadata to WandB config")

        # 4. Create comprehensive metrics tables
        # Combined summary table (most prominent)
        all_rows = []
        if basic_metrics:
            for key, value in sorted(basic_metrics.items()):
                all_rows.append(["Accuracy", key, f"{value:.6f}"])
        if physics_metrics:
            for key, value in sorted(physics_metrics.items()):
                all_rows.append(["Physics", key, f"{value:.6f}"])

        if all_rows:
            wandb_ctx.log_table(
                "Evaluation Summary",
                columns=["Category", "Metric", "Value"],
                data=all_rows
            )
            print(f"✓ Created evaluation summary table with {len(all_rows)} metrics")

        # Separate category tables for detailed view
        if basic_metrics:
            basic_rows = [[key, f"{value:.6f}"] for key, value in sorted(basic_metrics.items())]
            wandb_ctx.log_table(
                "eval/accuracy_metrics",
                columns=["Metric", "Value"],
                data=basic_rows
            )

        if physics_metrics:
            physics_rows = [[key, f"{value:.6f}"] for key, value in sorted(physics_metrics.items())]
            wandb_ctx.log_table(
                "eval/physics_diagnostics",
                columns=["Physics Check", "Value"],
                data=physics_rows
            )
            print(f"✓ Created {len(physics_metrics)} physics diagnostic entries")

        # 4. Log images
        if "plot_mse" in outputs:
            wandb_ctx.log_image("eval/mse_histogram", outputs["plot_mse"])
        if "plot_mae" in outputs:
            wandb_ctx.log_image("eval/mae_histogram", outputs["plot_mae"])
        if "plot_latent_heatmap" in outputs:
            wandb_ctx.log_image("eval/latent_heatmap", outputs["plot_latent_heatmap"])
        if "plot_latent_spectrum" in outputs:
            wandb_ctx.log_image("eval/latent_spectrum", outputs["plot_latent_spectrum"])

        # 5. Save all output files
        for output_path in outputs.values():
            wandb_ctx.save_file(output_path)

        print("✓ Uploaded evaluation outputs to WandB")

    # Note: Don't call wandb_ctx.finish() - orchestrator owns the run!

    _print_report(report, outputs, args.print_json)


if __name__ == "__main__":
    main()
