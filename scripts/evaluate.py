#!/usr/bin/env python
from __future__ import annotations

"""Evaluate latent operator checkpoints on PDEBench datasets."""

import argparse
import json
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
from ups.utils.monitoring import init_monitoring_session

# Use spawn to allow CUDA in DataLoader workers during evaluation
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def make_operator(cfg: Dict[str, Any]) -> LatentOperator:
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
    config = LatentOperatorConfig(
        latent_dim=dim,
        pdet=PDETransformerConfig(**pdet_cfg),
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
    args = parser.parse_args()

    cfg = load_config(args.config)

    operator = make_operator(cfg)
    operator.load_state_dict(torch.load(args.operator, map_location="cpu"))

    diffusion_model = None
    if args.diffusion:
        diffusion_model = make_diffusion(cfg)
        diffusion_model.load_state_dict(torch.load(args.diffusion, map_location="cpu"))

    reward_model = None
    ttc_runtime_cfg = None
    ttc_cfg = cfg.get("ttc")
    if ttc_cfg and ttc_cfg.get("enabled"):
        device = torch.device(args.device)
        reward_model = build_reward_model_from_config(ttc_cfg, cfg.get("latent", {}).get("dim", 32), device).to(device)
        sampler_cfg = ttc_cfg.get("sampler", {})
        tau_range = sampler_cfg.get("tau_range", [0.3, 0.7])
        ttc_runtime_cfg = TTCConfig(
            steps=ttc_cfg.get("steps", 1),
            dt=ttc_cfg.get("dt", cfg.get("training", {}).get("dt", 0.1)),
            candidates=ttc_cfg.get("candidates", 4),
            beam_width=ttc_cfg.get("beam_width", ttc_cfg.get("beam", 1)),
            horizon=ttc_cfg.get("horizon", 1),
            tau_range=(float(tau_range[0]), float(tau_range[1])),
            noise_std=float(sampler_cfg.get("noise_std", 0.0)),
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
    
    # Upload all report files to W&B
    try:
        import wandb
        if wandb.run is not None:
            for output_type, output_path in outputs.items():
                wandb.save(str(output_path), base_path=str(output_path.parent.parent))
                print(f"Uploaded {output_type} report to W&B: {output_path.name}")
    except Exception as e:
        print(f"Note: Could not upload reports to W&B: {e}")

    session = init_monitoring_session(cfg, component="evaluation", file_path=args.log_path)
    
    # Log metrics with eval/ prefix for better organization
    eval_metrics = {
        f"eval/{k}": v for k, v in report.metrics.items()
    }
    session.log(eval_metrics)
    
    # Log extra info
    if report.extra:
        eval_extra = {f"eval/{k}": v for k, v in report.extra.items()}
        session.log(eval_extra)
    
    # Log images with eval/ prefix
    if "plot_mse" in outputs:
        session.log_image("eval/mse_histogram", outputs["plot_mse"])
    if "plot_mae" in outputs:
        session.log_image("eval/mae_histogram", outputs["plot_mae"])
    if "plot_latent_heatmap" in outputs:
        session.log_image("eval/latent_heatmap", outputs["plot_latent_heatmap"])
    if "plot_latent_spectrum" in outputs:
        session.log_image("eval/latent_spectrum", outputs["plot_latent_spectrum"])

    if session.run is not None:
        try:  # pragma: no cover
            session.run.summary.update({f"eval/{k}": v for k, v in report.metrics.items()})
            if report.extra:
                session.run.summary.update({f"eval_extra/{k}": v for k, v in report.extra.items()})
        except Exception:
            pass

    session.finish()

    _print_report(report, outputs, args.print_json)


if __name__ == "__main__":
    main()
