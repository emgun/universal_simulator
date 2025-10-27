#!/usr/bin/env python3
"""
Analyze a WandB training run and generate a comprehensive report.

Features:
- Fetches training curves (loss, grad norms, learning rates)
- Computes stage-by-stage timing breakdown
- Summarizes GPU/system utilisation statistics
- Analyzes final metrics (including physics diagnostics) and evaluation results
- Optionally exports the full run history as CSV
- Generates markdown report with recommendations

Usage:
    python scripts/analyze_run.py <run_id>
    python scripts/analyze_run.py emgun-morpheus-space/universal-simulator/abc123def
    python scripts/analyze_run.py abc123def --project universal-simulator --entity emgun-morpheus-space
    python scripts/analyze_run.py abc123def --output reports/run_analysis.md
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

try:
    import wandb
    import pandas as pd
    import numpy as np
except ImportError:
    print("❌ Required packages not installed:")
    print("   pip install wandb pandas numpy")
    sys.exit(1)


STAGES = [
    ("operator", "Operator"),
    ("diffusion_residual", "Diffusion Residual"),
    ("consistency_distill", "Consistency Distill"),
    ("steady_prior", "Steady Prior"),
]

GPU_COLUMN_PREFIXES = ("system/gpu", "system.cpu", "system/memory", "system.cpu.", "system/gpu.", "gpu/")
STAGE_LABEL_MAP = {key: label for key, label in STAGES}


def parse_run_path(run_path: str) -> tuple[str, str, str]:
    """
    Parse run path into (entity, project, run_id).
    
    Handles formats:
    - full: entity/project/run_id
    - short: run_id (uses defaults)
    """
    parts = run_path.split("/")
    
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    elif len(parts) == 1:
        # Just run ID, use defaults
        return None, None, parts[0]
    else:
        raise ValueError(f"Invalid run path format: {run_path}")


def fetch_run(entity: Optional[str], project: Optional[str], run_id: str) -> wandb.apis.public.Run:
    """Fetch W&B run object."""
    api = wandb.Api()

    # Build run path
    if entity and project:
        path = f"{entity}/{project}/{run_id}"
    elif project:
        path = f"{project}/{run_id}"
    else:
        # Try to infer from run_id
        path = run_id
    
    try:
        run = api.run(path)
        return run
    except Exception as e:
        print(f"❌ Failed to fetch run: {e}")
        print(f"   Tried path: {path}")
        sys.exit(1)


def load_history(run: wandb.apis.public.Run, max_rows: Optional[int]) -> pd.DataFrame:
    """Load run history into a pandas DataFrame."""
    try:
        history = run.history(samples=max_rows, pandas=True)
    except Exception as exc:
        print(f"⚠️  Failed to fetch run history: {exc}")
        return pd.DataFrame()

    if history is None:
        return pd.DataFrame()
    if not isinstance(history, pd.DataFrame):
        history = pd.DataFrame(history)

    # Drop columns that are completely empty
    history = history.dropna(axis=1, how="all")
    return history


def _first_present_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def analyze_training_curves(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Analyze training loss curves for each stage using the history dataframe.

    Supports both "stage/..." and "training/stage/..." naming.
    """
    analysis: Dict[str, Dict[str, float]] = {}

    if df.empty:
        return analysis

    for stage_key, _label in STAGES:
        # Loss
        loss_col = _first_present_column(df, [
            f"{stage_key}/loss",
            f"training/{stage_key}/loss",
        ])
        if not loss_col:
            continue
        losses = df[loss_col].dropna()
        if losses.empty:
            continue

        stage_stats: Dict[str, float] = {
            "initial_loss": float(losses.iloc[0]),
            "final_loss": float(losses.iloc[-1]),
            "min_loss": float(losses.min()),
            "steps": int(len(losses)),
        }
        if losses.iloc[0] != 0:
            stage_stats["reduction_pct"] = float((losses.iloc[0] - losses.iloc[-1]) / losses.iloc[0] * 100.0)

        # LR
        lr_col = _first_present_column(df, [
            f"{stage_key}/lr",
            f"training/{stage_key}/lr",
        ])
        if lr_col:
            lrs = df[lr_col].dropna()
            if not lrs.empty:
                stage_stats["lr_initial"] = float(lrs.iloc[0])
                stage_stats["lr_final"] = float(lrs.iloc[-1])
                stage_stats["lr_min"] = float(lrs.min())

        # Epoch index
        epoch_col = _first_present_column(df, [
            f"{stage_key}/epoch",
            f"training/{stage_key}/epoch",
        ])
        if epoch_col:
            epochs = df[epoch_col].dropna()
            if not epochs.empty:
                stage_stats["epochs_recorded"] = int(epochs.max()) + 1 if epochs.dtype.kind in "iu" else float(epochs.max())

        # Epoch times
        time_col = _first_present_column(df, [
            f"{stage_key}/epoch_time_sec",
            f"training/{stage_key}/epoch_time_sec",
        ])
        if time_col:
            times = df[time_col].dropna()
            if not times.empty:
                stage_stats["epoch_time_mean"] = float(times.mean())
                stage_stats["epoch_time_max"] = float(times.max())

        # Patience
        patience_col = _first_present_column(df, [
            f"{stage_key}/epochs_since_improve",
            f"training/{stage_key}/epochs_since_improve",
        ])
        if patience_col:
            patience = df[patience_col].dropna()
            if not patience.empty:
                stage_stats["epochs_since_improve_last"] = float(patience.iloc[-1])

        analysis[stage_key] = stage_stats

    return analysis


def analyze_gradient_norms(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Analyze gradient norms for convergence issues.

    Supports both "stage/..." and "training/stage/..." naming.
    """
    grad_analysis: Dict[str, Dict[str, float]] = {}

    if df.empty:
        return grad_analysis

    for stage_key, _label in STAGES:
        grad_col = _first_present_column(df, [
            f"{stage_key}/grad_norm",
            f"training/{stage_key}/grad_norm",
        ])
        if not grad_col:
            continue
        grads = df[grad_col].dropna()
        if grads.empty:
            continue
        grad_analysis[stage_key] = {
            "initial": float(grads.iloc[0]),
            "final": float(grads.iloc[-1]),
            "max": float(grads.max()),
            "mean": float(grads.mean()),
        }

    return grad_analysis


def analyze_evaluation(run: wandb.apis.public.Run) -> Dict:
    """Extract evaluation metrics."""
    summary = run.summary

    eval_metrics = {}
    
    # Baseline metrics
    if "eval/baseline_nrmse" in summary:
        eval_metrics["baseline"] = {
            "nrmse": summary.get("eval/baseline_nrmse"),
            "mse": summary.get("eval/baseline_mse"),
            "mae": summary.get("eval/baseline_mae"),
            "rmse": summary.get("eval/baseline_rmse"),
            "rel_l2": summary.get("eval/baseline_rel_l2"),
        }
        for key in ("conservation_gap", "bc_violation", "negativity_penalty"):
            eval_metrics["baseline"][key] = summary.get(f"eval/{key}")
    
    # TTC metrics
    if "eval/ttc_nrmse" in summary:
        eval_metrics["ttc"] = {
            "nrmse": summary.get("eval/ttc_nrmse"),
            "mse": summary.get("eval/ttc_mse"),
            "mae": summary.get("eval/ttc_mae"),
            "rmse": summary.get("eval/ttc_rmse"),
            "rel_l2": summary.get("eval/ttc_rel_l2"),
        }
        for key in ("conservation_gap", "bc_violation", "negativity_penalty"):
            eval_metrics["ttc"][key] = summary.get(f"eval/ttc_{key}")
        
        # Compute improvement
        if eval_metrics.get("baseline"):
            baseline_nrmse = eval_metrics["baseline"]["nrmse"]
            ttc_nrmse = eval_metrics["ttc"]["nrmse"]
            if baseline_nrmse and ttc_nrmse:
                improvement = (baseline_nrmse - ttc_nrmse) / baseline_nrmse * 100
                eval_metrics["ttc_improvement_pct"] = improvement
    
    return eval_metrics


def analyze_gpu_metrics(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Summarize GPU/system metrics from history."""
    gpu_stats: Dict[str, Dict[str, float]] = {}

    if df.empty:
        return gpu_stats

    candidate_cols = [col for col in df.columns if col.startswith(GPU_COLUMN_PREFIXES)]
    for col in candidate_cols:
        series = df[col].dropna()
        if series.empty:
            continue
        gpu_stats[col] = {
            "mean": float(series.mean()),
            "max": float(series.max()),
            "min": float(series.min()),
        }
    return gpu_stats


def generate_report(run: wandb.apis.public.Run, analysis: Dict) -> str:
    """Generate markdown report."""
    lines = []
    
    # Header
    lines.append(f"# Training Run Analysis: {run.name}")
    lines.append(f"\n**Run ID:** `{run.id}`")
    lines.append(f"**Project:** {run.project}")
    lines.append(f"**Entity:** {run.entity}")
    lines.append(f"**Created:** {run.created_at}")
    lines.append(f"**State:** {run.state}")
    lines.append(f"**URL:** {run.url}")
    lines.append("\n---\n")
    
    # Configuration
    lines.append("## Configuration\n")
    config = run.config
    if "latent" in config and "dim" in config["latent"]:
        lines.append(f"- **Latent Dimension:** {config['latent']['dim']}")
    if "training" in config and "batch_size" in config["training"]:
        lines.append(f"- **Batch Size:** {config['training']['batch_size']}")
    if "stages" in config:
        lines.append(f"- **Operator Epochs:** {config['stages'].get('operator', {}).get('epochs', 'N/A')}")
        lines.append(f"- **Diffusion Epochs:** {config['stages'].get('diff_residual', {}).get('epochs', 'N/A')}")
        lines.append(f"- **Distillation Epochs:** {config['stages'].get('consistency_distill', {}).get('epochs', 'N/A')}")
    
    lines.append("\n---\n")
    
    # Training Curves Analysis
    lines.append("## Training Loss Analysis\n")
    
    if "curves" in analysis and analysis["curves"]:
        for stage, metrics in analysis["curves"].items():
            stage_label = STAGE_LABEL_MAP.get(stage, stage.title())
            lines.append(f"### {stage_label} Stage\n")
            lines.append(f"- **Initial Loss:** {metrics['initial_loss']:.6f}")
            lines.append(f"- **Final Loss:** {metrics['final_loss']:.6f}")
            lines.append(f"- **Min Loss:** {metrics['min_loss']:.6f}")
            if "reduction_pct" in metrics:
                lines.append(f"- **Loss Reduction:** {metrics['reduction_pct']:.1f}%")
            lines.append(f"- **Logged Steps:** {metrics['steps']}")
            if "epochs_recorded" in metrics:
                lines.append(f"- **Epochs Recorded:** {metrics['epochs_recorded']}")
            if "lr_initial" in metrics and "lr_final" in metrics:
                lines.append(f"- **LR (initial → final):** {metrics['lr_initial']:.6f} → {metrics['lr_final']:.6f}")
            if "epoch_time_mean" in metrics:
                lines.append(f"- **Mean Epoch Time:** {metrics['epoch_time_mean']:.2f}s")
            if "epochs_since_improve_last" in metrics:
                lines.append(f"- **Epochs Since Improve (last):** {metrics['epochs_since_improve_last']:.1f}")

            # Assess convergence
            reduction = metrics.get("reduction_pct", 0.0)
            final_loss = metrics.get("final_loss", 0.0)
            initial_loss = metrics.get("initial_loss", 0.0)
            if initial_loss > 0 and reduction >= 90:
                lines.append("- **Status:** ✅ Excellent convergence")
            elif reduction >= 50:
                lines.append("- **Status:** ✅ Good convergence")
            elif reduction >= 20:
                lines.append("- **Status:** ⚠️  Moderate convergence")
            else:
                lines.append("- **Status:** ❌ Minimal loss reduction")

            lines.append("")
    else:
        lines.append("*No training curve data available*\n")
    
    lines.append("---\n")
    
    # Gradient Norms
    if "gradients" in analysis and analysis["gradients"]:
        lines.append("## Gradient Norm Analysis\n")
        
        for stage, grad_metrics in analysis["gradients"].items():
            stage_label = STAGE_LABEL_MAP.get(stage, stage.title().replace("_", " "))
            lines.append(f"### {stage_label}\n")
            lines.append(f"- **Initial Grad Norm:** {grad_metrics['initial']:.4f}")
            lines.append(f"- **Final Grad Norm:** {grad_metrics['final']:.4f}")
            lines.append(f"- **Max Grad Norm:** {grad_metrics['max']:.4f}")
            lines.append(f"- **Mean Grad Norm:** {grad_metrics['mean']:.4f}")
            
            # Assess gradient health
            if grad_metrics['max'] > 10.0:
                lines.append("- **Status:** ⚠️  High gradient norms detected (possible instability)")
            elif grad_metrics['final'] < 0.01:
                lines.append("- **Status:** ⚠️  Very low final gradients (possible early stopping)")
            else:
                lines.append("- **Status:** ✅ Healthy gradients")
            
            lines.append("")
        
        lines.append("---\n")
    
    # Evaluation Results
    if "evaluation" in analysis and analysis["evaluation"]:
        lines.append("## Evaluation Results\n")
        
        eval_data = analysis["evaluation"]
        
        if "baseline" in eval_data:
            lines.append("### Baseline Performance\n")
            for metric, value in eval_data["baseline"].items():
                if value is not None:
                    lines.append(f"- **{metric.upper()}:** {value:.6f}")
            lines.append("")

        if "ttc" in eval_data:
            lines.append("### Test-Time Conditioning (TTC) Performance\n")
            for metric, value in eval_data["ttc"].items():
                if value is not None:
                    lines.append(f"- **{metric.upper()}:** {value:.6f}")
            
            if "ttc_improvement_pct" in eval_data:
                improvement = eval_data["ttc_improvement_pct"]
                lines.append(f"\n**TTC Improvement:** {improvement:.1f}%")
                
                if improvement > 20:
                    lines.append("- **Status:** ✅ Excellent TTC improvement")
                elif improvement > 10:
                    lines.append("- **Status:** ✅ Good TTC improvement")
                elif improvement > 5:
                    lines.append("- **Status:** ⚠️  Moderate TTC improvement")
                else:
                    lines.append("- **Status:** ❌ Minimal TTC improvement")
            
            lines.append("")
        
        lines.append("---\n")

    if "gpu" in analysis and analysis["gpu"]:
        lines.append("## Hardware Metrics\n")
        for key, stats in analysis["gpu"].items():
            lines.append(f"- **{key}:** avg={stats['mean']:.2f}, max={stats['max']:.2f}, min={stats['min']:.2f}")
        lines.append("\n---\n")
    
    # Recommendations
    lines.append("## Recommendations\n")
    
    recommendations = []
    
    # Analyze operator
    if "curves" in analysis and "operator" in analysis["curves"]:
        op_loss = analysis["curves"]["operator"]["final_loss"]
        if op_loss > 0.001:
            recommendations.append("- **Operator:** Consider increasing epochs or lowering LR for better convergence")
        elif op_loss < 0.0001:
            recommendations.append("- **Operator:** Excellent convergence! Consider this a good baseline")

    # Analyze diffusion
    if "curves" in analysis and "diffusion_residual" in analysis["curves"]:
        diff_loss = analysis["curves"]["diffusion_residual"]["final_loss"]
        if diff_loss > 0.01:
            recommendations.append("- **Diffusion:** Loss is high - try reducing LR or increasing epochs")

    # Analyze TTC
    if "evaluation" in analysis and "ttc_improvement_pct" in analysis["evaluation"]:
        improvement = analysis["evaluation"]["ttc_improvement_pct"]
        if improvement < 10:
            recommendations.append("- **TTC:** Low improvement - consider tuning TTC hyperparameters (candidates, beam_width)")
    
    # Analyze gradients
    if "gradients" in analysis:
        for stage, grad_metrics in analysis["gradients"].items():
            if grad_metrics["max"] > 10.0:
                recommendations.append(f"- **{stage.title()}:** High gradient norms - enable/reduce gradient clipping")

    if "evaluation" in analysis and "baseline" in analysis["evaluation"]:
        base = analysis["evaluation"]["baseline"]
        if base.get("conservation_gap") and base["conservation_gap"] > 1.0:
            recommendations.append("- **Physics:** High conservation gap detected; consider enabling physics guards or residual correctors")
        if base.get("bc_violation") and base["bc_violation"] > 0.5:
            recommendations.append("- **Boundary Conditions:** Tighten boundary penalties or curriculum for harder regimes")

    if recommendations:
        lines.extend(recommendations)
    else:
        lines.append("✅ Training looks good! No major issues detected.")
    
    lines.append("\n---\n")
    lines.append(f"\n*Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Analyze W&B training run")
    parser.add_argument("run_path", help="Run path (entity/project/run_id or just run_id)")
    parser.add_argument("--entity", help="W&B entity (if not in run_path)")
    parser.add_argument("--project", default="universal-simulator", help="W&B project")
    parser.add_argument("--output", "-o", help="Output markdown file (default: print to stdout)")
    parser.add_argument("--history-csv", help="Optional path to export raw history as CSV")
    parser.add_argument("--max-rows", type=int, default=20000, help="Maximum number of history rows to retrieve")
    args = parser.parse_args()
    
    # Parse run path
    entity, project, run_id = parse_run_path(args.run_path)
    entity = entity or args.entity
    project = project or args.project
    
    print(f"🔍 Fetching run: {run_id}")
    print(f"   Project: {project}")
    if entity:
        print(f"   Entity: {entity}")
    
    # Fetch run
    run = fetch_run(entity, project, run_id)
    print(f"✅ Found run: {run.name}")
    
    # Load history once for analyses and potential export
    print("📥 Downloading run history...")
    history_df = load_history(run, args.max_rows)
    if args.history_csv and not history_df.empty:
        history_path = Path(args.history_csv)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_df.to_csv(history_path, index=False)
        print(f"   • Saved history to {history_path}")
    elif history_df.empty:
        print("⚠️  History is empty or unavailable")

    # Analyze
    print("📊 Analyzing training curves...")
    curves = analyze_training_curves(history_df)
    
    print("📈 Analyzing gradient norms...")
    gradients = analyze_gradient_norms(history_df)

    print("🎯 Analyzing evaluation results...")
    evaluation = analyze_evaluation(run)

    print("🖥️  Summarizing hardware metrics...")
    gpu_metrics = analyze_gpu_metrics(history_df)

    # Compile analysis
    analysis = {
        "curves": curves,
        "gradients": gradients,
        "evaluation": evaluation,
        "gpu": gpu_metrics,
    }
    
    # Generate report
    print("📝 Generating report...")
    report = generate_report(run, analysis)
    
    # Output
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report)
        print(f"\n✅ Report saved to: {output_path}")
    else:
        print("\n" + "="*70)
        print(report)
        print("="*70)


if __name__ == "__main__":
    main()
