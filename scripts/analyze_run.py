#!/usr/bin/env python3
"""
Analyze a WandB training run and generate a comprehensive report.

Features:
- Fetches training curves (loss, grad norms, learning rates)
- Computes stage-by-stage timing breakdown
- Analyzes final metrics and evaluation results
- Generates markdown report with plots and recommendations

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


def analyze_training_curves(run: wandb.apis.public.Run) -> Dict:
    """Analyze training loss curves for each stage."""
    history = run.scan_history()
    df = pd.DataFrame(history)
    
    analysis = {}
    
    # Operator stage
    if "operator/loss" in df.columns:
        op_losses = df["operator/loss"].dropna()
        if len(op_losses) > 0:
            analysis["operator"] = {
                "initial_loss": float(op_losses.iloc[0]),
                "final_loss": float(op_losses.iloc[-1]),
                "min_loss": float(op_losses.min()),
                "reduction": float((op_losses.iloc[0] - op_losses.iloc[-1]) / op_losses.iloc[0] * 100),
                "steps": len(op_losses),
            }
    
    # Diffusion stage
    if "diffusion_residual/loss" in df.columns:
        diff_losses = df["diffusion_residual/loss"].dropna()
        if len(diff_losses) > 0:
            analysis["diffusion"] = {
                "initial_loss": float(diff_losses.iloc[0]),
                "final_loss": float(diff_losses.iloc[-1]),
                "min_loss": float(diff_losses.min()),
                "reduction": float((diff_losses.iloc[0] - diff_losses.iloc[-1]) / diff_losses.iloc[0] * 100),
                "steps": len(diff_losses),
            }
    
    # Consistency distillation
    if "consistency_distill/loss" in df.columns:
        cons_losses = df["consistency_distill/loss"].dropna()
        if len(cons_losses) > 0:
            analysis["consistency"] = {
                "initial_loss": float(cons_losses.iloc[0]),
                "final_loss": float(cons_losses.iloc[-1]),
                "min_loss": float(cons_losses.min()),
                "steps": len(cons_losses),
            }
    
    return analysis


def analyze_gradient_norms(run: wandb.apis.public.Run) -> Dict:
    """Analyze gradient norms for convergence issues."""
    history = run.scan_history()
    df = pd.DataFrame(history)
    
    grad_analysis = {}
    
    for stage in ["operator", "diffusion_residual", "consistency_distill"]:
        grad_col = f"{stage}/grad_norm"
        if grad_col in df.columns:
            grads = df[grad_col].dropna()
            if len(grads) > 0:
                grad_analysis[stage] = {
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
        }
    
    # TTC metrics
    if "eval/ttc_nrmse" in summary:
        eval_metrics["ttc"] = {
            "nrmse": summary.get("eval/ttc_nrmse"),
            "mse": summary.get("eval/ttc_mse"),
            "mae": summary.get("eval/ttc_mae"),
            "rmse": summary.get("eval/ttc_rmse"),
        }
        
        # Compute improvement
        if eval_metrics.get("baseline"):
            baseline_nrmse = eval_metrics["baseline"]["nrmse"]
            ttc_nrmse = eval_metrics["ttc"]["nrmse"]
            if baseline_nrmse and ttc_nrmse:
                improvement = (baseline_nrmse - ttc_nrmse) / baseline_nrmse * 100
                eval_metrics["ttc_improvement_pct"] = improvement
    
    return eval_metrics


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
            lines.append(f"### {stage.title()} Stage\n")
            lines.append(f"- **Initial Loss:** {metrics['initial_loss']:.6f}")
            lines.append(f"- **Final Loss:** {metrics['final_loss']:.6f}")
            lines.append(f"- **Min Loss:** {metrics['min_loss']:.6f}")
            if "reduction" in metrics:
                lines.append(f"- **Loss Reduction:** {metrics['reduction']:.1f}%")
            lines.append(f"- **Training Steps:** {metrics['steps']}")
            
            # Assess convergence
            if metrics['final_loss'] < metrics['initial_loss'] * 0.1:
                lines.append("- **Status:** ✅ Good convergence")
            elif metrics['final_loss'] < metrics['initial_loss'] * 0.5:
                lines.append("- **Status:** ⚠️  Moderate convergence")
            else:
                lines.append("- **Status:** ❌ Poor convergence")
            
            lines.append("")
    else:
        lines.append("*No training curve data available*\n")
    
    lines.append("---\n")
    
    # Gradient Norms
    if "gradients" in analysis and analysis["gradients"]:
        lines.append("## Gradient Norm Analysis\n")
        
        for stage, grad_metrics in analysis["gradients"].items():
            lines.append(f"### {stage.title().replace('_', ' ')}\n")
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
    if "curves" in analysis and "diffusion" in analysis["curves"]:
        diff_loss = analysis["curves"]["diffusion"]["final_loss"]
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
    
    # Analyze
    print("📊 Analyzing training curves...")
    curves = analyze_training_curves(run)
    
    print("📈 Analyzing gradient norms...")
    gradients = analyze_gradient_norms(run)
    
    print("🎯 Analyzing evaluation results...")
    evaluation = analyze_evaluation(run)
    
    # Compile analysis
    analysis = {
        "curves": curves,
        "gradients": gradients,
        "evaluation": evaluation,
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

