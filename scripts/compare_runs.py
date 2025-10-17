#!/usr/bin/env python3
"""
Compare multiple WandB training runs side-by-side.

Features:
- Config diff highlighting
- Loss/metric table including physics diagnostics
- Identifies which changes improved/degraded performance

Usage:
    python scripts/compare_runs.py run1_id run2_id
    python scripts/compare_runs.py abc123 def456 ghi789 --output reports/comparison.md
    python scripts/compare_runs.py abc123 def456 --project universal-simulator --entity emgun-morpheus-space
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List
from datetime import datetime

try:
    import wandb
    import pandas as pd
    import numpy as np
except ImportError:
    print("❌ Required packages not installed:")
    print("   pip install wandb pandas numpy")
    sys.exit(1)


def fetch_runs(run_ids: List[str], entity: str, project: str) -> List[wandb.apis.public.Run]:
    """Fetch multiple runs."""
    api = wandb.Api()
    runs = []
    
    for run_id in run_ids:
        try:
            if entity and project:
                path = f"{entity}/{project}/{run_id}"
            elif project:
                path = f"{project}/{run_id}"
            else:
                path = run_id
            
            run = api.run(path)
            runs.append(run)
            print(f"✅ Fetched: {run.name} ({run_id})")
        except Exception as e:
            print(f"❌ Failed to fetch {run_id}: {e}")
            sys.exit(1)
    
    return runs


def extract_metrics(run: wandb.apis.public.Run) -> Dict:
    """Extract key metrics from a run."""
    summary = run.summary
    
    return {
        "operator_final_loss": summary.get("operator/loss"),
        "diffusion_final_loss": summary.get("diffusion_residual/loss"),
        "consistency_final_loss": summary.get("consistency_distill/loss"),
        "operator_grad_norm": summary.get("operator/grad_norm"),
        "diffusion_grad_norm": summary.get("diffusion_residual/grad_norm"),
        "eval_baseline_nrmse": summary.get("eval/baseline_nrmse"),
        "eval_ttc_nrmse": summary.get("eval/ttc_nrmse"),
        "eval_ttc_improvement": summary.get("eval/ttc_improvement_pct"),
        "eval_conservation_gap": summary.get("eval/conservation_gap"),
        "eval_bc_violation": summary.get("eval/bc_violation"),
        "eval_negativity_penalty": summary.get("eval/negativity_penalty"),
        "eval_baseline_rel_l2": summary.get("eval/baseline_rel_l2"),
        "eval_ttc_rel_l2": summary.get("eval/ttc_rel_l2"),
        "eval_ttc_conservation_gap": summary.get("eval/ttc_conservation_gap"),
        "eval_ttc_bc_violation": summary.get("eval/ttc_bc_violation"),
        "eval_ttc_negativity_penalty": summary.get("eval/ttc_negativity_penalty"),
    }


def extract_config_subset(config: Dict) -> Dict:
    """Extract relevant config parameters for comparison."""
    subset = {}
    
    # Latent dimension
    if "latent" in config:
        subset["latent_dim"] = config["latent"].get("dim")
    
    # Training params
    if "training" in config:
        training = config["training"]
        subset["batch_size"] = training.get("batch_size")
        subset["time_stride"] = training.get("time_stride")
        subset["grad_clip"] = training.get("grad_clip")
        subset["compile"] = training.get("compile")
        subset["num_workers"] = training.get("num_workers")
    
    # Stages
    if "stages" in config:
        stages = config["stages"]
        
        if "operator" in stages:
            op = stages["operator"]
            subset["op_epochs"] = op.get("epochs")
            if "optimizer" in op:
                subset["op_lr"] = op["optimizer"].get("lr")
                subset["op_weight_decay"] = op["optimizer"].get("weight_decay")
        
        if "diff_residual" in stages:
            diff = stages["diff_residual"]
            subset["diff_epochs"] = diff.get("epochs")
            if "optimizer" in diff:
                subset["diff_lr"] = diff["optimizer"].get("lr")
        
        if "consistency_distill" in stages:
            cons = stages["consistency_distill"]
            subset["cons_epochs"] = cons.get("epochs")
    
    # TTC
    if "ttc" in config:
        ttc = config["ttc"]
        subset["ttc_enabled"] = ttc.get("enabled")
        subset["ttc_candidates"] = ttc.get("candidates")
        subset["ttc_beam_width"] = ttc.get("beam_width")
    
    return subset


def compare_configs(runs: List[wandb.apis.public.Run]) -> str:
    """Generate config comparison table."""
    lines = []
    lines.append("## Configuration Comparison\n")
    
    # Extract configs
    configs = [extract_config_subset(run.config) for run in runs]
    
    # Get all unique keys
    all_keys = set()
    for config in configs:
        all_keys.update(config.keys())
    
    # Build table
    lines.append("| Parameter | " + " | ".join([f"Run {i+1} ({run.id[:7]})" for i, run in enumerate(runs)]) + " |")
    lines.append("|" + "---|" * (len(runs) + 1))
    
    for key in sorted(all_keys):
        values = [str(config.get(key, "N/A")) for config in configs]
        
        # Check if values differ
        unique_values = set(values)
        if len(unique_values) > 1:
            # Highlight differences with **bold**
            row = f"| **{key}** | " + " | ".join([f"**{v}**" for v in values]) + " |"
        else:
            row = f"| {key} | " + " | ".join(values) + " |"
        
        lines.append(row)
    
    return "\n".join(lines) + "\n"


def compare_metrics(runs: List[wandb.apis.public.Run]) -> str:
    """Generate metrics comparison table."""
    lines = []
    lines.append("## Performance Metrics Comparison\n")
    
    # Extract metrics
    all_metrics = [extract_metrics(run) for run in runs]
    
    # Build table
    lines.append("| Metric | " + " | ".join([f"Run {i+1} ({run.id[:7]})" for i, run in enumerate(runs)]) + " | Best |")
    lines.append("|" + "---|" * (len(runs) + 2))
    
    metric_names = {
        "operator_final_loss": "Operator Final Loss",
        "diffusion_final_loss": "Diffusion Final Loss",
        "consistency_final_loss": "Consistency Final Loss",
        "operator_grad_norm": "Operator Grad Norm",
        "diffusion_grad_norm": "Diffusion Grad Norm",
        "eval_baseline_nrmse": "Eval Baseline NRMSE",
        "eval_ttc_nrmse": "Eval TTC NRMSE",
        "eval_ttc_improvement": "TTC Improvement %",
        "eval_baseline_rel_l2": "Eval Baseline Rel L2",
        "eval_ttc_rel_l2": "Eval TTC Rel L2",
        "eval_conservation_gap": "Baseline Conservation Gap",
        "eval_bc_violation": "Baseline BC Violation",
        "eval_negativity_penalty": "Baseline Negativity Penalty",
        "eval_ttc_conservation_gap": "TTC Conservation Gap",
        "eval_ttc_bc_violation": "TTC BC Violation",
        "eval_ttc_negativity_penalty": "TTC Negativity Penalty",
    }
    
    for key, display_name in metric_names.items():
        values = [metrics.get(key) for metrics in all_metrics]
        
        # Skip if all None
        if all(v is None for v in values):
            continue
        
        # Find best (lowest for losses, highest for improvement)
        numeric_values = [v for v in values if v is not None]
        if not numeric_values:
            continue
        
        if "improvement" in key.lower():
            best_value = max(numeric_values)
            best_idx = values.index(best_value)
        else:
            best_value = min(numeric_values)
            best_idx = values.index(best_value)
        
        # Format values
        formatted_values = []
        for i, v in enumerate(values):
            if v is None:
                formatted_values.append("N/A")
            else:
                # Format number
                if abs(v) < 0.001:
                    formatted = f"{v:.2e}"
                else:
                    formatted = f"{v:.6f}"
                
                # Bold if best
                if i == best_idx:
                    formatted = f"**{formatted}**"
                
                formatted_values.append(formatted)
        
        best_run_name = runs[best_idx].name
        row = f"| {display_name} | " + " | ".join(formatted_values) + f" | Run {best_idx+1} |"
        lines.append(row)
    
    return "\n".join(lines) + "\n"


def generate_insights(runs: List[wandb.apis.public.Run]) -> str:
    """Generate insights about what changed between runs."""
    lines = []
    lines.append("## Key Insights\n")
    
    if len(runs) != 2:
        lines.append("*Insights available for 2-run comparisons only*\n")
        return "\n".join(lines)
    
    # Compare configs
    config1 = extract_config_subset(runs[0].config)
    config2 = extract_config_subset(runs[1].config)
    
    differences = []
    for key in set(config1.keys()) | set(config2.keys()):
        val1 = config1.get(key)
        val2 = config2.get(key)
        if val1 != val2:
            differences.append(f"- **{key}:** {val1} → {val2}")
    
    if differences:
        lines.append("### Configuration Changes\n")
        lines.extend(differences)
        lines.append("")
    
    # Compare metrics
    metrics1 = extract_metrics(runs[0])
    metrics2 = extract_metrics(runs[1])
    
    improvements = []
    degradations = []
    
    for key in metrics1.keys():
        val1 = metrics1.get(key)
        val2 = metrics2.get(key)
        
        if val1 is None or val2 is None:
            continue
        
        # Determine if improvement or degradation
        if "improvement" in key.lower():
            # Higher is better
            if val2 > val1:
                improvements.append(f"- **{key}:** {val1:.3f}% → {val2:.3f}% (+{val2-val1:.3f}%)")
            elif val2 < val1:
                degradations.append(f"- **{key}:** {val1:.3f}% → {val2:.3f}% ({val2-val1:.3f}%)")
        else:
            # Lower is better
            if val2 < val1:
                pct_change = (val1 - val2) / val1 * 100
                improvements.append(f"- **{key}:** {val1:.6f} → {val2:.6f} (-{pct_change:.1f}%)")
            elif val2 > val1:
                pct_change = (val2 - val1) / val1 * 100
                degradations.append(f"- **{key}:** {val1:.6f} → {val2:.6f} (+{pct_change:.1f}%)")
    
    if improvements:
        lines.append("### ✅ Improvements\n")
        lines.extend(improvements)
        lines.append("")
    
    if degradations:
        lines.append("### ❌ Degradations\n")
        lines.extend(degradations)
        lines.append("")
    
    if not improvements and not degradations:
        lines.append("*No significant performance differences detected*\n")
    
    return "\n".join(lines)


def generate_report(runs: List[wandb.apis.public.Run]) -> str:
    """Generate comparison report."""
    lines = []
    
    # Header
    lines.append(f"# Training Run Comparison ({len(runs)} runs)\n")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Run overview
    lines.append("## Runs Overview\n")
    for i, run in enumerate(runs, 1):
        lines.append(f"{i}. **{run.name}** (`{run.id}`)")
        lines.append(f"   - Created: {run.created_at}")
        lines.append(f"   - State: {run.state}")
        lines.append(f"   - URL: {run.url}")
    lines.append("\n---\n")
    
    # Config comparison
    lines.append(compare_configs(runs))
    lines.append("---\n")
    
    # Metrics comparison
    lines.append(compare_metrics(runs))
    lines.append("---\n")
    
    # Insights
    lines.append(generate_insights(runs))
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compare W&B training runs")
    parser.add_argument("run_ids", nargs="+", help="Run IDs to compare (2 or more)")
    parser.add_argument("--entity", help="W&B entity")
    parser.add_argument("--project", default="universal-simulator", help="W&B project")
    parser.add_argument("--output", "-o", help="Output markdown file (default: print to stdout)")
    args = parser.parse_args()
    
    if len(args.run_ids) < 2:
        print("❌ Please provide at least 2 run IDs to compare")
        sys.exit(1)
    
    print(f"🔍 Fetching {len(args.run_ids)} runs...")
    runs = fetch_runs(args.run_ids, args.entity, args.project)
    
    print("📊 Comparing runs...")
    report = generate_report(runs)
    
    # Output
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report)
        print(f"\n✅ Comparison report saved to: {output_path}")
    else:
        print("\n" + "="*70)
        print(report)
        print("="*70)


if __name__ == "__main__":
    main()
