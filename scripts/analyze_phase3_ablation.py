#!/usr/bin/env python3
"""Analyze Phase 3 ablation study results.

Compares U-shaped vs Pure transformer architectures across different token counts
and attention mechanisms.

Usage:
    python scripts/analyze_phase3_ablation.py --output-dir reports/phase3
"""
import argparse
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd


def fetch_phase3_runs() -> List[Any]:
    """Fetch all Phase 3 runs from WandB."""
    try:
        import wandb
        api = wandb.Api()
        runs = api.runs(
            "emgun-morpheus-space/universal-simulator",
            filters={"tags": {"$in": ["phase3", "upt"]}},
        )
        return list(runs)
    except ImportError:
        print("❌ wandb not installed. Install with: pip install wandb")
        return []
    except Exception as e:
        print(f"❌ Failed to fetch runs: {e}")
        return []


def extract_run_data(runs: List[Any]) -> pd.DataFrame:
    """Extract relevant data from runs."""
    results = []

    for run in runs:
        config = run.config
        summary = run.summary

        # Extract key configuration
        arch_type = config.get("operator", {}).get("architecture_type", "pdet_unet")
        latent_cfg = config.get("latent", {})
        pdet_cfg = config.get("operator", {}).get("pdet", {})

        results.append({
            "run_id": run.id,
            "name": run.name,
            "state": run.state,
            "created_at": run.created_at,
            "tokens": latent_cfg.get("tokens"),
            "latent_dim": latent_cfg.get("dim"),
            "architecture": arch_type,
            "attention": pdet_cfg.get("attention_type", "channel_separated"),
            "depth": pdet_cfg.get("depth") or sum(pdet_cfg.get("depths", [])),
            "drop_path": pdet_cfg.get("drop_path", 0.0),
            "hidden_dim": pdet_cfg.get("hidden_dim"),
            # Evaluation metrics
            "nrmse_baseline": summary.get("eval/baseline_nrmse"),
            "nrmse_ttc": summary.get("eval/ttc_nrmse"),
            "mse_baseline": summary.get("eval/baseline_mse"),
            "mse_ttc": summary.get("eval/ttc_mse"),
            # Training metrics
            "operator_final_loss": summary.get("operator/final_loss"),
            "training_time_hours": summary.get("_runtime", 0) / 3600,
            # Physics metrics
            "conservation_gap": summary.get("eval/ttc_conservation_gap"),
            "bc_violation": summary.get("eval/ttc_bc_violation"),
            "negativity_penalty": summary.get("eval/ttc_negativity_penalty"),
        })

    df = pd.DataFrame(results)

    # Compute TTC improvement
    if "nrmse_baseline" in df.columns and "nrmse_ttc" in df.columns:
        df["ttc_improvement_pct"] = (
            (df["nrmse_baseline"] - df["nrmse_ttc"]) / df["nrmse_baseline"] * 100
        )

    return df


def compare_architectures(df: pd.DataFrame, output_dir: Path):
    """Compare U-shaped vs Pure transformer at same token counts."""
    print("\n" + "=" * 70)
    print("ARCHITECTURE COMPARISON")
    print("=" * 70)

    for tokens in sorted(df["tokens"].unique()):
        subset = df[df["tokens"] == tokens].sort_values("nrmse_ttc")

        print(f"\n{tokens} Tokens:")
        print("-" * 70)

        for _, row in subset.iterrows():
            arch_label = "U-shaped" if row["architecture"] == "pdet_unet" else "Pure"
            attn_label = row["attention"]
            print(
                f"  {arch_label:10} | {attn_label:18} | "
                f"NRMSE: {row['nrmse_baseline']:.6f} → {row['nrmse_ttc']:.6f} | "
                f"Improvement: {row['ttc_improvement_pct']:+.1f}%"
            )

    # Save comparison table
    comparison_file = output_dir / "architecture_comparison.csv"
    df.to_csv(comparison_file, index=False)
    print(f"\n✅ Saved comparison data: {comparison_file}")


def plot_results(df: pd.DataFrame, output_dir: Path):
    """Generate comparison plots."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("⚠️  matplotlib not installed, skipping plots")
        return

    # Plot 1: NRMSE comparison by token count
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for tokens in sorted(df["tokens"].unique()):
        subset = df[df["tokens"] == tokens]

        # Baseline NRMSE
        x_labels = [
            f"{row['architecture'][:4]}\\n{row['attention'][:4]}"
            for _, row in subset.iterrows()
        ]
        x_pos = np.arange(len(x_labels))

        axes[0].bar(x_pos, subset["nrmse_baseline"], alpha=0.7, label=f"{tokens} tokens")
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels(x_labels, rotation=45, ha="right")

        # TTC NRMSE
        axes[1].bar(x_pos, subset["nrmse_ttc"], alpha=0.7, label=f"{tokens} tokens")
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(x_labels, rotation=45, ha="right")

    axes[0].set_title("Baseline NRMSE")
    axes[0].set_ylabel("NRMSE")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].set_title("TTC NRMSE")
    axes[1].set_ylabel("NRMSE")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plot_file = output_dir / "architecture_comparison.png"
    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    print(f"✅ Saved plot: {plot_file}")
    plt.close()


def generate_report(df: pd.DataFrame, output_path: Path):
    """Generate markdown report."""
    from datetime import datetime

    report = f"""# Phase 3 Ablation Study Results

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

Total runs analyzed: {len(df)}
Completed runs: {len(df[df['state'] == 'finished'])}
Failed runs: {len(df[df['state'] == 'failed'])}

## Results by Token Count

"""

    for tokens in sorted(df["tokens"].unique()):
        subset = df[df["tokens"] == tokens].sort_values("nrmse_ttc")

        report += f"### {tokens} Tokens\n\n"
        report += "| Architecture | Attention | Baseline NRMSE | TTC NRMSE | Improvement | Training Time |\n"
        report += "|--------------|-----------|----------------|-----------|-------------|---------------|\n"

        for _, row in subset.iterrows():
            arch_label = "U-shaped" if row["architecture"] == "pdet_unet" else "Pure"
            report += (
                f"| {arch_label} | {row['attention']} | "
                f"{row['nrmse_baseline']:.6f} | {row['nrmse_ttc']:.6f} | "
                f"{row['ttc_improvement_pct']:+.1f}% | "
                f"{row['training_time_hours']:.1f}h |\n"
            )

        # Find best config for this token count
        if not subset.empty:
            best = subset.iloc[0]
            report += f"\n**Best configuration**: {best['name']} "
            report += f"(NRMSE: {best['nrmse_ttc']:.6f})\n\n"

    # Hypothesis testing
    report += "## Hypothesis Testing\n\n"

    # H1: Pure transformer matches U-shaped at 128 tokens
    tokens_128 = df[df["tokens"] == 128]
    if len(tokens_128) >= 2:
        pure_128 = tokens_128[tokens_128["architecture"] == "pdet_stack"]
        unet_128 = tokens_128[tokens_128["architecture"] == "pdet_unet"]

        if not pure_128.empty and not unet_128.empty:
            pure_nrmse = pure_128["nrmse_ttc"].min()
            unet_nrmse = unet_128["nrmse_ttc"].min()
            diff_pct = abs(pure_nrmse - unet_nrmse) / unet_nrmse * 100

            status = "✅ PASS" if diff_pct <= 5 else "❌ FAIL"
            report += f"**H1: Pure matches U-shaped at 128 tokens** {status}\n"
            report += f"- Pure: {pure_nrmse:.6f}, U-shaped: {unet_nrmse:.6f} "
            report += f"(diff: {diff_pct:.1f}%)\n\n"

    # H2: Pure transformer outperforms U-shaped at 256 tokens
    tokens_256 = df[df["tokens"] == 256]
    if len(tokens_256) >= 2:
        pure_256 = tokens_256[tokens_256["architecture"] == "pdet_stack"]
        unet_256 = tokens_256[tokens_256["architecture"] == "pdet_unet"]

        if not pure_256.empty and not unet_256.empty:
            pure_nrmse = pure_256["nrmse_ttc"].min()
            unet_nrmse = unet_256["nrmse_ttc"].min()
            improvement_pct = (unet_nrmse - pure_nrmse) / unet_nrmse * 100

            status = "✅ PASS" if improvement_pct >= 10 else "❌ FAIL"
            report += f"**H2: Pure outperforms U-shaped at 256 tokens** {status}\n"
            report += f"- Pure: {pure_nrmse:.6f}, U-shaped: {unet_nrmse:.6f} "
            report += f"(improvement: {improvement_pct:+.1f}%)\n\n"

    # H3: Standard attention comparable to channel-separated
    pure_std = df[(df["architecture"] == "pdet_stack") & (df["attention"] == "standard")]
    pure_chan = df[(df["architecture"] == "pdet_stack") & (df["attention"] == "channel_separated")]

    if not pure_std.empty and not pure_chan.empty:
        std_nrmse = pure_std["nrmse_ttc"].min()
        chan_nrmse = pure_chan["nrmse_ttc"].min()
        diff_pct = abs(std_nrmse - chan_nrmse) / chan_nrmse * 100

        status = "✅ PASS" if diff_pct <= 5 else "❌ FAIL"
        report += f"**H3: Standard attention comparable to channel-separated** {status}\n"
        report += f"- Standard: {std_nrmse:.6f}, Channel-sep: {chan_nrmse:.6f} "
        report += f"(diff: {diff_pct:.1f}%)\n\n"

    # Recommendations
    report += "## Recommendations\n\n"

    # Best config by token count
    report += "### Production Configuration Recommendations\n\n"
    for tokens in sorted(df["tokens"].unique()):
        subset = df[df["tokens"] == tokens].sort_values("nrmse_ttc")
        if not subset.empty:
            best = subset.iloc[0]
            arch_label = "U-shaped" if best["architecture"] == "pdet_unet" else "Pure stacked"
            report += f"**{tokens} tokens**: Use {arch_label} transformer with {best['attention']} attention\n"
            report += f"- Config: `{best['name']}.yaml`\n"
            report += f"- Expected NRMSE: {best['nrmse_ttc']:.6f}\n\n"

    # Write report
    with open(output_path, "w") as f:
        f.write(report)

    print(f"✅ Saved report: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze Phase 3 ablation study")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/phase3"),
        help="Output directory for reports and plots",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed information",
    )
    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PHASE 3 ABLATION STUDY ANALYSIS")
    print("=" * 70)

    # Fetch runs from WandB
    print("\nFetching Phase 3 runs from WandB...")
    runs = fetch_phase3_runs()

    if not runs:
        print("❌ No runs found. Make sure runs are tagged with 'phase3' and 'upt'")
        return

    print(f"✅ Found {len(runs)} runs")

    # Extract data
    print("\nExtracting run data...")
    df = extract_run_data(runs)

    if df.empty:
        print("❌ No data extracted from runs")
        return

    print(f"✅ Extracted data from {len(df)} runs")

    # Display summary
    if args.verbose:
        print("\nRun summary:")
        print(df[["name", "tokens", "architecture", "attention", "nrmse_ttc"]].to_string())

    # Compare architectures
    compare_architectures(df, args.output_dir)

    # Generate plots
    print("\nGenerating plots...")
    plot_results(df, args.output_dir)

    # Generate report
    print("\nGenerating markdown report...")
    generate_report(df, args.output_dir / "ablation_report.md")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {args.output_dir}")
    print("  - architecture_comparison.csv")
    print("  - architecture_comparison.png")
    print("  - ablation_report.md")


if __name__ == "__main__":
    main()
