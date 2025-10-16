#!/usr/bin/env python3
"""
Add SOTA-comparable metrics (nRMSE, relative L2) to existing evaluation results.

This script recomputes normalized metrics from existing evaluation data
to enable comparison with state-of-the-art PDE solver benchmarks.
"""

import json
import math
from pathlib import Path
import argparse


def compute_sota_metrics(report_json_path: Path) -> dict:
    """
    Compute nRMSE and relative L2 from existing evaluation results.

    Note: We need the raw data to compute these properly. This is a placeholder
    that estimates from MSE, but proper implementation requires re-running evaluation.
    """
    with open(report_json_path) as f:
        data = json.load(f)

    metrics = data['metrics']
    mse = metrics['mse']
    rmse = metrics['rmse']

    # For Burgers equation, typical solution magnitudes are O(1-10)
    # This is an estimate - real nRMSE requires target statistics
    # For proper SOTA comparison, re-run evaluation with updated code

    print(f"\n⚠️  Note: Proper nRMSE/rel_l2 computation requires re-running evaluation")
    print(f"    Current metrics are estimates based on typical Burgers solution magnitudes")
    print(f"\n📊 Existing metrics from {report_json_path.name}:")
    print(f"    MSE:  {mse:.6f}")
    print(f"    RMSE: {rmse:.6f}")

    # Estimate assuming target RMS ~ 1.0 (typical for normalized Burgers)
    estimated_target_rms = 1.0
    nrmse_estimate = rmse / estimated_target_rms

    print(f"\n📐 Estimated SOTA metrics (assuming target RMS ≈ {estimated_target_rms}):")
    print(f"    nRMSE:     {nrmse_estimate:.6f}")
    print(f"    Relative L2: {nrmse_estimate:.6f}")

    return {
        'mse': mse,
        'rmse': rmse,
        'nrmse_estimate': nrmse_estimate,
        'rel_l2_estimate': nrmse_estimate,
        'note': 'Estimates only - re-run evaluation for accurate values'
    }


def main():
    parser = argparse.ArgumentParser(description='Estimate SOTA metrics from existing results')
    parser.add_argument('report_json', type=Path, help='Path to evaluation JSON report')
    args = parser.parse_args()

    if not args.report_json.exists():
        print(f"Error: {args.report_json} not found")
        return 1

    metrics = compute_sota_metrics(args.report_json)

    print(f"\n✅ To get accurate nRMSE and relative L2:")
    print(f"    Re-run evaluation with updated code:")
    print(f"    PYTHONPATH=src python scripts/evaluate.py --config <config> ...")

    return 0


if __name__ == '__main__':
    exit(main())
