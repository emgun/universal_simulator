#!/usr/bin/env python3
"""
Validate training optimizations by comparing baseline and optimized configs.

Usage:
    python scripts/validate_optimizations.py \
        --baseline configs/train_pdebench_2task_baseline_ddp_4gpu_original.yaml \
        --optimized configs/train_pdebench_2task_baseline_ddp_4gpu.yaml \
        --epochs 20 \
        --seed 42
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import yaml


def run_training(config_path: str, epochs: int, seed: int, output_log: Path) -> subprocess.CompletedProcess:
    """Run training with given config.

    Args:
        config_path: Path to YAML config file
        epochs: Number of epochs to train
        seed: Random seed
        output_log: Path to save training logs

    Returns:
        CompletedProcess result
    """
    cmd = [
        "python", "scripts/train.py",
        "--config", config_path,
        "--stage", "operator",
        "--epochs", str(epochs),
        "--seed", str(seed),
    ]

    print(f"Running: {' '.join(cmd)}")
    print(f"Logging to: {output_log}")

    with output_log.open("w") as f:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        f.write(result.stdout)

    return result


def parse_metrics(log_path: Path) -> Dict[str, Any]:
    """Extract metrics from training logs.

    Parses epoch times, final loss, memory usage, etc. from training output.

    Args:
        log_path: Path to training log file

    Returns:
        Dictionary of extracted metrics
    """
    if not log_path.exists():
        print(f"Warning: Log file not found: {log_path}")
        return {}

    metrics = {
        "avg_epoch_time": None,
        "final_loss": None,
        "memory_gb": None,
        "epoch_times": [],
        "losses": [],
    }

    with log_path.open() as f:
        for line in f:
            # Parse epoch time (format: "Epoch 1/20 completed in 5.2 min")
            epoch_time_match = re.search(r"Epoch \d+/\d+ completed in ([\d.]+) min", line)
            if epoch_time_match:
                epoch_time = float(epoch_time_match.group(1))
                metrics["epoch_times"].append(epoch_time)

            # Parse training loss (format: "train_loss: 0.00023")
            loss_match = re.search(r"train_loss:\s*([\d.e-]+)", line)
            if loss_match:
                loss = float(loss_match.group(1))
                metrics["losses"].append(loss)

            # Parse GPU memory (format: "GPU 0: 15.2 GB")
            memory_match = re.search(r"GPU \d+:\s*([\d.]+)\s*GB", line)
            if memory_match:
                memory_gb = float(memory_match.group(1))
                if metrics["memory_gb"] is None or memory_gb > metrics["memory_gb"]:
                    metrics["memory_gb"] = memory_gb

    # Calculate averages
    if metrics["epoch_times"]:
        # Skip first 2 epochs (warmup) and last 1 (checkpoint overhead)
        valid_times = metrics["epoch_times"][2:-1] if len(metrics["epoch_times"]) > 3 else metrics["epoch_times"]
        metrics["avg_epoch_time"] = sum(valid_times) / len(valid_times) if valid_times else None

    if metrics["losses"]:
        metrics["final_loss"] = metrics["losses"][-1]

    return metrics


def compare_runs(baseline_metrics: Dict[str, Any], optimized_metrics: Dict[str, Any]) -> Tuple[float, float]:
    """Compare baseline and optimized runs.

    Args:
        baseline_metrics: Metrics from baseline run
        optimized_metrics: Metrics from optimized run

    Returns:
        Tuple of (speedup, loss_delta)
    """
    print("\n" + "="*80)
    print("VALIDATION RESULTS")
    print("="*80)

    # Speedup calculation
    if baseline_metrics.get("avg_epoch_time") and optimized_metrics.get("avg_epoch_time"):
        speedup = baseline_metrics["avg_epoch_time"] / optimized_metrics["avg_epoch_time"]
        print(f"\n✓ Speedup: {speedup:.2f}x")
        print(f"  Baseline: {baseline_metrics['avg_epoch_time']:.2f} min/epoch")
        print(f"  Optimized: {optimized_metrics['avg_epoch_time']:.2f} min/epoch")
        print(f"  Time saved: {baseline_metrics['avg_epoch_time'] - optimized_metrics['avg_epoch_time']:.2f} min/epoch")
    else:
        speedup = None
        print("\n✗ Speedup: Could not calculate (missing epoch times)")

    # Loss comparison
    if baseline_metrics.get("final_loss") is not None and optimized_metrics.get("final_loss") is not None:
        loss_delta = abs(baseline_metrics["final_loss"] - optimized_metrics["final_loss"])
        loss_pct = (loss_delta / baseline_metrics["final_loss"]) * 100 if baseline_metrics["final_loss"] > 0 else 0
        print(f"\n✓ Loss delta: {loss_delta:.6f} ({loss_pct:.1f}%)")
        print(f"  Baseline: {baseline_metrics['final_loss']:.6f}")
        print(f"  Optimized: {optimized_metrics['final_loss']:.6f}")
    else:
        loss_delta = None
        print("\n✗ Loss delta: Could not calculate (missing final loss)")

    # Memory comparison
    if baseline_metrics.get("memory_gb") and optimized_metrics.get("memory_gb"):
        memory_increase = optimized_metrics["memory_gb"] - baseline_metrics["memory_gb"]
        memory_increase_pct = (memory_increase / baseline_metrics["memory_gb"]) * 100
        print(f"\n✓ Memory: {baseline_metrics['memory_gb']:.1f}GB → {optimized_metrics['memory_gb']:.1f}GB")
        print(f"  Increase: +{memory_increase:.1f}GB ({memory_increase_pct:+.1f}%)")
    else:
        print("\n✗ Memory: Could not compare (missing memory usage)")

    print("\n" + "="*80)

    return speedup, loss_delta


def validate_results(speedup: Optional[float], loss_delta: Optional[float],
                    target_speedup: float = 3.0, max_loss_delta: float = 0.001) -> bool:
    """Validate that results meet success criteria.

    Args:
        speedup: Measured speedup factor
        loss_delta: Measured loss difference
        target_speedup: Minimum required speedup (default: 3.0x)
        max_loss_delta: Maximum allowed loss delta (default: 0.001)

    Returns:
        True if validation passed, False otherwise
    """
    passed = True

    print("\nVALIDATION GATES:")
    print("-" * 80)

    # Speedup gate
    if speedup is not None:
        if speedup >= target_speedup:
            print(f"✓ Speedup gate: PASSED ({speedup:.2f}x >= {target_speedup:.1f}x)")
        else:
            print(f"✗ Speedup gate: FAILED ({speedup:.2f}x < {target_speedup:.1f}x)")
            passed = False
    else:
        print(f"✗ Speedup gate: SKIPPED (no data)")
        passed = False

    # Loss delta gate
    if loss_delta is not None:
        if loss_delta < max_loss_delta:
            print(f"✓ Loss delta gate: PASSED ({loss_delta:.6f} < {max_loss_delta:.6f})")
        else:
            print(f"✗ Loss delta gate: FAILED ({loss_delta:.6f} >= {max_loss_delta:.6f})")
            passed = False
    else:
        print(f"✗ Loss delta gate: SKIPPED (no data)")
        passed = False

    print("-" * 80)

    return passed


def main():
    parser = argparse.ArgumentParser(description="Validate training optimizations")
    parser.add_argument("--baseline", required=True, help="Path to baseline config YAML")
    parser.add_argument("--optimized", required=True, help="Path to optimized config YAML")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs to train")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--target-speedup", type=float, default=3.0, help="Target speedup (default: 3.0x)")
    parser.add_argument("--max-loss-delta", type=float, default=0.001, help="Maximum loss delta (default: 0.001)")
    parser.add_argument("--skip-baseline", action="store_true", help="Skip baseline run (use existing logs)")
    parser.add_argument("--skip-optimized", action="store_true", help="Skip optimized run (use existing logs)")
    args = parser.parse_args()

    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    baseline_log = logs_dir / "baseline_validation.log"
    optimized_log = logs_dir / "optimized_validation.log"

    # Run baseline
    if not args.skip_baseline:
        print("\n" + "="*80)
        print("RUNNING BASELINE CONFIGURATION")
        print("="*80)
        baseline_result = run_training(args.baseline, args.epochs, args.seed, baseline_log)
        if baseline_result.returncode != 0:
            print(f"✗ Baseline training failed with exit code {baseline_result.returncode}")
            print(f"  Check logs at: {baseline_log}")
            sys.exit(1)
    else:
        print(f"\nSkipping baseline run, using existing log: {baseline_log}")

    # Run optimized
    if not args.skip_optimized:
        print("\n" + "="*80)
        print("RUNNING OPTIMIZED CONFIGURATION")
        print("="*80)
        optimized_result = run_training(args.optimized, args.epochs, args.seed, optimized_log)
        if optimized_result.returncode != 0:
            print(f"✗ Optimized training failed with exit code {optimized_result.returncode}")
            print(f"  Check logs at: {optimized_log}")
            sys.exit(1)
    else:
        print(f"\nSkipping optimized run, using existing log: {optimized_log}")

    # Parse and compare
    print("\n" + "="*80)
    print("PARSING METRICS")
    print("="*80)

    baseline_metrics = parse_metrics(baseline_log)
    print(f"\nBaseline metrics: {len(baseline_metrics.get('epoch_times', []))} epochs, "
          f"final_loss={baseline_metrics.get('final_loss', 'N/A')}")

    optimized_metrics = parse_metrics(optimized_log)
    print(f"Optimized metrics: {len(optimized_metrics.get('epoch_times', []))} epochs, "
          f"final_loss={optimized_metrics.get('final_loss', 'N/A')}")

    speedup, loss_delta = compare_runs(baseline_metrics, optimized_metrics)

    # Validate
    passed = validate_results(speedup, loss_delta, args.target_speedup, args.max_loss_delta)

    # Save results
    results = {
        "baseline": baseline_metrics,
        "optimized": optimized_metrics,
        "speedup": speedup,
        "loss_delta": loss_delta,
        "validation_passed": passed,
        "config": {
            "baseline_config": args.baseline,
            "optimized_config": args.optimized,
            "epochs": args.epochs,
            "seed": args.seed,
            "target_speedup": args.target_speedup,
            "max_loss_delta": args.max_loss_delta,
        }
    }

    results_path = logs_dir / "validation_results.json"
    with results_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Exit with appropriate code
    if passed:
        print("\n✓ All validation gates passed!")
        sys.exit(0)
    else:
        print("\n✗ Validation failed. See results above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
