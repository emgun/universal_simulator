"""Benchmark distributed training speedup."""

import argparse
import json
import subprocess
import time
from pathlib import Path


def benchmark(nproc: int, config: str, epochs: int = 5) -> dict:
    """Run training and measure time.

    Args:
        nproc: Number of GPUs
        config: Config file path
        epochs: Number of epochs to train

    Returns:
        Dictionary with benchmark results
    """
    cmd = [
        "torchrun",
        f"--nproc_per_node={nproc}",
        "--nnodes=1",
        "--node_rank=0",
        "--master_addr=localhost",
        "--master_port=29500",
        "scripts/train.py",
        "--config",
        config,
        "--stage",
        "operator",
        "--epochs",
        str(epochs),
    ]

    print(f"Running: {' '.join(cmd)}")
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start

    if result.returncode != 0:
        raise RuntimeError(f"Training failed: {result.stderr}")

    return {
        "nproc": nproc,
        "config": config,
        "epochs": epochs,
        "elapsed_sec": elapsed,
        "sec_per_epoch": elapsed / epochs,
    }


def main():
    """Run benchmarks for 1, 2, 4 GPUs."""
    parser = argparse.ArgumentParser(description="Benchmark distributed training speedup")
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of epochs to benchmark (default: 5)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results.json",
        help="Output file for results (default: benchmark_results.json)",
    )
    args = parser.parse_args()

    configs = {
        1: "configs/train_pdebench_2task_baseline.yaml",
        2: "configs/train_pdebench_2task_baseline_ddp.yaml",
        4: "configs/train_pdebench_11task_ddp.yaml",
    }

    results = []

    for nproc, config in configs.items():
        # Check if config exists
        if not Path(config).exists():
            print(f"Skipping {nproc}-GPU benchmark: {config} not found")
            continue

        print(f"\n{'=' * 60}")
        print(f"Benchmarking {nproc}-GPU training")
        print("=" * 60)

        try:
            result = benchmark(nproc=nproc, config=config, epochs=args.epochs)
            results.append(result)

            print(f"Completed in {result['elapsed_sec']:.1f} sec")
            print(f"Throughput: {result['sec_per_epoch']:.1f} sec/epoch")
        except Exception as e:
            print(f"Error benchmarking {nproc}-GPU: {e}")
            continue

    if not results:
        print("No benchmarks completed successfully")
        return

    # Compute speedup
    baseline = results[0]["sec_per_epoch"]

    print(f"\n{'=' * 60}")
    print("Speedup Summary")
    print("=" * 60)
    for result in results:
        speedup = baseline / result["sec_per_epoch"]
        sec_per_epoch = result["sec_per_epoch"]
        print(f"{result['nproc']}-GPU: {speedup:.2f}× speedup ({sec_per_epoch:.1f} sec/epoch)")

    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
