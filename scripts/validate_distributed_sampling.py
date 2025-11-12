#!/usr/bin/env python
"""Validate task distribution across ranks for distributed multi-task training."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ups.data.latent_pairs import build_latent_pair_loader
from ups.utils.config_loader import load_config


def validate_task_distribution(config_path: str):
    """
    Check that each rank sees balanced task distribution.

    This script should be run with torchrun:
        torchrun --nproc_per_node=2 scripts/validate_distributed_sampling.py config.yaml
    """
    # Initialize distributed
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Load config and build loader
    cfg = load_config(config_path)
    train_loader = build_latent_pair_loader(cfg)

    # Collect task names for this rank
    task_counts = Counter()
    for batch in train_loader:
        task_names = batch.get("task_names", [])
        task_counts.update(task_names)

    # Gather counts from all ranks
    all_counts = [None] * world_size
    dist.all_gather_object(all_counts, dict(task_counts))

    if rank == 0:
        print(f"\nTask Distribution Validation (world_size={world_size})")
        print("=" * 60)

        # Check balance across ranks
        all_tasks = set()
        for counts in all_counts:
            all_tasks.update(counts.keys())

        for task in sorted(all_tasks):
            counts_per_rank = [counts.get(task, 0) for counts in all_counts]
            mean = sum(counts_per_rank) / len(counts_per_rank)
            std = (sum((c - mean) ** 2 for c in counts_per_rank) / len(counts_per_rank)) ** 0.5

            print(f"\nTask: {task}")
            print(f"  Counts per rank: {counts_per_rank}")
            print(f"  Mean: {mean:.1f}, Std: {std:.1f}")
            if mean > 0:
                print(f"  Balance: {'✓ GOOD' if std / mean < 0.1 else '✗ IMBALANCED'}")
            else:
                print("  Balance: N/A (no samples)")

        print("\n" + "=" * 60)

    dist.destroy_process_group()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: torchrun --nproc_per_node=N scripts/validate_distributed_sampling.py <config_path>"
        )
        sys.exit(1)
    validate_task_distribution(sys.argv[1])
