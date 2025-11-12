#!/usr/bin/env python
"""Find optimal per-GPU batch size before OOM."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ups.data.latent_pairs import build_latent_pair_loader
from ups.models.latent_operator import build_operator
from ups.utils.config_loader import load_config


def tune_batch_size(config_path: str, start: int = 4, end: int = 32, step: int = 4):
    """Find maximum batch size that doesn't OOM.

    Args:
        config_path: Path to training config file
        start: Starting batch size to test
        end: Maximum batch size to test
        step: Increment between tests

    This script should be run with torchrun for distributed tuning:
        torchrun --nproc_per_node=2 scripts/tune_batch_size.py config.yaml

    The script will iteratively test increasing batch sizes until OOM,
    then report the maximum working batch size.
    """
    # Setup distributed (if running under torchrun)
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = dist.get_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")

        if rank == 0:
            print(f"Tuning batch size with {dist.get_world_size()} GPUs")
    else:
        rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Tuning batch size on single device: {device}")

    cfg = load_config(config_path)

    max_working = 0

    for batch_size in range(start, end + 1, step):
        if rank == 0:
            print(f"\n{'='*60}")
            print(f"Testing batch_size={batch_size}")
            print("=" * 60)

        # Update config
        cfg["training"]["batch_size"] = batch_size

        try:
            # Build model and data
            train_loader = build_latent_pair_loader(cfg)
            operator = build_operator(cfg).to(device)

            if dist.is_initialized():
                from torch.nn.parallel import DistributedDataParallel as DDP

                operator = DDP(
                    operator,
                    device_ids=[local_rank],
                    output_device=local_rank,
                    static_graph=True,
                )

            # Test one batch
            batch = next(iter(train_loader))
            z0 = batch["z0"].to(device)
            z1 = batch["z1"].to(device)
            cond = {k: v.to(device) for k, v in batch.get("cond", {}).items()}

            from ups.core.latent_state import LatentState

            state = LatentState(z=z0, t=torch.tensor(0.0, device=device), cond=cond)

            pred = operator(state, dt=torch.tensor(0.1, device=device))
            loss = torch.nn.functional.mse_loss(pred.z, z1)
            loss.backward()

            # Success!
            max_working = batch_size
            allocated = torch.cuda.memory_allocated() / 1e9
            peak = torch.cuda.max_memory_allocated() / 1e9

            if rank == 0:
                print(f"✓ SUCCESS: batch_size={batch_size}")
                print(f"  Allocated: {allocated:.2f} GB")
                print(f"  Peak memory: {peak:.2f} GB")

            # Cleanup
            del operator, train_loader, batch, z0, z1, cond, state, pred, loss
            torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                if rank == 0:
                    print(f"✗ OOM: batch_size={batch_size}")
                torch.cuda.empty_cache()
                break  # Stop increasing
            else:
                # Re-raise non-OOM errors
                raise

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Maximum batch size: {max_working}")
        print(f"{'='*60}\n")

    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()

    return max_working


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/tune_batch_size.py <config_path> [start] [end] [step]")
        print("   Or: torchrun --nproc_per_node=N scripts/tune_batch_size.py <config_path>")
        sys.exit(1)

    config_path = sys.argv[1]
    start = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    end = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    step = int(sys.argv[4]) if len(sys.argv) > 4 else 4

    tune_batch_size(config_path, start, end, step)
