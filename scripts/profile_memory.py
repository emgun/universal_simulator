#!/usr/bin/env python
"""Profile GPU memory usage during training."""

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


def profile_memory(config_path: str, stage: str = "operator"):
    """Profile memory usage for a given config and stage.

    Args:
        config_path: Path to training config file
        stage: Training stage (operator, diff_residual, etc.)

    This script should be run with torchrun for distributed profiling:
        torchrun --nproc_per_node=2 scripts/profile_memory.py config.yaml
    """
    # Setup distributed (if running under torchrun)
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")

        if rank == 0:
            print(f"Profiling memory with {dist.get_world_size()} GPUs")
    else:
        rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Profiling memory on single device: {device}")

    # Load config and build model
    cfg = load_config(config_path)
    train_loader = build_latent_pair_loader(cfg)
    operator = build_operator(cfg).to(device)

    # Wrap with DDP if distributed
    if dist.is_initialized():
        from torch.nn.parallel import DistributedDataParallel as DDP

        operator = DDP(
            operator,
            device_ids=[local_rank],
            output_device=local_rank,
            static_graph=True,
        )

    # Profile forward + backward pass
    torch.cuda.reset_peak_memory_stats()

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Memory Profiling: {config_path}")
        print(f"Stage: {stage}")
        print(f"{'='*60}\n")

    for i, batch in enumerate(train_loader):
        if i >= 5:  # Profile first 5 batches
            break

        # Forward
        z0 = batch["z0"].to(device)
        z1 = batch["z1"].to(device)
        cond = {k: v.to(device) for k, v in batch.get("cond", {}).items()}

        from ups.core.latent_state import LatentState

        state = LatentState(z=z0, t=torch.tensor(0.0, device=device), cond=cond)

        pred = operator(state, dt=torch.tensor(0.1, device=device))
        loss = torch.nn.functional.mse_loss(pred.z, z1)

        # Backward
        loss.backward()

        # Print memory stats (rank 0 only)
        if rank == 0:
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            peak = torch.cuda.max_memory_allocated() / 1e9

            print(f"Batch {i+1}:")
            print(f"  Allocated: {allocated:.2f} GB")
            print(f"  Reserved:  {reserved:.2f} GB")
            print(f"  Peak:      {peak:.2f} GB")

        # Clear gradients for next iteration
        operator.zero_grad(set_to_none=True)

    # Final summary
    if rank == 0:
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"\n{'='*60}")
        print(f"Peak Memory Usage: {peak:.2f} GB")
        print(f"{'='*60}\n")

    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/profile_memory.py <config_path> [stage]")
        print("   Or: torchrun --nproc_per_node=N scripts/profile_memory.py <config_path>")
        sys.exit(1)

    config_path = sys.argv[1]
    stage = sys.argv[2] if len(sys.argv) > 2 else "operator"

    profile_memory(config_path, stage)
