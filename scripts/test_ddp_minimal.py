#!/usr/bin/env python3
"""Minimal DDP test script to isolate initialization issues.

Usage:
    # Single process (for comparison)
    python scripts/test_ddp_minimal.py

    # Multi-GPU with torchrun
    torchrun --nproc_per_node=2 --nnodes=1 --master_addr=localhost --master_port=29500 scripts/test_ddp_minimal.py

Expected behavior:
    - Both ranks should print their rank and world_size
    - Both ranks should perform a simple all_reduce operation
    - Script should exit cleanly with success message
"""
import os
import sys
import traceback

import torch
import torch.distributed as dist


def main():
    print("=" * 80)
    print("MINIMAL DDP TEST SCRIPT")
    print("=" * 80)

    # Step 1: Log environment variables
    print("\n[STEP 1] Environment Variables:")
    print(f"  RANK = {os.environ.get('RANK', 'NOT_SET')}")
    print(f"  LOCAL_RANK = {os.environ.get('LOCAL_RANK', 'NOT_SET')}")
    print(f"  WORLD_SIZE = {os.environ.get('WORLD_SIZE', 'NOT_SET')}")
    print(f"  MASTER_ADDR = {os.environ.get('MASTER_ADDR', 'NOT_SET')}")
    print(f"  MASTER_PORT = {os.environ.get('MASTER_PORT', 'NOT_SET')}")

    # Step 2: Check PyTorch and CUDA
    print("\n[STEP 2] PyTorch and CUDA Info:")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  CUDA device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"    GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")
    print(
        f"  NCCL available: {torch.cuda.nccl.is_available() if hasattr(torch.cuda, 'nccl') else 'N/A'}"
    )
    if hasattr(torch.cuda, "nccl") and torch.cuda.nccl.is_available():
        print(f"  NCCL version: {torch.cuda.nccl.version()}")

    # Step 3: Initialize DDP if RANK is set
    if "RANK" not in os.environ:
        print("\n[RESULT] RANK env var not set - running in single-process mode")
        print(
            "[RESULT] To test DDP, run with: torchrun --nproc_per_node=2 scripts/test_ddp_minimal.py"
        )
        print("[SUCCESS] Single-process test completed")
        return 0

    print("\n[STEP 3] Initializing DDP...")
    try:
        # Parse environment variables
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        print(f"  Parsed: rank={rank}, local_rank={local_rank}, world_size={world_size}")

        # Initialize process group
        print(f"  [Rank {rank}] Calling dist.init_process_group(backend='nccl')...")
        dist.init_process_group(backend="nccl")
        print(f"  [Rank {rank}] init_process_group() succeeded!")

        # Verify initialization
        assert dist.is_initialized(), "dist.is_initialized() returned False"
        assert dist.get_rank() == rank, f"Rank mismatch: {dist.get_rank()} != {rank}"
        assert (
            dist.get_world_size() == world_size
        ), f"World size mismatch: {dist.get_world_size()} != {world_size}"
        print(f"  [Rank {rank}] Verification passed (backend={dist.get_backend()})")

    except Exception as e:
        print(f"\n[ERROR] DDP initialization failed at rank {os.environ.get('RANK', 'UNKNOWN')}")
        print(f"  Exception: {type(e).__name__}: {e}")
        traceback.print_exc()
        print("\n[DEBUGGING INFO]")
        print("  NCCL/CUDA environment variables:")
        for key in sorted(os.environ.keys()):
            if "NCCL" in key or "CUDA" in key or "TORCH" in key:
                print(f"    {key}={os.environ[key]}")
        return 1

    # Step 4: Set CUDA device
    print(f"\n[STEP 4] Setting CUDA device:")
    try:
        print(f"  [Rank {rank}] Setting device to cuda:{local_rank}")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        print(f"  [Rank {rank}] Current device: {torch.cuda.current_device()}")
        assert torch.cuda.current_device() == local_rank
        print(f"  [Rank {rank}] Device set successfully")
    except Exception as e:
        print(f"  [Rank {rank}] [ERROR] Failed to set device: {e}")
        traceback.print_exc()
        dist.destroy_process_group()
        return 1

    # Step 5: Test all_reduce communication
    print(f"\n[STEP 5] Testing DDP communication:")
    try:
        # Create a tensor with rank value
        tensor = torch.tensor([float(rank)], device=device)
        print(f"  [Rank {rank}] Before all_reduce: tensor={tensor.item()}")

        # All-reduce (sum)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        expected_sum = sum(range(world_size))
        print(f"  [Rank {rank}] After all_reduce: tensor={tensor.item()}, expected={expected_sum}")

        # Verify
        assert (
            abs(tensor.item() - expected_sum) < 1e-6
        ), f"All-reduce failed: {tensor.item()} != {expected_sum}"
        print(f"  [Rank {rank}] All-reduce test PASSED")

    except Exception as e:
        print(f"  [Rank {rank}] [ERROR] Communication test failed: {e}")
        traceback.print_exc()
        dist.destroy_process_group()
        return 1

    # Step 6: Synchronize and cleanup
    print(f"\n[STEP 6] Synchronizing and cleaning up:")
    try:
        dist.barrier()
        print(f"  [Rank {rank}] Barrier synchronized")
        dist.destroy_process_group()
        print(f"  [Rank {rank}] Process group destroyed")
    except Exception as e:
        print(f"  [Rank {rank}] [WARNING] Cleanup issue: {e}")

    # Success!
    print(f"\n{'=' * 80}")
    print(f"[SUCCESS] Rank {rank} completed all tests successfully!")
    print(f"{'=' * 80}")
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n[FATAL ERROR] Unhandled exception: {e}")
        traceback.print_exc()
        sys.exit(1)
