#!/usr/bin/env python3
"""Minimal DDP test script to isolate initialization issues.

Usage:
    # Single process (for comparison)
    python scripts/test_ddp_minimal.py

    # Multi-GPU with torchrun
    torchrun --nproc_per_node=2 --nnodes=1 --master_addr=localhost --master_port=29500 scripts/test_ddp_minimal.py

Simulated OOM testing:
    torchrun --nproc_per_node=2 scripts/test_ddp_minimal.py --simulate-oom-rank 1 --simulate-oom-step 0
    # or export UPS_SIMULATE_OOM_RANK / UPS_SIMULATE_OOM_STEP and run without flags

Expected behavior:
    - Both ranks should print their rank and world_size
    - Both ranks should perform a simple all_reduce operation
    - Script should exit cleanly with success message
"""
import argparse
import os
import sys
import traceback
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ups.training.distributed_utils import maybe_empty_cache, sync_error_flag


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal DDP diagnostic")
    parser.add_argument("--simulate-oom-rank", type=int, default=None)
    parser.add_argument("--simulate-oom-step", type=int, default=None)
    parser.add_argument(
        "--simulate-oom-steps",
        type=int,
        default=1,
        help="Number of synthetic steps to run when simulating OOM (default: 1)",
    )
    args = parser.parse_args()

    env_rank = os.environ.get("UPS_SIMULATE_OOM_RANK")
    env_step = os.environ.get("UPS_SIMULATE_OOM_STEP")
    if args.simulate_oom_rank is None and env_rank is not None:
        try:
            args.simulate_oom_rank = int(env_rank)
        except ValueError:
            print(f"[SIM-OOM] Ignoring invalid UPS_SIMULATE_OOM_RANK={env_rank!r}")
    if args.simulate_oom_step is None and env_step is not None:
        try:
            args.simulate_oom_step = int(env_step)
        except ValueError:
            print(f"[SIM-OOM] Ignoring invalid UPS_SIMULATE_OOM_STEP={env_step!r}")
    return args


def main():
    args = _parse_args()
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
    if hasattr(torch.cuda, "nccl"):
        try:
            nccl_available = torch.cuda.nccl.is_available()
        except TypeError:
            nccl_available = "unknown"
        print(f"  NCCL available: {nccl_available}")
        if nccl_available:
            try:
                print(f"  NCCL version: {torch.cuda.nccl.version()}")
            except (AttributeError, RuntimeError):
                print("  NCCL version: unavailable")
    else:
        print("  NCCL available: N/A")

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

    if (
        args.simulate_oom_rank is not None
        and args.simulate_oom_step is not None
        and "RANK" in os.environ
    ):
        print("\n[STEP 6] Simulated OOM coordination test:")
        synthetic_steps = max(1, args.simulate_oom_steps)
        for step in range(synthetic_steps):
            local_failure = False
            try:
                if rank == args.simulate_oom_rank and step == args.simulate_oom_step:
                    raise RuntimeError("CUDA out of memory (simulated)")
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    local_failure = True
                    maybe_empty_cache(True)
                    print(f"  [Rank {rank}] Simulated OOM triggered at step {step}")
                else:
                    raise
            should_skip = sync_error_flag(local_failure, device, True)
            if should_skip:
                if not local_failure:
                    print(
                        f"  [Rank {rank}] Received remote OOM signal at step {step}, skipping synthetic work"
                    )
                continue
            print(f"  [Rank {rank}] Synthetic step {step} completed without simulated OOM")

    # Step 7: Synchronize and cleanup
    print(f"\n[STEP 7] Synchronizing and cleaning up:")
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
