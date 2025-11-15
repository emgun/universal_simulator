from __future__ import annotations

"""Helpers for coordinating errors across distributed ranks."""

import os
from typing import Optional

import torch


SIM_OOM_RANK_ENV = "UPS_SIMULATE_OOM_RANK"
SIM_OOM_STEP_ENV = "UPS_SIMULATE_OOM_STEP"
SIM_OOM_STAGE_ENV = "UPS_SIMULATE_OOM_STAGE"


def sync_error_flag(flag: bool, device: torch.device, is_distributed: bool) -> bool:
    """Return True if any rank reported an error."""

    if not is_distributed:
        return flag

    import torch.distributed as dist

    tensor = torch.tensor(1 if flag else 0, device=device, dtype=torch.int32)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return bool(tensor.item())


def maybe_empty_cache(flag: bool) -> None:
    """Free cached CUDA memory if requested."""

    if flag and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _env_to_int(name: str) -> Optional[int]:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        print(f"[SIM-OOM] Ignoring invalid value for {name!r}: {value!r}")
        return None


def maybe_trigger_simulated_oom(stage: str, batch_index: int, rank: int) -> None:
    """Raise a fake CUDA OOM if env vars target this rank/batch."""

    target_rank = _env_to_int(SIM_OOM_RANK_ENV)
    target_step = _env_to_int(SIM_OOM_STEP_ENV)
    target_stage = os.environ.get(SIM_OOM_STAGE_ENV)

    if target_rank is None or target_step is None:
        return
    if target_rank != rank:
        return
    if target_stage and target_stage.lower() != stage.lower():
        return
    if target_step != batch_index:
        return

    print(
        f"[SIM-OOM] Rank {rank} simulating CUDA OOM during {stage} batch {batch_index}. "
        f"(env {SIM_OOM_RANK_ENV}={target_rank}, {SIM_OOM_STEP_ENV}={target_step})"
    )
    raise RuntimeError("CUDA out of memory (simulated for testing)")
