---
title: Fix Distributed Training Hang via Coordinated OOM Handling
date: 2025-11-15
owner: Codex (CLI)
status: complete
tags:
  - ddp
  - distributed-training
  - pdebench
  - implementation-plan
related_research:
  - thoughts/shared/research/2025-11-15-ddp-hang-vast-status.md
---

## Goal
Ensure multi-GPU PDEBench training no longer deadlocks when only one rank encounters an out-of-memory exception. The fix must provide a shared signaling mechanism, deterministic tests, and operational guidance so Vast.ai launches reliably produce WandB metrics.

## Current Behavior
- OOM handling in `scripts/train.py` broadcasts `skip_flag` only on the rank that raised the exception (`scripts/train.py:1246-1263`, `1550-1603`, `2036-2052`).
- Other ranks stay inside their forward/backward pass, never call the broadcast, and block forever.
- Result: torchrun hangs before logging a single batch; WandB runs remain “running” with `historyStep=-1`.

## Plan of Record

### 1. Shared Distributed Error Helper
- Add `ups/training/distributed_utils.py` exposing `sync_error_flag(flag, device, is_distributed)` and `maybe_empty_cache(flag)`.
- Implementation: convert boolean to tensor, perform `dist.all_reduce` (SUM) when distributed so every rank sees whether **any** peer failed.
- Keep helpers no-op in single-GPU mode.

### 2. Refactor Stage Loops
- Operator stage (`scripts/train.py:1180-1340`):
  - Wrap batch body in `try/except/finally`; set `oom_flag` inside `except`.
  - In `finally`, call `sync_error_flag`. If result is true, zero grads, skip logging/metrics, `continue`.
  - Remove old `dist.broadcast` usage.
- Diffusion residual + consistency stages (`scripts/train.py:1500-2100`):
  - Apply identical pattern around teacher forward, diffusion computation, and consistency chunking so every collective stays aligned.
  - Guard EMA/scheduler updates so they only run when `skip_flag` is false.

### 3. Deterministic OOM Simulation Hook
- Add debug-only env/config (e.g., `SIMULATE_OOM_RANK` + `SIMULATE_OOM_STEP`) that raises an artificial `"CUDA out of memory"` `RuntimeError` at the desired rank/batch.
- Ensure hook is disabled by default and clearly logged when active.
- Document usage in `docs/ddp_debugging_improvements.md`.

### 4. Automated Tests
- Extend `scripts/test_ddp_minimal.py` (or add new script) to run under torchrun with the simulation hook, verifying all ranks exit cleanly.
- Update `tests/integration/test_distributed_training.py`:
  - Add `test_oom_skip_sync` launching `torchrun --nproc_per_node=2 scripts/train.py … --stage operator --epochs 1` with the simulation knobs enabled.
  - Assert return code 0 and presence of the skip log.
- Provide `make test-ddp-oom` target for local/CI use.

### 5. Operational Integration
- Update `.vast/onstart.sh` to optionally run the simulation test before the expensive training run.
- Refresh docs (`docs/ddp_debugging_improvements.md`, relevant thoughts) with:
  - Description of the new sync helper.
  - Instructions for enabling the simulation and interpreting WandB/log output.

## Out of Scope
- No changes to FSDP/FSDP2 selection, data loading, latent cache, or optimizer logic.
- NCCL port-forward warnings in Vast logs are treated as operational noise unless they persist after the fix.

## Success Criteria

### Automated
- `python scripts/test_ddp_minimal.py --simulate-oom-rank 1 --simulate-oom-step 0` succeeds under `torchrun --nproc_per_node=2`.
- `pytest tests/integration/test_distributed_training.py -k oom_skip_sync` passes.
- Existing multi-GPU integration tests (`test_2gpu_training`, `test_4gpu_training`) remain green.
- `make test-ddp-oom` (new target) exits 0.

### Manual
- `torchrun --nproc_per_node=2 scripts/train.py --config configs/train_pdebench_2task_baseline_ddp.yaml --stage operator --epochs 1` logs batches on WandB without hanging.
- Vast.ai launch produces a WandB run with actual history rather than staying stuck at `historyStep=-1`.
- Simulated OOM prints the expected skip warning once per batch and training resumes thereafter.
