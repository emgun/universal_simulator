---
date: 2025-11-15T18:40:45Z
researcher: Codex (CLI)
git_commit: d50d625369e42b724a265d3b2db4099b5ac6cc22
branch: feature/distributed-training-ddp
repository: universal_simulator
topic: "Distributed DDP hang location + Vast/W&B status"
tags:
  - research
  - ddp
  - distributed-training
  - wandb
  - vast
status: in_progress
last_updated: 2025-11-15
last_updated_by: Codex (CLI)
---

# Research: Distributed DDP hang location + Vast/W&B status

**Date**: 2025-11-15T18:40:45Z (UTC)  
**Researcher**: Codex (CLI)  
**Git Commit**: `d50d625369e42b724a265d3b2db4099b5ac6cc22`  
**Branch**: `feature/distributed-training-ddp`  
**Repository**: `universal_simulator`

## Research Question
Investigate where distributed training could be hanging and report on the current Vast.ai instance together with the WandB run it is driving.

## Summary
Distributed training currently stalls when any rank other than rank 0 hits an out-of-memory exception, because the exception handler calls `dist.broadcast()` only on the failing rank. The other ranks remain in their forward/backward path and never invoke the broadcast, so the process group deadlocks. This single-rank broadcast pattern exists in the operator, diffusion, and consistency stages in `scripts/train.py`. The Vast.ai launcher (`.vast/onstart.sh`) provisions a two-GPU run via `torchrun` + `scripts/run_fast_to_sota.py`, which in turn writes WandB metadata to `FAST_TO_SOTA_WANDB_INFO`. The WandB project currently shows multiple runs (`train-20251115_025031`, `train-20251115_022525`, etc.) stuck in the `running` state with zero history rows, which aligns with a hang before the first metric is logged. Instance `27899066` is in the package-installation phase (pip installing `universal-physics-stack`), repeatedly reporting `Error: remote port forwarding failed for listen port 19066`, so training has not yet progressed far enough for a new run to appear.

## Detailed Findings

### OOM skip signaling only runs on the throwing rank
- The operator stage wraps the main training step in `try/except RuntimeError`. When `.lower()` of the error string contains `"out of memory"`, rank 0 prints a warning and the code attempts to “notify” other ranks by broadcasting `skip_flag = torch.tensor(1, device=device)` with `dist.broadcast(skip_flag, src=0)` (`scripts/train.py:1246-1263`). However, the broadcast is invoked only inside the `except` path of the throwing rank. If rank 1 triggers the OOM, rank 0 is still in the forward pass and never enters the handler, so rank 1 blocks in `dist.broadcast` waiting for rank 0’s matching call. This is the precise hang that shows up in multi-GPU runs that reach the first large batch.

### Diffusion/consistency stages repeat the one-sided broadcast
- The diffusion residual stage wraps both the teacher-forward (`predicted = operator(state, dt_tensor)`) and the actual diffusion step in identical `except RuntimeError` blocks, each calling `dist.broadcast(skip_flag, src=0)` from whichever rank faulted (`scripts/train.py:1550-1603`). The consistency distillation loop uses the same pattern when chunking through `distill_fn` work (`scripts/train.py:2036-2052`). Because those branches also live only in the throwing process, they exhibit the same deadlock whenever the non-zero rank OOMs during the heavy distillation workloads.

### Vast launch flow for current run
- `.vast/onstart.sh:1-103` clones `feature/distributed-training-ddp`, installs the editable package, downloads the `advection1d` and `darcy2d` PDEBench shards, precomputes latent caches, exports NCCL tuning variables, and finally launches `torchrun --nproc_per_node=2 scripts/run_fast_to_sota.py … --wandb-tags vast,ddp,2gpu` (lines 80-87).  
- `scripts/run_fast_to_sota.py:703-824` shows that the launcher runs config validation/data checks/dry-run before the `train` step, sets `FAST_TO_SOTA_WANDB_INFO`, and expects `train.py` to persist WandB metadata back to disk. The running Vast instance (`27899066`) we inspected via `vastai show instance` reports 2×A100_SXM4, $1.8753/hr, and the logs are dominated by `pip install -e .[dev]` output at `Sat Nov 15 18:39 UTC`, indicating the provisioning sequence above is still in progress and training has not yet started.

### WandB runs show zero history rows
- `scripts/train.py:2574-2603` creates the training run ID (`train-<timestamp>`) on rank 0 and writes it to the `FAST_TO_SOTA_WANDB_INFO` path propagated from `run_fast_to_sota.py`.  
- Querying the project (`wandb.Api().runs('emgun-morpheus-space/universal-simulator')`) shows that the latest launches—`train-20251115_025031`, `train-20251115_022525`, `train-20251115_014747`, and `train-20251115_000037`—are all still in `state="running"` with `lastHistoryStep = -1` and empty history DataFrames as of 18:40 UTC. The URLs (for example, https://wandb.ai/emgun-morpheus-space/universal-simulator/runs/train-20251115_025031) match the IDs recorded on the Vast orchestrator, so the code reaches WandB initialization but hangs before logging the first batch, consistent with the OOM broadcast deadlock.

## Code References
- `scripts/train.py:1246-1263` – Operator-stage OOM handler broadcasts `skip_flag` only from the process that caught the exception.  
- `scripts/train.py:1550-1603` – Diffusion residual stage repeats the same single-rank broadcast for both teacher forward and diffusion loss computation.  
- `scripts/train.py:2036-2052` – Consistency distillation chunking uses the same OOM broadcast pattern inside `distill_fn`.  
- `scripts/train.py:2574-2603` – WandB context creation and `FAST_TO_SOTA_WANDB_INFO` persistence for torchrun launches.  
- `.vast/onstart.sh:1-103` – Vast.ai bootstrap script that installs dependencies, downloads PDEBench data, and launches the two-GPU torchrun workflow.  
- `scripts/run_fast_to_sota.py:703-824` – Orchestration logic that wraps validation/dry-run, sets the WandB info path, and executes `scripts/train.py`.

## Architecture Documentation
The distributed training pipeline is coordinated from `.vast/onstart.sh`, which ensures dependencies and data are prepared before invoking `scripts/run_fast_to_sota.py`. That orchestrator performs validation/dry-run gates, then either calls `torchrun … scripts/train.py …` itself or relies on the outer torchrun (when `RANK` is already set). During training, `scripts/train.py` initializes DDP via `setup_distributed()` and wraps each stage in `try/except` blocks intended to synchronize OOM skips. Those handlers directly touch `torch.distributed` primitives; because they live strictly within the throwing rank, they introduce collective calls (broadcasts) that other ranks never execute when they remain inside the forward pass, causing the full process group to stall. WandB integration is done manually: `train.py` constructs a run ID and writes it to `FAST_TO_SOTA_WANDB_INFO`, which `run_fast_to_sota.py` later reads to annotate artifacts/metadata.

## Historical Context (from thoughts/)
- `thoughts/shared/research/2025-11-13-ddp-crash-investigation.md` documents the November 13 wave of DDP crashes (zero WandB history) and already suspected initialization/hang issues in multi-GPU runs.  
- `thoughts/shared/plans/2025-11-12-distributed-training-multi-task.md` captures the design decisions for multi-task DDP training, including the requirement for balanced sampling and OOM safety across ranks—context for why the skip-broadcast blocks were added.

## Related Research
- `thoughts/shared/research/2025-11-13-ddp-crash-investigation.md` – earlier investigation of failed DDP launches and WandB discrepancies.

## Open Questions
- How should the OOM skip signal be propagated so that every rank participates in the same collective (e.g., using `all_reduce` or wrapping the broadcast in a block executed by all ranks)?
- Once the current Vast instance finishes installing packages, does the hang reproduce immediately on the first real batch, confirming the OOM path as the culprit?
- The Vast logs repeatedly report `Error: remote port forwarding failed for listen port 19066`; does that affect SSH/log streaming or is it benign noise from concurrent log fetches?
