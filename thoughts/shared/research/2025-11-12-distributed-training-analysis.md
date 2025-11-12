---
date: 2025-11-12T20:43:11Z
researcher: Emery Gunselman
git_commit: db340949c4fe2e32a7e55702b9fe9ebb41a2a28e
branch: feature--UPT
repository: universal_simulator
topic: "Best Way to Incorporate and Optimize Distributed Training"
tags: [research, distributed-training, multi-gpu, performance-optimization, pytorch-ddp, training-infrastructure]
status: complete
last_updated: 2025-11-12
last_updated_by: Emery Gunselman
---

# Research: Best Way to Incorporate and Optimize Distributed Training

**Date**: 2025-11-12T20:43:11Z
**Researcher**: Emery Gunselman
**Git Commit**: db340949c4fe2e32a7e55702b9fe9ebb41a2a28e
**Branch**: feature--UPT
**Repository**: universal_simulator

## Research Question

What is the best way to incorporate and optimize distributed training into the Universal Physics Stack codebase?

## Executive Summary

The Universal Physics Stack **does not currently implement distributed training** (no PyTorch DDP, DataParallel, or Lightning). Training runs on **single-GPU VastAI instances** with multi-worker DataLoaders for data parallelism.

**Key Finding**: For the current workload (Burgers 32-dim, ~25min on A100, ~$1.25/run), **distributed training may not provide significant value** compared to existing optimizations. The codebase would benefit more from:
1. **Enabling existing optimizations** (latent cache precomputation - already built but disabled)
2. **Multi-dataset scaling** (training on all 11 PDEBench datasets simultaneously)
3. **Longer training runs** (current bottleneck is startup overhead, not training time)

**If distributed training is still desired**, the recommended approach is **PyTorch DDP with minimal architectural changes** (estimated 2-3 days implementation).

## Current State

### Training Architecture

**Single-Device Training**
- Device selection: `scripts/train.py:491-493` (hardcoded single device)
  ```python
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Using device: {device}")
  ```
- All models placed on single device: `scripts/train.py:1171`
- No rank/world_size awareness anywhere in codebase

**Multi-Stage Pipeline**
- Sequential stages: operator → diffusion → consistency → steady
- Orchestrated by `scripts/run_fast_to_sota.py` (Lines 350-1240)
- Each stage runs as subprocess via `scripts/train.py`
- Context passed via environment variables (`WANDB_CONTEXT_FILE`)

**Training Loop**
- Core loop: `scripts/train.py:599-868`
- Single-device batch processing
- Gradient accumulation support (for consistency distillation)
- EMA model tracking
- Early stopping with patience

### Existing Parallelization

**DataLoader Multi-Worker Parallelism**
- `src/ups/data/parallel_cache.py:270-322` - `build_parallel_latent_loader()`
- Default: `num_workers=4`, `pin_memory=True`, `persistent_workers=True`
- Multiprocessing start method: `spawn` (CUDA-safe)
- Documentation claims: "4-8× faster than num_workers=0" (`parallel_cache.py:11`)

**Performance Optimizations**
- **torch.compile**: `scripts/train.py:382-413` - Graph compilation with mode selection
- **Mixed Precision (AMP)**: `autocast` + `GradScaler` throughout `scripts/train.py`
- **Muon+AdamW Hybrid Optimizer**: `src/ups/training/muon_optimizer.py` (2-3x faster convergence claim)
- **Persistent Workers**: Avoids DataLoader worker respawn overhead
- **Prefetch Factor**: 4 batches ahead (`scripts/train.py:1155`)

### Configuration

**Key Training Parameters** (from `configs/train_burgers_32dim.yaml`):
```yaml
training:
  batch_size: 16         # Single GPU batch size
  num_workers: 4         # DataLoader workers
  pin_memory: true       # Faster CPU→GPU transfer
  persistent_workers: true
  amp: true              # Mixed precision
  compile: true          # torch.compile
  grad_clip: 1.0         # Gradient clipping
```

**No Distributed Config**
- No `world_size`, `rank`, `local_rank` parameters
- No `torch.distributed` backend configuration
- No DDP-specific settings

## Code References

### Critical Barriers to Distributed Training

1. **Hardcoded Single Device** - `scripts/train.py:491-493`
   ```python
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   ```

2. **No Rank Awareness** - All training code assumes rank 0

3. **Subprocess Orchestration** - `scripts/run_fast_to_sota.py:350-1240`
   - Launches `scripts/train.py` as subprocess
   - Not compatible with `torch.distributed.launch`

4. **Checkpoint Management** - `src/ups/utils/checkpoint_manager.py`
   - Saves only `model.state_dict()` (no optimizer/scheduler state)
   - No distributed checkpoint coordination

5. **WandB Context Sharing** - `src/ups/utils/wandb_context.py`
   - Shares context via environment variable file
   - Not DDP-aware (would create multiple runs)

### Key Files for Distributed Training Modifications

| File | Lines | What Needs Changing |
|------|-------|---------------------|
| `scripts/train.py` | 491-493 | Device initialization → distributed init |
| `scripts/train.py` | 1171 | Model placement → DDP wrapper |
| `scripts/train.py` | 599-868 | Training loop → add rank guards |
| `scripts/run_fast_to_sota.py` | 350-1240 | Subprocess launch → torchrun integration |
| `src/ups/data/parallel_cache.py` | 270-322 | DataLoader → DistributedSampler |
| `src/ups/utils/wandb_context.py` | N/A | Logging → rank 0 only |
| `src/ups/utils/checkpoint_manager.py` | N/A | Save/load → distributed coordination |

## Architecture Documentation

### Current Training Flow

```
run_fast_to_sota.py (orchestrator)
├── Validation phase
├── Training phase
│   ├── Stage 1: Operator (subprocess → train.py)
│   ├── Stage 2: Diffusion (subprocess → train.py)
│   ├── Stage 3: Consistency (subprocess → train.py)
│   └── Stage 4: Steady Prior (subprocess → train.py)
├── Small eval phase (subprocess → evaluate.py)
├── Full eval phase (subprocess → evaluate.py)
├── Gating checks
└── Leaderboard update
```

**Key Constraint**: Each stage runs as independent subprocess with no inter-process communication except checkpoints and environment variables.

### PyTorch Patterns Already in Use

**Positive Indicators** (DDP-compatible):
- ✅ `torch.compile` - Works with DDP (use `static_graph=True`)
- ✅ Mixed precision (AMP) - DDP-compatible
- ✅ Gradient clipping - Just needs `model.module` access in DDP
- ✅ EMA models - Can track DDP-wrapped model
- ✅ State dict checkpointing - Add `model.module.state_dict()` for DDP

**Potential Issues**:
- ❌ Subprocess orchestration - Need to replace with `torchrun`
- ❌ WandB logging - Need rank guards
- ❌ No DistributedSampler usage
- ❌ No process group initialization

## Historical Context (from thoughts/)

### Training Optimization Plans

1. **Training Overhead Optimization** (`thoughts/shared/plans/2025-10-28-training-overhead-optimization.md`)
   - **Goal**: Reduce VastAI startup from 14-25min to 6-9min (60% reduction)
   - **Method**: GPU cache precomputation, parallel downloads, persistent cache
   - **Status**: Already implemented but disabled (latent cache features exist in `parallel_cache.py`)
   - **Impact**: $0.30-0.50 savings per run
   - **Relevance**: This addresses the actual bottleneck (startup, not training)

2. **Checkpoint Resume System** (`thoughts/shared/plans/2025-10-28-checkpoint-resume-system.md`)
   - Intelligent checkpoint resumption for interrupted training
   - Auto-resume capabilities with `--auto-resume` flag
   - Stage tracking via `stage_status.json`
   - **Relevance**: More valuable than distributed training for current workload

3. **Muon Optimizer Integration** (`thoughts/shared/plans/2025-10-30-muon-optimizer-integration.md`)
   - **Status**: Already integrated
   - Hybrid Muon+AdamW approach
   - Claims: 2-3x faster convergence, 35% faster optimizer step
   - **Relevance**: Already achieving speedups without distributed training

4. **PDEBench Multi-Dataset Scaling** (`thoughts/shared/plans/2025-11-07-pdebench-multi-dataset-scaling.md`)
   - Train on all 11 PDEBench datasets simultaneously
   - Complexity-based curriculum
   - **Relevance**: THIS is where distributed training could add value (large multi-task batch)

### No Prior Distributed Training Work

**Search Results**: No documents in `thoughts/` directory specifically about:
- Multi-GPU training strategies
- PyTorch DDP implementation
- Distributed training architecture
- Multi-node training on VastAI

**Interpretation**: Team has prioritized single-GPU optimization over distributed training.

## Recommendations

### Strategic Assessment: Do You Need Distributed Training?

**Current Workload Analysis** (Burgers 32-dim baseline):
- Training time: ~25 minutes on A100
- Cost: ~$1.25 per run
- Bottleneck: **Startup overhead (14-25min)**, not training time
- Improvement potential from DDP: ~10-30% speedup (assumes 2-4 GPUs, linear scaling)
- Actual savings: ~2-7 minutes, ~$0.20-0.50

**Latent Cache Optimization** (already built, just disabled):
- Startup time reduction: 60% (14-25min → 6-9min)
- Savings per run: $0.30-0.50
- Implementation effort: **Enable existing feature** (1 day)

**Verdict**: For current workload, **latent cache optimization > distributed training**.

### When Distributed Training DOES Make Sense

1. **Multi-Dataset Training** (PDEBench scaling plan)
   - Training on 11 datasets simultaneously
   - Larger effective batch sizes needed
   - **DDP could enable 2-4x larger batches** across GPUs

2. **Longer Training Runs** (if you scale to 192-dim latent, 100+ epochs)
   - Current: Training time < startup time
   - At scale: Training time > startup time → DDP provides value

3. **Hyperparameter Sweeps**
   - Multiple runs with different configs
   - **Better handled by parallel VastAI instances** (current approach)
   - DDP doesn't help here

### Recommended Approach: PyTorch DDP (If Needed)

**Why DDP over Alternatives:**
- ✅ **PyTorch DDP**: Minimal code changes, efficient, well-tested
- ❌ PyTorch Lightning: Large refactor, overkill for current needs
- ❌ Horovod: Extra dependency, not PyTorch-native
- ❌ DataParallel: Deprecated, slower than DDP

**Implementation Roadmap** (2-3 days):

#### Phase 1: Core DDP Integration (1 day)

**Step 1.1: Distributed Initialization**
```python
# scripts/train.py - Replace lines 491-493
import torch.distributed as dist

def setup_distributed():
    """Initialize distributed training."""
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        return device, True  # device, is_distributed
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return device, False

device, is_distributed = setup_distributed()
rank = dist.get_rank() if is_distributed else 0
world_size = dist.get_world_size() if is_distributed else 1
```

**Step 1.2: DDP Model Wrapping**
```python
# scripts/train.py - After line 1171
if is_distributed:
    operator = torch.nn.parallel.DistributedDataParallel(
        operator,
        device_ids=[local_rank],
        output_device=local_rank,
        static_graph=True  # Required for torch.compile
    )
```

**Step 1.3: DistributedSampler**
```python
# src/ups/data/parallel_cache.py - Modify build_parallel_latent_loader()
from torch.utils.data.distributed import DistributedSampler

def build_parallel_latent_loader(..., is_distributed=False, rank=0, world_size=1):
    sampler = DistributedSampler(dataset, rank=rank, world_size=world_size) if is_distributed else None
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),  # Only shuffle if no sampler
        ...
    )
```

**Step 1.4: Rank Guards for Logging**
```python
# src/ups/utils/wandb_context.py
def log_training_metric(self, stage, metric, value, step):
    if dist.is_initialized() and dist.get_rank() != 0:
        return  # Only rank 0 logs
    # ... existing logging code
```

#### Phase 2: Checkpoint & Orchestration (1 day)

**Step 2.1: DDP-Aware Checkpoints**
```python
# Saving (rank 0 only)
if rank == 0:
    state_dict = model.module.state_dict() if is_distributed else model.state_dict()
    torch.save(state_dict, path)

# Loading (all ranks)
state_dict = torch.load(path, map_location=device)
if is_distributed:
    model.module.load_state_dict(state_dict)
else:
    model.load_state_dict(state_dict)
```

**Step 2.2: Replace Subprocess Launch with torchrun**
```python
# scripts/run_fast_to_sota.py - Modify subprocess calls
if num_gpus > 1:
    cmd = [
        "torchrun",
        f"--nproc_per_node={num_gpus}",
        "scripts/train.py",
        "--config", config_path,
        "--stage", stage
    ]
else:
    cmd = ["python", "scripts/train.py", "--config", config_path, "--stage", stage]

subprocess.run(cmd, check=True)
```

#### Phase 3: VastAI Multi-GPU Integration (0.5 days)

**Step 3.1: Update Instance Selection**
```python
# scripts/vast_launch.py - Add multi-GPU filter
gpu_filter = f'gpu_ram >= 48 reliability > 0.95 num_gpus={num_gpus} disk_space >= 64'
```

**Step 3.2: Update Onstart Script**
```bash
# .vast/onstart.sh - Use torchrun
torchrun \
  --nproc_per_node=2 \  # Or 4 for 4-GPU instances
  scripts/run_fast_to_sota.py \
  --train-config /workspace/universal_simulator/configs/train_burgers_32dim.yaml \
  ...
```

#### Phase 4: Testing & Validation (0.5 days)

**Test Cases:**
1. Single-GPU run (backward compatibility)
2. 2-GPU DDP run (local)
3. 4-GPU DDP run (VastAI)
4. Checkpoint save/load with DDP
5. WandB logging (verify single run, not duplicates)
6. Gradient accumulation with DDP

**Expected Speedups:**
- 2 GPUs: 1.7-1.9x (90-95% scaling efficiency)
- 4 GPUs: 3.2-3.6x (80-90% scaling efficiency)
- Not 2x/4x due to communication overhead

### Alternative Optimizations (Higher ROI)

**Option 1: Enable Latent Cache Precomputation** (1 day)
- Modify configs: Enable `latent_cache_dir` in training configs
- Update VastAI onstart: Run `precompute_latent_cache.py` before training
- Expected impact: 60% startup time reduction, $0.30-0.50 savings per run
- **Recommendation: Do this FIRST** (already built, just disabled)

**Option 2: Increase Batch Size** (0 days)
- Current: `batch_size=16` (conservative for 48GB A100)
- Potential: `batch_size=32` or higher (test OOM limits)
- Expected impact: 10-20% speedup (better GPU utilization)
- No code changes, just config tuning

**Option 3: Multi-Instance Parallel Runs** (current approach)
- Launch 4 separate VastAI instances with different hyperparameters
- No code changes needed
- Better for exploration than single multi-GPU run
- **Already the team's approach** (good!)

## Implementation Priority Ranking

**For Current Workload (Burgers 32-dim, 25min runs):**

1. **Enable Latent Cache** (1 day, $0.30-0.50/run savings) ⭐⭐⭐⭐⭐
2. **Increase Batch Size** (0 days, 10-20% speedup) ⭐⭐⭐⭐
3. **Checkpoint Resume System** (2 days, robustness) ⭐⭐⭐⭐
4. **Distributed Training** (3 days, 10-30% speedup) ⭐⭐

**For Multi-Dataset Training (PDEBench scaling):**

1. **Enable Latent Cache** (critical for 11 datasets) ⭐⭐⭐⭐⭐
2. **Distributed Training** (enables larger effective batches) ⭐⭐⭐⭐⭐
3. **Increase Batch Size** (per-GPU batch size) ⭐⭐⭐⭐
4. **Multi-Instance Parallel** (run multiple scaling experiments) ⭐⭐⭐

## Open Questions

1. **What is the actual GPU utilization during training?**
   - If <70%, distributed training won't help (underutilized GPUs)
   - Profile with: `nvidia-smi dmon -s u -c 100` during training

2. **What is the current batch size headroom?**
   - Test: Increase `batch_size` until OOM
   - If can double batch size, DDP may not be needed

3. **Is the bottleneck data loading or compute?**
   - Profile with: `torch.profiler` or `py-spy`
   - If data loading, increase `num_workers` (not DDP)

4. **What is the target scale for multi-dataset training?**
   - 11 datasets × current batch size = need DDP?
   - Or can you run 11 separate single-GPU runs?

5. **Do you plan to use multi-GPU VastAI instances?**
   - Cost: 2×A100 (~$1.50/hr) vs 1×A100 ($0.80/hr)
   - Availability: Multi-GPU instances rarer, higher reliability?

## Related Research

- `thoughts/shared/plans/2025-10-28-training-overhead-optimization.md` - Latent cache optimization
- `thoughts/shared/plans/2025-10-28-checkpoint-resume-system.md` - Checkpoint resumption
- `thoughts/shared/plans/2025-10-30-muon-optimizer-integration.md` - Optimizer speedups (already done)
- `thoughts/shared/plans/2025-11-07-pdebench-multi-dataset-scaling.md` - Multi-dataset scaling (where DDP helps)

## Next Steps

**Recommended Action Plan:**

1. **Profile Current Training** (1-2 hours)
   ```bash
   # Add to scripts/train.py
   python -m torch.utils.bottleneck scripts/train.py --config configs/train_burgers_32dim.yaml --stage operator --epochs 1
   ```
   - Check GPU utilization, data loading time, compute time

2. **Enable Latent Cache** (1 day)
   - Update `configs/train_burgers_32dim.yaml`: Set `latent_cache_dir`
   - Test VastAI run with cache enabled
   - Measure startup time reduction

3. **Test Batch Size Scaling** (1 hour)
   - Try `batch_size=24, 32, 48` until OOM
   - Measure speedup per batch size increase

4. **Decision Point**: After steps 1-3, reassess if DDP is still needed
   - If GPU util <70%: Optimize compute, not distribute
   - If batch size can double: May not need DDP
   - If latent cache cuts startup 60%: Training time less critical

5. **If DDP Still Needed**: Implement Phase 1-4 roadmap above (3 days)

## Conclusion

**Best Way to Incorporate Distributed Training:**
- **Short Answer**: PyTorch DDP with 3-day implementation roadmap
- **Better Answer**: Enable latent cache first (60% startup reduction, 1 day) → Then reassess if DDP is needed
- **Best Answer**: Profile → Optimize → Scale only if needed

The codebase is **well-positioned** for DDP (clean PyTorch, torch.compile, AMP), but the **current workload doesn't justify it**. Focus on:
1. Enabling existing optimizations (latent cache)
2. Profiling to find real bottlenecks
3. Scaling batch size before scaling GPUs
4. Reserve DDP for multi-dataset training phase

**Total Estimated Effort:**
- Profiling + latent cache + batch tuning: **2 days**
- DDP implementation (if needed): **+3 days**
- Total: **2-5 days** depending on findings
