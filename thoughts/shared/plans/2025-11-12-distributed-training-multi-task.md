# Distributed Training for Multi-Task PDEBench Implementation Plan

## Overview

This plan implements production-ready PyTorch Distributed Data Parallel (DDP) training for the Universal Physics Stack, optimized for full PDEBench suite scaling (11 tasks). The implementation balances **memory capacity** (enable larger batches/more tasks) and **training speed** (reduce wall-clock time), with flexible GPU support (2-GPU and 4-GPU VastAI instances).

**Key Value Propositions:**
1. **Memory Capacity**: 2×A100 = 160GB, 4×A100 = 320GB (vs 80GB single-GPU)
2. **Larger Per-GPU Batches**: 4 → 8-12 (reduce gradient accumulation overhead)
3. **Multi-Task Scaling**: 2-task baseline → 11-task PDEBench suite
4. **Training Speed**: 1.7-3.6× speedup (depending on GPU count and scaling efficiency)

## Current State Analysis

### Existing Infrastructure

**Training Architecture** (`scripts/train.py:491-493, 1171`):
- Single-device initialization: `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`
- No distributed process group initialization
- No DDP model wrapping
- Subprocess-based orchestration incompatible with `torchrun`

**Multi-Task Training** (`src/ups/data/latent_pairs.py:753-843`):
- Uses simple `ConcatDataset` with NO balanced sampling (despite config `strategy: "balanced"`)
- Task metadata tracked via `LatentPair.task_name` (line 260)
- Per-task metrics logged to WandB (optional, `scripts/train.py:788-851`)
- No `DistributedSampler` support

**Memory Constraints** (`configs/train_pdebench_2task_baseline.yaml:50-57`):
- Batch size: **4** (reduced from 8 to fit 80GB A100 with UPT inverse losses)
- Gradient accumulation: **12 steps** (effective batch = 48)
- Parallel encoding: **disabled** (`num_workers=0` to avoid OOM)
- UPT inverse losses: ~5-10 GB memory overhead per batch

**Data Loading** (`src/ups/data/latent_pairs.py:732-741`):
```python
loader_kwargs = {
    "batch_size": batch_size,
    "shuffle": True,  # No sampler
    "collate_fn": collate_fn,
    "num_workers": num_workers,
    "pin_memory": pin_memory,
    "persistent_workers": num_workers > 0,
}
```

### Key Discoveries

1. **Memory is the Primary Bottleneck**: Config explicitly documents batch size reduction from 8 → 4 due to "UPT inverse losses" on 80GB A100
2. **No Existing DDP Infrastructure**: Clean slate implementation required
3. **Multi-Task Balancing Not Implemented**: Config has `task_sampling.strategy: "balanced"` but code uses naive `ConcatDataset`
4. **Task-Level Metrics Already Tracked**: Infrastructure exists for per-task logging (`task_names` in batch)
5. **Conservation Checks Disabled**: Commit `db34094` notes "multi-task crash" with conservation penalty
6. **Extensive OOM Recovery**: 4 distinct OOM catch points across training stages (`scripts/train.py:779-1326`)
7. **Spawn-based Multiprocessing**: Already uses `mp.set_start_method("spawn")` for CUDA safety (NOT for distributed training)

## Desired End State

### Functional Requirements

**After Phase 5 completion:**

1. **Flexible GPU Support**:
   - Config parameter: `training.num_gpus: 2` or `4`
   - Automatic DDP initialization based on `RANK` environment variable
   - Backward compatible: Single-GPU mode still works (`num_gpus: 1`)

2. **Task-Aware Distributed Sampling**:
   - Custom `MultiTaskDistributedSampler` ensures balanced task distribution across ranks
   - Each GPU sees same proportion of each task per epoch
   - Maintains statistical parity for multi-task training

3. **Memory Capacity Improvements**:
   - Per-GPU batch size: 8-12 (vs current 4)
   - Gradient accumulation: 4-6 steps (vs current 12)
   - Parallel encoding re-enabled: `num_workers=8`
   - Total effective batch size: 96-192 (2-GPU) or 192-384 (4-GPU)

4. **Training Speed Improvements**:
   - Expected speedup: 1.7-1.9× (2-GPU), 3.2-3.6× (4-GPU)
   - Reduced startup time via parallel data loading
   - Better GPU utilization (less memory-constrained)

5. **Production Features**:
   - Rank guards for WandB logging (only rank 0)
   - DDP-aware checkpointing (`model.module.state_dict()`)
   - Per-rank OOM recovery and error handling
   - Task-level metrics aggregation across ranks
   - VastAI onstart script generation with `torchrun`

### Verification Criteria

**To verify the implementation is complete:**

1. **Single-GPU Backward Compatibility**:
   ```bash
   python scripts/train.py --config configs/train_pdebench_2task_baseline.yaml --stage operator --epochs 1
   # Should complete without DDP warnings
   ```

2. **2-GPU Training**:
   ```bash
   torchrun --nproc_per_node=2 scripts/train.py --config configs/train_pdebench_2task_baseline_ddp.yaml --stage operator --epochs 1
   # Should show: rank 0 and rank 1 logs, balanced task sampling, single WandB run
   ```

3. **11-Task Scaling**:
   ```bash
   torchrun --nproc_per_node=4 scripts/train.py --config configs/train_pdebench_11task_ddp.yaml --stage operator --epochs 5
   # Should complete with balanced task distribution, larger batch size, faster training
   ```

4. **VastAI Integration**:
   ```bash
   python scripts/vast_launch.py launch --config configs/train_pdebench_11task_ddp.yaml --auto-shutdown
   # Should provision 4×A100 instance, generate torchrun onstart script, complete training
   ```

5. **Performance Benchmarks**:
   - 2-GPU training: 1.7-1.9× faster than single-GPU baseline
   - 4-GPU training: 3.2-3.6× faster than single-GPU baseline
   - Per-GPU batch size increased to 8-12
   - Gradient accumulation reduced to 4-6 steps

## What We're NOT Doing

**Explicitly out of scope to prevent scope creep:**

1. **PyTorch Lightning Migration**: Sticking with native DDP (minimal refactor)
2. **Gradient Checkpointing**: Already complex enough without activation checkpointing
3. **Model Parallelism**: Only data parallelism (DDP), not tensor/pipeline parallelism
4. **Multi-Node Training**: Single multi-GPU instance only (not cluster)
5. **Dynamic GPU Count**: Fixed at launch time (no auto-scaling)
6. **FSDP (Fully Sharded Data Parallel)**: Model fits in single GPU memory, no need
7. **Mixed-Precision Optimizations Beyond Existing**: Keep current AMP setup
8. **Task Curriculum Learning**: Balanced sampling only, no dynamic weighting
9. **Distributed Hyperparameter Sweeps**: Single run per instance (use parallel instances)
10. **ZeRO Optimizer**: Not needed for current model size

## Implementation Approach

### High-Level Strategy

**Core Principle**: Minimal code changes to `scripts/train.py` and `src/ups/data/latent_pairs.py`, maximum compatibility with existing infrastructure.

**Phased Implementation**:
1. **Phase 1**: Core DDP infrastructure (distributed init, model wrapping, rank guards)
2. **Phase 2**: Task-aware distributed sampling (custom sampler for balanced multi-task)
3. **Phase 3**: Memory optimization and batch scaling (leverage increased capacity)
4. **Phase 4**: VastAI integration (torchrun launcher, flexible GPU config)
5. **Phase 5**: Production hardening (checkpointing, testing, benchmarks)

**Key Design Decisions**:
- **PyTorch DDP over Lightning**: Native PyTorch, less refactor, production-ready
- **Custom Sampler over WeightedRandomSampler**: Task-aware, respects rank/world_size
- **Config-Driven GPU Count**: `training.num_gpus` parameter, backward compatible
- **Rank 0 Logging Only**: Clean WandB integration, no run proliferation
- **Static Graphs for torch.compile**: DDP-compatible compilation with `static_graph=True`

---

## Phase 1: Core DDP Infrastructure

### Overview

Implement basic PyTorch DDP support with distributed initialization, DDP model wrapping, and rank guards for logging. Test with 2-GPU setup on single-task Burgers baseline before multi-task complexity.

### Changes Required

#### 1. Distributed Initialization

**File**: `scripts/train.py`

**Current** (lines 491-493):
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
```

**New** (replace lines 491-493):
```python
import torch.distributed as dist

def setup_distributed():
    """Initialize distributed training if RANK environment variable is set."""
    if "RANK" in os.environ:
        # torchrun sets these automatically
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")

        if rank == 0:
            print(f"Distributed training initialized: {world_size} GPUs")

        return device, True, rank, world_size, local_rank
    else:
        # Single-GPU mode (backward compatible)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device} (single-GPU mode)")
        return device, False, 0, 1, None

device, is_distributed, rank, world_size, local_rank = setup_distributed()
```

#### 2. DDP Model Wrapping

**File**: `scripts/train.py`

**Current** (line 1171):
```python
operator = operator.to(device)
```

**New** (after line 1171):
```python
operator = operator.to(device)

# Wrap with DDP if distributed
if is_distributed:
    from torch.nn.parallel import DistributedDataParallel as DDP
    operator = DDP(
        operator,
        device_ids=[local_rank],
        output_device=local_rank,
        static_graph=True,  # Required for torch.compile compatibility
        find_unused_parameters=False,  # All params used in operator
    )
    if rank == 0:
        print(f"Operator wrapped with DDP on device {local_rank}")
```

**Repeat for diffusion, distill, steady models** (similar pattern at their initialization points)

#### 3. Rank Guards for WandB Logging

**File**: `src/ups/utils/wandb_context.py`

**Modify all logging methods** (e.g., `log_training_metric`, `log_eval_summary`, `update_config`):

**Add rank guard at top of each method**:
```python
def log_training_metric(self, stage: str, metric: str, value: float, step: int):
    """Log a training metric (time series)."""
    # Only rank 0 logs to WandB
    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return

    # ... existing logging code ...
```

**Files to update**:
- `src/ups/utils/wandb_context.py` (all logging methods)
- `scripts/train.py` (print statements → conditional on rank == 0)

#### 4. Configuration Parameter

**File**: Add new training parameter to all configs

**Example** (`configs/train_pdebench_2task_baseline_ddp.yaml`):
```yaml
training:
  # Distributed training
  num_gpus: 2  # Number of GPUs (1 = single-GPU, 2/4 = DDP)

  # Existing parameters...
  batch_size: 8  # Per-GPU batch size (increased from 4)
  accum_steps: 6  # Reduced from 12 (effective batch = 8*6*2 = 96)
```

### Success Criteria

#### Automated Verification

- [ ] **Single-GPU backward compatibility**: `python scripts/train.py --config configs/train_pdebench_2task_baseline.yaml --stage operator --epochs 1` completes without errors
- [ ] **DDP initialization**: `torchrun --nproc_per_node=2 scripts/train.py --config configs/train_pdebench_2task_baseline_ddp.yaml --stage operator --epochs 1` shows "Distributed training initialized: 2 GPUs" (rank 0 only)
- [ ] **Model forward pass**: Training loop executes without DDP errors (no "accessing tensor output" errors)
- [ ] **Checkpoint save/load**: `checkpoints/op_latest.ckpt` saved by rank 0 only, loadable by all ranks
- [ ] **WandB logging**: Only 1 run created (not 2), metrics logged from rank 0 only

#### Manual Verification

- [ ] Verify both GPUs are utilized: `nvidia-smi dmon -s u -c 10` shows >70% utilization on both GPUs
- [ ] Check WandB dashboard: Only 1 run, no duplicate metrics
- [ ] Verify training loss decreases normally (no DDP-related convergence issues)
- [ ] Confirm checkpoint file size is correct (not doubled)

**Implementation Note**: After completing this phase and all automated verification passes, pause here for manual confirmation from the human that the manual testing was successful before proceeding to Phase 2.

---

## Phase 2: Task-Aware Distributed Sampling

### Overview

Implement custom `MultiTaskDistributedSampler` to ensure balanced task distribution across ranks. This is critical for 11-task PDEBench scaling - naive `DistributedSampler` would treat `ConcatDataset` as flat index list, leading to imbalanced task distribution per rank.

### Changes Required

#### 1. Create MultiTaskDistributedSampler

**File**: `src/ups/data/task_samplers.py` (NEW FILE)

**Content**:
```python
"""Task-aware distributed samplers for multi-task training."""

from __future__ import annotations

import math
from typing import Iterator

import torch
import torch.distributed as dist
from torch.utils.data import Sampler


class MultiTaskDistributedSampler(Sampler[int]):
    """
    Distributed sampler that maintains balanced task sampling across ranks.

    Given a list of per-task dataset sizes, ensures each rank sees the same
    proportion of each task per epoch. Respects distributed training semantics
    (rank, world_size) while maintaining task balance.

    Args:
        task_sizes: List of dataset sizes per task (e.g., [1000, 800, 1200] for 3 tasks)
        num_replicas: Number of processes (default: world_size)
        rank: Rank of current process (default: current rank)
        shuffle: Whether to shuffle samples within each task (default: True)
        seed: Random seed for shuffling (default: 0)
        drop_last: Whether to drop incomplete batches (default: False)

    Example:
        >>> task_sizes = [1000, 800, 1200]  # 3 tasks
        >>> sampler = MultiTaskDistributedSampler(task_sizes, num_replicas=2, rank=0)
        >>> len(sampler)  # 1500 (total 3000 / 2 replicas)
        1500
    """

    def __init__(
        self,
        task_sizes: list[int],
        num_replicas: int | None = None,
        rank: int | None = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size() if dist.is_initialized() else 1
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank() if dist.is_initialized() else 0

        if rank >= num_replicas or rank < 0:
            raise ValueError(f"Invalid rank {rank}, should be in [0, {num_replicas-1}]")

        self.task_sizes = task_sizes
        self.num_tasks = len(task_sizes)
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed

        # Compute per-task samples per replica
        self.per_task_num_samples = []
        for size in task_sizes:
            if self.drop_last and size % self.num_replicas != 0:
                num_samples = math.ceil((size - self.num_replicas) / self.num_replicas)
            else:
                num_samples = math.ceil(size / self.num_replicas)
            self.per_task_num_samples.append(num_samples)

        self.total_size = sum(self.per_task_num_samples) * self.num_replicas
        self.num_samples = sum(self.per_task_num_samples)

        # Compute task offsets in ConcatDataset (cumulative sizes)
        self.task_offsets = [0]
        for size in task_sizes[:-1]:
            self.task_offsets.append(self.task_offsets[-1] + size)

    def __iter__(self) -> Iterator[int]:
        """Generate indices for this rank."""
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        indices = []

        # For each task, generate indices and select this rank's subset
        for task_idx, (size, offset, num_samples) in enumerate(
            zip(self.task_sizes, self.task_offsets, self.per_task_num_samples)
        ):
            # Generate task-local indices [0, size)
            if self.shuffle:
                task_indices = torch.randperm(size, generator=g).tolist()
            else:
                task_indices = list(range(size))

            # Pad if needed (for balanced distribution)
            if not self.drop_last:
                padding_size = num_samples * self.num_replicas - len(task_indices)
                if padding_size > 0:
                    task_indices += task_indices[:padding_size]
            else:
                # Drop tail to make evenly divisible
                task_indices = task_indices[:num_samples * self.num_replicas]

            # Select this rank's subset (every num_replicas-th element)
            rank_indices = task_indices[self.rank : len(task_indices) : self.num_replicas]

            # Convert to global indices (add task offset)
            rank_indices = [idx + offset for idx in rank_indices]

            indices.extend(rank_indices)

        # Shuffle across tasks if requested (maintains per-task balance in expectation)
        if self.shuffle:
            # Shuffle with task-preserving property (interleave tasks)
            perm = torch.randperm(len(indices), generator=g).tolist()
            indices = [indices[i] for i in perm]

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for this sampler (for shuffling)."""
        self.epoch = epoch
```

#### 2. Integrate MultiTaskDistributedSampler

**File**: `src/ups/data/latent_pairs.py`

**Modify `build_latent_pair_loader()`** (lines 711-843):

**Add imports** (top of file):
```python
from ups.data.task_samplers import MultiTaskDistributedSampler
```

**Replace DataLoader creation** (lines 732-741):

**Current**:
```python
loader_kwargs = {
    "batch_size": batch_size,
    "shuffle": True,
    "collate_fn": collate_fn,
    "num_workers": num_workers,
    "pin_memory": pin_memory,
    "persistent_workers": num_workers > 0,
}
if prefetch_factor is not None and num_workers > 0:
    loader_kwargs["prefetch_factor"] = prefetch_factor

return DataLoader(combined, **loader_kwargs)
```

**New**:
```python
# Check if distributed training is active
import torch.distributed as dist
is_distributed = dist.is_initialized()

# For multi-task, use task-aware sampler
if len(task_list) > 1 and is_distributed:
    # Extract per-task sizes
    task_sizes = [len(ds) for ds in latent_datasets]
    sampler = MultiTaskDistributedSampler(
        task_sizes=task_sizes,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=True,
        seed=train_cfg.get("seed", 0),
        drop_last=False,
    )
    shuffle = False  # Sampler handles shuffling
elif is_distributed:
    # Single-task distributed training
    from torch.utils.data.distributed import DistributedSampler
    sampler = DistributedSampler(
        combined,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=True,
        seed=train_cfg.get("seed", 0),
    )
    shuffle = False
else:
    # Single-GPU mode
    sampler = None
    shuffle = True

loader_kwargs = {
    "batch_size": batch_size,
    "shuffle": shuffle,
    "sampler": sampler,
    "collate_fn": collate_fn,
    "num_workers": num_workers,
    "pin_memory": pin_memory,
    "persistent_workers": num_workers > 0,
}
if prefetch_factor is not None and num_workers > 0:
    loader_kwargs["prefetch_factor"] = prefetch_factor

return DataLoader(combined, **loader_kwargs)
```

#### 3. Set Epoch for Sampler (Shuffling)

**File**: `scripts/train.py`

**Add before training loop** (e.g., line 606):
```python
# For distributed training, set epoch for sampler (enables shuffling)
if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
    train_loader.sampler.set_epoch(epoch)
```

**Repeat for diffusion and distill stages**

#### 4. Validation: Task Distribution Check

**File**: `scripts/validate_distributed_sampling.py` (NEW FILE)

**Content**:
```python
"""Validate task distribution across ranks for distributed multi-task training."""

import torch
import torch.distributed as dist
from collections import Counter
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
    train_loader = build_latent_pair_loader(cfg, split="train")

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
            print(f"  Balance: {'✓ GOOD' if std / mean < 0.1 else '✗ IMBALANCED'}")

        print("\n" + "=" * 60)

    dist.destroy_process_group()


if __name__ == "__main__":
    import sys
    validate_task_distribution(sys.argv[1])
```

### Success Criteria

#### Automated Verification

- [ ] **MultiTaskDistributedSampler creation**: `pytest tests/unit/test_task_samplers.py -v` passes (test sampler logic)
- [ ] **2-task balanced distribution**: `torchrun --nproc_per_node=2 scripts/validate_distributed_sampling.py configs/train_pdebench_2task_baseline_ddp.yaml` shows std/mean < 10% for both tasks
- [ ] **11-task balanced distribution**: `torchrun --nproc_per_node=4 scripts/validate_distributed_sampling.py configs/train_pdebench_11task_ddp.yaml` shows std/mean < 10% for all 11 tasks
- [ ] **Training loop**: `torchrun --nproc_per_node=2 scripts/train.py --config configs/train_pdebench_2task_baseline_ddp.yaml --stage operator --epochs 2` completes without errors
- [ ] **Per-task metrics**: WandB run shows separate metrics for each task (advection1d/loss, darcy2d/loss)

#### Manual Verification

- [ ] Manually inspect task distribution output - verify counts are balanced across ranks
- [ ] Check WandB dashboard - verify per-task metrics are logged correctly
- [ ] Run with 11 tasks - verify all tasks are represented in each rank's batches
- [ ] Verify no task is dropped or over-represented (visual inspection)

**Implementation Note**: After completing this phase and all automated verification passes, pause here for manual confirmation from the human that the manual testing was successful before proceeding to Phase 3.

---

## Phase 3: Memory Optimization & Batch Scaling

### Overview

Leverage increased memory capacity from multi-GPU setup to:
1. Increase per-GPU batch size (4 → 8 or 12)
2. Reduce gradient accumulation (12 → 4 or 6)
3. Re-enable parallel encoding (`num_workers > 0`)
4. Profile and tune memory usage

### Changes Required

#### 1. Update Config Parameters

**File**: `configs/train_pdebench_2task_baseline_ddp.yaml` (NEW FILE, based on baseline)

**Changes**:
```yaml
training:
  # Distributed training
  num_gpus: 2  # 2×A100 = 160GB total

  # Memory optimization (increased from single-GPU baseline)
  batch_size: 8            # Per-GPU (was 4, now 8 = 2× increase)
  accum_steps: 6           # Reduced from 12 (effective batch = 8*6*2 = 96)

  # Re-enable parallel encoding (was 0 for single-GPU OOM)
  num_workers: 8           # 8 workers per GPU
  use_parallel_encoding: true
  pin_memory: true
  prefetch_factor: 4

  # Keep existing optimizations
  cache_dir: data/latent_cache
  latent_cache_dtype: float32
  amp: true
  compile: true
  compile_mode: reduce-overhead
```

**Create separate configs for 4-GPU**:

**File**: `configs/train_pdebench_11task_ddp.yaml` (NEW FILE)

```yaml
data:
  task: [advection1d, darcy2d, burgers1d, diff_react1d, swe2d, ns2d_cond, comp_ns2d, react_diff2d, diff_sorp2d, shallow_water2d_varied, ns2d_turb]
  # All 11 PDEBench tasks

training:
  num_gpus: 4              # 4×A100 = 320GB total
  batch_size: 12           # Per-GPU (12*4 = 48 samples per step)
  accum_steps: 4           # Effective batch = 12*4*4 = 192

  num_workers: 8
  use_parallel_encoding: true
  pin_memory: true
  prefetch_factor: 4

  # Larger model for 11-task capacity
latent:
  dim: 192                 # Increased from 128
  tokens: 192

operator:
  pdet:
    input_dim: 192
    hidden_dim: 576        # 3× latent (was 384)
    depth: 16              # Deeper for 11 tasks (was 12)
```

#### 2. Memory Profiling Script

**File**: `scripts/profile_memory.py` (NEW FILE)

**Content**:
```python
"""Profile GPU memory usage during training."""

import torch
import torch.distributed as dist
from ups.data.latent_pairs import build_latent_pair_loader
from ups.models.latent_operator import build_operator
from ups.utils.config_loader import load_config


def profile_memory(config_path: str, stage: str = "operator"):
    """Profile memory usage for a given config and stage."""
    # Setup
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load config and build model
    cfg = load_config(config_path)
    train_loader = build_latent_pair_loader(cfg, split="train")
    operator = build_operator(cfg).to(device)

    # Wrap with DDP if distributed
    if dist.is_initialized():
        from torch.nn.parallel import DistributedDataParallel as DDP
        operator = DDP(operator, device_ids=[local_rank], output_device=local_rank)

    # Profile forward + backward pass
    torch.cuda.reset_peak_memory_stats()

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

        # Print memory stats
        if not dist.is_initialized() or dist.get_rank() == 0:
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            peak = torch.cuda.max_memory_allocated() / 1e9

            print(f"\nBatch {i+1}:")
            print(f"  Allocated: {allocated:.2f} GB")
            print(f"  Reserved:  {reserved:.2f} GB")
            print(f"  Peak:      {peak:.2f} GB")

    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    import sys
    profile_memory(sys.argv[1])
```

#### 3. Batch Size Tuning Script

**File**: `scripts/tune_batch_size.py` (NEW FILE)

**Content**:
```python
"""Find optimal per-GPU batch size before OOM."""

import torch
import torch.distributed as dist
from ups.data.latent_pairs import build_latent_pair_loader
from ups.models.latent_operator import build_operator
from ups.utils.config_loader import load_config


def tune_batch_size(config_path: str, start: int = 4, end: int = 32, step: int = 4):
    """Binary search for max batch size that doesn't OOM."""
    # Setup distributed
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = dist.get_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank = 0
        device = torch.device("cuda")

    cfg = load_config(config_path)

    max_working = 0

    for batch_size in range(start, end + 1, step):
        if rank == 0:
            print(f"\n{'='*60}")
            print(f"Testing batch_size={batch_size}")
            print('='*60)

        # Update config
        cfg["training"]["batch_size"] = batch_size

        try:
            # Build model and data
            train_loader = build_latent_pair_loader(cfg, split="train")
            operator = build_operator(cfg).to(device)

            if dist.is_initialized():
                from torch.nn.parallel import DistributedDataParallel as DDP
                operator = DDP(operator, device_ids=[local_rank], output_device=local_rank)

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
                raise

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Maximum batch size: {max_working}")
        print('='*60)

    if dist.is_initialized():
        dist.destroy_process_group()

    return max_working


if __name__ == "__main__":
    import sys
    tune_batch_size(sys.argv[1])
```

### Success Criteria

#### Automated Verification

- [ ] **Memory profiling**: `torchrun --nproc_per_node=2 scripts/profile_memory.py configs/train_pdebench_2task_baseline_ddp.yaml` shows peak memory < 70GB per GPU
- [ ] **Batch size tuning**: `torchrun --nproc_per_node=2 scripts/tune_batch_size.py configs/train_pdebench_2task_baseline_ddp.yaml` finds max batch size ≥ 8
- [ ] **Training with larger batches**: `torchrun --nproc_per_node=2 scripts/train.py --config configs/train_pdebench_2task_baseline_ddp.yaml --stage operator --epochs 5` completes without OOM
- [ ] **11-task training**: `torchrun --nproc_per_node=4 scripts/train.py --config configs/train_pdebench_11task_ddp.yaml --stage operator --epochs 5` completes successfully
- [ ] **Parallel encoding**: num_workers=8 doesn't cause CUDA serialization errors

#### Manual Verification

- [ ] Monitor GPU memory during training: `nvidia-smi dmon -s mu -c 100` shows consistent memory usage < 75GB per GPU
- [ ] Verify batch size increase improves throughput: samples/sec with batch_size=8 > 1.5× baseline batch_size=4
- [ ] Check gradient accumulation reduction: Effective batch size maintained (96-192)
- [ ] Verify no memory leaks: Memory usage stable over 10+ epochs

**Implementation Note**: After completing this phase and all automated verification passes, pause here for manual confirmation from the human that the manual testing was successful before proceeding to Phase 4.

---

## Phase 4: Cloud Provider Integration (VastAI & Vultr)

### Overview

Integrate distributed training with **both VastAI and Vultr** GPU cloud providers:
1. Unified launch interface for both providers
2. Config-driven GPU count (`training.num_gpus`)
3. Provider-specific multi-GPU instance filtering
4. `torchrun` launcher in startup scripts
5. Flexible support for 2-GPU and 4-GPU configs

**Provider Comparison**:

| Feature | VastAI | Vultr |
|---------|--------|-------|
| **Cost** | $2-3/hr (2×A100) | $3-4/hr (2×A100) |
| **Availability** | High (spot market) | High (on-demand) |
| **Reliability** | Variable (community hosts) | High (managed infrastructure) |
| **API** | `vastai` CLI | `vultr-cli` CLI + Python client |
| **Multi-GPU** | 2-8×GPU instances | 2-8×GPU (bare metal and VM) |
| **NVLink** | Depends on host | Yes (HGX A100 bare metal) |
| **Best For** | Cost-sensitive experiments | Production workloads |

### Changes Required

#### 1. Create Unified Cloud Provider Interface

**File**: `scripts/cloud_providers.py` (NEW FILE)

**Content**:
```python
"""Unified interface for GPU cloud providers (VastAI, Vultr)."""

from __future__ import annotations

import abc
import subprocess
import json
from dataclasses import dataclass
from typing import Optional


@dataclass
class GPUInstance:
    """Unified GPU instance representation."""
    instance_id: str
    provider: str
    num_gpus: int
    gpu_model: str
    gpu_ram_gb: int
    cost_per_hour: float
    status: str
    ssh_host: Optional[str] = None
    ssh_port: Optional[int] = None


class CloudProvider(abc.ABC):
    """Abstract base class for GPU cloud providers."""

    @abc.abstractmethod
    def search_instances(
        self,
        num_gpus: int,
        min_gpu_ram: int = 80,
        gpu_model: Optional[str] = None,
    ) -> list[GPUInstance]:
        """Search for available GPU instances."""
        pass

    @abc.abstractmethod
    def create_instance(
        self,
        num_gpus: int,
        min_gpu_ram: int = 80,
        gpu_model: Optional[str] = None,
        startup_script: Optional[str] = None,
    ) -> GPUInstance:
        """Create a new GPU instance."""
        pass

    @abc.abstractmethod
    def destroy_instance(self, instance_id: str) -> None:
        """Destroy a GPU instance."""
        pass

    @abc.abstractmethod
    def get_instance_status(self, instance_id: str) -> GPUInstance:
        """Get instance status and details."""
        pass


class VastAIProvider(CloudProvider):
    """VastAI GPU cloud provider."""

    def search_instances(
        self,
        num_gpus: int,
        min_gpu_ram: int = 80,
        gpu_model: Optional[str] = None,
    ) -> list[GPUInstance]:
        """Search VastAI for available instances."""
        gpu_filter = f"gpu_ram >= {min_gpu_ram} reliability > 0.95 num_gpus={num_gpus} disk_space >= 64"
        if gpu_model:
            gpu_filter += f" gpu_name={gpu_model}"

        cmd = ["vastai", "search", "offers", gpu_filter, "--order", "dph_total", "--raw"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        instances = []
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            data = json.loads(line)
            instances.append(
                GPUInstance(
                    instance_id=str(data["id"]),
                    provider="vastai",
                    num_gpus=data["num_gpus"],
                    gpu_model=data["gpu_name"],
                    gpu_ram_gb=data["gpu_ram"],
                    cost_per_hour=data["dph_total"],
                    status="available",
                )
            )

        return instances[:10]  # Top 10 by price

    def create_instance(
        self,
        num_gpus: int,
        min_gpu_ram: int = 80,
        gpu_model: Optional[str] = None,
        startup_script: Optional[str] = None,
    ) -> GPUInstance:
        """Create VastAI instance."""
        # Search for best offer
        instances = self.search_instances(num_gpus, min_gpu_ram, gpu_model)
        if not instances:
            raise RuntimeError("No instances available")

        best_offer = instances[0]
        offer_id = best_offer.instance_id

        # Create instance with onstart script
        cmd = ["vastai", "create", "instance", offer_id]
        if startup_script:
            cmd.extend(["--onstart-cmd", startup_script])

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        instance_id = result.stdout.strip()

        return GPUInstance(
            instance_id=instance_id,
            provider="vastai",
            num_gpus=best_offer.num_gpus,
            gpu_model=best_offer.gpu_model,
            gpu_ram_gb=best_offer.gpu_ram_gb,
            cost_per_hour=best_offer.cost_per_hour,
            status="creating",
        )

    def destroy_instance(self, instance_id: str) -> None:
        """Destroy VastAI instance."""
        cmd = ["vastai", "destroy", "instance", instance_id]
        subprocess.run(cmd, check=True)

    def get_instance_status(self, instance_id: str) -> GPUInstance:
        """Get VastAI instance status."""
        cmd = ["vastai", "show", "instances", "--raw"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        for line in result.stdout.strip().split("\n"):
            data = json.loads(line)
            if str(data["id"]) == instance_id:
                return GPUInstance(
                    instance_id=instance_id,
                    provider="vastai",
                    num_gpus=data["num_gpus"],
                    gpu_model=data["gpu_name"],
                    gpu_ram_gb=data["gpu_ram"],
                    cost_per_hour=data["dph_total"],
                    status=data["actual_status"],
                    ssh_host=data.get("ssh_host"),
                    ssh_port=data.get("ssh_port"),
                )

        raise ValueError(f"Instance {instance_id} not found")


class VultrProvider(CloudProvider):
    """Vultr GPU cloud provider."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize Vultr provider with API key."""
        import os
        self.api_key = api_key or os.environ.get("VULTR_API_KEY")
        if not self.api_key:
            raise ValueError("VULTR_API_KEY environment variable not set")

    def search_instances(
        self,
        num_gpus: int,
        min_gpu_ram: int = 80,
        gpu_model: Optional[str] = None,
    ) -> list[GPUInstance]:
        """Search Vultr for available GPU plans."""
        # Use vultr-cli to list GPU plans
        cmd = ["vultr-cli", "plans", "list", "--type", "vhf", "--output", "json"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        plans = json.loads(result.stdout)

        instances = []
        for plan in plans:
            # Filter by GPU specs
            if "gpu" not in plan.get("id", "").lower():
                continue

            # Parse GPU count from plan name (e.g., "vhf-8c-32gb-a100-2gpu")
            plan_name = plan.get("id", "")
            if f"{num_gpus}gpu" not in plan_name.lower():
                continue

            # Parse GPU RAM from plan description
            # Vultr A100 plans have 80GB HBM2e per GPU
            gpu_ram = 80 if "a100" in plan_name.lower() else 48

            if gpu_ram < min_gpu_ram:
                continue

            if gpu_model and gpu_model.lower() not in plan_name.lower():
                continue

            instances.append(
                GPUInstance(
                    instance_id=plan["id"],
                    provider="vultr",
                    num_gpus=num_gpus,
                    gpu_model=gpu_model or "A100",
                    gpu_ram_gb=gpu_ram,
                    cost_per_hour=plan.get("monthly_cost", 0) / 730,  # Approximate hourly
                    status="available",
                )
            )

        return instances

    def create_instance(
        self,
        num_gpus: int,
        min_gpu_ram: int = 80,
        gpu_model: Optional[str] = None,
        startup_script: Optional[str] = None,
    ) -> GPUInstance:
        """Create Vultr GPU instance."""
        # Find appropriate plan
        plans = self.search_instances(num_gpus, min_gpu_ram, gpu_model)
        if not plans:
            raise RuntimeError("No Vultr plans available for specified GPU config")

        best_plan = plans[0]
        plan_id = best_plan.instance_id

        # Create instance
        cmd = [
            "vultr-cli", "instance", "create",
            "--region", "ewr",  # Newark (closest to most US users)
            "--plan", plan_id,
            "--os", "387",  # Ubuntu 22.04
            "--label", f"ups-ddp-{num_gpus}gpu",
            "--output", "json",
        ]

        if startup_script:
            # Write startup script to temp file
            import tempfile
            with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
                f.write(startup_script)
                script_path = f.name

            cmd.extend(["--script-id", script_path])

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        instance_data = json.loads(result.stdout)

        return GPUInstance(
            instance_id=instance_data["id"],
            provider="vultr",
            num_gpus=best_plan.num_gpus,
            gpu_model=best_plan.gpu_model,
            gpu_ram_gb=best_plan.gpu_ram_gb,
            cost_per_hour=best_plan.cost_per_hour,
            status="creating",
        )

    def destroy_instance(self, instance_id: str) -> None:
        """Destroy Vultr instance."""
        cmd = ["vultr-cli", "instance", "delete", instance_id]
        subprocess.run(cmd, check=True)

    def get_instance_status(self, instance_id: str) -> GPUInstance:
        """Get Vultr instance status."""
        cmd = ["vultr-cli", "instance", "get", instance_id, "--output", "json"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        data = json.loads(result.stdout)

        return GPUInstance(
            instance_id=instance_id,
            provider="vultr",
            num_gpus=int(data.get("gpu_count", 0)),
            gpu_model=data.get("gpu_type", "unknown"),
            gpu_ram_gb=80,  # Assume A100
            cost_per_hour=0,  # Not returned by get
            status=data["status"],
            ssh_host=data.get("main_ip"),
            ssh_port=22,
        )


def get_provider(provider_name: str) -> CloudProvider:
    """Factory function to get cloud provider instance."""
    if provider_name.lower() == "vastai":
        return VastAIProvider()
    elif provider_name.lower() == "vultr":
        return VultrProvider()
    else:
        raise ValueError(f"Unknown provider: {provider_name}. Choose 'vastai' or 'vultr'.")
```

#### 2. Update Launch Script for Both Providers

**File**: `scripts/cloud_launch.py` (NEW FILE, replaces `vast_launch.py`)

**Content**:
```python
"""Unified GPU cloud launcher for VastAI and Vultr."""

import argparse
from pathlib import Path
from ups.utils.config_loader import load_config
from scripts.cloud_providers import get_provider


def generate_startup_script(config_path: str, num_gpus: int) -> str:
    """Generate startup script with torchrun for distributed training."""
    config_name = Path(config_path).stem

    script = f"""#!/bin/bash
set -e

# Update system
apt-get update && apt-get install -y git python3-pip

# Clone repository
cd /workspace
git clone https://github.com/your-org/universal_simulator.git
cd universal_simulator

# Install dependencies
pip install -e .[dev]

# Download training data
python scripts/download_data.py

"""

    if num_gpus > 1:
        script += f"""
# Start distributed training with torchrun
echo "Starting distributed training with {num_gpus} GPUs..."
torchrun \\
  --nproc_per_node={num_gpus} \\
  --nnodes=1 \\
  --node_rank=0 \\
  --master_addr=localhost \\
  --master_port=29500 \\
  scripts/run_fast_to_sota.py \\
    --train-config configs/{config_name}.yaml
"""
    else:
        script += f"""
# Start single-GPU training
echo "Starting single-GPU training..."
python scripts/run_fast_to_sota.py \\
  --train-config configs/{config_name}.yaml
"""

    return script


def main():
    parser = argparse.ArgumentParser(description="Launch distributed training on cloud GPU providers")
    parser.add_argument("--config", required=True, help="Path to training config")
    parser.add_argument("--provider", required=True, choices=["vastai", "vultr"], help="Cloud provider")
    parser.add_argument("--dry-run", action="store_true", help="Search instances without creating")
    parser.add_argument("--auto-shutdown", action="store_true", help="Auto-shutdown after training")

    args = parser.parse_args()

    # Load config and extract GPU requirements
    cfg = load_config(args.config)
    num_gpus = cfg.get("training", {}).get("num_gpus", 1)
    min_gpu_ram = 80  # A100 default

    print(f"Launching on {args.provider.upper()} with {num_gpus}×GPU")

    # Get provider
    provider = get_provider(args.provider)

    # Search instances
    print(f"Searching for {num_gpus}×GPU instances with {min_gpu_ram}GB+ VRAM...")
    instances = provider.search_instances(num_gpus, min_gpu_ram)

    if not instances:
        print("ERROR: No instances available")
        return

    print(f"\nFound {len(instances)} available instances:")
    for i, inst in enumerate(instances[:5]):
        print(f"  {i+1}. {inst.num_gpus}×{inst.gpu_model} ({inst.gpu_ram_gb}GB) - ${inst.cost_per_hour:.2f}/hr")

    if args.dry_run:
        print("\nDry-run mode: not creating instance")
        return

    # Generate startup script
    startup_script = generate_startup_script(args.config, num_gpus)

    # Create instance
    print("\nCreating instance...")
    instance = provider.create_instance(
        num_gpus=num_gpus,
        min_gpu_ram=min_gpu_ram,
        startup_script=startup_script,
    )

    print(f"\nInstance created!")
    print(f"  Instance ID: {instance.instance_id}")
    print(f"  Provider: {instance.provider}")
    print(f"  GPUs: {instance.num_gpus}×{instance.gpu_model}")
    print(f"  Cost: ${instance.cost_per_hour:.2f}/hr")
    print(f"\nMonitor logs:")
    if instance.provider == "vastai":
        print(f"  vastai logs {instance.instance_id}")
    elif instance.provider == "vultr":
        print(f"  vultr-cli instance get {instance.instance_id}")


if __name__ == "__main__":
    main()
```

#### 3. Legacy VastAI Script Compatibility

**File**: `scripts/vast_launch.py`

**Update to use new unified interface**:
```python
"""VastAI launcher (legacy wrapper for cloud_launch.py)."""

import sys
import subprocess

# Forward to unified cloud launcher
args = sys.argv[1:]
cmd = ["python", "scripts/cloud_launch.py", "--provider", "vastai"] + args
subprocess.run(cmd)
```

#### 4. Update Orchestrator for DDP

**File**: `scripts/run_fast_to_sota.py`

**Modify subprocess calls** (lines 350-1240):

**Current**:
```python
cmd = ["python", "scripts/train.py", "--config", config_path, "--stage", stage]
subprocess.run(cmd, check=True)
```

**New**:
```python
# Check if already running under torchrun (RANK env var set)
if "RANK" in os.environ:
    # Already in distributed context, just call train.py directly
    cmd = ["python", "scripts/train.py", "--config", config_path, "--stage", stage]
else:
    # Single-GPU or need to launch torchrun
    num_gpus = cfg.get("training", {}).get("num_gpus", 1)
    if num_gpus > 1:
        cmd = [
            "torchrun",
            f"--nproc_per_node={num_gpus}",
            "--nnodes=1",
            "--node_rank=0",
            "--master_addr=localhost",
            "--master_port=29500",
            "scripts/train.py",
            "--config", config_path,
            "--stage", stage,
        ]
    else:
        cmd = ["python", "scripts/train.py", "--config", config_path, "--stage", stage]

subprocess.run(cmd, check=True)
```

#### 5. Config Validation for num_gpus

**File**: `scripts/validate_config.py`

**Add validation check** (around line 270):
```python
def validate_distributed_config(cfg: dict) -> list[str]:
    """Validate distributed training configuration."""
    errors = []

    num_gpus = cfg.get("training", {}).get("num_gpus", 1)

    if num_gpus < 1:
        errors.append("training.num_gpus must be >= 1")

    if num_gpus > 8:
        errors.append("training.num_gpus > 8 not recommended (diminishing returns)")

    # Check batch size is reasonable for GPU count
    batch_size = cfg.get("training", {}).get("batch_size", 8)
    if batch_size * num_gpus < 16:
        errors.append(f"Effective batch size ({batch_size}*{num_gpus}={batch_size*num_gpus}) is very small")

    # Warn if multi-task without balanced strategy
    task = cfg.get("data", {}).get("task", [])
    if isinstance(task, list) and len(task) > 1 and num_gpus > 1:
        strategy = cfg.get("data", {}).get("task_sampling", {}).get("strategy", "")
        if strategy != "balanced":
            errors.append("Multi-task DDP requires task_sampling.strategy='balanced'")

    return errors
```

#### 6. Setup Instructions for Vultr

**File**: `docs/vultr_setup.md` (NEW FILE)

**Content**:
```markdown
# Vultr GPU Cloud Setup

## 1. Create Vultr Account

Sign up at https://www.vultr.com/ and verify your account.

## 2. Generate API Key

1. Navigate to Account → API
2. Click "Enable API"
3. Copy your API key

## 3. Install Vultr CLI

**macOS**:
```bash
brew install vultr/vultr-cli/vultr-cli
```

**Linux**:
```bash
wget https://github.com/vultr/vultr-cli/releases/latest/download/vultr-cli_linux_amd64.tar.gz
tar -xzf vultr-cli_linux_amd64.tar.gz
sudo mv vultr-cli /usr/local/bin/
```

## 4. Configure Authentication

```bash
export VULTR_API_KEY="your-api-key-here"
echo 'export VULTR_API_KEY="your-api-key-here"' >> ~/.bashrc  # or ~/.zshrc
```

## 5. Verify Setup

```bash
vultr-cli account info
vultr-cli plans list --type vhf --output json
```

## 6. Usage

```bash
# Launch on Vultr (2×A100)
python scripts/cloud_launch.py \
  --config configs/train_pdebench_2task_baseline_ddp.yaml \
  --provider vultr \
  --auto-shutdown

# Launch on VastAI (2×A100)
python scripts/cloud_launch.py \
  --config configs/train_pdebench_2task_baseline_ddp.yaml \
  --provider vastai \
  --auto-shutdown
```

## Provider Selection Guide

**Choose Vultr when:**
- Need high reliability (production workloads)
- Want NVLink/NVSwitch for 8×GPU clusters
- Budget allows $3-4/hr for 2×A100

**Choose VastAI when:**
- Cost-sensitive experiments ($2-3/hr for 2×A100)
- Okay with variable host reliability
- Need quick spot availability
```

### Success Criteria

#### Automated Verification

- [ ] **Unified cloud provider**: `python scripts/cloud_launch.py --provider vastai --config configs/train_pdebench_2task_baseline_ddp.yaml --dry-run` succeeds
- [ ] **Vultr integration**: `python scripts/cloud_launch.py --provider vultr --config configs/train_pdebench_2task_baseline_ddp.yaml --dry-run` succeeds
- [ ] **VastAI legacy wrapper**: `python scripts/vast_launch.py launch --config configs/train_pdebench_2task_baseline_ddp.yaml --dry-run` forwards to unified launcher
- [ ] **Config validation**: `python scripts/validate_config.py configs/train_pdebench_2task_baseline_ddp.yaml` passes distributed checks
- [ ] **Local 2-GPU test**: `torchrun --nproc_per_node=2 scripts/run_fast_to_sota.py --train-config configs/train_pdebench_2task_baseline_ddp.yaml --train-only` completes all stages
- [ ] **Orchestrator subprocess handling**: Training stages execute under torchrun context without re-launching

#### Manual Verification

**VastAI Testing:**
- [ ] Launch VastAI instance: `python scripts/cloud_launch.py --provider vastai --config configs/train_pdebench_2task_baseline_ddp.yaml --auto-shutdown`
- [ ] Verify 2×A100 instance provisions correctly
- [ ] SSH and check logs: `vastai ssh <instance_id>` → `tail -f /workspace/universal_simulator/nohup.out`
- [ ] Confirm distributed training starts: "Distributed training initialized: 2 GPUs"
- [ ] Verify WandB: Only 1 run created

**Vultr Testing:**
- [ ] Launch Vultr instance: `python scripts/cloud_launch.py --provider vultr --config configs/train_pdebench_2task_baseline_ddp.yaml --auto-shutdown`
- [ ] Verify 2×A100 instance provisions correctly
- [ ] SSH and check logs: `ssh root@<vultr-ip>` → `tail -f /workspace/universal_simulator/nohup.out`
- [ ] Confirm distributed training starts: "Distributed training initialized: 2 GPUs"
- [ ] Verify WandB: Only 1 run created

**Cost Comparison:**
- [ ] Compare VastAI vs Vultr costs for 2×A100 (2-hour training run)
- [ ] Document reliability differences in `docs/cloud_provider_comparison.md`

**Implementation Note**: After completing this phase and all automated verification passes, pause here for manual confirmation from the human that the manual testing was successful before proceeding to Phase 5.

---

## Phase 5: Production Hardening & Testing

### Overview

Final production features and comprehensive testing:
1. DDP-aware checkpointing (rank 0 only saves, all ranks load)
2. Distributed metrics aggregation
3. Error recovery and fault tolerance
4. End-to-end testing (2-GPU, 4-GPU, 11-task)
5. Performance benchmarks

### Changes Required

#### 1. DDP-Aware Checkpointing

**File**: `src/ups/utils/checkpoint_manager.py`

**Modify save and load methods**:

**Save** (rank 0 only):
```python
def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    path: str,
    is_distributed: bool = False,
    rank: int = 0,
) -> None:
    """Save checkpoint (rank 0 only in distributed mode)."""
    if is_distributed and rank != 0:
        return  # Only rank 0 saves

    # Unwrap DDP if needed
    model_state = model.module.state_dict() if is_distributed else model.state_dict()

    checkpoint = {
        "model": model_state,
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }

    torch.save(checkpoint, path)
    if rank == 0:
        print(f"Checkpoint saved to {path}")
```

**Load** (all ranks):
```python
def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    path: str,
    device: torch.device,
    is_distributed: bool = False,
) -> int:
    """Load checkpoint (all ranks in distributed mode)."""
    checkpoint = torch.load(path, map_location=device)

    # Load into DDP-wrapped or unwrapped model
    if is_distributed:
        model.module.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint["model"])

    optimizer.load_state_dict(checkpoint["optimizer"])
    epoch = checkpoint["epoch"]

    print(f"Checkpoint loaded from {path} (epoch {epoch})")
    return epoch
```

**Update all checkpoint save/load calls in `scripts/train.py`** to pass `is_distributed` and `rank`.

#### 2. Distributed Metrics Aggregation

**File**: `scripts/train.py`

**Add metrics aggregation helper**:
```python
def aggregate_metrics(
    metrics: dict[str, float],
    world_size: int,
    rank: int,
) -> dict[str, float]:
    """Aggregate metrics across all ranks (mean)."""
    if world_size == 1:
        return metrics  # Single-GPU, no aggregation needed

    import torch.distributed as dist

    aggregated = {}
    for key, value in metrics.items():
        # Convert to tensor
        tensor = torch.tensor(value, device=f"cuda:{rank}")

        # All-reduce (sum)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

        # Average
        aggregated[key] = (tensor / world_size).item()

    return aggregated
```

**Use in training loop** (e.g., line 850):
```python
# Aggregate per-task metrics across ranks
if is_distributed:
    per_task_metrics = aggregate_metrics(per_task_metrics, world_size, rank)

# Log aggregated metrics (rank 0 only)
if rank == 0:
    for task_name, task_loss in per_task_metrics.items():
        wandb_ctx.log_training_metric("operator", f"{task_name}/loss", task_loss, epoch)
```

#### 3. Error Recovery for Distributed Training

**File**: `scripts/train.py`

**Enhance OOM recovery** (e.g., lines 779-783):
```python
try:
    # Forward pass
    pred = operator(state, dt=dt_tensor)
    loss = criterion(pred.z, z1)
except RuntimeError as e:
    if "out of memory" in str(e).lower():
        torch.cuda.empty_cache()

        # In distributed mode, all ranks must skip together
        if is_distributed:
            import torch.distributed as dist
            # Broadcast skip signal from rank 0
            skip_flag = torch.tensor(1, device=device)
            dist.broadcast(skip_flag, src=0)

        if rank == 0:
            print(f"Warning: OOM encountered in operator step, skipping batch (all ranks)")
        continue
    else:
        raise
```

#### 4. End-to-End Testing Suite

**File**: `tests/integration/test_distributed_training.py` (NEW FILE)

**Content**:
```python
"""Integration tests for distributed training."""

import pytest
import subprocess
import os
from pathlib import Path


def run_torchrun(nproc: int, script: str, config: str, extra_args: list[str] = None) -> int:
    """Helper to run torchrun command."""
    cmd = [
        "torchrun",
        f"--nproc_per_node={nproc}",
        "--nnodes=1",
        "--node_rank=0",
        "--master_addr=localhost",
        "--master_port=29500",
        script,
        "--config", config,
    ]
    if extra_args:
        cmd.extend(extra_args)

    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode


@pytest.mark.slow
@pytest.mark.gpu
def test_2gpu_training():
    """Test 2-GPU distributed training completes successfully."""
    returncode = run_torchrun(
        nproc=2,
        script="scripts/train.py",
        config="configs/train_pdebench_2task_baseline_ddp.yaml",
        extra_args=["--stage", "operator", "--epochs", "2"],
    )
    assert returncode == 0, "2-GPU training failed"


@pytest.mark.slow
@pytest.mark.gpu
def test_4gpu_training():
    """Test 4-GPU distributed training completes successfully."""
    returncode = run_torchrun(
        nproc=4,
        script="scripts/train.py",
        config="configs/train_pdebench_11task_ddp.yaml",
        extra_args=["--stage", "operator", "--epochs", "2"],
    )
    assert returncode == 0, "4-GPU training failed"


@pytest.mark.slow
@pytest.mark.gpu
def test_task_distribution():
    """Test task distribution is balanced across ranks."""
    returncode = run_torchrun(
        nproc=2,
        script="scripts/validate_distributed_sampling.py",
        config="configs/train_pdebench_2task_baseline_ddp.yaml",
    )
    assert returncode == 0, "Task distribution validation failed"


@pytest.mark.slow
@pytest.mark.gpu
def test_checkpoint_save_load():
    """Test DDP checkpointing (save rank 0, load all ranks)."""
    # Train for 1 epoch, save checkpoint
    returncode = run_torchrun(
        nproc=2,
        script="scripts/train.py",
        config="configs/train_pdebench_2task_baseline_ddp.yaml",
        extra_args=["--stage", "operator", "--epochs", "1"],
    )
    assert returncode == 0, "Training failed"

    # Verify checkpoint exists
    assert Path("checkpoints/op_latest.ckpt").exists(), "Checkpoint not saved"

    # Resume from checkpoint
    returncode = run_torchrun(
        nproc=2,
        script="scripts/train.py",
        config="configs/train_pdebench_2task_baseline_ddp.yaml",
        extra_args=["--stage", "operator", "--epochs", "2", "--resume"],
    )
    assert returncode == 0, "Resume from checkpoint failed"


@pytest.mark.slow
@pytest.mark.gpu
def test_wandb_single_run():
    """Test that DDP creates only 1 WandB run (not per rank)."""
    # This is a manual verification test - prints warning
    print("WARNING: This test requires manual WandB dashboard verification")
    print("After running, check WandB dashboard to ensure only 1 run was created")

    returncode = run_torchrun(
        nproc=2,
        script="scripts/train.py",
        config="configs/train_pdebench_2task_baseline_ddp.yaml",
        extra_args=["--stage", "operator", "--epochs", "1"],
    )
    assert returncode == 0, "Training failed"
```

#### 5. Performance Benchmarking Script

**File**: `scripts/benchmark_distributed.py` (NEW FILE)

**Content**:
```python
"""Benchmark distributed training speedup."""

import time
import subprocess
import json


def benchmark(nproc: int, config: str, epochs: int = 5) -> dict:
    """Run training and measure time."""
    cmd = [
        "torchrun",
        f"--nproc_per_node={nproc}",
        "scripts/train.py",
        "--config", config,
        "--stage", "operator",
        "--epochs", str(epochs),
    ]

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
    configs = {
        1: "configs/train_pdebench_2task_baseline.yaml",
        2: "configs/train_pdebench_2task_baseline_ddp.yaml",
        4: "configs/train_pdebench_11task_ddp.yaml",
    }

    results = []

    for nproc, config in configs.items():
        print(f"\n{'='*60}")
        print(f"Benchmarking {nproc}-GPU training")
        print('='*60)

        result = benchmark(nproc=nproc, config=config, epochs=5)
        results.append(result)

        print(f"Completed in {result['elapsed_sec']:.1f} sec")
        print(f"Throughput: {result['sec_per_epoch']:.1f} sec/epoch")

    # Compute speedup
    baseline = results[0]["sec_per_epoch"]

    print(f"\n{'='*60}")
    print("Speedup Summary")
    print('='*60)
    for result in results:
        speedup = baseline / result["sec_per_epoch"]
        print(f"{result['nproc']}-GPU: {speedup:.2f}× speedup ({result['sec_per_epoch']:.1f} sec/epoch)")

    # Save results
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
```

### Success Criteria

#### Automated Verification

- [ ] **Unit tests**: `pytest tests/unit/test_task_samplers.py tests/unit/test_checkpoint_manager.py -v` all pass
- [ ] **Integration tests**: `pytest tests/integration/test_distributed_training.py -v -m gpu` all pass (requires 4×GPU)
- [ ] **Checkpoint save/load**: Verify only rank 0 saves, all ranks load successfully
- [ ] **Metrics aggregation**: Per-task metrics are correctly averaged across ranks
- [ ] **Error recovery**: OOM recovery synchronizes across all ranks

#### Manual Verification

- [ ] **End-to-end 2-GPU**: Run full pipeline on 2-GPU instance, verify completion and correctness
- [ ] **End-to-end 4-GPU**: Run 11-task training on 4-GPU instance, verify all tasks trained
- [ ] **WandB dashboard**: Verify only 1 run created, all metrics logged correctly
- [ ] **Performance benchmarks**: Run `scripts/benchmark_distributed.py`, verify 1.7-3.6× speedup
- [ ] **VastAI production run**: Launch real VastAI instance with `--auto-shutdown`, verify successful completion

**Implementation Note**: After completing this phase and all automated verification passes, this is the final phase. Conduct comprehensive manual testing including end-to-end VastAI runs before considering the implementation complete.

---

## Testing Strategy

### Unit Tests

**New tests to create**:
1. **`tests/unit/test_task_samplers.py`**:
   - Test `MultiTaskDistributedSampler` index generation
   - Verify balanced task distribution across ranks
   - Test edge cases (uneven task sizes, drop_last)

2. **`tests/unit/test_checkpoint_manager.py`**:
   - Test DDP-aware save/load
   - Verify rank 0 only saves
   - Test state dict unwrapping

3. **`tests/unit/test_distributed_utils.py`**:
   - Test metrics aggregation
   - Test rank guards
   - Test error synchronization

### Integration Tests

**New integration tests** (`tests/integration/test_distributed_training.py`):
1. 2-GPU training completion
2. 4-GPU training completion
3. Task distribution validation
4. Checkpoint save/load cycle
5. WandB single run verification

### Manual Testing Steps

**Phase 1-5 Manual Verification Checklist**:

1. **Local Multi-GPU Testing** (requires 2-4 GPUs):
   ```bash
   # 2-GPU test
   torchrun --nproc_per_node=2 scripts/train.py \
     --config configs/train_pdebench_2task_baseline_ddp.yaml \
     --stage operator --epochs 5

   # 4-GPU test
   torchrun --nproc_per_node=4 scripts/train.py \
     --config configs/train_pdebench_11task_ddp.yaml \
     --stage operator --epochs 5
   ```

2. **VastAI Integration Testing**:
   ```bash
   # Launch 2-GPU instance
   python scripts/vast_launch.py launch \
     --config configs/train_pdebench_2task_baseline_ddp.yaml \
     --auto-shutdown

   # Monitor logs
   vastai logs <instance_id> -f

   # SSH and verify
   vastai ssh <instance_id>
   tail -f /workspace/universal_simulator/nohup.out
   ```

3. **Performance Validation**:
   ```bash
   # Run benchmarks
   python scripts/benchmark_distributed.py

   # Verify speedup
   cat benchmark_results.json
   # Expected: 1.7-1.9× (2-GPU), 3.2-3.6× (4-GPU)
   ```

4. **WandB Dashboard Verification**:
   - Navigate to WandB project
   - Verify only 1 run created per training job
   - Check per-task metrics are logged correctly
   - Verify no duplicate metrics or runs

5. **Memory Usage Validation**:
   ```bash
   # Profile memory
   torchrun --nproc_per_node=2 scripts/profile_memory.py \
     configs/train_pdebench_2task_baseline_ddp.yaml

   # Tune batch size
   torchrun --nproc_per_node=2 scripts/tune_batch_size.py \
     configs/train_pdebench_2task_baseline_ddp.yaml
   ```

## Performance Considerations

### Expected Speedups

**Scaling Efficiency**:
- **2-GPU**: 1.7-1.9× speedup (85-95% efficiency)
- **4-GPU**: 3.2-3.6× speedup (80-90% efficiency)

**Factors affecting efficiency**:
1. **Communication Overhead**: Gradient synchronization (NCCL collective)
2. **Load Imbalance**: Task distribution skew across ranks
3. **DataLoader Bottleneck**: Multi-worker data loading saturation
4. **Batch Size**: Larger batches → better GPU utilization → better scaling

### Memory Capacity Improvements

**Single-GPU Baseline** (80GB A100):
- Batch size: 4
- Gradient accumulation: 12
- Effective batch: 48
- Peak memory: ~75GB (95% utilization)

**2-GPU** (2×80GB = 160GB total):
- Batch size per GPU: 8 (2× increase)
- Gradient accumulation: 6 (50% reduction)
- Effective batch: 96 (2× increase)
- Peak memory per GPU: ~65GB (81% utilization)

**4-GPU** (4×80GB = 320GB total):
- Batch size per GPU: 12 (3× increase)
- Gradient accumulation: 4 (67% reduction)
- Effective batch: 192 (4× increase)
- Peak memory per GPU: ~70GB (88% utilization)

### Bottleneck Analysis

**Potential Bottlenecks**:
1. **NCCL Bandwidth**: ~300-400 GB/s for NVLink (acceptable for model size)
2. **DataLoader**: May need to increase `num_workers` per GPU (8 → 12)
3. **Task Imbalance**: MultiTaskDistributedSampler mitigates this
4. **Small Batch OOM**: If batch_size < 8 per GPU, may not scale well

**Mitigation Strategies**:
- Use `static_graph=True` in DDP for torch.compile compatibility
- Increase `prefetch_factor` for DataLoader
- Profile with `torch.profiler` to identify bottlenecks
- Consider gradient checkpointing if memory-bound (future work)

## Migration Notes

### Backward Compatibility

**Single-GPU mode** remains fully functional:
- If `training.num_gpus` not set or `= 1`, uses single-GPU path
- No DDP overhead
- Existing configs continue to work without modification

### Upgrading Existing Configs

**To enable distributed training**:
1. Add `training.num_gpus: 2` (or 4)
2. Increase `training.batch_size` (e.g., 4 → 8)
3. Decrease `training.accum_steps` (e.g., 12 → 6)
4. Set `data.task_sampling.strategy: "balanced"` for multi-task
5. Re-enable parallel encoding: `training.num_workers: 8`

**Example migration**:
```bash
# Copy baseline config
cp configs/train_pdebench_2task_baseline.yaml \
   configs/train_pdebench_2task_baseline_ddp.yaml

# Edit new config (add num_gpus, adjust batch_size, accum_steps)
# ... manual edits ...

# Validate
python scripts/validate_config.py configs/train_pdebench_2task_baseline_ddp.yaml

# Test locally (if 2-GPU available)
torchrun --nproc_per_node=2 scripts/train.py \
  --config configs/train_pdebench_2task_baseline_ddp.yaml \
  --stage operator --epochs 1
```

### VastAI Instance Selection

**Recommended instances**:
- **2×A100 80GB**: Most cost-effective (~$2-3/hr), good availability
- **4×A100 80GB**: High capacity (~$4-6/hr), lower availability

**Search filters**:
```bash
vastai search offers 'gpu_ram >= 80 reliability > 0.95 num_gpus=2 disk_space >= 64' --order 'dph_total'
vastai search offers 'gpu_ram >= 80 reliability > 0.95 num_gpus=4 disk_space >= 64' --order 'dph_total'
```

### Data Management

**Latent Cache**:
- Cache is **device-agnostic** (computed once, shared across GPUs)
- Use RAM preload mode for best multi-GPU performance
- Set `cache_dir: data/latent_cache` in config

**Dataset Downloading**:
- Download happens once per instance (not per GPU)
- Uses B2 cloud storage (same as single-GPU)
- Test/val data from WandB artifacts (same as single-GPU)

---

## Phase 6: PyTorch Lightning Migration (OPTIONAL)

### Purpose and triggers
- Only start after native DDP/FSDP paths in `scripts/train.py` are stable. Lightning buys easier strategy switching and callbacks, but costs 7-10 days and a loop refactor.
- Trigger when you need FSDP/DeepSpeed-style scaling or Lightning ecosystem features (Ray Tune, Hydra-style config composition). Skip if current DDP covers capacity and the team prefers explicit control.

### Baseline behaviors to mirror (no regressions)
- **Distributed bootstrap**: RANK/LOCAL_RANK driven setup with backend fallback in `scripts/train.py:100-207`.
- **Operator wrapping**: DDP vs optional FSDP2 plus compile in `scripts/train.py:894-937`.
- **Data loading**: Multi-task builder + `MultiTaskDistributedSampler`/`DistributedSampler` wired inside `build_latent_pair_loader` in `src/ups/data/latent_pairs.py:793-1012`.
- **Logging/resume hooks**: JSONL + Wandb via `TrainingLogger` and patience/grad-clip helpers in `scripts/train.py:330-380` and `_grad_clip_value/_get_patience` around `scripts/train.py:250-290`.
- **Error/oom handling**: Rank-wide sync + simulated OOM guards in `src/ups/training/distributed_utils.py:1-74`.

### Deliverables (Lightning side)
1) **LightningModule (operator-only first)** in `src/ups/training/lightning_modules.py`  
   - Reuse `LatentState`, `compute_nrmse`, spectral loss, and inverse-loss toggles from `scripts/train.py` so outputs/metrics match native runs.  
   - Call `maybe_trigger_simulated_oom` early in the training step to keep failure parity; honor grad clip/EMA and metric aggregation semantics.
2) **LightningDataModule** in `src/ups/data/lightning_datamodule.py`  
   - Wrap `build_latent_pair_loader(cfg)` for train/val/test; set `replace_sampler_ddp=False` so Lightning preserves the sampler created inside the loader.  
   - Surface the same config knobs (latent cache, num_workers/prefetch, rollout_horizon, task mixing).
3) **Lightning trainer entrypoint** in `scripts/train_lightning.py`  
   - Keep CLI parity with `scripts/train.py` (`--config`, `--stage`). Map `training.num_gpus` → `devices`, `training.precision`/`amp` → `precision`, `training.grad_clip` → `gradient_clip_val`, and `training.accum_steps` → `accumulate_grad_batches`.  
   - Strategy selection: `"ddp"` default; allow `"fsdp"` when `training.use_fsdp2` is set. Keep Wandb optional; ensure only rank 0 logs.  
   - Checkpoint naming mirrors `checkpoint.dir` defaults from native training.
4) **Multi-stage orchestrator** in `scripts/run_lightning_pipeline.py`  
   - Sequentially run operator → (optional) diff_residual/steady_prior using the Lightning trainer, keeping the same stage gating by `stages.*.epochs`.  
   - Preserve global_step continuity for logging comparisons across stages.
5) **Parity + rollout tests**  
   - Side-by-side runs on 1 GPU (`python scripts/train.py ...` vs `python scripts/train_lightning.py ...`) and 2 GPU (`torchrun --nproc_per_node=2 ...`) to confirm loss curves, throughput, and checkpoint sizes align within noise.  
   - Load a Lightning checkpoint into the native model to verify state_dict compatibility.

### Implementation steps (recommended order)
- **Phase 6.1**: Scaffold the LightningModule/DataModule with read-only wrappers around existing builders; no control-flow changes.  
- **Phase 6.2**: Single-stage operator training on 1 GPU; verify metrics/logs/ckpts match native.  
- **Phase 6.3**: Enable DDP/FSDP strategies; validate samplers are not replaced and Wandb runs remain single.  
- **Phase 6.4**: Wire the multi-stage pipeline + resume-from-checkpoint; check patience/early-stop parity.  
- **Phase 6.5**: Optional: expose strategy switches (DDP/FSDP/DeepSpeed) via config flag; update README/QUICKSTART.

### Effort comparison

| Approach | Effort | Risk | When to Use |
|----------|--------|------|-------------|
| **Native DDP (Phases 1-5)** | 5-7 days | Low | Default choice |
| **Lightning Migration (Phase 6)** | +7-10 days | Medium | Need FSDP/DeepSpeed or Lightning callbacks |
| **Total (DDP + Lightning)** | 12-17 days | Medium | Future-proofing |

### Success criteria

#### Automated
- [ ] `python scripts/train_lightning.py --config configs/train_pdebench_2task_baseline_ddp.yaml --stage operator` completes with loss/throughput within 5% of native.  
- [ ] `torchrun --nproc_per_node=2 scripts/train_lightning.py --config configs/train_pdebench_2task_baseline_ddp.yaml --stage operator` keeps a single Wandb run and uses task-balanced sampling.  
- [ ] Switching `strategy=fsdp` (when `training.use_fsdp2` is true) trains without Lightning/torch.compile conflicts.  
- [ ] Native loader can consume a Lightning checkpoint (state_dict keys match).

#### Manual
- [ ] Compare Lightning vs native gradient norms/ckpt sizes; confirm early-stopping/patience behaves the same.  
- [ ] Multi-stage Lightning pipeline runs to completion with the same stage ordering and resume semantics.  
- [ ] Optional DeepSpeed/Fabric strategies only after DDP parity is proven.

**Implementation note**: Keep Lightning side-by-side with native scripts (no deletion) until parity is proven; default path for production remains native DDP unless scaling gaps resurface.

---

### Lightning Migration Plan (Phase 6) with 4-GPU + speed/compile focus

**Scope**: Add Lightning-based training alongside native DDP/FSDP, keeping native as default until parity is proven. Target 1/2/4 GPU runs, compile safeguards, and throughput sanity.

**What must mirror native**
- Distributed bootstrap/backends: `scripts/train.py:100-207`
- Operator DDP/FSDP2 wrapping + compile toggles: `scripts/train.py:894-937`
- Multi-task loaders/samplers: `src/ups/data/latent_pairs.py:793-1012`
- OOM/error sync: `src/ups/training/distributed_utils.py:1-74`
- Logging/grad-clip/patience: `scripts/train.py:250-380`

**Tasks**
1) Lightning scaffolding (operator-first)
   - Add `src/ups/training/lightning_modules.py`: Operator LightningModule using `LatentState`, `compute_nrmse`, spectral/inverse loss toggles; call `maybe_trigger_simulated_oom` early; keep grad clip/EMA semantics.
   - Add `training.compile` flag: default off; when on, attempt `torch.compile` with a clear fallback path if Lightning/compile is unstable.
2) Lightning DataModule
   - Add `src/ups/data/lightning_datamodule.py`: wrap `build_latent_pair_loader(cfg)` for train/val/test; `replace_sampler_ddp=False` so Lightning preserves `MultiTaskDistributedSampler`/`DistributedSampler`.
   - Expose loader knobs: cache, num_workers, prefetch_factor, rollout_horizon, task mixing, pin_memory, persistent_workers.
3) Trainer entrypoint
   - Add `scripts/train_lightning.py`: CLI `--config --stage`; map `training.num_gpus` → devices (supports 1/2/4), `training.precision`/`amp` → precision, `training.grad_clip` → `gradient_clip_val`, `training.accum_steps` → `accumulate_grad_batches`.
   - Strategy: default `ddp`; allow `fsdp` when `training.use_fsdp2` is set. WandB rank-0 only; checkpoint naming mirrors `checkpoint.dir`. Pass through `deterministic`/`benchmark` to match cudnn settings.
4) Multi-stage orchestration
   - Add `scripts/run_lightning_pipeline.py`: sequential operator → optional diff_residual/steady_prior gated by `stages.*.epochs`; preserve stage ordering and global_step continuity; accept `--devices` override for 2/4 GPUs.
5) Speed/compile safeguards
   - Keep `static_graph` behavior when not compiling; only enable compile via flag and fallback cleanly on errors.
   - Ensure DataLoader non-blocking transfers where applicable; enable pin_memory + persistent_workers when `num_workers>0`; allow `prefetch_factor`.
   - For 4-GPU: log backend selection; allow NCCL env tweaks (e.g., `NCCL_P2P_DISABLE=1`) if instability is observed.
6) Refactor/simplify (non-breaking)
   - Centralize config→trainer mapping helper to avoid duplication between native and Lightning entrypoints.
   - Leave native scripts intact; Lightning lives side-by-side until parity/throughput confirmed.

**Success criteria**
- Automated
  - [ ] `python scripts/train_lightning.py --config configs/train_pdebench_2task_baseline_ddp.yaml --stage operator` completes; loss/throughput within 5% of native.
  - [ ] `torchrun --nproc_per_node=2 scripts/train_lightning.py --config configs/train_pdebench_2task_baseline_ddp.yaml --stage operator` single WandB run; task-balanced sampling preserved.
  - [ ] `torchrun --nproc_per_node=4 scripts/train_lightning.py --config configs/train_pdebench_2task_baseline_ddp.yaml --stage operator` stable NCCL/DDP; scaling vs 2-GPU within expected bounds (~near-linear until comms dominate).
  - [ ] `strategy=fsdp` path runs when `training.use_fsdp2` is true; no Lightning/compile conflicts.
  - [ ] Lightning checkpoint loads into native model (state_dict key compatibility).
- Manual
  - [ ] Compare Lightning vs native gradient norms, early-stop/patience, checkpoint sizes.
  - [ ] Multi-stage Lightning pipeline completes with same stage ordering/resume semantics.
  - [ ] Compile toggle: when `training.compile=true`, verify throughput improvement and clean fallback on error.
  - [ ] Performance sanity on target hardware (2- and 4-GPU) meets acceptable throughput.

## References

### Related Documents

- **Distributed Training Research**: `thoughts/shared/research/2025-11-12-distributed-training-analysis.md`
- **Multi-Dataset Scaling Plan**: `thoughts/shared/plans/2025-11-07-pdebench-multi-dataset-scaling.md`
- **Training Overhead Optimization**: `thoughts/shared/plans/2025-10-28-training-overhead-optimization.md`
- **Checkpoint Resume System**: `thoughts/shared/plans/2025-10-28-checkpoint-resume-system.md`

### Key Code Locations

- **Training Script**: `scripts/train.py` (lines 491-493, 1171, 606)
- **Data Loading**: `src/ups/data/latent_pairs.py` (lines 711-843)
- **WandB Logging**: `src/ups/utils/wandb_context.py`
- **Orchestrator**: `scripts/run_fast_to_sota.py` (lines 350-1240)
- **VastAI Launcher**: `scripts/vast_launch.py`

### External Resources

- **PyTorch DDP Tutorial**: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
- **torchrun Documentation**: https://pytorch.org/docs/stable/elastic/run.html
- **NCCL Performance Tuning**: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/tuning.html
- **VastAI Multi-GPU Guide**: https://vast.ai/docs/gpu/multiple-gpus

### Similar Implementations

- **Hugging Face Accelerate**: https://huggingface.co/docs/accelerate/usage_guides/distributed
- **PyTorch Lightning DDP**: https://lightning.ai/docs/pytorch/stable/advanced/model_parallel.html
- **DeepSpeed**: https://www.deepspeed.ai/ (future consideration for FSDP)

---

**Plan Status**: Draft (awaiting review)
**Created**: 2025-11-12T21:10:07Z
**Author**: Emery Gunselman
**Git Commit**: db340949c4fe2e32a7e55702b9fe9ebb41a2a28e
**Branch**: feature--UPT
**Estimated Effort**:
- **Phases 1-5 (Native DDP)**: 5-7 days (40-56 hours) - RECOMMENDED
- **Phase 6 (Lightning Migration)**: +7-10 days (OPTIONAL, only if needed)
- **Total (if including Lightning)**: 12-17 days
**Complexity**: High (multi-task DDP with flexible GPU support)
