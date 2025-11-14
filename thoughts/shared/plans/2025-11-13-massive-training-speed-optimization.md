# Massive Training Speed Optimization Implementation Plan

## Overview

Implement comprehensive training speed optimizations targeting **5-10x speedup** (conservative: 3-5x) while maintaining model performance. This plan implements all 5 tiers of optimizations identified in the research document, from low-risk production techniques to experimental cutting-edge approaches.

## Current State Analysis

**Baseline Performance** (4-GPU DDP config):
- Epoch time: ~7-8 min/epoch (already optimized from 11 min via gradient accumulation)
- Training time (40 epochs): ~5 hours
- GPU memory: ~15GB/GPU (conservative, <20% of 80GB A100)
- GPU utilization: 98% during compute

**Already Implemented:**
- DDP with NCCL backend (`scripts/train.py:57-171`)
- Mixed precision training (AMP) - enabled by default
- Gradient accumulation (batch_size=8, accum_steps=3)
- Persistent workers - auto-enabled (`latent_pairs.py:795`)
- Latent caching to disk
- Flash Attention via `F.scaled_dot_product_attention`

**Major Gaps:**
- torch.compile disabled (code exists but `compile: false`)
- No activation checkpointing
- NCCL only has debug vars, missing performance tuning
- No CPU offload optimizer
- Conservative memory usage (only 15GB/80GB used)

## Desired End State

**Target Performance** (after all optimizations):
- Epoch time: **1-2 min/epoch** (5-7x faster)
- Training time (40 epochs): **<1 hour** (vs 5 hours baseline)
- GPU memory: **30-40GB/GPU** (2-3x higher batch sizes)
- GPU utilization: **>95%** sustained
- Model quality: Within **3% of baseline** validation NRMSE

**Verification:**
- Run baseline training and log WandB metrics
- Run optimized training with same seed
- Compare: epoch time, GPU memory, final NRMSE, TTC improvement
- All automated tests pass
- No OOM errors during 10-epoch validation run

## What We're NOT Doing

- **No H100-specific optimizations yet** (Tier 5) - save for future hardware upgrade
- **No architectural changes to encoder/decoder** - only transformer operator optimization
- **No reduction in model capacity** - maintain 128-dim latent, 384 hidden dim, depth 12
- **No changes to loss functions** - keep all UPT inverse losses and physics priors (may adjust frequency)
- **No multi-node training** - keep single-node 4-GPU setup
- **No mixed batch sizes across GPUs** - maintain uniform per-GPU batching

## Implementation Approach

**Strategy:** Incremental, validated optimization with rollback capability

1. **Phase 1-2 (Tier 1):** Low-risk, high-impact optimizations (2-2.5x speedup)
2. **Phase 3-4 (Tier 2):** Medium-risk quick wins (additional 1.3-1.5x)
3. **Phase 5-6 (Tier 3):** Memory → speed trade-offs (enable larger batches)
4. **Phase 7-8 (Tier 4):** Advanced experimental optimizations
5. **Phase 9 (Validation):** Full pipeline validation and benchmarking

Each phase includes:
- Specific code changes with file:line references
- Configuration updates
- Automated and manual success criteria
- Rollback procedure if validation fails

---

## Phase 1: Enable torch.compile and NCCL Tuning

### Overview
Enable PyTorch JIT compilation and optimize NCCL communication for immediate 15-30% speedup with zero risk. These are production-grade techniques with graceful fallbacks.

### Changes Required

#### 1. Enable torch.compile in Config
**File**: `configs/train_pdebench_2task_baseline_ddp_4gpu.yaml`
**Changes**: Update training section

```yaml
training:
  # ... existing config ...
  amp: true
  compile: true                    # CHANGE: false → true
  compile_mode: reduce-overhead    # NEW: Best for batch_size=8-12
  grad_clip: null
```

**Why `reduce-overhead` mode:**
- Uses CUDA graphs to eliminate Python overhead
- Best for moderate batch sizes (8-16)
- Conservative, stable choice for production

#### 2. Add NCCL Performance Tuning Variables
**File**: `scripts/vast_launch.py`
**Location**: Lines 298-304 (replace debug-only settings)

**Current code:**
```python
# NCCL debug env vars (lines 298-304)
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export TORCH_DISTRIBUTED_DEBUG=DETAIL
```

**New code:**
```python
# NCCL performance optimization
export NCCL_NSOCKS_PERTHREAD=4      # Increase socket parallelism
export NCCL_SOCKET_NTHREADS=2       # Thread pool for async ops
export NCCL_MIN_NCHANNELS=4         # Minimum communication channels
export NCCL_P2P_DISABLE=0           # Enable NVLink peer-to-peer
export NCCL_IB_DISABLE=0            # Enable InfiniBand if available

# Optional debug (only if NCCL_DEBUG_LEVEL is set)
export NCCL_DEBUG=${NCCL_DEBUG_LEVEL:-WARN}  # Default WARN, not INFO
```

**Why these settings:**
- Increases parallelism for AllReduce operations
- Enables hardware acceleration (NVLink)
- Reduces synchronization latency by 5-10%

#### 3. Verify torch.compile in Training Script
**File**: `scripts/train.py`
**Location**: Lines 550-581 (`_maybe_compile` function)

**Validation**: Confirm this code exists and is correct (no changes needed):
```python
def _maybe_compile(model: nn.Module, cfg: dict, name: str) -> nn.Module:
    """Optionally compile a model with torch.compile when enabled."""
    compile_enabled = bool(cfg.get("training", {}).get("compile", False))
    if compile_enabled and "teacher" not in name:
        try:
            compile_mode = training_cfg.get("compile_mode", "default")
            compiled = torch.compile(model, mode=compile_mode, fullgraph=False)
            return compiled
        except Exception:
            logger.warning(f"Failed to compile {name}, falling back to eager")
            return model
    return model
```

**Action**: No code changes needed, just verify this function is called for operator model.

### Success Criteria

#### Automated Verification:
- [x] Config validation passes: `python scripts/validate_config.py configs/train_pdebench_2task_baseline_ddp_4gpu.yaml`
- [ ] Training starts without errors: First 2 epochs complete
- [ ] torch.compile succeeds: Check logs for "Compiled model" message (not "falling back to eager")
- [ ] No NCCL errors: `grep -i "nccl.*error" logs/*.log` returns empty
- [ ] GPU utilization maintained: `>=95%` during training

#### Manual Verification:
- [ ] Epoch time improves: Measure 5-epoch average, expect **15-30% faster** (7 min → 5-6 min)
- [ ] Training loss converges normally: Within 5% of baseline at epoch 10
- [ ] No new warnings in logs: Check for NCCL warnings or compile failures
- [ ] Memory usage unchanged: Should remain ~15GB/GPU

**Implementation Note**: After completing automated verification, run a 10-epoch test with same seed as baseline. Compare epoch times and loss curves. If epoch time improves by <10%, investigate torch.compile fallback in logs. If NCCL errors appear, revert NCCL settings and use debug mode to diagnose.

---

## Phase 2: Selective Activation Checkpointing

### Overview
Implement activation checkpointing in transformer layers to reduce memory by 40%, enabling 2x larger batch sizes. This trades 20% backward compute for 2x overall speedup via larger batches.

### Changes Required

#### 1. Add Checkpointing to TransformerLayer
**File**: `src/ups/core/blocks_pdet.py`
**Location**: Lines 124-135 (TransformerLayer class)

**Current code:**
```python
class TransformerLayer(nn.Module):
    def __init__(self, dim: int, group_size: int, num_heads: int, mlp_ratio: float = 2.0) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = ChannelSeparatedSelfAttention(dim, group_size=group_size, num_heads=num_heads)
        self.ff = FeedForward(dim, hidden_dim=hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ff(x)
        return x
```

**New code:**
```python
from torch.utils.checkpoint import checkpoint

class TransformerLayer(nn.Module):
    def __init__(self, dim: int, group_size: int, num_heads: int, mlp_ratio: float = 2.0,
                 use_checkpoint: bool = False) -> None:  # NEW parameter
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = ChannelSeparatedSelfAttention(dim, group_size=group_size, num_heads=num_heads)
        self.ff = FeedForward(dim, hidden_dim=hidden)
        self.use_checkpoint = use_checkpoint  # NEW

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Only checkpoint during training, not inference
        if self.use_checkpoint and self.training:
            # Checkpoint attention sublayer
            x = x + checkpoint(self._attn_forward, x, use_reentrant=False)
            # Checkpoint FFN sublayer
            x = x + checkpoint(self._ffn_forward, x, use_reentrant=False)
        else:
            # Original eager execution
            x = x + self.attn(self.attn_norm(x))
            x = x + self.ff(x)
        return x

    def _attn_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Isolated attention forward for checkpointing."""
        return self.attn(self.attn_norm(x))

    def _ffn_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Isolated FFN forward for checkpointing."""
        return self.ff(x)
```

#### 2. Add use_checkpoint Parameter to PDETransformer
**File**: `src/ups/core/blocks_pdet.py`
**Location**: Lines 168-234 (PDETransformer class)

**Changes in `__init__`:**
```python
@dataclass
class PDETransformerConfig:
    # ... existing fields ...
    use_activation_checkpoint: bool = False  # NEW field

class PDETransformer(nn.Module):
    def __init__(self, cfg: PDETransformerConfig):
        super().__init__()
        # ... existing code ...

        # Update layer creation to pass use_checkpoint
        def make_layer():
            return TransformerLayer(
                dim=cfg.hidden_dim,
                group_size=cfg.group_size,
                num_heads=cfg.num_heads,
                mlp_ratio=cfg.mlp_ratio,
                use_checkpoint=cfg.use_activation_checkpoint  # NEW
            )

        # Apply to all encoder/decoder/bottleneck layers
        # ... rest of __init__ unchanged ...
```

#### 3. Add Config Parameter
**File**: `configs/train_pdebench_2task_baseline_ddp_4gpu.yaml`
**Location**: Lines 32-45 (operator.pdet section)

```yaml
operator:
  architecture_type: pdet_stack
  pdet:
    input_dim: 128
    hidden_dim: 384
    depth: 12
    num_heads: 8
    attention_type: standard
    qk_norm: true
    mlp_ratio: 4.0
    drop_path: 0.1
    dropout: 0.0
    use_activation_checkpoint: true  # NEW: Enable checkpointing
```

#### 4. Increase Batch Size and Reduce Accumulation
**File**: `configs/train_pdebench_2task_baseline_ddp_4gpu.yaml`
**Location**: Lines 54-56 (training section)

**Current:**
```yaml
training:
  batch_size: 8
  accum_steps: 3
```

**New:**
```yaml
training:
  batch_size: 16    # DOUBLED (checkpointing saves 40% memory)
  accum_steps: 2    # REDUCED (maintain effective batch ~128-192)
```

**Math:**
- Old effective batch: 8 × 3 × 4 GPUs = 96
- New effective batch: 16 × 2 × 4 GPUs = 128 (+33% larger)
- Memory: 15GB → ~18GB per GPU (still conservative)
- DDP sync: 3 → 2 per optimizer step (33% less communication)

### Success Criteria

#### Automated Verification:
- [x] Code compiles: `python -c "from src.ups.core.blocks_pdet import TransformerLayer; print('OK')"`
- [x] Config validation passes: `python scripts/validate_config.py configs/train_pdebench_2task_baseline_ddp_4gpu.yaml`
- [ ] Training starts without OOM: First 5 epochs complete
- [ ] Memory usage increased but safe: **18-22GB/GPU** (check with `nvidia-smi`)
- [ ] Checkpointing is active: Search logs for checkpoint-related messages

#### Manual Verification:
- [ ] Epoch time improves: Expect **1.3-1.6x faster** than Phase 1 (5 min → 3-4 min)
- [ ] Loss convergence unchanged: Within 5% of baseline at epoch 20
- [ ] GPU utilization maintained: **>=93%** (slight drop due to recomputation)
- [ ] No OOM errors in 20-epoch test run

**Implementation Note**: If OOM occurs, reduce batch_size incrementally: 16→14→12. If epoch time is slower than Phase 1 despite larger batches, checkpointing overhead is too high - consider checkpointing every other layer instead of all layers.

---

## Phase 3: Loss Frequency Optimization

### Overview
Reduce inverse loss computation frequency from every batch to every 4th batch, saving 25-30% epoch time. Inverse losses are primarily for regularization and can tolerate noisier gradients.

### Changes Required

#### 1. Update Inverse Loss Frequency
**File**: `configs/train_pdebench_2task_baseline_ddp_4gpu.yaml`
**Location**: Lines 77-84 (training.inverse_loss_frequency)

**Current:**
```yaml
training:
  lambda_inv_enc: 0.01
  lambda_inv_dec: 0.01
  use_inverse_losses: true
  inverse_loss_frequency: 1      # Compute every batch
```

**New:**
```yaml
training:
  lambda_inv_enc: 0.01
  lambda_inv_dec: 0.01
  use_inverse_losses: true
  inverse_loss_frequency: 4      # CHANGE: Compute every 4th batch (75% reduction)
```

#### 2. Reduce Query Sampling Count
**File**: `configs/train_pdebench_2task_baseline_ddp_4gpu.yaml`
**Location**: Lines 86-94 (training.query_sampling)

**Current:**
```yaml
training:
  query_sampling:
    enabled: true
    num_queries: 2048    # Sample 2k points per batch
    strategy: uniform
```

**New:**
```yaml
training:
  query_sampling:
    enabled: true
    num_queries: 1024    # HALVE: 2048 → 1024 (50% fewer decoder queries)
    strategy: uniform    # Could try "stratified" for better coverage
```

**Rationale:**
- Inverse encoding loss dominates compute time (30-50% of forward pass)
- Query sampling applies to inverse_encoding only (decoder queries)
- Halving queries reduces decoder compute by 50%
- Combined with frequency=4: (1/4 batches) × (1/2 queries) = **87.5% reduction** in inverse encoding compute

#### 3. Verify Loss Computation Logic
**File**: `src/ups/training/losses.py`
**Location**: Search for `inverse_loss_frequency` usage

**Action**: Verify that the frequency parameter correctly skips inverse loss computation. Look for code like:
```python
if step % cfg.inverse_loss_frequency == 0:
    inv_loss = compute_inverse_loss(...)
```

If this logic doesn't exist, it needs to be implemented in the training loop.

### Success Criteria

#### Automated Verification:
- [x] Config validation passes: `python scripts/validate_config.py configs/train_pdebench_2task_baseline_ddp_4gpu.yaml`
- [ ] Training starts normally: First 5 epochs complete
- [ ] Inverse losses logged less frequently: Check WandB for `inverse_encoding_loss` logged every 4 steps
- [ ] No errors in loss computation: `grep -i "loss.*error" logs/*.log` returns empty

#### Manual Verification:
- [ ] Epoch time improves: Expect **20-30% faster** than Phase 2 (3-4 min → 2.5-3 min)
- [ ] Training loss still converges: May be slightly noisier but within 5% of baseline at epoch 30
- [ ] Final validation NRMSE: Within **3% of baseline** (some degradation acceptable)
- [ ] Inverse loss magnitudes reasonable: Check WandB, should be similar to baseline when computed

**Implementation Note**: If validation NRMSE degrades >5%, try intermediate frequency (inverse_loss_frequency: 2) or increase num_queries back to 1536. The goal is to balance speed with performance.

---

## Phase 4: DataLoader Optimization

### Overview
Fine-tune DataLoader parameters for optimal throughput. Persistent workers are already enabled, but we can optimize num_workers and prefetch_factor for 4-GPU setup.

### Changes Required

#### 1. Increase num_workers for Better I/O Parallelism
**File**: `configs/train_pdebench_2task_baseline_ddp_4gpu.yaml`
**Location**: Lines 63-66 (training.num_workers)

**Current:**
```yaml
training:
  num_workers: 8               # 8 workers per GPU
  use_parallel_encoding: true
  pin_memory: true
  prefetch_factor: 4
```

**New:**
```yaml
training:
  num_workers: 12              # INCREASE: 3 workers per GPU (12 total for 4-GPU)
  use_parallel_encoding: true
  pin_memory: true
  prefetch_factor: 2           # REDUCE: 4 → 2 (diminishing returns, saves RAM)
```

**Rationale:**
- Multi-task training (advection1d + darcy2d) benefits from more workers
- prefetch_factor=4 is overkill for cached data, wastes RAM
- 3 workers per GPU is optimal for I/O-bound multi-task loading

#### 2. Verify persistent_workers is Enabled
**File**: `src/ups/data/latent_pairs.py`
**Location**: Line 795

**Action**: Confirm this line exists (no changes needed):
```python
persistent_workers = num_workers > 0  # Auto-enable when workers > 0
```

**Status**: Already enabled, no action required.

### Success Criteria

#### Automated Verification:
- [x] Config validation passes: `python scripts/validate_config.py configs/train_pdebench_2task_baseline_ddp_4gpu.yaml`
- [ ] Training starts with 12 workers: Check logs for "num_workers: 12"
- [ ] Worker processes persistent: Processes don't restart between epochs
- [ ] No I/O bottlenecks: GPU utilization stays **>93%**

#### Manual Verification:
- [ ] Epoch time improves: Expect **5-10% faster** than Phase 3 (2.5-3 min → 2.3-2.8 min)
- [ ] Data loading not a bottleneck: Profile with `torch.profiler` to confirm <5% data loading time
- [ ] No excessive RAM usage: Workers should use <8GB total across all GPUs
- [ ] Loss curves identical: No impact on training dynamics

**Implementation Note**: If GPU utilization drops below 90%, data loading is a bottleneck. Increase num_workers further (16 or 20) or check if latent cache is being used properly.

---

## Phase 5: CPU Offload Optimizer (Experimental)

### Overview
Offload optimizer state (AdamW momentum and variance) to CPU, freeing 8-10GB GPU memory to enable larger batch sizes. This is experimental and depends on fast CPU-GPU interconnect.

### Changes Required

#### 1. Install torchao Dependency
**File**: `pyproject.toml` or `requirements.txt`

**Action**: Add dependency (if not already present):
```toml
[project.dependencies]
torchao = ">=0.5.0"  # CPU offload optimizer
```

**Install**: `pip install torchao`

#### 2. Wrap Hybrid Optimizer with CPU Offload
**File**: `src/ups/training/hybrid_optimizer.py`
**Location**: After existing code (add new function)

**New code:**
```python
from typing import Optional
try:
    from torchao.optim import CPUOffloadOptimizer
    TORCHAO_AVAILABLE = True
except ImportError:
    TORCHAO_AVAILABLE = False

def create_hybrid_optimizer_with_offload(
    model: nn.Module,
    cfg: dict,
    offload_to_cpu: bool = False,
) -> Optimizer:
    """
    Create Muon+AdamW hybrid optimizer, optionally with CPU offload.

    Args:
        model: The model to optimize
        cfg: Config dict with optimizer params
        offload_to_cpu: If True and torchao available, offload optimizer state to CPU

    Returns:
        Optimizer (possibly wrapped with CPUOffloadOptimizer)
    """
    # Create base hybrid optimizer (existing function)
    optimizer = create_hybrid_optimizer(model, cfg)

    # Optionally wrap with CPU offload
    if offload_to_cpu:
        if not TORCHAO_AVAILABLE:
            logger.warning("torchao not available, skipping CPU offload")
            return optimizer

        logger.info("Wrapping optimizer with CPUOffloadOptimizer")
        optimizer = CPUOffloadOptimizer(
            optimizer,
            offload_gradients=False,  # Keep gradients on GPU for DDP
        )

    return optimizer
```

#### 3. Update Training Script to Use CPU Offload
**File**: `scripts/train.py`
**Location**: Search for `create_hybrid_optimizer` calls (multiple locations)

**Changes**: Replace optimizer creation calls:

**Before:**
```python
optimizer = create_hybrid_optimizer(model, cfg)
```

**After:**
```python
from src.ups.training.hybrid_optimizer import create_hybrid_optimizer_with_offload

offload_enabled = cfg.get("training", {}).get("cpu_offload_optimizer", False)
optimizer = create_hybrid_optimizer_with_offload(model, cfg, offload_to_cpu=offload_enabled)
```

**Locations to update:**
- Operator stage optimizer creation (~line 730)
- Diffusion stage optimizer creation (~line 860)
- Consistency distillation stage (~line 960)

#### 4. Add Config Parameter
**File**: `configs/train_pdebench_2task_baseline_ddp_4gpu.yaml`
**Location**: Lines 50-75 (training section)

**New parameter:**
```yaml
training:
  # ... existing config ...
  cpu_offload_optimizer: true  # NEW: Offload optimizer state to CPU
  batch_size: 20               # INCREASE from 16 (CPU offload frees ~8GB)
  accum_steps: 2               # Keep same
```

**Math:**
- Memory freed: ~8GB (AdamW state for all parameters)
- New memory budget: 18GB → 26GB available
- New batch size: 16 → 20 (+25%)
- Effective batch: 20 × 2 × 4 = 160

### Success Criteria

#### Automated Verification:
- [x] torchao installs successfully: `pip list | grep torchao`
- [x] Code imports without errors: `python -c "from src.ups.training.hybrid_optimizer import wrap_optimizer_with_cpu_offload"`
- [x] Config validation passes: `python scripts/validate_config.py configs/train_pdebench_2task_baseline_ddp_4gpu.yaml`
- [ ] Training starts without OOM: First 5 epochs complete
- [ ] Memory usage correct: **24-28GB/GPU** (increased but safe)

#### Manual Verification:
- [ ] Epoch time improves: Expect **15-25% faster** than Phase 4 despite CPU overhead (2.5 min → 2-2.2 min)
- [ ] CPU-GPU transfer overhead acceptable: Profile with `nvprof` or `nsys`, should be <5% of step time
- [ ] Loss convergence normal: Within 5% of baseline at epoch 40
- [ ] Optimizer updates correct: Weights are being updated (check gradient norms in logs)

**Implementation Note**: If CPU-GPU transfer overhead is >10% (epoch time doesn't improve), revert CPU offload. This optimization requires PCIe 4.0+ for good performance. On slower interconnects, the overhead negates the batch size benefit.

**Rollback Procedure**: Set `cpu_offload_optimizer: false` and `batch_size: 16` in config.

---

## Phase 6: Mixed Precision BF16

### Overview
Switch from FP16 to BF16 mixed precision for better numerical stability and eliminate GradScaler overhead. BF16 has same dynamic range as FP32 and is natively supported on A100.

### Changes Required

#### 1. Add BF16 Autocast Support
**File**: `scripts/train.py`
**Location**: Search for autocast usage in training loop

**Current pattern** (likely in training loop):
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler(enabled=cfg.training.amp)

with autocast(enabled=cfg.training.amp):  # Defaults to FP16
    output = model(input)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**New pattern:**
```python
from torch.cuda.amp import autocast, GradScaler

# Determine dtype based on config
amp_enabled = cfg.training.get("amp", False)
amp_dtype = cfg.training.get("amp_dtype", "bfloat16")  # NEW config param

if amp_dtype == "bfloat16":
    autocast_dtype = torch.bfloat16
    use_scaler = False  # BF16 doesn't need gradient scaling
elif amp_dtype == "float16":
    autocast_dtype = torch.float16
    use_scaler = True
else:
    autocast_dtype = torch.float32
    use_scaler = False

scaler = GradScaler(enabled=use_scaler)

with autocast(enabled=amp_enabled, dtype=autocast_dtype):
    output = model(input)
    loss = criterion(output, target)

if use_scaler:
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
else:
    loss.backward()
    optimizer.step()
```

#### 2. Add Config Parameter
**File**: `configs/train_pdebench_2task_baseline_ddp_4gpu.yaml`
**Location**: Lines 71-73 (training section)

**Current:**
```yaml
training:
  amp: true
  compile: true
  compile_mode: reduce-overhead
```

**New:**
```yaml
training:
  amp: true
  amp_dtype: bfloat16    # NEW: Use BF16 instead of FP16
  compile: true
  compile_mode: reduce-overhead
```

#### 3. Update All Training Stages
**File**: `scripts/train.py`

**Action**: Apply the autocast/scaler changes to:
- Operator training loop (~line 1040)
- Diffusion training loop (~line 1140)
- Consistency distillation loop (~line 1240)

Ensure all three stages respect the `amp_dtype` config parameter.

### Success Criteria

#### Automated Verification:
- [x] Config validation passes: `python scripts/validate_config.py configs/train_pdebench_2task_baseline_ddp_4gpu.yaml`
- [ ] BF16 is used: Check logs for "autocast dtype: torch.bfloat16"
- [ ] No GradScaler overhead: Search logs, should NOT see "GradScaler enabled"
- [ ] No NaN losses: First 10 epochs complete without NaN
- [ ] Training numerically stable: Loss curves smooth, no spikes

#### Manual Verification:
- [ ] Epoch time improves: Expect **5-10% faster** than Phase 5 (2.2 min → 2-2.1 min)
- [ ] Memory bandwidth reduced: Profile with `nvidia-smi dmon` to see lower memory I/O
- [ ] Loss convergence identical: Within 3% of baseline at full training
- [ ] No numerical issues: Gradients don't explode or vanish (check gradient norms)

**Implementation Note**: If NaN losses occur, BF16 may have insufficient precision for some operations. Add explicit FP32 casting for sensitive operations (e.g., layer norms, loss computation). Alternatively, use `amp_dtype: float16` with GradScaler.

---

## Phase 7: FSDP2 for Multi-GPU Setup (Experimental)

### Overview
Replace DDP with Fully Sharded Data Parallel v2 (FSDP2) to shard model parameters across GPUs, freeing 7-10% memory per GPU. This enables larger batch sizes or deeper models. **Experimental: more complex checkpoint handling.** Supports 2+ GPU setups.

### Changes Required

#### 1. Add FSDP2 Import and Wrapper Function
**File**: `scripts/train.py`
**Location**: After DDP setup code (~line 171)

**New code:**
```python
from torch.distributed.fsdp import fully_shard
from torch.distributed.fsdp.api import ShardingStrategy, CPUOffload

def setup_fsdp2(
    model: nn.Module,
    cfg: dict,
) -> nn.Module:
    """
    Wrap model with FSDP2 (Fully Sharded Data Parallel v2).

    Requires PyTorch 2.4+. Shards parameters across GPUs to save memory.

    Args:
        model: Model to wrap
        cfg: Config dict with FSDP settings

    Returns:
        FSDP-wrapped model
    """
    fsdp_cfg = cfg.get("training", {}).get("fsdp", {})

    # Configure sharding strategy
    strategy = ShardingStrategy.FULL_SHARD  # Shard params, gradients, and optimizer state

    # Optional CPU offload (aggressive memory saving)
    cpu_offload = CPUOffload(offload_params=False)  # Keep params on GPU for speed

    logger.info(f"Wrapping model with FSDP2, strategy={strategy}")

    # Apply FSDP2 wrapper
    model = fully_shard(
        model,
        strategy=strategy,
        cpu_offload=cpu_offload,
    )

    return model
```

#### 2. Add Conditional FSDP/DDP Selection
**File**: `scripts/train.py`
**Location**: Replace DDP wrapping code (~line 666-680 for operator)

**Before:**
```python
if distributed:
    model = DDP(model, device_ids=[local_rank], static_graph=True)
```

**After:**
```python
if distributed:
    use_fsdp = cfg.get("training", {}).get("use_fsdp2", False)
    num_gpus = cfg.get("training", {}).get("num_gpus", 1)

    # Use FSDP2 for 2+ GPUs if enabled
    if use_fsdp and num_gpus >= 2:
        logger.info(f"Using FSDP2 for {num_gpus}-GPU distributed training")
        model = setup_fsdp2(model, cfg)
    else:
        logger.info("Using DDP for distributed training")
        model = DDP(model, device_ids=[local_rank], static_graph=True)
```

**Apply to all stages**: Operator, diffusion, consistency distillation wrappers.

#### 3. Update Checkpoint Save/Load for FSDP
**File**: `scripts/train.py`
**Location**: Search for `torch.save` and `torch.load` calls

**FSDP checkpoint saving:**
```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig

def save_checkpoint_fsdp(model, optimizer, path, cfg):
    """Save checkpoint compatible with FSDP."""
    if isinstance(model, FSDP):
        # Use FSDP state dict API
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            state_dict = model.state_dict()

            # Only rank 0 saves
            if dist.get_rank() == 0:
                torch.save({
                    "model": state_dict,
                    "optimizer": optimizer.state_dict(),
                    "config": cfg,
                }, path)
    else:
        # Standard DDP checkpoint (existing code)
        # ... existing save logic ...
```

**FSDP checkpoint loading:**
```python
def load_checkpoint_fsdp(model, path):
    """Load checkpoint compatible with FSDP."""
    checkpoint = torch.load(path, map_location="cpu")

    if isinstance(model, FSDP):
        # Use FSDP load API
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
            model.load_state_dict(checkpoint["model"])
    else:
        # Standard load (existing code)
        model.load_state_dict(checkpoint["model"])

    return checkpoint
```

#### 4. Add Config Parameter

**4-GPU Config:**
**File**: `configs/train_pdebench_2task_baseline_ddp_4gpu.yaml`
**Location**: Lines 50-58 (training section)

```yaml
training:
  num_gpus: 4
  use_fsdp2: true           # NEW: Enable FSDP2 for 4-GPU
  batch_size: 24            # INCREASE from 20 (FSDP saves ~3GB/GPU)
  accum_steps: 2
```

**2-GPU Config (Optional):**
**File**: `configs/train_pdebench_2task_baseline_ddp.yaml`
**Location**: Lines 50-58 (training section)

```yaml
training:
  num_gpus: 2
  use_fsdp2: false          # Optional: Set true to enable FSDP2 for 2-GPU
  batch_size: 30            # Can increase to 32-34 if FSDP enabled
  accum_steps: 2
  amp_dtype: bfloat16       # NEW: BF16 for Phase 6
```

**Math:**
- Memory freed: ~3GB (sharded parameters and gradients)
- New memory budget: 26GB → 29GB available
- New batch size: 20 → 24 (+20%)
- Effective batch: 24 × 2 × 4 = 192

### Success Criteria

#### Automated Verification:
- [x] PyTorch version check: `python -c "import torch; assert torch.__version__ >= '2.4'"` (PyTorch 2.9.0 installed)
- [x] FSDP2 imports: `python -c "from torch.distributed.fsdp import FullyShardedDataParallel as FSDP"` (Successfully imports)
- [x] Config validation passes: `python scripts/validate_config.py configs/train_pdebench_2task_baseline_ddp_4gpu.yaml`
- [ ] Training starts without errors: First 5 epochs complete
- [ ] Memory usage correct: **26-30GB/GPU** (higher batch, but still safe)
- [ ] Checkpoints save/load correctly: Test checkpoint at epoch 5, load and resume

#### Manual Verification:
- [ ] Epoch time improves: Expect **10-20% faster** than Phase 6 (2 min → 1.7-1.9 min)
- [ ] Communication overhead acceptable: FSDP should have similar or better scaling than DDP
- [ ] Loss convergence normal: Within 5% of baseline at full training
- [ ] Checkpoint compatibility: Old DDP checkpoints won't load, but new FSDP checkpoints work

**Implementation Note**: FSDP2 is more complex than DDP. If checkpoint loading fails or communication overhead is high (>10% slower than Phase 6), revert to DDP by setting `use_fsdp2: false`. FSDP2 works best with large models (>1B params); benefits may be marginal for smaller models.

**Rollback Procedure**:
1. Set `use_fsdp2: false` in config
2. Revert batch_size to Phase 6 value (20)
3. Delete FSDP checkpoints (incompatible with DDP)
4. Restart training from last DDP checkpoint

---

## Phase 8: Advanced Architecture Optimizations (Experimental)

### Overview
Explore advanced optimizations: FlexAttention (PyTorch 2.5+), pure transformer architecture, and fused optimizer kernels. These are experimental and may require PyTorch upgrades.

### Changes Required

#### 1. Check PyTorch Version and Upgrade if Needed

**Current requirement**: PyTorch 2.3+
**FlexAttention requires**: PyTorch 2.5+

**Action**: Check and optionally upgrade
```bash
python -c "import torch; print(torch.__version__)"

# If < 2.5, upgrade (test in isolated environment first)
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Risk**: PyTorch upgrades can break compatibility. Test thoroughly before applying to production.

#### 2. Implement FlexAttention (Optional, Requires 2.5+)

**File**: `src/ups/core/attention.py`
**Location**: Add new attention class

**New code:**
```python
try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    FLEX_ATTENTION_AVAILABLE = False

class FlexSelfAttention(nn.Module):
    """
    Self-attention using FlexAttention API (PyTorch 2.5+).

    Automatically fuses to FlashAttention kernels via torch.compile.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qk_norm: bool = True,
        dropout: float = 0.0,
        qkv_bias: bool = False,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        self.proj = nn.Linear(dim, dim)
        self.dropout_p = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, tokens, dim)

        Returns:
            (batch, tokens, dim)
        """
        B, N, C = x.shape

        # QKV projection
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply QK norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # FlexAttention (fuses to FlashAttention via compile)
        # Note: FlexAttention expects (B, num_heads, N, head_dim) format
        attn_out = flex_attention(
            q, k, v,
            scale=self.scale,
            enable_gqa=False,  # Not using grouped-query attention
        )  # (B, num_heads, N, head_dim)

        # Reshape and project
        attn_out = attn_out.transpose(1, 2).reshape(B, N, C)  # (B, N, C)
        out = self.proj(attn_out)

        return out
```

**Integration**: Update `StandardSelfAttention` to optionally use `FlexSelfAttention` backend:
```python
def __init__(self, ..., use_flex: bool = False):
    if use_flex and FLEX_ATTENTION_AVAILABLE:
        self.attn_impl = FlexSelfAttention(...)
    else:
        self.attn_impl = StandardSelfAttentionImpl(...)
```

#### 3. Add Pure Transformer Architecture Option

**File**: `configs/train_pdebench_2task_baseline_ddp_4gpu.yaml`
**Location**: Lines 32-34 (operator section)

**Current:**
```yaml
operator:
  architecture_type: pdet_stack  # Pure transformer (not pdet_unet)
```

**Alternative (U-Net transformer):**
```yaml
operator:
  architecture_type: pdet_unet  # U-shaped hierarchical transformer
```

**Decision**: Keep `pdet_stack` (pure transformer) as it's already recommended for the current setup. This is already optimal.

**No changes needed** unless testing architectural comparisons.

#### 4. Enable Fused Optimizer Kernels

**File**: `src/ups/training/hybrid_optimizer.py`
**Location**: AdamW creation

**Current:**
```python
adamw_params = [p for p in model.parameters() if p.ndim == 1]  # 1D params
adamw = torch.optim.AdamW(adamw_params, lr=lr, ...)
```

**New:**
```python
# Enable fused kernels if available (PyTorch 2.0+)
adamw = torch.optim.AdamW(
    adamw_params,
    lr=lr,
    weight_decay=weight_decay,
    betas=betas,
    eps=eps,
    fused=True,  # NEW: Fuse optimizer kernels
)
```

**Note**: Fused kernels may not support Muon optimizer. Only apply to AdamW portion of hybrid optimizer.

### Success Criteria

#### Automated Verification:
- [x] PyTorch version (if upgraded): `>=2.5.0` for FlexAttention (2.9.0 installed)
- [x] FlexAttention available (if 2.5+): `python -c "from torch.nn.attention.flex_attention import flex_attention"`
- [x] Fused AdamW enabled: Added fused=True to all AdamW optimizer creations in scripts/train.py
- [x] Config validation passes: `python scripts/validate_config.py configs/train_pdebench_2task_baseline_ddp_4gpu.yaml`
- [ ] Training starts without errors: First 5 epochs complete

#### Manual Verification:
- [ ] Epoch time improves: Expect **5-15% faster** if FlexAttention used (1.7 min → 1.5-1.6 min)
- [ ] Fused optimizer speedup: Expect **3-5% faster** optimizer steps
- [ ] Loss convergence normal: Within 5% of baseline at full training
- [ ] No regressions from PyTorch upgrade: All tests pass

**Implementation Note**: FlexAttention is cutting-edge and may have bugs or compatibility issues. If training fails or is slower, revert to standard attention. Fused optimizer is lower risk and should be kept.

**Rollback Procedure**:
1. If PyTorch 2.5 causes issues, downgrade to 2.3/2.4
2. If FlexAttention fails, set `use_flex: false` in attention config
3. If fused optimizer fails, remove `fused=True` parameter

---

## Phase 9: Full Pipeline Validation and Benchmarking

### Overview
Run comprehensive validation to measure cumulative speedup, verify model quality, and document all optimizations for production use.

**Supported Configurations**:
- **2-GPU Setup**: `configs/train_pdebench_2task_baseline_ddp.yaml` (optimized) vs `configs/train_pdebench_2task_baseline_ddp_original.yaml` (baseline)
- **4-GPU Setup**: `configs/train_pdebench_2task_baseline_ddp_4gpu.yaml` (optimized) vs `configs/train_pdebench_2task_baseline_ddp_4gpu_original.yaml` (baseline)

Both configurations have complete Phase 1-8 optimizations applied and baseline snapshots for A/B comparison.

### Changes Required

#### 1. Create Validation Script
**File**: `scripts/validate_optimizations.py` (NEW)

**Purpose**: Compare baseline vs optimized configurations

**Implementation:**
```python
#!/usr/bin/env python3
"""
Validate training optimizations by comparing baseline and optimized configs.

Usage:
    python scripts/validate_optimizations.py \
        --baseline configs/train_pdebench_2task_baseline_ddp_4gpu_original.yaml \
        --optimized configs/train_pdebench_2task_baseline_ddp_4gpu.yaml \
        --epochs 20 \
        --seed 42
"""

import argparse
import subprocess
import json
from pathlib import Path

def run_training(config_path, epochs, seed):
    """Run training with given config."""
    cmd = [
        "python", "scripts/train.py",
        "--config", config_path,
        "--stage", "operator",
        "--epochs", str(epochs),
        "--seed", str(seed),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    return result

def parse_metrics(log_path):
    """Extract metrics from training logs."""
    # Parse epoch times, final loss, memory usage, etc.
    # Implementation depends on log format
    pass

def compare_runs(baseline_metrics, optimized_metrics):
    """Compare baseline and optimized runs."""
    speedup = baseline_metrics["avg_epoch_time"] / optimized_metrics["avg_epoch_time"]
    loss_delta = abs(baseline_metrics["final_loss"] - optimized_metrics["final_loss"])

    print(f"Speedup: {speedup:.2f}x")
    print(f"Loss delta: {loss_delta:.6f}")
    print(f"Memory: {baseline_metrics['memory_gb']:.1f}GB → {optimized_metrics['memory_gb']:.1f}GB")

    return speedup, loss_delta

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--optimized", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Running baseline...")
    baseline_result = run_training(args.baseline, args.epochs, args.seed)

    print("Running optimized...")
    optimized_result = run_training(args.optimized, args.epochs, args.seed)

    # Parse and compare
    baseline_metrics = parse_metrics("logs/baseline.log")
    optimized_metrics = parse_metrics("logs/optimized.log")

    speedup, loss_delta = compare_runs(baseline_metrics, optimized_metrics)

    # Validation gates
    assert speedup >= 3.0, f"Speedup {speedup:.2f}x < target 3.0x"
    assert loss_delta < 0.001, f"Loss delta {loss_delta:.6f} too large"

    print("✓ All validation gates passed!")
```

#### 2. Create Baseline Config Snapshots
**Files**:
- `configs/train_pdebench_2task_baseline_ddp_original.yaml` (2-GPU baseline)
- `configs/train_pdebench_2task_baseline_ddp_4gpu_original.yaml` (4-GPU baseline)

**Action**: Create baseline snapshots for both 2-GPU and 4-GPU setups to preserve pre-optimization state for A/B comparison.

**2-GPU Snapshot**:
```bash
# Create baseline with: compile: false, use_activation_checkpoint: false,
# batch_size: 8, accum_steps: 6, num_workers: 8, prefetch_factor: 4,
# cpu_offload_optimizer: false, no amp_dtype (defaults to float16)
```

**4-GPU Snapshot**:
```bash
# Create baseline with same settings as 2-GPU
# batch_size: 8, accum_steps: 3 (for 4 GPUs)
```

#### 3. Run A/B Comparison
**4-GPU Command:**
```bash
python scripts/validate_optimizations.py \
    --baseline configs/train_pdebench_2task_baseline_ddp_4gpu_original.yaml \
    --optimized configs/train_pdebench_2task_baseline_ddp_4gpu.yaml \
    --epochs 20 \
    --seed 42
```

**2-GPU Command:**
```bash
python scripts/validate_optimizations.py \
    --baseline configs/train_pdebench_2task_baseline_ddp_original.yaml \
    --optimized configs/train_pdebench_2task_baseline_ddp.yaml \
    --epochs 20 \
    --seed 42
```

#### 4. Document Optimizations
**File**: `docs/training_optimizations.md` (NEW)

**Content**: Summary of all applied optimizations, performance gains, and rollback procedures.

### Success Criteria

#### Automated Verification:
- [x] Validation script created and imports successfully: `scripts/validate_optimizations.py`
- [x] 2-GPU baseline config created and valid: `configs/train_pdebench_2task_baseline_ddp_original.yaml`
- [x] 2-GPU optimized config valid: `configs/train_pdebench_2task_baseline_ddp.yaml`
- [x] 4-GPU baseline config created and valid: `configs/train_pdebench_2task_baseline_ddp_4gpu_original.yaml`
- [x] 4-GPU optimized config valid: `configs/train_pdebench_2task_baseline_ddp_4gpu.yaml`
- [x] Code imports successfully: TransformerLayer, HybridOptimizer, FSDP2, FlexAttention all available
- [x] PyTorch version check: 2.9.0 (>= 2.5 for FlexAttention)
- [x] Dependencies verified: FSDP2 available, torchao optional (graceful fallback)
- [x] Documentation created: `docs/training_optimizations.md` (covers both 2-GPU and 4-GPU setups)
- [ ] Full 20-epoch validation run (2-GPU): Both baseline and optimized complete (PENDING - requires VastAI)
- [ ] Full 20-epoch validation run (4-GPU): Both baseline and optimized complete (PENDING - requires VastAI)
- [ ] Speedup target met: **>=3.0x faster** for both configs (PENDING - requires validation run)
- [ ] Loss delta acceptable: **<0.001** MSE difference for both configs (PENDING - requires validation run)
- [ ] Memory usage safe: **<50GB/GPU** for both configs (PENDING - requires validation run)
- [ ] All unit tests pass: `pytest tests/unit/` (PENDING - test file not yet created)
- [ ] All integration tests pass: `pytest tests/integration/` (PENDING)

#### Manual Verification:
- [ ] WandB comparison: Loss curves nearly identical, optimized faster
- [ ] Validation NRMSE: Within **3%** of baseline
- [ ] Test-time conditioning (TTC): Within **5%** of baseline improvement
- [ ] Physics gate scores: Similar mass/energy conservation metrics
- [ ] Visual inspection: Rollout predictions look correct
- [ ] Production readiness: All configs documented, rollback procedures tested

**Implementation Note**: If speedup is <3x, review each phase's contribution. Likely culprits: torch.compile not working (check logs), FSDP overhead too high, or CPU offload negating benefits. If loss delta >0.001, investigate: BF16 precision issues, activation checkpointing bugs, or loss frequency too aggressive.

---

## Testing Strategy

### Unit Tests

**New test file**: `tests/unit/test_optimizations.py`

```python
import torch
import pytest
from src.ups.core.blocks_pdet import TransformerLayer

def test_activation_checkpointing():
    """Test that checkpointing doesn't change forward output."""
    layer_ckpt = TransformerLayer(dim=128, group_size=32, num_heads=4, use_checkpoint=True)
    layer_eager = TransformerLayer(dim=128, group_size=32, num_heads=4, use_checkpoint=False)

    # Copy weights
    layer_ckpt.load_state_dict(layer_eager.state_dict())

    # Test forward
    x = torch.randn(2, 64, 128)
    layer_ckpt.train()
    layer_eager.train()

    out_ckpt = layer_ckpt(x)
    out_eager = layer_eager(x)

    assert torch.allclose(out_ckpt, out_eager, atol=1e-5)

def test_bf16_autocast():
    """Test BF16 autocast doesn't cause NaNs."""
    from torch.cuda.amp import autocast

    model = TransformerLayer(dim=128, group_size=32, num_heads=4).cuda()
    x = torch.randn(2, 64, 128).cuda()

    with autocast(dtype=torch.bfloat16):
        out = model(x)

    assert not torch.isnan(out).any()
    assert out.dtype == torch.float32  # Output should be upcast

def test_cpu_offload_optimizer():
    """Test CPU offload optimizer updates weights correctly."""
    try:
        from torchao.optim import CPUOffloadOptimizer
    except ImportError:
        pytest.skip("torchao not available")

    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    optimizer = CPUOffloadOptimizer(optimizer, offload_gradients=False)

    # Run one step
    x = torch.randn(2, 10)
    loss = model(x).sum()
    loss.backward()
    optimizer.step()

    # Weights should have changed
    assert not torch.allclose(model.weight, torch.zeros_like(model.weight))
```

**Run tests:**
```bash
pytest tests/unit/test_optimizations.py -v
```

### Integration Tests

**Test full training pipeline with optimizations:**

```bash
# Short 5-epoch test with all optimizations
python scripts/train.py \
    --config configs/train_pdebench_2task_baseline_ddp_4gpu.yaml \
    --stage operator \
    --epochs 5 \
    --seed 42

# Check logs for errors
grep -i "error\|warning\|nan" logs/*.log

# Verify checkpoint saved
ls checkpoints/op_epoch_5.ckpt
```

### Manual Testing Steps

1. **Visual inspection of rollouts:**
   - Load checkpoint from optimized training
   - Generate rollout predictions
   - Compare with baseline rollouts visually
   - Check for artifacts, instabilities, or quality degradation

2. **Memory profiling:**
   ```bash
   # Use nvidia-smi to monitor memory during training
   watch -n 1 nvidia-smi

   # Or use PyTorch profiler
   python scripts/train.py --config configs/... --profile
   ```

3. **Performance profiling:**
   ```bash
   # Profile with nsys for detailed analysis
   nsys profile -o profile.nsys-rep python scripts/train.py ...

   # Visualize in Nsight Systems
   nsys-ui profile.nsys-rep
   ```

4. **Long-run stability test:**
   - Run full 40-epoch training on VastAI
   - Monitor WandB for loss spikes, NaNs, or convergence issues
   - Compare final test NRMSE with baseline
   - Verify TTC improvement similar to baseline

---

## Performance Considerations

### Expected Memory Usage Progression

| Phase | Batch Size | Memory/GPU | Change |
|-------|-----------|-----------|---------|
| Baseline | 8 | 15GB | - |
| Phase 1 (compile) | 8 | 15GB | Same |
| Phase 2 (ckpt) | 16 | 18GB | +3GB (larger batch) |
| Phase 3 (loss freq) | 16 | 18GB | Same |
| Phase 4 (dataloader) | 16 | 18GB | Same |
| Phase 5 (CPU offload) | 20 | 22GB | +4GB (larger batch) |
| Phase 6 (BF16) | 20 | 22GB | Same |
| Phase 7 (FSDP2) | 24 | 26GB | +4GB (larger batch) |
| Phase 8 (advanced) | 24 | 26GB | Same |

**Safety margin:** 26GB / 80GB = 32.5% utilization (plenty of headroom)

### Expected Epoch Time Progression

| Phase | Epoch Time | Speedup (Cumulative) | Speedup (This Phase) |
|-------|-----------|---------------------|---------------------|
| Baseline | 7.5 min | 1.0x | - |
| Phase 1 | 5.5 min | 1.36x | 1.36x (compile + NCCL) |
| Phase 2 | 3.8 min | 1.97x | 1.45x (larger batch) |
| Phase 3 | 2.9 min | 2.59x | 1.31x (loss freq) |
| Phase 4 | 2.7 min | 2.78x | 1.07x (dataloader) |
| Phase 5 | 2.2 min | 3.41x | 1.23x (CPU offload) |
| Phase 6 | 2.0 min | 3.75x | 1.10x (BF16) |
| Phase 7 | 1.7 min | 4.41x | 1.18x (FSDP2) |
| Phase 8 | 1.5 min | 5.0x | 1.13x (advanced) |

**Conservative estimate:** 3-4x speedup (Phases 1-6)
**Optimistic estimate:** 5-7x speedup (all phases)

### Cost Projections

**Baseline** (40 epochs × 7.5 min = 300 min = 5 hours):
- Cost @ $1.89/hr: **$9.45**

**After Phase 6** (40 epochs × 2 min = 80 min = 1.33 hours):
- Cost @ $1.89/hr: **$2.51** (73% cost reduction)

**After Phase 8** (40 epochs × 1.5 min = 60 min = 1 hour):
- Cost @ $1.89/hr: **$1.89** (80% cost reduction)

---

## Migration Notes

### Config File Changes Summary

**File**: `configs/train_pdebench_2task_baseline_ddp_4gpu.yaml`

**Changes from baseline:**
```yaml
operator:
  pdet:
    use_activation_checkpoint: true  # Phase 2

training:
  num_gpus: 4
  batch_size: 24                     # Increased 8→24 (Phases 2,5,7)
  accum_steps: 2                     # Reduced 3→2 (Phase 2)
  num_workers: 12                    # Increased 8→12 (Phase 4)
  prefetch_factor: 2                 # Reduced 4→2 (Phase 4)

  amp: true
  amp_dtype: bfloat16                # Phase 6
  compile: true                      # Phase 1
  compile_mode: reduce-overhead      # Phase 1

  cpu_offload_optimizer: true        # Phase 5
  use_fsdp2: true                    # Phase 7

  inverse_loss_frequency: 4          # Phase 3 (was 1)
  query_sampling:
    num_queries: 1024                # Phase 3 (was 2048)
```

### VastAI Script Changes

**File**: `scripts/vast_launch.py`

**Lines 298-304**: Replace NCCL debug vars with performance vars (Phase 1)

### Code Changes Summary

**New files:**
- `scripts/validate_optimizations.py` - Validation script
- `tests/unit/test_optimizations.py` - Unit tests
- `docs/training_optimizations.md` - Documentation

**Modified files:**
- `src/ups/core/blocks_pdet.py` - Add checkpointing (Phase 2)
- `src/ups/training/hybrid_optimizer.py` - Add CPU offload wrapper (Phase 5)
- `scripts/train.py` - Add BF16 autocast (Phase 6), FSDP2 support (Phase 7)
- `src/ups/core/attention.py` - Add FlexAttention (Phase 8, optional)

---

## References

- Original research: `thoughts/shared/research/2025-11-13-massive-training-speed-optimization.md`
- Recent DDP optimization: `thoughts/shared/research/2025-11-13-ddp-performance-optimization.md`
- Cutting-edge techniques: `thoughts/shared/research/2025-11-13-cutting-edge-training-optimizations.md`
- PyTorch torch.compile: https://pytorch.org/docs/stable/generated/torch.compile.html
- Activation checkpointing: https://pytorch.org/blog/activation-checkpointing-techniques/
- FSDP2: https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html
- FlexAttention: https://pytorch.org/blog/flexattention/
- torchao CPU offload: https://github.com/pytorch/ao/tree/main/torchao/optim
