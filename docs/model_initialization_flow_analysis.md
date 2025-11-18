# Model Initialization and FSDP Wrapping Flow Analysis

**Date:** 2025-11-18
**Purpose:** Identify potential SIGSEGV crash points during DDP/FSDP model initialization

## Executive Summary

Analysis of model initialization reveals **critical ordering issue**: buffers are registered on CPU during `__init__`, then the entire model is moved to CUDA with `.to(device)`, and finally wrapped with FSDP/DDP. The buffer registration in `LogSpacedRelativePositionBias` and `AdaLNConditioner` creates CPU tensors that get moved to CUDA, which could cause CUDA contamination when combined with shared memory in distributed training.

## Critical Finding: Buffer Registration Without Device Specification

### Issue 1: `shifted_window.py` - LogSpacedRelativePositionBias

**File:** `/Users/emerygunselman/Code/universal_simulator/src/ups/core/shifted_window.py`
**Lines:** 155-162

```python
def __init__(self, window_size: Iterable[int] | int, num_heads: int) -> None:
    super().__init__()
    self.window_size = _to_2tuple(window_size)
    self.num_heads = num_heads

    coords_h = torch.arange(self.window_size[0])  # ⚠️ Created on CPU
    coords_w = torch.arange(self.window_size[1])  # ⚠️ Created on CPU
    coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1)
    coords_flat = coords.view(-1, 2).float()  # (window_area, 2)
    rel = coords_flat[:, None, :] - coords_flat[None, :, :]  # (N, N, 2)
    # Signed log distance keeps directionality but compresses large offsets.
    log_rel = torch.sign(rel) * torch.log1p(rel.abs())
    self.register_buffer("log_relative_positions", log_rel)  # ⚠️ CPU buffer registered
```

**Problem:**
- Tensors created on CPU by default
- Buffer registered on CPU
- Later moved to CUDA with `.to(device)` call
- In distributed training with shared memory, this could cause CUDA context contamination

### Issue 2: `conditioning.py` - AdaLNConditioner

**File:** `/Users/emerygunselman/Code/universal_simulator/src/ups/core/conditioning.py`
**Lines:** 38-43, 70

```python
def __init__(self, cfg: ConditioningConfig) -> None:
    super().__init__()
    # ... other initialization ...

    # Start from neutral modulation (scale≈1, shift≈0, gate≈1).
    nn.init.zeros_(embed[-1].weight)  # ⚠️ CPU initialization
    nn.init.zeros_(embed[-1].bias)    # ⚠️ CPU initialization

    self.register_buffer("_dummy", torch.zeros(1), persistent=False)  # ⚠️ CPU buffer

def forward(self, cond: Mapping[str, torch.Tensor]) -> Mapping[str, torch.Tensor]:
    # ...
    if total is None:
        device = next(self.parameters()).device
        total = torch.zeros(batch, self.cfg.latent_dim * 3, device=device)  # ✓ Correct device inference
```

**Problem:**
- `_dummy` buffer registered on CPU
- Weight/bias initialization uses `nn.init` which operates on CPU tensors during construction
- Later moved to CUDA with `.to(device)` call

## Complete Model Initialization Flow

### Native Training (scripts/train.py)

**Function:** `train_operator()` (lines 899-1094)

```
1. Load data loader (line 900)
   └─> dataset_loader(cfg)

2. Create operator model (line 901) ⚠️ CRITICAL POINT
   └─> make_operator(cfg)
       ├─> Creates LatentOperatorConfig
       └─> Instantiates LatentOperator(config)
           ├─> TimeEmbedding.__init__() (line 66)
           │   └─> nn.Linear layers (CPU)
           ├─> time_to_latent = nn.Linear() (line 67)
           ├─> PDETransformerBlock.__init__() or PureTransformer.__init__() (line 79/81)
           │   └─> Creates all sub-modules on CPU:
           │       ├─> TransformerLayer modules
           │       │   ├─> ChannelSeparatedSelfAttention
           │       │   │   ├─> q_proj, k_proj, v_proj, out_proj (nn.Linear)
           │       │   │   └─> RMSNorm (nn.Parameter)
           │       │   └─> FeedForward (nn.Sequential with LayerNorm, Linear, GELU)
           │       └─> No LogSpacedRelativePositionBias in current code (not used)
           ├─> AdaLNConditioner.__init__() (line 87) ⚠️ IF conditioning enabled
           │   ├─> nn.init.zeros_() on embed layers
           │   └─> register_buffer("_dummy", torch.zeros(1)) ⚠️ CPU buffer
           └─> output_norm = nn.LayerNorm() (line 91)

3. Setup distributed training (line 912) ⚠️ BEFORE .to(device)
   └─> setup_distributed()
       ├─> Parses RANK, LOCAL_RANK, WORLD_SIZE from env
       ├─> Initializes NCCL/Gloo process group
       ├─> Sets torch.cuda.set_device(local_rank)
       └─> Returns device, is_distributed, rank, world_size, local_rank

4. Move operator to CUDA (line 914) ⚠️ CRITICAL: AFTER distributed init
   └─> operator.to(device)
       └─> Recursively moves all parameters and buffers to CUDA
           ├─> All nn.Parameter objects → CUDA
           └─> All registered buffers → CUDA (including _dummy, log_relative_positions)

5. Wrap with FSDP2/DDP (lines 917-939) ⚠️ CRITICAL: AFTER .to(device)
   └─> IF use_fsdp and num_gpus >= 2:
       └─> setup_fsdp2(operator, cfg, local_rank)
           ├─> Creates MixedPrecision policy
           ├─> Creates auto_wrap_policy (line 737-740)
           └─> Wraps with FSDP (line 745-752) ⚠️ POTENTIAL CRASH POINT
               ├─> sharding_strategy=FULL_SHARD
               ├─> device_id=local_rank
               └─> FSDP inspects all parameters/buffers
                   └─> May trigger CUDA operations on shared memory
   ELSE:
       └─> Wrap with DDP (lines 931-936)
           └─> DDP(operator, device_ids=[local_rank], ...)

6. Optional torch.compile (line 941)
   └─> _maybe_compile(operator, cfg, "operator")

7. Create encoder/decoder for inverse losses (lines 943-1004)
   └─> GridEncoder and AnyPointDecoder created and moved to device
```

### Lightning Training (scripts/train_lightning.py)

**Function:** `main()` (lines 35-139)

```
1. Load config (line 52)

2. Create Lightning module (line 77) ⚠️ CRITICAL POINT
   └─> OperatorLightningModule(cfg)
       └─> __init__() (line 204-218)
           ├─> super().__init__() (Lightning base class)
           ├─> self.operator = _build_operator(cfg) ⚠️ Same as make_operator()
           │   └─> Creates LatentOperator on CPU (same flow as above)
           └─> Optional torch.compile (line 218)

3. Create DataModule (line 81)
   └─> UPSDataModule(cfg)

4. Create Trainer (line 116-135) ⚠️ CRITICAL: Lightning handles device movement
   └─> pl.Trainer(
           accelerator="gpu",
           devices=num_gpus,
           strategy="ddp" or "fsdp",  # Line 62-64
           precision="bf16-mixed",
           ...
       )

5. trainer.fit(model, datamodule) (line 137) ⚠️ CRITICAL: Lightning internals
   └─> Lightning automatically:
       ├─> Calls setup_distributed() internally (via ddp/fsdp plugin)
       ├─> Moves model to device (calls .to(device))
       └─> Wraps model with DDP/FSDP ⚠️ POTENTIAL CRASH POINT
           └─> Same buffer registration issue as native training
```

## Identified Issues and Crash Points

### Issue A: CPU Buffer Registration → CUDA Movement in Distributed Context

**When:** During model initialization before distributed setup
**Where:** `AdaLNConditioner.__init__()` and `LogSpacedRelativePositionBias.__init__()`
**Why it crashes:**

1. **Construction phase (CPU):**
   ```python
   self.register_buffer("_dummy", torch.zeros(1))  # CPU tensor
   ```

2. **Distributed setup:**
   ```python
   torch.cuda.set_device(local_rank)  # Sets CUDA context
   dist.init_process_group(backend="nccl")  # Initializes NCCL
   ```

3. **Device movement:**
   ```python
   operator.to(device)  # Moves all buffers CPU → CUDA
   ```
   - This creates new CUDA tensors
   - In shared memory context, may contaminate CUDA device state

4. **FSDP wrapping:**
   ```python
   model = FSDP(model, device_id=local_rank, ...)
   ```
   - FSDP inspects all parameters and buffers
   - May perform allgather/shard operations
   - **CRASH**: SIGSEGV if buffers have residual CPU/shared memory references

### Issue B: Custom Parameter Initialization (nn.init calls)

**When:** During module construction
**Where:** `AdaLNConditioner.__init__()` line 39-40
**Code:**
```python
nn.init.zeros_(embed[-1].weight)
nn.init.zeros_(embed[-1].bias)
```

**Why it might crash:**
- `nn.init.*` operates in-place on parameter tensors
- Creates CPU tensors during construction
- Later moved to CUDA with `.to(device)`
- In distributed training with DataLoader workers + shared memory:
  - Main process: Creates parameters on CPU
  - DataLoader workers: May try to access same memory
  - CUDA contamination when moved to GPU

### Issue C: Order of Operations in train.py

**Current order (PROBLEMATIC):**
```
1. make_operator(cfg)           # Creates model on CPU
2. setup_distributed()          # Initializes NCCL, sets CUDA device
3. operator.to(device)          # Moves CPU → CUDA ⚠️ RISKY with shared memory
4. wrap_with_fsdp/ddp()         # Wraps model ⚠️ CRASH POINT
5. _maybe_compile()             # Compiles model
```

**Issue:** Model created on CPU, then moved to CUDA after distributed setup, then wrapped.

### Issue D: Lightning Training Order

**Lightning internal order (HIDDEN):**
```
1. OperatorLightningModule.__init__()  # Creates model on CPU
2. trainer.fit()
   ├─> Lightning DDPStrategy/FSDPStrategy setup
   │   ├─> Initializes distributed process group
   │   └─> Moves model to device
   └─> Wraps model with DDP/FSDP ⚠️ CRASH POINT
```

**Issue:** Same as native training, but hidden inside Lightning framework.

## Specific Line Numbers Where Crashes Could Occur

### Native Training (scripts/train.py)

| Line | Code | Reason |
|------|------|--------|
| 914 | `operator.to(device)` | Moving CPU buffers to CUDA in distributed context |
| 925 | `operator = setup_fsdp2(operator, cfg, local_rank)` | FSDP wrapping after device movement |
| 745-752 | `model = FSDP(model, ...)` | FSDP inspecting buffers with mixed CPU/CUDA state |
| 931-936 | `operator = DDP(operator, ...)` | DDP wrapping after device movement |

### Lightning Training (scripts/train_lightning.py)

| Line | Code | Reason |
|------|------|--------|
| 77 | `model = OperatorLightningModule(cfg)` | Model creation with CPU buffers |
| 137 | `trainer.fit(model, datamodule=datamodule)` | Lightning internal device movement + wrapping |
| N/A | Lightning internal FSDP/DDP wrapping | Hidden inside `pl.Trainer` strategy |

### Model Modules (src/ups/)

| File | Line | Code | Reason |
|------|------|------|--------|
| `conditioning.py` | 43 | `self.register_buffer("_dummy", torch.zeros(1))` | CPU buffer registration |
| `conditioning.py` | 39-40 | `nn.init.zeros_(embed[-1].weight/bias)` | In-place CPU initialization |
| `shifted_window.py` | 162 | `self.register_buffer("log_relative_positions", log_rel)` | CPU buffer registration |
| `latent_operator.py` | 66-67 | `self.time_embed = TimeEmbedding(...)` | nn.Linear on CPU |
| `latent_operator.py` | 91 | `self.output_norm = nn.LayerNorm(...)` | LayerNorm on CPU |

## Root Cause Analysis

### Primary Root Cause: Device Ambiguity During Construction

The model is constructed without explicit device specification:
```python
# Current (PROBLEMATIC)
model = LatentOperator(config)  # Created on CPU by default
model.to(device)                # Moved to CUDA later

# In distributed context with shared memory:
# - DataLoader workers create tensors in shared memory
# - Model buffers reference CPU memory
# - .to(device) creates CUDA copies
# - FSDP/DDP wrapping tries to allgather/shard
# - SIGSEGV: Mixed CPU/CUDA/shared memory references
```

### Secondary Root Cause: Buffer Registration Without Device

Buffers registered without device specification default to CPU:
```python
# Current (PROBLEMATIC)
self.register_buffer("_dummy", torch.zeros(1))  # CPU

# Should be (if model created on device):
# Device inferred from model parameters during .to() call
# But in distributed context, this inference may fail
```

### Tertiary Root Cause: Distributed Setup Order

Distributed setup happens before model device placement:
```python
# Current order:
1. Create model (CPU)
2. Setup distributed (NCCL, set CUDA device)
3. Move model to CUDA
4. Wrap with FSDP/DDP

# This creates a window where:
# - CUDA context is active (step 2)
# - Model is still on CPU (step 2)
# - Device movement creates new tensors (step 3)
# - Wrapping may see inconsistent state (step 4)
```

## Potential Crash Scenarios

### Scenario 1: FSDP Allgather on Mixed Buffers
```
1. Model created on CPU with buffers
2. Distributed init sets CUDA context
3. .to(device) moves buffers to CUDA
4. FSDP wrapping calls _allgather_buffers()
5. CRASH: SIGSEGV if buffer has residual CPU reference
```

### Scenario 2: DDP Broadcast on Shared Memory
```
1. Model created on CPU
2. DataLoader workers create shared memory tensors
3. .to(device) moves model to CUDA
4. DDP wrapping calls broadcast_parameters()
5. CRASH: SIGSEGV if shared memory tensor accessed after CUDA movement
```

### Scenario 3: Buffer Access During Forward Pass
```
1. Model wrapped with FSDP
2. Forward pass accesses _dummy buffer
3. Buffer was moved from CPU → CUDA → sharded
4. CRASH: SIGSEGV if buffer reference is stale
```

## Recommendations

### Fix 1: Create Model Directly on Target Device
```python
# In make_operator():
def make_operator(cfg: dict, device: torch.device = None) -> LatentOperator:
    # ... config creation ...
    model = LatentOperator(config)
    if device is not None:
        model = model.to(device)  # Move before returning
    return model

# In train_operator():
device, is_distributed, rank, world_size, local_rank = setup_distributed()
operator = make_operator(cfg, device=device)  # Create on device directly
# No operator.to(device) call needed
```

### Fix 2: Specify Device in Buffer Registration
```python
# In AdaLNConditioner.__init__():
def __init__(self, cfg: ConditioningConfig, device: torch.device = None) -> None:
    super().__init__()
    # ... other init ...

    device = device or torch.device('cpu')  # Default to CPU if not specified
    self.register_buffer("_dummy", torch.zeros(1, device=device), persistent=False)

# In LogSpacedRelativePositionBias.__init__():
def __init__(self, window_size, num_heads, device: torch.device = None) -> None:
    super().__init__()
    device = device or torch.device('cpu')

    coords_h = torch.arange(self.window_size[0], device=device)
    coords_w = torch.arange(self.window_size[1], device=device)
    # ... rest of init ...
    self.register_buffer("log_relative_positions", log_rel)  # Already on device
```

### Fix 3: Change Initialization Order
```python
# Current (PROBLEMATIC):
operator = make_operator(cfg)           # CPU
device, ... = setup_distributed()       # CUDA context
operator.to(device)                     # CPU → CUDA movement
wrap_with_fsdp(operator)                # Wrapping

# Proposed (SAFER):
device, ... = setup_distributed()       # CUDA context
operator = make_operator(cfg)           # Still CPU (can't avoid in Python)
operator.to(device)                     # CPU → CUDA movement (before wrapping)
# BUT: Still has race condition with shared memory

# Best (SAFEST):
device, ... = setup_distributed()       # CUDA context
with torch.cuda.device(device):         # Set default device
    operator = make_operator(cfg)       # Creates on CUDA by default
wrap_with_fsdp(operator)                # Wrapping (no device movement needed)
```

### Fix 4: Disable Shared Memory for Distributed Training
```python
# In dataset_loader():
if is_distributed:
    # Disable shared memory to avoid CUDA contamination
    num_workers = 0  # Force single-process data loading
else:
    num_workers = cfg.get("data", {}).get("num_workers", 4)
```

## Testing Plan

1. **Verify buffer devices after construction:**
   ```python
   model = make_operator(cfg)
   for name, buf in model.named_buffers():
       print(f"{name}: {buf.device}")
   ```

2. **Test distributed initialization order:**
   ```python
   # Add logging to train.py:
   print(f"[INIT] Rank {rank}: Before operator creation")
   operator = make_operator(cfg)
   print(f"[INIT] Rank {rank}: After operator creation, before .to(device)")
   operator.to(device)
   print(f"[INIT] Rank {rank}: After .to(device), before FSDP wrapping")
   operator = setup_fsdp2(operator, cfg, local_rank)
   print(f"[INIT] Rank {rank}: After FSDP wrapping")
   ```

3. **Test with/without shared memory:**
   ```bash
   # Test with shared memory (current):
   torchrun --nproc_per_node=2 scripts/train_lightning.py --config <config>

   # Test without shared memory:
   # Set num_workers=0 in config
   torchrun --nproc_per_node=2 scripts/train_lightning.py --config <config>
   ```

## Conclusion

**Most Likely Crash Point:** Line 914 in `scripts/train.py` (`operator.to(device)`) or line 925 (`setup_fsdp2()`) when combined with shared memory in DataLoader workers.

**Root Cause:** Buffers registered on CPU during construction, then moved to CUDA after distributed setup, causing CUDA contamination in shared memory context.

**Recommended Fix:** Implement Fix 3 (change initialization order with `torch.cuda.device()` context manager) + Fix 4 (disable shared memory for distributed training).
