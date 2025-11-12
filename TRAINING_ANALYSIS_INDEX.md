# Training Infrastructure Analysis - Index

## Quick Navigation

This analysis documents the current training infrastructure of Universal Physics Stack (UPS), with focus on distributed training readiness.

### Main Document

**File**: `TRAINING_INFRASTRUCTURE_ANALYSIS.md` (698 lines)

Complete technical analysis with exact line numbers and code snippets.

---

## Document Sections

### 1. Training Orchestration Architecture
- Entry point: `scripts/run_fast_to_sota.py` (Lines 350-1240)
- Pipeline structure and subprocess management
- Metadata management and gating system
- **Key insight**: Subprocess-based coordination with environment variables for WandB context

### 2. Training Script Architecture  
- Entry point: `scripts/train.py` (26,539 lines)
- 4-stage sequential training pipeline
- Single-device architecture (no distributed support)
- **Key insight**: Critical limitation - hardcoded to single CUDA device or CPU

### 3. Training Loop Implementation
- Core loop: Lines 599-868 in `train.py`
- Per-epoch iteration with gradient accumulation
- Mixed precision (AMP) support with GradScaler
- EMA (exponential moving average) implementation
- Early stopping with patience mechanism
- **Key insight**: Full support for gradient accumulation but no distributed synchronization

### 4. Data Loading Pipeline
- Factory: `build_latent_pair_loader()` (Lines 711-892)
- DataLoader configuration with auto-calculated worker counts
- Multi-task support via ConcatDataset
- Optional latent caching in float16
- Custom collate function for variable-length trajectories
- **Key insight**: Synchronous encoding with potential GPU-CPU transfer bottleneck

### 5. Optimizer & Scheduler Configuration
- Optimizer types: Adam, AdamW, SGD, Muon+AdamW hybrid
- Scheduler types: StepLR, CosineAnnealingLR, ReduceLROnPlateau
- Stage-specific overrides supported
- **Key insight**: Rich optimizer/scheduler support but no distributed gradient averaging

### 6. Model Initialization and Compilation
- Model factory: `make_operator()` (Lines 226-255)
- Optional torch.compile support with safe defaults
- Explicit device placement without distributed awareness
- **Key insight**: torch.compile enabled but with suppressed errors and no fullgraph mode

### 7. Logging and Monitoring
- WandBContext class for centralized logging
- Clean architecture: one run per pipeline
- Time series via `log_training_metric()`
- Summary scalars via `log_eval_summary()`
- Dual output: JSONL file + WandB
- **Key insight**: Excellent logging architecture but no distributed rank awareness

### 8. Seeding and Determinism
- `set_seed()` function (Lines 97-125)
- Configurable seed (default: 17)
- Optional deterministic algorithms
- CUBLAS workspace configuration
- **Key insight**: Determinism supported but not tested at scale

### 9. Loss Computation
- NRMSE and spectral energy losses
- Loss bundle computation with physics priors
- Inverse loss support for physical field reconstruction
- **Key insight**: Advanced loss functions but no distributed loss normalization

### 10. Batch Processing and Conditioning
- Batch unpacking with backward compatibility
- Device transfer for all tensors
- Per-task metrics tracking
- **Key insight**: Per-task tracking supports multi-task learning but not distributed aggregation

### 11. Configuration Management
- YAML config loading with recursive includes
- Stage-specific parameter overrides
- Full configuration hierarchy support
- **Key insight**: Rich config system but no distributed parameter broadcast

### 12. Current Limitations
- NO distributed training support (critical)
- Single-device only
- No DataParallel or DistributedDataParallel
- Checkpoints save state_dict only (no optimizer/scheduler state)
- No resumable training state
- **Key insight**: Fundamental architectural changes needed for distributed training

---

## Key Files and Line Numbers

### Core Training
| Component | File | Lines | Function |
|-----------|------|-------|----------|
| Orchestration | `scripts/run_fast_to_sota.py` | 350-1240 | `main()` |
| Training Script | `scripts/train.py` | 1-50 | Imports & init |
| Seed Setup | `scripts/train.py` | 97-125 | `set_seed()` |
| Model Creation | `scripts/train.py` | 226-255 | `make_operator()` |
| Optimizer Factory | `scripts/train.py` | 258-338 | `_create_optimizer()` |
| Scheduler Factory | `scripts/train.py` | 341-375 | `_create_scheduler()` |
| Compilation | `scripts/train.py` | 382-413 | `_maybe_compile()` |
| Device Selection | `scripts/train.py` | 491-493 | `torch.device()` |
| Operator Training | `scripts/train.py` | 480-894 | `train_operator()` |
| Training Loop | `scripts/train.py` | 599-868 | Epoch/batch iteration |

### Data Loading
| Component | File | Lines | Function |
|-----------|------|-------|----------|
| DataLoader Factory | `src/ups/data/latent_pairs.py` | 711-892 | `build_latent_pair_loader()` |
| Configuration | `src/ups/data/latent_pairs.py` | 732-741 | Loader kwargs |
| Custom Collate | `src/ups/data/latent_pairs.py` | 895-988 | `latent_pair_collate()` |
| Batch Unpacking | `src/ups/data/latent_pairs.py` | 991-1003 | `unpack_batch()` |

### Logging
| Component | File | Lines | Function |
|-----------|------|-------|----------|
| WandB Context | `src/ups/utils/wandb_context.py` | 27-43 | `WandBContext` dataclass |
| Training Metrics | `src/ups/utils/wandb_context.py` | 45-80 | `log_training_metric()` |
| Evaluation Metrics | `src/ups/utils/wandb_context.py` | 82-109 | `log_eval_summary()` |
| Tables | `src/ups/utils/wandb_context.py` | 111-141 | `log_table()` |
| File Upload | `src/ups/utils/wandb_context.py` | 161-176 | `save_file()` |
| Monitoring Session | `src/ups/utils/monitoring.py` | 18-96 | `MonitoringSession` |

---

## Configuration Structure

### Key Training Parameters
```yaml
training:
  batch_size: 16              # Per-device batch size
  num_workers: auto           # DataLoader workers (auto = cpu_count // 4)
  pin_memory: true            # Enable pin_memory if CUDA available
  amp: false                  # Mixed precision training
  compile: false              # torch.compile
  ema_decay: 0.999            # EMA decay (0 = disabled)
  grad_clip: 1.0              # Gradient clipping norm
  accum_steps: 1              # Gradient accumulation
  patience: 10                # Early stopping patience
  lr: 1e-3                    # Learning rate
  weight_decay: 0.0           # Weight decay
```

### Stage-Specific Configuration
```yaml
stages:
  operator:
    epochs: 25
    optimizer:
      name: adamw
      lr: 1e-3
    scheduler:
      name: cosineannealinglr
      t_max: 25
  diffusion_residual:
    epochs: 10
  consistency_distill:
    epochs: 5
  steady_prior:
    epochs: 0
```

---

## Distributed Training Readiness Assessment

### Current Status: NOT READY

### Critical Barriers
1. **Device Management** (Line 491-493)
   - Hardcoded single device
   - No rank/world_size concepts
   
2. **Data Distribution** (Line 732-741)
   - No DistributedSampler
   - No data sharding across ranks
   
3. **Model Distribution** (Lines 382-413, 493)
   - No DistributedDataParallel wrapping
   - No gradient synchronization
   
4. **Checkpointing** (Lines 868-878)
   - No optimizer/scheduler state saved
   - No distributed checkpoint coordination
   
5. **Logging** (src/ups/utils/wandb_context.py)
   - No rank awareness
   - Would spam logs from multiple ranks

### Required Changes for Distributed Training

#### Phase 1: Distributed Setup
- Add rank/world_size from environment
- Initialize distributed backend (NCCL/GLOO)
- Device selection based on rank

#### Phase 2: DataLoader Distribution
- Use DistributedSampler for data sharding
- Set shuffle=False on sampler, True on DataLoader
- Synchronize batch sizes across ranks

#### Phase 3: Model Distribution
- Wrap model in DistributedDataParallel
- Set find_unused_parameters=True for mixed models
- Handle gradient synchronization

#### Phase 4: Checkpointing
- Save optimizer state with model
- Coordinator rank saves, others verify
- Add distributed barrier before checkpoint

#### Phase 5: Logging
- Gate logging to rank 0 only
- Use DistributedSampler's set_epoch() for reproducibility
- Aggregate metrics before logging

---

## How to Use This Analysis

1. **For understanding current system**: Read sections in order (1-14)
2. **For finding specific code**: Use the "Key Files and Line Numbers" tables
3. **For distributed training planning**: Focus on section 12 and readiness assessment
4. **For config changes**: Reference "Configuration Structure" section
5. **For implementation**: Use exact line numbers to locate code

---

## Code Examples

### Current Single-Device Setup
```python
# scripts/train.py, Lines 491-493
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
operator.to(device)
```

### DataLoader Creation
```python
# src/ups/data/latent_pairs.py, Lines 732-741
loader_kwargs = {
    "batch_size": batch,
    "shuffle": True,
    "collate_fn": latent_pair_collate,
    "num_workers": num_workers,
    "pin_memory": pin_memory,
    "persistent_workers": num_workers > 0,
}
```

### Gradient Accumulation
```python
# scripts/train.py, Lines 810-824
do_step = ((i + 1) % accum_steps == 0) or ((i + 1) == num_batches)
if do_step:
    if use_amp:
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(...)
        scaler.step(optimizer)
        scaler.update()
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(...)
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
```

### Logging
```python
# src/ups/utils/wandb_context.py, Lines 45-80
def log_training_metric(self, stage, metric, value, step):
    if not self.enabled or self.run is None:
        return
    key = f"training/{stage}/{metric}"
    step_key = f"training/{stage}/step"
    self.run.log({key: value, step_key: step})
```

---

## Next Steps

1. **Review** the main analysis document (`TRAINING_INFRASTRUCTURE_ANALYSIS.md`)
2. **Identify** specific components relevant to your distributed training needs
3. **Reference** exact line numbers when diving into code
4. **Plan** changes based on the readiness assessment section

---

**Document Generated**: Medium thoroughness level analysis
**Focus Areas**: Infrastructure relevant to distributed training
**Analysis Approach**: Line-by-line code review with architecture overview
