# Universal Physics Stack - Training Pipeline Infrastructure Documentation

## Overview

This document describes the complete training pipeline infrastructure for the Universal Physics Stack (UPS), including multi-stage training, data management, loss functions, and checkpoint handling. The training system is discretization-agnostic and supports grid, mesh, and particle physics domains.

---

## 1. Training Loop Implementation

### File: `/Users/emerygunselman/Code/universal_simulator/src/ups/training/loop_train.py`

**Purpose**: Core curriculum-driven training loop for the latent operator.

**Key Components**:

- **LatentTrainer Class** (lines 27-106):
  - Manages curriculum-driven operator training
  - Handles batch cycling through multiple curriculum stages
  - Implements gradient clipping and EMA (exponential moving average) updates
  - Key methods:
    - `train()` (lines 56-74): Main training loop with curriculum cycling
    - `_train_step()` (lines 76-105): Computes loss bundle for a single batch
    - `_apply_ema()` (lines 49-54): Updates EMA model weights

- **CurriculumConfig Dataclass** (lines 18-24):
  - Configures curriculum-based training stages
  - Parameters:
    - `stages`: Multiple training stage configurations
    - `rollout_lengths`: Rollout horizon for each stage
    - `max_steps`: Maximum training steps
    - `grad_clip`: Optional gradient clipping value
    - `ema_decay`: Optional EMA decay coefficient

**EMA (Exponential Moving Average)**:
- Optional model averaging for improved stability
- Implemented with configurable decay rate
- Freezes EMA model gradients to reduce memory

---

## 2. Multi-Stage Training Pipeline

### File: `/Users/emerygunselman/Code/universal_simulator/scripts/train.py`

The training pipeline consists of 4 sequential stages + evaluation:

### **Stage 1: Operator Training**
**Function**: `train_operator()` (lines 400-693)

- Trains the latent evolution operator (PDE-Transformer)
- Loads dataset via `build_latent_pair_loader()` returning latent pairs
- Loss computation includes:
  - Forward prediction loss: MSE between predicted and target latent states
  - Optional spectral loss: Frequency-domain accuracy
  - Optional rollout loss: Multi-step prediction accuracy
  - Optional inverse losses (UPT): Ensures latent invertibility
  
- Checkpoint saved to: `checkpoints/operator.pt`
- EMA checkpoint (if enabled): `checkpoints/operator_ema.pt`
- Configuration section: `stages.operator`

**Inverse Loss Details** (for UPT - Universal Physics Transformer):
- `lambda_inv_enc`: Ensures latent can decode back to input fields
- `lambda_inv_dec`: Ensures decoded fields can re-encode to latent
- Curriculum learning applied: warmup from 0 weight, gradual increase
- Lines 417-473: Encoder/decoder initialization for inverse losses

### **Stage 2: Diffusion Residual Training**
**Function**: `train_diffusion()` (lines 695-839)

- Loads pre-trained operator as frozen teacher
- Trains diffusion model to predict residuals between operator and targets
- Uses stochastic tau sampling for broader supervision
- Loss includes:
  - MSE residual loss
  - Spectral energy loss (optional)
  - Relative NRMSE loss (optional)

- Checkpoint saved to: `checkpoints/diffusion_residual.pt`
- EMA checkpoint: `checkpoints/diffusion_residual_ema.pt`
- Configuration section: `stages.diff_residual`

**Key insight**: Diffusion learns correction on top of deterministic operator

### **Stage 3: Consistency Distillation**
**Function**: `train_consistency()` (lines 841-1020)

- Distills the diffusion model into a few-step sampler
- Uses teacher-student paradigm for loss
- Reduces inference cost from multiple diffusion steps to few steps
- Checkpoint saved to: `checkpoints/diffusion_residual.pt` (overwrites)
- Configuration section: `stages.consistency_distill`

**Implementation details** (lines 841-920):
- Teacher: Diffusion model (frozen)
- Student: Same model being trained
- Loss: MSE between student and teacher outputs at sampled tau values
- DistillationConfig controls tau sampling

### **Stage 4: Steady-State Prior Training**
**Function**: `train_steady_prior()` (lines 1022-1309)

- Trains optional prior for steady-state equilibrium prediction
- Checkpoint saved to: `checkpoints/steady_prior.pt`
- Configuration section: `stages.steady_prior`

### **Stage 5: Evaluation**
**Function**: `train_all_stages()` (lines 1516-1733)

- Orchestrates all 4 stages sequentially
- Clears GPU cache between stages (line 1582, 1600, 1618)
- Runs baseline and optional TTC evaluation
- Logs comprehensive metrics to WandB

**Multi-Stage Orchestration** (lines 1568-1632):
- Each stage checks `_stage_epochs(cfg, stage)` (lines 391-397)
- Skips stage if epochs <= 0
- Increments global_step counter for WandB logging
- Main entry point: `main()` (lines 1735-1824)

---

## 3. Data Management & Loading

### File: `/Users/emerygunselman/Code/universal_simulator/src/ups/data/datasets.py`

**Dataset Classes**:

1. **GridZarrDataset** (lines 24-131):
   - Zarr-backed grid dataset for spatial PDE data
   - Expected layout: `coords (N, 2)`, `fields/<name> (T, H, W, C)`, metadata
   - Lazy-loads grid data on demand
   - Returns Sample dict with fields, coords, metadata

2. **MeshZarrDataset** (lines 133-195):
   - Graph-based mesh dataset with cached Laplacian operators
   - Stores connectivity as CSR sparse matrix
   - Supports static and time-varying meshes

3. **ParticleZarrDataset** (lines 197-258):
   - Particle dynamics dataset with neighbor graphs
   - Includes cached neighbor connectivity indices
   - Supports variable particle counts

### File: `/Users/emerygunselman/Code/universal_simulator/src/ups/data/latent_pairs.py`

**Latent Pair Generation** (lines 245-783):

1. **LatentPair Dataclass** (lines 245-254):
   - Stores (z0, z1) latent state pairs
   - Optional future rollout states
   - Optional inverse loss fields (UPT)

2. **GridLatentPairDataset** (lines 257-413):
   - Converts PDEBench/Grid data to latent pairs
   - Encoder-based transformation: fields → latent via `GridEncoder`
   - Caching mechanism (PyTorch format):
     - Check cache: `cache_dir / f"sample_{idx:05d}.pt"`
     - Cache structure: `{"latent": tensor, "params": dict, "bc": dict}`
     - Time-stride downsampling support (line 360-363)
   
   - **Inverse Loss Support** (lines 251-254, 378-411):
     - Stores input fields, coords, metadata for inverse loss computation
     - Supports UPT training requiring encoder/decoder gradients

3. **GraphLatentPairDataset** (lines 472-543):
   - Converts mesh/particle data to latent pairs
   - Per-timestep encoding for graph structures

4. **Collate Functions**:
   - `latent_pair_collate()` (lines 723-783): Main collate for training batches
   - Handles variable-length sequences and inverse loss fields
   - Returns dict: `{"z0", "z1", "cond", "future", "input_fields", "coords", "meta"}`

### File: `/Users/emerygunselman/Code/universal_simulator/src/ups/data/parallel_cache.py`

**Cache Strategies** (lines 1-362):

1. **RawFieldDataset** (lines 33-88):
   - Workers load raw fields without encoding
   - Avoids CUDA device mismatch issues
   - Checks cache first, returns fields for on-demand encoding

2. **PreloadedCacheDataset** (lines 90-167):
   - RAM preloading strategy
   - Entire cache loaded into memory for instant access
   - ~90%+ GPU utilization when cache fits in RAM
   - Suitable for smaller datasets (< 20GB)

3. **Collate with Encoding** (lines 169-256):
   - `make_collate_with_encoding()`: Creates collate function with GPU encoding
   - Workers return raw fields, main process encodes on GPU
   - Hybrid approach: uses cache if available, encodes on-demand otherwise

4. **Helper Functions**:
   - `check_cache_complete()` (lines 313-324): Validates cache status
   - `estimate_cache_size_mb()` (lines 327-349): Memory estimation
   - `check_sufficient_ram()` (lines 351-361): RAM availability check

### File: `/Users/emerygunselman/Code/universal_simulator/src/ups/data/pdebench.py`

**PDEBench Integration** (lines 47-138):

- **PDEBenchDataset** (lines 47-138):
  - HDF5-based loader for benchmarks (burgers1d, advection1d, darcy2d, navier_stokes2d)
  - Auto-detects data format from file structure
  - Supports parameter and boundary condition aggregation
  - Optional normalization: mean/std normalization of fields

- **Task Specifications** (lines 23-28):
  - Task specs define field/parameter/BC keys for each benchmark
  - Example: `burgers1d` uses `data` field key with no parameters

### File: `/Users/emerygunselman/Code/universal_simulator/src/ups/data/schemas.py`

**Unified Sample Schema** (lines 11-117):

- **Sample TypedDict** (lines 11-37):
  - `kind`: "grid" | "mesh" | "particles"
  - `coords`: Spatial coordinates [N, d]
  - `connect`: Optional edge connectivity [E, 2]
  - `fields`: Dict of physical fields {name: [N, C_f]}
  - `bc`, `params`, `geom`: Problem metadata
  - `time`, `dt`: Temporal information
  - `meta`: Additional metadata (grid_shape, cache_path, etc.)

- **Validation** (lines 62-117):
  - `validate_sample()`: Strict schema validation with informative errors
  - Checks field dimensions match coordinates
  - Validates data types (float32/64, int64)

---

## 4. Loss Functions and Triplet/Contrastive Losses

### File: `/Users/emerygunselman/Code/universal_simulator/src/ups/training/losses.py`

**Loss Components** (lines 13-282):

### Core Loss Functions:

1. **MSE Loss** (lines 21-22):
   - `mse()`: Wrapper around F.mse_loss
   - Base loss for all prediction tasks

2. **One-Step Loss** (lines 102-103):
   - `one_step_loss()`: MSE between next predicted and target latent
   - Primary objective: `L_forward = lambda_forward * MSE(pred_next, target_next)`
   - Weight parameter: `lambda_forward` (default: 1.0)

3. **Rollout Loss** (lines 106-109):
   - `rollout_loss()`: Multi-step sequence MSE
   - Objective: `L_rollout = lambda_rollout * MSE(pred_seq, target_seq)`
   - Trains model to maintain accuracy over trajectories
   - Weight parameter: `lambda_rollout` (optional, default: 0.0)

4. **Spectral Loss** (lines 112-116):
   - `spectral_loss()`: Frequency-domain MSE
   - Objective: `L_spec = lambda_spectral * MSE(|FFT(pred)|, |FFT(target)|)`
   - Enforces correct frequency content
   - Weight parameter: `lambda_spectral` (optional)

5. **Consistency Loss** (lines 119-122):
   - `consistency_loss()`: Mean value preservation
   - Objective: `L_consistency = MSE(mean(pred), mean(target))`
   - Ensures conservation properties

### UPT Inverse Losses (Phase 1.5):

6. **Inverse Encoding Loss** (lines 25-60):
   - Name: `L_inv_enc`
   - Purpose: Ensures latent can decode back to input fields
   - Flow: input_fields → [already encoded] → decoder → reconstructed_fields
   - Loss: `MSE(reconstructed_fields, input_fields)` in physical space
   - Parameters:
     - `input_fields`: Original physical fields dict {field_name: (B, N, C)}
     - `latent`: Encoded latent (B, tokens, latent_dim)
     - `decoder`: AnyPointDecoder instance
     - `input_positions`: Spatial coordinates (B, N, coord_dim)
     - `weight`: Multiplier for loss (default: 1.0)

7. **Inverse Decoding Loss** (lines 63-99):
   - Name: `L_inv_dec`
   - Purpose: Ensures decoder output can re-encode to latent
   - Flow: latent → decoder → decoded_fields → encoder → reconstructed_latent
   - Loss: `MSE(reconstructed_latent, latent)` in latent space
   - Completes invertibility requirement
   - Original latent detached to avoid double backprop

### Curriculum Learning for Inverse Losses:

8. **Curriculum Weight Scheduling** (lines 134-165):
   - `compute_inverse_loss_curriculum_weight()`:
     - Phase 1 (epochs 0-warmup_epochs): weight = 0.0
     - Phase 2 (warmup to 2*warmup_epochs): linear ramp from 0 to base_weight
     - Phase 3 (>2*warmup_epochs): weight = min(base_weight, max_weight)
   - Parameters:
     - `warmup_epochs`: Default 15, epochs before introducing inverse loss
     - `max_weight`: Default 0.05, maximum allowed inverse loss weight
   - Prevents gradient explosion during early training

### Loss Bundle Composition:

9. **compute_operator_loss_bundle()** (lines 168-251):
   - Comprehensive loss combining multiple terms
   - Parameters (all optional):
     - Forward loss: `pred_next`, `target_next` → `L_forward`
     - Rollout loss: `pred_rollout`, `target_rollout` → `L_rollout`
     - Spectral loss: `spectral_pred`, `spectral_target` → `L_spec`
     - Inverse encoding: `input_fields`, `encoded_latent`, `decoder` → `L_inv_enc`
     - Inverse decoding: `decoder`, `encoder`, `query_positions` → `L_inv_dec`
   
   - Weight configuration via `weights` dict:
     - `lambda_forward`: Forward loss weight
     - `lambda_rollout`: Rollout loss weight
     - `lambda_spectral`: Spectral loss weight
     - `lambda_inv_enc`: Inverse encoding weight
     - `lambda_inv_dec`: Inverse decoding weight
     - `inverse_loss_warmup_epochs`: Warmup duration (default 15)
     - `inverse_loss_max_weight`: Max inverse weight (default 0.05)
   
   - Applies curriculum learning when `current_epoch` provided
   - Returns `LossBundle` with total loss and component dict

10. **compute_loss_bundle()** (lines 254-281):
    - Backward-compatible wrapper for forward/rollout/spectral losses
    - Deprecated; use `compute_operator_loss_bundle()` for new code

### Loss Bundle Structure:

```python
@dataclass
class LossBundle:
    total: Tensor        # Weighted sum of all components
    components: Dict[str, Tensor]  # Individual loss terms
```

### LossBundle Component Names:
- `L_forward`: One-step prediction loss
- `L_rollout`: Multi-step sequence loss
- `L_spec`: Spectral energy loss
- `L_inv_enc`: Inverse encoding loss (UPT)
- `L_inv_dec`: Inverse decoding loss (UPT)

---

## 5. Checkpoint Management and Stage Progression

### Checkpoint Directory Structure

```
checkpoints/
├── operator.pt              # Stage 1 output
├── operator_ema.pt          # Optional EMA average
├── diffusion_residual.pt    # Stage 2/3 output
├── diffusion_residual_ema.pt # Optional EMA average
├── diffusion_residual_epoch_1.pt  # Periodic checkpoints
├── diffusion_residual_epoch_2.pt
├── steady_prior.pt          # Stage 4 output
└── metadata.json            # Run metadata
```

### Checkpoint Saving

**Operator Stage** (lines 669-692):
- Saves best operator state (by validation loss)
- Path: `checkpoints/operator.pt`
- Also saves EMA model if enabled: `checkpoints/operator_ema.pt`
- Uploads to WandB: `wandb_ctx.save_file(operator_path)`

**Diffusion Stage** (lines 823-839):
- Saves diffusion model at epoch intervals (configurable)
- Path: `checkpoints/diffusion_residual.pt`
- Periodic checkpoints: `diffusion_residual_epoch_{epoch}.pt`
- Uploads to WandB after training completes

**Consistency Stage** (lines 992-1020):
- Overwrites diffusion checkpoint with distilled version
- Path: `checkpoints/diffusion_residual.pt`

**Steady Prior Stage** (lines 1287-1309):
- Path: `checkpoints/steady_prior.pt`

### Stage Progression

**Load Operator** (for Diffusion training):
```python
# Lines 704-710 in train_diffusion()
operator = make_operator(cfg)
op_path = checkpoint_dir / "operator.pt"
if op_path.exists():
    operator_state = torch.load(op_path, map_location="cpu")
    operator_state = _strip_compiled_prefix(operator_state)  # Remove torch.compile wrapper
    operator.load_state_dict(operator_state)
```

**Load Diffusion** (for Consistency training):
```python
# Lines 864-869 in train_consistency()
diff_path = checkpoint_dir / "diffusion_residual.pt"
if diff_path.exists():
    diff_state = torch.load(diff_path, map_location="cpu")
    diff_state = _strip_compiled_prefix(diff_state)
    diff.load_state_dict(diff_state)
```

### Configuration-Driven Stage Skipping

**Helper Function** `_stage_epochs()` (lines 391-397):
```python
def _stage_epochs(cfg: dict, stage: str) -> int:
    """Return configured epochs for a stage; 0 means skip."""
    value = cfg.get("stages", {}).get(stage, {}).get("epochs", 0)
    return int(value) if value is not None else 0
```

**Usage** (lines 1569-1579):
```python
op_epochs = _stage_epochs(cfg, "operator")
if op_epochs > 0:
    train_operator(cfg, wandb_ctx=wandb_ctx, global_step=global_step)
```

### State Dictionary Handling

**Strip Compiled Prefix** (lines 82-88):
- torch.compile() prefixes parameter names with `_orig_mod.`
- `_strip_compiled_prefix()` removes this for checkpoints saved with compiled models
- Ensures checkpoint compatibility across compilation modes

### GPU Cache Management

**Between-Stage Cleanup** (lines 1581-1584, 1599-1602, 1617-1620):
```python
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("✓ Cleared GPU cache")
```

### Checkpoint Metadata

**Metadata JSON** (in `run_fast_to_sota.py`, lines 92-120):
- Records config hash, creation timestamp, architecture fingerprint
- Tracks training completion status
- Maps back to original config for reproducibility

---

## 6. Training Configuration

### Configuration Sections

**Data Configuration** (`data`):
```yaml
data:
  task: "burgers1d"              # PDEBench task
  split: "train"                 # Data split
  root: "/path/to/pdebench"      # Data root
  field_name: "u"                # Primary field name
  patch_size: 4                  # Encoder patch size
  latent_dim: 32                 # Encoder latent dim
  latent_len: 16                 # Number of latent tokens
```

**Latent Configuration** (`latent`):
```yaml
latent:
  dim: 32        # Latent vector dimension
  tokens: 16     # Number of latent tokens
```

**Training Configuration** (`training`):
```yaml
training:
  batch_size: 16                 # Batch size
  amp: true                      # Automatic mixed precision
  compile: true                  # torch.compile enabled
  compile_mode: "default"        # Compile mode
  grad_clip: 1.0                 # Gradient clipping
  ema_decay: 0.999               # EMA decay (optional)
  dt: 0.1                        # Time step
  use_inverse_losses: false      # Enable UPT losses
  lambda_inv_enc: 0.0            # Inverse encoding weight
  lambda_inv_dec: 0.0            # Inverse decoding weight
  lambda_forward: 1.0            # Forward loss weight
  lambda_rollout: 0.0            # Rollout loss weight
  lambda_spectral: 0.0           # Spectral loss weight
  latent_cache_dir: "data/latent_cache"
  latent_cache_dtype: "float16"  # Cache precision
  num_workers: 4                 # DataLoader workers
  pin_memory: true               # Pinned memory
  accum_steps: 1                 # Gradient accumulation
  checkpoint_interval: 0         # Save every N epochs (0=never)
```

**Stage Configuration** (`stages.<stage_name>`):
```yaml
stages:
  operator:
    epochs: 25
    batch_size: 32
    optimizer:
      name: "adamw"
      lr: 1e-3
    scheduler:
      name: "cosineannealinglr"
      t_max: 25
    patience: 5
    grad_clip: 1.0
    ema_decay: 0.999
  
  diff_residual:
    epochs: 15
    optimizer:
      name: "adamw"
      lr: 5e-4
    scheduler:
      name: "cosineannealinglr"
      t_max: 15
    patience: 5
  
  consistency_distill:
    epochs: 10
    batch_size: 8
    optimizer:
      name: "adamw"
      lr: 1e-4
    scheduler:
      name: "cosineannealinglr"
      t_max: 10
  
  steady_prior:
    epochs: 5
    optimizer:
      name: "adamw"
      lr: 1e-3
```

---

## 7. Key Training Utilities

### Helper Functions in train.py

| Function | Lines | Purpose |
|----------|-------|---------|
| `load_config()` | 91-93 | Load YAML config |
| `set_seed()` | 96-124 | Set random seeds + determinism |
| `ensure_checkpoint_dir()` | 127-131 | Create checkpoint directory |
| `dataset_loader()` | 216-222 | Create data loader from config |
| `make_operator()` | 225-242 | Instantiate latent operator |
| `_create_optimizer()` | 245-258 | Create optimizer from config |
| `_create_scheduler()` | 261-295 | Create LR scheduler |
| `_amp_enabled()` | 298-299 | Check AMP status |
| `_maybe_compile()` | 302-333 | Optional torch.compile |
| `_grad_clip_value()` | 335-340 | Get grad clip per-stage |
| `_get_ema_decay()` | 343-347 | Get EMA decay per-stage |
| `_init_ema()` | 350-355 | Initialize EMA model |
| `_update_ema()` | 358-361 | Apply EMA update |
| `_get_patience()` | 364-369 | Get early stopping patience |
| `_sample_tau()` | 372-382 | Sample tau for diffusion |
| `_should_stop()` | 385-388 | Check early stop condition |
| `_stage_epochs()` | 391-397 | Get stage epoch count |

### Training Logger Class (lines 134-213)

- **Purpose**: Unified logging to file + WandB
- **Methods**:
  - `log()`: Log epoch metrics (loss, lr, grad_norm, etc.)
  - `close()`: Cleanup (no-op, run managed by orchestrator)
  - `get_global_step()`: Retrieve global step counter
- **Output**: JSONL file + WandB time series

---

## 8. WandB Integration

**WandBContext** (referenced throughout train.py):
- Single context per pipeline run
- Manages all logging centrally
- Time series: `log_training_metric(stage, metric, value, step)`
- Scalars: `log_eval_summary(metrics, prefix="eval")`
- Alerts: `alert(title, text, level)`
- Checkpoints: `save_file(path)`
- Artifacts: `log_artifact(artifact)`

**Integration Points**:
- Operator training loss (line 614): logs loss components
- Stage completion alerts (lines 688-692, 836-839, 1007-1011, 1302-1308)
- Evaluation metrics (lines 1415, 1434-1441)
- GPU info (lines 1559-1564)

---

## 9. Summary

### Training Pipeline Flow

```
┌─────────────────────────────────────────────────────────────┐
│  Main Entry: train_all_stages() / main()                    │
│             Scripts: train.py                                │
└────┬────────────────────────────────────────────────────────┘
     │
     ├─→ Stage 1: train_operator()
     │   ├─ Load: PDEBench data → GridLatentPairDataset
     │ ├─ Model: LatentOperator (PDE-Transformer)
     │   ├─ Loss: compute_operator_loss_bundle()
     │   │   - L_forward: MSE(pred_next, target_next)
     │   │   - L_rollout: MSE(pred_seq, target_seq)
     │   │   - L_spec: FFT-based frequency loss
     │   │   - L_inv_enc/L_inv_dec: UPT invertibility
     │   └─ Output: checkpoints/operator.pt
     │
     ├─→ Stage 2: train_diffusion()
     │   ├─ Load: Operator (frozen teacher)
     │   ├─ Model: DiffusionResidual
     │   ├─ Loss: MSE(diffusion_output, operator_residual)
     │   └─ Output: checkpoints/diffusion_residual.pt
     │
     ├─→ Stage 3: train_consistency()
     │   ├─ Load: DiffusionResidual (teacher)
     │   ├─ Model: DiffusionResidual (student)
     │   ├─ Loss: Distillation MSE
     │   └─ Output: checkpoints/diffusion_residual.pt (overwrite)
     │
     ├─→ Stage 4: train_steady_prior()
     │   ├─ Load: Dataset
     │   ├─ Model: SteadyPrior
     │   ├─ Loss: MSE(prior_output, target_equilibrium)
     │   └─ Output: checkpoints/steady_prior.pt
     │
     └─→ Stage 5: Evaluation
         ├─ Baseline: Operator + optional Diffusion
         ├─ TTC: Test-time conditioning with rewards
         └─ Metrics: NRMSE, MSE, MAE, Physics checks
```

### Key Data Structures

- **LatentPair**: (z0, z1, cond, future, input_fields, coords, meta)
- **Sample**: Unified schema for grid/mesh/particle data
- **LossBundle**: (total, components)
- **CurriculumConfig**: Multi-stage training configuration

### Key Files Summary

| Path | Purpose |
|------|---------|
| `src/ups/training/loop_train.py` | Curriculum-driven training loop |
| `src/ups/training/losses.py` | Loss functions (forward, inverse, spectral) |
| `src/ups/training/consistency_distill.py` | Diffusion distillation |
| `src/ups/data/datasets.py` | Zarr dataset loaders (grid, mesh, particle) |
| `src/ups/data/latent_pairs.py` | Latent pair dataset generation + caching |
| `src/ups/data/parallel_cache.py` | Parallel encoding + cache strategies |
| `src/ups/data/pdebench.py` | PDEBench benchmark loader |
| `src/ups/data/schemas.py` | Unified Sample schema |
| `scripts/train.py` | Main training orchestrator |
| `scripts/run_fast_to_sota.py` | Full pipeline orchestrator |

