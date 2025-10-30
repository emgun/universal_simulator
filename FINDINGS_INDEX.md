# Training Pipeline Infrastructure - Detailed Findings Index

## File Locations (Absolute Paths)

### Core Training Implementation
- `/Users/emerygunselman/Code/universal_simulator/src/ups/training/loop_train.py`
- `/Users/emerygunselman/Code/universal_simulator/src/ups/training/losses.py`
- `/Users/emerygunselman/Code/universal_simulator/src/ups/training/consistency_distill.py`
- `/Users/emerygunselman/Code/universal_simulator/scripts/train.py`
- `/Users/emerygunselman/Code/universal_simulator/scripts/run_fast_to_sota.py`

### Data Management
- `/Users/emerygunselman/Code/universal_simulator/src/ups/data/datasets.py`
- `/Users/emerygunselman/Code/universal_simulator/src/ups/data/latent_pairs.py`
- `/Users/emerygunselman/Code/universal_simulator/src/ups/data/parallel_cache.py`
- `/Users/emerygunselman/Code/universal_simulator/src/ups/data/pdebench.py`
- `/Users/emerygunselman/Code/universal_simulator/src/ups/data/schemas.py`

---

## Specific Implementation Details

### 1. LatentTrainer (Training Loop)
**File**: `/Users/emerygunselman/Code/universal_simulator/src/ups/training/loop_train.py`
**Lines**: 27-106

**Key Methods**:
- `LatentTrainer.__init__()` (lines 30-47): Initialization with device, optimizer, dataloader
- `LatentTrainer.train()` (lines 56-74): Main curriculum loop with cyclic batch iteration
- `LatentTrainer._train_step()` (lines 76-105): Single step loss computation
- `LatentTrainer._apply_ema()` (lines 49-54): EMA weight update

**EMA Configuration**:
- Optional feature controlled by `curriculum.ema_decay`
- Creates frozen copy of model (`ema_model`)
- Updates with decay factor: `p_ema = decay * p_ema + (1-decay) * p`

### 2. Multi-Stage Training Functions
**File**: `/Users/emerygunselman/Code/universal_simulator/scripts/train.py`

#### Stage 1: Operator
- **Function**: `train_operator()` (lines 400-693)
- **Key Lines**:
  - Encoder/decoder init for inverse losses: lines 417-473
  - Main training loop: lines 492-666
  - Loss computation: line 559-591
  - Best state tracking: lines 652-659
  - Checkpoint save: lines 669-685
- **Output**: `checkpoints/operator.pt` + `operator_ema.pt`

#### Stage 2: Diffusion
- **Function**: `train_diffusion()` (lines 695-839)
- **Key Lines**:
  - Operator loading: lines 704-711
  - Training loop: lines 753-830
  - Residual computation: line 776
  - Tau sampling: line 778
  - Checkpoint save: lines 833-839
- **Output**: `checkpoints/diffusion_residual.pt`

#### Stage 3: Consistency Distillation
- **Function**: `train_consistency()` (lines 841-1020)
- **Key Lines**:
  - Config adjustment for batch size: lines 854-856
  - Model loading: lines 864-881
  - Teacher-student loss: lines 913-920
- **Output**: `checkpoints/diffusion_residual.pt` (overwrite)

#### Stage 4: Steady Prior
- **Function**: `train_steady_prior()` (lines 1022-1309)
- **Key Lines**:
  - Independent training setup: lines 1022-1080
  - Checkpoint save: lines 1287-1300
- **Output**: `checkpoints/steady_prior.pt`

#### Pipeline Orchestration
- **Function**: `train_all_stages()` (lines 1516-1733)
- **Key Lines**:
  - Stage 1: lines 1568-1579
  - Stage 2: lines 1586-1598
  - Stage 3: lines 1604-1616
  - Stage 4: lines 1622-1633
  - GPU cache clearing: lines 1581-1584, 1599-1602, 1617-1620
  - Evaluation: lines 1634-1707
  - WandB finish: line 1728

- **Entry Point**: `main()` (lines 1735-1824)

### 3. Loss Functions
**File**: `/Users/emerygunselman/Code/universal_simulator/src/ups/training/losses.py`

#### LossBundle Structure (lines 13-18)
```python
@dataclass
class LossBundle:
    total: Tensor                      # Weighted sum
    components: Dict[str, Tensor]      # Individual terms
```

#### Core Loss Functions
1. `mse()` (lines 21-22): F.mse_loss wrapper
2. `one_step_loss()` (lines 102-103): L_forward = MSE(pred_next, target_next)
3. `rollout_loss()` (lines 106-109): L_rollout = MSE(sequences)
4. `spectral_loss()` (lines 112-116): L_spec = MSE(|FFT|)
5. `consistency_loss()` (lines 119-122): Mean preservation
6. `edge_total_variation()` (lines 125-131): TV regularization

#### UPT Inverse Losses
1. `inverse_encoding_loss()` (lines 25-60)
   - Ensures: latent → decoder → reconstructed_fields matches input
   - Loss: MSE in physical space

2. `inverse_decoding_loss()` (lines 63-99)
   - Ensures: latent → decoder → re-encoder matches latent
   - Loss: MSE in latent space

#### Curriculum Learning
- `compute_inverse_loss_curriculum_weight()` (lines 134-165)
  - Phase 1 (0 to warmup_epochs): weight = 0
  - Phase 2 (warmup to 2*warmup): linear ramp
  - Phase 3 (>2*warmup): weight = min(base_weight, max_weight)

#### Loss Bundle Composition
- `compute_operator_loss_bundle()` (lines 168-251)
  - Input parameters (all optional):
    - `pred_next`, `target_next` → L_forward
    - `pred_rollout`, `target_rollout` → L_rollout
    - `spectral_pred`, `spectral_target` → L_spec
    - `input_fields`, `encoded_latent`, `decoder` → L_inv_enc
    - `decoder`, `encoder`, `query_positions` → L_inv_dec
  - Weight keys:
    - `lambda_forward`, `lambda_rollout`, `lambda_spectral`
    - `lambda_inv_enc`, `lambda_inv_dec`
    - `inverse_loss_warmup_epochs`, `inverse_loss_max_weight`
  - Returns: LossBundle with robustly summed components

### 4. Dataset Classes
**File**: `/Users/emerygunselman/Code/universal_simulator/src/ups/data/datasets.py`

1. **GridZarrDataset** (lines 24-131)
   - Zarr layout: `coords (N,2)`, `fields/<name> (T,H,W,C)`
   - Returns: Sample dict with fields, coords, time, metadata
   - __getitem__: lines 94-130

2. **MeshZarrDataset** (lines 133-195)
   - Stores CSR sparse Laplacian in cache
   - __getitem__: lines 161-194

3. **ParticleZarrDataset** (lines 197-258)
   - Neighbor graph as CSR indices/indptr
   - __getitem__: lines 219-257

### 5. Latent Pair Dataset Generation
**File**: `/Users/emerygunselman/Code/universal_simulator/src/ups/data/latent_pairs.py`

#### LatentPair Dataclass (lines 245-254)
```python
@dataclass
class LatentPair:
    z0: torch.Tensor                           # Initial state
    z1: torch.Tensor                           # Next state
    cond: Dict[str, torch.Tensor]              # Conditioning
    future: Optional[torch.Tensor] = None      # Multi-step rollout
    input_fields: Optional[Dict[...]] = None   # UPT inverse loss
    coords: Optional[torch.Tensor] = None      # UPT inverse loss
    meta: Optional[Dict] = None                # Metadata
```

#### GridLatentPairDataset (lines 257-413)
- **Caching**: `cache_dir / f"sample_{idx:05d}.pt"`
- **Cache format**: `{"latent", "params", "bc"}`
- **Time stride**: lines 360-363
- **Inverse loss support**: lines 378-411
  - Stores input_fields, coords, meta per sample

#### GraphLatentPairDataset (lines 472-543)
- Per-timestep encoding for mesh/particle data

#### Collate Function (lines 723-783)
- `latent_pair_collate()`: Main collate for training
- Returns dict: `{"z0", "z1", "cond", "future", "input_fields", "coords", "meta"}`
- Handles optional inverse loss fields

#### Loader Builder (lines 588-720)
- `build_latent_pair_loader()`: Creates complete DataLoader
- Supports:
  - Single task or list of tasks (multi-dataset mixing)
  - Grid, mesh, particle data
  - Caching with dtype conversion
  - Worker configuration

### 6. Cache Strategies
**File**: `/Users/emerygunselman/Code/universal_simulator/src/ups/data/parallel_cache.py`

1. **RawFieldDataset** (lines 33-88)
   - Returns raw fields for worker-based loading
   - __getitem__: returns `{"idx", "cached", "fields", "params", "bc", "cache_path"}`

2. **PreloadedCacheDataset** (lines 90-167)
   - Preloads entire cache to RAM
   - __init__: lines 97-132 (cache loading)
   - __getitem__: lines 138-166 (instant access)

3. **make_collate_with_encoding()** (lines 169-256)
   - Creates collate function with GPU encoding
   - Encodes on-demand if not cached
   - Saves to cache after encoding

4. **Helper Functions**
   - `check_cache_complete()` (lines 313-324)
   - `estimate_cache_size_mb()` (lines 327-349)
   - `check_sufficient_ram()` (lines 351-361)

### 7. PDEBench Integration
**File**: `/Users/emerygunselman/Code/universal_simulator/src/ups/data/pdebench.py`

- **PDEBenchDataset** (lines 47-138)
  - __init__: lines 50-124 (HDF5 loading)
  - __getitem__: lines 129-138 (single sample)
  - Supports multi-shard loading
  - Optional field normalization

- **Task Specs** (lines 23-28)
  - Maps task names to HDF5 keys
  - Supports: burgers1d, advection1d, darcy2d, navier_stokes2d

### 8. Data Schema
**File**: `/Users/emerygunselman/Code/universal_simulator/src/ups/data/schemas.py`

- **Sample TypedDict** (lines 11-37)
  - kind, coords, connect, fields, bc, params, geom, time, dt, meta

- **Validation** (lines 62-117)
  - `validate_sample()`: Strict schema checking
  - Field dimension validation
  - Type checking (float, int tensors)

### 9. Checkpoint Management
**File**: `/Users/emerygunselman/Code/universal_simulator/scripts/train.py`

- **ensure_checkpoint_dir()** (lines 127-131)
  - Creates `checkpoints/` directory

- **_strip_compiled_prefix()** (lines 82-88)
  - Removes `_orig_mod.` prefix from torch.compile

- **_stage_epochs()** (lines 391-397)
  - Returns configured epochs for stage (0 to skip)

- **Stage-specific Save Locations**:
  - Operator: line 671
  - Diffusion: line 836
  - Consistency: overwrites diffusion
  - Steady Prior: line 1293

### 10. Helper Utilities

**Configuration Loading** (lines 91-93)
- `load_config()`: YAML loading

**Seed & Determinism** (lines 96-124)
- `set_seed()`: Random seed + torch config

**Dataset Loading** (lines 216-222)
- `dataset_loader()`: DataLoader creation from config

**Model Creation** (lines 225-242)
- `make_operator()`: Instantiate LatentOperator

**Optimizer & Scheduler** (lines 245-295)
- `_create_optimizer()`: Adam/AdamW/SGD
- `_create_scheduler()`: StepLR/CosineAnnealingLR/ReduceLROnPlateau

**Optimization Utilities**
- `_amp_enabled()` (lines 298-299): Check AMP
- `_maybe_compile()` (lines 302-333): torch.compile wrapper
- `_grad_clip_value()` (lines 335-340): Per-stage grad clip
- `_get_ema_decay()` (lines 343-347): Per-stage EMA decay

**EMA Management**
- `_init_ema()` (lines 350-355): Create frozen copy
- `_update_ema()` (lines 358-361): Apply EMA update

**Early Stopping**
- `_get_patience()` (lines 364-369): Get patience per-stage
- `_should_stop()` (lines 385-388): Check stop condition

**Training Logger** (lines 134-213)
- `TrainingLogger` class for JSONL + WandB logging
- log() method: line 155-206
- Tracks: loss, lr, grad_norm, epochs_since_improve

### 11. Key Configuration Sections

**Data** (example):
```yaml
data:
  task: "burgers1d"
  split: "train"
  root: "/path/to/pdebench"
  field_name: "u"
  patch_size: 4
```

**Training** (example):
```yaml
training:
  batch_size: 16
  amp: true
  compile: true
  grad_clip: 1.0
  ema_decay: 0.999
  dt: 0.1
  use_inverse_losses: false
  lambda_forward: 1.0
  latent_cache_dir: "data/latent_cache"
```

**Stages** (example):
```yaml
stages:
  operator:
    epochs: 25
    optimizer: {name: adamw, lr: 1e-3}
    scheduler: {name: cosineannealinglr, t_max: 25}
```

### 12. WandB Integration Points

- **Training metrics logging** (line 614)
- **Stage completion alerts** (lines 688-692, 836-839, 1007-1011, 1302-1308)
- **Checkpoint upload** (lines 681, 834, 1003, 1299)
- **Evaluation summary** (lines 1415, 1434-1441)
- **Model metrics tables** (lines 1483-1513)

---

## Summary Statistics

- **Total Lines Documented**: ~3,000+
- **Functions Documented**: 50+
- **Classes Documented**: 15+
- **Configuration Options**: 40+
- **Loss Functions**: 10+
- **Dataset Loaders**: 3
- **Cache Strategies**: 3

All paths are absolute and verified.
All line numbers are current.
