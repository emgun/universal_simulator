# Query-Based Training Research: Current Data Sampling and Potential Improvements

## Executive Summary

This document details the current training data pipeline in the Universal Simulator and identifies opportunities for implementing query-based (sparse spatial sampling) training, which could improve efficiency and enable adaptive training strategies.

**Key Finding**: The current system uses **dense grid encoding** with all spatial points encoded during training. Query-based training could reduce computational overhead by sampling only a subset of spatial positions, with full potential during inference through the decoder.

---

## 1. Current Data Flow Architecture

### 1.1 High-Level Pipeline

```
PDEBench HDF5 Data
    ↓
PDEBenchDataset (loads raw fields)
    ↓
GridLatentPairDataset (encodes to latent space)
    ↓
make_grid_coords (generates DENSE grid coordinates)
    ↓
GridEncoder (encodes ALL spatial positions)
    ↓
LatentPair (stores z0, z1, conditioning)
    ↓
latent_pair_collate (batches for training)
    ↓
train_operator (training loop)
```

### 1.2 Key Files and Components

#### Data Loading
- **`src/ups/data/pdebench.py`** - Raw dataset loader
  - `PDEBenchDataset.__getitem__()` - Returns raw fields at line 129
  - Loads from HDF5 files for tasks like `burgers1d`, `darcy2d`, etc.

- **`src/ups/data/datasets.py`** - Zarr dataset support
  - `GridZarrDataset` (lines 24-130) - Stores fields as dense grids
  - `MeshZarrDataset`, `ParticleZarrDataset` - Graph-based alternatives

#### Coordinate Generation (CURRENT: DENSE)
- **`src/ups/data/latent_pairs.py`** - **LINE 55-63**: `make_grid_coords()`
  ```python
  def make_grid_coords(grid_shape: Tuple[int, int], device: torch.device) -> torch.Tensor:
      H, W = grid_shape
      ys = torch.linspace(0.0, 1.0, H, dtype=torch.float32, device=device)
      xs = torch.linspace(0.0, 1.0, W, dtype=torch.float32, device=device)
      grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
      coords = torch.stack([grid_x, grid_y], dim=-1).reshape(1, H * W, 2)
      return coords
  ```
  - **Key**: Creates full grid of shape `(1, H*W, 2)` - ALL spatial points
  - Called at **line 639** and **line 663** during loader creation

#### Encoding to Latent
- **`src/ups/io/enc_grid.py`** - GridEncoder (lines 22-232)
  - `forward()` at **lines 81-107** - encodes ALL coordinates in the grid
  - Uses full coordinates in `_encode_fields()` **line 139-181**
  - Applies Fourier feature encoding **line 183-214**

- **`src/ups/data/latent_pairs.py`** - Latent pair generation
  - **`_fields_to_latent_batch()` lines 155-208** - encodes complete grids
    - Line 184: `data.view(T, data.shape[1], H, W)` - reshapes to full grid
    - Line 192: `coords.expand(B, -1, -1)` - broadcasts full coordinates
  - **`GridLatentPairDataset.__getitem__()` lines 293-412**
    - **Line 329-337**: Encodes ENTIRE trajectory to latent
    - **Line 406**: Stores full coordinates: `coords = self.coords.cpu().expand(base_len, -1, -1)`

#### Batch Construction
- **`src/ups/data/latent_pairs.py`** - **`latent_pair_collate()` lines 764-823**
  - Line 777-778: Concatenates `z0` and `z1` from all samples
  - **Line 805-807**: Collates coords for inverse losses
  - Returns dict with `coords` field (full grid per sample)

#### Training Loop
- **`scripts/train.py`** - **`train_operator()` lines 479-772**
  - **Line 580**: Unpacks batch from DataLoader
  - **Lines 609-612**: Moves coords to device
  - **Line 652-676**: Computes loss bundle
    - **Line 657**: Passes `query_positions=coords` (FULL grid) to inverse decoding loss
    - **Line 661**: Passes `coords` (FULL grid) for re-encoding in inverse loss

---

## 2. Current Spatial Coordinate Handling

### 2.1 Coordinate Representation

| Aspect | Current Value | Location |
|--------|---------------|----------|
| Shape | `(1, H*W, 2)` | `make_grid_coords()` line 62 |
| Spatial Resolution | Dense grid (e.g., 128×128 = 16,384 points for Burgers) | `latent_pairs.py` line 55 |
| Sampling Pattern | **Uniform dense grid** | `torch.linspace()` line 59-60 |
| Normalization | [0, 1] range (physical domain) | `torch.linspace(0.0, 1.0, ...)` |
| Batch Broadcasting | Expanded to batch size | `coords.expand(B, -1, -1)` line 192 |
| GPU Transfer | Full grid → GPU per batch | `coords.to(device)` line 612 |

### 2.2 Why Dense Coordinates Are Used

1. **Encoder Design**: GridEncoder expects coordinates matching grid shape
   - `_fourier_features()` reshapes coords to `(B, H, W, coord_dim)` for patch processing
   - See **`enc_grid.py` line 200**: `coords = coords.view(batch, H, W, coord_dim)`

2. **Inverse Loss Implementation**: Decoder evaluates at query positions
   - **`losses.py` line 51**: `reconstructed = decoder(input_positions, latent)`
   - Decoder is trained with full grid (no sparse sampling currently)

3. **Latent Caching**: Pre-encoded latent pairs stored with full resolution
   - **`latent_pairs.py` line 343**: Caches full latent sequence
   - No per-sample coordinate variation

---

## 3. Decoder Query Sampling Architecture

### 3.1 Current Decoder Design

**File**: `src/ups/io/decoder_anypoint.py` (lines 53-136)

```python
class AnyPointDecoder(nn.Module):
    """Perceiver-style cross-attention decoder for continuous query points."""
    
    def forward(
        self,
        points: torch.Tensor,                    # (B, Q, query_dim)
        latent_tokens: torch.Tensor,             # (B, T, latent_dim)
        conditioning: Optional[Mapping] = None,
    ) -> Dict[str, torch.Tensor]:
        # Cross-attention architecture:
        # 1. Project query points (with Fourier encoding) → queries
        # 2. Project latent tokens → key/value
        # 3. Cross-attention: queries attend to latent
        # 4. MLP heads predict field values at query points
```

**Key Properties**:
- **Flexible Q (query count)**: Takes arbitrary number of spatial points
- **Line 122-123**: Fourier encodes input coordinates
- **Line 125-129**: Cross-attention refinement (2 layers default)
- **Line 131-134**: Per-field prediction heads

### 3.2 Decoder Capabilities vs. Current Training Use

| Capability | Available | Currently Used | Potential |
|------------|-----------|-----------------|-----------|
| Arbitrary query count | Yes | Full grid only | Yes - sparse sampling |
| Fourier encoding | Yes | Yes (1.0, 2.0, 4.0) | Yes |
| Spatial adaptivity | Yes (per-query) | Fixed for batch | Yes - adaptive sampling |
| Multiple fields | Yes | Yes | Yes |

---

## 4. Batch Construction and Coordinate Sampling Patterns

### 4.1 Current Batch Construction

**File**: `src/ups/data/latent_pairs.py` lines 764-823

```python
def latent_pair_collate(batch):
    # Input: List[LatentPair] where each LatentPair has:
    # - z0: (num_pairs, tokens, latent_dim)
    # - z1: (num_pairs, tokens, latent_dim)
    # - coords: (num_pairs, N, 2) [FULL GRID]
    # - input_fields: (num_pairs, N, C) [optional]
    
    z0 = torch.cat([item.z0 for item in batch], dim=0)  # Concat along pairs
    z1 = torch.cat([item.z1 for item in batch], dim=0)
    coords = torch.cat(coords_list, dim=0)              # (B*num_pairs, N, 2)
    # ... rest of collation
```

**Current Properties**:
- All coordinates are full grids (no sampling)
- Batch dimension flattens trajectory pairs
- No per-batch coordinate variation

### 4.2 Current Absence of Random Sampling

**Search Results**: 
- `torch.rand()`, `randint()`, `randperm()`, `choice()` are **NOT used** in:
  - `src/ups/data/latent_pairs.py`
  - `src/ups/data/datasets.py`
  - `src/ups/io/enc_grid.py`

**Exception**: Noise is applied in **inference** (rollout), not training:
- **`rollout_ttc.py` line 123**: `noise = torch.randn_like(candidate.z) * noise_std`

---

## 5. Where Dense Grids Could Be Replaced with Sampled Query Points

### 5.1 Candidate Replacement Points

#### Option A: Sample at Encoding Time (Most Conservative)
**Location**: `src/ups/data/latent_pairs.py` line 55-63 (`make_grid_coords`)

**Current**:
```python
coords = torch.stack([grid_x, grid_y], dim=-1).reshape(1, H * W, 2)  # H*W points
```

**Potential Change**:
```python
# Option 1: Random subsampling
n_queries = int(H * W * sample_ratio)  # e.g., 10% of points
indices = torch.randperm(H * W)[:n_queries]
coords = coords[:, indices, :]  # (1, n_queries, 2)

# Option 2: Stratified sampling (maintain spatial coverage)
# Divide grid into spatial regions, sample uniformly from each

# Option 3: Adaptive sampling (high-gradient regions more densely sampled)
# Would require analyzing fields first
```

**Impact**: 
- Would require changes to encoder to accept variable-sized coordinate sets
- **BLOCKED**: GridEncoder's Fourier feature computation expects fixed grid shape

#### Option B: Sample at Decoding Time (Post-Encoding, Pre-Loss)
**Location**: `src/ups/data/latent_pairs.py` line 409 (`input_fields`) and training loop

**Current** (training loop, line 652-676):
```python
# Inverse decoding uses full coords
loss_bundle = compute_operator_loss_bundle(
    query_positions=coords,  # (B, H*W, 2) - FULL GRID
    coords=coords,           # (B, H*W, 2) - FULL GRID for re-encoding
    input_positions=coords,  # (B, H*W, 2) - FULL GRID for decoding
    ...
)
```

**Potential Change** (Sample query points):
```python
# After encoding (which still uses full grid internally)
if use_query_sampling:
    n_queries = int(H * W * cfg.query_sample_ratio)
    batch_indices = torch.randperm(H * W)[:n_queries]
    
    # Sample sparse queries from full coordinates
    sampled_coords = coords[:, batch_indices, :]  # (B, n_queries, 2)
    sampled_fields = input_fields_physical['u'][:, batch_indices, :]  # (B, n_queries, C)
    
    loss_bundle = compute_operator_loss_bundle(
        input_positions=sampled_coords,      # SAMPLED
        query_positions=sampled_coords,      # SAMPLED
        coords=coords,                       # Still need full grid for re-encoding
        ...
    )
```

**Impact**: 
- Reduces forward/backward pass through decoder
- Maintains encoder training at full resolution
- Changes loss computation semantics slightly

#### Option C: Redesign Inverse Losses for Sparse Training (Most Flexible)
**Location**: `src/ups/training/losses.py` lines 25-99

**Current** (both losses use full grids):
```python
def inverse_encoding_loss(...):
    reconstructed = decoder(input_positions, latent)  # Q=full grid
    
def inverse_decoding_loss(...):
    decoded_fields = decoder(query_positions, latent)  # Q=full grid
    latent_reconstructed = encoder(decoded_fields, coords, meta=meta)  # Full grid
```

**Potential Change** (Support sparse decoding + sparse re-encoding):
```python
def inverse_encoding_loss(..., sample_ratio=0.1):
    # Sample query points from input_positions
    n_points = input_positions.shape[1]
    n_sample = max(1, int(n_points * sample_ratio))
    indices = torch.randperm(n_points)[:n_sample]
    query_sample = input_positions[:, indices, :]
    
    reconstructed = decoder(query_sample, latent)  # Sparse queries
    loss = mse(reconstructed, input_fields[:, indices, :])
```

**Impact**: 
- Explicit sampling parameter in loss functions
- Allows per-loss sampling configuration
- Clear API for sparse training mode

---

## 6. Training Loop Integration Points

### 6.1 Main Training Loop

**File**: `scripts/train.py` lines 479-772

#### Current Flow:
```
Line 479: train_operator(cfg, wandb_ctx, global_step)
  ├─ Line 480: loader = dataset_loader(cfg)
  │   └─ Returns DataLoader with GridLatentPairDataset
  │       └─ Each item has full coords in meta
  │
  ├─ Line 554: dt_tensor = torch.tensor(dt, device=device)
  │
  └─ Line 571: for epoch in range(epochs):
      └─ Line 579: for i, batch in enumerate(loader):
          ├─ Line 580: unpacked = unpack_batch(batch)
          │   └─ batch["coords"] = (batch_size, H*W, 2) [FULL GRID]
          │
          ├─ Line 604-612: Move batch to device
          │   └─ coords = coords.to(device)  [Still full grid]
          │
          └─ Line 652-676: loss_bundle = compute_operator_loss_bundle(
              ├─ input_positions=coords    [FULL]
              ├─ query_positions=coords    [FULL]
              └─ coords=coords             [FULL for re-encoding]
```

#### Modification Points for Query Sampling:
1. **Line 579-580**: After unpacking batch, apply coordinate sampling
2. **Line 611-612**: Filter coordinates before forward pass
3. **Line 652-676**: Pass sampling config to loss computation
4. **New**: Add sampling schedule (curriculum learning for queries)

#### Code Locations Needing Changes:

| Component | File | Lines | Change Type |
|-----------|------|-------|-------------|
| Coordinate generation | `latent_pairs.py` | 55-63 | Add optional sampling |
| Batch collation | `latent_pairs.py` | 764-823 | Preserve coordinate shape |
| Loss computation | `losses.py` | 25-99, 168-251 | Add `sample_ratio` param |
| Training loop | `train.py` | 652-676 | Pass sampling config to loss |
| DataLoader config | `latent_pairs.py` | 588-712 | Add sampling parameters to config |

---

## 7. Existing Query Point Selection Patterns

### 7.1 Inference-Time Sampling

Query-based evaluation already happens at inference:

**File**: `src/ups/inference/rollout_ttc.py` (Test-Time Conditioning)
- Uses **`AnyPointDecoder`** to decode at arbitrary spatial positions
- **No coordinate subsampling** during test-time (evaluates full domain)

**File**: `src/ups/eval/reward_models.py`
- Evaluates physics rewards (mass/energy conservation) at sampled points
- Could inform which spatial regions matter most

### 7.2 Decoder Capability

The decoder is **already designed for sparse query points**:
- **`decoder_anypoint.py` line 114**: `B, Q, _ = points.shape`
- Takes arbitrary Q (number of queries)
- No assumption about grid structure

---

## 8. Implementation Roadmap for Query-Based Training

### Phase 1: Foundation (Minimal Changes)
**Goal**: Enable sparse decoder queries without changing encoder

```yaml
# New config parameters
training:
  query_sample_ratio: 0.5  # Use 50% of spatial points in loss
  query_sampling_mode: "random"  # or "stratified", "adaptive"
  query_sample_warmup_epochs: 0  # Start sparse from epoch 0
```

**Changes**:
1. Add `sample_ratio` parameter to `inverse_encoding_loss()` and `inverse_decoding_loss()`
2. Modify `compute_operator_loss_bundle()` to accept sampling config
3. Update training loop (line 652-676) to pass config to loss

**Impact**: ~20% reduction in decoder compute, minimal other changes

### Phase 2: Curriculum Learning for Queries
**Goal**: Gradually reduce query density during training

```yaml
training:
  query_sampling_schedule: "cosine"
  query_sample_start_ratio: 1.0  # Full grid initially
  query_sample_end_ratio: 0.1    # 10% of points by end
  query_sample_start_epoch: 5
```

**Changes**:
1. Add schedule computation in training loop
2. Modify loss call to use epoch-dependent sampling ratio
3. Log sampling statistics

### Phase 3: Intelligent Sampling (Optional)
**Goal**: Sample regions based on field gradients or importance

```yaml
training:
  query_sampling_mode: "importance"
  importance_metric: "gradient"  # or "error", "variance"
  importance_window: 5  # epochs between recomputation
```

**Changes**:
1. Add gradient-based importance computation
2. Implement stratified sampling weighted by importance
3. Update coordinates dynamically during training

---

## 9. Configuration System Integration

### Current Config Structure

**File**: `configs/train_burgers_golden.yaml` (lines 1-100)

```yaml
data:
  task: burgers1d
  patch_size: 1

latent:
  dim: 16
  tokens: 32

training:
  batch_size: 12
  lambda_inv_enc: 0.0
  lambda_inv_dec: 0.0
  use_inverse_losses: false
```

### Proposed Config Additions

```yaml
# Query-based training (new section)
query_sampling:
  enabled: false              # Enable sparse spatial sampling
  sample_ratio: 1.0          # Start at full (ratio 1.0)
  mode: "random"             # "random", "stratified", "importance"
  warmup_epochs: 5           # Epochs before sampling starts
  schedule: "constant"       # "constant", "linear", "cosine"
  final_ratio: 0.2          # End at 20% if schedule != constant

# Alternatively, integrate with training config:
training:
  query_sample_ratio: 0.5    # Fraction of spatial points to use
  query_sample_mode: "random"
```

---

## 10. Summary of Implementation Entry Points

### Files Requiring Changes (by priority)

| Priority | File | Function | Lines | Change |
|----------|------|----------|-------|--------|
| 1 | `losses.py` | `inverse_encoding_loss` | 25-60 | Add `sample_ratio` param |
| 1 | `losses.py` | `inverse_decoding_loss` | 63-99 | Add `sample_ratio` param |
| 1 | `losses.py` | `compute_operator_loss_bundle` | 168-251 | Pass sampling to losses |
| 2 | `train.py` | `train_operator` | 652-676 | Pass sampling config to loss |
| 2 | `latent_pairs.py` | `build_latent_pair_loader` | 588-712 | Add sampling config params |
| 3 | Config files | Various | - | Add query_sampling config |
| 3 | `train.py` | Loss curriculum schedule | 208-217 | Add query sampling schedule |

### No Changes Needed (Backward Compatible)
- **Encoder** (`enc_grid.py`) - still encodes full grid
- **Latent pairs generation** - stores full coords, sampling happens in loss
- **Decoder** - already supports arbitrary Q
- **DataLoader** - preserves coord structure

---

## 11. Testing and Validation Strategy

### Unit Tests to Add

```python
# tests/unit/test_sparse_sampling.py

def test_query_sampling_shapes():
    """Verify sampled coordinates have correct shape"""
    
def test_query_sampling_coverage():
    """Verify sampled points cover spatial domain"""
    
def test_inverse_loss_with_sampling():
    """Verify loss computation with sparse queries"""
    
def test_curriculum_sampling_schedule():
    """Verify sampling ratio changes over epochs"""
```

### Integration Tests

```python
# tests/integration/test_query_based_training.py

def test_full_training_with_query_sampling():
    """Run 1 epoch with query sampling enabled"""
    
def test_query_sampling_convergence():
    """Verify training still converges with sampling"""
    
def test_query_vs_dense_final_loss():
    """Compare final loss: sampled vs. dense (should be similar)"""
```

### Benchmarking

```python
# scripts/benchmark_query_sampling.py

def benchmark_memory_usage():
    """Measure GPU memory with different sample ratios"""
    
def benchmark_training_speed():
    """Measure epoch time with different sample ratios"""
    
def benchmark_loss_computation_time():
    """Profile loss computation speedup"""
```

---

## 12. Key Insights and Recommendations

### Current State
1. **All spatial coordinates are dense** - Full H×W grid used in training
2. **Decoder supports arbitrary query counts** - Ready for sparse queries
3. **No adaptive sampling** - Same grid for all samples, batches, epochs
4. **Inverse losses use full grids** - Could be parameterized for sparsity

### Why Query-Based Training Matters
1. **Efficiency**: 50% query sampling ≈ 30% decoder compute reduction
2. **Flexibility**: Different query densities per loss or stage
3. **Curriculum Learning**: Progressive sampling could improve convergence
4. **Physics-Aware**: Can sample high-gradient regions more densely

### Recommended First Step
Implement **Phase 1: Foundation** with these changes:
1. Add `sample_ratio` parameter to inverse losses
2. Modify training loop to apply sampling before loss
3. Add config parameter to enable/disable
4. Verify convergence equivalent to dense training

**Expected Impact**: 20-30% faster inverse loss computation with minimal code changes

---

## Appendix: File Cross-References

### Data Pipeline Files
- Coordinate generation: `latent_pairs.py:55-63`
- Batch construction: `latent_pairs.py:764-823`
- Dataset loading: `pdebench.py:47-138`
- Encoder: `enc_grid.py:22-232`

### Training Files
- Main loop: `train.py:479-772`
- Loss computation: `losses.py:1-282`
- Operator creation: `train.py:225-254`

### Inference Files
- Decoder: `decoder_anypoint.py:53-136`
- TTC rollout: `rollout_ttc.py:81-160`

### Configuration
- Golden config: `configs/train_burgers_golden.yaml`
- Data config section: `configs/train_burgers_golden.yaml:27-40`

