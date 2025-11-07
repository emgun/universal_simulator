# Query-Based Training: Quick Reference

## Current State: Dense Grid Training

All spatial coordinates are **dense uniform grids** during training:

```
Training Data Pipeline:
┌─────────────────────────────────────────────────────────────────┐
│ Raw Fields (H×W grid)                                           │
│        ↓                                                         │
│ make_grid_coords() generates:                                   │
│   ys = linspace(0, 1, H)                                        │
│   xs = linspace(0, 1, W)                                        │
│   coords = (1, H*W, 2) ← FULL GRID                             │
│        ↓                                                         │
│ GridEncoder encodes at ALL spatial positions                    │
│   _fourier_features() reshapes to (B, H, W, coord_dim)        │
│        ↓                                                         │
│ Inverse Losses use FULL coordinates:                            │
│   query_positions = coords  (H*W points)                        │
│   input_positions = coords  (H*W points)                        │
└─────────────────────────────────────────────────────────────────┘
```

## Key Files (File:Line References)

### Coordinate Generation (DENSE → SAMPLE)
**`src/ups/data/latent_pairs.py:55-63`**
```python
def make_grid_coords(grid_shape, device):
    H, W = grid_shape
    ys = torch.linspace(0.0, 1.0, H)      # line 59
    xs = torch.linspace(0.0, 1.0, W)      # line 60
    grid_y, grid_x = torch.meshgrid(...)  # line 61
    coords = torch.stack([...]).reshape(1, H * W, 2)  # line 62
    return coords
```
**Status**: Dense grid generation
**Opportunity**: Could add optional sampling, but blocks encoder Fourier features

### Batch Construction (Full Coords)
**`src/ups/data/latent_pairs.py:764-823`**
```python
def latent_pair_collate(batch):
    z0 = torch.cat([item.z0 for item in batch], dim=0)
    z1 = torch.cat([item.z1 for item in batch], dim=0)
    coords = torch.cat(coords_list, dim=0)  # line 805-807
    # Returns dict with full coords
```
**Status**: Preserves full grid from dataset
**No Change Needed**: Just passes data through

### Training Loop (Applies coords to loss)
**`scripts/train.py:652-676`**
```python
loss_bundle = compute_operator_loss_bundle(
    query_positions=coords,          # line 657 - FULL GRID
    input_positions=coords,          # line 661 - FULL GRID
    coords=coords,                   # line 661 - FULL GRID
    # ... other inputs
)
```
**Status**: Passes full coords to loss functions
**Primary Modification Point**: Apply sampling before passing to loss

### Loss Functions (Use full coords)
**`src/ups/training/losses.py:25-99`**
```python
def inverse_encoding_loss(input_fields, latent, decoder, input_positions, ...):
    reconstructed = decoder(input_positions, latent)  # line 51
    # input_positions = (B, H*W, 2) currently

def inverse_decoding_loss(latent, decoder, encoder, query_positions, coords, ...):
    decoded_fields = decoder(query_positions, latent)  # line 93
    # query_positions = (B, H*W, 2) currently
```
**Status**: Hard-coded for full grids
**Modification Point**: Add `sample_ratio` parameter

### Decoder (Already Flexible)
**`src/ups/io/decoder_anypoint.py:53-136`**
```python
def forward(self, points, latent_tokens, ...):
    B, Q, _ = points.shape  # line 114
    # Takes arbitrary Q (number of queries)
    # Ready for sparse sampling!
```
**Status**: Already supports arbitrary query counts
**No Change Needed**: Decoder is ready

## Implementation Path: Phase 1 (Foundation)

### Step 1: Modify Loss Functions
**File**: `src/ups/training/losses.py`

Add `sample_ratio=None` parameter to both inverse losses:

```python
def inverse_encoding_loss(
    input_fields,
    latent,
    decoder,
    input_positions,
    weight=1.0,
    sample_ratio=None,  # NEW
):
    if sample_ratio is not None and sample_ratio < 1.0:
        # Sample subset of query points
        n_points = input_positions.shape[1]
        n_sample = max(1, int(n_points * sample_ratio))
        indices = torch.randperm(n_points)[:n_sample]
        input_positions = input_positions[:, indices, :]
        input_fields = {k: v[:, indices, :] for k, v in input_fields.items()}
    
    reconstructed = decoder(input_positions, latent)
    # ... rest unchanged
```

### Step 2: Modify Loss Bundle
**File**: `src/ups/training/losses.py:168-251`

Add sampling to `compute_operator_loss_bundle()`:

```python
def compute_operator_loss_bundle(
    *,
    # ... existing params ...
    weights=None,
    query_sample_ratio=None,  # NEW
    current_epoch=None,
):
    # Extract sample ratio from weights or use explicit param
    sample_ratio = weights.get("query_sample_ratio", query_sample_ratio) if weights else query_sample_ratio
    
    # Pass to inverse losses
    if all(x is not None for x in [...]):
        comp["L_inv_enc"] = inverse_encoding_loss(
            # ... existing args ...
            sample_ratio=sample_ratio,  # NEW
        )
```

### Step 3: Modify Training Loop
**File**: `scripts/train.py:652-676`

Extract sampling config and pass to loss:

```python
# Around line 649
query_sample_ratio = float(train_cfg.get("query_sample_ratio", 1.0))

loss_bundle = compute_operator_loss_bundle(
    # ... existing args ...
    weights={
        # ... existing weights ...
        "query_sample_ratio": query_sample_ratio,  # NEW
    },
)
```

### Step 4: Add Config Parameter
**File**: `configs/train_*.yaml`

```yaml
training:
  query_sample_ratio: 1.0    # Default: full grid (no sampling)
  # Set to 0.5 for 50% sampling
  # Set to 0.1 for 10% sampling
```

## Performance Impact

### Estimated Speedup (Phase 1)
- **50% sampling (0.5 ratio)**: ~25-30% faster inverse loss computation
- **Memory**: ~15-20% reduction in GPU memory for decoder forward/backward
- **Accuracy**: Negligible impact if sample_ratio ≥ 0.3 (based on related work)

### Backward Compatibility
- Default `query_sample_ratio: 1.0` = full grid = current behavior
- No changes to encoder, decoder architecture
- Latent caching unchanged
- All existing configs work without modification

## Testing Checklist

- [ ] Test loss computation with `sample_ratio=1.0` (should match original)
- [ ] Test loss computation with `sample_ratio=0.5`
- [ ] Verify sampled coordinates are properly filtered
- [ ] Run 1 epoch with sampling enabled
- [ ] Compare convergence: sampled vs. dense (should be similar)
- [ ] Benchmark wall-clock time improvement
- [ ] Check GPU memory usage reduction

## Future Phases (Optional)

### Phase 2: Curriculum Learning
```yaml
training:
  query_sampling_schedule: "cosine"
  query_sample_start_ratio: 1.0     # Full grid initially
  query_sample_end_ratio: 0.2       # 20% by end
  query_sample_start_epoch: 10      # Begin scheduling at epoch 10
```

### Phase 3: Adaptive Sampling
```yaml
training:
  query_sampling_mode: "importance"  # vs "random"
  importance_metric: "gradient"      # vs "error", "variance"
  importance_window: 5               # Recompute every 5 epochs
```

## References

- Full Research: `/docs/query_based_training_research.md`
- Decoder Design: `src/ups/io/decoder_anypoint.py` lines 53-136
- Loss Functions: `src/ups/training/losses.py` lines 1-282
- Training Loop: `scripts/train.py` lines 479-772

---

**Last Updated**: 2025-11-05  
**Status**: Phase 1 (Foundation) ready for implementation
