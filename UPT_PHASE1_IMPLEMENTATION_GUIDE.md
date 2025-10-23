# UPT Phase 1 Implementation Guide: Inverse Losses

**Date**: 2025-01-23
**Status**: ✅ Code Complete - Data Pipeline Integration Pending
**Estimated Time to Full Integration**: 1-2 days

---

## 🎯 What Was Accomplished

### ✅ Completed

1. **UPT Loss Functions Implemented** (`src/ups/training/losses.py`)
   - `upt_inverse_encoding_loss()` - Lines 58-123
   - `upt_inverse_decoding_loss()` - Lines 126-197
   - Full docstrings with algorithm descriptions
   - Query point sampling for computational efficiency
   - Backward compatible with legacy losses

2. **Training Script Updated** (`scripts/train.py`)
   - Added UPT loss imports (line 46)
   - Added config parameters for UPT losses (lines 428-439)
   - Added documented placeholder for integration (lines 481-523)
   - Preserved existing training loop (no breaking changes)

3. **Configuration Created** (`configs/train_burgers_upt_losses.yaml`)
   - Based on golden config for fair comparison
   - Added `lambda_inv_enc`, `lambda_inv_dec` weights
   - Added query point sampling parameters
   - Comprehensive implementation notes

4. **Analysis Document** (`UPT_INTEGRATION_ANALYSIS.md`)
   - 11 identified gaps between UPS and UPT
   - Prioritized improvement roadmap
   - Expected impact estimates
   - Risk assessments

---

## 🚧 What's Needed for Full Integration

### Required Changes

#### 1. Data Pipeline Modifications (`src/ups/data/latent_pairs.py`)

**Current State**: Data loader only provides pre-encoded latent pairs
```python
# Current batch format
batch = {
    "z0": Tensor,      # (B, tokens, latent_dim)
    "z1": Tensor,      # (B, tokens, latent_dim)
    "cond": Dict[str, Tensor],
    "future": Optional[Tensor],
}
```

**Required State**: Must also provide original fields and coordinates
```python
# Required batch format
batch = {
    "z0": Tensor,      # (B, tokens, latent_dim)
    "z1": Tensor,      # (B, tokens, latent_dim)
    "cond": Dict[str, Tensor],
    "future": Optional[Tensor],
    # NEW: For UPT inverse losses
    "fields_orig": Dict[str, Tensor],  # (B, num_points, channels)
    "coords": Tensor,                   # (B, num_points, coord_dim)
    "meta": Dict[str, Any],             # {grid_shape, ...}
}
```

**Implementation Path**:

**Option A: Modify `GridLatentPairDataset.__getitem__()` (lines 290-350)**
```python
def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
    # ... existing code to load sample ...

    # EXISTING: Encode to latent
    latent_seq = _fields_to_latent_batch(...)

    # NEW: Also keep original fields and coords
    return {
        "latent": latent_seq,
        "params": params_cpu,
        "bc": bc_cpu,
        # NEW additions
        "fields_orig": {"u": fields[start:end+1]},  # Original physical fields
        "coords": self.coords.cpu(),                 # Spatial coordinates
        "meta": {"grid_shape": self.grid_shape},     # Metadata
    }
```

**Option B: Add separate dataset class `GridLatentPairDatasetWithFields`**
- Inherit from `GridLatentPairDataset`
- Override `__getitem__()` to include original fields
- Use conditional flag in config to select dataset class

**Recommendation**: Option A - simpler, but increases memory usage ~2x

#### 2. Encoder/Decoder Module Access (`scripts/train.py`)

**Current State**: Encoder created in data loader, not accessible in training loop

**Required Changes**:

**Step 1**: Store encoder/decoder in latent dataset
```python
# In GridLatentPairDataset.__init__()
class GridLatentPairDataset(Dataset):
    def __init__(self, base, encoder, ...):
        self.encoder = encoder  # Already exists
        # Store for later retrieval
```

**Step 2**: Retrieve from data loader
```python
# In train_operator()
loader = dataset_loader(cfg)

# NEW: Get encoder from dataset
if hasattr(loader.dataset, 'encoder'):
    encoder = loader.dataset.encoder
else:
    encoder = None

# NEW: Create decoder
if encoder is not None:
    from ups.io.decoder_anypoint import AnyPointDecoder, AnyPointDecoderConfig
    decoder_cfg = AnyPointDecoderConfig(
        latent_dim=cfg["latent"]["dim"],
        query_dim=2,  # 2D for Burgers/grids
        hidden_dim=cfg["operator"]["pdet"]["hidden_dim"],
        num_layers=2,
        num_heads=4,
        output_channels={"u": 1},  # Burgers has 1-channel u field
    )
    decoder = AnyPointDecoder(decoder_cfg).to(device).eval()
else:
    decoder = None
```

**Step 3**: Update lines 438-439 in train.py
```python
# BEFORE:
encoder = None  # TODO: Pass encoder from data loader
decoder = None  # TODO: Create decoder module

# AFTER:
# encoder and decoder set above (see Step 2)
```

**Step 4**: Uncomment UPT loss integration (lines 497-522)
```python
# Remove the '#' comments to enable
if use_upt_losses and encoder is not None and decoder is not None:
    if "fields_orig" in batch and "coords" in batch:
        # ... loss computation code ...
```

---

## 📋 Step-by-Step Integration Checklist

### Phase 1A: Minimal Working Integration (1 day)

- [ ] **Modify `GridLatentPairDataset.__getitem__()`**
  - [ ] Add `"fields_orig"` to return dict
  - [ ] Add `"coords"` to return dict
  - [ ] Add `"meta"` to return dict
  - [ ] Test: Verify batch dict has new keys

- [ ] **Update `train_operator()` in `scripts/train.py`**
  - [ ] Retrieve encoder from dataset
  - [ ] Create decoder module
  - [ ] Uncomment UPT loss integration code (lines 497-522)
  - [ ] Test: Run 1 epoch without errors

- [ ] **Validate Locally**
  ```bash
  python scripts/train.py \
    --config configs/train_burgers_upt_losses.yaml \
    --stage operator \
    --epochs 1
  ```
  - [ ] Training completes without errors
  - [ ] UPT losses appear in logs
  - [ ] Loss values are reasonable (< 1.0)

### Phase 1B: Full Training Run (1 day)

- [ ] **Launch Full Training**
  ```bash
  python scripts/vast_launch.py launch \
    --config configs/train_burgers_upt_losses.yaml \
    --auto-shutdown
  ```

- [ ] **Monitor Training**
  - [ ] Check WandB dashboard for loss curves
  - [ ] Verify `loss_inv_enc` and `loss_inv_dec` decrease
  - [ ] Check total training time (expect +10-20% vs golden)

- [ ] **Evaluate Results**
  ```bash
  python scripts/compare_runs.py \
    <golden_run_id> \
    <upt_losses_run_id>
  ```
  - [ ] Compare NRMSE (expect -10 to -20%)
  - [ ] Check correlation time (expect improvement)
  - [ ] Analyze rollout stability

---

## 💡 Implementation Tips

### Memory Management

**Problem**: Adding original fields to batch doubles memory usage

**Solutions**:
1. **Reduce batch size**: Drop from 12 → 8 or 6
2. **Use gradient accumulation**: Increase `accum_steps` to maintain effective batch size
3. **Sample fewer query points**: Use 1024-2048 instead of all points

### Computational Cost

**Expected Overhead**: +15-25% training time

**Why**:
- Inverse encoding: 1 decoder forward pass
- Inverse decoding: 1 decoder + 1 encoder forward pass
- Total: ~2 extra forward passes per batch

**Optimization**:
- Reduce `inv_enc_query_points` and `inv_dec_query_points`
- Use AMP (already enabled)
- Consider torch.compile for decoder

### Debugging

**If inverse losses are NaN/Inf**:
1. Check decoder output shapes match `fields_orig`
2. Verify coordinate shapes are correct
3. Add gradient clipping (already enabled at 1.0)
4. Reduce loss weights (try 0.1 instead of 0.5)

**If training is slow**:
1. Profile with `torch.profiler`
2. Reduce query points
3. Reduce decoder depth (2 layers → 1 layer)

---

## 📊 Expected Results

### Baseline (Golden Config)
- **Operator final loss**: ~0.00023
- **Validation NRMSE**: ~0.078
- **Training time**: ~14.5 min (RTX 4090)

### With UPT Inverse Losses (Expected)
- **Operator final loss**: ~0.00015-0.00020 (10-30% improvement)
- **Validation NRMSE**: ~0.060-0.070 (10-20% improvement)
- **Training time**: ~17-18 min (+15-20% overhead)
- **Correlation time**: Improved (longer stable rollouts)

### Success Criteria
- ✅ Both inverse losses decrease during training
- ✅ Validation NRMSE < 0.075 (better than golden's 0.078)
- ✅ Training time < 20 min (acceptable overhead)
- ✅ No OOM errors with batch_size=8

---

## 🔬 Validation Tests

### Unit Tests (`tests/unit/test_losses.py`)

Create tests for UPT loss functions:

```python
def test_upt_inverse_encoding_loss():
    """Test inverse encoding loss computation."""
    batch_size, num_points, coord_dim = 4, 64, 2
    latent_dim, num_tokens = 16, 32

    # Mock data
    fields = {"u": torch.randn(batch_size, num_points, 1)}
    coords = torch.randn(batch_size, num_points, coord_dim)
    latent = torch.randn(batch_size, num_tokens, latent_dim)

    # Mock decoder
    class MockDecoder(nn.Module):
        def forward(self, points, latent_tokens):
            B, Q, _ = points.shape
            return {"u": torch.randn(B, Q, 1)}

    decoder = MockDecoder()

    # Compute loss
    loss = upt_inverse_encoding_loss(
        fields, coords, latent, decoder, num_query_points=32
    )

    # Assertions
    assert loss.ndim == 0  # Scalar
    assert loss.item() >= 0  # Non-negative
    assert not torch.isnan(loss)
```

### Integration Test

```bash
# Test 1 epoch with UPT losses
pytest tests/integration/test_train_with_upt_losses.py -v
```

---

## 📚 Reference Documentation

### Key Files Modified

| File | Lines | Changes |
|------|-------|---------|
| `src/ups/training/losses.py` | 58-197 | Added UPT loss functions |
| `scripts/train.py` | 46, 428-439, 481-523 | Imports, config, placeholder |
| `configs/train_burgers_upt_losses.yaml` | All | New experimental config |

### Related Documents

- **`UPT_INTEGRATION_ANALYSIS.md`**: Full gap analysis and roadmap
- **`UPT_docs/UPT_Implementation_Plan.md`**: Original UPT implementation guide
- **`UPT_docs/UPT_Arch_Train_Scaling.md`**: UPT architecture details

### External References

- **UPT Paper**: arXiv:2402.12365, Section 5.2 (Inverse Losses)
- **Perceiver-IO**: Cross-attention decoder architecture
- **NeRF**: Fourier feature positional encoding

---

## 🚀 Next Steps After Phase 1

Once inverse losses are working:

### Phase 2: Scale-Up Experiment (2-4 weeks)
- Increase latent tokens from 16 → 256
- Increase latent dim from 16 → 192
- Benchmark on UPT-17M configuration
- Expected +20-40% NRMSE improvement

### Phase 3: Architecture Simplification (4-6 weeks)
- Replace U-shaped PDE-Transformer with pure transformer
- Simplify training code
- Improve scalability

### Phase 4: Advanced Features (6-8 weeks)
- Query-based training (sample random points)
- Physics priors (divergence penalties)
- CFD encoder for mesh/particle domains

---

## ❓ FAQ

**Q: Why not just train with inverse losses disabled first?**
A: The config works as-is with `use_upt_inverse_losses=false`. The data pipeline changes are needed to enable them.

**Q: Will this break existing checkpoints?**
A: No. The model architecture is unchanged. Old checkpoints load fine.

**Q: Can I test UPT losses without modifying data pipeline?**
A: Not easily. The losses require original fields which aren't in current batches.

**Q: What if I only want inverse encoding loss, not inverse decoding?**
A: Set `lambda_inv_dec=0.0` in config. Inverse decoding is more expensive (2 forward passes).

**Q: How do I disable UPT losses for ablation study?**
A: Set `use_upt_inverse_losses=false` in config. Uses golden config behavior exactly.

---

## 📞 Support

For questions or issues:
1. Check this guide and `UPT_INTEGRATION_ANALYSIS.md`
2. Review UPT paper Section 5.2
3. Create GitHub issue with error logs

---

**Ready to integrate?** Start with Phase 1A checklist above. Expected time: 1 day for minimal integration, 2 days for full validation.
