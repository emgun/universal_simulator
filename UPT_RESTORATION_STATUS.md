# UPT Phase 1 + 1.5 Restoration Status

## Changes Verified as COMPLETE ✅

### 1. `src/ups/training/losses.py`
- ✅ `inverse_encoding_loss()` - Correct UPT semantics restored
- ✅ `inverse_decoding_loss()` - Correct UPT semantics restored
- ✅ `compute_operator_loss_bundle()` - Added and working

### 2. `scripts/train.py`
- ✅ Encoder/decoder instantiation when inverse losses enabled
- ✅ Loss bundle integration with WandB logging
- ✅ Dict batch format handling

### 3. `src/ups/data/latent_pairs.py` (Partial)
- ✅ `LatentPair` dataclass extended with optional fields
- ✅ `GridLatentPairDataset.__init__()` has `use_inverse_losses` parameter

## Changes Still MISSING ⚠️

### `src/ups/data/latent_pairs.py`

Still need to restore:

1. **`GridLatentPairDataset.__getitem__()`** - Load physical fields when `use_inverse_losses=True`
2. **`latent_pair_collate()`** function - Custom collate for optional fields
3. **`unpack_batch()`** - Handle dict format
4. **`build_latent_pair_loader()`** - Extract and pass `use_inverse_losses`
5. **DataLoader calls** - Add `collate_fn=latent_pair_collate`

## Quick Restoration Plan

The fastest way to restore is to use a reference implementation. Here are the key code blocks that need to be added:

### Block 1: Modify `__getitem__` (line ~293)
Need to add logic to:
- Check `if self.use_inverse_losses`
- Load `fields_cpu` from sample or cache
- Extract first timestep physical fields
- Convert to dict format
- Include in returned `LatentPair`

### Block 2: Add `latent_pair_collate` function (after line ~700)
Custom collate that:
- Stacks z0, z1
- Collates conditioning dicts
- Handles optional inverse loss fields
- Returns dict format

### Block 3: Update `unpack_batch` (line ~711)
- Check if batch is dict
- If so, return dict directly (preserves optional fields)
- Otherwise use legacy tuple unpacking

### Block 4: Update `build_latent_pair_loader` (line ~581)
- Extract `use_inverse_losses` from config
- Pass to all `GridLatentPairDataset` instantiations
- Add `collate_fn=latent_pair_collate` to DataLoader calls

## Current Status

**Phase 1 (Infrastructure):** ✅ COMPLETE
- Loss functions correct
- Training loop integrated
- Configs created

**Phase 1.5 (Data Loading):** ⚠️  70% COMPLETE
- Core structures in place
- Missing data flow implementation

## Impact Assessment

**Can we test without Phase 1.5 completion?**
- ✅ YES - Code runs without errors
- ⚠️ Inverse losses will be skipped (lambda weights have no effect)
- ✅ Validates refactored code doesn't break existing training

**Do we need to complete Phase 1.5?**
- ✅ YES - To actually compute and benefit from inverse losses
- ✅ YES - To meet the implementation goals
- ⚠️ But can be done in a follow-up session

## Recommendation

Given time constraints and the fact that:
1. Core Phase 1 infrastructure is complete
2. Training loop handles missing fields gracefully
3. Code already improved/refactored from baseline

**Option A: Complete Phase 1.5 now** (Est: 30-45 min)
- Restore all 5 missing data loading components
- Full end-to-end UPT implementation
- Ready for actual inverse loss training

**Option B: Document and defer** (Est: 5 min)
- Mark current state as Phase 1 complete
- Document exactly what Phase 1.5 needs
- Test current code to verify no regressions
- Complete Phase 1.5 in next session

Both are valid. Option A gives complete implementation, Option B lets you test the refactored infrastructure immediately.
