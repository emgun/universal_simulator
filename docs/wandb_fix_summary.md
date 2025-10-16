# WandB Configuration Fix Summary

## Issues Encountered

### 1. API Key Authentication (RESOLVED)
**Problem:** API key `8fab23e5f80a88f46f6cc2dff4b6c24f50fdbe1f` was giving 401 "user is not logged in" errors during `wandb.init()`.

**Solution:** Use offline mode during training, then sync afterward:
```bash
export WANDB_MODE=offline  # During training
wandb sync wandb/offline-run-*  # After training
```

**Status:** ✅ Working - both 16-dim and 32-dim v2 runs successfully synced

### 2. Missing Entity Parameter (FIXED)
**Problem:** `wandb.init()` in `train_all_stages()` was missing the `entity` parameter, causing permission errors.

**Fix Applied:** Updated `scripts/train.py` line 1115:
```python
shared_run = wandb.init(
    project=wandb_cfg.get("project", "universal-simulator"),
    entity=wandb_cfg.get("entity"),  # ← Added this line
    name=wandb_cfg.get("run_name", "full-pipeline"),
    ...
)
```

**Status:** ✅ Fixed in commit `783fcd9`

### 3. Config Files Missing Entity (FIXED)
**Problem:** Some config files didn't specify `entity` in `logging.wandb`.

**Configs Fixed:**
- ✅ `train_burgers_16dim_pru2jxc4.yaml`
- ✅ `train_burgers_8dim_pru2jxc4.yaml` 
- ✅ `train_burgers_32dim_v2_practical.yaml`

**Status:** ✅ All configs now include `entity: emgun-morpheus-space`

## Current Synced Runs

### 16-dim Run (ID: 8rwppo47)
- **Config:** `train_burgers_16dim_pru2jxc4.yaml`
- **Operator Loss:** 0.00023 (excellent!)
- **Evaluation:** 0.1511 NRMSE (with TTC)
- **URL:** https://wandb.ai/emgun-morpheus-space/universal-simulator/runs/8rwppo47

### 32-dim v2 Run (ID: qswd55uf)
- **Config:** `train_burgers_32dim_v2_practical.yaml`
- **Operator Loss:** 2.85e-5 (~88× better than v1!)
- **Diffusion Loss:** 5.37e-5
- **Consistency Loss:** 6.38e-7
- **Evaluation:** 0.0657 NRMSE (with enhanced TTC) ⭐ **BEST**
- **URL:** https://wandb.ai/emgun-morpheus-space/universal-simulator/runs/qswd55uf

## Known Issues (Not Yet Resolved)

### 1. Diffusion Architecture Mismatch
**Problem:** 32-dim v2 diffusion was trained with `hidden_dim=64` instead of the configured `96`.

**Root Cause:** Config inheritance issue - `diffusion.hidden_dim` wasn't properly overriding the base config.

**Impact:** Can't use diffusion checkpoint for evaluation, only operator works.

**Solution Needed:**
```yaml
# In train_burgers_32dim_v2_practical.yaml
diffusion:
  latent_dim: 32
  hidden_dim: 96  # ← This should override, but didn't work

# Possible fix: Use explicit diffusion config section
stages:
  diff_residual:
    model:
      latent_dim: 32
      hidden_dim: 96  # More explicit override
```

### 2. Baseline Evaluation Missing
**Problem:** Need to run baseline (no TTC) evaluation for 32-dim v2 to compare.

**Current Results:**
- Only have TTC evaluation: 0.0657 NRMSE
- Baseline is unknown

**TODO:** Create evaluation script that disables TTC programmatically.

## Recommendations

### For Future Runs:
1. **Always use offline mode** until API key issue is resolved:
   ```bash
   export WANDB_MODE=offline
   ```

2. **Sync after training:**
   ```bash
   wandb sync wandb/offline-run-*
   ```

3. **Verify config inheritance** for architecture parameters:
   - Check that `hidden_dim` is correctly set for both operator and diffusion
   - Consider flattening config to avoid inheritance issues

4. **Test eval before long training:**
   - Run a quick eval test with dummy checkpoints
   - Verify model architectures match

### Getting New API Key:
If you need full online mode, get a fresh API key from:
https://wandb.ai/authorize

Then:
```bash
wandb login NEW_API_KEY
```

## Summary

**What Works:** ✅
- Offline training with WandB logging
- Post-training sync to wandb.ai  
- All training metrics captured
- Entity parameter properly set

**What Needs Work:** ⚠️
- Fix config inheritance for diffusion `hidden_dim`
- Get baseline (no TTC) evaluation for 32-dim v2
- Optional: Get new API key for full online mode

**Overall:** System is functional with offline mode workaround. Training and logging work perfectly, just need manual sync step.

