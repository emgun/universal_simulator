# Overnight SOTA Sweep Status Report

**Date**: 2025-10-21
**Status**: Partial completion - 3 runs trained, eval failed due to config mismatch

---

## Summary

**Training**: ✅ 3/15 configs completed training successfully
**Evaluation**: ❌ Failed due to architecture mismatch (latent_tokens: 16 vs 32)
**Fix Applied**: Updated `small_eval_burgers.yaml` to use latent_tokens=32 (commit 2df191a)

---

## Completed Training Runs

### Round A: Optimizer Grid (3/9 completed)

| Config | WandB Run | Status | Notes |
|--------|-----------|--------|-------|
| `round_a_lr20e5_w3pct` | [703lg447](https://wandb.ai/emgun-morpheus-space/universal-simulator/runs/703lg447) | ✅ Training complete | Instance 27072218 |
| `round_a_lr20e5_w5pct` | [4dataza9](https://wandb.ai/emgun-morpheus-space/universal-simulator/runs/4dataza9) | ✅ Training complete | Instance 27072220 |
| `round_a_lr29e5_w6pct` | [39wifkv8](https://wandb.ai/emgun-morpheus-space/universal-simulator/runs/39wifkv8) | ✅ Training complete | Instance 27072227 |

**Hyperparameters**:
- LR: [2e-4, 2e-4, 3e-4]
- Warmup: [3%, 5%, 6%]
- EMA: 0.9995
- Architecture: hidden_dim=96, num_heads=6

---

## Issue: Evaluation Failure

**Error**: `RuntimeError: Checkpoint architecture mismatch detected (latent_tokens: checkpoint=32, config=16)`

**Root Cause**:
- Training configs use `latent.tokens: 32` (from `train_burgers_32dim.yaml`)
- Default `small_eval_burgers.yaml` overrode to `tokens: 16` for faster eval
- Evaluation cannot load checkpoints with mismatched architecture

**Fix Applied**:
```yaml
# configs/small_eval_burgers.yaml (line 8)
latent:
  tokens: 32  # Changed from 16 to match training
```

**Commit**: 2df191a - "Fix small_eval_burgers.yaml: use latent_tokens=32 to match training"

---

## Remaining Runs

### Still Loading (6/15)
- Instances: 27073037, 27073052, 27073057, 27073060, 27073068, 27073072
- Status: Loading for 3+ hours, likely stuck or failed
- **Recommendation**: Destroy and relaunch with fixed eval config

### Exited Early (6/15)
- Instances: 27073033, 27073040, 27073047, 27073059, 27073065, 27073071
- Likely hit same eval error and auto-shutdown
- **Recommendation**: Check WandB for partial training progress

---

## Next Steps

### Option 1: Wait for WandB Training Metrics
- Check the 3 completed runs in WandB for operator/diffusion loss curves
- If training converged well, proceed with separate evaluation

### Option 2: Run Evaluation Separately
Since checkpoints are saved in WandB artifacts, we can:
1. Download checkpoints from WandB runs
2. Run `scripts/evaluate.py` with fixed eval configs
3. Get proper test NRMSE scores

### Option 3: Relaunch Missing 12 Configs
- Fix applied (latent_tokens=32), safe to relaunch
- Destroy stuck instances
- Relaunch with updated configs

---

## Configurations Matrix

### Round A: Optimizer Grid (9 total)

| LR \\ Warmup | 3% | 5% | 6% |
|-------------|----|----|-----|
| **2e-4** | ✅ Complete | ✅ Complete | ❌ Missing |
| **3e-4** | ❌ Missing | ❌ Missing | ✅ Complete |
| **4.5e-4** | ❌ Missing | ❌ Missing | ❌ Missing |

### Round B: Capacity Scaling (3 total)
- ❌ `round_b_cap64` - Missing
- ❌ `round_b_cap96` - Missing
- ❌ `round_b_cap128` - Missing

### Round C: Hybrid Best (3 total)
- ❌ `round_c_cap128_ttc` - Missing
- ❌ `round_c_extended` - Missing
- ❌ `round_c_tokens48` - Missing

---

## Lessons Learned

1. **Architecture Consistency**: Eval configs must exactly match training architecture
2. **Config Validation**: Should validate eval configs match training before long runs
3. **VastAI Reliability**: RTX 4090 more reliable than RTX 5880Ada for launches
4. **Instance Monitoring**: Need better real-time monitoring for multi-instance sweeps

---

## Cost Summary

- **3 completed runs**: ~$1.20 (3 × $0.35/hr × 1hr)
- **12 failed/stuck instances**: ~$1.50 (wasted)
- **Total spent**: ~$2.70
- **Target budget**: $5.25 for 15 runs
- **Remaining budget**: $2.55

---

## Recommendations

**Immediate** (within 1 hour):
1. Check WandB runs for training metrics
2. Destroy all stuck/exited instances (save costs)
3. Decide: re-eval existing checkpoints OR relaunch missing configs

**Short-term** (next day):
1. Relaunch 12 missing configs with fixed eval config
2. Monitor first 2-3 runs to completion
3. Verify eval works correctly with latent_tokens=32

**Long-term**:
1. Add config validation script to check train/eval consistency
2. Create monitoring dashboard for multi-instance sweeps
3. Implement checkpoint-only training mode (skip eval, run separately)

---

**Status**: Waiting for decision on next steps
**Last Updated**: 2025-10-21 07:15 PDT
