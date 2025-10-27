# Oscillation Fix Configurations

This document describes the new configs created to address the operator loss oscillation issue documented in `reports/OSCILLATION_ANALYSIS.md`.

## Problem Summary

The 16-dim UPT configs exhibited pathological training behavior:
- **50% oscillation rate** (12 increases, 12 decreases over 24 transitions)
- **Bimodal loss distribution** (2.96σ separation between regimes)
- **Never converged** (best loss 1.536, target <0.001)
- **Large jumps** (up to 0.76 magnitude)

**Root causes identified:**
1. **Information bottleneck** (512 dims insufficient for task)
2. **Encoder drift** (no inverse losses)
3. **Insufficient depth** (3 layers vs 8-12 recommended)
4. **Learning rate slightly high**

## Solution Configs

### 1. Full Fix: `train_burgers_upt_64dim.yaml` (Recommended)

**Implements all 4 fixes from analysis (Scenario C):**

| Parameter | Old Value | New Value | Rationale |
|-----------|-----------|-----------|-----------|
| `latent.dim` | 16 | 64 | 4x capacity |
| `latent.tokens` | 32 | 64 | 2x capacity |
| **Total capacity** | **512 dims** | **4096 dims** | **8x increase** |
| `operator.pdet.depths` | [1,1,1] | [3,3,2] | 3→8 layers |
| `operator.pdet.hidden_dim` | 96 | 192 | Scale proportionally |
| `stages.operator.optimizer.lr` | 1e-3 | 5e-4 | More conservative |
| `training.lambda_inv_enc` | 0.5 | 0.5 | ✅ Already set |
| `training.lambda_inv_dec` | 0.5 | 0.5 | ✅ Already set |

**Expected results:**
- ✅ Operator loss: **<0.001** (1600x improvement)
- ✅ **No oscillation** (smooth monotonic decrease)
- ✅ **Optimal performance**
- ⏱️ Training time: ~35-40 min (vs 25 min for 16-dim)

**When to use:**
- Production runs requiring best accuracy
- When training time is not critical
- Validating the full oscillation hypothesis

### 2. Capacity-Only Fix: `train_burgers_upt_64dim_shallow.yaml` (Comparison)

**Implements capacity fix alone (Scenario B):**

| Parameter | Old Value | New Value | Rationale |
|-----------|-----------|-----------|-----------|
| `latent.dim` | 16 | 64 | 4x capacity |
| `latent.tokens` | 32 | 64 | 2x capacity |
| `operator.pdet.depths` | [1,1,1] | [1,1,1] | **Keep shallow** |
| `stages.operator.optimizer.lr` | 1e-3 | 5e-4 | More conservative |

**Expected results:**
- ✅ Operator loss: **<0.01** (160x improvement)
- ✅ **No oscillation** (smooth convergence)
- ⚠️ Not optimal (10x worse than 8-layer version)
- ⏱️ Training time: ~25-30 min (faster than 8-layer)

**When to use:**
- Quick validation of capacity hypothesis
- Faster experimentation
- Scientific comparison to isolate depth effect

## Usage

### Quick Start

```bash
# Recommended: Full fix (8 layers, 64-dim)
python scripts/vast_launch.py launch \
  --config configs/train_burgers_upt_64dim.yaml \
  --auto-shutdown

# Comparison: Shallow fix (3 layers, 64-dim)
python scripts/vast_launch.py launch \
  --config configs/train_burgers_upt_64dim_shallow.yaml \
  --auto-shutdown
```

### Local Testing

```bash
# Validate config
python scripts/validate_config.py configs/train_burgers_upt_64dim.yaml

# Dry run (1 epoch, operator only)
python scripts/train.py \
  --config configs/train_burgers_upt_64dim.yaml \
  --stage operator \
  --override stages.operator.epochs=1
```

### Monitoring

Watch for these key indicators during training:

**Healthy training (expected):**
```
Epoch 0:  2.0
Epoch 5:  0.5   ✅ Exponential decay
Epoch 10: 0.1   ✅ Smooth descent
Epoch 15: 0.01  ✅ Converging
Epoch 20: 0.002 ✅ Stable
Epoch 25: 0.001 ✅ Target reached
```

**Unhealthy training (if problem persists):**
```
Epoch 0:  2.3
Epoch 6:  2.2   ⚠️ Slow progress
Epoch 12: 1.9
Epoch 18: 2.1   ❌ JUMP (oscillating)
Epoch 24: 1.8
```

**WandB charts to monitor:**
- `operator/loss_total` - Should decrease monotonically
- `operator/loss_inv_enc` - Encoder stability
- `operator/loss_inv_dec` - Decoder stability
- Standard deviation of loss over 10 epochs - Should be <1% of mean

## Verification Plan

To validate the oscillation analysis:

1. **Run baseline** (already done):
   ```bash
   # 16-dim, no inverse losses, 3 layers
   # Result: Loss ~1.5, 50% oscillation ❌
   configs/train_burgers_upt_nocache.yaml
   ```

2. **Run shallow 64-dim** (new):
   ```bash
   # 64-dim, inverse losses, 3 layers
   # Expected: Loss ~0.01, no oscillation ✅
   configs/train_burgers_upt_64dim_shallow.yaml
   ```

3. **Run deep 64-dim** (new):
   ```bash
   # 64-dim, inverse losses, 8 layers
   # Expected: Loss <0.001, no oscillation ✅
   configs/train_burgers_upt_64dim.yaml
   ```

4. **Compare results**:
   ```bash
   python scripts/compare_runs.py \
     <baseline_run_id> \
     <shallow_64dim_run_id> \
     <deep_64dim_run_id>
   ```

**Success criteria:**
- ✅ Shallow 64-dim: 10-100x better than baseline
- ✅ Deep 64-dim: 1.5-2x better than shallow
- ✅ Both show smooth convergence (no oscillation)
- ✅ Loss curves match analysis predictions

If all criteria met → **Oscillation diagnosis validated!**

## Cost & Resource Estimates

### GPU Memory

| Config | Latent Size | Model Size | GPU Memory | Max Batch Size (A100 40GB) |
|--------|-------------|------------|------------|---------------------------|
| 16-dim (baseline) | 512 dims | ~12M params | ~4 GB | 16 |
| 64-dim shallow | 4096 dims | ~48M params | ~8 GB | 8 |
| 64-dim deep (8 layers) | 4096 dims | ~75M params | ~12 GB | 8 |

### Training Time & Cost (VastAI, A100 @ $1.89/hr)

| Config | Epochs | Time | Cost | Expected Loss |
|--------|--------|------|------|---------------|
| 16-dim baseline | 25 | ~25 min | $0.79 | 1.536 ❌ |
| 64-dim shallow | 25 | ~30 min | $0.95 | <0.01 ✅ |
| 64-dim deep (8 layers) | 25 | ~40 min | $1.26 | <0.001 ✅ |

**Recommendation:** Use 64-dim deep for production. The extra $0.47 buys 1600x better accuracy.

### Disk Space

| Config | Cache | Compile | Total | Notes |
|--------|-------|---------|-------|-------|
| nocache | 0 GB | 0 GB | ~20 GB | Slow, safe for small disks |
| optimized | ~500 GB | ~200 GB | ~800 GB | Fast, needs large disk |

Both new configs use `compile: false` to avoid disk issues.

## Troubleshooting

### If oscillation persists:

1. **Check dimensions match:**
   ```bash
   python scripts/validate_config.py configs/train_burgers_upt_64dim.yaml
   ```

2. **Verify inverse losses enabled:**
   ```yaml
   training:
     use_inverse_losses: true  # Must be true
     lambda_inv_enc: 0.5       # Must be >0
     lambda_inv_dec: 0.5       # Must be >0
   ```

3. **Check depth correctly set:**
   ```yaml
   operator:
     pdet:
       depths: [3, 3, 2]  # Should sum to 8
   ```

4. **Inspect WandB logs:**
   - Look for `loss_inv_enc` and `loss_inv_dec` metrics
   - If missing → inverse losses not working
   - If present but high → encoder/decoder mismatch

### If OOM (out of memory):

```yaml
training:
  batch_size: 8 → 4     # Reduce batch size
  accum_steps: 6 → 12   # Increase accumulation
```

Effective batch remains: 4 × 12 = 48 ✅

### If training too slow:

Try shallow version first:
```bash
python scripts/vast_launch.py launch \
  --config configs/train_burgers_upt_64dim_shallow.yaml \
  --auto-shutdown
```

## References

- **Primary analysis:** `reports/OSCILLATION_ANALYSIS.md`
- **Root causes:** Information bottleneck, encoder drift, insufficient depth
- **Solution:** 64-dim + inverse losses + 8 layers + lower LR
- **Expected improvement:** 1600x better final loss, smooth convergence

## Next Steps

1. ✅ Create configs (done)
2. ⏳ Run shallow 64-dim validation
3. ⏳ Run deep 64-dim validation
4. ⏳ Compare to baseline
5. ⏳ Update leaderboard if successful
6. ⏳ Promote to production config

---

*Created: 2025-10-26*
*Based on: OSCILLATION_ANALYSIS.md*
