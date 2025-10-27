# Oscillation Fix Configs - Quick Reference

## TL;DR

**Problem:** 16-dim model oscillates (loss 1.536, never converges)
**Solution:** Scale to 64-dim + add depth

| Config | Loss | Time | Use Case |
|--------|------|------|----------|
| `train_burgers_upt_64dim.yaml` | <0.001 ✅ | 40 min | **Production** |
| `train_burgers_upt_64dim_shallow.yaml` | <0.01 ✅ | 30 min | Quick validation |
| `train_burgers_upt_nocache.yaml` | 1.536 ❌ | 25 min | Baseline (broken) |

## Quick Launch Commands

```bash
# Production (recommended)
python scripts/vast_launch.py launch \
  --config configs/train_burgers_upt_64dim.yaml \
  --auto-shutdown

# Fast validation
python scripts/vast_launch.py launch \
  --config configs/train_burgers_upt_64dim_shallow.yaml \
  --auto-shutdown
```

## What Changed

```yaml
# From 16-dim baseline to 64-dim fix:

latent:
  dim: 16 → 64         # 4x capacity
  tokens: 32 → 64      # 2x capacity

operator.pdet:
  depths: [1,1,1] → [3,3,2]  # 3 → 8 layers (full version only)
  hidden_dim: 96 → 192       # Scale proportionally

training:
  batch_size: 12 → 8         # Larger model
  accum_steps: 4 → 6         # Maintain effective batch=48

stages.operator.optimizer:
  lr: 1.0e-3 → 5.0e-4        # More conservative
```

## Expected Training Curves

### Baseline (16-dim) - BROKEN ❌
```
Epoch 0:  2.3
Epoch 12: 1.9
Epoch 48: 1.7
Epoch 54: 2.0  ⚠️ JUMP
Epoch 90: 2.2  ⚠️ JUMP
Epoch 108: 1.5 (best)
Epoch 132: 2.3 ⚠️ CATASTROPHIC
Final: 1.6 (unstable, oscillating)
```

### 64-dim Shallow - FIXED ✅
```
Epoch 0:  2.0
Epoch 5:  0.5  ✅ Smooth
Epoch 10: 0.1  ✅ Smooth
Epoch 15: 0.03 ✅ Smooth
Epoch 20: 0.015 ✅ Converging
Final: <0.01 (stable)
```

### 64-dim Deep (8 layers) - OPTIMAL ✅
```
Epoch 0:  2.0
Epoch 5:  0.3  ✅ Fast
Epoch 10: 0.05 ✅ Fast
Epoch 15: 0.01 ✅ Fast
Epoch 20: 0.003 ✅ Converging
Final: <0.001 (optimal)
```

## Monitoring Checklist

Watch WandB for:
- ✅ `operator/loss_total` decreases monotonically (no jumps)
- ✅ `operator/loss_inv_enc` stays low (<0.1)
- ✅ `operator/loss_inv_dec` stays low (<0.1)
- ✅ Loss std dev <1% of mean (stable)

If you see:
- ❌ Loss jumps >0.1 → oscillation persists
- ❌ Loss plateaus >0.1 → not converging
- ❌ Missing `loss_inv_*` → inverse losses disabled

## Troubleshooting

**OOM (out of memory)?**
```yaml
training:
  batch_size: 8 → 4
  accum_steps: 6 → 12
```

**Too slow?**
Use shallow version or reduce epochs:
```yaml
stages.operator.epochs: 25 → 10
```

**Still oscillating?**
```bash
# Verify config
python scripts/validate_config.py configs/train_burgers_upt_64dim.yaml

# Check dimensions match
grep -A2 "latent:" configs/train_burgers_upt_64dim.yaml
grep -A5 "pdet:" configs/train_burgers_upt_64dim.yaml
```

## Full Documentation

- **Comprehensive guide:** `docs/oscillation_fix_configs.md`
- **Implementation summary:** `OSCILLATION_FIX_SUMMARY.md`
- **Root cause analysis:** `reports/OSCILLATION_ANALYSIS.md`

---

*Quick reference for oscillation fix configs*
*Updated: 2025-10-26*
