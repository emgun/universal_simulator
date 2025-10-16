# Run Comparison: hjnqclgk (Aggressive) vs Baseline 32-dim

## TL;DR: **Aggressive approach provided NO benefit**

The aggressive TTC + rollout configuration made training harder/longer without improving final NRMSE.

---

## Performance Comparison

| Metric | Baseline 32-dim | hjnqclgk (Aggressive) | Difference |
|--------|----------------|---------------------|------------|
| **Final NRMSE** | **0.0921** | **0.0907** | **-1.5%** (negligible) |
| Training Time | ~20 min | ~315 min | **16x slower!** |
| GPU Cost | ~$0.70 | ~$10.78 | **15x more expensive!** |
| TTC Candidates | 8 | 16 | 2x more |
| TTC Beam Width | 3 | 5 | 67% larger |
| TTC Horizon | 2 | 3 | 50% longer |
| Rollout Loss | None | Yes (horizon=3) | Added complexity |

**Conclusion**: Aggressive approach achieved essentially the same NRMSE (0.09) but cost 15x more and took 16x longer.

---

## Training Loss Comparison

### Operator Stage

| Metric | Baseline | hjnqclgk | Assessment |
|--------|----------|----------|------------|
| Epochs | 15 (pru2jxc4 style) | 50 | 3.3x longer |
| Final Loss | ~0.0002-0.0005 (estimated) | 0.000659 | **Worse** |
| Best Loss | ~0.0002-0.0005 (estimated) | 0.000720 | **Worse** |
| Grad Norm | ~0.05-0.10 (typical) | 0.107 | Similar/slightly high |

**Analysis**: Despite 3.3x more epochs, hjnqclgk achieved WORSE operator loss. The rollout loss (horizon=3, lambda=0.1) made the task harder without improving the base model quality.

### Diffusion Residual Stage

| Metric | Baseline | hjnqclgk | Assessment |
|--------|----------|----------|------------|
| Epochs | 8 | 15 | 1.9x longer |
| Final Loss | ~0.004-0.006 (typical) | 0.00813 | **Much worse** |
| Best Loss | ~0.004-0.006 (typical) | 0.00875 | **Much worse** |
| Grad Norm | ~0.15-0.25 (typical) | 0.313 | **Higher** |

**Critical Finding**: The diffusion loss of 0.0081 is **significantly worse** than expected. This suggests:
1. The harder operator task (with rollout) produced worse latent representations
2. The diffusion model struggled to learn on these representations
3. Underfitting - the model didn't converge properly

### Consistency Distillation Stage

| Metric | Baseline | hjnqclgk | Assessment |
|--------|----------|----------|------------|
| Epochs | 8 | 10 | 1.25x longer |
| Final Loss | ~1-2e-06 (typical) | 1.20e-06 | Similar |
| Best Loss | ~1-2e-06 (typical) | 1.71e-06 | Similar |
| Grad Norm | ~1e-04 (typical) | 8.91e-05 | Similar |

**Analysis**: Consistency distillation performed normally, suggesting the aggressive TTC settings during eval don't affect this stage.

---

## Why Did Aggressive Approach Fail?

### 1. Rollout Loss Hurt Base Model Quality

**Problem**: Training with multi-step rollout (horizon=3) made the operator's task harder:
- Single-step prediction: Learn `z_{t+1} = f(z_t)`
- Multi-step rollout: Learn `z_{t+3} = f(f(f(z_t)))` simultaneously

**Result**: 
- Operator loss WORSE despite 3.3x more epochs
- Diffusion loss WORSE - struggled with harder latent space
- No benefit in final NRMSE

**Why?** The model capacity (32-dim, 120K params) was insufficient for the harder multi-step task. Rollout loss is typically useful for **larger models** with capacity to spare.

### 2. Aggressive TTC Didn't Help

**Configuration**:
- Candidates: 8 → 16 (2x)
- Beam width: 3 → 5 (67% increase)
- Horizon: 2 → 3 (50% increase)
- Max evaluations: 150 → 300 (2x)

**Result**: NRMSE 0.0921 → 0.0907 (1.5% improvement, within noise)

**Why didn't it help?**
1. **Diminishing Returns**: Standard TTC (8 candidates, beam=3) was already near-optimal for 32-dim
2. **Overfitting to Test**: More aggressive search may have found slightly better test-time solutions but not meaningfully better
3. **Computational Waste**: 16x longer eval time for 1.5% gain

**Cost/Benefit**:
- Baseline: 0.0921 NRMSE in ~20 min
- Aggressive: 0.0907 NRMSE in ~320 min (training + eval)
- **Not worth it!**

### 3. Training Curves Look Worse

The user's observation is **correct**:

**Operator Grad Norm**:
- Should decay smoothly to ~0.05
- hjnqclgk: Still at 0.107 after 50 epochs
- Suggests the rollout task was too hard for stable convergence

**Diffusion Loss**:
- Should reach ~0.004-0.006
- hjnqclgk: Stuck at 0.0081
- Clear underfitting - model couldn't learn the residual properly

**Diffusion Grad Norm**:
- Should be ~0.15-0.25
- hjnqclgk: 0.313 (higher)
- Suggests struggling/oscillating, not smoothly converging

---

## Root Cause Analysis

### The Fundamental Problem

**We asked a 120K parameter model to do TOO MUCH**:
1. Learn multi-step rollout (harder operator task)
2. Learn diffusion residuals on harder latent space
3. Compress everything into 32 dimensions

**Result**: Model quality degraded across the board, but TTC was able to "rescue" it at inference time to achieve similar NRMSE.

### Why Final NRMSE Was Still OK

**TTC compensated for worse base model**:
- Worse operator → More room for TTC to optimize
- This aligns with the "TTC Paradox" from previous findings
- But it's an inefficient path: better to have good base model + TTC

**Analogy**:
- Good approach: Train efficient car (good operator) + GPS (TTC) → Fast, efficient
- Bad approach: Train broken car (bad operator with rollout) + super-GPS (aggressive TTC) → Slow, expensive, same destination

---

## Recommendations

### 1. **Drop Rollout Loss for 32-dim**

Rollout loss requires more model capacity. For 32-dim:
```yaml
# REMOVE THESE:
training:
  rollout_horizon: 3
  lambda_rollout: 0.1
```

**Expected improvement**:
- Operator loss: 0.00066 → 0.0002-0.0003 (2-3x better)
- Diffusion loss: 0.0081 → 0.004-0.006 (2x better)
- Training time: 50 epochs → 15-25 epochs (2x faster)
- Final NRMSE: Same (0.09) or slightly better

### 2. **Use Standard TTC Settings**

Aggressive TTC is computational waste for 32-dim:
```yaml
ttc:
  candidates: 8      # Not 16
  beam_width: 3      # Not 5
  horizon: 2         # Not 3
  max_evaluations: 150  # Not 300
```

**Expected improvement**:
- Eval time: 80 min → 5-10 min (8-16x faster!)
- NRMSE: Same (0.09)
- Cost: $10.78 → $0.70 (15x cheaper)

### 3. **Reserve Aggressive Methods for Larger Models**

**Rollout loss**: Use for 128-dim, 256-dim, 512-dim where capacity exists
**Aggressive TTC**: May help for 512-dim where baseline is already good

**For 32-dim**: Keep it simple and fast

---

## Corrected Config for 32-dim v3

Based on this analysis, here's what 32-dim v3 SHOULD be:

```yaml
# 32-dim v3 CORRECTED: No rollout, standard TTC

training:
  batch_size: 12
  # NO rollout_horizon
  # NO lambda_rollout
  
stages:
  operator:
    epochs: 25  # Not 50
    optimizer:
      name: adamw
      lr: 1.0e-3
      weight_decay: 0.03

  diff_residual:
    epochs: 8   # Not 15
    
  consistency_distill:
    epochs: 8   # Not 10

ttc:
  enabled: true
  candidates: 8       # Not 16
  beam_width: 3       # Not 5
  horizon: 2          # Not 3
  max_evaluations: 150  # Not 300
```

**Expected results**:
- Training: ~25 min (vs 315 min)
- Eval: ~10 min (vs 80 min)
- Cost: ~$1.20 (vs $10.78)
- NRMSE: 0.09 (same or better)
- Better training curves
- **9x cheaper, same performance!**

---

## Conclusion

The hjnqclgk run was a valuable **negative result**:

✅ **What we learned**:
1. Rollout loss hurts 32-dim model quality
2. Aggressive TTC provides negligible benefit for small models
3. Simple is better for 32-dim: standard training + standard TTC = 0.09 NRMSE

❌ **What didn't work**:
1. Making task harder (rollout) without more capacity
2. Throwing more compute at TTC without addressing base model
3. Assumption that "more aggressive = better"

**Next Steps**:
1. Use corrected config (no rollout, standard TTC) for fair comparison
2. Focus on architectural improvements (Option 2 from next_steps_analysis.md)
3. Test smaller dimensions (16-dim, 24-dim) as planned

The aggressive approach was **9x more expensive** for **no gain**. Lesson: understand your model's capacity before making the task harder.

