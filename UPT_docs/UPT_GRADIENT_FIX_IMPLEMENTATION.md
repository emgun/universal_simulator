# UPT Gradient Explosion Fix - Implementation Summary

**Date**: 2025-01-27
**Status**: Diagnostic test running on VastAI instance 27336827
**Branch**: feature--UPT

---

## Problem Summary

Previous UPT training run (train-20251027_060629) showed:
- **Gradient explosion**: 61.6 billion max gradient norm
- **Poor convergence**: Operator loss only reduced 59.8% (vs 100% for golden baseline)
- **Accuracy degradation**: 27.4% worse NRMSE than baseline
- **BUT**: 70.5% better conservation gap - proving the physics approach works!

**Root cause**: Inverse loss weights (0.5) were too large and poorly balanced with forward losses.

---

## Fixes Implemented

### 1. Aggressive Gradient Clipping

**File**: `configs/train_burgers_upt_fixed.yaml`, `configs/test_upt_fixed_diagnostic.yaml`

```yaml
training:
  grad_clip: 0.3  # Much stronger (was 1.0)
  grad_clip_per_param: true  # Clip each parameter group separately
```

### 2. Much Smaller Inverse Loss Weights (100x reduction!)

**File**: `configs/train_burgers_upt_fixed.yaml`, `configs/test_upt_fixed_diagnostic.yaml`

```yaml
training:
  lambda_inv_enc: 0.001  # Was 0.5 → caused gradient explosion
  lambda_inv_dec: 0.001  # Start conservative
```

### 3. Curriculum Learning

**Implementation**: Added to `src/ups/training/losses.py`

#### New Function: `compute_inverse_loss_curriculum_weight()`

```python
def compute_inverse_loss_curriculum_weight(
    epoch: int,
    base_weight: float,
    warmup_epochs: int = 15,
    max_weight: float = 0.05,
) -> float:
    """Compute curriculum-scheduled weight for inverse losses.

    Implements gradual ramp-up to prevent gradient explosion:
    - Epochs 0-warmup_epochs: weight = 0 (pure forward training)
    - Epochs warmup_epochs to warmup_epochs*2: linear ramp from 0 to base_weight
    - Epochs > warmup_epochs*2: weight = min(base_weight, max_weight)
    """
    if epoch < warmup_epochs:
        # Phase 1: Pure forward training
        return 0.0
    elif epoch < warmup_epochs * 2:
        # Phase 2: Linear ramp-up
        progress = (epoch - warmup_epochs) / warmup_epochs
        return min(base_weight * progress, max_weight)
    else:
        # Phase 3: Full weight (but capped at max_weight)
        return min(base_weight, max_weight)
```

#### Modified Function: `compute_operator_loss_bundle()`

Added `current_epoch` parameter and curriculum weight application:

```python
def compute_operator_loss_bundle(
    *,
    # ... existing parameters ...
    current_epoch: Optional[int] = None,  # NEW
) -> LossBundle:
    """Compute full loss bundle with curriculum learning support."""

    # Apply curriculum learning to inverse loss weights if epoch is provided
    lambda_inv_enc = weights.get("lambda_inv_enc", 0.0)
    lambda_inv_dec = weights.get("lambda_inv_dec", 0.0)

    if current_epoch is not None:
        warmup_epochs = weights.get("inverse_loss_warmup_epochs", 15)
        max_weight = weights.get("inverse_loss_max_weight", 0.05)

        lambda_inv_enc = compute_inverse_loss_curriculum_weight(
            current_epoch, lambda_inv_enc, warmup_epochs, max_weight
        )
        lambda_inv_dec = compute_inverse_loss_curriculum_weight(
            current_epoch, lambda_inv_dec, warmup_epochs, max_weight
        )

    # Rest of function uses curriculum-adjusted weights...
```

**Config parameters**:
```yaml
training:
  inverse_loss_warmup_epochs: 15  # Pure forward for epochs 0-15
  inverse_loss_max_weight: 0.05   # Never exceed 5% of total loss
```

### 4. Extended Training Duration

**File**: `configs/train_burgers_upt_fixed.yaml`

```yaml
stages:
  operator:
    epochs: 35  # Increased from 25 to allow recovery after warmup
```

---

## Files Created/Modified

### New Configs

1. **`configs/train_burgers_upt_fixed.yaml`**
   - Full 35-epoch production config
   - All gradient fixes applied
   - Enabled: compile, AMP, TTC, evaluation
   - Expected runtime: ~35-40 min on A100
   - WandB run name: `burgers-upt-fixed`

2. **`configs/test_upt_fixed_diagnostic.yaml`**
   - Quick 5-epoch diagnostic
   - Disabled: compile (faster startup), TTC, evaluation
   - Expected runtime: ~5-7 min on A100
   - WandB run name: `burgers-upt-diagnostic`

### Modified Code

1. **`src/ups/training/losses.py`**
   - Added `compute_inverse_loss_curriculum_weight()` function
   - Modified `compute_operator_loss_bundle()` to support curriculum learning
   - Fixed typo: `none` → `None` in line 220

---

## Success Criteria (Diagnostic Run)

**Monitoring**: VastAI instance 27336827

### Critical Metrics

1. **Gradient Norms**: Stay < 100 throughout training (vs 61.6 billion in previous run)
2. **Loss Balance**: `L_inv / L_forward` ratio < 0.2 (inverse should not dominate)
3. **Convergence**: Operator loss decreases steadily each epoch
4. **No Instabilities**: No NaN/Inf values, no extreme spikes

### Expected Behavior by Epoch

```
Epoch 0-5 (Diagnostic):
  - L_inv_enc = 0.0 (curriculum warmup phase)
  - L_inv_dec = 0.0 (curriculum warmup phase)
  - L_forward dominates (pure forward training)
  - Operator loss: ~1.0 → ~0.1 expected

Full Run Epochs 0-35:
  - Epochs 0-15: Pure forward (L_inv = 0.0)
  - Epochs 15-30: Linear ramp-up (L_inv: 0 → 0.001)
  - Epochs 30-35: Full weight (L_inv = min(0.001, 0.05) = 0.001)
```

---

## Next Steps

### Phase 1: Diagnostic Validation (Current)

**Status**: Running on instance 27336827
**Duration**: ~5-7 minutes
**Monitor**:
- Bash process 3cf2c1: Startup logs (4 min)
- Bash process 910f44: Training progress (8 min)

**Decision point**: If gradients stay stable and loss decreases properly → proceed to Phase 2

### Phase 2: Full Training Run (If Phase 1 succeeds)

**Command**:
```bash
python scripts/vast_launch.py launch \
  --config configs/train_burgers_upt_fixed.yaml \
  --auto-shutdown
```

**Expected outcomes**:
- Gradient norms < 100 (eliminate explosion)
- Operator final loss < 0.001 (proper convergence)
- NRMSE ≤ 0.072 (match or beat golden baseline)
- Conservation gap < 1.0 (maintain UPT improvement)
- **Overall**: 5-10% improvement over golden baseline

---

## Technical Notes

### Curriculum Learning Strategy

The curriculum gradually introduces inverse losses to prevent the gradient shock that caused the previous explosion:

**Phase 1 (Epochs 0-15)**: Pure Forward Training
- Inverse weight = 0.0
- Model learns basic latent evolution without physics constraints
- Establishes stable parameter regime

**Phase 2 (Epochs 15-30)**: Gradual Introduction
- Inverse weight ramps linearly from 0 to base_weight
- Smooth transition prevents gradient spikes
- Model adapts to physics constraints incrementally

**Phase 3 (Epochs 30+)**: Full Physics-Informed Training
- Inverse weight = min(base_weight, max_weight) = min(0.001, 0.05) = 0.001
- Capped at 5% of total loss to prevent domination
- Balances accuracy with physics constraints

### Why This Should Work

1. **Historical evidence**: 70.5% conservation improvement proves physics approach works
2. **Root cause addressed**: 100x smaller weights + aggressive clipping prevents explosion
3. **Smooth introduction**: Curriculum prevents gradient shock
4. **Safety bounds**: max_weight cap prevents loss imbalance

### Comparison to Previous Run

| Parameter | Previous (Failed) | Fixed (Current) |
|-----------|------------------|-----------------|
| `lambda_inv_enc` | 0.5 | 0.001 (100x smaller) |
| `lambda_inv_dec` | 0.5 | 0.001 (100x smaller) |
| `grad_clip` | 1.0 | 0.3 (3.3x stronger) |
| `grad_clip_per_param` | false | true |
| Curriculum | None | 15 epoch warmup |
| Max gradient | 61.6 billion | Target: < 100 |
| Operator loss | 0.740 (poor) | Target: < 0.001 |

---

## Related Documents

- `reports/UPT_ANALYSIS_SUMMARY.md` - Full analysis of gradient explosion issue
- `thoughts/shared/plans/2025-01-23-upt-inverse-losses-implementation.md` - Implementation plan
- `UPT_docs/UPT_INTEGRATION_ANALYSIS.md` - Gap analysis and architecture details

---

**Last Updated**: 2025-01-27
**Next Check**: Monitor instance 27336827 logs in ~8 minutes
