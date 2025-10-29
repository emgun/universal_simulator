# ARM TTC Fixes Applied to Ablation Configs

**Date**: 2025-10-28
**Status**: ✅ All configs validated

---

## Summary

Applied ARM fixes to all three UPT ablation configs to enable proper Test-Time Conditioning (TTC). The fixes address the 0% TTC improvement issue by correcting physics assumptions and increasing candidate diversity.

---

## Configs Updated

1. ✅ `configs/ablation_upt_64tokens_fixed.yaml`
2. ✅ `configs/ablation_upt_128tokens_fixed.yaml`
3. ✅ `configs/ablation_upt_256tokens_fixed.yaml`

---

## Fixes Applied (All Configs)

### 1. Enable Lookahead
```yaml
horizon: 2  # Was 1 (disabled beam search)
```
**Impact**: Activates multi-step planning with beam search

### 2. Fix Physics Assumptions for Burgers PDE
```yaml
weights:
  mass: 0.0      # Was 1.0 - Burgers does NOT conserve mass
  energy: 0.0    # Was 1.0 - Burgers does NOT conserve energy
  penalty_negative: 1.0  # Was 0.5 - Only penalize unphysical negatives
```
**Impact**: Removes invalid conservation penalties for dissipative PDEs

### 3. Increase Candidate Diversity
```yaml
tau_range: [0.05, 0.95]  # Was [0.15, 0.85] - wider diffusion range
```
**Impact**: More diverse candidate generation

### 4. Enable Debug Logging
```yaml
debug: true  # New - detailed ARM output for analysis
```
**Impact**: Visibility into reward variance and candidate selection

---

## Model-Specific Noise Tuning

Noise levels adjusted based on model capacity:

| Model Size | Tokens | Latent Dim | noise_std | Rationale |
|------------|--------|------------|-----------|-----------|
| **Small** | 64 | 64 | 0.12 | Higher noise for more exploration |
| **Medium** | 128 | 128 | 0.10 | Moderate noise (baseline) |
| **Large** | 256 | 256 | 0.08 | Lower noise (more stable predictions) |

### Noise Schedules Updated

**64 tokens**:
```yaml
noise_schedule: [0.15, 0.12, 0.08]  # Was [0.08, 0.05, 0.02]
```

**128 tokens**:
```yaml
noise_schedule: [0.12, 0.10, 0.06]  # Was [0.08, 0.05, 0.02]
```

**256 tokens**:
```yaml
noise_schedule: [0.10, 0.08, 0.05]  # Was [0.05, 0.03, 0.01]
```

**Rationale**: Smaller models need more stochasticity to explore diverse solutions. Larger models have better internal representations and benefit from lower noise.

---

## TTC Parameters by Model Size

### 64 Tokens (Small Model)
```yaml
candidates: 8       # Fewer candidates (less compute)
beam_width: 3       # Narrower beam
horizon: 2          # ✅ FIXED (was 1)
steps: 3            # Moderate rollout length
max_evaluations: 150
```

### 128 Tokens (Medium Model)
```yaml
candidates: 12      # More candidates
beam_width: 4       # Wider beam
horizon: 2          # ✅ FIXED (was 1)
steps: 3            # Moderate rollout length
max_evaluations: 175
```

### 256 Tokens (Large Model)
```yaml
candidates: 16      # Most candidates (best predictions)
beam_width: 5       # Widest beam
horizon: 2          # ✅ FIXED (was 1)
steps: 5            # Longer rollout (stable)
max_evaluations: 200
residual_threshold: 0.3  # Tighter threshold (was 0.35)
```

**Rationale**: Larger models can afford more computation and benefit from tighter thresholds and longer rollouts.

---

## Expected Impact

### Before Fixes (Current Results)
- TTC improvement: ~0-2%
- Reward variance: Very low (< 1e-6)
- Candidate selection: Near random
- Lookahead: Disabled

### After Fixes (Expected)
- TTC improvement: **5-10%** (conservative estimate)
- Reward variance: Higher (> 1e-4)
- Candidate selection: Best candidate chosen
- Lookahead: Active (multi-step planning)

### Best Case (Per Paper)
- TTC improvement: **88%** (if ARM aligns with NRMSE)

---

## Validation Results

All configs passed validation with 5/5 fixes applied:

```bash
# 64 tokens
python scripts/validate_arm_config.py configs/ablation_upt_64tokens_fixed.yaml
# ✅ VALIDATION PASSED - noise_std: 0.12

# 128 tokens
python scripts/validate_arm_config.py configs/ablation_upt_128tokens_fixed.yaml
# ✅ VALIDATION PASSED - noise_std: 0.10

# 256 tokens
python scripts/validate_arm_config.py configs/ablation_upt_256tokens_fixed.yaml
# ✅ VALIDATION PASSED - noise_std: 0.08
```

---

## Testing Strategy

### Option A: Test Single Config (Quick)
```bash
# Start with medium model (128 tokens) - best balance
python scripts/vast_launch.py launch \
  --config configs/ablation_upt_128tokens_fixed.yaml \
  --auto-shutdown
```

**Time**: ~30 min
**Cost**: ~$1.50

### Option B: Test All Configs (Comprehensive)
```bash
# Run all three in parallel
python scripts/vast_launch.py launch \
  --config configs/ablation_upt_64tokens_fixed.yaml \
  --auto-shutdown &

python scripts/vast_launch.py launch \
  --config configs/ablation_upt_128tokens_fixed.yaml \
  --auto-shutdown &

python scripts/vast_launch.py launch \
  --config configs/ablation_upt_256tokens_fixed.yaml \
  --auto-shutdown &

# Wait for all to complete
wait
```

**Time**: ~60-90 min (parallel)
**Cost**: ~$4-6

---

## Key Differences from Original ARM Fix Config

### `eval_burgers_arm_fixed.yaml` (16-dim baseline)
```yaml
candidates: 16
noise_std: 0.10
steps: 1
```

### Ablation Configs (64/128/256-dim)
- **Scaled candidates**: 8/12/16 by model size
- **Scaled noise**: 0.12/0.10/0.08 (inversely with model size)
- **More steps**: 3/3/5 (longer rollouts for ablation study)

**Rationale**: Ablation configs test different architectures, so we scale TTC parameters accordingly to ensure fair comparison.

---

## Success Criteria

### Per Config Metrics

| Config | Baseline NRMSE | TTC Target | Min Improvement |
|--------|----------------|------------|-----------------|
| 64 tokens | TBD | < baseline × 0.95 | 5% |
| 128 tokens | TBD | < baseline × 0.95 | 5% |
| 256 tokens | TBD | < baseline × 0.95 | 5% |

### Debug Validation
- ✅ Reward variance > 1e-4
- ✅ Best candidate chosen > 80% of time
- ✅ Lookahead active (horizon > 1)
- ✅ Debug logs show reward components

---

## Analysis Plan

After running, check for:

1. **Reward Variance** (in step logs):
   ```python
   # From _ttc_step_logs.json
   variance = std(rewards_per_step)
   print(f"Mean variance: {variance}")  # Should be > 1e-4
   ```

2. **TTC Improvement**:
   ```python
   improvement = 100 * (baseline_nrmse - ttc_nrmse) / baseline_nrmse
   print(f"TTC improvement: {improvement}%")  # Target: > 5%
   ```

3. **Model Size Scaling**:
   ```python
   # Compare improvements across model sizes
   # Expected: Larger models benefit more from TTC
   ```

---

## Troubleshooting

### If TTC Still Shows < 2% Improvement

**Check**:
1. Step logs for reward variance
2. Conservation penalties actually disabled (mass=0, energy=0)
3. Lookahead executing (horizon=2)
4. Noise schedule applied

**Actions**:
1. Increase noise_std by 50% (e.g., 0.12 → 0.18)
2. Try different penalty_negative weights (0.5, 1.0, 2.0)
3. Consider PRM implementation (see roadmap)

---

## Next Steps

### Immediate
1. ✅ Configs updated and validated
2. ⏳ Run evaluation on VastAI
3. ⏳ Analyze results

### After Results
- **If improvement > 5%**: Document success, use these configs for production
- **If improvement < 5%**: Proceed with PRM implementation
- **If 256-token shows best TTC gain**: Consider it as new baseline

---

## Related Documents

- `ARM_CRITICAL_ISSUES_ANALYSIS.md` - Technical analysis of ARM bugs
- `ARM_FIXES_SUMMARY.md` - Quick reference for fixes
- `TTC_DURING_TRAINING_ANALYSIS.md` - Training integration analysis
- `thoughts/shared/research/2025-10-28-ttc-prm-arm-implementation-roadmap.md` - PRM plan (if needed)

---

## Changelog

### 2025-10-28 - Initial ARM Fixes
- Applied 5 critical ARM fixes to all ablation configs
- Scaled noise by model size (64: 0.12, 128: 0.10, 256: 0.08)
- Enabled debug logging for all configs
- Updated noise schedules for higher diversity
- Validated all configs successfully

---

## Configuration File Locations

```
configs/
  ablation_upt_64tokens_fixed.yaml   # Small model (64 tokens × 64 dim)
  ablation_upt_128tokens_fixed.yaml  # Medium model (128 tokens × 128 dim)
  ablation_upt_256tokens_fixed.yaml  # Large model (256 tokens × 256 dim)
  eval_burgers_arm_fixed.yaml        # Baseline (32 tokens × 16 dim)
```

---

## Summary Table

| Config | Latent Dim | Tokens | Candidates | Noise | TTC Steps | Validated |
|--------|-----------|---------|------------|-------|-----------|-----------|
| 64-token | 64 | 64 | 8 | 0.12 | 3 | ✅ |
| 128-token | 128 | 128 | 12 | 0.10 | 3 | ✅ |
| 256-token | 256 | 256 | 16 | 0.08 | 5 | ✅ |
| Baseline | 16 | 32 | 16 | 0.10 | 1 | ✅ |

**All fixes applied**: Lookahead (horizon=2), Conservation disabled (mass=0, energy=0), Wide tau range [0.05, 0.95], Debug enabled

---

**Status**: Ready for VastAI testing 🚀
