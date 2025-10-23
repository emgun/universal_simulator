# Burgers-Golden Evaluation Variance Analysis

**Date:** 2025-10-23
**Analyzed Runs:** Last 4 finished burgers-golden runs
**Project:** emgun-morpheus-space/universal-simulator

## Executive Summary

The burgers-golden configuration shows **significant and problematic variance** in evaluation results across runs with the same configuration. Baseline NRMSE varies by 50% and conservation gap varies by 1540% (15x), making results unreliable and non-reproducible.

## Key Findings

### 1. Evaluation Metric Variance (CRITICAL)

| Metric | Min | Max | Mean | Std | CV |
|--------|-----|-----|------|-----|-----|
| **Baseline NRMSE** | 0.0831 | 0.1247 | 0.0994 | 0.0195 | **19.7%** |
| **TTC NRMSE** | 0.0831 | 0.1247 | 0.0994 | 0.0195 | **19.7%** |
| **Conservation Gap** | 0.87 | 13.43 | 6.40 | 6.41 | **100.2%** |

**Issue:** The expected performance from CLAUDE.md is NRMSE ~0.078, but actual results range from 0.083 to 0.125, with most runs performing worse than documented.

### 2. Training Loss Variance

| Stage | Mean | Std | CV |
|-------|------|-----|-----|
| **Operator** | 0.001040 | 0.000039 | **3.8%** ✅ |
| **Diffusion** | 0.004583 | 0.001779 | **38.8%** ⚠️ |
| **Consistency** | 0.000002 | 0.000000 | **<1%** ✅ |

**Finding:** Operator and consistency stages train reliably, but **diffusion has 122% variance** (loss ranges from 0.0029 to 0.0065).

### 3. Counter-Intuitive Correlations

| Correlation | Value | Interpretation |
|-------------|-------|----------------|
| Diffusion loss ↔ Baseline NRMSE | **-0.394** | Lower diffusion loss = HIGHER eval error! |
| Diffusion loss ↔ Conservation gap | **-0.590** | Lower diffusion loss = WORSE physics! |
| Operator loss ↔ Baseline NRMSE | -0.003 | Operator performance doesn't predict eval |

**Critical Insight:** The diffusion stage exhibits **inverse correlation** with evaluation performance. When diffusion achieves lower training loss, the model performs WORSE on evaluation. This strongly suggests **overfitting** in the diffusion stage.

### 4. Test-Time Conditioning (TTC) is Ineffective

| Run | Baseline NRMSE | TTC NRMSE | Improvement |
|-----|----------------|-----------|-------------|
| train-20251023_012202 | 0.1247 | 0.1247 | **-0.003%** ❌ |
| train-20251023_000127 | 0.0831 | 0.0831 | **-0.011%** ❌ |
| train-20251022_221628 | 0.1047 | 0.1047 | **-0.010%** ❌ |
| 19pkjths | 0.0849 | 0.0849 | **-0.016%** ❌ |

**Finding:** TTC provides NO improvement and actually degrades performance slightly. The expected 25x improvement from CLAUDE.md is not being realized.

### 5. Missing Configuration Logging

Recent runs (train-20251023_*) show **empty config dictionaries** in WandB summary, though training history is captured. This makes it impossible to verify:
- Random seed (if any)
- Data splits
- Hyperparameters
- Architecture settings

## Root Causes

### Primary: No Random Seed Control ⚠️

None of the analyzed runs log a random seed in their config. This means:
- **PyTorch initialization** is random each run
- **Data shuffling** differs between runs
- **TTC sampling** varies randomly
- **Diffusion noise** schedules may differ

**Impact:** Training and evaluation are fundamentally non-reproducible.

### Secondary: Diffusion Overfitting 🔴

The strong inverse correlation (-0.59) between diffusion loss and conservation gap suggests:
1. Diffusion model learns spurious patterns that hurt physics conservation
2. Training loss doesn't measure what matters for generalization
3. Early stopping may be needed for diffusion stage

### Tertiary: TTC Misconfiguration 🔴

TTC shows zero improvement, which indicates:
- Analytical rewards (conservation, BC violation) may not be implemented correctly
- Beam search may not be exploring diverse enough candidates
- Decoder may have dimensional mismatch issues (though config claims consistency)

### Quaternary: Config Logging Broken 🟡

Recent WandB runs don't capture config in summary, making debugging impossible.

## Detailed Run Breakdown

### Run: train-20251023_012202 (WORST)
- **Baseline NRMSE:** 0.1247 (59% worse than expected)
- **Conservation Gap:** 13.43 (worst physics)
- **Diffusion Loss:** 0.0029 (BEST training, worst eval! 🚩)
- **Operator Loss:** 0.0011

### Run: train-20251023_000127 (BEST)
- **Baseline NRMSE:** 0.0831 (6% worse than expected, but closest)
- **Conservation Gap:** 0.87 (best physics)
- **Diffusion Loss:** 0.0044 (moderate)
- **Operator Loss:** 0.0011

### Run: train-20251022_221628 (MEDIUM)
- **Baseline NRMSE:** 0.1047 (34% worse than expected)
- **Conservation Gap:** 4.90 (medium)
- **Diffusion Loss:** 0.0065 (WORST training, medium eval)
- **Operator Loss:** 0.0010

## Recommendations

### Immediate Actions (P0)

1. **Add Deterministic Training**
   ```yaml
   # In configs/train_burgers_golden.yaml
   seed: 42  # Or any fixed seed
   training:
     deterministic: true
     benchmark: false  # Disable cudnn benchmarking for reproducibility
   ```

2. **Fix Config Logging**
   - Investigate why recent runs don't log config to WandB summary
   - Likely issue in `src/ups/utils/wandb_context.py`
   - Ensure `wandb.config.update()` is called after initialization

3. **Disable or Debug Diffusion Stage**
   - Current diffusion hurts performance when it trains "well"
   - Options:
     - Skip diffusion entirely (set epochs to 0)
     - Add stronger regularization (increase weight_decay from 0.015 to 0.05)
     - Implement early stopping based on validation conservation gap
     - Reduce learning rate from 5e-5 to 1e-5

4. **Fix or Disable TTC**
   - TTC provides zero benefit currently
   - Options:
     - Verify analytical reward implementation in `src/ups/eval/reward_models.py`
     - Check decoder dimensional consistency
     - Increase noise variance in TTC sampling
     - Or simply disable TTC for now (`ttc.enabled: false`)

### Short-Term (P1)

5. **Implement Physics-Aware Early Stopping**
   - Stop diffusion training based on validation conservation gap, not training loss
   - Add validation callback that checks physics diagnostics

6. **Reduce Diffusion Epochs**
   - Currently 8 epochs - try reducing to 3-5
   - The inverse correlation suggests less diffusion training may help

7. **Add Validation Set Tracking**
   - Log val/operator/loss, val/diffusion/loss during training
   - Use validation loss for early stopping instead of training loss

### Medium-Term (P2)

8. **Hyperparameter Sweep for Diffusion**
   - Systematically vary: learning rate, weight_decay, epochs
   - Optimize for eval conservation gap, not training loss

9. **Investigate Conservation Gap Calculation**
   - Verify implementation in `src/ups/eval/physics_checks.py`
   - Ensure it's computed consistently across runs
   - Check if certain test examples dominate the metric

10. **Document Actual Performance**
    - Update CLAUDE.md with realistic expectations
    - Current docs claim NRMSE ~0.078, actual is 0.083-0.125

## Success Criteria for Next Runs

A successful fix should achieve:
- **Reproducibility:** CV < 5% for baseline NRMSE across 3 runs with same seed
- **Performance:** Baseline NRMSE < 0.085 consistently
- **Physics:** Conservation gap < 2.0 consistently
- **TTC Benefit:** TTC improvement > 5% (or disable if not achievable)

## References

- **Config:** `configs/train_burgers_golden.yaml`
- **Best Run (19pkjths):** https://wandb.ai/emgun-morpheus-space/universal-simulator/runs/19pkjths
- **Worst Run:** https://wandb.ai/emgun-morpheus-space/universal-simulator/runs/train-20251023_012202
- **Analysis Scripts:** `scripts/analyze_run.py`, `scripts/compare_runs.py`
