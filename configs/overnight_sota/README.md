# Overnight SOTA Sweep

**Goal**: Achieve <0.035 NRMSE (state-of-the-art) through systematic hyperparameter optimization

**Status**: ✅ All 15 runs launched successfully

**Launch Time**: 2025-10-21 03:15 UTC
**Expected Completion**: ~10:45 UTC (~7.5 hours)

---

## Strategy

Based on `docs/fast_to_sota_playbook.md` and `docs/parallel_runs_playbook.md`:

### Round A: Optimizer Grid (9 runs)
**Purpose**: Find best learning rate and warmup combination

| Config | LR | Warmup | EMA | Notes |
|--------|-----|--------|-----|-------|
| `round_a_lr20e5_w3pct` | 2e-4 | 3% | 0.9995 | Conservative |
| `round_a_lr20e5_w5pct` | 2e-4 | 5% | 0.9995 | Conservative + standard warmup |
| `round_a_lr20e5_w6pct` | 2e-4 | 6% | 0.9995 | Conservative + long warmup |
| `round_a_lr29e5_w3pct` | 3e-4 | 3% | 0.9995 | **Baseline (expected best)** |
| `round_a_lr29e5_w5pct` | 3e-4 | 5% | 0.9995 | Baseline + standard warmup |
| `round_a_lr29e5_w6pct` | 3e-4 | 6% | 0.9995 | Baseline + long warmup |
| `round_a_lr45e5_w3pct` | 4.5e-4 | 3% | 0.9995 | Aggressive |
| `round_a_lr45e5_w5pct` | 4.5e-4 | 5% | 0.9995 | Aggressive + standard warmup |
| `round_a_lr45e5_w6pct` | 4.5e-4 | 6% | 0.9995 | Aggressive + long warmup |

**Expected Winner**: `round_a_lr29e5_w5pct` or `round_a_lr45e5_w3pct`

---

### Round B: Capacity Scaling (3 runs)
**Purpose**: Find optimal model capacity vs compute trade-off

| Config | Hidden Dim | Num Heads | Group Size | Params | Expected Impact |
|--------|-----------|-----------|------------|--------|-----------------|
| `round_b_cap64` | 64 | 4 | 8 | ~750K | Faster, may underfit |
| `round_b_cap96` | 96 | 6 | 12 | ~1.5M | **Baseline** |
| `round_b_cap128` | 128 | 8 | 16 | ~2.5M | More capacity, slower |

All use LR=3e-4 (expected best from Round A)

**Expected Winner**: `round_b_cap96` or `round_b_cap128` (if worth the cost)

---

### Round C: Hybrid Best (3 runs)
**Purpose**: Combine multiple improvements for maximum performance

| Config | Key Features | Hypothesis |
|--------|--------------|------------|
| `round_c_cap128_ttc` | hidden=128, 12 TTC candidates, 200 max evals | Enhanced capacity + better uncertainty quantification |
| `round_c_extended` | 35/10/10 epochs (+40% training) | More training → better convergence |
| `round_c_tokens48` | 48 latent tokens (+50%), batch=10 | Higher resolution latents, slight memory pressure |

**Expected Winner**: `round_c_cap128_ttc` or `round_c_extended`

---

## Instances

```
Instances: 27071339-27071394 (15 RTX 4090s)
Average cost: $0.35/hr
Expected runtime: 30 min/run
Total cost: ~$5.25
```

---

## Tracking

All runs tracked in WandB with consistent naming:

- **Project**: `universal-simulator`
- **Entity**: `emgun-morpheus-space`
- **Group**: `overnight-sota`
- **Run Names**: Match config names (e.g., `round_a_lr29e5_w5pct`)
- **Tags**:
  - `overnight-sota` (all)
  - `round-a`, `round-b`, `round-c` (by round)
  - Hyperparameter-specific tags (e.g., `lr=3e-4`, `warmup=5%`)

**W&B Dashboard**: https://wandb.ai/emgun-morpheus-space/universal-simulator

---

## Analysis Plan

After completion (~10:45 UTC):

1. **Initial Triage** (filter by physics gates)
   - Mass conservation error < 0.01
   - Energy conservation error < 0.05
   - No negative densities

2. **Round A Winners** (top 2-3 by NRMSE)
   - Identify best LR × Warmup combination
   - Check training stability (loss curves)
   - Verify TTC improvement over baseline

3. **Round B Winners** (best capacity)
   - Compare NRMSE vs training time
   - Calculate performance per Joule/step
   - Check if larger models justify cost

4. **Round C Winners** (hybrid approaches)
   - Compare against best Round A/B
   - Identify winning combination
   - Check for overfitting (train/val gap)

5. **SOTA Assessment**
   - Best NRMSE vs target (<0.035)
   - If target not met: design Round D based on insights
   - If target met: validate on additional test sets

---

## Next Steps

### If Best NRMSE < 0.035:
✅ **SOTA achieved!**
- Run full evaluation on all test splits
- Generate performance report
- Promote to production config

### If Best NRMSE ≥ 0.035:
🔬 **Round D: Local Optimization**

Based on Round A-C results, likely candidates:
- **Curriculum Learning**: Progressive horizon expansion
- **Data Augmentation**: Noise injection, tau jittering
- **Architecture Tweaks**: Deeper models, attention window tuning
- **Regularization**: Dropout, weight decay, spectral penalties
- **Ensemble Methods**: Multi-model averaging

---

## Files Generated

- **Configs**: `configs/overnight_sota/*.yaml` (15 configs + manifest)
- **Scripts**:
  - `scripts/generate_overnight_sweep.py` (config generator)
  - `scripts/launch_overnight_sweep.py` (launcher)
- **Logs**: `logs/overnight_sota_launch.log` (launch record)

---

## Quick Commands

```bash
# Monitor instances
vastai show instances

# Check W&B runs
# Visit: https://wandb.ai/emgun-morpheus-space/universal-simulator?workspace=user-emgun

# Analyze results (after completion)
python scripts/analyze_sweep.py --group overnight-sota --target 0.035

# Re-launch failed runs (if any)
python scripts/launch_overnight_sweep.py --retry-failed
```

---

## Success Metrics

| Metric | Target | Current Best | Round A Goal | Round B Goal | Round C Goal |
|--------|--------|--------------|--------------|--------------|--------------|
| NRMSE (test) | <0.035 | 0.0921 | <0.065 | <0.050 | **<0.035** |
| Mass conservation | <0.01 | 0.003 | <0.005 | <0.005 | <0.005 |
| Energy conservation | <0.05 | 0.018 | <0.03 | <0.03 | <0.03 |
| Training time | <30 min | 25 min | <30 min | <35 min | <40 min |

---

**Status**: 🚀 Runs in progress...
**Next Check**: ~04:00 UTC (30% complete)
**Final Results**: ~10:45 UTC

