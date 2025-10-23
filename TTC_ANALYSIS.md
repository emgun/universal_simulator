# Test-Time Conditioning (TTC) Analysis

**Date:** 2025-10-23
**Issue:** TTC completes successfully but provides 0% improvement across all recent runs

## Status Summary

✅ **TTC IS working** - evaluation completes and metrics are logged
❌ **TTC provides ZERO improvement** - all recent runs show 0.00-0.02% improvement
⚠️ **Auto-shutdown fails** - instances hang after TTC completes (separate issue)

## Historical Performance Data

| Run Date | Run ID | Baseline NRMSE | TTC NRMSE | Improvement |
|----------|--------|----------------|-----------|-------------|
| 2025-10-23 | train-20251023_032316 | **0.0651** | 0.0651 | **0.02%** |
| 2025-10-23 | train-20251023_032317 | 0.0742 | 0.0742 | -0.00% |
| 2025-10-23 | train-20251023_032045 | 0.0803 | 0.0803 | 0.00% |
| 2025-10-23 | train-20251023_012202 | 0.1247 | 0.1247 | -0.00% |
| 2025-10-23 | train-20251023_000127 | 0.0831 | 0.0831 | -0.01% |
| 2025-10-22 | train-20251022_221628 | 0.1047 | 0.1047 | -0.01% |
| 2025-10-22 | 19pkjths | 0.0849 | 0.0849 | -0.02% |
| 2025-10-22 | **ptxr87mw** (ref) | 0.0782 | 0.0782 | **-0.01%** |
| 2025-10-21 | dlf1s9ic | 1.1389 | 1.1390 | -0.00% |
| 2025-10-21 | gj4k53mk | 1.8853 | 1.8853 | -0.00% |

**Finding:** TTC has provided ZERO measurable improvement in ALL runs dating back to at least Oct 21.

## CLAUDE.md Documentation Discrepancy

**Claim in CLAUDE.md:**
> "TTC NRMSE: ~0.078 (minimal improvement, but stable)"
> "Reference run: ptxr87mw (NRMSE: 0.0782 - 25x improvement)"

**Reality:**
- Run ptxr87mw: Baseline 0.0782 → TTC 0.0782 (improvement: -0.01%)
- The "25x improvement" claim appears to be **incorrect or outdated**

## Why TTC Isn't Helping

### Evidence from Logs

From instance 27168505 logs, TTC beam search rewards:
```
[INFO] step=0 rewards=[-554.54, -554.77, -554.63, -554.80, -554.73, -554.62, -554.59, -554.76]
[INFO] step=0 rewards=[-588.32, -588.28, -588.44, -588.37, -588.45, -588.76, -588.56, -588.39]
```

**Analysis:**
- All 8 candidates have rewards within **~0.5%** of each other
- Beam search is choosing between nearly identical options
- The analytical reward function is **not discriminating** between good and bad trajectories

### Root Cause Hypotheses

#### 1. Reward Function Not Sensitive Enough
The analytical rewards (mass, energy, momentum conservation) may be:
- Too coarse-grained to detect quality differences
- Dominated by prediction errors rather than physics violations
- Negative across all candidates (all violate physics similarly)

#### 2. Candidates Too Similar
With only 8 candidates and noise_std=0.015, the TTC sampler may be:
- Generating very similar trajectories
- Not exploring diverse enough regions
- Noise is too small to create meaningful variation

#### 3. Baseline Already Excellent
At baseline NRMSE ~0.065-0.080:
- The model already satisfies physics well
- Little room for TTC to improve via physics rewards
- Any gains are within measurement noise

#### 4. Decoder Quality Issues
If the decoder used for reward computation is inaccurate:
- Decoded fields don't match true physics
- Conservation calculations are meaningless
- Rewards don't correlate with actual solution quality

## Configuration Analysis

Current TTC config (from train_burgers_golden.yaml):

```yaml
ttc:
  enabled: true
  steps: 1
  candidates: 8
  beam_width: 3
  horizon: 1
  residual_threshold: 0.35
  gamma: 1.0
  max_evaluations: 150

  sampler:
    tau_range: [0.15, 0.85]
    noise_std: 0.015              # ← Very low noise
    noise_schedule: [0.03, 0.015, 0.005]

  reward:
    analytical_weight: 1.0
    grid: [64, 64]
    mass_field: rho
    energy_field: e
    momentum_field: []

    weights:
      mass: 1.2
      energy: 0.15                # ← Energy weighted very low
      penalty_negative: 0.6
```

**Issues:**
1. **Low noise_std (0.015)** - may not create diverse candidates
2. **Energy weight very low (0.15)** - may miss energy violations
3. **No momentum conservation** - missing physics constraint
4. **Only 8 candidates** - small search space

## Debugging Plan

### Phase 1: Verify Reward Function (High Priority)

1. **Log reward components separately**
   ```python
   # In reward_models.py, log:
   - mass_conservation_gap (per candidate)
   - energy_conservation_gap (per candidate)
   - negativity_penalty (per candidate)
   - total_reward (per candidate)
   ```

2. **Check if rewards correlate with NRMSE**
   - Do candidates with better rewards actually have lower error?
   - If not, reward function is broken

3. **Visualize decoded fields**
   - Are decoded fields physically reasonable?
   - Do they match ground truth?

### Phase 2: Increase Candidate Diversity (Medium Priority)

1. **Increase noise_std**
   ```yaml
   noise_std: 0.05  # Increase from 0.015
   ```

2. **Increase candidate count**
   ```yaml
   candidates: 16  # Increase from 8
   ```

3. **Expand beam width**
   ```yaml
   beam_width: 5  # Increase from 3
   ```

### Phase 3: Improve Reward Function (Medium Priority)

1. **Balance reward weights**
   ```yaml
   weights:
     mass: 1.0
     energy: 1.0      # Increase from 0.15
     penalty_negative: 0.5
   ```

2. **Add momentum conservation** (if applicable to Burgers)
   ```yaml
   momentum_field: [u]  # If velocity field available
   ```

3. **Add spectral energy check**
   - Penalize unphysical high-frequency modes
   - Reward smooth, physically plausible solutions

### Phase 4: Validate on Known Cases (Low Priority)

1. **Create synthetic test**
   - Generate trajectory with known physics violation
   - Verify TTC can detect and correct it

2. **Compare to ground truth**
   - Use exact Burgers solution if available
   - Measure if TTC moves closer to truth

## Auto-Shutdown Issue (Separate)

**Problem:** Instances don't shutdown after TTC completes

**Current behavior:**
1. Training completes ✓
2. Baseline eval completes ✓
3. TTC eval completes ✓
4. WandB run finishes ✓
5. Script continues running (logs show TTC reward computation) ❌
6. Auto-shutdown never triggers ❌

**Possible causes:**
1. TTC eval enters infinite loop
2. Script waits for subprocess that never terminates
3. Auto-shutdown code never reached

**Fix:**
1. Add timeout to TTC evaluation
2. Add explicit shutdown after WandB finish
3. Or use VastAI's built-in auto-shutdown (max_runtime)

## Recommendations

### Immediate (Do First)
1. ✅ **Keep TTC enabled** - user has seen gains historically
2. **Add reward component logging** - understand what's happening
3. **Increase noise_std to 0.05** - create more diverse candidates

### Short-term (This Week)
4. **Fix auto-shutdown** - add timeout or explicit destroy
5. **Increase energy weight to 1.0** - balance physics rewards
6. **Test with 16 candidates** - expand search space

### Long-term (When Time Permits)
7. **Validate reward function** - ensure it measures what we want
8. **Add spectral reward** - penalize unphysical modes
9. **Hyperparameter sweep TTC** - optimize candidates, noise, weights

## Success Criteria

TTC should be considered "working" when:
1. **Improvement > 5%** on baseline NRMSE
2. **Reward variance > 5%** across candidates (showing discrimination)
3. **Best reward correlates** with lowest NRMSE (r > 0.5)

## References

- Config: `configs/train_burgers_golden.yaml:143-185`
- Reward model: `src/ups/eval/reward_models.py:80-150`
- TTC runner: `src/ups/eval/pdebench_runner.py` (TTC integration)
- Analysis doc: `analysis_eval_variance.md`
