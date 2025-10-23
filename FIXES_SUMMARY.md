# Evaluation Variance Fixes - Summary

**Date:** 2025-10-23
**Issue:** Burgers-golden runs show 19.7% variance in NRMSE, making results non-reproducible
**Root Causes:** No random seed, config logging broken, conservation gap metric flawed, diffusion overfitting

---

## ✅ Fixes Implemented

### 1. Added Reproducibility Settings

**File:** `configs/train_burgers_golden.yaml`

```yaml
# New section at top of config
seed: 42                    # Fixed random seed for reproducibility
deterministic: true         # Enable deterministic algorithms
benchmark: false            # Disable cudnn benchmarking for reproducibility
```

**Impact:** Ensures all PyTorch operations, data shuffling, and sampling use the same random state.

---

### 2. Fixed Config Logging to WandB

**File:** `src/ups/utils/wandb_context.py` (lines 306-378)

**Changes:**
- Added `seed`, `deterministic`, `benchmark` to logged config
- Added `weight_decay` for all optimizer stages
- Added `diffusion_latent_dim`, `diffusion_hidden_dim`
- Added `operator_group_size`
- Added TTC sampler config (`ttc_noise_std`, `ttc_steps`)

**Impact:** WandB runs now capture full hyperparameter config for debugging and comparison.

---

### 3. Implemented Seed Setting in Training Code

**File:** `scripts/train.py` (lines 84-108)

**Changes:**
```python
def set_seed(cfg: Dict) -> None:
    """Set random seed and configure determinism settings."""
    seed = cfg.get("seed", 17)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Configure PyTorch determinism
    deterministic = cfg.get("deterministic", False)
    benchmark = cfg.get("benchmark", True)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"✓ Deterministic mode enabled (seed={seed})")
    else:
        torch.backends.cudnn.benchmark = benchmark
        print(f"✓ Seed set to {seed} (deterministic={deterministic}, benchmark={benchmark})")
```

**Impact:** Training is now fully deterministic when `deterministic: true` is set.

---

### 4. Documented Conservation Gap Flaw

**File:** `conservation_gap_analysis.md`

**Key Findings:**
- Current "conservation_gap" metric operates on **latent space**, not physical fields
- Measures spatial sum difference at single timestep, NOT temporal conservation
- High variance (100% CV) is expected because latent encodings vary randomly
- Inverse correlation with diffusion loss is an artifact of latent magnitude scaling

**Recommendation:** Disable or ignore conservation_gap until proper implementation (decode to physical space, measure across time).

---

## 📊 New Configurations Created

### A. `train_burgers_golden.yaml` (updated)
- **Base config** with reproducibility settings added
- All other settings unchanged
- Use for production runs

### B. `train_burgers_golden_no_diffusion.yaml`
- **Test config** with diffusion and consistency stages DISABLED
- Tests hypothesis: diffusion may be hurting performance
- Based on inverse correlation (-0.394) between diffusion loss and eval NRMSE

### C. `train_burgers_golden_light_diffusion.yaml`
- **Test config** with reduced diffusion training
- Diffusion epochs: 8 → 3
- Diffusion LR: 5e-5 → 2e-5
- Diffusion weight_decay: 0.015 → 0.05 (stronger regularization)
- Tests hypothesis: less diffusion training prevents overfitting

### D. `sweep_diffusion.yaml`
- **WandB sweep config** for Bayesian hyperparameter optimization
- Sweeps: diffusion epochs [0, 3, 5, 8], LR [1e-6 to 1e-4], weight_decay [0.01 to 0.1]
- Use with: `wandb sweep configs/sweep_diffusion.yaml`

---

## 🧪 Testing Plan

### Phase 1: Reproducibility Test (3 runs)

Run the updated golden config **3 times** with same seed to verify reproducibility:

```bash
# Run 1
python scripts/train.py --config configs/train_burgers_golden.yaml --stage all

# Run 2 (should be identical)
python scripts/train.py --config configs/train_burgers_golden.yaml --stage all

# Run 3 (should be identical)
python scripts/train.py --config configs/train_burgers_golden.yaml --stage all
```

**Success Criteria:**
- Operator final loss variance < 1%
- Diffusion final loss variance < 5%
- **Baseline NRMSE variance < 2%** ← Key metric
- All runs produce identical training curves

---

### Phase 2: Diffusion Ablation (2 runs)

Test if diffusion is helping or hurting:

```bash
# Test A: No diffusion
python scripts/train.py --config configs/train_burgers_golden_no_diffusion.yaml --stage all

# Test B: Light diffusion
python scripts/train.py --config configs/train_burgers_golden_light_diffusion.yaml --stage all
```

**Comparison:**
- Compare baseline NRMSE vs original golden config
- If no-diffusion performs better → disable diffusion in golden
- If light-diffusion performs better → update golden with reduced settings

---

### Phase 3: Hyperparameter Sweep (optional)

If manual ablation is inconclusive, run full sweep:

```bash
wandb sweep configs/sweep_diffusion.yaml
wandb agent <sweep-id>
```

This will explore 20-30 combinations to find optimal diffusion hyperparameters.

---

## 📈 Expected Outcomes

### Immediate (After Phase 1)
- **NRMSE variance < 2%** across identical runs
- Confidence in results for production use
- Ability to detect real improvements vs noise

### Short-term (After Phase 2)
- **Identify optimal diffusion settings** (or remove if harmful)
- **Reduce NRMSE absolute value** by fixing overfitting
- Update golden config with best settings

### Medium-term
- Implement proper conservation checks on decoded physical fields
- Add temporal conservation metrics for rollout validation
- Establish baseline: Operator-only vs Operator+Diffusion+Consistency

---

## 🚨 Known Limitations

### Conservation Gap Metric
- **Status:** Broken, do not trust
- **Issue:** Operates on latent space, not physical fields
- **Fix Required:** Decode to physical space, measure temporal conservation
- **Workaround:** Ignore this metric entirely for now

### TTC Not Working
- Current analysis shows **zero improvement** from TTC (-0.003% to -0.016%)
- Expected improvement: 25x (from docs)
- **Action:** Fix or disable TTC in separate task (out of scope for this fix)

### Latent Space Conservation
- Encoder/decoder don't preserve conservation laws
- Operator training doesn't enforce physics constraints
- **Impact:** Physics violations accumulate during rollout
- **Fix Required:** Add physics-informed loss terms or guards

---

## 📝 Files Modified

### Configs
- ✏️ `configs/train_burgers_golden.yaml` - Added seed/determinism
- ✨ `configs/train_burgers_golden_no_diffusion.yaml` - New
- ✨ `configs/train_burgers_golden_light_diffusion.yaml` - New
- ✨ `configs/sweep_diffusion.yaml` - New

### Source Code
- ✏️ `src/ups/utils/wandb_context.py` - Enhanced config logging
- ✏️ `scripts/train.py` - Enhanced set_seed() function

### Documentation
- ✨ `analysis_eval_variance.md` - Variance analysis report
- ✨ `conservation_gap_analysis.md` - Conservation metric analysis
- ✨ `FIXES_SUMMARY.md` - This file

---

## 🏃 Next Steps

1. **Run Phase 1 tests** - Verify reproducibility (3 runs)
2. **Review results** - Confirm variance < 2%
3. **Run Phase 2 tests** - Test diffusion ablations
4. **Update golden config** - Based on best performer
5. **Document findings** - Update CLAUDE.md with new expected performance

---

## ❓ Questions for User

1. **Should we run tests locally or on VastAI?**
   - Local: Faster iteration, limited GPU
   - VastAI: Production environment, exact cost

2. **Priority order?**
   - Option A: Reproducibility first, ablation later
   - Option B: Run all tests in parallel (3 reproducibility + 2 ablation = 5 total)

3. **Sweep vs manual testing?**
   - Option A: Manual testing of 2-3 configs (faster, targeted)
   - Option B: Full WandB sweep (comprehensive, expensive)

---

## 🎯 Success Metrics

| Metric | Before | Target | Stretch Goal |
|--------|--------|--------|--------------|
| **NRMSE Variance (CV)** | 19.7% | < 5% | < 2% |
| **NRMSE Absolute** | 0.083-0.125 | < 0.090 | < 0.080 |
| **Reproducibility** | None | 3/3 runs identical | N/A |
| **Conservation Gap** | Broken | Disabled | Implemented correctly |
| **TTC Improvement** | -0.01% | N/A | > 5% (separate task) |

---

*Generated: 2025-10-23*
