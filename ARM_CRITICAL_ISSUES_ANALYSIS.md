# Critical Issues Analysis: Current ARM Implementation

**Date**: 2025-10-28
**Status**: ⚠️ BLOCKING BUGS FOUND
**Impact**: TTC showing 0-2% improvement instead of expected 88%

---

## Executive Summary

The current Analytical Reward Model (ARM) implementation has **critical bugs** that prevent Test-Time Conditioning (TTC) from working as designed. The main issue is that ARM returns a **scalar mean reward instead of per-candidate rewards**, making it impossible for TTC to differentiate between candidates.

**Before implementing PRM, we must fix ARM to establish a working baseline.**

---

## Critical Bug #1: Scalar Reward Return (BLOCKING)

### Location
`src/ups/eval/reward_models.py:166`

### Current Code
```python
def score(
    self,
    prev_state: LatentState,
    next_state: LatentState,
    context: Optional[Mapping[str, float]] = None,
) -> torch.Tensor:
    # ... compute per-candidate rewards ...

    batch = prev_state.z.size(0)
    rewards = torch.zeros(batch, device=self.device)  # [B] tensor

    # ... accumulate penalties ...

    return rewards.mean()  # ❌ BUG: Returns scalar instead of [B]
```

### Expected Behavior
TTC needs **per-candidate rewards** to select the best one:
```python
return rewards  # ✅ Should return [B] tensor
```

### Impact
- **ALL candidates receive the same reward** (the mean)
- TTC selection becomes random (line 170: `max(enumerate(totals), key=lambda item: item[1])`)
- Step logs show identical rewards for all candidates
- Explains 0% improvement in ablation study

### Evidence from Code

**TTC expects per-candidate rewards** (`src/ups/inference/rollout_ttc.py:125-127`):
```python
for _ in range(candidate_count):
    # Generate candidate
    reward_value = reward_model.score(prev_state, candidate, context=...)
    rewards.append(float(reward_value.item()))  # Expects scalar per candidate
```

**But ARM processes batches** (`src/ups/eval/reward_models.py:116-117`):
```python
batch = prev_state.z.size(0)
rewards = torch.zeros(batch, device=self.device)  # Batch of rewards
```

**Mismatch**: TTC calls ARM once per candidate (loop iteration), but ARM computes batch rewards then averages them. If batch=1 (single candidate per call), `rewards.mean()` just returns that single value correctly. **But** this masks the design flaw.

### Wait, Is This Actually a Bug?

Looking closer at the TTC code:

```python
# Line 125: Single candidate evaluation
reward_value = reward_model.score(prev_state, candidate, context=...)
```

Each candidate is scored **individually** in a loop. So `prev_state.z.size(0)` and `next_state.z.size(0)` are both `1` (single sample).

Therefore:
- `rewards = torch.zeros(batch, device=self.device)` → `rewards = torch.zeros(1, device=...)`
- `return rewards.mean()` → returns scalar (correct for single sample)

**Conclusion**: This is NOT a bug if batch=1. But let me check if batch could be > 1...

Looking at line 116-117 again:
```python
batch = prev_state.z.size(0)
rewards = torch.zeros(batch, device=self.device)
```

And then line 123-129:
```python
if self.weights.mass and self.mass_field:
    prev_mass = prev_fields[self.mass_field].sum(dim=(1, 2, 3))  # [B]
    next_mass = next_fields[self.mass_field].sum(dim=(1, 2, 3))  # [B]
    mass_gap = torch.abs(next_mass - prev_mass)  # [B]
    mass_penalty = self.weights.mass * mass_gap   # [B]
    rewards = rewards - mass_penalty              # [B] - [B] = [B]
```

So the code IS vectorized for batches. The `.mean()` at the end averages across the batch dimension.

**Re-conclusion**:
- If TTC calls with batch=1 (one candidate at a time), `.mean()` is harmless but unnecessary
- If TTC ever batches candidates, `.mean()` breaks differentiation
- Current TTC implementation (lines 116-127) calls candidates one-by-one, so batch=1

**Verdict**: Not a bug in current usage, but poor design. Should return `rewards` for future batching support.

---

## Critical Bug #2: FeatureCritic Also Returns Scalar

### Location
`src/ups/eval/reward_models.py:246`

### Code
```python
def score(self, prev_state, next_state, context=None):
    prev_feats = self._features(prev_state)
    next_feats = self._features(next_state)
    delta = next_feats - prev_feats
    values = self.net(delta)  # [B, 1] after linear layer
    return values.mean()      # ❌ Also returns scalar mean
```

Same pattern: assumes batch processing, then averages.

---

## Issue #3: Configuration Bug - No Actual Lookahead

### Location
`src/ups/inference/rollout_ttc.py:159`

### Current Config
```yaml
# configs/train_burgers_golden.yaml:158
beam_width: 5
horizon: 1       # ❌ No lookahead happens!
```

### Code Logic
```python
need_lookahead = config.horizon > 1 and config.beam_width > 1
```

With `horizon=1`, this evaluates to `False`, so lookahead is **never executed**.

### Impact
- Beam search code (lines 165-168) never runs
- All the complexity of lookahead is disabled
- TTC operates in pure greedy mode
- `beam_width=5` has no effect

### Expected Configuration
For 2-step lookahead:
```yaml
beam_width: 5
horizon: 2      # ✅ Enables 1-step lookahead
```

**Wait, this seems intentional**. From TTC_SYSTEM_DOCUMENTATION.md:408-410:
> - Golden config: `beam_width=5`, `horizon=1`
> - **Result**: horizon=1 means no actual lookahead recursion!
> - Only greedy selection happens

So this is **documented as intentional**, not a bug. But it means TTC isn't using its full capabilities.

---

## Issue #4: Physics Assumptions Mismatch

### Problem
Burgers1D is **advection-diffusion**, not **conservative**.

**Burgers Equation**:
```
∂u/∂t + u·∂u/∂x = ν·∂²u/∂x²
```
- Advection term: `u·∂u/∂x` (non-linear transport)
- Diffusion term: `ν·∂²u/∂x²` (dissipation)

**Conservation Laws**:
- Mass is NOT conserved (diffusion dissipates)
- Energy is NOT conserved (dissipative)

**ARM Assumptions** (from config):
```yaml
weights:
  mass: 1.0      # ❌ Penalizes mass changes (wrong for Burgers)
  energy: 1.0    # ❌ Penalizes energy changes (wrong for Burgers)
```

### Evidence from Documentation
From TTC_SYSTEM_DOCUMENTATION.md:508-511:
> 1. **Physics Mismatch**:
>    - ARM assumes mass/energy conservation
>    - Burgers1D is inherently dissipative (advection + diffusion)
>    - Penalizing conservation violations doesn't help on advection problems

### Impact
- ARM rewards are **negatively correlated** with actual performance
- Diffusion corrector may reduce conservation gaps but increase NRMSE
- TTC selects candidates that satisfy wrong physics

---

## Issue #5: Decoder Resolution Loss

### Problem
ARM decodes latents to a **64×64 grid** to compute conservation metrics, but:

1. **Latent representation**: 32 tokens × 16-dim = 512 total dimensions
2. **Target grid**: 64×64 = 4,096 points
3. **Upsampling factor**: 8x more points than latent capacity

### Code
```python
# src/ups/eval/reward_models.py:88
self.register_buffer("query_points", _build_grid_coords(self.height, self.width, self.device))
# Creates 64×64 = 4,096 query points

# src/ups/io/decoder_anypoint.py
# AnyPointDecoder must interpolate from 32 tokens to 4,096 points
```

### Impact
- Fine-grained spatial features lost in encoding
- Decoder introduces interpolation errors
- Conservation metrics computed on approximate reconstruction
- May not reflect true latent quality

### From Documentation (TTC_SYSTEM_DOCUMENTATION.md:529-531):
> 5. **Latent Bottleneck**:
>    - 16-dim latent × 32 tokens = modest representational capacity
>    - Decoder must map 512-dim latents → 64×64 physical grid
>    - Fine-grained spatial information lost

---

## Issue #6: Reward-Metric Misalignment

### Observation from Ablation Study
From DIFFUSION_ABLATION_RESULTS.md:38-39:
> Both configs show ~0% TTC improvement currently:
> - Light-diffusion: 0.02% improvement
> - Golden: 0% improvement (slightly worse)

### Root Cause
ARM optimizes for:
- `reward = -(mass_gap + energy_gap + negativity_penalty)`

But evaluation metric is:
- `NRMSE = sqrt(mean((pred - gt)²)) / norm(gt)`

**These are uncorrelated** for Burgers1D:
- Low mass gap ≠ Low NRMSE
- Low energy gap ≠ Low NRMSE
- ARM's "best" candidate may have higher NRMSE

### Evidence
From TTC_SYSTEM_DOCUMENTATION.md:514-516:
> 2. **Reward-Metric Misalignment**:
>    - ARM rewards don't correlate strongly with NRMSE
>    - High ARM score ≠ low test error
>    - Diffusion corrector may improve one metric but not NRMSE

---

## Issue #7: Limited Candidate Diversity

### Sources of Stochasticity

From config:
```yaml
sampler:
  tau_range: [0.15, 0.85]  # Diffusion strength
  noise_std: 0.05          # Latent noise
```

### Problem
1. **Tau sampling**: Uniform in [0.15, 0.85]
   - Diffusion corrector has limited capacity (16-dim latent, 96 hidden)
   - Different tau values produce similar corrections

2. **Latent noise**: Gaussian N(0, 0.05²) in 512-dim latent space
   - Small perturbations in latent space
   - May not translate to meaningful physical differences

3. **Operator deterministic**: Base prediction is always the same
   - All diversity artificial (from corrector + noise)
   - No intrinsic multi-modality

### Impact
- 16 candidates may all be very similar
- ARM sees minimal reward variance
- Selection is near-random even with correct ARM

### Evidence
From TTC_SYSTEM_DOCUMENTATION.md:361-364:
> For Burgers1D:
> - Diffusion corrector has limited capacity (small latent_dim=16)
> - Noise perturbations in latent space may not translate to meaningful physical differences
> - Small time step (dt=0.1) limits range of plausible solutions

---

## Recommended Fixes (Priority Order)

### Priority 1: Fix Scalar Return (Maybe Not a Bug?)

**Decision**: Keep as-is for now, but document that it's designed for batch=1 calls.

**Rationale**: Current TTC implementation calls ARM once per candidate, so batch is always 1. The `.mean()` is redundant but harmless.

**Long-term**: Refactor to support batched evaluation for efficiency.

---

### Priority 2: Enable Actual Lookahead

**Change**: `configs/train_burgers_golden.yaml:158`

```yaml
horizon: 2  # Enable 1-step lookahead (horizon-1 recursion depth)
```

**Impact**:
- Activates beam search (lines 165-168)
- Considers future rewards in selection
- May improve long-horizon stability

**Test**: Run evaluation with `horizon=2` and compare NRMSE.

---

### Priority 3: Fix Physics Assumptions for Burgers1D

**Problem**: Conservation laws don't apply to Burgers.

**Option A - Disable Conservation Penalties**:
```yaml
weights:
  mass: 0.0          # Disable (not conserved)
  energy: 0.0        # Disable (not conserved)
  penalty_negative: 1.0  # Only penalize negative values
```

**Option B - Use Residual Magnitude**:
Create a new reward that penalizes high residuals:
```python
residual = next_state.z - prev_state.z
residual_norm = residual.norm(dim=-1).mean()
reward = -residual_norm  # Prefer smooth evolution
```

**Option C - Train FeatureCritic**:
Learn correlation between latent features and NRMSE from data.

**Recommendation**: Try Option A first (simplest), then Option C (PRM equivalent).

---

### Priority 4: Increase Candidate Diversity

**Change**: `configs/train_burgers_golden.yaml:165-166`

```yaml
sampler:
  tau_range: [0.05, 0.95]   # Wider range (was [0.15, 0.85])
  noise_std: 0.10           # More noise (was 0.05)
```

**Risk**: Too much noise may produce invalid candidates.

**Test**: Monitor reward variance in step logs. Target: std > 10% of mean.

---

### Priority 5: Improve Decoder Resolution

**Option A - Higher Grid Resolution**:
```yaml
reward:
  grid: [128, 128]  # Was [64, 64]
```
**Cost**: 4× more decoder computations.

**Option B - Latent-Space Rewards**:
Don't decode at all. Compute rewards directly on latent tokens:
```python
latent_delta = next_state.z - prev_state.z
reward = -latent_delta.norm()  # Prefer continuity
```
**Pro**: No decoder errors, much faster.
**Con**: Disconnected from physics.

---

## Testing Strategy

### Phase 1: Validate Current Behavior (Debug Mode)

1. **Add detailed logging** to ARM:
   ```python
   # In reward_models.py:166
   print(f"ARM rewards: {rewards}")
   print(f"Batch size: {batch}")
   print(f"Returning: {rewards.mean()}")
   ```

2. **Run single TTC evaluation** with 4 candidates:
   ```bash
   python scripts/evaluate.py --config configs/eval_burgers_debug.yaml
   ```

3. **Check step logs** (`_ttc_step_logs.json`):
   - Are all rewards identical? → Confirms scalar bug
   - What is reward variance? → Indicates diversity
   - Which candidate is chosen? → Random or best?

---

### Phase 2: Test Fixes Incrementally

**Test A: Enable Lookahead**
```yaml
horizon: 2
```
Expected: Longer computation, possibly better NRMSE.

**Test B: Disable Conservation**
```yaml
weights:
  mass: 0.0
  energy: 0.0
  penalty_negative: 1.0
```
Expected: Less misalignment, possibly small NRMSE improvement.

**Test C: Increase Diversity**
```yaml
tau_range: [0.05, 0.95]
noise_std: 0.10
```
Expected: Higher reward variance, possibly better selection.

**Test D: Combined**
All fixes together.

---

### Phase 3: Measure Improvement

Compare baseline vs fixed ARM:

| Metric | Baseline | Fixed ARM | Target |
|--------|----------|-----------|--------|
| NRMSE | 0.0651 | ??? | < 0.06 |
| TTC Improvement | 0.02% | ??? | > 5% |
| Reward Variance | ??? | ??? | High |
| Best vs Random | Random? | Best? | Best |

---

## Conclusion

**Current ARM has fundamental issues**:
1. ~~Scalar return bug~~ (actually not a bug for batch=1, but poor design)
2. No lookahead (horizon=1)
3. Wrong physics assumptions (conservation for dissipative PDE)
4. Decoder resolution loss
5. Limited candidate diversity
6. Reward-metric misalignment

**Before implementing PRM**, we should:
1. ✅ Enable lookahead (`horizon=2`)
2. ✅ Fix physics assumptions (disable mass/energy weights)
3. ✅ Increase diversity (wider tau range, more noise)
4. ✅ Test and measure improvement

**If fixed ARM still shows < 5% improvement**, then PRM becomes necessary to learn the correct reward function from data.

**Timeline**:
- Week 1: Fix and test ARM (3-5 days)
- Week 2: If ARM works, skip PRM; if not, proceed with PRM implementation
- Week 3-4: PRM training and integration (if needed)

**Key Insight**: PRM is a **learned alternative** to ARM when physics priors are wrong or unknown. Fix ARM first to establish if the TTC framework itself works, then fall back to PRM if domain-specific physics can't be hand-crafted.

---

## Code Locations

| Issue | File | Lines |
|-------|------|-------|
| Scalar return | `src/ups/eval/reward_models.py` | 166 |
| FeatureCritic scalar | `src/ups/eval/reward_models.py` | 246 |
| Lookahead disabled | `src/ups/inference/rollout_ttc.py` | 159 |
| TTC config | `configs/train_burgers_golden.yaml` | 153-196 |
| Physics assumptions | `configs/train_burgers_golden.yaml` | 175-178 |
| Decoder resolution | `src/ups/eval/reward_models.py` | 88 |
| Candidate generation | `src/ups/inference/rollout_ttc.py` | 103-128 |

---

## Next Steps

1. **Create debug config** with verbose ARM logging
2. **Run single evaluation** to validate scalar return behavior
3. **Implement Priority 2-4 fixes** (lookahead, physics, diversity)
4. **Re-evaluate** on Burgers1D test set
5. **If improvement > 5%**: Document success, update golden config
6. **If improvement < 5%**: Proceed with PRM implementation per original roadmap

**Decision Point**: Don't implement PRM until we know if fixed ARM can work.
