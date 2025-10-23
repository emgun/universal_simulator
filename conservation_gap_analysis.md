# Conservation Gap Calculation Analysis

## Problem Summary

The `conservation_gap` metric shows extreme variance (0.87 to 13.43, 100.2% CV) across identical runs of burgers-golden, making it useless as a physics diagnostic. Root cause analysis reveals **fundamental flaws in the implementation**.

## Current Implementation

### Code Location
`src/ups/eval/pdebench_runner.py:178`

```python
# Physics-style diagnostics computed in latent coordinates
flat_pred = pred.reshape(pred.size(0), -1)
flat_target = target.reshape(target.size(0), -1)
conservation_batch = (flat_pred.sum(dim=1) - flat_target.sum(dim=1)).abs()
```

### What It Actually Measures

The current implementation:
1. **Operates on LATENT representations**, not physical fields (see comment on line 175)
2. **Compares spatial sum of prediction vs target** at a single timestep
3. **Does NOT measure conservation across time** (which is what conservation laws require)

### Why This Is Wrong

**Conservation laws** (mass, energy, momentum) state that certain quantities must be **constant over time**:

```
∫ ρ(x, t+Δt) dx = ∫ ρ(x, t) dx  (mass conservation)
```

The correct conservation gap should be:

```python
# For each trajectory
mass_t0 = physical_field[t=0].sum()
mass_t1 = physical_field[t=1].sum()
conservation_gap = |mass_t1 - mass_t0|
```

The current implementation instead computes:

```python
# At a single timestep
spatial_sum_pred = latent_pred.sum()  # Sum across space in latent space
spatial_sum_target = latent_target.sum()  # Sum across space in latent space
conservation_gap = |spatial_sum_pred - spatial_sum_target|
```

This is **not a conservation check**. It's measuring:
- How well the latent prediction sums to the same value as the latent target
- At a single point in time
- In latent space (which has no physical meaning)

## Why Variance Is High

The high variance (100% CV) occurs because:

1. **Latent representations are not unique**: Different training runs produce different latent encodings for the same physical solution
2. **Latent space has no conservation properties**: The encoder/decoder don't preserve spatial integrals
3. **Metric is meaningless**: Comparing latent sums measures encoding differences, not physics violations

### Example

Run 1:
- Latent encoding happens to have sum = 0.5
- Latent target has sum = 0.3
- **Conservation gap = 0.2** ✗ (meaningless)

Run 2:
- Latent encoding happens to have sum = 3.1
- Latent target has sum = 2.9
- **Conservation gap = 0.2** ✗ (same gap, different encoding)

Run 3:
- Latent encoding happens to have sum = 10.2
- Latent target has sum = 12.0
- **Conservation gap = 1.8** ✗ (different gap, just different latent scale)

None of these numbers tell you anything about actual mass/energy conservation in the physical solution!

## Correlation with Diffusion Loss

The analysis found **inverse correlation** between diffusion loss and conservation gap (-0.59):
- Lower diffusion training loss → Higher "conservation gap"
- This makes no physical sense

**Explanation**: Diffusion training optimizes for matching latent targets, which may increase the magnitude of latent values. Higher magnitude latent values → higher sums → higher "conservation gap" (even though physics is fine).

## Correct Implementation

### Option 1: Temporal Conservation (Proper Physics)

```python
def compute_temporal_conservation_gap(
    predictions: torch.Tensor,  # [batch, time, space, channels]
    field_idx: int = 0,  # Which channel (e.g., density)
) -> torch.Tensor:
    """
    Measure conservation violation across time.

    For true conservation: ∫ ρ(x,t) dx should be constant for all t
    """
    # Integrate over space for each timestep
    integrals = predictions[..., field_idx].sum(dim=-1)  # [batch, time]

    # Measure variance across time (should be zero for perfect conservation)
    initial = integrals[:, 0:1]  # [batch, 1]
    deviations = (integrals - initial).abs()  # [batch, time]

    # Return mean absolute deviation from initial value
    return deviations.mean()
```

### Option 2: Decoded Field Conservation

```python
def compute_decoded_conservation_gap(
    operator: LatentOperator,
    decoder: Decoder,
    latent_t0: LatentState,
    latent_t1: LatentState,
    query_points: torch.Tensor,
) -> float:
    """
    Measure conservation in PHYSICAL space, not latent space.
    """
    # Decode to physical fields
    field_t0 = decoder(latent_t0, query_points)  # [batch, n_points, channels]
    field_t1 = decoder(latent_t1, query_points)  # [batch, n_points, channels]

    # Compute spatial integrals
    mass_t0 = field_t0[..., 0].sum(dim=-1)  # Assuming channel 0 is density
    mass_t1 = field_t1[..., 0].sum(dim=-1)

    # Conservation gap
    return (mass_t1 - mass_t0).abs().mean().item()
```

### Option 3: Disable Until Fixed

The simplest short-term fix:

```python
# In pdebench_runner.py
# conservation_batch = (flat_pred.sum(dim=1) - flat_target.sum(dim=1)).abs()
# DISABLED: This metric is meaningless in latent space
conservation_batch = torch.zeros(pred.size(0), device=pred.device)
```

Then update docs to clarify that conservation diagnostics are not yet implemented.

## Recommendations

### Immediate (P0)

1. **Disable the current conservation_gap metric** - it's actively misleading
2. **Remove from evaluation reporting** or mark as "Not Implemented"
3. **Update analysis_eval_variance.md** to clarify this is not a real conservation check

### Short-term (P1)

4. **Implement Option 2**: Compute conservation on decoded physical fields
   - Requires decoder integration into evaluation
   - Adds overhead but provides real physics validation

### Medium-term (P2)

5. **Implement Option 1**: Temporal conservation on rollout predictions
   - Requires evaluation on multi-step rollouts
   - Most physically meaningful

6. **Add energy conservation**: Not just mass
   ```python
   energy = 0.5 * (u**2).sum()  # Kinetic energy for Burgers
   ```

7. **Add spectral conservation**: High-frequency energy should dissipate, not appear
   ```python
   spectrum_t0 = torch.fft.rfft(field_t0).abs()
   spectrum_t1 = torch.fft.rfft(field_t1).abs()
   # Check energy cascade direction
   ```

## Impact on Current Analysis

Given this finding, the `analysis_eval_variance.md` should be updated:

1. **Conservation gap variance is expected** - it's measuring random latent encoding differences
2. **Conservation gap is NOT a physics diagnostic** - ignore these values entirely
3. **Focus on NRMSE variance** - that's the real signal (19.7% CV is still problematic)
4. **Diffusion correlation makes sense** - it's just an artifact of latent magnitudes

## Testing

To verify the new implementation:

```python
# Perfect conservation test
def test_conservation_metric():
    # Create a solution with exact mass conservation
    t0 = torch.randn(10, 100)  # 10 samples, 100 spatial points
    t1 = t0 + torch.randn(10, 100) * 0.01  # Small perturbation
    t1 = t1 * (t0.sum(dim=1, keepdim=True) / t1.sum(dim=1, keepdim=True))  # Force conservation

    gap = compute_temporal_conservation_gap(torch.stack([t0, t1], dim=1), field_idx=0)
    assert gap < 1e-6, f"Conservation should be perfect, got gap={gap}"
```

## References

- Current implementation: `src/ups/eval/pdebench_runner.py:175-178`
- Alternative implementation: `src/ups/eval/metrics.py:26-29` (also flawed, but different)
- Physics checks module: `src/ups/eval/physics_checks.py:24-31` (also flawed)
