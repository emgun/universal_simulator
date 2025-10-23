# WandB Step Metric Fix

**Date:** 2025-01-22
**Status:** ✅ Fixed and tested

---

## Problem

Training charts appeared in WandB dashboard but **were completely empty** (no data points) even though metrics were being logged without errors.

### Symptoms
- Operator stage charts visible but empty
- No data points on loss/lr/grad_norm charts
- Other stages (diffusion_residual, consistency_distill) also empty
- No error messages

---

## Root Cause

**The fix documented in `WANDB_FIXES_SUMMARY.md` was never actually implemented in the code.**

In `src/ups/utils/wandb_context.py`, the `log_training_metric()` method at line 67 was logging:

```python
# BROKEN CODE
self.run.log({key: value}, step=step)
```

But WandB requires the step metric to exist **as actual data**, not just as a parameter:

```python
# FIXED CODE
step_key = f"training/{stage}/step"
self.run.log({key: value, step_key: step}, step=step)
```

### Why This Matters

WandB's charting works like this:

1. **Define the relationship** (lines 350-360 in `create_wandb_context()`):
   ```python
   wandb.define_metric("training/operator/step")
   wandb.define_metric("training/operator/*", step_metric="training/operator/step")
   ```
   This tells WandB: "Use `training/operator/step` as the x-axis for all `training/operator/*` charts"

2. **Log the data** (the broken part):
   ```python
   # Training code logs: training/operator/loss = 0.1 (with step=100 parameter)
   ```

3. **WandB tries to chart it**:
   - Looks for x-axis data: `training/operator/step`
   - **Cannot find it** (because it was never logged as data!)
   - Result: Empty chart

---

## The Fix

### Code Changes

**File:** `src/ups/utils/wandb_context.py` (lines 45-78)

```python
def log_training_metric(
    self, stage: str, metric: str, value: float, step: int
) -> None:
    """Log training metrics as time series.

    IMPORTANT: This logs BOTH the metric AND the step metric. WandB requires
    the step metric to exist as data for define_metric() to work correctly.
    Without logging the step metric, charts will appear empty!
    """
    if not self.enabled or self.run is None:
        return

    key = f"training/{stage}/{metric}"
    step_key = f"training/{stage}/step"  # ← NEW

    try:
        # Log both the metric AND the step metric for proper charting
        # WandB needs the step metric to exist as data for define_metric() to work
        self.run.log({key: value, step_key: step}, step=step)  # ← FIXED
    except Exception:
        pass
```

**Key change:** Now logs both `training/operator/loss: 0.1` AND `training/operator/step: 100` in the same call.

### Test Updates

**File:** `tests/unit/test_wandb_context.py`

1. **Updated `test_log_training_metric()`** (line 67-80):
   ```python
   # Now verifies both the metric AND step metric are logged
   mock_wandb_run.log.assert_called_once_with(
       {"training/operator/loss": 0.5, "training/operator/step": 10},
       step=10
   )
   ```

2. **Added `test_log_training_metric_multiple_stages()`** (line 83-113):
   - Tests operator, diffusion_residual, and consistency_distill stages
   - Verifies each stage gets its own step metric namespace

---

## Verification

### Unit Tests
```bash
pytest tests/unit/test_wandb_context.py -v
# ✅ 24/24 tests passed
```

### Syntax Validation
```bash
python -m py_compile src/ups/utils/wandb_context.py
# ✅ Syntax valid
```

### Related Tests
```bash
pytest tests/unit/test_training_logger.py -v
# ✅ 1 passed
```

---

## Expected Behavior After Fix

### Before (Broken)
```
WandB Dashboard:
├─ training/operator/loss (chart exists but empty)
├─ training/operator/lr (chart exists but empty)
└─ No diffusion_residual or consistency_distill charts
```

### After (Fixed)
```
WandB Dashboard:
├─ training/operator/loss ✅ (line chart with data points)
├─ training/operator/lr ✅ (line chart with data points)
├─ training/operator/grad_norm ✅ (line chart with data points)
├─ training/diffusion_residual/loss ✅ (line chart with data points)
├─ training/diffusion_residual/lr ✅ (line chart with data points)
├─ training/consistency_distill/loss ✅ (line chart with data points)
└─ All charts use proper step-based x-axis
```

---

## Why This Bug Existed

The bug was documented as "fixed" in `WANDB_FIXES_SUMMARY.md` but the actual code implementation was never completed. This is a classic case of:

1. **Documentation ahead of implementation** - The fix was designed and documented
2. **Missing verification** - Tests weren't updated to verify the new behavior
3. **Silent failure** - WandB didn't error, it just showed empty charts

---

## Lessons Learned

1. **Tests must verify behavior, not just code paths**
   - Old test checked that `log()` was called
   - Didn't check **what** was passed to `log()`

2. **WandB's API is subtle**
   - `define_metric(step_metric="foo")` doesn't create data
   - You must also log `foo` as actual data
   - The `step=` parameter is NOT the same as logging the step metric

3. **Documentation ≠ Implementation**
   - Always verify that documented fixes actually exist in code
   - Run the actual tests to confirm behavior

---

## Related Documents

- `WANDB_FIXES_SUMMARY.md` - Original fix documentation (now updated)
- `WANDB_CLEAN_ARCHITECTURE.md` - Architecture design
- `WANDB_IMPLEMENTATION_SUMMARY.md` - Implementation summary

---

## Commit Message

```
Fix WandB empty training charts by logging step metrics

Problem: Training charts appeared in WandB but were completely empty.

Root cause: The step metric (e.g., training/operator/step) was defined
via wandb.define_metric() but never actually logged as data. WandB
requires the step metric to exist as data points to use it as an x-axis.

Fix: Updated log_training_metric() to log both the metric AND the step
metric in the same call:
  self.run.log({key: value, step_key: step}, step=step)

This fix was documented in WANDB_FIXES_SUMMARY.md but never actually
implemented in the code. Now properly implemented and tested.

Changes:
- src/ups/utils/wandb_context.py: Log step metrics as data
- tests/unit/test_wandb_context.py: Add verification tests
- WANDB_FIXES_SUMMARY.md: Mark as implemented
- WANDB_STEP_METRIC_FIX.md: Detailed fix documentation

Tests: ✅ 24/24 passing
```

---

**Fix completed by:** Claude Code
**Verification status:** ✅ All tests passing
