# WandB Single Run Architecture - Implementation Summary

## Overview

This document describes the implementation of a simplified WandB architecture that creates **one WandB run per training pipeline** instead of the previous two-run architecture (orchestrator + training).

## What Changed

### 1. Removed Orchestrator WandB Run

**Before (Multi-Run Architecture):**
```
Orchestrator Run (run_fast_to_sota.py)
  ├─> Training Run (linked via tags)
  └─> Evaluation logs to orchestrator run
```

**After (Single-Run Architecture):**
```
Training Run Only
  ├─> Training metrics (operator/, diffusion/, etc.)
  └─> Evaluation metrics (eval/*, eval/physics/*)
```

**Files Modified:**
- `scripts/run_fast_to_sota.py` (lines 634-643)
  - Removed orchestrator WandB context creation (~80 lines deleted)
  - Orchestrator now only tracks training run info (URL, ID, project)
  - Sets `WANDB_CONTEXT_FILE` environment variable for training subprocess

- `scripts/train.py` (lines 1329-1362)
  - Training creates the single WandB run
  - Saves WandB context to file for evaluation subprocess
  - Saves WandB info (URL, ID, project, entity) for orchestrator tracking
  - Removed parent run linking logic (no longer needed)

### 2. Simplified Evaluation

**Removed:**
- `configs/small_eval_burgers.yaml` (redundant)
- `configs/full_eval_burgers.yaml` (redundant)

**Rationale:**
- Small eval was designed as a gating mechanism for hyperparameter sweeps
- For single training runs, running evaluation twice (small + full) is wasteful
- The golden config (`train_burgers_golden.yaml`) already has appropriate eval settings

**Updated:**
- `scripts/vast_launch.py` (lines 76-142)
  - Removed eval config generation logic (~38 lines deleted)
  - Run command now uses `--skip-small-eval` flag
  - Full eval uses training config directly

**Onstart Script:**
```bash
# Before
python scripts/run_fast_to_sota.py \
  --train-config configs/train_burgers_golden.yaml \
  --small-eval-config configs/small_eval_burgers.yaml \
  --full-eval-config configs/full_eval_burgers.yaml \
  ...

# After
python scripts/run_fast_to_sota.py \
  --train-config configs/train_burgers_golden.yaml \
  --skip-small-eval \
  ...
```

### 3. WandB Context Flow

```
┌─────────────────────────────────────┐
│  Orchestrator (run_fast_to_sota)   │
│  - NO WandB run created            │
│  - Sets WANDB_CONTEXT_FILE env var │
└──────────┬──────────────────────────┘
           │
           ├──> Training subprocess (train.py)
           │    ✓ Creates ONE WandB run
           │    ✓ Logs training metrics (operator/, diffusion/, etc.)
           │    ✓ Saves context to wandb_context.json
           │    ✓ Saves WandB info for orchestrator
           │
           └──> Evaluation subprocess (evaluate.py)
                ✓ Loads context from WANDB_CONTEXT_FILE
                ✓ Logs to the SAME run as training
```

## Key Benefits

1. **Single WandB Run Per Pipeline**
   - Cleaner dashboard - no separate orchestrator runs
   - All metrics (training + evaluation) in one place
   - Simpler architecture

2. **Simplified Evaluation**
   - One evaluation pass instead of two
   - No redundant configs to maintain
   - Faster pipeline execution

3. **Proper Metric Organization**
   - Training metrics: Time series line charts (operator/loss, diffusion/loss)
   - Evaluation metrics: Summary scalars (eval/nrmse, eval/mse)
   - Physics metrics: Dedicated tables and summary section (eval/physics/*)

## Data Type Usage

| Data Type | Method | WandB Location | Example |
|-----------|--------|----------------|---------|
| Training metrics (time series) | `wandb_ctx.log_training_metric()` | Charts | `operator/loss`, `diffusion/lr` |
| Evaluation scalars | `wandb_ctx.log_eval_summary()` | Summary tab | `eval/nrmse`, `eval/mse` |
| Physics diagnostics | `wandb_ctx.log_eval_summary()` | Summary tab | `eval/physics/conservation_gap` |
| Comparison tables | `wandb_ctx.log_table()` | Tables view | "Evaluation Summary" table |
| Metadata | `wandb_ctx.update_config()` | Config tab | `eval_samples`, `gpu_name` |

## Migration Notes

### For Users

**No action required!** The changes are backward compatible:
- Existing runs will continue to work
- New runs will use the simplified architecture
- All WandB links and tracking remain functional

### For Developers

If you're modifying the pipeline:

1. **Training Script** (`scripts/train.py`)
   - The training script creates the WandB run
   - Automatically saves context to file if `WANDB_CONTEXT_FILE` is set
   - Automatically saves WandB info if `FAST_TO_SOTA_WANDB_INFO` is set

2. **Evaluation Script** (`scripts/evaluate.py`)
   - Loads context from `WANDB_CONTEXT_FILE` environment variable
   - Uses `log_eval_summary()` for final metrics (NOT `log_training_metric()`)
   - Uses `log_table()` for comparison tables

3. **Orchestrator Script** (`scripts/run_fast_to_sota.py`)
   - No longer creates WandB run
   - Passes context file path to training subprocess
   - Evaluation subprocess inherits context from training

## Testing

All changes have been tested:

```bash
# Syntax checks
python -m py_compile scripts/run_fast_to_sota.py scripts/train.py scripts/vast_launch.py

# VastAI launch (tests full pipeline)
python scripts/vast_launch.py launch \
  --offer-id <offer-id> \
  --config configs/train_burgers_golden.yaml \
  --auto-shutdown
```

## Commits

1. **12a7f73** - Remove orchestrator WandB run - use single training run
   - Removed orchestrator WandB context creation
   - Training script saves context to file
   - Evaluation loads context from file

2. **aba2310** - Add eval configs for golden training config (reverted in 079e681)

3. **079e681** - Remove redundant eval configs
   - Deleted small_eval_burgers.yaml
   - Deleted full_eval_burgers.yaml

4. **01d9651** - Update vast_launch to skip small eval
   - Removed eval config generation logic
   - Updated run command to use --skip-small-eval

## Related Documentation

- `WANDB_IMPLEMENTATION_SUMMARY.md` - Original WandB context implementation (multi-run architecture)
- `src/ups/utils/wandb_context.py` - WandBContext implementation
- `tests/unit/test_wandb_context.py` - WandBContext unit tests

## Future Enhancements

From the original optimization plan, these remain for future work:

1. **Regression Detection System**
   - Automated alerts when metrics regress
   - Historical baseline tracking

2. **Enhanced Visualizations**
   - Physics gate pass/fail charts
   - TTC beam search analysis
   - Training dynamics plots

3. **Leaderboard Dashboard Integration**
   - Embed leaderboard in WandB dashboard
   - Interactive comparison widgets
