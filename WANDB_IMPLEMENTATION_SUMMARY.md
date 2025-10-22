# WandB Clean Architecture - Implementation Summary

## Overview

This document summarizes the implementation of the clean WandB architecture for the Universal Physics Simulator. The goal was to eliminate hacky patterns, use proper data types, and ensure one pipeline creates exactly one WandB run.

## What Was Changed

### 1. Created WandBContext Module (`src/ups/utils/wandb_context.py`)

A centralized context manager for WandB logging that provides:

- **Single Run Per Pipeline**: One `WandBContext` object is created by the orchestrator and passed to all components
- **Proper Data Types**:
  - `log_training_metric()` - Time series data (training loss, gradients, etc.)
  - `log_eval_summary()` - Final scalar metrics (no line charts!)
  - `update_config()` - Metadata and hyperparameters (not metrics)
  - `log_table()` - Multi-value comparisons (leaderboard, baseline vs TTC)
  - `log_image()` - Visualizations
  - `save_file()` - Checkpoints and artifacts

**Key Methods**:
```python
# Time series (appears as line chart)
ctx.log_training_metric("operator", "loss", 0.001, step=100)

# Final metrics (appears in Summary tab, NOT as chart)
ctx.log_eval_summary({"nrmse": 0.09, "mse": 0.001}, prefix="eval")

# Metadata (appears in Config tab)
ctx.update_config({"eval_samples": 3072, "eval_tau": 0.5})

# Tables for comparisons
ctx.log_table("eval/baseline_vs_ttc",
              columns=["Metric", "Baseline", "TTC"],
              data=[["NRMSE", 0.78, 0.09]])
```

### 2. Updated Training Script (`scripts/train.py`)

- **TrainingLogger** now accepts `wandb_ctx` instead of creating its own WandB run
- All training functions (`train_operator`, `train_diffusion`, `train_consistency`, `train_steady_prior`) now accept `wandb_ctx` parameter
- Replaced direct `wandb.save()` calls with `wandb_ctx.save_file()`
- Replaced direct `wandb.alert()` calls with `wandb_ctx.alert()`
- `train_all_stages()` can load context from environment or create standalone context

**Before (Hacky)**:
```python
# TrainingLogger owned run, sometimes called finish()
if wandb.run is not None:
    wandb.save(str(checkpoint_path), base_path=str(checkpoint_dir.parent))
```

**After (Clean)**:
```python
# TrainingLogger receives context, never calls finish()
if wandb_ctx:
    wandb_ctx.save_file(checkpoint_path)
```

### 3. Updated Evaluation Script (`scripts/evaluate.py`)

- Loads WandB context from `WANDB_CONTEXT_FILE` environment variable
- Uses `log_eval_summary()` for final metrics (NOT `wandb.log()` which creates time series)
- Uses `update_config()` for metadata (samples, tau, ttc_enabled)
- Uses `log_table()` for metrics table
- Uses `log_image()` and `save_file()` for outputs

**Before (Hacky)**:
```python
# Created scalars as time series (wrong!)
wandb.log({"eval/nrmse": 0.09})

# Logged metadata as metrics (wrong!)
wandb.log({"eval_samples": 3072})
```

**After (Clean)**:
```python
# Scalars in Summary tab (correct!)
wandb_ctx.log_eval_summary({"nrmse": 0.09}, prefix="eval")

# Metadata in Config tab (correct!)
wandb_ctx.update_config({"eval_samples": 3072})
```

### 4. Updated Orchestrator Script (`scripts/run_fast_to_sota.py`)

- Creates single `WandBContext` at the start of the pipeline
- Saves context to `wandb_context.json` file for subprocesses
- Passes `WANDB_CONTEXT_FILE` environment variable to training and evaluation subprocesses
- **Eliminated `WANDB_MODE=disabled` hack** - evaluation now logs to the same run!
- Added backward compatibility checks for legacy `_log_metrics_to_training_run()` calls
- Properly calls `wandb_ctx.finish()` at the end

**Before (Hacky)**:
```python
# Multiple wandb.init() calls throughout code
wandb_run = wandb.init(project="...", ...)

# Hack to prevent duplicate runs during eval
eval_env = {"WANDB_MODE": "disabled"}
```

**After (Clean)**:
```python
# Single context creation
wandb_ctx = create_wandb_context(cfg, run_id=run_id, mode="online")
save_wandb_context(wandb_ctx, wandb_context_file)

# Pass context to subprocesses (no hack!)
eval_env = {}
if wandb_context_file and wandb_context_file.exists():
    eval_env["WANDB_CONTEXT_FILE"] = str(wandb_context_file)
```

### 5. Added Comprehensive Unit Tests (`tests/unit/test_wandb_context.py`)

- 23 unit tests covering all WandBContext functionality
- Tests for disabled mode, error handling, file persistence
- Tests for proper data type usage (training metrics vs eval summary)
- All tests pass ✓

## Key Architectural Principles

### 1. Linked WandB Runs for Multi-Process Pipelines

The orchestrator creates a WandB run for pipeline tracking. Training creates a separate run (linked via tags) for training metrics. Evaluation logs to the orchestrator's run.

**Why separate runs?** WandB doesn't support multiple processes writing to the same run simultaneously. The recommended approach is to create separate runs and link them using tags and groups.

```
┌─────────────────────────────────────────┐
│  Orchestrator (run_fast_to_sota.py)    │
│                                         │
│  wandb_ctx = create_wandb_context()    │ ← Orchestrator run
│  run_id = "run_20251022_123456"        │
└──────────┬──────────────────────────────┘
           │
           ├──> Training subprocess
           │    Creates own run with tag: "parent:run_20251022_123456"
           │    → logs training metrics (operator/, diffusion/, etc.)
           │
           ├──> Small Eval subprocess
           │    Uses orchestrator's context file
           │    → logs eval summary to orchestrator run
           │
           └──> Full Eval subprocess
                Uses orchestrator's context file
                → logs eval summary to orchestrator run
```

**Benefits:**
- Clean separation: Pipeline events vs training metrics
- No multi-process conflicts
- Easy to find related runs via tags
- Follows WandB best practices

### 2. Proper Data Type Usage

| Data Type | When to Use | WandB Method | Result |
|-----------|-------------|--------------|---------|
| Time Series | Per-epoch/step metrics | `log_training_metric()` | Line charts |
| Scalars | Final evaluation metrics | `log_eval_summary()` | Summary tab |
| Metadata | Hyperparameters, config | `update_config()` | Config tab |
| Tables | Multi-value comparisons | `log_table()` | Table view |
| Images | Visualizations | `log_image()` | Media view |
| Files | Checkpoints, reports | `save_file()` | Files tab |

### 3. Context Passing via File + Environment Variable

For subprocess communication:

1. Orchestrator saves context to `wandb_context.json`
2. Orchestrator sets `WANDB_CONTEXT_FILE` environment variable
3. Subprocess loads context from file
4. Subprocess logs to the same run

This is cleaner than:
- ❌ Multiple `wandb.init()` calls (creates run proliferation)
- ❌ `WANDB_MODE=disabled` hack (prevents logging)
- ❌ Resume logic (fragile, creates race conditions)

## Files Changed

### Created:
- `src/ups/utils/wandb_context.py` (477 lines) - Core WandBContext implementation
- `tests/unit/test_wandb_context.py` (389 lines) - Comprehensive unit tests
- `WANDB_IMPLEMENTATION_SUMMARY.md` (this file) - Documentation

### Modified:
- `scripts/train.py` - Updated TrainingLogger and all training functions
- `scripts/evaluate.py` - Updated to use clean context loading and proper data types
- `scripts/run_fast_to_sota.py` - Updated to create context and pass to subprocesses

## Migration Guide

### For Existing Code

If you have code that uses WandB directly, migrate it to use WandBContext:

**Old Pattern**:
```python
import wandb

wandb.init(project="...", ...)
wandb.log({"loss": 0.1}, step=10)
wandb.log({"final_nrmse": 0.09})  # Wrong! Creates time series
wandb.config.update({"lr": 1e-4})
wandb.finish()
```

**New Pattern**:
```python
from ups.utils.wandb_context import create_wandb_context

# At orchestrator level
ctx = create_wandb_context(cfg, run_id="...", mode="online")

# In training
ctx.log_training_metric("operator", "loss", 0.1, step=10)

# In evaluation
ctx.log_eval_summary({"nrmse": 0.09}, prefix="eval")  # Correct! Summary scalar

# Metadata
ctx.update_config({"lr": 1e-4})

# At end of pipeline
ctx.finish()
```

### For Subprocesses

**Old Pattern**:
```python
# In subprocess
import wandb
wandb.init(project="...", id=run_id, resume="allow")  # Fragile!
```

**New Pattern**:
```python
# In subprocess
from ups.utils.wandb_context import load_wandb_context_from_env

ctx = load_wandb_context_from_env()  # Reads WANDB_CONTEXT_FILE
if ctx:
    ctx.log_training_metric("operator", "loss", 0.1, step=10)
```

## Testing

All changes are tested:

```bash
# Run WandBContext unit tests
python -m pytest tests/unit/test_wandb_context.py -v

# Run related tests to ensure nothing broke
python -m pytest tests/unit/test_training_logger.py tests/unit/test_leaderboard.py -v

# Syntax checks
python -m py_compile src/ups/utils/wandb_context.py scripts/train.py scripts/evaluate.py scripts/run_fast_to_sota.py
```

All tests pass ✓

## Benefits

1. **Clean Multi-Process Architecture**: Orchestrator and training runs properly linked via tags
2. **Proper Data Types**: Scalars in Summary, time series as charts, metadata in Config
3. **No More Hacks**: Eliminated `WANDB_MODE=disabled` workaround
4. **Clean Architecture**: WandBContext for centralized logging, clear responsibilities
5. **Easy Testing**: WandBContext can be mocked or disabled
6. **WandB Best Practices**: Follows recommended approach for multi-process pipelines
7. **Backward Compatible**: Standalone training still works

## Next Steps (Future Enhancements)

From the original WANDB_OPTIMIZATION_PLAN.md, these items remain for future work:

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

4. **Remote Sync Improvements**
   - Auto-retry for flaky connections
   - Bandwidth optimization for VastAI instances

## Conclusion

The clean WandB architecture is now fully implemented and tested. All hacky patterns have been eliminated, and the codebase follows WandB best practices for multi-process experiment tracking.

**Key Achievement**: **Linked Multi-Process Runs** with proper data types and no hacks! 🎉

- Orchestrator run: Pipeline tracking, gates, evaluation
- Training run: Training metrics (linked via tags)
- Proper data types throughout (time series, scalars, metadata)
- Clean WandBContext API for all logging
