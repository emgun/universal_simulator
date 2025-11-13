# DDP Debugging Improvements

**Date**: 2025-11-13
**Status**: Complete
**Git Branch**: feature/distributed-training-ddp

## Overview

Implemented comprehensive debugging improvements to systematically diagnose and resolve DDP (Distributed Data Parallel) crash issues identified in the research document at `thoughts/shared/research/2025-11-13-ddp-crash-investigation.md`.

## Problem Summary

All 9 multi-GPU (2×GPU) DDP training runs from November 13, 2025 crashed:
- **7 out of 9 runs**: Crashed immediately (0 WandB history events)
- **2 out of 9 runs**: Progressed to steps 0-3 before crashing
- **Anomaly**: VastAI logs showed "Training pipeline completed" but WandB marked runs as "crashed"

## Debugging Improvements Implemented

### 1. Enhanced DDP Initialization Logging (`scripts/train.py`)

**Location**: `scripts/train.py:58-143`

**Changes**:
- Added comprehensive logging at every critical step of DDP initialization
- Log all environment variables (`RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT`)
- Log PyTorch/CUDA/NCCL version information
- Log GPU availability and device count
- Log before/after `dist.init_process_group()` call
- Log before/after CUDA device assignment
- Added detailed error logging with full tracebacks

**Example Output**:
```
[DDP-DEBUG] setup_distributed() called
[DDP-DEBUG] RANK=0
[DDP-DEBUG] LOCAL_RANK=0
[DDP-DEBUG] WORLD_SIZE=2
[DDP-DEBUG] MASTER_ADDR=localhost
[DDP-DEBUG] MASTER_PORT=29500
[DDP-DEBUG] Checking GPU availability...
[DDP-DEBUG] torch.cuda.is_available() = True
[DDP-DEBUG] torch.cuda.device_count() = 2
[DDP-DEBUG] CUDA version: 12.1
[DDP-DEBUG] NCCL available: True
[DDP-DEBUG] NCCL version: (2, 18, 3)
[DDP-DEBUG] Attempting dist.init_process_group(backend='nccl')...
[DDP-DEBUG] dist.init_process_group(backend='nccl') completed successfully
```

### 2. Error Handling with Try-Except Blocks (`scripts/train.py`)

**Location**: `scripts/train.py:79-141`

**Changes**:
- Wrapped environment variable parsing in try-except
- Wrapped GPU availability checks in try-except
- Wrapped `dist.init_process_group()` in try-except with detailed error reporting
- Wrapped CUDA device assignment in try-except
- Added traceback printing for all exceptions
- Added NCCL/CUDA environment variable dump on errors

**Benefits**:
- Prevents silent crashes
- Provides actionable error information
- Helps identify exactly which operation is failing

### 3. Gloo Backend Fallback Option (`scripts/train.py`)

**Location**: `scripts/train.py:102-141`

**Changes**:
- Implemented automatic fallback from NCCL to Gloo if NCCL fails
- Added `DDP_BACKEND` environment variable to override default backend
- Gloo is CPU-based and slower but more compatible (useful for debugging)
- Warns user when fallback is used

**Usage**:
```bash
# Force Gloo backend (for debugging NCCL issues)
export DDP_BACKEND=gloo
python scripts/train.py --config ...
```

**Example Output (on fallback)**:
```
[DDP-ERROR] dist.init_process_group(backend='nccl') failed: ...
[DDP-WARNING] NCCL failed, trying Gloo fallback...
[DDP-DEBUG] Attempting dist.init_process_group(backend='gloo')...
[DDP-DEBUG] dist.init_process_group(backend='gloo') completed successfully
[DDP-WARNING] Fell back to Gloo backend (NCCL failed)
[DDP-WARNING] Gloo is CPU-based and slower - for debugging only!
```

### 4. Minimal DDP Test Script (`scripts/test_ddp_minimal.py`)

**Location**: `scripts/test_ddp_minimal.py` (new file)

**Purpose**: Isolate DDP initialization issues from training complexity

**Features**:
- Tests only DDP core functionality (no training code)
- Comprehensive logging at each step
- Tests process group initialization
- Tests CUDA device assignment
- Tests all_reduce communication
- Works with single-process (for baseline) or multi-GPU

**Usage**:
```bash
# Single-process baseline test
python scripts/test_ddp_minimal.py

# Multi-GPU test with torchrun
torchrun --nproc_per_node=2 --nnodes=1 \
  --master_addr=localhost --master_port=29500 \
  scripts/test_ddp_minimal.py
```

**Expected Output (success)**:
```
================================================================================
MINIMAL DDP TEST SCRIPT
================================================================================

[STEP 1] Environment Variables:
  RANK = 0
  LOCAL_RANK = 0
  WORLD_SIZE = 2
  ...

[STEP 2] PyTorch and CUDA Info:
  PyTorch version: 2.3.1
  CUDA available: True
  ...

[STEP 3] Initializing DDP...
  [Rank 0] Calling dist.init_process_group(backend='nccl')...
  [Rank 0] init_process_group() succeeded!
  ...

[STEP 4] Setting CUDA device:
  [Rank 0] Setting device to cuda:0
  [Rank 0] Device set successfully

[STEP 5] Testing DDP communication:
  [Rank 0] Before all_reduce: tensor=0.0
  [Rank 0] After all_reduce: tensor=1.0, expected=1
  [Rank 0] All-reduce test PASSED

[STEP 6] Synchronizing and cleaning up:
  [Rank 0] Barrier synchronized
  [Rank 0] Process group destroyed

================================================================================
[SUCCESS] Rank 0 completed all tests successfully!
================================================================================
```

### 5. NCCL Debug Environment Variables (`scripts/vast_launch.py`)

**Location**: `scripts/vast_launch.py:298-301`

**Changes**:
- Added `NCCL_DEBUG=INFO` for verbose NCCL output
- Added `NCCL_DEBUG_SUBSYS=ALL` for all subsystem debugging
- Added `TORCH_DISTRIBUTED_DEBUG=DETAIL` for PyTorch distributed debugging

**Environment Variables Set**:
```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export TORCH_DISTRIBUTED_DEBUG=DETAIL
```

**Benefits**:
- Reveals NCCL communication setup issues
- Shows GPU-to-GPU communication failures
- Helps diagnose CUDA driver/hardware issues

### 6. Fixed VastAI Error Handling (`scripts/vast_launch.py`)

**Location**: `scripts/vast_launch.py:306-345`

**Previous Behavior**:
```bash
torchrun ... || echo "⚠️  Training exited with code $?"
echo "✓ Training pipeline completed"
```
- Masked failures with `|| echo`
- Always printed "completed" even on failure
- VastAI couldn't detect training failures

**New Behavior**:
```bash
set +e  # Temporarily disable exit on error
torchrun ...
TRAIN_EXIT_CODE=$?
set -e  # Re-enable exit on error

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
  echo "✓ Training pipeline completed successfully"
else
  echo "✗ Training pipeline failed with exit code $TRAIN_EXIT_CODE"
  echo "Check logs above for error details"
  exit $TRAIN_EXIT_CODE
fi
```

**Benefits**:
- Properly captures and reports training exit codes
- VastAI can detect failures
- WandB state matches actual training outcome
- Clear success/failure messaging

## Testing Strategy

### Phase 1: Local Minimal Test

Test DDP initialization in isolation:

```bash
# Run minimal DDP test locally (if you have multi-GPU)
torchrun --nproc_per_node=2 scripts/test_ddp_minimal.py
```

**Expected**: All steps complete successfully, both ranks communicate

### Phase 2: VastAI Minimal Test

Test DDP on VastAI infrastructure:

```bash
# Launch VastAI instance with minimal test
python scripts/vast_launch.py launch \
  --config configs/train_pdebench_2task_baseline_ddp.yaml \
  --num-gpus 2 \
  --auto-shutdown \
  --dry-run
```

Then manually SSH and run:
```bash
cd /workspace/universal_simulator
torchrun --nproc_per_node=2 scripts/test_ddp_minimal.py
```

**Expected**: Detailed logs revealing the exact failure point

### Phase 3: Full Training Test

Once minimal test passes, launch full training:

```bash
python scripts/vast_launch.py launch \
  --config configs/train_pdebench_2task_baseline_ddp.yaml \
  --auto-shutdown
```

**Monitor logs** for:
- `[DDP-DEBUG]` messages showing initialization progress
- `NCCL_DEBUG` messages showing GPU communication
- Proper error messages if failure occurs
- Correct exit code and success/failure message

## Key Files Modified

1. **`scripts/train.py`**
   - Enhanced `setup_distributed()` with comprehensive logging
   - Added error handling and Gloo fallback

2. **`scripts/vast_launch.py`**
   - Added NCCL debug environment variables
   - Fixed error handling and exit code propagation

3. **`scripts/test_ddp_minimal.py`** (new)
   - Minimal DDP test script for isolation

## Diagnostic Workflow

If DDP crashes still occur after these changes:

1. **Check VastAI logs** for `[DDP-DEBUG]` and `[DDP-ERROR]` messages
2. **Identify failure point**:
   - Before `dist.init_process_group()`? → Environment variable issue
   - During `dist.init_process_group()`? → NCCL/GPU communication issue
   - After `dist.init_process_group()`? → CUDA device assignment issue
3. **Check NCCL debug output** for specific errors
4. **Try Gloo fallback**: Add `export DDP_BACKEND=gloo` to onstart script
5. **Run minimal test** to isolate from training complexity

## Expected Log Patterns

### Success Pattern

```
[DDP-DEBUG] setup_distributed() called
[DDP-DEBUG] RANK=0
[DDP-DEBUG] Parsed env vars: rank=0, local_rank=0, world_size=2
[DDP-DEBUG] Attempting dist.init_process_group(backend='nccl')...
[DDP-DEBUG] dist.init_process_group(backend='nccl') completed successfully
[DDP-DEBUG] Setting CUDA device to 0...
[DDP-DEBUG] Device set to cuda:0
[DDP-INFO] Distributed training initialized: 2 GPUs
[DDP-DEBUG] setup_distributed() returning successfully for rank 0
```

### Failure Pattern (example: NCCL init failure)

```
[DDP-DEBUG] setup_distributed() called
[DDP-DEBUG] RANK=0
[DDP-DEBUG] Parsed env vars: rank=0, local_rank=0, world_size=2
[DDP-DEBUG] Attempting dist.init_process_group(backend='nccl')...
[DDP-ERROR] dist.init_process_group(backend='nccl') failed: <error details>
[DDP-ERROR] Exception type: RuntimeError
<full traceback>
[DDP-DEBUG] Printing NCCL/CUDA environment variables:
  NCCL_DEBUG=INFO
  CUDA_VISIBLE_DEVICES=...
[DDP-WARNING] NCCL failed, trying Gloo fallback...
[DDP-DEBUG] Attempting dist.init_process_group(backend='gloo')...
```

## Next Steps

1. **Deploy changes** to VastAI with next multi-GPU run
2. **Monitor logs** for comprehensive debug output
3. **Analyze failure point** if crashes persist
4. **Share logs** with team for collaborative debugging
5. **Update research doc** with findings from new debug output

## Related Documentation

- **Investigation Report**: `thoughts/shared/research/2025-11-13-ddp-crash-investigation.md`
- **DDP Implementation**: `thoughts/shared/plans/2025-11-12-distributed-training-multi-task.md`
- **Config Used**: `configs/train_pdebench_2task_baseline_ddp.yaml`

## Summary

These debugging improvements transform DDP crash investigation from "silent failure with no logs" to "comprehensive diagnostic output at every step". The enhanced logging, error handling, fallback options, and proper exit code propagation will enable rapid identification of the root cause.

**Status**: Ready for deployment to VastAI for testing.
