---
date: 2025-11-13T14:45:59Z
researcher: Claude (via emerygunselman)
git_commit: 2595dca3b6a49adc35c18515fe81c0f73c5b5569
branch: feature/distributed-training-ddp
repository: universal_simulator
topic: "DDP Setup Crash Investigation: Multi-GPU Training Failures on Nov 13, 2025"
tags: [research, ddp, distributed-training, multi-gpu, crash-analysis, debugging]
status: complete
last_updated: 2025-11-13
last_updated_by: Claude
---

# Research: DDP Setup Crash Investigation

**Date**: 2025-11-13T14:45:59Z
**Researcher**: Claude (via emerygunselman)
**Git Commit**: `2595dca3b6a49adc35c18515fe81c0f73c5b5569`
**Branch**: `feature/distributed-training-ddp`
**Repository**: universal_simulator

## Research Question

Why are the new DDP (Distributed Data Parallel) setup runs crashing when training with 2 GPUs? All multi-GPU runs from November 13, 2025 are showing "crashed" state in WandB, with most having 0 history events.

## Summary

**Root Cause**: Multi-GPU DDP runs are failing during initialization, before any training metrics are logged. While the DDP implementation appears complete with all necessary safety fixes (CUDA IPC, WandB rank checking, distributed sampling), the runs are marked as "crashed" in WandB despite VastAI logs showing "Training pipeline completed."

**Key Findings**:
1. **All Nov 13 multi-GPU runs (2×GPU) crashed** - 9 runs attempted, all failed
2. **Most crashes happen immediately** - 7 out of 9 runs have 0 history events (crashed before first metric logged)
3. **Previous single-GPU runs (Nov 11) succeeded** - DDP code works in single-GPU mode
4. **DDP implementation appears sound** - All known safety fixes are in place (CPU dataset, RANK env var checking, distributed sampler)
5. **Suspicious pattern**: VastAI logs show "Training pipeline completed" but WandB shows "crashed"

**Likely Issues**:
1. **`torchrun` launch failure** - Multi-GPU launcher may be failing to spawn processes correctly
2. **NCCL backend initialization failure** - GPU communication setup may be failing
3. **Silent crash during DDP init** - Process may be crashing during `dist.init_process_group()` call
4. **WandB state mismatch** - Run marked as crashed even though some execution occurred

## Detailed Findings

### 1. Crash Pattern Analysis

#### Nov 13, 2025 Multi-GPU Runs (All Crashed)

| Run ID | Created (UTC) | GPU Count | Batch Size | History Events | Runtime | Crash Point |
|--------|---------------|-----------|------------|----------------|---------|-------------|
| train-20251113_012108 | 01:21:09 | 2 | 8 | 0 | N/A | Immediate |
| train-20251113_013449 | 01:34:49 | 2 | 8 | 2233 | 1768s | Step 3 |
| train-20251113_023511 | 02:35:13 | 2 | 4 | 374 | 224s | Step 0 |
| train-20251113_023512 | 02:35:13 | 2 | 4 | 0 | N/A | Immediate |
| train-20251113_024604 | 02:46:05 | 2 | 4 | 0 | N/A | Immediate |
| train-20251113_051243 | 05:12:44 | 2 | 4 | 0 | N/A | Immediate |
| train-20251113_055056 | 05:50:57 | 2 | 4 | 0 | N/A | Immediate |
| train-20251113_060416 | 06:04:18 | 2 | 4 | 0 | N/A | Immediate |
| train-20251113_061624 | 06:16:25 | 2 | 4 | 0 | N/A | Immediate |

**Patterns**:
- **7 out of 9** runs crashed immediately (0 history events)
- **2 out of 9** runs progressed to training (steps 0-3) before crashing
- **All** runs used 2-GPU configuration
- **Later runs** (after ~02:35) used updated config with batch_size=4 (OOM mitigation)
- **Earlier runs** used batch_size=8 (old config values)

#### Nov 11, 2025 Single-GPU Runs (Mixed Success/Crash)

| Run ID | Created (UTC) | GPU Count | State | History Events | Notes |
|--------|---------------|-----------|-------|----------------|-------|
| train-20251111_034102 | 03:41:03 | 1 | **finished** | 26,999 | ✅ Successful |
| train-20251111_033035 | 03:30:40 | N/A | crashed | 4 | Early crash |
| train-20251111_031848 | 03:18:49 | 1 | crashed | 36 | Step 4 |
| train-20251111_025348 | 02:53:52 | 1 | crashed | Large | Step 19 |
| train-20251111_000831 | 00:08:34 | 1 | crashed | Large | Step 19 |

**Key Observation**: At least one single-GPU run succeeded (train-20251111_034102), showing the codebase can complete training in single-GPU mode.

### 2. DDP Implementation Review

#### DDP Infrastructure is Complete

The DDP implementation (5-phase rollout on Nov 12, 2025) includes all necessary components:

**Phase 1: Core DDP** ([`scripts/train.py:58-84`](scripts/train.py#L58-L84))
- ✅ Automatic multi-GPU detection via `RANK` env var
- ✅ DDP initialization with NCCL backend
- ✅ Model wrapping with `DistributedDataParallel`
- ✅ Device placement per rank

**Phase 2: Distributed Sampling** ([`src/ups/data/latent_pairs.py:914-948`](src/ups/data/latent_pairs.py#L914-L948))
- ✅ `DistributedSampler` for single-task
- ✅ `MultiTaskDistributedSampler` for multi-task balanced sampling
- ✅ Epoch-based shuffling with `set_epoch()`

**Phase 3: CUDA IPC Safety** ([`src/ups/data/latent_pairs.py:767-771,821-824,333-363`](src/ups/data/latent_pairs.py))
- ✅ Dataset device forced to CPU (commit `2595dca`)
- ✅ Encoder kept on CPU, temp moved to GPU during encoding (commit `291041c`)
- ✅ Coordinates created on CPU
- ✅ Cached latents stored on CPU for `pin_memory` compatibility

**Phase 4: WandB Rank Checking** ([`scripts/train.py:2158-2165`](scripts/train.py#L2158-L2165))
- ✅ Check `RANK` env var (not `dist.is_initialized()`) for early detection (commit `dfcc418`)
- ✅ Only rank 0 initializes WandB
- ✅ Rank guards in all logging methods

**Phase 5: Production Hardening**
- ✅ OOM synchronization across ranks
- ✅ Checkpoint saving (rank 0 only)
- ✅ Metrics aggregation (`all_reduce`)
- ✅ Config validation for DDP

### 3. Configuration Analysis

**Config Used**: [`configs/train_pdebench_2task_baseline_ddp.yaml`](configs/train_pdebench_2task_baseline_ddp.yaml)

**Key Settings**:
```yaml
training:
  num_gpus: 2                    # 2×A100 = 160GB total
  batch_size: 4                  # Per-GPU (reduced from 8 for OOM mitigation)
  accum_steps: 12                # Effective batch = 4*12*2 = 96
  num_workers: 8                 # 8 workers per GPU
  use_parallel_encoding: true
  pin_memory: true
  prefetch_factor: 4
  cache_dir: data/latent_cache
  compile: false                 # torch.compile disabled for debugging

data:
  task: [advection1d, darcy2d]
  task_sampling:
    strategy: "balanced"         # MultiTaskDistributedSampler

logging:
  wandb:
    enabled: true
    project: universal-simulator
    entity: emgun-morpheus-space
    run_name: pdebench-2task-baseline-ddp
    tags: [distributed, ddp, 2gpu, pdebench, multi-task]
```

**Configuration appears correct** - all DDP-specific settings are properly configured.

### 4. VastAI Instance Analysis

**Instance ID**: 27823831
**Status**: exited
**GPUs**: 2× A100_SXM4
**Uptime**: N/A (exited)

**Log Excerpt** (final lines):
```
[validate-config] ✅ Config is valid and ready for training!
[validate-data] ✅ All data validation checks PASSED
[dry-run] ✅ Estimation complete (skipped validation)
[train] /opt/conda/bin/python3.11 scripts/train.py --config ...
wandb: Syncing run pdebench-2task-baseline-ddp
wandb: 🚀 View run at https://wandb.ai/emgun-morpheus-space/universal-simulator/runs/train-20251113_061624
✓ Training pipeline completed
stopping instance 27823831.
```

**Critical Observation**: VastAI logs show "Training pipeline completed" but WandB run `train-20251113_061624` shows as "crashed" with 0 history events.

### 5. Launch Mechanism Analysis

#### Expected Flow for Multi-GPU Launch

**File**: [`scripts/run_fast_to_sota.py:724-760`](scripts/run_fast_to_sota.py#L724-L760)

```python
# Check if already running under torchrun
if "RANK" in os.environ:
    # Already in distributed context, just call train.py directly
    train_cmd = [PYTHON, "scripts/train.py", ...]
else:
    # Single-GPU or need to launch torchrun
    num_gpus = cfg.get("training", {}).get("num_gpus", 1)
    if num_gpus > 1:
        train_cmd = [
            "torchrun",
            f"--nproc_per_node={num_gpus}",
            "--nnodes=1",
            "--node_rank=0",
            "--master_addr=localhost",
            "--master_port=29500",
            "scripts/train.py",
            ...
        ]
```

#### Potential Launch Issues

1. **`torchrun` not available or incorrect version**
   - VastAI instance may be using PyTorch 2.7.0 (from logs)
   - `torchrun` should be available in PyTorch 2.0+, but version compatibility issues possible

2. **NCCL backend initialization failure**
   - Multi-GPU communication setup may fail silently
   - NCCL requires proper CUDA driver and GPU connectivity
   - No error logs captured if failure occurs during `dist.init_process_group()`

3. **Port conflicts**
   - Default port 29500 may be in use
   - No fallback port mechanism

4. **Environment variable propagation**
   - `RANK`, `LOCAL_RANK`, `WORLD_SIZE` may not be set correctly by `torchrun`
   - Subprocess environment may not inherit parent variables

### 6. Comparison with Successful Single-GPU Run

**Successful Run**: `train-20251111_034102` (Nov 11, 03:41 UTC)
- **State**: finished
- **GPU count**: 1
- **History events**: 26,999 (full training)
- **Runtime**: 30,850s (~8.5 hours)
- **Config**: Likely `train_pdebench_2task_baseline.yaml` (single-GPU version)

**Key Difference**: Single-GPU mode **does not use `torchrun`** or DDP wrapping, avoiding all multi-GPU complexity.

## Code References

### DDP Initialization
- [`scripts/train.py:58-84`](scripts/train.py#L58-L84) - `setup_distributed()` function
- [`scripts/train.py:575`](scripts/train.py#L575) - Operator training DDP setup
- [`scripts/train.py:579-589`](scripts/train.py#L579-L589) - DDP model wrapping

### CUDA IPC Safety Fixes
- [`src/ups/data/latent_pairs.py:767-771`](src/ups/data/latent_pairs.py#L767-L771) - Force dataset device to CPU (commit `2595dca`)
- [`src/ups/data/latent_pairs.py:821-824`](src/ups/data/latent_pairs.py#L821-L824) - Encoder kept on CPU
- [`src/ups/data/latent_pairs.py:333-363`](src/ups/data/latent_pairs.py#L333-L363) - Temp GPU move during encoding

### WandB Rank Checking
- [`scripts/train.py:2158-2165`](scripts/train.py#L2158-L2165) - RANK env var check (commit `dfcc418`)
- [`src/ups/utils/wandb_context.py:68-82`](src/ups/utils/wandb_context.py#L68-L82) - Rank guards in logging

### Distributed Sampling
- [`src/ups/data/latent_pairs.py:914-948`](src/ups/data/latent_pairs.py#L914-L948) - DistributedSampler setup
- [`src/ups/data/task_samplers.py:13-131`](src/ups/data/task_samplers.py#L13-L131) - MultiTaskDistributedSampler

### Launch Orchestration
- [`scripts/run_fast_to_sota.py:724-760`](scripts/run_fast_to_sota.py#L724-L760) - torchrun launch logic
- [`scripts/vast_launch.py:292-308`](scripts/vast_launch.py#L292-L308) - VastAI onstart script generation

## Architecture Documentation

### DDP Initialization Sequence (Multi-GPU)

```
User launches VastAI instance with num_gpus=2
  ↓
onstart.sh generated with: torchrun --nproc_per_node=2 scripts/run_fast_to_sota.py
  ↓
torchrun spawns 2 processes, sets RANK={0,1}, LOCAL_RANK={0,1}, WORLD_SIZE=2
  ↓
Each process: run_fast_to_sota.py detects RANK env var → skips nested torchrun
  ↓
Each process: Calls scripts/train.py directly
  ↓
Each process: train_all_stages() → WandB init (RANK env var check)
  ├─ Rank 0: wandb.init() → Creates WandB run
  └─ Rank 1: Skipped (rank != 0)
  ↓
Each process: train_operator() called
  ↓
Each process: setup_distributed() called
  ├─ Checks RANK env var → Detects DDP mode
  ├─ Calls dist.init_process_group(backend="nccl")  ← POTENTIAL FAILURE POINT
  ├─ Sets device to cuda:{local_rank}
  └─ Returns (device, is_distributed=True, rank, world_size, local_rank)
  ↓
Each process: Model wrapped with DDP
  ↓
Each process: Training loop begins
  ↓
Rank 0: Logs metrics to WandB
Rank 1-N: Skip logging (rank guards)
```

**Critical Point**: The crash appears to occur **before** the first training metric is logged, suggesting failure during:
1. `torchrun` process spawning
2. `dist.init_process_group()` call
3. Model wrapping with DDP
4. DataLoader initialization with DistributedSampler

### VastAI "Training pipeline completed" Anomaly

**Observation**: VastAI logs show `✓ Training pipeline completed` but WandB shows "crashed" with 0 events.

**Possible Explanations**:
1. **Success message is premature** - Printed before actual training starts
2. **Silent crash after message** - Process crashes after printing success but before WandB finalization
3. **WandB finalization failure** - Run not properly marked as finished, defaults to "crashed"
4. **Exit code 0 despite crash** - Script exits cleanly but WandB detects abnormal termination

## Historical Context (from thoughts/)

### DDP Implementation Timeline (Nov 12, 2025)

**Source**: [`thoughts/shared/plans/2025-11-12-distributed-training-multi-task.md`](thoughts/shared/plans/2025-11-12-distributed-training-multi-task.md)

**12:30 PM** - Multi-task conservation penalty crash fix
**2:01 PM** - Phase 1: Core DDP infrastructure
**2:04 PM** - Phase 2: Task-aware distributed sampling
**2:14 PM** - Phase 3: Memory optimization
**4:14 PM** - Phase 4: Cloud provider integration
**5:05 PM** - Phase 5: Production hardening
**9:05 PM** - **First CUDA IPC fix** (encoder to CPU) - commit `291041c`
**9:43 PM** - **WandB rank check fix** (use RANK env var) - commit `dfcc418`
**10:09 PM** - **Final DDP safety fix** (force dataset device to CPU) - commit `2595dca`

**Key Insight**: All DDP work happened in a single day (Nov 12), with 3 rounds of debugging for CUDA IPC and WandB issues. The implementation is **very recent** and may have **untested edge cases**.

### Known Issues (from historical commits)

**Issue 1: "Producer process terminated" / CUDA IPC Conflicts**
- **Fixed**: Commits `291041c`, `2595dca`
- **Solution**: Keep dataset on CPU, temp move encoder to GPU during encoding

**Issue 2: WandB Duplicate Run Errors (409 Conflict)**
- **Fixed**: Commit `dfcc418`
- **Solution**: Check `RANK` env var instead of `dist.is_initialized()`

**Issue 3: Multi-Task Conservation Penalty Crash**
- **Fixed**: Commit `db34094`
- **Solution**: Disable conservation penalty (`lambda_conservation: 0.0`)

**Issue 4: Out of Memory (OOM)**
- **Mitigated**: Reduced batch_size from 8 to 4
- **Status**: Config updated, but some runs still used old values

## Potential Root Causes (Prioritized)

### 1. `torchrun` Launch Failure (HIGH PROBABILITY)

**Evidence**:
- 7 out of 9 runs crashed immediately (0 history events)
- No training metrics logged at all
- VastAI logs show success message but WandB shows crashed

**Hypothesis**: `torchrun` is failing to spawn processes correctly, or spawned processes are crashing during initialization.

**Debugging Steps**:
1. Check if `torchrun` is available: `which torchrun`
2. Verify PyTorch version supports multi-GPU: `python -c "import torch; print(torch.__version__)"`
3. Test basic torchrun: `torchrun --nproc_per_node=2 --nnodes=1 python -c "import torch.distributed as dist; dist.init_process_group('nccl'); print(f'Rank {dist.get_rank()}')"`
4. Add verbose logging: `torchrun --log_dir=logs/torchrun ...`

### 2. NCCL Backend Initialization Failure (HIGH PROBABILITY)

**Evidence**:
- Crashes occur during DDP init phase (before metrics)
- NCCL requires specific CUDA/driver setup
- No error logs captured in VastAI output

**Hypothesis**: `dist.init_process_group(backend="nccl")` is failing silently due to:
- Incompatible CUDA/NCCL versions
- GPU communication setup issues
- Missing NCCL environment variables

**Debugging Steps**:
1. Check NCCL version: `python -c "import torch; print(torch.cuda.nccl.version())"`
2. Test NCCL communication: Run simple 2-GPU NCCL test
3. Check NCCL debug output: `export NCCL_DEBUG=INFO`
4. Try fallback backend: Change to `backend="gloo"` (CPU-based, slower but more compatible)

### 3. Environment Variable Propagation Issues (MEDIUM PROBABILITY)

**Evidence**:
- `RANK` env var used for early detection
- `torchrun` sets these variables, but subprocess may not inherit them

**Hypothesis**: Environment variables not properly propagated to subprocesses.

**Debugging Steps**:
1. Add logging in `train.py` main: `print(f"RANK={os.environ.get('RANK')}, LOCAL_RANK={os.environ.get('LOCAL_RANK')}")`
2. Verify variables in VastAI logs
3. Check subprocess creation in `run_fast_to_sota.py`

### 4. Port Conflicts (LOW PROBABILITY)

**Evidence**:
- Default port 29500 hardcoded
- Multiple runs attempted in short succession

**Hypothesis**: Previous runs may hold port, causing initialization failure.

**Debugging Steps**:
1. Check if port is in use: `netstat -tuln | grep 29500`
2. Add random port selection
3. Increase port: `--master_port=29600`

### 5. Silent Exception in DDP Init (MEDIUM PROBABILITY)

**Evidence**:
- No error logs in VastAI output
- WandB shows crashed but no details

**Hypothesis**: Exception during DDP setup is not being logged or caught properly.

**Debugging Steps**:
1. Add try-except around `dist.init_process_group()`:
   ```python
   try:
       dist.init_process_group(backend="nccl")
   except Exception as e:
       print(f"[RANK {rank}] DDP init failed: {e}")
       import traceback
       traceback.print_exc()
       raise
   ```
2. Add logging before/after critical operations
3. Check stderr output (may be separate from stdout in VastAI)

### 6. WandB State Confusion (LOW PROBABILITY)

**Evidence**:
- VastAI shows "completed" but WandB shows "crashed"
- 0 history events suggests no `wandb.log()` calls

**Hypothesis**: WandB is marking run as crashed due to missing finalization call.

**Debugging Steps**:
1. Add explicit `wandb.finish()` call at end
2. Check if `atexit` handlers are running
3. Verify WandB API key and connectivity

## Next Steps for Debugging

### Immediate Actions (High Priority)

1. **Add Comprehensive Logging**
   - Log `RANK`, `LOCAL_RANK`, `WORLD_SIZE` at script start
   - Log before/after `dist.init_process_group()`
   - Log before/after DDP model wrapping
   - Log before/after DataLoader creation

2. **Enable NCCL Debug Output**
   - Add `export NCCL_DEBUG=INFO` to VastAI onstart script
   - Check for NCCL errors in logs

3. **Test Basic DDP Functionality**
   - Create minimal test script (10 lines) that just initializes DDP
   - Run with `torchrun --nproc_per_node=2`
   - Verify both ranks print "Hello from rank X"

4. **Try Gloo Backend**
   - Change `backend="nccl"` to `backend="gloo"` temporarily
   - If successful, narrows down to NCCL-specific issue

### Medium Priority

5. **Check VastAI GPU Configuration**
   - Verify both GPUs are visible: `nvidia-smi`
   - Check CUDA version: `nvcc --version`
   - Check NCCL installation: `ldconfig -p | grep nccl`

6. **Review VastAI Startup Script**
   - Examine `.vast/onstart.sh` for correct `torchrun` invocation
   - Verify all environment variables are exported

7. **Test with Single-GPU DDP**
   - Run DDP with `num_gpus=1` (should behave like non-DDP)
   - Isolates multi-GPU-specific issues

### Long-term Fixes

8. **Improve Error Handling**
   - Add try-except blocks around all DDP operations
   - Implement proper error propagation to WandB
   - Add health checks before expensive operations

9. **Add DDP Validation Script**
   - Create `scripts/validate_ddp.py` to test multi-GPU setup
   - Run before expensive training jobs

10. **Implement Gradual Rollout**
    - Test 2-GPU locally before VastAI
    - Add `--dry-run-ddp` flag to test without actual training
    - Implement DDP smoke test (1 epoch, small batch)

## Related Research

**Previous DDP Documentation**:
- [`thoughts/shared/research/2025-11-12-distributed-training-analysis.md`](thoughts/shared/research/2025-11-12-distributed-training-analysis.md) - Strategic assessment and implementation roadmap
- [`thoughts/shared/plans/2025-11-12-distributed-training-multi-task.md`](thoughts/shared/plans/2025-11-12-distributed-training-multi-task.md) - Full 5-phase implementation plan
- [`thoughts/shared/research/2025-11-12-data-flow-cache-requirements.md`](thoughts/shared/research/2025-11-12-data-flow-cache-requirements.md) - Latent cache + DDP interactions

**Related Commits**:
- `2595dca` - Force dataset device to CPU for DDP safety
- `dfcc418` - Fix WandB rank check: use RANK env var
- `291041c` - Fix DDP + latent cache CUDA IPC conflicts
- `a7a3c4a` - Phase 5: Production hardening and testing
- `7d0a009` - Phase 1: Core DDP Infrastructure Implementation

## Open Questions

1. **Why do VastAI logs show "Training pipeline completed" but WandB shows "crashed"?**
   - Is the success message printed prematurely?
   - Is there a crash after the message but before WandB finalization?
   - Is WandB incorrectly marking runs as crashed?

2. **Why are 7 out of 9 runs crashing immediately (0 history events)?**
   - Is `torchrun` failing to spawn processes?
   - Is `dist.init_process_group()` failing silently?
   - Is there an exception during imports that's not being logged?

3. **Why did 2 out of 9 runs progress to training (steps 0-3)?**
   - What was different about these runs?
   - Were they using different configurations?
   - Did they hit a later failure point (OOM, data loading issue)?

4. **Is the DDP implementation actually being used?**
   - Are the `RANK` environment variables being set correctly by `torchrun`?
   - Is the code path going through DDP or falling back to single-GPU?

5. **Are there VastAI-specific issues?**
   - Does multi-GPU work locally but fail on VastAI?
   - Are there networking or GPU communication restrictions on VastAI?

## Recommendations

### For Immediate Debugging

1. **Add verbose logging** to `scripts/train.py` at every critical point
2. **Enable NCCL debug output** with `export NCCL_DEBUG=INFO`
3. **Create minimal DDP test** to verify basic multi-GPU functionality
4. **Try Gloo backend** as fallback to isolate NCCL issues
5. **Check VastAI GPU visibility** with `nvidia-smi` and NCCL tests

### For Longer-term Robustness

1. **Implement comprehensive error handling** around all DDP operations
2. **Create DDP validation script** to test setup before expensive runs
3. **Add health checks** for GPU availability and communication
4. **Test locally** before deploying to VastAI
5. **Consider gradual rollout**: Test 2-GPU locally → Test 2-GPU on VastAI → Scale to 4-GPU

### Alternative Approaches

1. **Focus on single-GPU optimization** first (latent cache, batch size)
   - According to research, single-GPU training is only ~25 min
   - DDP may not provide significant value for this workload
2. **Use parallel single-GPU instances** for hyperparameter sweeps instead of DDP
3. **Reserve DDP for 11-task PDEBench** where memory capacity is critical

---

**Investigation Status**: Complete
**Next Action**: Implement comprehensive logging and run minimal DDP test to isolate failure point.
