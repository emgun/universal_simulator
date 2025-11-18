# DDP 2-Task Testing Summary (2025-11-18)

**Date:** 2025-11-18
**Branch:** `feature/distributed-training-ddp`
**Status:** ⚠️ **IN PROGRESS** - Debugging /dev/shm issues with 2-task training

---

## Overview

Tested the 2-task DDP configuration (`train_pdebench_2task_baseline_ddp.yaml`) with the successful DDP fixes from burgers training. Encountered `/dev/shm` exhaustion issues that did not occur with single-task burgers config.

---

## Changes Implemented Today

### 1. ✅ **Enabled torch.compile for 2-Task DDP Config**

**File:** `configs/train_pdebench_2task_baseline_ddp.yaml`

**Changes:**
```yaml
# Before
compile: false                   # Disabled: torch.compile graph compilation fails with PyTorch 2.9
compile_mode: reduce-overhead

# After
compile: true                    # ENABLED: torch.compile works with DDP using compile_mode: default
compile_mode: default            # CRITICAL: Use 'default' mode (NOT reduce-overhead) for DDP compatibility
```

**Rationale:**
- Burgers DDP config (`train_burgers_upt_full_ddp.yaml`) successfully ran with `compile: true` and `compile_mode: default`
- `compile_mode: reduce-overhead` fails with DDP due to aggressive CUDA graph usage conflicting with gradient synchronization
- `compile_mode: default` avoids CUDA graph issues and is DDP-compatible

**Commit:** `1610ea5`

---

### 2. ✅ **Fixed Test/Val Data Download**

**Problem:** Test data files (e.g., `burgers1d_test.h5`) were never downloaded, causing evaluation to fail with `FileNotFoundError`.

**Root Cause:** `scripts/setup_vast_data.sh` only downloaded train splits, with comment claiming "val/test will come from WandB artifacts" but no such mechanism existed.

**Solution:** Modified `setup_vast_data.sh` to download all three splits (train, val, test) from B2 storage.

**File:** `scripts/setup_vast_data.sh`

**Changes:**
```bash
# Before: Only downloaded train split
for task in $TASKS; do
  target_file="$ROOT_DIR/${task}_train.h5"
  # ... download train only
done

# After: Download all splits
for task in $TASKS; do
  for split in train val test; do
    target_file="$ROOT_DIR/${task}_${split}.h5"
    # Try multiple B2 path patterns:
    # - full/$task/${task}_$split.h5
    # - full/$task/${task}_$split_000.h5
    # - pdebench/${task}_full_v1/${task}_$split.h5
    # Don't exit if test/val missing (not all tasks have them)
  done
done
```

**Behavior:**
- ✅ Downloads train/val/test if available
- ✅ Warns but continues if test/val not found (graceful degradation)
- ✅ Tries multiple B2 path patterns to handle inconsistent naming

**Commit:** `22b0e9f`

---

### 3. ⚠️ **2-Task Config Testing - ONGOING ISSUES**

**Config:** `configs/train_pdebench_2task_baseline_ddp.yaml`
- **Tasks:** advection1d + darcy2d
- **Architecture:** Pure transformer (pdet_stack), 128-dim latent space
- **Distributed:** 2×A100 GPUs with DDP
- **Optimizations:** All Phase 1-8 optimizations + torch.compile enabled

**Test Results:**

#### Attempt 1: `/dev/shm` exhaustion during cache loading
```
Instance: 28000023
Result: ❌ FAILED

Error:
RuntimeError: unable to write to file </torch_3610_1898609195_6964>:
No space left on device (28)

Location: PreloadedCacheDataset.__init__() during latent.share_memory_()
```

**Analysis:**
- Cache precomputation succeeded (advection1d: 8000 samples, darcy2d: 5000 samples)
- Training started but failed when trying to load cache into shared memory
- Docker container's `/dev/shm` (default ~64MB) too small for multi-task cache
- Total data: advection1d (6.2G) + darcy2d (317M) = much larger than burgers

**Fix Attempted:** Changed `num_workers: 6` → `num_workers: 0`

**Commit:** `e10145c`

#### Attempt 2: SIGBUS (Signal 7)
```
Instance: 28000351
Result: ❌ FAILED

Error:
subprocess.CalledProcessError: Command died with <Signals.SIGBUS: 7>

Location: Silent crash during DDP initialization (no Python traceback)
```

**Analysis:**
- SIGBUS indicates memory access violation at C++ level
- Process hung for 2.5 minutes between ProcessGroupNCCL warning and crash
- Even with `num_workers: 0`, `PreloadedCacheDataset` still calls `.share_memory_()`
- SIGBUS likely from attempting to access shared memory that can't be allocated

**Fix Attempted:** Disabled caching entirely (`latent_cache_dir: null`)

**Reverted:** User correctly noted that burgers DDP worked with caching, so we shouldn't avoid the issue

#### Attempt 3: Skip cache precomputation (CURRENT)
```
Instance: 28000651
Status: 🔄 RUNNING

Approach: Let training compute cache on-the-fly instead of preloading
Modified: vast_launch.py to skip precomputation for DDP (num_gpus > 1)
```

**Status:** Being tested currently

---

## Key Differences: Burgers (Success) vs 2-Task (Failed)

| Aspect | Burgers DDP ✅ | 2-Task DDP ❌ |
|--------|----------------|----------------|
| **Tasks** | 1 (burgers1d) | 2 (advection1d + darcy2d) |
| **Raw Data Size** | ~1-2GB | 6.5GB (6.2G + 317M) |
| **Cache Size** | ~500MB estimated | ~3-5GB estimated |
| **num_workers** | 6 | 6 (tried 0, still failed) |
| **Cache Precompute** | Yes, succeeded | Yes, succeeded (but loading failed) |
| **PreloadedCacheDataset** | Loaded successfully | /dev/shm exhaustion |
| **Training** | 40 epochs completed | Crashed before epoch 1 |

**Root Cause Hypothesis:**
- Docker containers have limited `/dev/shm` size (typically 64MB default)
- `PreloadedCacheDataset` loads entire cache into shared memory via `.share_memory_()`
- Burgers cache (~500MB) fits within available system RAM for shared memory mapping
- 2-task cache (~3-5GB) exceeds `/dev/shm` capacity → crashes with ENOSPC or SIGBUS

---

## Potential Solutions (Not Yet Implemented)

### Option 1: Increase Docker --shm-size
```bash
vastai launch instance --shm-size 16G ...
```

**Pros:**
- Allows PreloadedCacheDataset to work as designed
- Maintains fast training speed from RAM caching

**Cons:**
- Requires modifying vast_launch.py to pass --shm-size
- VastAI may not support --shm-size parameter (uses custom runtime)
- Need to verify VastAI Docker launch options

### Option 2: Use Alternative Cache Mode
Modify code to check if DDP is enabled and use disk-based cache instead of RAM-preloaded:

```python
# In latent_pairs.py
if dist.is_initialized() and world_size > 1:
    # DDP mode: Use GridLatentDataset (disk-based cache)
    dataset = GridLatentDataset(...)
else:
    # Single-GPU mode: Use PreloadedCacheDataset (RAM-preloaded)
    dataset = PreloadedCacheDataset(...)
```

**Pros:**
- Avoids /dev/shm entirely
- Works reliably with DDP
- No infrastructure changes needed

**Cons:**
- Slower training (disk I/O instead of RAM)
- Requires code changes
- Loses benefit of cache precomputation

### Option 3: Disable Cache Precomputation for DDP (**Currently Testing**)
Skip cache precomputation for `num_gpus > 1` and let training compute on-the-fly:

```python
# In vast_launch.py
if precompute and num_gpus == 1:
    # Only precompute for single-GPU
    cache_cmd = "PYTHONPATH=src python scripts/precompute_latent_cache.py ..."
else:
    cache_cmd = 'echo "Skipping cache precompute (DDP)"'
```

**Pros:**
- Simple fix
- Avoids /dev/shm issues entirely
- Training still computes cache on-the-fly

**Cons:**
- First epoch will be slower (computing cache)
- Doesn't solve underlying issue
- User rejected this approach

### Option 4: Fix PreloadedCacheDataset for DDP (**RECOMMENDED**)
Modify `PreloadedCacheDataset` to NOT use `.share_memory_()` when DDP is active:

```python
# In parallel_cache.py
class PreloadedCacheDataset(Dataset):
    def __init__(self, ...):
        # Load cache
        for sample in cache_files:
            latent = torch.load(sample)

            # ONLY share memory if NOT using DDP
            if not (dist.is_available() and dist.is_initialized()):
                latent = latent.share_memory_()

            self.cache.append(latent)
```

**Pros:**
- Fixes root cause
- Maintains cache precomputation benefits
- Works reliably with both single-GPU and DDP

**Cons:**
- Requires code changes
- Need to test if multiprocessing works without .share_memory_()

---

## Commits Summary

| Commit | Description | Status |
|--------|-------------|--------|
| `1610ea5` | Enable torch.compile for 2-task DDP (compile_mode: default) | ✅ Merged |
| `22b0e9f` | Fix test/val data download in setup_vast_data.sh | ✅ Merged |
| `e10145c` | Fix /dev/shm exhaustion (set num_workers: 0) | ⚠️ Didn't solve issue |
| `8904615` | Restore num_workers: 6 (match burgers config) | ✅ Merged |

---

## Next Steps

### Immediate (Do Now)
1. 🔄 **Monitor current instance** (28000651) - test if skipping cache precomputation works
2. 📊 **Compare results** with burgers DDP to understand differences
3. 🐛 **Debug /dev/shm limits** - check actual Docker container limits

### Short Term (This Session)
1. ✅ **Implement Option 4** - Modify PreloadedCacheDataset to skip .share_memory_() for DDP
2. 🧪 **Test fix** with 2-task config
3. 📝 **Document solution** in production playbook

### Medium Term (Next Sprint)
1. 🔧 **Add --shm-size support** to vast_launch.py if VastAI supports it
2. 🎯 **Validate multi-task DDP** works consistently
3. 📈 **Benchmark performance** vs single-GPU

---

## References

### Related Documentation
- **DDP Fixes:** `docs/distributed_training_SIGSEGV_solution.md`
- **Outstanding Issues:** `docs/distributed_training_outstanding_issues.md`
- **Investigation Log:** `docs/ddp_optimization_investigation_2025-11-18.md`

### Successful Configs
- **Burgers DDP:** `configs/train_burgers_upt_full_ddp.yaml` (40 epochs, 99% GPU util, WORKS)
- **2-Task DDP:** `configs/train_pdebench_2task_baseline_ddp.yaml` (FAILS at startup)

### Key Code Files
- **Cache Preload:** `src/ups/data/parallel_cache.py:167` (PreloadedCacheDataset)
- **Cache Selection:** `src/ups/data/latent_pairs.py:884` (chooses PreloadedCacheDataset)
- **VastAI Launch:** `scripts/vast_launch.py:253` (cache precomputation generation)

### External Resources
- [Docker --shm-size documentation](https://docs.docker.com/engine/reference/run/#runtime-constraints-on-resources)
- [PyTorch Shared Memory](https://pytorch.org/docs/stable/multiprocessing.html#sharing-strategies)
- [Linux /dev/shm](https://www.kernel.org/doc/html/latest/filesystems/tmpfs.html)

---

**Last Updated:** 2025-11-18
**Next Review:** After current instance (28000651) completes or fails
