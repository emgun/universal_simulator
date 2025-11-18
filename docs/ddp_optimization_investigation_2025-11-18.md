# DDP Training Optimization Investigation - 2025-11-18

## Executive Summary

Investigation into 2-GPU DDP training performance bottlenecks for UPT Phase 4 Burgers 1D model. Initial DDP runs showed no speedup (60s/epoch on 2 GPUs vs expected 15-20s). Root cause identified as I/O bottleneck from `num_workers=0` requirement in PreloadedCacheDataset. Implemented shared memory fix to enable multi-worker data loading, but encountered silent crashes during training initialization.

**Current Status:** Debugging shared memory implementation crashes.

---

## Timeline of Investigations

### Investigation 1: Initial DDP Run (Unoptimized)
**Instance:** 27968251
**Config:** `train_burgers_upt_full_ddp.yaml`
**Parameters:**
- batch_size: 10 per GPU
- accum_steps: 2
- num_queries: 2048
- num_workers: 0 (forced by PreloadedCacheDataset)

**Results:**
- Epoch time: **60s/epoch**
- GPU utilization: Fluctuating 30-89%
- Memory: 50.8GB per GPU
- **No DDP speedup observed**

**Diagnosis:** I/O bound due to single-threaded data loading (`num_workers=0`)

---

### Investigation 2: Batch Size Optimization
**Instance:** 27968588
**Config:** `train_burgers_upt_full_ddp_optimized.yaml`
**Changes:**
- batch_size: 10 → 32 (3.2x increase)
- accum_steps: 2 → 1 (removed sync overhead)
- num_queries: 2048 → 1024 (reduced query sampling)

**Results:**
- Epoch time: **66s/epoch** (WORSE!)
- GPU utilization: Still fluctuating 0-100%
- Memory: 50.8GB per GPU
- **Larger batches made I/O bottleneck worse**

**Diagnosis:** Larger batch_size = more data to load per batch with same single thread

---

### Investigation 3: RAM Disk + Smaller Batches
**Instance:** 27969033
**Config:** `train_burgers_upt_full_ddp_ramdisk.yaml`
**Changes:**
- batch_size: 32 → 20 (reduced I/O per batch)
- latent_cache_dir: `data/latent_cache` → `/dev/shm/latent_cache` (RAM disk)
- Cache: 13GB in tmpfs (RAM-backed filesystem)

**Results:**
- Epoch time: **58s/epoch** (marginal 8% improvement)
- GPU utilization: Still spiky (10-69%)
- Memory: 32GB per GPU (down from 50GB)
- **RAM disk helped but not enough**

**Diagnosis:** RAM disk eliminated disk I/O, but single-threaded CPU loading still bottleneck

---

### Investigation 4: Shared Memory Multi-Worker Fix

**Root Cause Analysis:**
```python
# PreloadedCacheDataset loads cache into main process RAM
self.cache[idx] = {"latent": data["latent"].float()}  # Not accessible to workers
```

**Problem:** PyTorch DataLoader with `num_workers > 0` spawns worker processes that cannot access main process memory.

**Solution Implemented:**
```python
# Move tensors to shared memory
latent = data["latent"].float().share_memory_()  # Multi-process accessible!
params = params.share_memory_() if params else None
bc = bc.share_memory_() if bc else None
```

**Files Modified:**
- `src/ups/data/parallel_cache.py`: Added `.share_memory_()` to all tensors
- `src/ups/data/latent_pairs.py`: Removed `num_workers=0` restriction

**Expected Impact:**
- 1 worker → 6-8 workers (parallel data loading)
- Epoch time: 60s → 15-20s (3-4x speedup)
- GPU utilization: Sustained 90%+

---

## Shared Memory Implementation Crashes

### Crash Instance 1: 27969292
**Timeline:**
- 01:24: Instance start
- 01:27: Cache precomputation (8 workers, 13GB)
- 01:28:02: WandB init
- 01:28:34: "Training pipeline completed successfully" (32s later)
- Status: Instance exited

**WandB Analysis:**
- Run: `train-20251118_012802`
- State: **CRASHED**
- Events logged: 0
- No training metrics

**Warning in Logs:**
```
[W1118 01:27:51] Producer process terminated before shared CUDA tensors released
See Note [Sharing CUDA tensors]
```

**Key Observation:** No "📦 Preloading ... samples into shared RAM" message from PreloadedCacheDataset.__init__()

### Crash Instance 2: 27969597
**Timeline:**
- Similar pattern to Instance 1
- Cache precomputation completed successfully
- WandB init at 01:39:21
- Silent crash
- "Training pipeline completed successfully" (false positive)

**WandB Analysis:**
- Run: `train-20251118_013920`
- State: "running" (stale)
- Events logged: 0

**Same Warning:**
```
[W1118 01:39:08] Producer process terminated before shared CUDA tensors released
```

---

## Current Hypothesis

### The Silent Crash

**Evidence:**
1. Cache precomputation completes successfully
2. WandB initializes
3. Training script reports "completed successfully"
4. BUT: No PreloadedCacheDataset initialization message
5. No training epochs logged
6. WandB shows crashed/0 events

**Hypothesis:**
The PreloadedCacheDataset is never being instantiated. Training crashes during dataset creation, before `__init__()` completes. The `run_fast_to_sota.py` wrapper catches the exception and reports success anyway.

### Potential Root Causes

**1. Shared Memory Pickling Issue**
- DataLoader pickles dataset to send to workers
- Shared memory tensors may not pickle correctly
- Workers fail to reconstruct dataset

**2. CUDA Tensor Contamination**
- Cache precomputation uses CUDA (`--device cuda`)
- Warning about "shared CUDA tensors" suggests GPU tensors leaked into cache files
- `.share_memory_()` may fail on tensors that were CUDA at some point

**3. Dataset Factory Logic**
- `latent_pairs.py` may be falling back to different dataset type
- Shared memory dataset never selected due to cache validation failure
- Error not propagated properly

---

## Technical Details

### PreloadedCacheDataset Flow

```python
# Expected flow (not happening):
1. __init__() starts
2. print("📦 Preloading {num_samples} cache files into shared RAM...")
3. Load each cache file
4. Call .share_memory_() on tensors
5. print("✅ Preloaded {loaded} samples into shared RAM (multi-process safe)")

# Actual flow:
1. __init__() never starts OR crashes before first print
2. Silent failure
3. Training script catches exception
4. Reports "success"
```

### Cache Structure

```
/dev/shm/latent_cache/burgers1d_train/
├── sample_00000.pt  # Contains {latent: Tensor, params: dict, bc: dict}
├── sample_00001.pt
├── ...
└── sample_01999.pt  # 2000 samples total
```

### Shared Memory Modification

**Before:**
```python
data = torch.load(cache_path, map_location="cpu")
self.cache[idx] = {
    "latent": data["latent"].float(),  # Regular CPU tensor
    "params": data.get("params"),
    "bc": data.get("bc"),
}
```

**After:**
```python
data = torch.load(cache_path, map_location="cpu")
latent = data["latent"].float().share_memory_()  # Shared memory tensor
params = data.get("params")
if params is not None and isinstance(params, torch.Tensor):
    params = params.share_memory_()
bc = data.get("bc")
if bc is not None and isinstance(bc, torch.Tensor):
    bc = bc.share_memory_()

self.cache[idx] = {"latent": latent, "params": params, "bc": bc}
```

---

## GPU Configuration Consideration

### User Question: GPU VRAM Caching
**Q:** Could we bypass CPU entirely and load cache directly into GPU VRAM?

**Analysis:**
- **Pros:** Zero CPU→GPU transfer, fastest access
- **Cons:**
  - Limited VRAM: 80GB/GPU - 32GB (model+batch) = 48GB free
  - Cache is 13GB × 2 GPUs = 26GB total
  - DDP requires separate copy per GPU
  - Reduces room for larger batches

**Recommendation:** Current approach (RAM cache + `pin_memory=True` + async transfer) is better:
- Leverages full 160GB system RAM
- Async CPU→GPU transfer overlaps with compute
- Leaves full VRAM for model scaling

---

## Performance Baselines

| Configuration | Epoch Time | GPU Util | Memory/GPU | Status |
|--------------|-----------|----------|------------|--------|
| Baseline (1-GPU, disk cache) | ~60s | 70-90% | ~32GB | Working |
| DDP unoptimized (2-GPU, disk) | 60s | 30-89% | 50.8GB | Working (no speedup) |
| DDP + large batches | 66s | 0-100% | 50.8GB | Working (worse!) |
| DDP + RAM disk | 58s | 10-69% | 32GB | Working (8% better) |
| DDP + shared memory | CRASHED | N/A | N/A | **Investigating** |

**Expected with working multi-worker:**
- Epoch time: 15-20s (3-4x speedup)
- GPU utilization: 90%+
- Memory: 32-40GB/GPU

---

## Next Steps (Deep Debug)

### 1. Verify Cache File Integrity
- [ ] Check if cache files contain CUDA tensors
- [ ] Verify all tensors are properly CPU before saving
- [ ] Add explicit `.cpu()` calls before `.share_memory_()`

### 2. Test Shared Memory Isolation
- [ ] Create minimal test script
- [ ] Load single cache file
- [ ] Call `.share_memory_()`
- [ ] Verify pickle/unpickle works
- [ ] Test in DataLoader with workers

### 3. Add Detailed Error Logging
- [ ] Modify `run_fast_to_sota.py` to print exceptions
- [ ] Add try/except in PreloadedCacheDataset.__init__
- [ ] Log to file instead of stdout
- [ ] SSH into live instance during training start

### 4. Alternative Approaches
- [ ] **Memory-mapped files** (mmap) instead of shared memory
- [ ] **Separate worker initialization** - load cache in each worker
- [ ] **Shared memory manager** - use multiprocessing.Manager
- [ ] **Pre-fork cache loading** - load before DataLoader spawn

---

## Code References

**Modified Files:**
- `src/ups/data/parallel_cache.py:186-219` - Shared memory implementation
- `src/ups/data/latent_pairs.py:882-891` - Removed `_requires_main_process` flag
- `src/ups/data/latent_pairs.py:957-959` - Removed `num_workers=0` enforcement

**Configs:**
- `configs/train_burgers_upt_full_ddp_optimized.yaml` - Batch size optimization
- `configs/train_burgers_upt_full_ddp_ramdisk.yaml` - RAM disk + shared memory

**Git Commits:**
- `b85c7f0` - Fix: Enable num_workers > 0 for PreloadedCacheDataset with shared memory
- `835fe95` - Add RAM disk optimized DDP config and onstart script

---

## Lessons Learned

1. **I/O bottlenecks dominate** - GPU speedup irrelevant if data loading is single-threaded
2. **Larger batches ≠ better** - When I/O bound, larger batches make it worse
3. **RAM disk helps marginally** - 8% improvement, but doesn't solve core issue
4. **Silent failures are dangerous** - Script reporting "success" while crashing
5. **Shared memory is complex** - Interactions with multiprocessing, pickling, CUDA are subtle

---

## References

- PyTorch Tensor Sharing: https://pytorch.org/docs/stable/multiprocessing.html#sharing-cuda-tensors
- CUDA IPC Warning: https://pytorch.org/docs/stable/notes/multiprocessing.html#cuda-ips-operations
- DataLoader Workers: https://pytorch.org/docs/stable/data.html#multi-process-data-loading

---

**Document Status:** Living document, updated during active investigation
**Last Updated:** 2025-11-18 01:45 UTC
**Investigator:** Claude Code
**Branch:** `feature/distributed-training-ddp`
