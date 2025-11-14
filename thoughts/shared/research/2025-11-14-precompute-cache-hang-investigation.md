---
date: 2025-11-14T21:46:06+0000
researcher: Emery Gunselman
git_commit: b2d8929f8a5194d693f094fdccd96c8ba49711bf
branch: feature/distributed-training-ddp
repository: emgun/universal_simulator
topic: "Why is precompute latent cache hanging?"
tags: [research, codebase, latent-cache, precompute, hang, parallel-processing, multiprocessing]
status: complete
last_updated: 2025-11-14
last_updated_by: Emery Gunselman
---

# Research: Why is Precompute Latent Cache Hanging?

**Date**: 2025-11-14T21:46:06+0000
**Researcher**: Emery Gunselman
**Git Commit**: b2d8929f8a5194d693f094fdccd96c8ba49711bf
**Branch**: feature/distributed-training-ddp
**Repository**: emgun/universal_simulator

## Research Question

Why is the precompute latent cache hanging, and what are the potential blocking points in the system?

## Summary

The latent cache precomputation system is designed to speed up training by 3-5x through pre-encoding physical fields into latent space tokens. However, **the most recent commit (b2d8929, Nov 14 2025) shows that latent cache precomputation is currently hanging and has been temporarily disabled** with `cache_dir: null` in the DDP training config.

The investigation identified **four primary potential hang points**:

1. **Worker initialization with CUDA context** (most likely) - occurs during PyTorch DataLoader's first iteration when workers spawn and initialize encoder models on GPU
2. **HDF5 file loading** - blocking I/O on `h5py.File()` operations with no timeout handling
3. **Torch serialization** - CPU-bound `torch.save()` operations with large tensor payloads
4. **Disk I/O** - blocking `write_bytes()` operations during cache file writes

**Critical finding**: The codebase has **no timeout mechanisms, no watchdog processes, and no graceful worker termination** on any blocking operations, meaning any hang will block indefinitely.

## Detailed Findings

### Current Status: Cache Disabled Due to Hang

**Evidence**: Commit `b2d8929` (Nov 14, 2025)
**File**: `configs/train_pdebench_2task_baseline_ddp.yaml:79`
**Change**:
```yaml
# BEFORE
cache_dir: data/latent_cache

# AFTER
cache_dir: null  # TEMPORARY: Disabled due to precompute hang
```

**Workaround**: Training now uses on-demand encoding (4-8x slower) instead of precomputed cache.

### System Architecture

The latent cache system uses a multi-process architecture with three distinct operating modes:

#### Mode 1: Parallel Encoding (Default for Precompute)

**Implementation**: `src/ups/data/parallel_cache.py:270-322` (`build_parallel_latent_loader`)

**Architecture**:
- **Worker processes**: Load raw HDF5 data via `RawFieldDataset.__getitem__` (lines 59-87)
- **Main process**: Performs GPU encoding via `CollateFnWithEncoding.__call__` (lines 195-267)
- **Benefit**: Avoids CUDA IPC (Inter-Process Communication) issues by keeping encoding in main process

**Configuration** (precompute_latent_cache.py:322-323):
```python
num_workers = 4  # Default, configurable via CLI
persistent_workers = False  # Workers die between epochs
pin_memory = True  # Faster CPU→GPU transfers
prefetch_factor = 2  # Read-ahead buffer size
```

**Process spawning** (precompute_latent_cache.py:307):
```python
multiprocessing.set_start_method('spawn', force=True)
```
Uses `'spawn'` for CUDA safety (slower than `'fork'` but required for GPU code).

#### Mode 2: PreloadedCacheDataset (RAM Preload)

**Implementation**: `src/ups/data/parallel_cache.py:90-167`

**When used**:
- Cache is complete (`check_cache_complete` returns True)
- Sufficient RAM available (`check_sufficient_ram` with 20% safety margin)

**Behavior**:
- Loads **ALL** cache files into RAM at initialization (lines 109-126)
- Returns data instantly from memory (no disk I/O during training)
- Achieves 90%+ GPU utilization

#### Mode 3: Legacy Single-Worker (Fallback)

**When used**: If parallel encoding fails with `AttributeError` or `TypeError`

**Behavior**:
- All operations (loading, encoding, caching) in main process
- No multiprocessing
- Slowest mode but most reliable

### Critical Blocking Points

#### 1. Worker Initialization with CUDA Context (Most Likely Hang Point)

**Location**: PyTorch DataLoader internals, triggered on first `for batch in loader:` iteration

**What happens**:
1. Main process calls `DataLoader.__iter__()`
2. PyTorch spawns `num_workers` processes using `multiprocessing.spawn()`
3. Each worker process initializes:
   - Imports dataset class
   - Creates dataset instance
   - **Loads encoder model** (if in dataset)
   - **Initializes CUDA context** (if encoder on GPU)

**Why it hangs**:
- **CUDA initialization failures**: If GPU resources exhausted or CUDA runtime conflicts
- **IPC deadlocks**: Workers try to share CUDA tensors across process boundaries
- **Memory exhaustion**: Each worker loads full encoder model (~100-500MB per worker)

**Evidence of related issues**:
- Commit `291041c` fixed "Producer process terminated" crashes by keeping encoder on CPU
- Commit `2595dca` "Force dataset device to CPU for DDP safety"

**Code location** (latent_pairs.py:334-363):
```python
# Fixed version: temporarily move encoder to device only during encoding
encoder_was_on = next(self.encoder.parameters()).device
if encoder_was_on != self.device:
    self.encoder.to(self.device)  # Temporary move for encoding

# ... encoding happens ...

if encoder_was_on != self.device:
    self.encoder.to(encoder_was_on)  # Move back to CPU to avoid IPC
```

**Why it can still hang**:
- Even with CPU storage, the `.to(device)` call in main process can hang if CUDA context issues exist
- Workers can timeout during model initialization if resources constrained

#### 2. HDF5 File Loading (Second Most Likely)

**Location**: `src/ups/data/pdebench.py:112-145`

**Blocking I/O operations**:
```python
for path in shard_paths:
    with h5py.File(path, "r") as f:  # ← Blocking file open, NO TIMEOUT
        f_fields = torch.from_numpy(f[spec.field_key][...]).float()  # ← Blocking array read
```

**Why it hangs**:
- **Filesystem issues**: NFS locks, remote storage timeouts, permission errors
- **File corruption**: HDF5 library hangs on corrupted file headers
- **Concurrent access**: Multiple workers accessing same file simultaneously (HDF5 not designed for this)

**Memory behavior**:
- Loads **entire dataset** into RAM (line 142: `torch.cat(fields_list)`)
- For multi-task: concatenates all shards
- Example: 2 tasks × 800 samples × 64×64×100 timesteps ≈ 10-20GB RAM

**No error handling**: No try/except around file operations, no timeout mechanism.

#### 3. GPU Encoding Operations

**Location**: `src/ups/data/latent_pairs.py:351-359` (`_fields_to_latent_batch`)

**Synchronization points**:
```python
encoder.to(self.device)  # ← Blocks until model transfer completes
fields.to(self.device, non_blocking=True)  # ← Queues transfer
latent = encoder(field_inputs, ...)  # ← GPU forward pass
encoder.to(encoder_was_on)  # ← Blocks until model transfer back
```

**Why it hangs**:
- **CUDA OOM**: If GPU memory exhausted, allocation hangs or crashes
- **CUDA context conflicts**: Multiple processes initializing CUDA simultaneously
- **cudnn initialization**: First GPU operation can hang if cudnn library issues

**No timeout**: All CUDA operations are synchronous with no timeout handling.

#### 4. Cache File Writing

**Location**: `src/ups/data/latent_pairs.py:377-380`

**Write sequence**:
```python
buffer = io.BytesIO()
torch.save(payload, buffer)  # ← CPU-bound serialization, can be slow
tmp_path.write_bytes(buffer.getvalue())  # ← Blocking disk I/O, NO TIMEOUT
tmp_path.replace(cache_path)  # ← Filesystem atomic rename
```

**Why it hangs**:
- **Disk full**: Write blocks if no space available
- **Slow storage**: Network-attached storage (NFS, S3 FUSE) can have very high latency
- **Large payloads**: 512-dim latent cache can be 5-20MB per sample, serialization takes seconds

**No error handling**: No try/except, no disk space checks, no timeout.

### Missing Safeguards

**Critical gaps in the implementation**:

1. **No timeouts on blocking operations**
   - HDF5 file opens
   - CUDA operations
   - Disk I/O
   - Worker initialization

2. **No watchdog process**
   - No monitoring of worker health
   - No heartbeat mechanism
   - No automatic worker restart on hang

3. **No graceful termination**
   - Hung workers never get killed
   - No cleanup on timeout
   - Main process waits indefinitely

4. **No progress monitoring beyond tqdm**
   - Progress bar updates only when batches complete
   - If workers hang before first batch, progress bar never appears
   - No per-worker progress tracking

### Known Issues and Recent Fixes

#### Issue 1: CUDA IPC Conflicts (Fixed in commit 291041c)

**Problem**: Encoder on GPU causes "Producer process terminated" in DDP training

**Root cause**:
- Workers tried to share CUDA tensors across processes
- PyTorch's multiprocessing can't share CUDA tensors in `'spawn'` mode
- Worker initialization hung or crashed

**Fix**: Keep encoder on CPU in dataset, temporarily move to device only in `__getitem__`

**Status**: Fixed for DDP training, but may still affect precompute script

#### Issue 2: Precompute Hang (ACTIVE, commit b2d8929, Nov 14 2025)

**Symptom**: Latent cache precomputation hangs indefinitely

**Workaround**: Disabled cache via `cache_dir: null` in config

**Status**: **Open**, no root cause identified

**Likely causes** (prioritized by probability):
1. Worker initialization with CUDA context conflicts
2. HDF5 file locks from concurrent worker access
3. Filesystem issues on remote storage (VastAI instances use network storage)
4. Memory pressure causing OOM during worker spawn

## Code References

### Main Entry Point
- `scripts/precompute_latent_cache.py:302` - `main()` function
- `scripts/precompute_latent_cache.py:307` - Multiprocessing setup with `'spawn'` method
- `scripts/precompute_latent_cache.py:188-221` - `_iter_dataset()` iteration loop

### Parallel Processing
- `src/ups/data/parallel_cache.py:270-322` - `build_parallel_latent_loader()` factory
- `src/ups/data/parallel_cache.py:33-88` - `RawFieldDataset` (worker-side loading)
- `src/ups/data/parallel_cache.py:169-268` - `CollateFnWithEncoding` (main process encoding)

### Cache Logic
- `src/ups/data/latent_pairs.py:303-380` - `GridLatentPairDataset.__getitem__()` with caching
- `src/ups/data/latent_pairs.py:334-363` - Encoder device management to avoid IPC issues
- `src/ups/data/latent_pairs.py:377-380` - Cache file writing

### Data Loading
- `src/ups/data/pdebench.py:112-145` - HDF5 file loading (potential hang point)
- `src/ups/data/pdebench.py:152-161` - `__getitem__()` returns fields

### Configuration
- `configs/train_pdebench_2task_baseline_ddp.yaml:79` - Cache disabled due to hang
- `configs/cache_precompute_defaults.yaml` - Default precompute settings

## Architecture Documentation

### Cache File Structure

Cache files are organized as:
```
data/latent_cache/
├── {task}_{split}/
│   ├── sample_00000.pt      # Contains {"latent", "params", "bc", optional "fields"}
│   ├── sample_00001.pt
│   ├── ...
│   └── .cache_metadata.json  # Metadata with config hash and timestamp
```

### Three Parallel Processing Approaches

1. **DataLoader Worker Processes**
   - PyTorch's built-in multiprocessing
   - Workers load raw HDF5, main process encodes on GPU
   - Default: 4 workers, spawn method

2. **Parallel Encoding Mode** (Optional)
   - `RawFieldDataset` in workers + `CollateFnWithEncoding` in main process
   - Avoids CUDA IPC by keeping encoder in main process
   - Fallback to legacy mode on errors

3. **PreloadedCacheDataset** (RAM Preload)
   - Loads entire cache into RAM at initialization
   - Used when cache complete and sufficient RAM available
   - Achieves 90%+ GPU utilization

### Memory and I/O Characteristics

**Memory usage**:
- PDEBench dataset: 10-20GB RAM (entire dataset loaded at init)
- Encoder model: 100-500MB per worker process
- Cache files: 5-20MB per sample for 512-dim latents
- PreloadedCache: 10-20GB RAM for full cache

**I/O patterns**:
- HDF5 reads: Sequential, blocking, no prefetch
- Cache writes: Synchronous, atomic rename
- No async I/O or parallel writes

## Historical Context (from thoughts/)

### Related Research Documents

1. **`thoughts/shared/research/2025-11-13-massive-training-speed-optimization.md`**
   - Documents that latent cache precomputation is "already built but disabled"
   - Notes cache provides 4-8x speedup when working
   - PreloadedCacheDataset achieves 90%+ GPU utilization

2. **`thoughts/shared/research/2025-11-13-ddp-crash-investigation.md`**
   - Documents CUDA IPC conflicts between DDP and latent cache
   - Commit `291041c` fixed "DDP + latent cache CUDA IPC conflicts"
   - Commit `2595dca` "Force dataset device to CPU for DDP safety"

3. **`thoughts/shared/research/2025-11-05-pdebench-data-pipeline.md`**
   - Complete cache architecture documentation
   - Documents three operating modes (PreloadedCache, Parallel Encoding, Legacy)
   - Cache completeness checks: `check_cache_complete`, `estimate_cache_size_mb`, `check_sufficient_ram`

### Historical Timeline

- **Nov 5, 2025**: PDEBench data pipeline documented with cache infrastructure
- **Nov 13, 2025**: DDP crash investigation identifies CUDA IPC conflicts with cache
- **Nov 13, 2025**: Speed optimization research documents cache as disabled
- **Nov 14, 2025**: **Cache explicitly disabled in config due to precompute hang** (commit b2d8929)

## Open Questions

1. **Which specific blocking point causes the hang?**
   - Need instrumentation/logging to identify exact location
   - Add timeout wrappers around HDF5 opens, CUDA operations, disk writes
   - Add per-worker heartbeat monitoring

2. **Why doesn't the hang occur in DDP training mode?**
   - DDP training uses on-demand encoding (no precompute script)
   - Different DataLoader configuration?
   - Different memory/CUDA initialization pattern?

3. **Does the hang occur locally or only on VastAI instances?**
   - VastAI uses network storage (could cause HDF5 hangs)
   - Different CUDA driver versions?
   - Test precompute script locally to isolate

4. **Is the hang deterministic or intermittent?**
   - Always hangs at same sample index?
   - Related to specific HDF5 files?
   - Timing-dependent (race condition)?

5. **Can we add timeout and recovery mechanisms?**
   - Wrap blocking operations in timeout decorators
   - Add worker health monitoring
   - Implement graceful restart on worker hang

## Recommended Next Steps

### Immediate Debugging

1. **Add instrumentation to precompute script**:
   ```python
   # In _iter_dataset(), add per-batch logging
   for i, batch in enumerate(loader):
       print(f"[{time.time()}] Batch {i} completed", flush=True)
   ```

2. **Test with num_workers=0** (no multiprocessing):
   ```bash
   python scripts/precompute_latent_cache.py \
     --config configs/cache_precompute_defaults.yaml \
     --num-workers 0 \
     --task burgers_train \
     --split train
   ```

3. **Test locally vs VastAI**:
   - Run precompute on local machine
   - Compare behavior with VastAI instance
   - Isolate storage/network vs code issues

### Add Timeout Mechanisms

1. **Wrap HDF5 operations**:
   ```python
   import signal

   def timeout_handler(signum, frame):
       raise TimeoutError("HDF5 operation timed out")

   signal.signal(signal.SIGALRM, timeout_handler)
   signal.alarm(30)  # 30 second timeout
   try:
       with h5py.File(path, "r") as f:
           fields = f[spec.field_key][...]
   finally:
       signal.alarm(0)
   ```

2. **Add DataLoader timeout**:
   ```python
   loader = DataLoader(..., timeout=60)  # PyTorch built-in
   ```

3. **Add worker watchdog**:
   ```python
   # Monitor worker processes, kill if no progress in 60s
   for worker in loader._workers:
       if worker.is_alive() and not worker.has_progress:
           worker.terminate()
   ```

### Alternative Approaches

1. **Disable multiprocessing entirely**:
   - Set `num_workers=0` in precompute script
   - Slower but more reliable

2. **Use sequential caching**:
   - Cache one sample at a time instead of batch
   - More disk I/O but simpler process model

3. **Move to on-demand caching**:
   - Keep `cache_dir` set but don't precompute
   - Let training populate cache lazily
   - Slower first epoch but no precompute hang

4. **Use different storage backend**:
   - Copy HDF5 files to local SSD on VastAI instances
   - Avoid network storage issues
   - Trade storage space for reliability
