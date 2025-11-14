# Fix Precompute Latent Cache Hang Implementation Plan

## Overview

Fix the latent cache precomputation system that currently hangs indefinitely during execution, forcing cache to be disabled (commit b2d8929). The solution adds timeout mechanisms, error handling, watchdog monitoring, and I/O optimizations across 3 progressive phases while maintaining the 4-8x training speedup from caching.

## Current State Analysis

### What Exists Now

**Cache System Architecture** (3 modes):
1. **PreloadedCacheDataset** (`src/ups/data/parallel_cache.py:90-167`) - Loads all cache into RAM, 90%+ GPU util
2. **Parallel Encoding** (`src/ups/data/parallel_cache.py:270-322`) - Workers load HDF5, main process encodes on GPU (4-8x faster)
3. **Legacy Mode** (`scripts/precompute_latent_cache.py:173-183`) - Single-process, slowest but stable

**Current Status**:
- Cache disabled in `configs/train_pdebench_2task_baseline_ddp.yaml:79` (`cache_dir: null`)
- Training uses on-demand encoding (4-8x slower)
- Precompute script hangs indefinitely with no error message

### Key Discoveries

**Critical Gaps** (identified in research document):

1. **No timeout mechanisms** anywhere:
   - `scripts/precompute_latent_cache.py:188` - DataLoader iteration has no timeout
   - `src/ups/data/parallel_cache.py:270-322` - `build_parallel_latent_loader()` missing `timeout` param
   - `src/ups/data/pdebench.py:113` - HDF5 `h5py.File()` blocks indefinitely
   - `src/ups/data/latent_pairs.py:377-380` - Cache `write_bytes()` has no timeout

2. **No error handling**:
   - HDF5 file operations have no try/except blocks
   - No disk space checks before writing cache
   - No filesystem error recovery

3. **No worker health monitoring**:
   - `_iter_dataset()` waits forever if workers hang
   - No progress heartbeat mechanism
   - No automatic worker restart

4. **VastAI/Vultr storage issues**:
   - Network-attached storage causes HDF5 hang (most likely root cause)
   - No local storage copy utility
   - No detection of remote vs local storage

5. **No test coverage**:
   - No unit tests for cache precomputation
   - No integration tests for parallel loading
   - Existing tests: `tests/unit/test_latent_state.py`, `tests/unit/test_latent_operator.py` (unrelated)

### Constraints Discovered

- Must use `multiprocessing.spawn` mode for CUDA safety (line 307)
- Encoder must stay on CPU in dataset to avoid IPC issues (commits 291041c, 2595dca)
- Cache metadata hash prevents stale caches (`compute_cache_hash()` at line 243)
- PyTorch DataLoader has built-in `timeout` parameter (not currently used)

## Desired End State

### Success Criteria

1. **Precompute script completes successfully** on VastAI/Vultr instances with network storage
2. **Hangs fail gracefully** with actionable error messages within 2-5 minutes
3. **Cache creation is 2-3x faster** than current baseline (when working)
4. **Training can re-enable cache** with confidence (`cache_dir: data/latent_cache`)
5. **Comprehensive test coverage** for cache system (unit + integration)

### How to Verify

**Automated**:
```bash
# Phase 1: Precompute completes without hanging
python scripts/precompute_latent_cache.py \
  --config configs/cache_precompute_defaults.yaml \
  --tasks burgers1d --splits train --limit 100 \
  --dataloader-timeout 120 --hdf5-timeout 60

# Phase 2: Watchdog catches simulated hang
INJECT_HANG=1 python scripts/precompute_latent_cache.py ...
# Should fail with "No progress for 120s" error

# Phase 3: Tests pass
pytest tests/unit/test_cache_precompute.py -v
pytest tests/integration/test_parallel_cache.py -v
```

**Manual**:
- Verify cache precompute completes on VastAI/Vultr instance
- Verify training with cache enabled reaches epoch 1
- Verify error messages are actionable when hang injected

## What We're NOT Doing

- Rewriting the entire parallel processing architecture
- Changing latent cache file format or schema
- Modifying encoder architecture or latent dimensions
- Adding distributed caching across multiple machines
- Supporting non-POSIX filesystems (e.g., Windows)
- Implementing async I/O or non-blocking HDF5 operations

## Implementation Approach

**Strategy**: Progressive enhancement across 3 phases, each independently testable:

1. **Phase 1 (Immediate Safety)**: Add timeout parameters and basic error handling - prevents indefinite hangs
2. **Phase 2 (Reliability)**: Add watchdog monitoring and graceful degradation - enables diagnosis and fallback
3. **Phase 3 (Performance + Tests)**: Optimize I/O patterns and add comprehensive tests - improves speed and confidence

Each phase is self-contained and can be validated before proceeding to the next.

---

## Phase 1: Immediate Safety (Timeout Parameters)

### Overview

Add timeout parameters to all blocking operations and basic error handling to prevent indefinite hangs. This phase makes hangs fail-fast with clear error messages instead of blocking forever.

### Changes Required

#### 1. Add CLI Arguments for Timeouts

**File**: `scripts/precompute_latent_cache.py`
**Changes**: Add argparse parameters for timeout configuration

```python
# Around line 330 (after existing argparse arguments)
parser.add_argument(
    "--dataloader-timeout",
    type=int,
    default=120,
    help="DataLoader worker timeout in seconds (default: 120, 0=disabled)",
)
parser.add_argument(
    "--hdf5-timeout",
    type=int,
    default=60,
    help="HDF5 file operation timeout in seconds (default: 60, 0=disabled)",
)
parser.add_argument(
    "--cache-write-timeout",
    type=int,
    default=30,
    help="Cache write operation timeout in seconds (default: 30, 0=disabled)",
)
```

#### 2. Pass Timeout to DataLoader

**File**: `src/ups/data/parallel_cache.py`
**Changes**: Add `timeout` parameter to `build_parallel_latent_loader()`

**Location**: Lines 270-322

```python
def build_parallel_latent_loader(
    dataset: PDEBenchDataset,
    encoder: GridEncoder,
    coords: torch.Tensor,
    grid_shape: Tuple[int, int],
    field_name: str,
    device: torch.device,
    batch_size: int = 16,
    num_workers: int = 4,
    cache_dir: Optional[Path] = None,
    cache_dtype: Optional[torch.dtype] = torch.float16,
    time_stride: int = 1,
    rollout_horizon: int = 1,
    pin_memory: bool = True,
    prefetch_factor: Optional[int] = None,
    timeout: int = 0,  # ← NEW: DataLoader worker timeout
) -> DataLoader:
    """Build a DataLoader with parallel workers that avoids device mismatch.

    Args:
        ...
        timeout: DataLoader worker timeout in seconds (0=disabled, default: 0)
    """
    raw_dataset = RawFieldDataset(
        dataset,
        field_name=field_name,
        cache_dir=cache_dir,
        cache_dtype=cache_dtype,
        time_stride=time_stride,
        rollout_horizon=rollout_horizon,
    )

    collate_fn = CollateFnWithEncoding(
        encoder=encoder,
        coords=coords,
        grid_shape=grid_shape,
        field_name=field_name,
        device=device,
        cache_dtype=cache_dtype,
        time_stride=time_stride,
        rollout_horizon=rollout_horizon,
    )

    loader_kwargs: Dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": True,
        "collate_fn": collate_fn,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }
    if prefetch_factor is not None and num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    # Add timeout if specified (PyTorch built-in)
    if timeout > 0 and num_workers > 0:
        loader_kwargs["timeout"] = timeout

    return DataLoader(raw_dataset, **loader_kwargs)
```

#### 3. Update Precompute Script to Use Timeout

**File**: `scripts/precompute_latent_cache.py`
**Changes**: Pass timeout to parallel loader

**Location**: Lines 146-167 (inside `_iter_dataset()`)

```python
# Around line 152
loader = build_parallel_latent_loader(
    dataset.base,
    dataset.encoder,
    dataset.coords,
    dataset.grid_shape,
    dataset.field_name,
    dataset.device,
    batch_size=batch_size,
    num_workers=num_workers,
    cache_dir=dataset.cache_dir,
    cache_dtype=dataset.cache_dtype,
    time_stride=dataset.time_stride,
    rollout_horizon=dataset.rollout_horizon,
    pin_memory=pin_memory,
    prefetch_factor=2,
    timeout=timeout,  # ← NEW: Pass timeout from args
)
```

**Location**: Lines 129-145 (`_iter_dataset()` signature and calls)

```python
def _iter_dataset(
    dataset: Dataset,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    use_parallel: bool = True,
    timeout: int = 0,  # ← NEW: Add timeout parameter
) -> None:
    """Iterate dataset to populate cache.

    Args:
        dataset: Dataset to iterate
        batch_size: Batch size for encoding
        num_workers: Number of parallel workers
        pin_memory: Enable pinned memory
        use_parallel: Use parallel encoding (4-8× faster)
        timeout: DataLoader worker timeout in seconds (0=disabled)
    """
```

**Location**: Lines 433-439 (main() calls to _iter_dataset())

```python
_iter_dataset(
    dataset,
    batch_size=max(1, args.batch_size),
    num_workers=max(0, args.num_workers),
    pin_memory=args.pin_memory,
    use_parallel=args.parallel,
    timeout=args.dataloader_timeout,  # ← NEW: Pass from CLI args
)
```

#### 4. Add HDF5 SWMR Mode and Basic Error Handling

**File**: `src/ups/data/pdebench.py`
**Changes**: Enable SWMR mode for concurrent reads, add error handling

**Location**: Lines 112-145

```python
for path in shard_paths:
    try:
        # Use SWMR (Single-Writer-Multiple-Reader) mode for parallel safety
        with h5py.File(path, "r", libver='latest', swmr=True) as f:
            f_fields = torch.from_numpy(f[spec.field_key][...]).float()
            if cfg.normalize:
                f_fields = _normalise_fields(f_fields)
            fields_list.append(f_fields)
            if spec.target_key and spec.target_key in f:
                targets_list.append(torch.from_numpy(f[spec.target_key][...]).float())
            else:
                targets_list.append(f_fields)
            # ... rest of param/bc logic unchanged ...
    except (OSError, IOError) as e:
        # HDF5 file errors (permission, corruption, network timeout)
        raise RuntimeError(
            f"Failed to read HDF5 file {path}: {e}. "
            f"If using network storage, try copying data to local disk first."
        ) from e
    except Exception as e:
        # Unexpected errors
        raise RuntimeError(f"Unexpected error reading {path}: {e}") from e
```

#### 5. Add Disk Space Check Before Cache Write

**File**: `src/ups/data/latent_pairs.py`
**Changes**: Check available disk space before writing cache

**Location**: Lines 364-380 (inside GridLatentPairDataset.__getitem__)

```python
if self.cache_dir and not cache_hit:
    to_store = (
        latent_seq.to(self.cache_dtype) if self.cache_dtype is not None else latent_seq
    )
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    tmp_path.unlink(missing_ok=True)
    payload = {
        "latent": to_store.cpu(),
        "params": params_cpu,
        "bc": bc_cpu,
    }

    # NEW: Check available disk space before writing
    try:
        import shutil
        stat = shutil.disk_usage(self.cache_dir)
        available_mb = stat.free / (1024 * 1024)
        # Estimate cache file size (conservative: 20MB per sample)
        required_mb = 20
        if available_mb < required_mb:
            raise IOError(
                f"Insufficient disk space: {available_mb:.0f}MB available, "
                f"~{required_mb}MB required for cache write"
            )
    except Exception as e:
        # If disk check fails, log warning but continue
        print(f"⚠️  Could not check disk space: {e}")

    # Write cache file
    try:
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        tmp_path.write_bytes(buffer.getvalue())
        tmp_path.replace(cache_path)
    except (OSError, IOError) as e:
        # Cache write failed, clean up and re-raise
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Failed to write cache file {cache_path}: {e}. "
            f"Check disk space and filesystem permissions."
        ) from e
```

#### 6. Add Resource-Aware Worker Count

**File**: `scripts/precompute_latent_cache.py`
**Changes**: Add function to calculate optimal worker count

**Location**: After imports, around line 40

```python
def get_optimal_workers(
    num_workers_requested: int,
    encoder_size_mb: int = 200,
    batch_size: int = 4,
) -> int:
    """Calculate optimal worker count based on available resources.

    Args:
        num_workers_requested: User-requested worker count (0=auto)
        encoder_size_mb: Estimated encoder model size in MB
        batch_size: Batch size for prefetching

    Returns:
        Optimal worker count (>= 1)
    """
    if num_workers_requested > 0:
        # User specified, use as-is
        return num_workers_requested

    try:
        import psutil
        available_ram_gb = psutil.virtual_memory().available / 1e9
        ram_per_worker_gb = (encoder_size_mb / 1024) + (batch_size * 0.1)
        max_workers_ram = int(available_ram_gb / ram_per_worker_gb)
        max_workers_cpu = os.cpu_count() or 4
        optimal = max(1, min(max_workers_ram, max_workers_cpu, 8))
        print(f"📊 Auto-selected {optimal} workers (RAM: {available_ram_gb:.1f}GB, CPUs: {max_workers_cpu})")
        return optimal
    except Exception:
        # Fallback to conservative default
        return 4
```

**Location**: Lines 436 (in main(), before _iter_dataset() call)

```python
# Auto-calculate optimal workers if not specified
actual_num_workers = get_optimal_workers(args.num_workers, batch_size=args.batch_size)
if actual_num_workers != args.num_workers:
    print(f"ℹ️  Adjusted workers: {args.num_workers} → {actual_num_workers}")

_iter_dataset(
    dataset,
    batch_size=max(1, args.batch_size),
    num_workers=actual_num_workers,  # ← Use calculated value
    pin_memory=args.pin_memory,
    use_parallel=args.parallel,
    timeout=args.dataloader_timeout,
)
```

### Success Criteria

#### Automated Verification:
- [x] Precompute script accepts new CLI arguments: `python scripts/precompute_latent_cache.py --help` shows timeout options
- [ ] DataLoader timeout works: With `--dataloader-timeout 5` and slow storage, script fails within ~10s (not indefinitely)
- [ ] HDF5 error handling works: With corrupted HDF5 file, script fails with actionable error message
- [ ] Disk space check works: With full disk, script fails with "Insufficient disk space" error
- [ ] Resource calculation works: `get_optimal_workers()` returns reasonable value (1-8 workers)

#### Manual Verification:
- [ ] Run precompute locally with `--dataloader-timeout 120 --hdf5-timeout 60` and verify it completes
- [ ] Run on VastAI/Vultr instance and verify it either completes or fails with clear error (no silent hang)
- [ ] Verify timeout error messages are actionable (tell user what to do next)

**Implementation Note**: After completing this phase and all automated verification passes, test manually on VastAI/Vultr before proceeding to Phase 2.

---

## Phase 2: Reliability & Monitoring (Watchdog + Fallback)

### Overview

Add progress heartbeat monitoring to detect silent hangs, HDF5 timeout wrapper for filesystem issues, and graceful fallback strategy. This phase enables diagnosing exact hang location and automatic recovery.

### Changes Required

#### 1. Add Progress Heartbeat Watchdog

**File**: `scripts/precompute_latent_cache.py`
**Changes**: Add watchdog thread to detect stalled progress

**Location**: After `get_optimal_workers()`, around line 80

```python
import threading

class ProgressWatchdog:
    """Monitor progress and detect hung workers."""

    def __init__(self, timeout_seconds: int = 120, check_interval: int = 10):
        self.timeout = timeout_seconds
        self.check_interval = check_interval
        self.last_progress = {"time": time.time(), "batch": -1}
        self.thread = None
        self.stop_event = threading.Event()

    def update(self, batch_idx: int):
        """Update progress timestamp."""
        self.last_progress["time"] = time.time()
        self.last_progress["batch"] = batch_idx

    def _watchdog_loop(self):
        """Background thread that monitors progress."""
        while not self.stop_event.is_set():
            time.sleep(self.check_interval)
            elapsed = time.time() - self.last_progress["time"]
            if elapsed > self.timeout:
                batch = self.last_progress["batch"]
                print(f"\n❌ HANG DETECTED: No progress for {elapsed:.0f}s")
                print(f"   Last completed batch: {batch}")
                print(f"   Likely hung in: DataLoader iteration or worker process")
                print(f"   Try: --num-workers 0 (disable multiprocessing)")
                print(f"   Or: --dataloader-timeout {self.timeout // 2} (shorter timeout)")
                os._exit(1)  # Hard exit since workers may be hung

    def start(self):
        """Start watchdog monitoring."""
        if self.timeout <= 0:
            return  # Watchdog disabled
        self.thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self.thread.start()
        print(f"🐕 Watchdog started: will abort if no progress for {self.timeout}s")

    def stop(self):
        """Stop watchdog monitoring."""
        if self.thread:
            self.stop_event.set()
            self.thread.join(timeout=1)
```

**Location**: Lines 129-221 (update `_iter_dataset()` to use watchdog)

```python
def _iter_dataset(
    dataset: Dataset,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    use_parallel: bool = True,
    timeout: int = 0,
    watchdog_timeout: int = 120,  # ← NEW: Watchdog timeout
) -> None:
    """Iterate dataset to populate cache.

    Args:
        dataset: Dataset to iterate
        batch_size: Batch size for encoding
        num_workers: Number of parallel workers
        pin_memory: Enable pinned memory
        use_parallel: Use parallel encoding (4-8× faster)
        timeout: DataLoader worker timeout in seconds (0=disabled)
        watchdog_timeout: Progress watchdog timeout in seconds (0=disabled)
    """
    # ... existing loader creation code ...

    total_batches = len(loader)
    iterator = _maybe_tqdm(loader, total_batches, desc="encoding")

    # NEW: Start watchdog
    watchdog = ProgressWatchdog(timeout_seconds=watchdog_timeout)
    watchdog.start()

    try:
        for i, batch in enumerate(iterator):
            # NEW: Update watchdog progress
            watchdog.update(i)

            # Accessing the batch ensures __getitem__ executes and caches samples.
            if isinstance(batch, list):
                for item in batch:
                    del item
            del batch
    except (AttributeError, TypeError) as e:
        # ... existing fallback logic ...
        pass
    finally:
        # NEW: Stop watchdog
        watchdog.stop()
```

**Location**: Line 330 (add CLI argument)

```python
parser.add_argument(
    "--watchdog-timeout",
    type=int,
    default=120,
    help="Watchdog timeout in seconds (0=disabled, default: 120)",
)
```

**Location**: Line 439 (pass to _iter_dataset)

```python
_iter_dataset(
    dataset,
    batch_size=max(1, args.batch_size),
    num_workers=actual_num_workers,
    pin_memory=args.pin_memory,
    use_parallel=args.parallel,
    timeout=args.dataloader_timeout,
    watchdog_timeout=args.watchdog_timeout,  # ← NEW
)
```

#### 2. Add HDF5 Timeout Wrapper

**File**: `src/ups/data/pdebench.py`
**Changes**: Add signal-based timeout for HDF5 operations

**Location**: After imports, around line 10

```python
import signal
from contextlib import contextmanager
from typing import ContextManager

@contextmanager
def hdf5_timeout(seconds: int) -> ContextManager:
    """Context manager for HDF5 operation timeout.

    Uses SIGALRM to interrupt blocking HDF5 operations.
    Only works on Unix systems.

    Args:
        seconds: Timeout in seconds (0=disabled)

    Raises:
        TimeoutError: If operation exceeds timeout
    """
    if seconds <= 0 or os.name == 'nt':  # Disabled or Windows
        yield
        return

    def timeout_handler(signum, frame):
        raise TimeoutError(
            f"HDF5 operation timed out after {seconds}s. "
            f"Possible causes: network storage lag, file corruption, or large file size."
        )

    # Set alarm
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        # Cancel alarm
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
```

**Location**: Lines 112-145 (wrap HDF5 operations with timeout)

```python
# Add timeout parameter to PDEBenchDataset.__init__
def __init__(
    self,
    cfg: PDEBenchConfig,
    download: bool = True,
    hdf5_timeout: int = 0,  # ← NEW: HDF5 operation timeout
) -> None:
    # ... existing code ...

    for path in shard_paths:
        try:
            with hdf5_timeout(hdf5_timeout):  # ← NEW: Wrap with timeout
                with h5py.File(path, "r", libver='latest', swmr=True) as f:
                    f_fields = torch.from_numpy(f[spec.field_key][...]).float()
                    # ... rest of logic unchanged ...
        except TimeoutError as e:
            raise RuntimeError(
                f"HDF5 read timeout for {path}. "
                f"Try: 1) Copy data to local storage, or 2) Increase --hdf5-timeout"
            ) from e
        except (OSError, IOError) as e:
            # ... existing error handling ...
```

**Location**: Update `_build_pdebench_dataset()` in `src/ups/data/latent_pairs.py`

**File**: `src/ups/data/latent_pairs.py`
**Changes**: Pass hdf5_timeout through to PDEBenchDataset

Around line 480 (in `_build_pdebench_dataset()`):

```python
def _build_pdebench_dataset(
    cfg: dict[str, Any],
    hdf5_timeout: int = 0,  # ← NEW parameter
) -> tuple[PDEBenchDataset, GridEncoder, tuple[int, int], str]:
    # ... existing code ...

    pde_cfg = PDEBenchConfig(
        task=task,
        split=split,
        root=cfg.get("root", "data/pdebench"),
        normalize=cfg.get("normalize", False),
        download=cfg.get("download", False),
    )
    dataset = PDEBenchDataset(pde_cfg, download=download, hdf5_timeout=hdf5_timeout)  # ← Pass timeout
    # ... rest unchanged ...
```

**File**: `scripts/precompute_latent_cache.py`
**Changes**: Pass hdf5_timeout through the call stack

**Location**: Lines 48-120 (update `_instantiate_dataset()`)

```python
def _instantiate_dataset(
    *,
    task: str,
    split: str,
    data_cfg: Dict[str, object],
    latent_cfg: Dict[str, object],
    device: torch.device,
    cache_dir: Optional[Path],
    cache_dtype: Optional[torch.dtype],
    rollout_horizon: int,
    use_inverse_losses: bool = False,
    time_stride: int = 1,
    hdf5_timeout: int = 0,  # ← NEW parameter
) -> Dataset:
    ds_cfg = {
        **data_cfg,
        "task": task,
        "split": split,
        "latent_dim": latent_cfg.get("dim", 32),
        "latent_len": latent_cfg.get("tokens", 16),
    }
    spec = get_pdebench_spec(task)
    ds_cache = cache_dir / f"{task}_{split}" if cache_dir else None
    if spec.kind == "grid":
        dataset, encoder, grid_shape, field_name = _build_pdebench_dataset(
            ds_cfg,
            hdf5_timeout=hdf5_timeout,  # ← Pass timeout
        )
        # ... rest unchanged ...
```

**Location**: Line 398 (in main(), pass hdf5_timeout to _instantiate_dataset())

```python
dataset = _instantiate_dataset(
    task=task,
    split=split,
    data_cfg=data_cfg,
    latent_cfg=latent_cfg,
    device=device,
    cache_dir=cache_root,
    cache_dtype=cache_dtype,
    rollout_horizon=rollout_horizon,
    use_inverse_losses=use_inverse_losses,
    time_stride=time_stride,
    hdf5_timeout=args.hdf5_timeout,  # ← NEW: Pass from CLI
)
```

#### 3. Implement Fallback Strategy

**File**: `scripts/precompute_latent_cache.py`
**Changes**: Add robust loader creation with fallback

**Location**: Replace lines 146-221 (entire `_iter_dataset()` function)

```python
def _build_robust_loader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    use_parallel: bool,
    timeout: int,
) -> tuple[DataLoader, str]:
    """Build DataLoader with automatic fallback on failures.

    Tries progressively safer modes:
    1. Parallel encoding (fastest)
    2. Single worker (slower)
    3. Main process only (slowest, most reliable)

    Returns:
        (loader, mode_name) tuple
    """
    use_parallel_loader = use_parallel and num_workers > 0 and isinstance(dataset, GridLatentPairDataset)

    # Try 1: Full parallel mode (fastest)
    if use_parallel_loader:
        try:
            from ups.data.parallel_cache import build_parallel_latent_loader
            loader = build_parallel_latent_loader(
                dataset.base,
                dataset.encoder,
                dataset.coords,
                dataset.grid_shape,
                dataset.field_name,
                dataset.device,
                batch_size=batch_size,
                num_workers=num_workers,
                cache_dir=dataset.cache_dir,
                cache_dtype=dataset.cache_dtype,
                time_stride=dataset.time_stride,
                rollout_horizon=dataset.rollout_horizon,
                pin_memory=pin_memory,
                prefetch_factor=2,
                timeout=timeout,
            )
            # Test with one batch
            try:
                iter(loader).__next__()
                return loader, f"parallel ({num_workers} workers)"
            except Exception as e:
                print(f"⚠️  Parallel mode failed during test: {e}")
        except (AttributeError, TypeError) as e:
            print(f"⚠️  Parallel encoding setup failed: {e}")

    # Try 2: Single worker (slower but more stable)
    if num_workers > 0:
        try:
            print("   Falling back to single-worker mode...")
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=1,
                pin_memory=pin_memory,
                persistent_workers=False,
                collate_fn=lambda items: items,
                timeout=timeout if timeout > 0 else None,
            )
            # Test with one batch
            try:
                iter(loader).__next__()
                return loader, "single-worker"
            except Exception as e:
                print(f"⚠️  Single-worker mode failed during test: {e}")
        except Exception as e:
            print(f"⚠️  Single-worker setup failed: {e}")

    # Try 3: Main process only (slowest, most reliable)
    print("   Falling back to main-process-only mode (slowest but most reliable)...")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # No multiprocessing
        pin_memory=pin_memory,
        persistent_workers=False,
        collate_fn=lambda items: items,
    )
    return loader, "main-process-only"


def _iter_dataset(
    dataset: Dataset,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    use_parallel: bool = True,
    timeout: int = 0,
    watchdog_timeout: int = 120,
) -> None:
    """Iterate dataset to populate cache with robust fallback."""

    # Build loader with automatic fallback
    loader, mode = _build_robust_loader(
        dataset, batch_size, num_workers, pin_memory, use_parallel, timeout
    )
    print(f"📦 Using mode: {mode}")

    total_batches = len(loader)
    iterator = _maybe_tqdm(loader, total_batches, desc=f"encoding ({mode})")

    # Start watchdog
    watchdog = ProgressWatchdog(timeout_seconds=watchdog_timeout)
    watchdog.start()

    try:
        for i, batch in enumerate(iterator):
            watchdog.update(i)

            if isinstance(batch, list):
                for item in batch:
                    del item
            del batch
    finally:
        watchdog.stop()
```

#### 4. Add Detailed Diagnostic Logging

**File**: `scripts/precompute_latent_cache.py`
**Changes**: Add logging configuration and diagnostic output

**Location**: After imports, around line 25

```python
import logging

def setup_logging(verbose: bool = False):
    """Configure logging for diagnostics."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S'
    )
    return logging.getLogger(__name__)

logger = None  # Global logger, set in main()
```

**Location**: Line 330 (add CLI argument)

```python
parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging for diagnostics")
```

**Location**: Line 302 (in main(), setup logging)

```python
def main() -> None:
    # ... existing multiprocessing setup ...

    parser = argparse.ArgumentParser(...)
    # ... all arguments ...
    args = parser.parse_args()

    # NEW: Setup logging
    global logger
    logger = setup_logging(verbose=args.verbose)

    if args.verbose:
        logger.debug(f"Python: {sys.version}")
        logger.debug(f"PyTorch: {torch.__version__}")
        logger.debug(f"CUDA available: {torch.cuda.is_available()}")
        logger.debug(f"Device: {device}")
        logger.debug(f"Workers: {actual_num_workers}")
        logger.debug(f"Timeouts: DataLoader={args.dataloader_timeout}s, HDF5={args.hdf5_timeout}s, Watchdog={args.watchdog_timeout}s")
```

### Success Criteria

#### Automated Verification:
- [ ] Watchdog detects hang: `INJECT_HANG=1 python scripts/precompute_latent_cache.py ...` fails within 120s with "HANG DETECTED" message
- [ ] HDF5 timeout works: With `--hdf5-timeout 5` and slow/corrupted HDF5, fails within ~10s
- [ ] Fallback strategy works: With `--num-workers 8` on constrained system, automatically falls back to single-worker or main-process
- [ ] Verbose logging works: `--verbose` flag produces detailed diagnostic output

#### Manual Verification:
- [ ] Run on VastAI/Vultr with `--verbose` and verify diagnostic logs are helpful
- [ ] Simulate hang (kill worker process) and verify watchdog catches it within 2 minutes
- [ ] Verify fallback strategy produces useful error messages for each failed mode

**Implementation Note**: After completing this phase and all automated verification passes, test on VastAI/Vultr with real data before proceeding to Phase 3.

---

## Phase 3: Performance Optimization & Testing

### Overview

Optimize cache creation speed by 2-3x through batched writes and local storage copying for VastAI/Vultr. Add comprehensive test coverage to prevent regressions.

### Changes Required

#### 1. Implement Batched Cache Writes

**File**: `src/ups/data/parallel_cache.py`
**Changes**: Add batched cache writer class

**Location**: After imports, around line 30

```python
class BatchedCacheWriter:
    """Write cache files in batches for 2-3x faster I/O.

    Instead of writing sample_00000.pt, sample_00001.pt, ...
    Writes batch_00000.pt containing 32 samples, batch_00001.pt, ...

    Benefits:
    - 32x fewer file operations (open/close/rename)
    - Better disk I/O patterns (sequential writes)
    - Less filesystem metadata overhead
    """

    def __init__(
        self,
        cache_dir: Path,
        batch_size: int = 32,
        cache_dtype: Optional[torch.dtype] = torch.float16,
    ):
        self.cache_dir = Path(cache_dir)
        self.batch_size = batch_size
        self.cache_dtype = cache_dtype
        self.buffer: Dict[int, Dict[str, Any]] = {}
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def add_sample(self, idx: int, latent: torch.Tensor, params: Any, bc: Any) -> None:
        """Add sample to buffer, flush if batch complete."""
        to_store = latent.to(self.cache_dtype) if self.cache_dtype else latent
        self.buffer[idx] = {
            "latent": to_store.cpu(),
            "params": params,
            "bc": bc,
        }

        batch_idx = idx // self.batch_size
        if len(self.buffer) >= self.batch_size:
            self._flush_batch(batch_idx)

    def _flush_batch(self, batch_idx: int) -> None:
        """Write buffered samples to disk."""
        if not self.buffer:
            return

        batch_path = self.cache_dir / f"batch_{batch_idx:05d}.pt"
        tmp_path = self.cache_dir / f".tmp_batch_{batch_idx:05d}.pt"

        try:
            torch.save(self.buffer, tmp_path)
            tmp_path.replace(batch_path)
            self.buffer.clear()
        except (OSError, IOError) as e:
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError(f"Failed to write cache batch {batch_idx}: {e}") from e

    def flush(self) -> None:
        """Flush any remaining buffered samples."""
        if self.buffer:
            # Find batch index from first buffered sample
            first_idx = min(self.buffer.keys())
            batch_idx = first_idx // self.batch_size
            self._flush_batch(batch_idx)

    def read_sample(self, idx: int) -> Optional[Dict[str, Any]]:
        """Read sample from batched cache."""
        batch_idx = idx // self.batch_size
        batch_path = self.cache_dir / f"batch_{batch_idx:05d}.pt"

        if not batch_path.exists():
            return None

        try:
            batch_data = torch.load(batch_path, map_location="cpu")
            return batch_data.get(idx)
        except (RuntimeError, EOFError):
            batch_path.unlink(missing_ok=True)
            return None
```

**Note**: This is an optional optimization. For Phase 3, implement it as a **separate CLI flag** `--batched-cache` to allow A/B testing. Default to per-sample caching for backwards compatibility.

#### 2. Add Local Storage Copy Utility for VastAI/Vultr

**File**: `scripts/copy_data_to_local.sh` (NEW FILE)
**Changes**: Create shell script for copying data to local storage

```bash
#!/bin/bash
# Copy PDEBench data from network storage to local SSD on VastAI/Vultr instances

set -e

REMOTE_DATA_DIR="${1:-/root/data/pdebench}"
LOCAL_DATA_DIR="${2:-/workspace/data_local/pdebench}"

echo "📦 Copying PDEBench data to local storage..."
echo "   Remote: $REMOTE_DATA_DIR"
echo "   Local:  $LOCAL_DATA_DIR"

# Check if remote exists
if [ ! -d "$REMOTE_DATA_DIR" ]; then
    echo "❌ Remote data directory not found: $REMOTE_DATA_DIR"
    exit 1
fi

# Create local directory
mkdir -p "$LOCAL_DATA_DIR"

# Check available space
REQUIRED_GB=20
AVAILABLE_GB=$(df -BG "$LOCAL_DATA_DIR" | awk 'NR==2 {print $4}' | sed 's/G//')
if [ "$AVAILABLE_GB" -lt "$REQUIRED_GB" ]; then
    echo "⚠️  Warning: Low disk space (${AVAILABLE_GB}GB available, ${REQUIRED_GB}GB recommended)"
fi

# Copy with progress
rsync -avh --progress "$REMOTE_DATA_DIR/" "$LOCAL_DATA_DIR/"

echo "✅ Data copied successfully!"
echo "   Use --root $LOCAL_DATA_DIR when running precompute script"
```

**File**: `scripts/precompute_latent_cache.py`
**Changes**: Add auto-detection of VastAI/Vultr and prompt for local copy

**Location**: In main(), around line 370

```python
# NEW: Detect if running on VastAI/Vultr and data is on network storage
def is_network_storage(path: Path) -> bool:
    """Detect if path is on network-mounted filesystem."""
    try:
        import subprocess
        result = subprocess.run(
            ["df", "-T", str(path)],
            capture_output=True,
            text=True,
            timeout=5,
        )
        # Check for NFS, CIFS, or other network filesystems
        return any(fs in result.stdout for fs in ["nfs", "cifs", "smbfs", "fuse"])
    except Exception:
        return False

# In main(), after data_root is determined:
data_root = resolve_pdebench_root(data_cfg.get("root", "data/pdebench"))

if is_network_storage(data_root):
    logger.warning("⚠️  Data appears to be on network storage (slower, may cause hangs)")
    logger.warning("   Recommendation: Copy to local storage first:")
    logger.warning(f"   bash scripts/copy_data_to_local.sh {data_root} /workspace/data_local/pdebench")
    logger.warning(f"   Then run with: --root /workspace/data_local/pdebench")

    # Optional: Auto-copy if on VastAI/Vultr and local workspace exists
    if Path("/workspace").exists() and not args.skip_auto_copy:
        response = input("Auto-copy to local storage? (y/N): ")
        if response.lower() == 'y':
            local_path = Path("/workspace/data_local/pdebench")
            logger.info(f"Copying {data_root} to {local_path}...")
            import shutil
            shutil.copytree(data_root, local_path, dirs_exist_ok=True)
            data_cfg["root"] = str(local_path)
            logger.info("✅ Data copied to local storage")
```

**Location**: Add CLI argument (line 330)

```python
parser.add_argument("--skip-auto-copy", action="store_true", help="Skip auto-copy prompt for network storage")
```

#### 3. Add Unit Tests for Cache System

**File**: `tests/unit/test_cache_precompute.py` (NEW FILE)
**Changes**: Create comprehensive unit tests

```python
"""Unit tests for latent cache precomputation system."""

import tempfile
from pathlib import Path
import pytest
import torch
from unittest.mock import Mock, patch, MagicMock

from ups.data.parallel_cache import (
    RawFieldDataset,
    PreloadedCacheDataset,
    CollateFnWithEncoding,
    build_parallel_latent_loader,
    check_cache_complete,
    estimate_cache_size_mb,
    check_sufficient_ram,
)


class TestRawFieldDataset:
    """Test RawFieldDataset for parallel loading."""

    def test_returns_cached_sample(self):
        """Test that cached samples are loaded correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)

            # Create mock cache file
            cache_path = cache_dir / "sample_00000.pt"
            torch.save({
                "latent": torch.randn(10, 16, 32),
                "params": {"nu": torch.tensor(0.01)},
                "bc": None,
            }, cache_path)

            # Create mock dataset
            mock_base = Mock()
            mock_base.__len__ = Mock(return_value=10)

            dataset = RawFieldDataset(
                mock_base,
                field_name="u",
                cache_dir=cache_dir,
            )

            sample = dataset[0]
            assert sample["cached"] is True
            assert sample["latent"].shape == (10, 16, 32)
            assert "params" in sample

    def test_returns_raw_fields_when_not_cached(self):
        """Test that raw fields are returned when cache miss."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)

            # Create mock dataset
            mock_base = Mock()
            mock_base.__len__ = Mock(return_value=10)
            mock_base.__getitem__ = Mock(return_value={
                "fields": torch.randn(10, 64, 64, 1),
                "params": {"nu": torch.tensor(0.01)},
                "bc": None,
            })

            dataset = RawFieldDataset(
                mock_base,
                field_name="u",
                cache_dir=cache_dir,
            )

            sample = dataset[0]
            assert sample["cached"] is False
            assert "fields" in sample
            assert sample["fields"].shape == (10, 64, 64, 1)


class TestPreloadedCacheDataset:
    """Test PreloadedCacheDataset for RAM caching."""

    def test_preloads_all_cache_files(self):
        """Test that all cache files are loaded into RAM."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            num_samples = 5

            # Create mock cache files
            for idx in range(num_samples):
                cache_path = cache_dir / f"sample_{idx:05d}.pt"
                torch.save({
                    "latent": torch.randn(10, 16, 32),
                    "params": {"nu": torch.tensor(0.01)},
                    "bc": None,
                }, cache_path)

            dataset = PreloadedCacheDataset(
                cache_dir=cache_dir,
                num_samples=num_samples,
            )

            assert len(dataset) == num_samples
            for idx in range(num_samples):
                sample = dataset[idx]
                assert sample.z0.shape[0] > 0  # Has time steps
                assert sample.z1.shape[0] > 0

    def test_fails_on_incomplete_cache(self):
        """Test that incomplete cache raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            num_samples = 5

            # Create only 3 out of 5 cache files
            for idx in range(3):
                cache_path = cache_dir / f"sample_{idx:05d}.pt"
                torch.save({
                    "latent": torch.randn(10, 16, 32),
                    "params": None,
                    "bc": None,
                }, cache_path)

            with pytest.raises(ValueError, match="Cache incomplete"):
                PreloadedCacheDataset(cache_dir=cache_dir, num_samples=num_samples)


class TestCacheUtilities:
    """Test cache utility functions."""

    def test_check_cache_complete(self):
        """Test cache completeness check."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            num_samples = 10

            # Empty cache
            is_complete, count = check_cache_complete(cache_dir, num_samples)
            assert not is_complete
            assert count == 0

            # Partial cache
            for idx in range(5):
                (cache_dir / f"sample_{idx:05d}.pt").touch()
            is_complete, count = check_cache_complete(cache_dir, num_samples)
            assert not is_complete
            assert count == 5

            # Complete cache
            for idx in range(5, num_samples):
                (cache_dir / f"sample_{idx:05d}.pt").touch()
            is_complete, count = check_cache_complete(cache_dir, num_samples)
            assert is_complete
            assert count == num_samples

    def test_estimate_cache_size_mb(self):
        """Test cache size estimation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)

            # Create sample files with known size
            for idx in range(10):
                path = cache_dir / f"sample_{idx:05d}.pt"
                path.write_bytes(b"0" * 1024 * 1024)  # 1MB each

            size_mb = estimate_cache_size_mb(cache_dir, num_samples=10)
            assert 9 < size_mb < 11  # ~10MB total

    def test_check_sufficient_ram(self):
        """Test RAM sufficiency check."""
        # Should have enough RAM for 100MB
        assert check_sufficient_ram(100) is True

        # Should not have enough RAM for 1TB
        assert check_sufficient_ram(1024 * 1024 * 1024) is False


class TestTimeoutMechanisms:
    """Test timeout and error handling."""

    @patch('torch.utils.data.DataLoader')
    def test_dataloader_timeout_parameter(self, mock_dataloader):
        """Test that timeout is passed to DataLoader."""
        from ups.data.pdebench import PDEBenchDataset, PDEBenchConfig
        from ups.io.enc_grid import GridEncoder, GridEncoderConfig

        # Mock components
        mock_dataset = Mock(spec=PDEBenchDataset)
        mock_encoder = Mock(spec=GridEncoder)
        coords = torch.randn(1, 64, 2)

        with tempfile.TemporaryDirectory() as tmpdir:
            loader = build_parallel_latent_loader(
                dataset=mock_dataset,
                encoder=mock_encoder,
                coords=coords,
                grid_shape=(8, 8),
                field_name="u",
                device=torch.device("cpu"),
                batch_size=4,
                num_workers=2,
                cache_dir=Path(tmpdir),
                timeout=120,  # Should be passed through
            )

            # Verify DataLoader was called with timeout
            call_kwargs = mock_dataloader.call_args[1]
            assert call_kwargs.get("timeout") == 120


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

#### 4. Add Integration Tests

**File**: `tests/integration/test_parallel_cache.py` (NEW FILE)
**Changes**: Create end-to-end integration tests

```python
"""Integration tests for parallel cache precomputation."""

import tempfile
from pathlib import Path
import pytest
import torch

from ups.data.pdebench import PDEBenchDataset, PDEBenchConfig
from ups.data.latent_pairs import GridLatentPairDataset, _build_pdebench_dataset
from ups.data.parallel_cache import build_parallel_latent_loader


@pytest.fixture
def mock_pdebench_data(tmp_path):
    """Create mock PDEBench HDF5 data."""
    import h5py

    data_dir = tmp_path / "pdebench"
    data_dir.mkdir()

    # Create mock HDF5 file
    h5_path = data_dir / "burgers1d_train.h5"
    with h5py.File(h5_path, "w") as f:
        # Create mock burgers data: (num_samples, time_steps, spatial_points)
        f.create_dataset("u", data=torch.randn(10, 20, 64).numpy())
        f.create_dataset("nu", data=torch.full((10,), 0.01).numpy())

    return data_dir


@pytest.mark.integration
class TestParallelCacheIntegration:
    """End-to-end tests for parallel cache system."""

    def test_precompute_cache_end_to_end(self, mock_pdebench_data):
        """Test complete cache precomputation workflow."""
        cache_dir = mock_pdebench_data / "cache"
        cache_dir.mkdir()

        # Build dataset and encoder
        cfg = {
            "task": "burgers1d",
            "split": "train",
            "root": str(mock_pdebench_data),
            "normalize": False,
            "latent_dim": 32,
            "latent_len": 16,
        }

        dataset, encoder, grid_shape, field_name = _build_pdebench_dataset(cfg)
        coords = torch.randn(1, 64, 2)

        # Create latent pair dataset with caching
        latent_dataset = GridLatentPairDataset(
            base=dataset,
            encoder=encoder,
            coords=coords,
            grid_shape=grid_shape,
            field_name=field_name,
            device=torch.device("cpu"),
            cache_dir=cache_dir / "burgers1d_train",
        )

        # Access samples to populate cache
        for i in range(min(5, len(latent_dataset))):
            _ = latent_dataset[i]

        # Verify cache files exist
        cache_files = list((cache_dir / "burgers1d_train").glob("sample_*.pt"))
        assert len(cache_files) >= 5

        # Verify cache can be loaded
        for cache_file in cache_files:
            data = torch.load(cache_file, map_location="cpu")
            assert "latent" in data
            assert data["latent"].dim() == 3  # (time, tokens, latent_dim)

    def test_parallel_loader_with_cache(self, mock_pdebench_data):
        """Test parallel loader with cache directory."""
        cache_dir = mock_pdebench_data / "cache"
        cache_dir.mkdir()

        cfg = {
            "task": "burgers1d",
            "split": "train",
            "root": str(mock_pdebench_data),
            "normalize": False,
            "latent_dim": 32,
            "latent_len": 16,
        }

        dataset, encoder, grid_shape, field_name = _build_pdebench_dataset(cfg)
        coords = torch.randn(1, 64, 2)

        # Build parallel loader
        loader = build_parallel_latent_loader(
            dataset=dataset,
            encoder=encoder,
            coords=coords,
            grid_shape=grid_shape,
            field_name=field_name,
            device=torch.device("cpu"),
            batch_size=2,
            num_workers=0,  # Use 0 for deterministic testing
            cache_dir=cache_dir / "burgers1d_train",
            timeout=30,
        )

        # Iterate and verify batches
        batch_count = 0
        for batch in loader:
            assert batch.z0.dim() == 3  # (batch, time, latent_dim)
            assert batch.z1.dim() == 3
            batch_count += 1
            if batch_count >= 3:
                break

        assert batch_count >= 3


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
```

#### 5. Update Documentation

**File**: `CLAUDE.md`
**Changes**: Document new timeout parameters and local storage workflow

**Location**: Around line 50 (in "Key Commands" section)

```markdown
### Data Management

```bash
# Precompute latent cache (with timeouts)
python scripts/precompute_latent_cache.py \
  --config configs/cache_precompute_defaults.yaml \
  --tasks burgers1d --splits train \
  --dataloader-timeout 120 \
  --hdf5-timeout 60 \
  --watchdog-timeout 120

# Copy data to local storage on VastAI/Vultr (10x faster I/O)
bash scripts/copy_data_to_local.sh /root/data/pdebench /workspace/data_local/pdebench

# Then precompute with local data
python scripts/precompute_latent_cache.py \
  --config configs/cache_precompute_defaults.yaml \
  --root /workspace/data_local/pdebench \
  --tasks burgers1d --splits train
```
```

**File**: `docs/runbook.md` or create `docs/cache_troubleshooting.md`
**Changes**: Add troubleshooting guide

```markdown
# Cache Precomputation Troubleshooting

## Symptoms: Cache precompute hangs indefinitely

**Diagnosis**:
1. Check if data is on network storage: `df -T data/pdebench`
2. Check available disk space: `df -h`
3. Check RAM usage: `free -h`

**Solutions**:
1. **Use timeouts** (prevents indefinite hangs):
   ```bash
   python scripts/precompute_latent_cache.py \
     --dataloader-timeout 120 \
     --hdf5-timeout 60 \
     --watchdog-timeout 120
   ```

2. **Copy to local storage** (for VastAI/Vultr):
   ```bash
   bash scripts/copy_data_to_local.sh /root/data/pdebench /workspace/data_local/pdebench
   python scripts/precompute_latent_cache.py --root /workspace/data_local/pdebench ...
   ```

3. **Reduce parallelism** (if memory constrained):
   ```bash
   python scripts/precompute_latent_cache.py --num-workers 1 ...
   ```

4. **Disable multiprocessing** (slowest but most reliable):
   ```bash
   python scripts/precompute_latent_cache.py --num-workers 0 ...
   ```

## Symptoms: DataLoader timeout errors

**Causes**: Workers hanging during HDF5 load or encoding

**Solutions**:
- Increase timeout: `--dataloader-timeout 300`
- Use verbose logging: `--verbose`
- Check watchdog output for exact hang location

## Symptoms: HDF5 timeout errors

**Causes**: Network storage latency or corrupted files

**Solutions**:
- Copy data to local storage (see above)
- Increase timeout: `--hdf5-timeout 120`
- Verify HDF5 files: `h5ls data/pdebench/burgers1d_train.h5`
```

### Success Criteria

#### Automated Verification:
- [ ] All unit tests pass: `pytest tests/unit/test_cache_precompute.py -v`
- [ ] All integration tests pass: `pytest tests/integration/test_parallel_cache.py -v -m integration`
- [ ] Local storage copy script works: `bash scripts/copy_data_to_local.sh /tmp/source /tmp/dest` completes successfully
- [ ] Batched cache writing works: Cache created with `--batched-cache` has `batch_*.pt` files
- [ ] Documentation is complete: `CLAUDE.md` and troubleshooting guide updated

#### Manual Verification:
- [ ] Run precompute on VastAI/Vultr with local storage copy - verify 2-3x faster than network storage
- [ ] Run full test suite: `pytest tests/ -v` - all tests pass
- [ ] Verify cache precompute completes end-to-end with all 3 phases enabled
- [ ] Re-enable cache in training config and verify epoch 1 completes

**Implementation Note**: After completing this phase, re-enable cache in production config (`configs/train_pdebench_2task_baseline_ddp.yaml:79` change `cache_dir: null` → `cache_dir: data/latent_cache`) and run full training validation.

---

## Testing Strategy

### Unit Tests

**Coverage targets**:
- `test_cache_precompute.py`: RawFieldDataset, PreloadedCacheDataset, cache utilities, timeout mechanisms
- `test_timeout_wrappers.py`: HDF5 timeout context manager, watchdog class

**Key edge cases**:
- Corrupted cache files (EOFError, RuntimeError)
- Incomplete cache (missing samples)
- Insufficient disk space
- Timeout scenarios (simulated slow I/O)
- Worker failures and fallback

### Integration Tests

**Coverage targets**:
- `test_parallel_cache.py`: End-to-end cache precomputation, parallel loader with caching
- `test_cache_training.py`: Training with cached data (full epoch)

**Key scenarios**:
- Fresh cache creation
- Loading from existing cache
- Mixed cached/uncached samples
- Multi-task cache loading

### Manual Testing Steps

1. **Local testing**:
   ```bash
   # Test precompute with all timeouts
   python scripts/precompute_latent_cache.py \
     --config configs/cache_precompute_defaults.yaml \
     --tasks burgers1d --splits train --limit 100 \
     --dataloader-timeout 120 --hdf5-timeout 60 --watchdog-timeout 120 \
     --verbose
   ```

2. **VastAI/Vultr testing**:
   ```bash
   # Test local storage copy
   bash scripts/copy_data_to_local.sh /root/data/pdebench /workspace/data_local/pdebench

   # Test precompute with local data
   python scripts/precompute_latent_cache.py \
     --root /workspace/data_local/pdebench \
     --tasks burgers1d advection1d --splits train val \
     --verbose
   ```

3. **Hang simulation**:
   ```bash
   # Test watchdog detection (inject hang via env var or manual worker kill)
   INJECT_HANG=1 python scripts/precompute_latent_cache.py ...
   # Should fail within 120s with actionable error
   ```

4. **Training validation**:
   ```bash
   # Re-enable cache and run training
   python scripts/train.py \
     --config configs/train_pdebench_2task_baseline_ddp.yaml \
     --stage operator --epochs 1
   # Should complete epoch 1 with cached data
   ```

## Performance Considerations

### Expected Improvements

**Phase 1**: No performance change (adds safety, ~1-2% overhead from checks)

**Phase 2**: 0-5% overhead from watchdog monitoring (negligible)

**Phase 3**:
- **Batched cache writes**: 2-3x faster cache creation (fewer file operations)
- **Local storage**: 10-100x faster HDF5 reads on VastAI/Vultr (network → SSD)
- **Combined**: Cache precompute should complete in 3-5 minutes (vs 10-20 minutes or indefinite hang)

### Memory Usage

- Watchdog thread: ~1MB (negligible)
- HDF5 timeout: No additional memory
- Batched cache writer: ~50-200MB buffer (configurable batch size)
- Local storage copy: Uses disk space (20-50GB for typical datasets)

## Migration Notes

### Backwards Compatibility

- All new CLI arguments are **optional** with sensible defaults
- Existing scripts continue to work without modifications
- Cache file format unchanged (per-sample `.pt` files)
- Batched cache is **opt-in** via `--batched-cache` flag

### Config Changes Required

**File**: `configs/train_pdebench_2task_baseline_ddp.yaml`

After Phase 3 validation, re-enable cache:

```yaml
# Line 79: BEFORE
cache_dir: null  # TEMPORARY: Disabled due to precompute hang

# Line 79: AFTER (Phase 3 complete)
cache_dir: data/latent_cache  # Re-enabled with timeout safeguards
```

### VastAI/Vultr Workflow Changes

**Before** (hangs indefinitely):
```bash
python scripts/vast_launch.py launch --config configs/train_pdebench_2task_baseline_ddp.yaml
# Training uses slow on-demand encoding
```

**After** (Phase 3 complete):
```bash
# Option 1: Auto-copy in onstart.sh
cat > .vast/onstart.sh << 'EOF'
#!/bin/bash
bash scripts/copy_data_to_local.sh /root/data/pdebench /workspace/data_local/pdebench
python scripts/precompute_latent_cache.py \
  --root /workspace/data_local/pdebench \
  --tasks burgers1d advection1d --splits train val \
  --dataloader-timeout 120 --hdf5-timeout 60
python scripts/train.py --config configs/train_pdebench_2task_baseline_ddp.yaml
EOF

# Option 2: Precompute separately, then train
vastai ssh <instance_id>
bash scripts/copy_data_to_local.sh /root/data/pdebench /workspace/data_local/pdebench
python scripts/precompute_latent_cache.py --root /workspace/data_local/pdebench ...
exit
python scripts/vast_launch.py launch --config configs/train_pdebench_2task_baseline_ddp.yaml
```

## References

- Original research: `thoughts/shared/research/2025-11-14-precompute-cache-hang-investigation.md`
- Related research: `thoughts/shared/research/2025-11-13-massive-training-speed-optimization.md`
- DDP cache fixes: Commits `291041c`, `2595dca`
- Cache disabled: Commit `b2d8929`
