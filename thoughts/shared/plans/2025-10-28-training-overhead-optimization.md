# Training Overhead Optimization Implementation Plan

## Overview

Reduce VastAI training startup overhead from 14-25 minutes to 6-9 minutes by enabling existing optimization features that are currently disabled or unused in production. The codebase already has GPU cache precomputation, parallel encoding, and RAM preloading capabilities implemented but not activated.

## Current State Analysis

### Startup Overhead Breakdown (Current: 14-25 minutes)

| Phase | Time | Status |
|-------|------|--------|
| Git clone & dependencies | 3-4 min | Unavoidable |
| Data downloads (sequential) | 1-3 min | **Optimizable** |
| **Latent cache precompute** | **5-15 min** | **LARGEST BOTTLENECK** |
| Training initialization | 1-2 min | Acceptable |

### Key Discoveries

**Problem 1: GPU Cache Precomputation Disabled**
- Location: `scripts/vast_launch.py:192`
- Current: Uses `--device cpu --no-parallel --num-workers 0`
- Impact: 5-15 min cache generation time
- Solution exists: `precompute_latent_cache.py` supports GPU + parallel mode

**Problem 2: PreloadedCacheDataset Never Used**
- Location: `src/ups/data/parallel_cache.py:90-167`
- Implementation exists with 90%+ GPU utilization capability
- Helper functions exist but never called:
  - `check_cache_complete()` (line 313)
  - `estimate_cache_size_mb()` (line 327)
  - `check_sufficient_ram()` (line 365)
- `build_latent_pair_loader()` always uses `GridLatentPairDataset` (disk I/O bound)

**Problem 3: Cache Wiped on Every Run**
- Location: `scripts/vast_launch.py:183`
- Current: `rm -rf data/latent_cache` on every launch
- No config hash checking for cache validity
- Forces full regeneration even when config unchanged

**Problem 4: Sequential Data Downloads**
- Location: `scripts/vast_launch.py:167-178`
- Downloads train/val/test files sequentially
- No parallelization despite 3 independent downloads

## Desired End State

### Target Metrics
- **Startup time**: 6-9 minutes (60% reduction from 14-25 min baseline)
- **Cache precompute**: 2-5 minutes (70% reduction from 5-15 min)
- **GPU utilization**: 90%+ during training (up from 60-70%)
- **Cost savings**: ~$0.30-0.50 per run at $1.89/hr

### Verification
1. Launch 3 training runs with same config
2. First run: Full cache generation (2-5 min)
3. Second run: Cache reused (skip generation, <30s overhead)
4. Third run with different `latent.dim`: Cache regenerated
5. Monitor GPU utilization with `nvidia-smi dmon` (expect 90%+)
6. Measure total startup time from instance launch to training start

## What We're NOT Doing

- **NOT** implementing Docker image caching (requires CI/CD setup, deferred to Phase 2)
- **NOT** implementing persistent VastAI instance pools (complex state management, deferred to Phase 3)
- **NOT** implementing remote cache service on B2 (overkill for current usage)
- **NOT** modifying training loop or model architecture
- **NOT** changing validation scripts (overhead already minimal at 4-9s)

## Implementation Approach

### Strategy
Focus on **activating existing optimization features** rather than building new ones. The codebase has most optimizations already implemented but disabled. This is a configuration and integration effort, not a ground-up development.

### Risk Mitigation
1. Test each optimization independently before combining
2. Add feature flags to rollback if issues arise
3. Preserve legacy mode as fallback option
4. Monitor GPU memory usage on first runs

## Phase 1: Enable GPU Cache Precomputation

### Overview
Change cache precomputation from CPU to GPU with parallel encoding. This is the single largest bottleneck (5-15 min → 2-5 min).

### Changes Required

#### 1. Update VastAI Launch Script
**File**: `scripts/vast_launch.py`
**Changes**: Line 192 cache precomputation command

```python
# BEFORE (line 192):
PYTHONPATH=src python scripts/precompute_latent_cache.py \
  --config {config_for_script} \
  --tasks burgers1d \
  --splits train val \
  --root data/pdebench \
  --cache-dir data/latent_cache \
  --cache-dtype float16 \
  --device cpu \              # ❌ Using CPU
  --batch-size 4 \
  --num-workers 0 \           # ❌ No parallelization
  --pin-memory \
  --no-parallel               # ❌ Parallel mode disabled

# AFTER:
PYTHONPATH=src python scripts/precompute_latent_cache.py \
  --config {config_for_script} \
  --tasks burgers1d \
  --splits train val \
  --root data/pdebench \
  --cache-dir data/latent_cache \
  --cache-dtype float16 \
  --device cuda \             # ✅ Use GPU
  --batch-size 16 \           # ✅ Larger batch (GPU can handle it)
  --num-workers 4 \           # ✅ Parallel workers
  --pin-memory \
  --parallel                  # ✅ Parallel mode enabled (default, but explicit)
```

**Rationale**:
- GPU encoding is 10-20× faster than CPU
- Parallel mode with 4 workers is 4-8× faster than legacy mode
- Combined: 40-160× speedup in cache generation

**Edge Cases**:
- If GPU OOM: Fallback to `--batch-size 8` or `--device cpu`
- If CUDA not available: Script already detects and falls back to CPU

### Success Criteria

#### Automated Verification
- [x] VastAI instance launches successfully: `python scripts/vast_launch.py launch --config configs/train_burgers_32dim.yaml --dry-run`
- [x] Onstart script generated without errors: `cat .vast/onstart.sh`
- [ ] Cache precomputation completes: Check instance logs for "✓ Latent precomputation finished"
- [ ] Training starts successfully: Check WandB for new run

#### Manual Verification
- [ ] Cache generation time: 2-5 minutes (down from 5-15 min)
- [ ] GPU utilization during cache gen: 80%+ (check with `nvidia-smi dmon`)
- [ ] Cache files created: `data/latent_cache/burgers1d_train/*.pt` exist
- [ ] No CUDA errors in logs
- [ ] Training runs normally after cache generation

**Implementation Note**: After completing automated verification and seeing cache generation time drop to 2-5 minutes, proceed to Phase 2.

---

## Phase 2: Implement Parallel Data Downloads

### Overview
Download train/val/test data files in parallel instead of sequentially to save 1-2 minutes during startup.

### Changes Required

#### 1. Parallelize rclone Downloads
**File**: `scripts/vast_launch.py`
**Changes**: Lines 167-178 data download section

```bash
# BEFORE (sequential):
if [ ! -f data/pdebench/burgers1d_train_000.h5 ]; then
  rclone copy B2TRAIN:pdebench/full/burgers1d/burgers1d_train_000.h5 data/pdebench/ --progress || exit 1
fi
if [ ! -f data/pdebench/burgers1d_val.h5 ] && [ ! -f data/pdebench/burgers1d_valid.h5 ]; then
  rclone copy B2TRAIN:PDEbench/pdebench/burgers1d_full_v1/burgers1d_val.h5 data/pdebench/ --progress || true
fi
# ... etc

# AFTER (parallel):
# Download all files in parallel (backgrounded)
if [ ! -f data/pdebench/burgers1d_train_000.h5 ]; then
  rclone copy B2TRAIN:pdebench/full/burgers1d/burgers1d_train_000.h5 data/pdebench/ --progress &
fi
if [ ! -f data/pdebench/burgers1d_val.h5 ] && [ ! -f data/pdebench/burgers1d_valid.h5 ]; then
  rclone copy B2TRAIN:PDEbench/pdebench/burgers1d_full_v1/burgers1d_val.h5 data/pdebench/ --progress &
fi
if [ ! -f data/pdebench/burgers1d_test.h5 ]; then
  rclone copy B2TRAIN:PDEbench/pdebench/burgers1d_full_v1/burgers1d_test.h5 data/pdebench/ --progress &
fi

# Wait for all downloads to complete
wait

# Continue with symlinking after downloads finish
ln -sf burgers1d_train_000.h5 data/pdebench/burgers1d_train.h5 || true
if [ -f data/pdebench/burgers1d_val.h5 ] && [ ! -f data/pdebench/burgers1d_valid.h5 ]; then
  mv -f data/pdebench/burgers1d_val.h5 data/pdebench/burgers1d_valid.h5 || true
fi
ln -sf burgers1d_valid.h5 data/pdebench/burgers1d_val.h5 || true
```

**Rationale**:
- Three independent downloads can run concurrently
- Network bandwidth is rarely the bottleneck (B2 S3 has good throughput)
- `wait` ensures all downloads complete before proceeding

**Edge Cases**:
- If a download fails: The `&` backgrounds the process, so errors won't halt the script
- The `|| true` already handles download failures gracefully
- Training will fail later if required files are missing (acceptable)

### Success Criteria

#### Automated Verification
- [x] Onstart script syntax valid: `bash -n .vast/onstart.sh`
- [ ] Downloads complete successfully: Check instance logs for "✓" after rclone commands
- [ ] All required files present: `ls data/pdebench/burgers1d_*.h5`
- [ ] Symlinks created correctly: `ls -la data/pdebench/burgers1d_train.h5`

#### Manual Verification
- [ ] Download time reduced: 60-120s (down from 100-200s)
- [ ] All three rclone processes run concurrently (check with `ps aux | grep rclone`)
- [ ] No file corruption: Files are valid HDF5 (check with `h5ls`)
- [ ] Training proceeds normally

**Implementation Note**: After downloads complete successfully in parallel, proceed to Phase 3.

---

## Phase 3: Persistent Cache with Config Hashing

### Overview
Stop wiping the cache on every run. Instead, compute a hash of cache-relevant config parameters and regenerate only when the config changes. This saves 5-15 minutes on subsequent runs with unchanged configs.

### Changes Required

#### 1. Add Cache Metadata and Hashing
**File**: `scripts/precompute_latent_cache.py`
**Changes**: Add hash computation and metadata storage

```python
# Add at top of file (after imports)
import hashlib

def compute_cache_hash(cfg: Dict[str, Any]) -> str:
    """Compute hash of config parameters that affect cache validity.

    Cache is invalidated when any of these change:
    - latent.dim (changes latent space dimensionality)
    - latent.tokens (changes latent sequence length)
    - encoder type/architecture
    - data.task (different PDE)
    - data.root (different data source)
    """
    cache_keys = {
        "latent_dim": cfg.get("latent", {}).get("dim"),
        "latent_tokens": cfg.get("latent", {}).get("tokens"),
        "encoder_patch_size": cfg.get("encoder", {}).get("patch_size"),
        "data_task": cfg.get("data", {}).get("task"),
        "data_root": cfg.get("data", {}).get("root"),
    }
    # Sort keys for deterministic hash
    serialized = json.dumps(cache_keys, sort_keys=True)
    return hashlib.sha256(serialized.encode()).hexdigest()[:12]

def save_cache_metadata(cache_dir: Path, cfg: Dict[str, Any]) -> None:
    """Save cache metadata for validation on subsequent runs."""
    metadata = {
        "config_hash": compute_cache_hash(cfg),
        "generated_at": time.time(),
        "latent_dim": cfg.get("latent", {}).get("dim"),
        "latent_tokens": cfg.get("latent", {}).get("tokens"),
        "data_task": cfg.get("data", {}).get("task"),
    }
    metadata_file = cache_dir / ".cache_metadata.json"
    metadata_file.write_text(json.dumps(metadata, indent=2))
    print(f"✓ Saved cache metadata: hash={metadata['config_hash']}")

def check_cache_valid(cache_dir: Path, cfg: Dict[str, Any]) -> bool:
    """Check if existing cache matches current config."""
    metadata_file = cache_dir / ".cache_metadata.json"
    if not metadata_file.exists():
        return False

    try:
        metadata = json.loads(metadata_file.read_text())
        current_hash = compute_cache_hash(cfg)
        is_valid = metadata.get("config_hash") == current_hash

        if is_valid:
            age_hours = (time.time() - metadata.get("generated_at", 0)) / 3600
            print(f"✓ Cache valid (hash={current_hash}, age={age_hours:.1f}h)")
        else:
            print(f"⚠ Cache invalid (hash mismatch: {metadata.get('config_hash')} != {current_hash})")

        return is_valid
    except (json.JSONDecodeError, KeyError):
        # Corrupted metadata, regenerate cache
        return False
```

**Location to call**: In `main()` function, before cache generation loop (around line 230):

```python
def main() -> None:
    # ... existing argument parsing ...

    # NEW: Check if cache is valid
    if cache_root.exists() and not args.overwrite:
        if check_cache_valid(cache_root, cfg):
            print("✅ Cache is valid and up-to-date, skipping regeneration")
            print("   Use --overwrite to force regeneration")
            return

    # ... existing cache generation loop ...

    # NEW: Save metadata after cache generation completes
    save_cache_metadata(cache_root, cfg)
```

#### 2. Update VastAI Launch Script to Preserve Cache
**File**: `scripts/vast_launch.py`
**Changes**: Line 183 cache cleanup logic

```bash
# BEFORE (line 183):
rm -rf data/latent_cache checkpoints/scale || true
mkdir -p data/latent_cache checkpoints/scale

# AFTER:
# Only clear checkpoints, preserve cache (will be validated by precompute script)
rm -rf checkpoints || true
mkdir -p data/latent_cache checkpoints
```

**Rationale**:
- Cache validation happens in precompute_latent_cache.py (via config hash check)
- If cache is valid, precompute script exits early (saves 5-15 min)
- If cache is invalid, precompute script regenerates automatically
- Checkpoints are still cleared (training always starts fresh)

#### 3. Add Cache Status to Validation Script
**File**: `scripts/validate_config.py`
**Changes**: Add cache validation check (optional but helpful)

```python
# Add new validation function after validate_checkpoints() (around line 344):

def validate_cache(cfg: Dict) -> List[Tuple[str, bool, str]]:
    """Validate latent cache status."""
    checks = []

    cache_dir = Path(cfg.get("training", {}).get("latent_cache_dir", "data/latent_cache"))

    if not cache_dir.exists():
        checks.append((
            "cache directory exists",
            False,
            f"Cache dir {cache_dir} not found (will be created on first run)"
        ))
        return checks

    # Check for metadata file
    metadata_file = cache_dir / ".cache_metadata.json"
    if not metadata_file.exists():
        checks.append((
            "cache metadata exists",
            False,
            "No metadata file found (cache will be regenerated)"
        ))
        return checks

    # Load and validate metadata
    try:
        import json
        metadata = json.loads(metadata_file.read_text())
        checks.append((
            "cache metadata valid",
            True,
            f"hash={metadata.get('config_hash', 'unknown')}"
        ))

        # Check if hash matches current config
        # (Import compute_cache_hash from precompute_latent_cache.py)
        # This is optional validation, not critical path

    except Exception as e:
        checks.append((
            "cache metadata parseable",
            False,
            f"Corrupted metadata: {e}"
        ))

    return checks
```

And add to `main()` validation flow (around line 410):

```python
    # After eval_checks
    cache_checks = validate_cache(cfg)
    print_results("Latent Cache", cache_checks)
    all_checks.extend(cache_checks)
```

### Success Criteria

#### Automated Verification
- [x] Config validation passes: `python scripts/validate_config.py configs/train_burgers_32dim.yaml`
- [ ] First run generates cache: Launch instance, check logs for "✓ Saved cache metadata"
- [ ] Second run reuses cache: Launch with same config, check logs for "✓ Cache valid" and early exit
- [ ] Hash mismatch detected: Change `latent.dim`, verify cache regenerates
- [ ] Metadata file created: `ls data/latent_cache/.cache_metadata.json`

#### Manual Verification
- [ ] First run time: Normal (14-25 min with cache generation)
- [ ] Second run time: Reduced by 5-15 min (cache skipped)
- [ ] Config change triggers regeneration: Modify latent.dim, verify cache rebuilt
- [ ] Training succeeds with cached data: No errors, normal metrics
- [ ] Cache persists across runs: Directory not wiped between launches

**Implementation Note**: After cache persistence working correctly, Phase 1 complete. Review metrics before proceeding to Phase 2.

---

## Phase 2: Enable PreloadedCacheDataset

### Overview
Once cache is reliably precomputed and persisted, eliminate disk I/O bottleneck during training by preloading cache into RAM. This improves GPU utilization from ~60-70% to 90%+ and reduces epoch time by 20-40%.

### Changes Required

#### 1. Add Cache Completeness Check to Data Loader Builder
**File**: `src/ups/data/latent_pairs.py`
**Changes**: Modify `build_latent_pair_loader()` function (around line 588)

```python
def build_latent_pair_loader(cfg: Dict[str, Any]) -> DataLoader:
    """Construct a DataLoader that yields latent pairs from data config."""

    # ... existing config extraction (lines 591-607) ...

    # NEW: Import helper functions
    from ups.data.parallel_cache import (
        check_cache_complete,
        estimate_cache_size_mb,
        check_sufficient_ram,
        PreloadedCacheDataset,
    )

    # ... existing loader_kwargs setup (lines 609-618) ...

    # NEW: Check if we can use PreloadedCacheDataset
    use_preloaded = False
    if cache_root and cache_root.exists():
        # Check each task's cache directory
        tasks_to_check = data_cfg.get("task")
        if not isinstance(tasks_to_check, (list, tuple)):
            tasks_to_check = [tasks_to_check] if tasks_to_check else []

        for task in tasks_to_check:
            ds_cache = cache_root / f"{task}_{split_name}"
            if ds_cache.exists():
                # Estimate dataset size (need to build dataset temporarily)
                temp_cfg = {**data_cfg, "task": task, "latent_dim": latent_cfg.get("dim", 32), "latent_len": latent_cfg.get("tokens", 16)}
                temp_dataset, _, _, _ = _build_pdebench_dataset(temp_cfg)
                num_samples = len(temp_dataset)

                # Check cache completeness
                cache_complete, num_cached = check_cache_complete(ds_cache, num_samples)

                if cache_complete:
                    # Check cache size and RAM availability
                    cache_size_mb = estimate_cache_size_mb(ds_cache, num_samples=min(10, num_samples))
                    has_sufficient_ram = check_sufficient_ram(cache_size_mb)

                    if has_sufficient_ram:
                        print(f"✅ Using PreloadedCacheDataset for {task}_{split_name}")
                        print(f"   Cache: {num_cached} samples, ~{cache_size_mb:.0f} MB")
                        use_preloaded = True
                    else:
                        print(f"⚠️  Insufficient RAM for PreloadedCacheDataset ({cache_size_mb:.0f} MB required)")
                        print(f"   Falling back to disk I/O mode (slower but memory-efficient)")
                        use_preloaded = False
                else:
                    print(f"⚠️  Cache incomplete for {task}_{split_name} ({num_cached}/{num_samples} samples)")
                    print(f"   Falling back to on-demand encoding (will populate cache)")
                    use_preloaded = False

    # ... rest of existing function ...

    # MODIFY: When building GridLatentPairDataset, conditionally use PreloadedCacheDataset
    if tasks and isinstance(tasks, str):
        # Single task case (around line 651)
        dataset, encoder, grid_shape, field_name = _build_pdebench_dataset(
            {**data_cfg, "latent_dim": latent_cfg.get("dim", 32), "latent_len": latent_cfg.get("tokens", 16)}
        )
        encoder = encoder.to(device)
        coords = make_grid_coords(grid_shape, device)
        ds_cache = cache_root / f"{tasks}_{split_name}" if cache_root and isinstance(tasks, str) else cache_root

        # NEW: Use PreloadedCacheDataset if cache is complete
        if use_preloaded and ds_cache and ds_cache.exists():
            latent_dataset = PreloadedCacheDataset(
                cache_dir=ds_cache,
                num_samples=len(dataset),
                time_stride=time_stride,
                rollout_horizon=rollout_horizon,
            )
        else:
            # Fallback to GridLatentPairDataset (original behavior)
            latent_dataset = GridLatentPairDataset(
                dataset,
                encoder,
                coords,
                grid_shape,
                field_name=field_name,
                device=device,
                cache_dir=ds_cache,
                cache_dtype=cache_dtype,
                time_stride=time_stride,
                rollout_horizon=rollout_horizon,
                use_inverse_losses=use_inverse_losses,
            )

        loader_kwargs["collate_fn"] = latent_pair_collate
        return DataLoader(latent_dataset, **loader_kwargs)

    # ... rest of function handles multi-task and other cases similarly ...
```

**Key Points**:
- Check cache completeness using existing helper functions
- Estimate RAM requirement and verify availability
- Fall back gracefully to disk I/O if RAM insufficient
- Preserve all existing behavior when cache not available

#### 2. Add RAM Check to Validation Script
**File**: `scripts/validate_data.py` or `scripts/validate_config.py`
**Changes**: Add RAM sufficiency check (optional but helpful)

```python
def validate_ram_for_cache(cfg: Dict) -> List[Tuple[str, bool, str]]:
    """Check if there's sufficient RAM for PreloadedCacheDataset."""
    checks = []

    from ups.data.parallel_cache import estimate_cache_size_mb, check_sufficient_ram

    cache_dir = Path(cfg.get("training", {}).get("latent_cache_dir", "data/latent_cache"))

    if not cache_dir.exists():
        checks.append((
            "RAM check skipped (no cache)",
            True,
            "Cache directory doesn't exist yet"
        ))
        return checks

    # Estimate cache size
    task = cfg.get("data", {}).get("task", "burgers1d")
    split = cfg.get("data", {}).get("split", "train")
    task_cache = cache_dir / f"{task}_{split}"

    if task_cache.exists():
        cache_size_mb = estimate_cache_size_mb(task_cache)
        has_ram = check_sufficient_ram(cache_size_mb)

        if has_ram:
            checks.append((
                "Sufficient RAM for PreloadedCacheDataset",
                True,
                f"Cache: ~{cache_size_mb:.0f} MB (RAM available)"
            ))
        else:
            checks.append((
                "RAM may be insufficient for PreloadedCacheDataset",
                False,
                f"Cache: ~{cache_size_mb:.0f} MB (will fall back to disk I/O)"
            ))
    else:
        checks.append((
            "RAM check skipped (cache not generated)",
            True,
            f"Task cache {task_cache} not found"
        ))

    return checks
```

### Success Criteria

#### Automated Verification
- [ ] Config validation passes: `python scripts/validate_config.py configs/train_burgers_32dim.yaml`
- [ ] Training starts successfully: Check WandB for new run
- [ ] PreloadedCacheDataset activated: Check logs for "✅ Using PreloadedCacheDataset"
- [ ] No OOM errors: Training completes without memory errors
- [ ] Fallback works: Test with insufficient RAM (artificially), verify disk I/O mode used

#### Manual Verification
- [ ] Cache preloading time: 10-30 seconds at dataset init (one-time cost)
- [ ] GPU utilization during training: 90%+ (check with `nvidia-smi dmon -s u`)
- [ ] Epoch time reduced: 20-40% faster than baseline (compare to Phase 1 runs)
- [ ] No disk I/O during training: Check with `iostat -x 1` (near-zero disk reads after first epoch)
- [ ] Training metrics unchanged: NRMSE and loss curves similar to baseline

**Implementation Note**: After GPU utilization reaches 90%+ and epoch time improves, Phase 2 complete.

---

## Testing Strategy

### Unit Tests

Create new test file: `tests/unit/test_cache_optimization.py`

```python
"""Unit tests for cache optimization features."""

import json
import tempfile
from pathlib import Path

import pytest
import torch

from ups.data.parallel_cache import (
    check_cache_complete,
    estimate_cache_size_mb,
    check_sufficient_ram,
    PreloadedCacheDataset,
)


def test_cache_hash_deterministic():
    """Verify cache hash is deterministic."""
    from scripts.precompute_latent_cache import compute_cache_hash

    cfg = {
        "latent": {"dim": 32, "tokens": 16},
        "data": {"task": "burgers1d", "root": "data/pdebench"},
    }

    hash1 = compute_cache_hash(cfg)
    hash2 = compute_cache_hash(cfg)
    assert hash1 == hash2, "Hash should be deterministic"


def test_cache_hash_changes_with_config():
    """Verify cache hash changes when relevant config changes."""
    from scripts.precompute_latent_cache import compute_cache_hash

    cfg1 = {"latent": {"dim": 32}}
    cfg2 = {"latent": {"dim": 64}}

    hash1 = compute_cache_hash(cfg1)
    hash2 = compute_cache_hash(cfg2)
    assert hash1 != hash2, "Hash should change when latent.dim changes"


def test_cache_completeness_check():
    """Test cache completeness check."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)

        # Empty cache
        complete, count = check_cache_complete(cache_dir, num_samples=10)
        assert not complete
        assert count == 0

        # Partial cache
        for i in range(5):
            (cache_dir / f"sample_{i:05d}.pt").write_text("dummy")
        complete, count = check_cache_complete(cache_dir, num_samples=10)
        assert not complete
        assert count == 5

        # Complete cache
        for i in range(5, 10):
            (cache_dir / f"sample_{i:05d}.pt").write_text("dummy")
        complete, count = check_cache_complete(cache_dir, num_samples=10)
        assert complete
        assert count == 10


def test_cache_size_estimation():
    """Test cache size estimation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)

        # Create dummy files with known size
        for i in range(10):
            (cache_dir / f"sample_{i:05d}.pt").write_bytes(b"x" * 1000)  # 1KB each

        size_mb = estimate_cache_size_mb(cache_dir, num_samples=10)
        assert 0.009 < size_mb < 0.011, f"Expected ~0.01 MB, got {size_mb}"


def test_ram_sufficiency_check():
    """Test RAM sufficiency check."""
    # Very small requirement should always pass
    assert check_sufficient_ram(required_mb=1.0)

    # Impossibly large requirement should fail
    assert not check_sufficient_ram(required_mb=1e12)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_preloaded_cache_dataset():
    """Test PreloadedCacheDataset loads and serves data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        num_samples = 5

        # Create dummy cache files
        for i in range(num_samples):
            data = {
                "latent": torch.randn(10, 8, 32),  # (T, tokens, dim)
                "params": None,
                "bc": None,
            }
            torch.save(data, cache_dir / f"sample_{i:05d}.pt")

        # Load dataset
        dataset = PreloadedCacheDataset(
            cache_dir=cache_dir,
            num_samples=num_samples,
            time_stride=1,
            rollout_horizon=1,
        )

        assert len(dataset) == num_samples

        # Get a sample
        pair = dataset[0]
        assert hasattr(pair, "z0")
        assert hasattr(pair, "z1")
        assert pair.z0.shape[-1] == 32, "Latent dim should be 32"
```

### Integration Tests

Add to `tests/integration/test_training_pipeline.py`:

```python
def test_cache_persistence_across_runs():
    """Verify cache persists and is reused across runs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir) / "cache"

        # First run: Generate cache
        # (Use minimal config and run for 1 epoch)
        # ...

        # Verify cache created
        assert cache_dir.exists()
        metadata_file = cache_dir / ".cache_metadata.json"
        assert metadata_file.exists()

        # Second run: Reuse cache
        # (Same config, verify cache not regenerated)
        # ...

        # Third run: Different config (invalidate cache)
        # (Change latent.dim, verify cache regenerated)
        # ...
```

### Manual Testing Steps

1. **Test GPU Cache Precomputation**:
   ```bash
   # Launch VastAI instance
   python scripts/vast_launch.py launch \
     --config configs/train_burgers_32dim.yaml \
     --auto-shutdown --dry-run

   # Check generated script uses GPU
   grep "device cuda" .vast/onstart.sh

   # Actually launch
   python scripts/vast_launch.py launch \
     --config configs/train_burgers_32dim.yaml \
     --auto-shutdown

   # Monitor GPU usage during cache generation
   vastai ssh <instance_id> "watch -n 1 nvidia-smi"

   # Verify cache generation time in logs
   vastai logs <instance_id> | grep "Latent precomputation finished"
   ```

2. **Test Cache Persistence**:
   ```bash
   # First run
   python scripts/vast_launch.py launch \
     --config configs/train_burgers_32dim.yaml \
     --auto-shutdown

   # Note cache generation time from logs

   # Second run (same config)
   python scripts/vast_launch.py launch \
     --config configs/train_burgers_32dim.yaml \
     --auto-shutdown

   # Should see "Cache valid" in logs and skip generation

   # Third run (change config)
   # Edit configs/train_burgers_32dim.yaml: latent.dim = 64
   python scripts/vast_launch.py launch \
     --config configs/train_burgers_32dim.yaml \
     --auto-shutdown

   # Should see "Cache invalid" and regenerate
   ```

3. **Test PreloadedCacheDataset**:
   ```bash
   # After cache is generated, launch training
   python scripts/train.py \
     --config configs/train_burgers_32dim.yaml \
     --stage operator --epochs 3

   # Monitor GPU utilization in separate terminal
   watch -n 1 nvidia-smi dmon -s u

   # Should see GPU util at 90%+ during training

   # Check logs for PreloadedCacheDataset activation
   grep "Using PreloadedCacheDataset" logs/*.log
   ```

## Performance Considerations

### GPU Memory
- **PreloadedCacheDataset** requires 10-20 GB RAM for typical 32-dim cache
- A100 40GB instances have sufficient RAM
- Smaller instances (16GB) may need to use disk I/O mode (automatic fallback)

### Cache Size
- 32-dim latent: ~500 MB - 2 GB per dataset
- 64-dim latent: ~2 GB - 8 GB per dataset
- 512-dim latent: ~10 GB - 40 GB per dataset

### Network Bandwidth
- Parallel downloads require ~100-300 Mbps combined
- VastAI instances typically have 1+ Gbps, so no bottleneck expected

### Disk I/O
- Cache files use float16 by default (50% size reduction)
- PreloadedCacheDataset eliminates disk I/O during training
- Checkpoint saves still incur disk I/O (unavoidable)

## Migration Notes

### Backward Compatibility
All changes are backward-compatible:
- Legacy mode (CPU, no-parallel) still works if GPU fails
- Disk I/O mode still works if RAM insufficient
- Cache regeneration still works if hash check fails

### Rollback Plan
If issues arise, rollback by:
1. Reverting `vast_launch.py` to use `--device cpu --no-parallel`
2. Adding `--overwrite` flag to force cache regeneration
3. Disabling PreloadedCacheDataset by setting a feature flag

### Monitoring
Track these metrics per run in WandB:
- `startup/cache_precompute_time_min` - Cache generation time
- `startup/cache_reused` - Boolean, was cache reused
- `startup/total_startup_time_min` - Total time from launch to training start
- `training/gpu_utilization_pct` - Average GPU util during first epoch
- `training/epoch_time_sec` - Time per epoch (should decrease with PreloadedCacheDataset)

## References

- **Original Analysis**: `reports/research/2025-10-28-training-overhead-analysis.md`
- **GPU Cache Implementation**: `scripts/precompute_latent_cache.py:88-140`
- **Parallel Cache System**: `src/ups/data/parallel_cache.py`
- **Helper Functions**: `parallel_cache.py:313-361`
- **Data Loader Builder**: `src/ups/data/latent_pairs.py:588-721`
- **VastAI Launch Script**: `scripts/vast_launch.py`

## Expected Timeline

### Phase 1: Quick Wins (5 days)
- Day 1: Enable GPU cache precomputation (4h), test on VastAI (2h)
- Day 2: Implement parallel downloads (2h), test (2h)
- Day 3: Add cache hashing logic (4h), test locally (2h)
- Day 4: Integrate cache persistence into VastAI (3h), test end-to-end (3h)
- Day 5: Buffer for bug fixes and documentation

### Phase 2: High-Impact (5 days)
- Day 6: Add cache completeness check (3h), RAM check (2h)
- Day 7: Integrate PreloadedCacheDataset into data loader (4h)
- Day 8: Test locally with small config (4h)
- Day 9: Test on VastAI with production config (6h)
- Day 10: Performance validation and documentation

**Total Estimated Time**: 10 days (2 weeks with buffer)

## Success Metrics

| Metric | Baseline | Target | Verification |
|--------|----------|--------|--------------|
| Startup time | 14-25 min | 6-9 min | Instance logs, WandB |
| Cache precompute | 5-15 min | 2-5 min | Precompute logs |
| Cache reuse | Never | Always (same config) | Check metadata file |
| GPU utilization | 60-70% | 90%+ | nvidia-smi dmon |
| Epoch time | Baseline | 20-40% faster | WandB metrics |
| Cost per run | $0.50 | $0.20 | VastAI billing |

---

**Created**: 2025-10-28
**Status**: Ready for implementation
**Estimated Effort**: 10 days (Phase 1: 5 days, Phase 2: 5 days)
**Expected Savings**: $0.30-0.50 per run, 60% startup time reduction
