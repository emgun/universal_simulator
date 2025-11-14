#!/usr/bin/env python
"""Materialise latent cache files for PDEBench datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    import yaml
except ImportError as exc:  # pragma: no cover - should always be present
    raise SystemExit("PyYAML is required to run this script") from exc

try:  # pragma: no cover - optional progress bar
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

import logging

from ups.data.datasets import MeshZarrDataset, ParticleZarrDataset
from ups.data.latent_pairs import (
    GraphLatentPairDataset,
    GridLatentPairDataset,
    _build_pdebench_dataset,
    make_grid_coords,
)
from ups.data.pdebench import get_pdebench_spec, resolve_pdebench_root
from ups.io.enc_mesh_particle import MeshParticleEncoder, MeshParticleEncoderConfig


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
        print(
            f"📊 Auto-selected {optimal} workers "
            f"(RAM: {available_ram_gb:.1f}GB, CPUs: {max_workers_cpu})"
        )
        return optimal
    except Exception:
        # Fallback to conservative default
        return 4


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
                print("   Likely hung in: DataLoader iteration or worker process")
                print("   Try: --num-workers 0 (disable multiprocessing)")
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


def _maybe_tqdm(iterator: Iterable, total: int | None, *, desc: str) -> Iterable:
    if tqdm is None:
        return iterator
    return tqdm(iterator, total=total, desc=desc)


def _instantiate_dataset(
    *,
    task: str,
    split: str,
    data_cfg: dict[str, object],
    latent_cfg: dict[str, object],
    device: torch.device,
    cache_dir: Path | None,
    cache_dtype: torch.dtype | None,
    rollout_horizon: int,
    use_inverse_losses: bool = False,
    time_stride: int = 1,
    hdf5_timeout: int = 0,
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
            ds_cfg, hdf5_timeout=hdf5_timeout
        )
        encoder = encoder.to(device)
        coords = make_grid_coords(grid_shape, device)
        return GridLatentPairDataset(
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

    data_root = resolve_pdebench_root(data_cfg.get("root"))
    zarr_path = data_root / f"{task}_{split}.zarr"
    if not zarr_path.exists():
        raise FileNotFoundError(f"Expected dataset at {zarr_path} for task '{task}'")

    if spec.kind == "mesh":
        base_dataset = MeshZarrDataset(str(zarr_path), group=task)
    elif spec.kind == "particles":
        base_dataset = ParticleZarrDataset(str(zarr_path), group=task)
    else:
        raise ValueError(f"Unsupported dataset kind '{spec.kind}' for precomputing cache")

    latent_dim = latent_cfg.get("dim", 32)
    latent_tokens = latent_cfg.get("tokens", 16)
    hidden_dim = data_cfg.get("hidden_dim", max(latent_dim * 2, 64))
    encoder_cfg = MeshParticleEncoderConfig(
        latent_len=latent_tokens,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        message_passing_steps=data_cfg.get("message_passing_steps", 3),
        supernodes=data_cfg.get("supernodes", 2048),
        use_coords=data_cfg.get("use_coords", True),
    )
    graph_encoder = MeshParticleEncoder(encoder_cfg).eval().to(device)
    return GraphLatentPairDataset(
        base_dataset,
        graph_encoder,
        kind=spec.kind,
        rollout_horizon=rollout_horizon,
        cache_dir=ds_cache,
        cache_dtype=cache_dtype,
        time_stride=time_stride,
    )


def _count_cached_samples(cache_dir: Path) -> int:
    if not cache_dir or not cache_dir.exists():
        return 0
    return sum(1 for _ in cache_dir.glob("sample_*.pt"))


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
    use_parallel_loader = (
        use_parallel and num_workers > 0 and isinstance(dataset, GridLatentPairDataset)
    )

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
    """Iterate dataset to populate cache with robust fallback.

    Args:
        dataset: Dataset to iterate
        batch_size: Batch size for encoding
        num_workers: Number of parallel workers
        pin_memory: Enable pinned memory
        use_parallel: Use parallel encoding (4-8× faster)
        timeout: DataLoader worker timeout in seconds (0=disabled)
        watchdog_timeout: Progress watchdog timeout in seconds (0=disabled)
    """

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


def _summarise_cache(cache_dir: Path) -> dict[str, float]:
    files = list(cache_dir.glob("sample_*.pt"))
    total_bytes = sum(f.stat().st_size for f in files)
    return {
        "num_samples": len(files),
        "total_mb": total_bytes / 1e6 if files else 0.0,
    }


def _load_config(path: str | None) -> dict[str, object]:
    if path is None:
        return {}
    cfg_path = Path(path)
    if not cfg_path.exists():  # pragma: no cover - early exit
        raise SystemExit(f"Config file {cfg_path} does not exist")
    with cfg_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def compute_cache_hash(cfg: dict[str, Any]) -> str:
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


def save_cache_metadata(cache_dir: Path, cfg: dict[str, Any]) -> None:
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


def check_cache_valid(cache_dir: Path, cfg: dict[str, Any]) -> bool:
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


def main() -> None:
    # Set multiprocessing start method to 'spawn' for CUDA compatibility
    # Must be done before any CUDA initialization
    import multiprocessing
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        # Already set, ignore
        pass

    parser = argparse.ArgumentParser(description="Precompute latent caches for PDEBench tasks")
    parser.add_argument("--config", default="configs/train_pdebench_scale.yaml", help="Training config to read defaults from")
    parser.add_argument("--tasks", nargs="+", default=["burgers1d"], help="One or more PDEBench tasks to process")
    parser.add_argument("--splits", nargs="+", default=["train"], help="Dataset splits to encode (train/val/test)")
    parser.add_argument("--root", default=None, help="Override data root directory")
    parser.add_argument("--cache-dir", default="data/latent_cache", help="Directory to write latent cache files")
    parser.add_argument("--cache-dtype", default=None, help="Torch dtype for cached tensors (e.g. float16, float32)")
    parser.add_argument("--latent-dim", type=int, default=None, help="Override latent dimensionality")
    parser.add_argument("--latent-len", type=int, default=None, help="Override latent token length")
    parser.add_argument("--device", default=None, help="Device to run encoders on (cuda, cpu)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for worker prefetching")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of DataLoader workers")
    parser.add_argument("--parallel", action="store_true", default=True, help="Use parallel encoding (4-8× faster, default: True)")
    parser.add_argument("--no-parallel", dest="parallel", action="store_false", help="Disable parallel encoding (use legacy mode)")
    parser.add_argument("--pin-memory", action="store_true", help="Enable pin_memory in DataLoader")
    parser.add_argument("--limit", type=int, default=0, help="Process at most this many samples per split (0 = full dataset)")
    parser.add_argument("--overwrite", action="store_true", help="Recompute even if cache already exists")
    parser.add_argument("--manifest", default=None, help="Optional path to write cache summary JSON")
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
    parser.add_argument(
        "--watchdog-timeout",
        type=int,
        default=120,
        help="Watchdog timeout in seconds (0=disabled, default: 120)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging for diagnostics")
    parser.add_argument("--skip-auto-copy", action="store_true", help="Skip auto-copy prompt for network storage")
    args = parser.parse_args()

    # Setup logging
    global logger
    logger = setup_logging(verbose=args.verbose)

    cfg = _load_config(args.config)
    data_cfg = dict(cfg.get("data", {}))
    if args.root is not None:
        data_cfg["root"] = args.root
    elif "root" not in data_cfg:
        data_cfg["root"] = "data/pdebench"

    # Detect network storage and prompt for local copy
    data_root = resolve_pdebench_root(data_cfg.get("root", "data/pdebench"))
    if is_network_storage(data_root):
        logger.warning("⚠️  Data appears to be on network storage (slower, may cause hangs)")
        logger.warning("   Recommendation: Copy to local storage first:")
        logger.warning(f"   bash scripts/copy_data_to_local.sh {data_root} /workspace/data_local/pdebench")
        logger.warning("   Then run with: --root /workspace/data_local/pdebench")

        # Optional: Auto-copy if on VastAI/Vultr and local workspace exists
        if Path("/workspace").exists() and not args.skip_auto_copy:
            try:
                response = input("Auto-copy to local storage? (y/N): ")
                if response.lower() == 'y':
                    local_path = Path("/workspace/data_local/pdebench")
                    logger.info(f"Copying {data_root} to {local_path}...")
                    import shutil
                    shutil.copytree(data_root, local_path, dirs_exist_ok=True)
                    data_cfg["root"] = str(local_path)
                    logger.info("✅ Data copied to local storage")
            except (KeyboardInterrupt, EOFError):
                logger.info("Skipping auto-copy")

    latent_cfg = dict(cfg.get("latent", {}))
    if args.latent_dim is not None:
        latent_cfg["dim"] = args.latent_dim
    if args.latent_len is not None:
        latent_cfg["tokens"] = args.latent_len

    cache_root = Path(args.cache_dir).resolve()
    cache_root.mkdir(parents=True, exist_ok=True)

    # Check if cache is valid and up-to-date
    if cache_root.exists() and not args.overwrite:
        if check_cache_valid(cache_root, cfg):
            print("✅ Cache is valid and up-to-date, skipping regeneration")
            print("   Use --overwrite to force regeneration")
            return

    cache_dtype = None
    if args.cache_dtype:
        if not hasattr(torch, args.cache_dtype):  # pragma: no cover - validation
            raise SystemExit(f"Unknown torch dtype '{args.cache_dtype}'")
        cache_dtype = getattr(torch, args.cache_dtype)
    elif cfg.get("training", {}).get("latent_cache_dtype"):
        dtype_name = cfg["training"]["latent_cache_dtype"]
        cache_dtype = getattr(torch, dtype_name)

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU")
        device = torch.device("cpu")

    # Verbose diagnostics
    if args.verbose:
        logger.debug(f"Python: {sys.version}")
        logger.debug(f"PyTorch: {torch.__version__}")
        logger.debug(f"CUDA available: {torch.cuda.is_available()}")
        logger.debug(f"Device: {device}")
        logger.debug(
            f"Timeouts: DataLoader={args.dataloader_timeout}s, "
            f"HDF5={args.hdf5_timeout}s, Watchdog={args.watchdog_timeout}s"
        )

    torch.set_grad_enabled(False)
    training_cfg = cfg.get("training", {})
    rollout_horizon = max(1, int(training_cfg.get("rollout_horizon", 1)))
    time_stride = max(1, int(training_cfg.get("time_stride", 1)))

    # Read UPT inverse losses flag from config
    use_inverse_losses = (
        training_cfg.get("use_inverse_losses", False) or
        training_cfg.get("lambda_inv_enc", 0.0) > 0 or
        training_cfg.get("lambda_inv_dec", 0.0) > 0
    )
    if use_inverse_losses:
        print("✓ UPT inverse losses enabled - physical fields will be cached")

    summary: dict[str, dict[str, float]] = {}
    start_time = time.time()
    for task in args.tasks:
        for split in args.splits:
            spec = get_pdebench_spec(task)
            if spec.kind == "mesh":
                print(f"[{task}:{split}] Mesh datasets are steady-state; latent caches are not applicable. Skipping.")
                summary[f"{task}_{split}"] = {"num_samples": 0, "total_mb": 0.0, "skipped": "mesh dataset"}
                continue
            ds_cache_dir = cache_root / f"{task}_{split}"
            existing = _count_cached_samples(ds_cache_dir)
            try:
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
                    hdf5_timeout=args.hdf5_timeout,
                )
            except (ValueError, FileNotFoundError) as exc:
                print(f"[{task}:{split}] skipping cache generation: {exc}")
                summary[f"{task}_{split}"] = {"num_samples": 0, "total_mb": 0.0, "skipped": str(exc)}
                continue
            total_samples = len(dataset)
            target_samples = total_samples if args.limit <= 0 else min(args.limit, total_samples)
            if args.limit > 0:
                dataset = Subset(dataset, list(range(target_samples)))
            else:
                target_samples = total_samples

            if existing == target_samples and not args.overwrite:
                print(f"[{task}:{split}] cache already contains {existing} samples – skipping")
                summary[f"{task}_{split}"] = _summarise_cache(ds_cache_dir)
                continue

            if args.overwrite and ds_cache_dir.exists():
                for file in ds_cache_dir.glob("sample_*.pt"):
                    file.unlink()

            # Auto-calculate optimal workers if not specified
            actual_num_workers = get_optimal_workers(args.num_workers, batch_size=args.batch_size)
            if actual_num_workers != args.num_workers:
                print(f"ℹ️  Adjusted workers: {args.num_workers} → {actual_num_workers}")

            if args.verbose:
                logger.debug(f"Workers: {actual_num_workers}")

            mode_str = "parallel" if args.parallel and actual_num_workers > 0 else "legacy"
            print(f"[{task}:{split}] encoding {target_samples} samples (cache dir: {ds_cache_dir}, mode: {mode_str})")
            split_start = time.time()
            _iter_dataset(
                dataset,
                batch_size=max(1, args.batch_size),
                num_workers=actual_num_workers,
                pin_memory=args.pin_memory,
                use_parallel=args.parallel,
                timeout=args.dataloader_timeout,
                watchdog_timeout=args.watchdog_timeout,
            )
            elapsed = time.time() - split_start
            stats = _summarise_cache(ds_cache_dir)
            stats["elapsed_min"] = elapsed / 60.0
            summary[f"{task}_{split}"] = stats
            print(f"[{task}:{split}] completed in {elapsed/60.0:.2f} min – {stats['num_samples']} samples cached")

    total_elapsed = time.time() - start_time
    print(f"Latency precomputation finished in {total_elapsed/60.0:.2f} minutes")

    # Save cache metadata for validation on subsequent runs
    save_cache_metadata(cache_root, cfg)

    if args.manifest:
        manifest_path = Path(args.manifest)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "generated_at": time.time(),
            "cache_dir": str(cache_root),
            "summary": summary,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Wrote summary to {manifest_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
