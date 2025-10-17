#!/usr/bin/env python
"""Materialise latent cache files for PDEBench datasets."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
from torch.utils.data import DataLoader, Subset

try:
    import yaml
except ImportError as exc:  # pragma: no cover - should always be present
    raise SystemExit("PyYAML is required to run this script") from exc

try:  # pragma: no cover - optional progress bar
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

from ups.data.latent_pairs import (
    GridLatentPairDataset,
    _build_pdebench_dataset,
    make_grid_coords,
)


def _maybe_tqdm(iterator: Iterable, total: Optional[int], *, desc: str) -> Iterable:
    if tqdm is None:
        return iterator
    return tqdm(iterator, total=total, desc=desc)


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
) -> GridLatentPairDataset:
    ds_cfg = {
        **data_cfg,
        "task": task,
        "split": split,
        "latent_dim": latent_cfg.get("dim", 32),
        "latent_len": latent_cfg.get("tokens", 16),
    }
    dataset, encoder, grid_shape, field_name = _build_pdebench_dataset(ds_cfg)
    encoder = encoder.to(device)
    coords = make_grid_coords(grid_shape, device)
    ds_cache = cache_dir / f"{task}_{split}" if cache_dir else None
    return GridLatentPairDataset(
        dataset,
        encoder,
        coords,
        grid_shape,
        field_name=field_name,
        device=device,
        cache_dir=ds_cache,
        cache_dtype=cache_dtype,
        rollout_horizon=rollout_horizon,
    )


def _count_cached_samples(cache_dir: Path) -> int:
    if not cache_dir or not cache_dir.exists():
        return 0
    return sum(1 for _ in cache_dir.glob("sample_*.pt"))


def _iter_dataset(
    dataset: GridLatentPairDataset,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    use_parallel: bool = True,
) -> None:
    """Iterate dataset to populate cache.
    
    Args:
        dataset: Dataset to iterate
        batch_size: Batch size for encoding
        num_workers: Number of parallel workers
        pin_memory: Enable pinned memory
        use_parallel: Use parallel encoding (4-8× faster)
    """
    if use_parallel and num_workers > 0:
        # Use parallel cache system for 4-8× speedup
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
        )
    else:
        # Legacy mode (slower, but stable)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,  # Force 0 for legacy to avoid device mismatch
            pin_memory=pin_memory,
            persistent_workers=False,
            collate_fn=lambda items: items,
        )
    
    total_batches = len(loader)
    iterator = _maybe_tqdm(loader, total_batches, desc="encoding")
    for batch in iterator:
        # Accessing the batch ensures __getitem__ executes and caches samples.
        if isinstance(batch, list):
            # Explicitly drop tensors to free GPU memory quickly.
            for item in batch:
                del item
        del batch


def _summarise_cache(cache_dir: Path) -> Dict[str, float]:
    files = list(cache_dir.glob("sample_*.pt"))
    total_bytes = sum(f.stat().st_size for f in files)
    return {
        "num_samples": len(files),
        "total_mb": total_bytes / 1e6 if files else 0.0,
    }


def _load_config(path: Optional[str]) -> Dict[str, object]:
    if path is None:
        return {}
    cfg_path = Path(path)
    if not cfg_path.exists():  # pragma: no cover - early exit
        raise SystemExit(f"Config file {cfg_path} does not exist")
    with cfg_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def main() -> None:
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
    args = parser.parse_args()

    cfg = _load_config(args.config)
    data_cfg = dict(cfg.get("data", {}))
    if args.root is not None:
        data_cfg["root"] = args.root
    elif "root" not in data_cfg:
        data_cfg["root"] = "data/pdebench"

    latent_cfg = dict(cfg.get("latent", {}))
    if args.latent_dim is not None:
        latent_cfg["dim"] = args.latent_dim
    if args.latent_len is not None:
        latent_cfg["tokens"] = args.latent_len

    cache_root = Path(args.cache_dir).resolve()
    cache_root.mkdir(parents=True, exist_ok=True)

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

    torch.set_grad_enabled(False)
    rollout_horizon = max(1, int(cfg.get("training", {}).get("rollout_horizon", 1)))

    summary: Dict[str, Dict[str, float]] = {}
    start_time = time.time()
    for task in args.tasks:
        for split in args.splits:
            ds_cache_dir = cache_root / f"{task}_{split}"
            existing = _count_cached_samples(ds_cache_dir)
            dataset = _instantiate_dataset(
                task=task,
                split=split,
                data_cfg=data_cfg,
                latent_cfg=latent_cfg,
                device=device,
                cache_dir=cache_root,
                cache_dtype=cache_dtype,
                rollout_horizon=rollout_horizon,
            )
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

            mode_str = "parallel" if args.parallel and args.num_workers > 0 else "legacy"
            print(f"[{task}:{split}] encoding {target_samples} samples (cache dir: {ds_cache_dir}, mode: {mode_str})")
            split_start = time.time()
            _iter_dataset(
                dataset,
                batch_size=max(1, args.batch_size),
                num_workers=max(0, args.num_workers),
                pin_memory=args.pin_memory,
                use_parallel=args.parallel,
            )
            elapsed = time.time() - split_start
            stats = _summarise_cache(ds_cache_dir)
            stats["elapsed_min"] = elapsed / 60.0
            summary[f"{task}_{split}"] = stats
            print(f"[{task}:{split}] completed in {elapsed/60.0:.2f} min – {stats['num_samples']} samples cached")

    total_elapsed = time.time() - start_time
    print(f"Latency precomputation finished in {total_elapsed/60.0:.2f} minutes")

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
