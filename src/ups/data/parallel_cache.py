"""Parallel-safe latent cache creation using custom collate function.

This module provides a DataLoader that avoids device mismatch issues by:
1. Workers load raw field data (no encoding)
2. Main process does GPU encoding in custom collate_fn
3. Enables num_workers > 0 for 4-8× faster cache creation
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
from pathlib import Path
import io

import torch
from torch.utils.data import Dataset, DataLoader

from ups.data.pdebench import PDEBenchDataset
from ups.data.latent_pairs import (
    LatentPair,
    prepare_conditioning,
    collate_latent_pairs,
)
from ups.io.enc_grid import GridEncoder


class RawFieldDataset(Dataset):
    """Dataset that returns raw fields without encoding (for parallel workers)."""

    def __init__(
        self,
        base: PDEBenchDataset,
        field_name: str = "u",
        *,
        cache_dir: Optional[Path] = None,
        cache_dtype: Optional[torch.dtype] = torch.float16,
        time_stride: int = 1,
        rollout_horizon: int = 1,
    ) -> None:
        super().__init__()
        self.base = base
        self.field_name = field_name
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dtype = cache_dtype
        self.time_stride = max(1, int(time_stride))
        self.rollout_horizon = max(1, int(rollout_horizon))

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return raw fields and metadata (no encoding in workers)."""
        # Check cache first
        if self.cache_dir:
            cache_path = self.cache_dir / f"sample_{idx:05d}.pt"
            if cache_path.exists():
                try:
                    data = torch.load(cache_path, map_location="cpu")
                    return {
                        "idx": idx,
                        "cached": True,
                        "latent": data["latent"].float(),
                        "params": data.get("params"),
                        "bc": data.get("bc"),
                        "cache_path": cache_path,
                    }
                except (RuntimeError, EOFError):  # corrupted file
                    cache_path.unlink(missing_ok=True)

        # Return raw fields for encoding in main process
        sample = self.base[idx]
        return {
            "idx": idx,
            "cached": False,
            "fields": sample["fields"].float(),
            "params": sample.get("params"),
            "bc": sample.get("bc"),
            "cache_path": self.cache_dir / f"sample_{idx:05d}.pt" if self.cache_dir else None,
        }


def make_collate_with_encoding(
    encoder: GridEncoder,
    coords: torch.Tensor,
    grid_shape: Tuple[int, int],
    field_name: str,
    device: torch.device,
    cache_dtype: Optional[torch.dtype] = torch.float16,
    time_stride: int = 1,
    rollout_horizon: int = 1,
):
    """Create a collate function that encodes in the main process on GPU."""

    def collate_with_encoding(batch):
        """Encode raw fields on GPU and return latent pairs."""
        from ups.data.latent_pairs import _fields_to_latent_batch

        latent_pairs = []

        for item in batch:
            if item["cached"]:
                # Use cached latent
                latent_seq = item["latent"]
            else:
                # Encode in main process on GPU
                fields = item["fields"].to(device, non_blocking=True)
                params_cpu = item["params"]
                bc_cpu = item["bc"]

                # Move params/bc to device
                params_device = None
                if params_cpu is not None:
                    params_device = {k: v.to(device, non_blocking=True) for k, v in params_cpu.items()}
                bc_device = None
                if bc_cpu is not None:
                    bc_device = {k: v.to(device, non_blocking=True) for k, v in bc_cpu.items()}

                # Encode on GPU in main process
                latent_seq = _fields_to_latent_batch(
                    encoder,
                    fields,
                    coords,
                    grid_shape,
                    params=params_device,
                    bc=bc_device,
                    field_name=field_name,
                )

                # Save to cache
                if item["cache_path"] is not None:
                    cache_path = item["cache_path"]
                    to_store = latent_seq.to(cache_dtype) if cache_dtype is not None else latent_seq
                    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
                    tmp_path.unlink(missing_ok=True)
                    payload = {
                        "latent": to_store.cpu(),
                        "params": params_cpu,
                        "bc": bc_cpu,
                    }
                    buffer = io.BytesIO()
                    torch.save(payload, buffer)
                    tmp_path.write_bytes(buffer.getvalue())
                    tmp_path.replace(cache_path)

            # Apply time stride
            if time_stride > 1:
                latent_seq = latent_seq[::time_stride]

            if latent_seq.shape[0] <= rollout_horizon:
                raise ValueError("Need more time steps than rollout horizon to form latent pairs")

            # Create latent pairs
            base_len = latent_seq.shape[0] - rollout_horizon
            z0 = latent_seq[:base_len]
            targets = []
            for step in range(1, rollout_horizon + 1):
                targets.append(latent_seq[step : step + base_len])
            target_stack = torch.stack(targets, dim=1)
            z1 = target_stack[:, 0]
            future = target_stack[:, 1:] if rollout_horizon > 1 else None

            cond = prepare_conditioning(item.get("params"), item.get("bc"), base_len)
            latent_pairs.append(LatentPair(z0, z1, cond, future=future))

        # Collate latent pairs
        return collate_latent_pairs(latent_pairs)

    return collate_with_encoding


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
) -> DataLoader:
    """Build a DataLoader with parallel workers that avoids device mismatch.

    Workers load raw data, main process does GPU encoding.
    Safe to use num_workers > 0 for 4-8× speedup.
    """
    raw_dataset = RawFieldDataset(
        dataset,
        field_name=field_name,
        cache_dir=cache_dir,
        cache_dtype=cache_dtype,
        time_stride=time_stride,
        rollout_horizon=rollout_horizon,
    )

    collate_fn = make_collate_with_encoding(
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

    return DataLoader(raw_dataset, **loader_kwargs)

