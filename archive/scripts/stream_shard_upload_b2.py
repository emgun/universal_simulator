#!/usr/bin/env python
from __future__ import annotations

"""Stream raw 1D PDEBench shards → write H5 shards locally → upload to B2 and delete.

Token budgeting: aim for ~tokens_per_param_ratio × model_params tokens.

Assumptions:
- Burgers1D/Advection1D H5 layout under data/pdebench/raw/1D/{Burgers,Advection}/Train/*.hdf5
- Each H5 contains trajectories shaped like (N, T, X[, C])

Usage example:
  python scripts/stream_shard_upload_b2.py burgers1d --out data/pdebench --bucket $B2_BUCKET \
    --prefix pdebench/full/burgers1d --artifact-name burgers1d_full

"""

import argparse
import math
import os
import shutil
import subprocess
from glob import glob
from pathlib import Path
from typing import Iterable, List

import h5py
import numpy as np

from ups.data.convert_pdebench import convert_files


def estimate_tokens_per_sample(h5_path: Path) -> int:
    with h5py.File(h5_path, "r") as f:
        # Heuristic: largest numeric dataset size is proxy for tokens
        best = 0
        def _visit(_, obj):
            nonlocal best
            if isinstance(obj, h5py.Dataset) and obj.dtype.kind in ("f", "i") and obj.ndim >= 2:
                # Per-sample tokens approximated by np.prod(shape[1:])
                tps = int(np.prod(obj.shape[1:]))
                best = max(best, tps)
        f.visititems(_visit)
        if best == 0:
            raise ValueError(f"No numeric dataset in {h5_path}")
        return best


def count_model_params(latent_dim: int, tokens: int, depth: int = 2) -> int:
    # Very coarse estimate; good enough for budgeting multiplier
    # Assume block has O(D^2 * T) params across a few projections
    return int((latent_dim * latent_dim * tokens) * depth)


def shard_and_upload(
    dataset_key: str,
    raw_root: Path,
    out_dir: Path,
    *,
    tokens_per_param_ratio: float,
    latent_dim: int,
    tokens: int,
    bucket: str,
    prefix: str,
    b2_key_id: str,
    b2_app_key: str,
    samples_per_file: int | None,
) -> List[Path]:
    patterns = {
        "burgers1d": "1D/Burgers/Train/*.hdf5",
        "advection1d": "1D/Advection/Train/*.hdf5",
    }
    pat = patterns.get(dataset_key)
    if not pat:
        raise SystemExit(f"Unsupported dataset: {dataset_key}")
    files = [Path(p) for p in sorted(glob(str(raw_root / pat)))]
    if not files:
        raise SystemExit(f"No raw files found for {dataset_key} under {raw_root}")

    tokens_per_sample = estimate_tokens_per_sample(files[0])
    target_tokens = int(tokens_per_param_ratio * count_model_params(latent_dim, tokens))
    est_samples_needed = max(1, target_tokens // max(1, tokens_per_sample))

    # Convert in shards of N files at a time until we exceed target tokens
    written_files: List[Path] = []
    total_samples = 0
    suffix = dataset_key.replace("1d", "1d").replace(" ", "_")
    out_dir.mkdir(parents=True, exist_ok=True)

    batch_index = 0
    index = 0
    while total_samples < est_samples_needed and index < len(files):
        batch = files[index:index + 8]  # stream in small batches of 8 files
        index += len(batch)
        out_path = out_dir / f"{dataset_key}_train_{batch_index:03d}.h5"
        print(f"Converting {len(batch)} files -> {out_path}")
        written = convert_files(batch, out_path, sample_size=samples_per_file)
        total_samples += written

        # Upload shard via rclone and delete local
        remote = f"UPSB2:{bucket}/{prefix}/{out_path.name}"
        env = os.environ.copy()
        env.setdefault("RCLONE_CONFIG_UPSB2_TYPE", "b2")
        env.setdefault("RCLONE_CONFIG_UPSB2_ACCOUNT", b2_key_id)
        env.setdefault("RCLONE_CONFIG_UPSB2_KEY", b2_app_key)
        subprocess.run(["rclone", "copyto", str(out_path), remote, "-P"], check=True, env=env)
        out_path.unlink(missing_ok=True)
        written_files.append(out_path)
        batch_index += 1

    print(f"Finished {dataset_key}: ~{total_samples} samples (~{total_samples * tokens_per_sample} tokens) vs target {target_tokens}")
    return written_files


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stream raw 1D PDEBench, shard to H5, upload to B2, delete local")
    p.add_argument("dataset", choices=["burgers1d", "advection1d"])
    p.add_argument("--raw-root", default="data/pdebench/raw", help="Root containing raw PDEBench")
    p.add_argument("--out", default="data/pdebench", help="Local temporary output directory")
    p.add_argument("--tokens-per-param-ratio", type=float, default=20.0)
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--tokens", type=int, default=64)
    p.add_argument("--bucket", default=os.environ.get("B2_BUCKET"), required=False)
    p.add_argument("--prefix", default="pdebench/full", help="Remote prefix inside bucket")
    p.add_argument("--samples-per-file", type=int, default=None, help="Optional subsampling per raw file")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    b2_key_id = os.environ.get("B2_KEY_ID")
    b2_app_key = os.environ.get("B2_APP_KEY")
    if not (b2_key_id and b2_app_key and args.bucket):
        raise SystemExit("B2 credentials and bucket must be provided via env or flags")

    shard_and_upload(
        args.dataset,
        Path(args.raw_root),
        Path(args.out),
        tokens_per_param_ratio=args.tokens_per_param_ratio,
        latent_dim=args.latent_dim,
        tokens=args.tokens,
        bucket=args.bucket,
        prefix=args.prefix,
        b2_key_id=b2_key_id,
        b2_app_key=b2_app_key,
        samples_per_file=args.samples_per_file,
    )


if __name__ == "__main__":
    main()


