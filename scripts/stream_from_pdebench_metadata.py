#!/usr/bin/env python
from __future__ import annotations

"""Stream PDEBench raw HDF5 files from metadata CSV URLs → shard → upload to B2 → delete.

Defaults to the official PDEBench metadata CSV hosted on GitHub.
Reference: https://github.com/pdebench/PDEBench?tab=readme-ov-file
"""

import argparse
import csv
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List

import h5py
import requests

from ups.data.convert_pdebench import convert_files


DEFAULT_METADATA = (
    "https://raw.githubusercontent.com/pdebench/PDEBench/main/"
    "pdebench/data_download/pdebench_data_urls.csv"
)

_LOG_FILE: Path | None = None
_LOG_FH = None


def _log(msg: str) -> None:
    print(msg)
    sys.stdout.flush()
    if _LOG_FH is not None:
        _LOG_FH.write(msg + "\n")
        _LOG_FH.flush()


def _download_file(url: str, dst: Path, max_retries: int = 3) -> None:
    # Try to fetch content length for progress
    size_mb = None
    try:
        h = requests.head(url, allow_redirects=True, timeout=30)
        cl = h.headers.get("Content-Length")
        if cl:
            size_mb = int(cl) / (1024 * 1024)
    except Exception:
        pass
    _log(f"Downloading: {url} ({size_mb:.1f} MB) -> {dst.name}" if size_mb else f"Downloading: {url} -> {dst.name}")

    for attempt in range(max_retries):
        try:
            bytes_done = 0
            last_report = 0.0
            report_step = 10 * 1024 * 1024  # 10 MB
            with requests.get(url, stream=True, timeout=(10, 600)) as r:  # 10 min timeout
                r.raise_for_status()
                with dst.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=2 * 1024 * 1024):  # 2 MB
                        if not chunk:
                            continue
                        f.write(chunk)
                        bytes_done += len(chunk)
                        if bytes_done - last_report >= report_step:
                            mb = bytes_done / (1024 * 1024)
                            if size_mb:
                                _log(f"  downloaded {mb:.1f}/{size_mb:.1f} MB ({mb/size_mb*100:.1f}%)")
                            else:
                                _log(f"  downloaded {mb:.1f} MB")
                            last_report = bytes_done
            _log(f"Download completed: {dst.name}")
            return
        except Exception as e:
            _log(f"Download attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                _log(f"Retrying in 5 seconds...")
                time.sleep(5)
            else:
                raise


def _estimate_tokens_per_sample(h5_path: Path) -> int:
    best = 0
    with h5py.File(h5_path, "r") as f:
        def _visit(_, obj):
            nonlocal best
            if isinstance(obj, h5py.Dataset) and obj.dtype.kind in ("f", "i") and obj.ndim >= 2:
                best_val = 1
                for dim in obj.shape[1:]:
                    best_val *= int(dim)
                best = max(best, best_val)
        f.visititems(_visit)
    if best == 0:
        raise ValueError(f"Could not estimate tokens per sample from {h5_path}")
    return best


def _count_model_params(latent_dim: int, tokens: int, depth: int = 2) -> int:
    return int(latent_dim * latent_dim * tokens * depth)


def _b2_env(b2_key_id: str, b2_app_key: str) -> dict:
    env = os.environ.copy()
    env.setdefault("RCLONE_CONFIG_UPSB2_TYPE", "b2")
    env.setdefault("RCLONE_CONFIG_UPSB2_ACCOUNT", b2_key_id)
    env.setdefault("RCLONE_CONFIG_UPSB2_KEY", b2_app_key)
    return env


def stream_from_metadata(
    dataset: str,
    split: str,
    meta_csv: Path,
    out_dir: Path,
    *,
    tokens_per_param_ratio: float,
    latent_dim: int,
    tokens: int,
    bucket: str,
    prefix: str,
    b2_key_id: str,
    b2_app_key: str,
    batch_size: int,
    samples_per_file: int | None,
) -> None:
    dataset = dataset.lower()
    split = split.lower()
    # scan all columns for dataset + split keywords and pick URL-like fields
    if dataset == "burgers1d":
        base_tokens = ("burgers", "1d")
    elif dataset == "advection1d":
        base_tokens = ("advection", "1d")
    else:
        raise SystemExit(f"Unsupported dataset: {dataset}")

    urls: List[str] = []
    with meta_csv.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            text = " ".join(str(v) for v in row.values()).lower()
            if all(tok in text for tok in base_tokens) and split in text:
                # Find any URL-like cell, prefer .hdf5/.h5
                candidates: List[str] = []
                for v in row.values():
                    s = str(v).strip()
                    if s.startswith("http"):
                        candidates.append(s)
                if not candidates:
                    continue
                preferred = None
                for c in candidates:
                    lc = c.lower()
                    if lc.endswith(".hdf5") or lc.endswith(".h5"):
                        preferred = c
                        break
                urls.append(preferred or candidates[0])

    if not urls:
        raise SystemExit(f"No URLs found for {dataset} {split} in metadata")

    _log(f"Found {len(urls)} candidate URLs for {dataset} {split}")
    out_dir.mkdir(parents=True, exist_ok=True)
    b2env = _b2_env(b2_key_id, b2_app_key)

    total_samples = 0
    target_tokens = int(tokens_per_param_ratio * _count_model_params(latent_dim, tokens))

    for i in range(0, len(urls), batch_size):
        batch_urls = urls[i:i + batch_size]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            local_files: List[Path] = []
            for u in batch_urls:
                fname = u.split("/")[-1]
                dst = tmp / fname
                _download_file(u, dst)
                local_files.append(dst)

            if not local_files:
                continue

            shard_name = f"{dataset}_{split}_{i//batch_size:03d}.h5"
            out_path = out_dir / shard_name
            _log(f"Converting {len(local_files)} files -> {out_path}")
            written = convert_files(local_files, out_path, sample_size=samples_per_file)
            total_samples += written

            tps = _estimate_tokens_per_sample(out_path)
            total_tokens = total_samples * tps
            _log(f"Estimated {tps} tokens/sample; cumulative samples {total_samples}; total tokens {total_tokens}")

            remote = f"UPSB2:{bucket}/{prefix}/{shard_name}" if prefix else f"UPSB2:{bucket}/{shard_name}"
            _log(f"Uploading shard to {remote}")
            subprocess.run(["rclone", "copyto", str(out_path), remote, "-P"], check=True, env=b2env)
            _log(f"Uploaded and removing local shard {out_path}")
            out_path.unlink(missing_ok=True)

            if total_tokens >= target_tokens:
                _log(f"Target tokens reached: {total_tokens} >= {target_tokens}")
                break


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stream PDEBench URLs → H5 shards → upload to B2 → delete")
    p.add_argument("--dataset", choices=["burgers1d", "advection1d"], required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--metadata-url", default=DEFAULT_METADATA, help="URL to metadata CSV")
    p.add_argument("--out", default="data/pdebench", help="Temp output dir for shards before upload")
    p.add_argument("--tokens-per-param-ratio", type=float, default=20.0)
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--tokens", type=int, default=64)
    p.add_argument("--bucket", default=os.environ.get("B2_BUCKET"))
    p.add_argument("--prefix", default="pdebench/full")
    p.add_argument("--batch-size", type=int, default=6, help="Number of raw files to stream per shard")
    p.add_argument("--samples-per-file", type=int, default=None)
    p.add_argument("--log-file", help="Optional log file path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    b2_key_id = os.environ.get("B2_KEY_ID")
    b2_app_key = os.environ.get("B2_APP_KEY")
    if not (args.bucket and b2_key_id and b2_app_key):
        raise SystemExit("Missing B2 credentials or bucket in environment")

    global _LOG_FH
    if args.log_file:
        log_path = Path(args.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        _LOG_FH = log_path.open("a", encoding="utf-8")

    # Fetch metadata CSV
    resp = requests.get(args.metadata_url, timeout=60)
    resp.raise_for_status()
    meta_path = Path(tempfile.gettempdir()) / "pdebench_download_metadata.csv"
    meta_path.write_bytes(resp.content)

    try:
        stream_from_metadata(
            args.dataset,
            args.split,
            meta_path,
            Path(args.out),
            tokens_per_param_ratio=args.tokens_per_param_ratio,
            latent_dim=args.latent_dim,
            tokens=args.tokens,
            bucket=args.bucket,
            prefix=args.prefix,
            b2_key_id=b2_key_id,
            b2_app_key=b2_app_key,
            batch_size=args.batch_size,
            samples_per_file=args.samples_per_file,
        )
    finally:
        if _LOG_FH is not None:
            _LOG_FH.close()


if __name__ == "__main__":
    main()
