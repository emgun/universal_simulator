#!/usr/bin/env python
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import os
import sys
from pathlib import Path
from typing import Iterable

import requests
import yaml

MANIFEST_PATH = Path(__file__).resolve().parents[1] / "docs" / "pdebench_manifest.yaml"
DATAFILE_URL = "https://darus.uni-stuttgart.de/api/access/datafile/{file_id}?format=original"
DEFAULT_CHUNK_SIZE = 1024 * 1024
DEFAULT_PART_SIZE = 256 * 1024 * 1024
DEFAULT_TIMEOUT = (30, 60)


def load_manifest(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"Manifest not found at {path}. Run the manifest fetch step first.")
    data = yaml.safe_load(path.read_text())
    files = data.get("files")
    if not isinstance(files, list):
        raise SystemExit("Invalid manifest format: 'files' missing or not a list.")
    return files


def find_entry(manifest: list[dict], logical_path: str) -> dict:
    for entry in manifest:
        if entry.get("path") == logical_path:
            return entry
    raise SystemExit(f"Path '{logical_path}' not found in manifest.")


def _part_ranges(total_size: int, part_size: int) -> list[tuple[int, int, int]]:
    if total_size <= 0:
        return []
    ranges = []
    for index, start in enumerate(range(0, total_size, part_size)):
        ranges.append((index, start, min(start + part_size - 1, total_size - 1)))
    return ranges


def _existing_checksum(path: Path) -> str:
    checksum = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(DEFAULT_CHUNK_SIZE), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def _request_with_retries(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    stream: bool = True,
    retries: int = 3,
    timeout: tuple[int, int] = DEFAULT_TIMEOUT,
) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, headers=headers, stream=stream, timeout=timeout)
            response.raise_for_status()
            return response
        except (requests.RequestException, requests.Timeout) as exc:
            last_error = exc
            print(f"request failed on attempt {attempt}/{retries}: {exc}", file=sys.stderr, flush=True)
    assert last_error is not None
    raise last_error


def _download_stream(url: str, dest: Path, expected_size: int | None, chunk_size: int) -> int:
    response = _request_with_retries(url, stream=True)
    total = 0
    with open(dest, "wb") as fh:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if not chunk:
                continue
            fh.write(chunk)
            total += len(chunk)
            if expected_size:
                pct = total / expected_size * 100
                sys.stdout.write(f"\rDownloaded {total/1024**2:.2f} MiB ({pct:.1f}%)")
                sys.stdout.flush()
    sys.stdout.write("\n")
    return total


def _download_part(
    url: str,
    part_path: Path,
    start: int,
    end: int,
    *,
    chunk_size: int,
    retries: int,
) -> int:
    expected_size = end - start + 1
    if part_path.exists() and part_path.stat().st_size == expected_size:
        return expected_size

    temp_path = part_path.with_suffix(part_path.suffix + ".tmp")
    if temp_path.exists():
        temp_path.unlink()

    headers = {"Range": f"bytes={start}-{end}"}
    response = _request_with_retries(url, headers=headers, stream=True, retries=retries)
    status_code = int(response.status_code)
    if status_code != 206:
        raise SystemExit(f"Range request {start}-{end} returned HTTP {status_code}; expected 206.")

    total = 0
    with open(temp_path, "wb") as fh:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if not chunk:
                continue
            fh.write(chunk)
            total += len(chunk)

    if total != expected_size:
        temp_path.unlink(missing_ok=True)
        raise SystemExit(
            f"Range request {start}-{end} wrote {total} bytes; expected {expected_size}."
        )
    temp_path.replace(part_path)
    return total


def _assemble_parts(dest: Path, part_paths: Iterable[Path]) -> int:
    total = 0
    temp_dest = dest.with_suffix(dest.suffix + ".tmp")
    if temp_dest.exists():
        temp_dest.unlink()
    with open(temp_dest, "wb") as out:
        for part_path in part_paths:
            with open(part_path, "rb") as part:
                for chunk in iter(lambda: part.read(DEFAULT_CHUNK_SIZE), b""):
                    out.write(chunk)
                    total += len(chunk)
    temp_dest.replace(dest)
    return total


def _download_ranges(
    url: str,
    dest: Path,
    expected_size: int,
    *,
    workers: int,
    part_size: int,
    chunk_size: int,
    retries: int,
) -> int:
    ranges = _part_ranges(expected_size, part_size)
    parts_dir = dest.with_suffix(dest.suffix + ".parts")
    parts_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Downloading {expected_size/1024**3:.2f} GiB as {len(ranges)} ranged parts "
        f"with {workers} workers",
        flush=True,
    )

    part_paths = [parts_dir / f"part-{index:05d}" for index, _, _ in ranges]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _download_part,
                url,
                part_paths[index],
                start,
                end,
                chunk_size=chunk_size,
                retries=retries,
            ): (index, start, end)
            for index, start, end in ranges
        }
        completed_bytes = 0
        for future in concurrent.futures.as_completed(futures):
            index, start, end = futures[future]
            part_bytes = future.result()
            completed_bytes += part_bytes
            pct = completed_bytes / expected_size * 100
            print(
                f"completed part {index + 1}/{len(ranges)} "
                f"({start}-{end}); aggregate {completed_bytes/1024**3:.2f} GiB ({pct:.1f}%)",
                flush=True,
            )

    total = _assemble_parts(dest, part_paths)
    if total != expected_size:
        raise SystemExit(f"Assembled {total} bytes for {dest}; expected {expected_size}.")
    return total


def download(
    entry: dict,
    dest: Path,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    part_size: int = DEFAULT_PART_SIZE,
    workers: int = 1,
    retries: int = 3,
) -> None:
    file_id = entry["file_id"]
    url = DATAFILE_URL.format(file_id=file_id)
    dest.parent.mkdir(parents=True, exist_ok=True)
    expected_size = entry.get("size_bytes")
    expected_checksum = entry.get("checksum")
    if dest.exists() and expected_size and dest.stat().st_size == int(expected_size):
        if not expected_checksum or _existing_checksum(dest) == expected_checksum:
            print(f"Already present {dest} ({dest.stat().st_size/1024**3:.2f} GiB)")
            return

    if expected_size and workers > 1:
        total = _download_ranges(
            url,
            dest,
            int(expected_size),
            workers=workers,
            part_size=part_size,
            chunk_size=chunk_size,
            retries=retries,
        )
    else:
        total = _download_stream(url, dest, int(expected_size) if expected_size else None, chunk_size)

    if expected_checksum:
        digest = _existing_checksum(dest)
        if digest != expected_checksum:
            raise SystemExit(
                f"Checksum mismatch for {dest}. Expected {expected_checksum}, got {digest}."
            )
    print(f"Saved {dest} ({total/1024**3:.2f} GiB)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a specific PDEBench file using the manifest")
    parser.add_argument(
        "logical_path",
        help="Path as listed in manifest, e.g. '1D/Burgers/Train/...' ",
    )
    parser.add_argument("--out", default="data/pdebench/raw", help="Root output directory")
    parser.add_argument("--manifest", default=str(MANIFEST_PATH), help="Path to pdebench_manifest.yaml")
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("PDEBENCH_DOWNLOAD_WORKERS", "8")),
        help="Parallel ranged download workers when manifest size is known",
    )
    parser.add_argument(
        "--part-size-mib",
        type=int,
        default=int(os.environ.get("PDEBENCH_DOWNLOAD_PART_SIZE_MIB", "256")),
        help="Ranged download part size in MiB",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=int(os.environ.get("PDEBENCH_DOWNLOAD_RETRIES", "3")),
        help="HTTP retry attempts per request",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_manifest(Path(args.manifest))
    entry = find_entry(manifest, args.logical_path)
    dest_root = Path(args.out)
    dest = dest_root / args.logical_path
    download(
        entry,
        dest,
        workers=max(1, args.workers),
        part_size=max(1, args.part_size_mib) * 1024 * 1024,
        retries=max(1, args.retries),
    )


if __name__ == "__main__":
    main()
