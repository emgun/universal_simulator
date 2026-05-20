#!/usr/bin/env python
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterable

import requests
import yaml

MANIFEST_PATH = Path(__file__).resolve().parents[1] / "docs" / "pdebench_manifest.yaml"
DATAFILE_URL = "https://darus.uni-stuttgart.de/api/access/datafile/{file_id}?format=original"
DEFAULT_CHUNK_SIZE = 1024 * 1024
DEFAULT_PART_SIZE = 256 * 1024 * 1024
DEFAULT_PART_TIMEOUT = 15 * 60
DEFAULT_RETRY_BACKOFF = 15.0
DEFAULT_TIMEOUT = (30, 60)
DEFAULT_SPLIT_AFTER_RETRIES = 2
DEFAULT_MIN_SPLIT_SIZE = 8 * 1024 * 1024


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
    part_timeout: int,
    retry_backoff: float = DEFAULT_RETRY_BACKOFF,
) -> int:
    expected_size = end - start + 1
    if part_path.exists() and part_path.stat().st_size == expected_size:
        return expected_size

    temp_path = part_path.with_suffix(part_path.suffix + ".tmp")
    if temp_path.exists():
        temp_path.unlink()

    headers = {"Range": f"bytes={start}-{end}"}
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        temp_path.unlink(missing_ok=True)
        try:
            response = _request_with_retries(
                url,
                headers=headers,
                stream=True,
                retries=1,
                timeout=(30, max(1, int(part_timeout))),
            )
            status_code = int(response.status_code)
            if status_code != 206:
                raise SystemExit(
                    f"Range request {start}-{end} returned HTTP {status_code}; expected 206."
                )

            total = 0
            started_at = time.monotonic()
            with open(temp_path, "wb") as fh:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if time.monotonic() - started_at > part_timeout:
                        raise TimeoutError(
                            f"Range request {start}-{end} exceeded {part_timeout}s part timeout"
                        )
                    if not chunk:
                        continue
                    fh.write(chunk)
                    total += len(chunk)

            if total != expected_size:
                raise OSError(
                    f"Range request {start}-{end} wrote {total} bytes; expected {expected_size}."
                )
            temp_path.replace(part_path)
            return total
        except SystemExit:
            raise
        except Exception as exc:
            last_error = exc
            print(
                f"part {start}-{end} failed on attempt {attempt}/{retries}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            if attempt < retries and retry_backoff > 0:
                time.sleep(retry_backoff * (2 ** (attempt - 1)))

    temp_path.unlink(missing_ok=True)
    assert last_error is not None
    raise last_error


def _download_part_to_file(
    url: str,
    dest: Path,
    start: int,
    end: int,
    *,
    chunk_size: int,
    retries: int,
    part_timeout: int,
    retry_backoff: float = DEFAULT_RETRY_BACKOFF,
    split_after_retries: int = DEFAULT_SPLIT_AFTER_RETRIES,
    min_split_size: int = DEFAULT_MIN_SPLIT_SIZE,
) -> int:
    expected_size = end - start + 1
    headers = {"Range": f"bytes={start}-{end}"}
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = _request_with_retries(
                url,
                headers=headers,
                stream=True,
                retries=1,
                timeout=(30, max(1, int(part_timeout))),
            )
            status_code = int(response.status_code)
            if status_code != 206:
                raise SystemExit(
                    f"Range request {start}-{end} returned HTTP {status_code}; expected 206."
                )

            total = 0
            offset = start
            started_at = time.monotonic()
            with open(dest, "r+b") as fh:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if time.monotonic() - started_at > part_timeout:
                        raise TimeoutError(
                            f"Range request {start}-{end} exceeded {part_timeout}s part timeout"
                        )
                    if not chunk:
                        continue
                    fh.seek(offset)
                    fh.write(chunk)
                    offset += len(chunk)
                    total += len(chunk)

            if total != expected_size:
                raise OSError(
                    f"Range request {start}-{end} wrote {total} bytes; expected {expected_size}."
                )
            return total
        except SystemExit:
            raise
        except Exception as exc:
            last_error = exc
            print(
                f"part {start}-{end} failed on attempt {attempt}/{retries}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            if (
                attempt >= split_after_retries
                and expected_size > max(1, min_split_size)
                and start < end
            ):
                midpoint = start + expected_size // 2 - 1
                print(
                    f"splitting timed-out range {start}-{end} into "
                    f"{start}-{midpoint} and {midpoint + 1}-{end}",
                    file=sys.stderr,
                    flush=True,
                )
                first = _download_part_to_file(
                    url,
                    dest,
                    start,
                    midpoint,
                    chunk_size=chunk_size,
                    retries=retries,
                    part_timeout=part_timeout,
                    retry_backoff=retry_backoff,
                    split_after_retries=split_after_retries,
                    min_split_size=min_split_size,
                )
                second = _download_part_to_file(
                    url,
                    dest,
                    midpoint + 1,
                    end,
                    chunk_size=chunk_size,
                    retries=retries,
                    part_timeout=part_timeout,
                    retry_backoff=retry_backoff,
                    split_after_retries=split_after_retries,
                    min_split_size=min_split_size,
                )
                return first + second
            if attempt < retries and retry_backoff > 0:
                time.sleep(retry_backoff * (2 ** (attempt - 1)))

    assert last_error is not None
    raise last_error


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


def _range_sidecar_path(temp_dest: Path) -> Path:
    return temp_dest.with_suffix(temp_dest.suffix + ".ranges.json")


def _range_key(start: int, end: int) -> str:
    return f"{start}-{end}"


def _load_completed_ranges(sidecar: Path, expected_size: int) -> set[str]:
    if not sidecar.exists():
        return set()
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return set()
    if int(payload.get("expected_size") or -1) != int(expected_size):
        return set()
    completed = payload.get("completed_ranges") or []
    return {str(item) for item in completed}


def _write_completed_ranges(sidecar: Path, expected_size: int, completed: set[str]) -> None:
    sidecar.write_text(
        json.dumps(
            {
                "expected_size": int(expected_size),
                "completed_ranges": sorted(completed),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _download_ranges(
    url: str,
    dest: Path,
    expected_size: int,
    *,
    workers: int,
    part_size: int,
    chunk_size: int,
    retries: int,
    part_timeout: int,
    retry_backoff: float,
    split_after_retries: int,
    min_split_size: int,
) -> int:
    ranges = _part_ranges(expected_size, part_size)
    temp_dest = dest.with_suffix(dest.suffix + ".tmp")
    sidecar = _range_sidecar_path(temp_dest)
    completed_ranges: set[str] = set()
    if temp_dest.exists() and temp_dest.stat().st_size == expected_size:
        completed_ranges = _load_completed_ranges(sidecar, expected_size)
    else:
        temp_dest.unlink(missing_ok=True)
        sidecar.unlink(missing_ok=True)
    if not temp_dest.exists():
        with open(temp_dest, "wb") as fh:
            fh.truncate(expected_size)
        _write_completed_ranges(sidecar, expected_size, completed_ranges)
    pending_ranges = [
        (index, start, end)
        for index, start, end in ranges
        if _range_key(start, end) not in completed_ranges
    ]
    completed_bytes = sum(
        end - start + 1
        for _, start, end in ranges
        if _range_key(start, end) in completed_ranges
    )
    print(
        f"Downloading {expected_size/1024**3:.2f} GiB as {len(ranges)} ranged parts "
        f"with {workers} workers",
        flush=True,
    )
    if completed_ranges:
        print(
            f"Resuming with {len(completed_ranges)}/{len(ranges)} completed parts "
            f"({completed_bytes/1024**3:.2f} GiB)",
            flush=True,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _download_part_to_file,
                url,
                temp_dest,
                start,
                end,
                chunk_size=chunk_size,
                retries=retries,
                part_timeout=part_timeout,
                retry_backoff=retry_backoff,
                split_after_retries=split_after_retries,
                min_split_size=min_split_size,
            ): (index, start, end)
            for index, start, end in pending_ranges
        }
        for future in concurrent.futures.as_completed(futures):
            index, start, end = futures[future]
            part_bytes = future.result()
            completed_bytes += part_bytes
            completed_ranges.add(_range_key(start, end))
            _write_completed_ranges(sidecar, expected_size, completed_ranges)
            pct = completed_bytes / expected_size * 100
            print(
                f"completed part {index + 1}/{len(ranges)} "
                f"({start}-{end}); aggregate {completed_bytes/1024**3:.2f} GiB ({pct:.1f}%)",
                flush=True,
            )

    total = temp_dest.stat().st_size
    if total != expected_size:
        raise SystemExit(f"Downloaded {total} bytes for {dest}; expected {expected_size}.")
    temp_dest.replace(dest)
    sidecar.unlink(missing_ok=True)
    return expected_size


def download(
    entry: dict,
    dest: Path,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    part_size: int = DEFAULT_PART_SIZE,
    workers: int = 1,
    retries: int = 3,
    part_timeout: int = DEFAULT_PART_TIMEOUT,
    retry_backoff: float = DEFAULT_RETRY_BACKOFF,
    split_after_retries: int = DEFAULT_SPLIT_AFTER_RETRIES,
    min_split_size: int = DEFAULT_MIN_SPLIT_SIZE,
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
            part_timeout=part_timeout,
            retry_backoff=retry_backoff,
            split_after_retries=split_after_retries,
            min_split_size=min_split_size,
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
    parser.add_argument(
        "--part-timeout",
        type=int,
        default=int(os.environ.get("PDEBENCH_DOWNLOAD_PART_TIMEOUT", str(DEFAULT_PART_TIMEOUT))),
        help="Maximum wall-clock seconds for one ranged part attempt",
    )
    parser.add_argument(
        "--retry-backoff",
        type=float,
        default=float(os.environ.get("PDEBENCH_DOWNLOAD_RETRY_BACKOFF", str(DEFAULT_RETRY_BACKOFF))),
        help="Initial seconds to sleep between ranged part retry attempts; doubles per attempt",
    )
    parser.add_argument(
        "--split-after-retries",
        type=int,
        default=int(
            os.environ.get("PDEBENCH_DOWNLOAD_SPLIT_AFTER_RETRIES", str(DEFAULT_SPLIT_AFTER_RETRIES))
        ),
        help="Split a timed-out ranged part into smaller ranges after this many failed attempts",
    )
    parser.add_argument(
        "--min-split-size-mib",
        type=int,
        default=int(os.environ.get("PDEBENCH_DOWNLOAD_MIN_SPLIT_SIZE_MIB", "8")),
        help="Smallest ranged part size, in MiB, eligible for timeout splitting",
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
        part_timeout=max(1, args.part_timeout),
        retry_backoff=max(0.0, args.retry_backoff),
        split_after_retries=max(1, args.split_after_retries),
        min_split_size=max(1, args.min_split_size_mib) * 1024 * 1024,
    )


if __name__ == "__main__":
    main()
