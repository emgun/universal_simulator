#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import requests
import yaml

MANIFEST_PATH = Path(__file__).resolve().parents[1] / "docs" / "pdebench_manifest.yaml"
DATAFILE_URL = "https://darus.uni-stuttgart.de/api/access/datafile/{file_id}?format=original"


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


def download(entry: dict, dest: Path, chunk_size: int = 1024 * 1024) -> None:
    file_id = entry["file_id"]
    url = DATAFILE_URL.format(file_id=file_id)
    dest.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    expected_size = entry.get("size_bytes")
    checksum = hashlib.md5()
    total = 0
    with open(dest, "wb") as fh:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                fh.write(chunk)
                checksum.update(chunk)
                total += len(chunk)
                if expected_size:
                    pct = total / expected_size * 100
                    sys.stdout.write(f"\rDownloaded {total/1024**2:.2f} MiB ({pct:.1f}%)")
                    sys.stdout.flush()
    sys.stdout.write("\n")
    expected_checksum = entry.get("checksum")
    if expected_checksum:
        digest = checksum.hexdigest()
        if digest != expected_checksum:
            raise SystemExit(
                f"Checksum mismatch for {dest}. Expected {expected_checksum}, got {digest}."
            )
    print(f"Saved {dest} ({total/1024**3:.2f} GiB)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a specific PDEBench file using the manifest")
    parser.add_argument("logical_path", help="Path as listed in manifest, e.g. '1D/Burgers/Train/...' ")
    parser.add_argument("--out", default="data/pdebench/raw", help="Root output directory")
    parser.add_argument("--manifest", default=str(MANIFEST_PATH), help="Path to pdebench_manifest.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_manifest(Path(args.manifest))
    entry = find_entry(manifest, args.logical_path)
    dest_root = Path(args.out)
    dest = dest_root / args.logical_path
    download(entry, dest)


if __name__ == "__main__":
    main()
