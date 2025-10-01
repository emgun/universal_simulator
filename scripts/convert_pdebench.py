#!/usr/bin/env python
from __future__ import annotations

import argparse
from glob import glob
from pathlib import Path

from ups.data.convert_pdebench import convert_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert PDEBench HDF5 shards into UPS-ready HDF5 file.")
    parser.add_argument("--pattern", required=True, help="Glob pattern for input HDF5 files")
    parser.add_argument("--out", required=True, help="Output HDF5 path (e.g. data/pdebench/burgers1d_train.h5)")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of files to include")
    parser.add_argument("--samples", type=int, default=None, help="Number of solutions to take from each file")
    args = parser.parse_args()

    files = sorted(glob(args.pattern))
    if not files:
        raise SystemExit(f"No files matched pattern: {args.pattern}")

    written = convert_files(
        [Path(f) for f in files],
        Path(args.out),
        limit=args.limit,
        sample_size=args.samples,
    )
    used = len(files) if args.limit is None else min(args.limit, len(files))
    print(f"Wrote {written} samples to {args.out} (from {used} files)")


if __name__ == "__main__":
    main()
