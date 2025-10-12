#!/usr/bin/env python
"""Create Burgers val/test splits from an existing train shard."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py


def slice_dataset(src_path: Path, val_path: Path, test_path: Path, val_count: int, test_count: int) -> None:
    if not src_path.exists():
        raise FileNotFoundError(src_path)

    with h5py.File(src_path, "r") as src_file:
        if "data" not in src_file:
            raise KeyError(f"{src_path} does not contain a 'data' dataset.")
        data = src_file["data"]
        total = data.shape[0]
        if total < val_count + test_count:
            raise ValueError(f"Need at least {val_count + test_count} samples, found {total} in {src_path}.")

        val_data = data[:val_count]
        test_data = data[val_count : val_count + test_count]

    val_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(val_path, "w") as val_file:
        val_file.create_dataset("data", data=val_data, compression="gzip", compression_opts=3)

    test_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(test_path, "w") as test_file:
        test_file.create_dataset("data", data=test_data, compression="gzip", compression_opts=3)

    print(f"Wrote {val_path} ({val_count} samples)")
    print(f"Wrote {test_path} ({test_count} samples)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate val/test splits from a Burgers1D train shard.")
    parser.add_argument("--train", required=True, help="Path to burgers1d_train_000.h5")
    parser.add_argument("--val", required=True, help="Output path for burgers1d_val.h5")
    parser.add_argument("--test", required=True, help="Output path for burgers1d_test.h5")
    parser.add_argument("--val-count", type=int, default=200, help="Number of validation trajectories")
    parser.add_argument("--test-count", type=int, default=200, help="Number of test trajectories")
    args = parser.parse_args()

    slice_dataset(Path(args.train), Path(args.val), Path(args.test), args.val_count, args.test_count)


if __name__ == "__main__":
    main()
