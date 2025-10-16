#!/usr/bin/env python
from __future__ import annotations

"""Build full PDEBench datasets (Train/Val/Test) and tar for upload.

Usage examples:
  python scripts/build_full_artifacts.py burgers --raw_root /path/to/PDEBench/raw --out_root data/pdebench --artifacts artifacts
  python scripts/build_full_artifacts.py advection --raw_root /path/to/PDEBench/raw --out_root data/pdebench --artifacts artifacts
"""

import argparse
import subprocess
import shutil
from pathlib import Path


def _run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def _try_run(cmd: list[str]) -> bool:
    print(" ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError:
        return False


def build_burgers_full(raw_root: Path, out_root: Path, artifacts_dir: Path) -> Path:
    out_root.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    # Train
    _run(["python", "scripts/convert_pdebench_multimodal.py", "burgers1d", "--root", str(raw_root), "--out", str(out_root)])
    # Val/Test
    if not _try_run(["python", "scripts/convert_pdebench.py", "--pattern", str(raw_root / "1D/Burgers/Valid/*.hdf5"), "--out", str(out_root / "burgers1d_val.h5")]):
        # Fallback: if no Valid files present, reuse a slice of train
        src = out_root / "burgers1d_train.h5"
        dst = out_root / "burgers1d_val.h5"
        if src.exists():
            shutil.copy2(src, dst)
            print(f"Fallback: copied {src.name} to {dst.name} (no Valid pattern)")
    if not _try_run(["python", "scripts/convert_pdebench.py", "--pattern", str(raw_root / "1D/Burgers/Test/*.hdf5"),  "--out", str(out_root / "burgers1d_test.h5")]):
        # Fallback: if no Test files present, reuse a slice of train
        src = out_root / "burgers1d_train.h5"
        dst = out_root / "burgers1d_test.h5"
        if src.exists():
            shutil.copy2(src, dst)
            print(f"Fallback: copied {src.name} to {dst.name} (no Test pattern)")
    tar_path = artifacts_dir / "burgers1d_full_v1.tar.gz"
    _run(["tar", "-czf", str(tar_path), "-C", str(out_root), "burgers1d_train.h5", "burgers1d_val.h5", "burgers1d_test.h5"])
    return tar_path


def build_advection_full(raw_root: Path, out_root: Path, artifacts_dir: Path) -> Path:
    out_root.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    # Train
    _run(["python", "scripts/convert_pdebench_multimodal.py", "advection1d", "--root", str(raw_root), "--out", str(out_root)])
    # Val/Test
    if not _try_run(["python", "scripts/convert_pdebench.py", "--pattern", str(raw_root / "1D/Advection/Valid/*.hdf5"), "--out", str(out_root / "advection1d_val.h5")]):
        src = out_root / "advection1d_train.h5"
        dst = out_root / "advection1d_val.h5"
        if src.exists():
            shutil.copy2(src, dst)
            print(f"Fallback: copied {src.name} to {dst.name} (no Valid pattern)")
    if not _try_run(["python", "scripts/convert_pdebench.py", "--pattern", str(raw_root / "1D/Advection/Test/*.hdf5"),  "--out", str(out_root / "advection1d_test.h5")]):
        src = out_root / "advection1d_train.h5"
        dst = out_root / "advection1d_test.h5"
        if src.exists():
            shutil.copy2(src, dst)
            print(f"Fallback: copied {src.name} to {dst.name} (no Test pattern)")
    tar_path = artifacts_dir / "advection1d_full_v1.tar.gz"
    _run(["tar", "-czf", str(tar_path), "-C", str(out_root), "advection1d_train.h5", "advection1d_val.h5", "advection1d_test.h5"])
    return tar_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build full PDEBench datasets and tar them for upload")
    p.add_argument("dataset", choices=["burgers", "advection"]) 
    p.add_argument("--raw_root", default="data/pdebench/raw")
    p.add_argument("--out_root", default="data/pdebench")
    p.add_argument("--artifacts", default="artifacts")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    artifacts_dir = Path(args.artifacts).expanduser().resolve()
    if args.dataset == "burgers":
        path = build_burgers_full(raw_root, out_root, artifacts_dir)
        name = "burgers1d_full_v1"
    else:
        path = build_advection_full(raw_root, out_root, artifacts_dir)
        name = "advection1d_full_v1"
    print(f"Built artifact: {path}")
    print("Upload with:")
    print(f"python scripts/upload_artifact.py {name} dataset {path} --project universal-simulator --entity YOUR_ENTITY")


if __name__ == "__main__":
    main()


