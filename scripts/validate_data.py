#!/usr/bin/env python3
"""
Validate training data integrity before starting expensive GPU runs.

Checks:
- Data files exist and are not corrupted
- HDF5/Zarr files are valid and readable
- Data shapes match expected dimensions
- Statistics are reasonable (no NaNs, infs)

Usage:
    python scripts/validate_data.py configs/train_burgers_32dim.yaml
    python scripts/validate_data.py --data-root data/pdebench --task burgers1d --split train
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional
import warnings

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

def check_file_exists(filepath: Path) -> Tuple[bool, str]:
    """Check if file exists and is accessible."""
    if not filepath.exists():
        return False, f"File not found: {filepath}"
    
    if not filepath.is_file():
        return False, f"Path is not a file: {filepath}"
    
    if filepath.stat().st_size == 0:
        return False, f"File is empty (0 bytes): {filepath}"
    
    return True, f"File exists ({filepath.stat().st_size / 1e6:.1f} MB)"


def validate_hdf5_file(filepath: Path) -> Tuple[bool, str, Optional[Dict]]:
    """Validate HDF5 file integrity and structure."""
    try:
        import h5py
    except ImportError:
        return False, "h5py not installed", None
    
    try:
        with h5py.File(filepath, 'r') as f:
            # Check if file can be opened
            keys = list(f.keys())
            
            if not keys:
                return False, "HDF5 file has no datasets", None
            
            # Try to read first dataset
            first_key = keys[0]
            data = f[first_key]
            shape = data.shape
            dtype = data.dtype
            
            # Try to read a small chunk
            if len(shape) > 0:
                chunk = data[0] if shape[0] > 0 else None
                if chunk is None:
                    return False, "Cannot read data from HDF5", None
            
            info = {
                "keys": keys,
                "first_dataset": first_key,
                "shape": shape,
                "dtype": str(dtype),
                "num_datasets": len(keys),
            }
            
            return True, f"Valid HDF5 with {len(keys)} datasets", info
            
    except OSError as e:
        return False, f"HDF5 file corrupted or invalid: {e}", None
    except Exception as e:
        return False, f"Error reading HDF5: {e}", None


def validate_data_statistics(filepath: Path, task: str) -> Tuple[bool, str]:
    """Check for NaNs, Infs, and reasonable value ranges."""
    try:
        import h5py
        import numpy as np
    except ImportError:
        return True, "Skipping statistics check (dependencies not available)"
    
    try:
        with h5py.File(filepath, 'r') as f:
            # Sample first dataset
            keys = list(f.keys())
            if not keys:
                return False, "No datasets to validate"
            
            first_key = keys[0]
            data = f[first_key]
            
            # Read a sample (first 10 timesteps or full if smaller)
            sample_size = min(10, data.shape[0]) if len(data.shape) > 0 else 1
            sample = np.array(data[:sample_size])
            
            # Check for NaNs
            if np.any(np.isnan(sample)):
                nan_pct = np.mean(np.isnan(sample)) * 100
                return False, f"Data contains NaNs ({nan_pct:.1f}%)"
            
            # Check for Infs
            if np.any(np.isinf(sample)):
                inf_pct = np.mean(np.isinf(sample)) * 100
                return False, f"Data contains Infs ({inf_pct:.1f}%)"
            
            # Check value range is reasonable (not all zeros, not absurdly large)
            data_min = float(np.min(sample))
            data_max = float(np.max(sample))
            data_mean = float(np.mean(sample))
            data_std = float(np.std(sample))
            
            if data_min == data_max:
                return False, f"Data is constant (all values = {data_min})"
            
            if abs(data_max) > 1e10:
                return False, f"Data has extreme values (max = {data_max:.2e})"
            
            stats_msg = f"Stats OK (min={data_min:.3f}, max={data_max:.3f}, mean={data_mean:.3f}, std={data_std:.3f})"
            return True, stats_msg
            
    except Exception as e:
        return False, f"Error checking statistics: {e}"


def validate_data_shapes(filepath: Path, expected_task: str) -> Tuple[bool, str]:
    """Validate data shapes match expected format for task."""
    try:
        import h5py
    except ImportError:
        return True, "Skipping shape validation (h5py not available)"
    
    try:
        with h5py.File(filepath, 'r') as f:
            keys = list(f.keys())
            if not keys:
                return False, "No datasets found"
            
            first_key = keys[0]
            shape = f[first_key].shape
            
            # Expected shapes for common tasks
            expected_dims = {
                "burgers1d": (3, 4),  # (T, X, C) or (T, C, X)
                "wave2d": (4,),       # (T, H, W, C)
                "heat2d": (4,),
                "advection": (3, 4),
            }
            
            if expected_task in expected_dims:
                expected = expected_dims[expected_task]
                if len(shape) not in expected:
                    return False, f"Unexpected shape {shape} for {expected_task} (expected {len(expected)}D)"
            
            return True, f"Shape {shape} looks reasonable"
            
    except Exception as e:
        return False, f"Error checking shapes: {e}"


def validate_dataset(data_root: Path, task: str, split: str) -> bool:
    """Validate entire dataset for a task/split."""
    print(f"\n{'='*70}")
    print(f"🔍 Validating Dataset: {task} ({split} split)")
    print(f"{'='*70}")
    
    # Determine expected filename
    expected_filename = f"{task}_{split}.h5"
    filepath = data_root / expected_filename
    
    all_passed = True
    
    # Check 1: File exists
    print("\n📁 Check 1: File Existence")
    print("─" * 70)
    exists, msg = check_file_exists(filepath)
    status = "✅" if exists else "❌"
    print(f"{status} {msg}")
    if not exists:
        all_passed = False
        return all_passed
    
    # Check 2: HDF5 integrity
    print("\n🔬 Check 2: HDF5 File Integrity")
    print("─" * 70)
    valid, msg, info = validate_hdf5_file(filepath)
    status = "✅" if valid else "❌"
    print(f"{status} {msg}")
    if info:
        print(f"   First dataset: {info['first_dataset']}")
        print(f"   Shape: {info['shape']}")
        print(f"   Dtype: {info['dtype']}")
    if not valid:
        all_passed = False
        return all_passed
    
    # Check 3: Data statistics
    print("\n📊 Check 3: Data Statistics")
    print("─" * 70)
    stats_ok, stats_msg = validate_data_statistics(filepath, task)
    status = "✅" if stats_ok else "❌"
    print(f"{status} {stats_msg}")
    if not stats_ok:
        all_passed = False
    
    # Check 4: Data shapes
    print("\n📐 Check 4: Data Shapes")
    print("─" * 70)
    shapes_ok, shapes_msg = validate_data_shapes(filepath, task)
    status = "✅" if shapes_ok else "❌"
    print(f"{status} {shapes_msg}")
    if not shapes_ok:
        all_passed = False
    
    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Validate training data integrity")
    parser.add_argument("config", nargs='?', help="Training config YAML (optional)")
    parser.add_argument("--data-root", type=str, help="Data root directory")
    parser.add_argument("--task", type=str, help="Task name (e.g., burgers1d)")
    parser.add_argument("--split", type=str, default="train", help="Data split (train/val/test)")
    parser.add_argument("--all-splits", action="store_true", help="Validate train, val, and test")
    args = parser.parse_args()
    
    # Load config if provided
    if args.config:
        try:
            from ups.utils.config_loader import load_config_with_includes
            cfg = load_config_with_includes(args.config)
            data_root = Path(cfg.get("data", {}).get("root", "data/pdebench"))
            task = cfg.get("data", {}).get("task", "burgers1d")
            split = args.split if not args.all_splits else None
        except Exception as e:
            print(f"❌ Failed to load config: {e}")
            sys.exit(1)
    else:
        # Use command-line args
        if not args.data_root or not args.task:
            print("❌ Either provide a config file or specify --data-root and --task")
            parser.print_help()
            sys.exit(1)
        
        data_root = Path(args.data_root)
        task = args.task
        split = args.split if not args.all_splits else None
    
    print(f"\n{'='*70}")
    print("🔍 DATA VALIDATION")
    print(f"{'='*70}")
    print(f"Data Root: {data_root}")
    print(f"Task: {task}")
    
    # Validate requested splits
    if args.all_splits:
        splits = ["train", "val", "test"]
    else:
        splits = [split]
    
    all_passed = True
    for split_name in splits:
        passed = validate_dataset(data_root, task, split_name)
        all_passed = all_passed and passed
    
    # Final summary
    print(f"\n{'='*70}")
    if all_passed:
        print("✅ All data validation checks PASSED")
        print(f"{'='*70}")
        print("\n💡 Data is ready for training!")
        sys.exit(0)
    else:
        print("❌ Some data validation checks FAILED")
        print(f"{'='*70}")
        print("\n⚠️  Fix data issues before training to avoid failures")
        sys.exit(1)


if __name__ == "__main__":
    main()

