#!/usr/bin/env python3
"""
Debug script to check cache file tensor properties.
Verifies all tensors are CPU-only and safe for shared memory.
"""
import torch
from pathlib import Path
import sys

def check_cache_file(cache_path: Path) -> dict:
    """Load cache file and check tensor properties."""
    try:
        data = torch.load(cache_path, map_location="cpu")

        results = {
            "path": str(cache_path),
            "success": True,
            "tensors": {},
        }

        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                results["tensors"][key] = {
                    "shape": tuple(value.shape),
                    "dtype": str(value.dtype),
                    "device": str(value.device),
                    "is_cuda": value.is_cuda,
                    "is_shared": value.is_shared(),
                    "requires_grad": value.requires_grad,
                }
            elif isinstance(value, dict):
                # Check nested dicts (params, bc)
                for nested_key, nested_val in value.items():
                    if isinstance(nested_val, torch.Tensor):
                        full_key = f"{key}.{nested_key}"
                        results["tensors"][full_key] = {
                            "shape": tuple(nested_val.shape),
                            "dtype": str(nested_val.dtype),
                            "device": str(nested_val.device),
                            "is_cuda": nested_val.is_cuda,
                            "is_shared": nested_val.is_shared(),
                            "requires_grad": nested_val.requires_grad,
                        }

        return results
    except Exception as e:
        return {
            "path": str(cache_path),
            "success": False,
            "error": str(e),
        }

def main():
    # Check first 5 cache files
    cache_dir = Path("data/latent_cache/burgers1d_train")

    if not cache_dir.exists():
        print(f"❌ Cache directory not found: {cache_dir}")
        print("   This script should run on a VastAI instance with cache populated")
        return 1

    cache_files = sorted(cache_dir.glob("sample_*.pt"))[:5]

    if not cache_files:
        print(f"❌ No cache files found in {cache_dir}")
        return 1

    print(f"🔍 Checking {len(cache_files)} cache files...")
    print()

    issues_found = []

    for cache_file in cache_files:
        print(f"📄 {cache_file.name}")
        results = check_cache_file(cache_file)

        if not results["success"]:
            print(f"  ❌ ERROR: {results['error']}")
            issues_found.append(results)
            continue

        for tensor_name, props in results["tensors"].items():
            print(f"  {tensor_name}:")
            print(f"    shape: {props['shape']}")
            print(f"    dtype: {props['dtype']}")
            print(f"    device: {props['device']}")

            # Check for issues
            if props["is_cuda"]:
                print(f"    ⚠️  WARNING: Tensor is on CUDA!")
                issues_found.append((cache_file, tensor_name, "CUDA tensor"))

            if props["is_shared"]:
                print(f"    ℹ️  Already in shared memory")

            if props["requires_grad"]:
                print(f"    ⚠️  WARNING: requires_grad=True")
                issues_found.append((cache_file, tensor_name, "requires_grad"))

        print()

    # Summary
    print("=" * 60)
    if issues_found:
        print(f"❌ Found {len(issues_found)} issues:")
        for issue in issues_found:
            if isinstance(issue, dict):
                print(f"  - {issue['path']}: {issue.get('error', 'Unknown error')}")
            else:
                cache_file, tensor_name, problem = issue
                print(f"  - {cache_file.name}/{tensor_name}: {problem}")
        return 1
    else:
        print("✅ All checked cache files are clean (CPU tensors only)")
        return 0

if __name__ == "__main__":
    sys.exit(main())
