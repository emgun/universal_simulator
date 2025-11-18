#!/usr/bin/env python3
"""
Minimal test for shared memory tensors in DataLoader with multiple workers.
Tests if .share_memory_() works correctly with multiprocessing.
"""
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import sys

class SharedMemoryTestDataset(Dataset):
    """Test dataset that loads tensors into shared memory."""

    def __init__(self, cache_dir: Path, num_samples: int):
        self.cache_dir = Path(cache_dir)
        self.num_samples = num_samples

        print(f"📦 Loading {num_samples} samples into shared memory...")
        self.cache = {}

        for idx in range(num_samples):
            cache_path = self.cache_dir / f"sample_{idx:05d}.pt"
            if not cache_path.exists():
                raise FileNotFoundError(f"Cache file not found: {cache_path}")

            try:
                data = torch.load(cache_path, map_location="cpu")

                # Explicitly ensure CPU and share memory
                latent = data["latent"].cpu().float().share_memory_()

                # Handle optional fields
                params = data.get("params")
                if params is not None and isinstance(params, torch.Tensor):
                    params = params.cpu().share_memory_()

                bc = data.get("bc")
                if bc is not None and isinstance(bc, torch.Tensor):
                    bc = bc.cpu().share_memory_()

                self.cache[idx] = {
                    "latent": latent,
                    "params": params,
                    "bc": bc,
                }

                if (idx + 1) % 100 == 0:
                    print(f"  Loaded {idx + 1}/{num_samples}...")

            except Exception as e:
                print(f"❌ Failed to load {cache_path}: {e}")
                raise

        print(f"✅ Loaded {len(self.cache)} samples into shared memory")

    def __len__(self) -> int:
        return len(self.cache)

    def __getitem__(self, idx: int):
        """Get sample from shared memory cache."""
        if idx not in self.cache:
            raise IndexError(f"Sample {idx} not in cache")

        sample = self.cache[idx]

        # Return tensors - they should be accessible from worker processes
        return {
            "idx": idx,
            "latent": sample["latent"],
            "params": sample.get("params"),
            "bc": sample.get("bc"),
        }

def worker_init_fn(worker_id):
    """Called in each worker process."""
    print(f"Worker {worker_id} initialized (PID: {torch.utils.data.get_worker_info().id})")

def test_dataloader(cache_dir: Path, num_workers: int = 4, batch_size: int = 8):
    """Test DataLoader with shared memory dataset."""

    # Count available samples
    cache_files = sorted(cache_dir.glob("sample_*.pt"))
    num_samples = min(100, len(cache_files))  # Test with first 100 samples

    if num_samples == 0:
        print(f"❌ No cache files found in {cache_dir}")
        return False

    print(f"🧪 Testing DataLoader with {num_workers} workers, batch_size={batch_size}")
    print(f"   Cache: {cache_dir}")
    print(f"   Samples: {num_samples}")
    print()

    try:
        # Create dataset
        dataset = SharedMemoryTestDataset(cache_dir, num_samples)

        # Create DataLoader
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn if num_workers > 0 else None,
        )

        print()
        print("🔄 Iterating batches...")

        # Test iteration
        for batch_idx, batch in enumerate(loader):
            print(f"  Batch {batch_idx}: {batch['idx'].tolist()}")

            # Verify tensors
            assert batch["latent"].shape[0] == len(batch["idx"])
            assert batch["latent"].device.type == "cpu"

            if batch_idx >= 2:  # Test first 3 batches
                break

        print()
        print("✅ DataLoader test PASSED!")
        print(f"   Shared memory + {num_workers} workers works correctly")
        return True

    except Exception as e:
        print()
        print(f"❌ DataLoader test FAILED!")
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    cache_dir = Path("data/latent_cache/burgers1d_train")

    if not cache_dir.exists():
        print(f"❌ Cache directory not found: {cache_dir}")
        print("   This script should run on a VastAI instance or after local cache creation")
        return 1

    print("=" * 70)
    print("Shared Memory DataLoader Test")
    print("=" * 70)
    print()

    # Test with 0 workers first (baseline)
    print("Test 1: num_workers=0 (baseline)")
    print("-" * 70)
    if not test_dataloader(cache_dir, num_workers=0, batch_size=4):
        return 1

    print()
    print()

    # Test with 4 workers
    print("Test 2: num_workers=4 (multiprocessing)")
    print("-" * 70)
    if not test_dataloader(cache_dir, num_workers=4, batch_size=8):
        return 1

    print()
    print("=" * 70)
    print("✅ All tests passed! Shared memory implementation works.")
    print("=" * 70)
    return 0

if __name__ == "__main__":
    sys.exit(main())
