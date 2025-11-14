"""Integration tests for parallel cache precomputation."""

import pytest
import torch

from ups.data.latent_pairs import GridLatentPairDataset, _build_pdebench_dataset
from ups.data.parallel_cache import build_parallel_latent_loader


@pytest.fixture
def mock_pdebench_data(tmp_path):
    """Create mock PDEBench HDF5 data."""
    import h5py

    data_dir = tmp_path / "pdebench"
    data_dir.mkdir()

    # Create mock HDF5 file
    h5_path = data_dir / "burgers1d_train.h5"
    with h5py.File(h5_path, "w") as f:
        # Create mock burgers data: (num_samples, time_steps, spatial_points)
        f.create_dataset("u", data=torch.randn(10, 20, 64).numpy())
        f.create_dataset("nu", data=torch.full((10,), 0.01).numpy())

    return data_dir


@pytest.mark.integration
class TestParallelCacheIntegration:
    """End-to-end tests for parallel cache system."""

    def test_precompute_cache_end_to_end(self, mock_pdebench_data):
        """Test complete cache precomputation workflow."""
        cache_dir = mock_pdebench_data / "cache"
        cache_dir.mkdir()

        # Build dataset and encoder
        cfg = {
            "task": "burgers1d",
            "split": "train",
            "root": str(mock_pdebench_data),
            "normalize": False,
            "latent_dim": 32,
            "latent_len": 16,
        }

        dataset, encoder, grid_shape, field_name = _build_pdebench_dataset(cfg)
        coords = torch.randn(1, 64, 2)

        # Create latent pair dataset with caching
        latent_dataset = GridLatentPairDataset(
            base=dataset,
            encoder=encoder,
            coords=coords,
            grid_shape=grid_shape,
            field_name=field_name,
            device=torch.device("cpu"),
            cache_dir=cache_dir / "burgers1d_train",
        )

        # Access samples to populate cache
        for i in range(min(5, len(latent_dataset))):
            _ = latent_dataset[i]

        # Verify cache files exist
        cache_files = list((cache_dir / "burgers1d_train").glob("sample_*.pt"))
        assert len(cache_files) >= 5

        # Verify cache can be loaded
        for cache_file in cache_files:
            data = torch.load(cache_file, map_location="cpu")
            assert "latent" in data
            assert data["latent"].dim() == 3  # (time, tokens, latent_dim)

    def test_parallel_loader_with_cache(self, mock_pdebench_data):
        """Test parallel loader with cache directory."""
        cache_dir = mock_pdebench_data / "cache"
        cache_dir.mkdir()

        cfg = {
            "task": "burgers1d",
            "split": "train",
            "root": str(mock_pdebench_data),
            "normalize": False,
            "latent_dim": 32,
            "latent_len": 16,
        }

        dataset, encoder, grid_shape, field_name = _build_pdebench_dataset(cfg)
        coords = torch.randn(1, 64, 2)

        # Build parallel loader
        loader = build_parallel_latent_loader(
            dataset=dataset,
            encoder=encoder,
            coords=coords,
            grid_shape=grid_shape,
            field_name=field_name,
            device=torch.device("cpu"),
            batch_size=2,
            num_workers=0,  # Use 0 for deterministic testing
            cache_dir=cache_dir / "burgers1d_train",
            timeout=30,
        )

        # Iterate and verify batches
        batch_count = 0
        for batch in loader:
            assert batch["z0"].dim() == 3  # (batch, tokens, latent_dim)
            assert batch["z1"].dim() == 3
            batch_count += 1
            if batch_count >= 3:
                break

        assert batch_count >= 3


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
