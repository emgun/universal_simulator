"""Unit tests for latent cache precomputation system."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch

from ups.data.parallel_cache import (
    BatchedCacheWriter,
    PreloadedCacheDataset,
    RawFieldDataset,
    build_parallel_latent_loader,
    check_cache_complete,
    check_sufficient_ram,
    estimate_cache_size_mb,
)


class TestRawFieldDataset:
    """Test RawFieldDataset for parallel loading."""

    def test_returns_cached_sample(self):
        """Test that cached samples are loaded correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)

            # Create mock cache file
            cache_path = cache_dir / "sample_00000.pt"
            torch.save({
                "latent": torch.randn(10, 16, 32),
                "params": {"nu": torch.tensor(0.01)},
                "bc": None,
            }, cache_path)

            # Create mock dataset
            mock_base = Mock()
            mock_base.__len__ = Mock(return_value=10)

            dataset = RawFieldDataset(
                mock_base,
                field_name="u",
                cache_dir=cache_dir,
            )

            sample = dataset[0]
            assert sample["cached"] is True
            assert sample["latent"].shape == (10, 16, 32)
            assert "params" in sample

    def test_returns_raw_fields_when_not_cached(self):
        """Test that raw fields are returned when cache miss."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)

            # Create mock dataset
            mock_base = Mock()
            mock_base.__len__ = Mock(return_value=10)
            mock_base.__getitem__ = Mock(return_value={
                "fields": torch.randn(10, 64, 64, 1),
                "params": {"nu": torch.tensor(0.01)},
                "bc": None,
            })

            dataset = RawFieldDataset(
                mock_base,
                field_name="u",
                cache_dir=cache_dir,
            )

            sample = dataset[0]
            assert sample["cached"] is False
            assert "fields" in sample
            assert sample["fields"].shape == (10, 64, 64, 1)


class TestPreloadedCacheDataset:
    """Test PreloadedCacheDataset for RAM caching."""

    def test_preloads_all_cache_files(self):
        """Test that all cache files are loaded into RAM."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            num_samples = 5

            # Create mock cache files
            for idx in range(num_samples):
                cache_path = cache_dir / f"sample_{idx:05d}.pt"
                torch.save({
                    "latent": torch.randn(10, 16, 32),
                    "params": {"nu": torch.tensor(0.01)},
                    "bc": None,
                }, cache_path)

            dataset = PreloadedCacheDataset(
                cache_dir=cache_dir,
                num_samples=num_samples,
            )

            assert len(dataset) == num_samples
            for idx in range(num_samples):
                sample = dataset[idx]
                assert sample.z0.shape[0] > 0  # Has time steps
                assert sample.z1.shape[0] > 0

    def test_fails_on_incomplete_cache(self):
        """Test that incomplete cache raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            num_samples = 5

            # Create only 3 out of 5 cache files
            for idx in range(3):
                cache_path = cache_dir / f"sample_{idx:05d}.pt"
                torch.save({
                    "latent": torch.randn(10, 16, 32),
                    "params": None,
                    "bc": None,
                }, cache_path)

            with pytest.raises(ValueError, match="Cache incomplete"):
                PreloadedCacheDataset(cache_dir=cache_dir, num_samples=num_samples)


class TestCacheUtilities:
    """Test cache utility functions."""

    def test_check_cache_complete(self):
        """Test cache completeness check."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            num_samples = 10

            # Empty cache
            is_complete, count = check_cache_complete(cache_dir, num_samples)
            assert not is_complete
            assert count == 0

            # Partial cache
            for idx in range(5):
                (cache_dir / f"sample_{idx:05d}.pt").touch()
            is_complete, count = check_cache_complete(cache_dir, num_samples)
            assert not is_complete
            assert count == 5

            # Complete cache
            for idx in range(5, num_samples):
                (cache_dir / f"sample_{idx:05d}.pt").touch()
            is_complete, count = check_cache_complete(cache_dir, num_samples)
            assert is_complete
            assert count == num_samples

    def test_estimate_cache_size_mb(self):
        """Test cache size estimation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)

            # Create sample files with known size
            for idx in range(10):
                path = cache_dir / f"sample_{idx:05d}.pt"
                path.write_bytes(b"0" * 1024 * 1024)  # 1MB each

            size_mb = estimate_cache_size_mb(cache_dir, num_samples=10)
            assert 9 < size_mb < 11  # ~10MB total

    def test_check_sufficient_ram(self):
        """Test RAM sufficiency check."""
        # Should have enough RAM for 100MB
        assert check_sufficient_ram(100) is True

        # Should not have enough RAM for 1TB
        assert check_sufficient_ram(1024 * 1024 * 1024) is False


class TestBatchedCacheWriter:
    """Test BatchedCacheWriter for batched cache I/O."""

    def test_batched_write_and_read(self):
        """Test that batched writes and reads work correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            writer = BatchedCacheWriter(cache_dir, batch_size=4)

            # Add 10 samples
            for idx in range(10):
                latent = torch.randn(10, 16, 32)
                params = {"nu": torch.tensor(0.01)}
                bc = None
                writer.add_sample(idx, latent, params, bc)

            # Flush remaining
            writer.flush()

            # Read back samples
            for idx in range(10):
                data = writer.read_sample(idx)
                assert data is not None
                assert data["latent"].shape == (10, 16, 32)

    def test_batch_files_created(self):
        """Test that batch files are created correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            writer = BatchedCacheWriter(cache_dir, batch_size=4)

            # Add 10 samples (should create 3 batch files: 0-3, 4-7, 8-9)
            for idx in range(10):
                latent = torch.randn(10, 16, 32)
                writer.add_sample(idx, latent, {}, None)

            writer.flush()

            # Check batch files exist
            batch_files = list(cache_dir.glob("batch_*.pt"))
            assert len(batch_files) == 3  # batch_00000, batch_00001, batch_00002


class TestTimeoutMechanisms:
    """Test timeout and error handling."""

    def test_dataloader_timeout_parameter(self):
        """Test that timeout is passed to DataLoader."""
        from ups.data.pdebench import PDEBenchDataset
        from ups.io.enc_grid import GridEncoder

        # Mock components
        mock_dataset = Mock(spec=PDEBenchDataset)
        mock_dataset.__len__ = Mock(return_value=10)
        mock_dataset.__getitem__ = Mock(return_value={
            "fields": torch.randn(10, 8, 8, 1),
            "params": None,
            "bc": None,
        })

        mock_encoder = Mock(spec=GridEncoder)
        coords = torch.randn(1, 64, 2)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Patch at the module where it's imported
            with patch('ups.data.parallel_cache.DataLoader') as mock_dataloader:
                # Configure mock to return a mock loader
                mock_dataloader.return_value = Mock()

                _ = build_parallel_latent_loader(
                    dataset=mock_dataset,
                    encoder=mock_encoder,
                    coords=coords,
                    grid_shape=(8, 8),
                    field_name="u",
                    device=torch.device("cpu"),
                    batch_size=4,
                    num_workers=2,
                    cache_dir=Path(tmpdir),
                    timeout=120,  # Should be passed through
                )

                # Verify DataLoader was called with timeout
                assert mock_dataloader.called
                call_kwargs = mock_dataloader.call_args[1]
                assert call_kwargs.get("timeout") == 120


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
