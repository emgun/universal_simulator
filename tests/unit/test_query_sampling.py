"""Unit tests for query sampling utilities."""

import torch
import pytest
from ups.training.query_sampling import (
    sample_uniform_queries,
    sample_stratified_queries,
    apply_query_sampling,
)


def test_uniform_sampling():
    """Test uniform random sampling without replacement."""
    total_points = 1024
    num_queries = 256
    indices = sample_uniform_queries(total_points, num_queries)

    assert indices.shape == (num_queries,)
    assert indices.max() < total_points
    assert indices.min() >= 0
    assert len(indices.unique()) == num_queries  # No duplicates


def test_uniform_sampling_all_points():
    """Test uniform sampling when num_queries >= total_points."""
    total_points = 100
    num_queries = 150  # Request more than available
    indices = sample_uniform_queries(total_points, num_queries)

    assert indices.shape == (total_points,)
    assert torch.equal(indices, torch.arange(total_points))


def test_stratified_sampling():
    """Test stratified sampling for coverage of all regions."""
    grid_shape = (32, 32)  # 1024 points
    num_queries = 256
    indices = sample_stratified_queries(grid_shape, num_queries)

    assert indices.shape == (num_queries,)
    assert indices.max() < 1024
    assert indices.min() >= 0

    # Verify coverage: at least one sample from each quadrant
    # Top-left quadrant: rows 0-15, cols 0-15
    top_left = (indices < 512) & (indices % 32 < 16)
    assert top_left.any(), "No samples from top-left quadrant"

    # Top-right quadrant: rows 0-15, cols 16-31
    top_right = (indices < 512) & (indices % 32 >= 16)
    assert top_right.any(), "No samples from top-right quadrant"

    # Bottom-left quadrant: rows 16-31, cols 0-15
    bottom_left = (indices >= 512) & (indices % 32 < 16)
    assert bottom_left.any(), "No samples from bottom-left quadrant"

    # Bottom-right quadrant: rows 16-31, cols 16-31
    bottom_right = (indices >= 512) & (indices % 32 >= 16)
    assert bottom_right.any(), "No samples from bottom-right quadrant"


def test_stratified_sampling_all_points():
    """Test stratified sampling when num_queries >= total_points."""
    grid_shape = (10, 10)
    num_queries = 150  # Request more than available
    indices = sample_stratified_queries(grid_shape, num_queries)

    assert indices.shape == (100,)
    assert torch.equal(indices, torch.arange(100))


def test_apply_query_sampling_uniform():
    """Test apply_query_sampling with uniform strategy."""
    B, H, W, C = 4, 64, 64, 1
    N = H * W
    fields = {"u": torch.randn(B, N, C)}
    coords = torch.rand(B, N, 2)

    sampled_fields, sampled_coords = apply_query_sampling(
        fields, coords, num_queries=1024, strategy="uniform"
    )

    assert sampled_fields["u"].shape == (B, 1024, C)
    assert sampled_coords.shape == (B, 1024, 2)

    # Verify sampling is consistent (same indices used for fields and coords)
    # We can't test exact values due to randomness, but shapes should match


def test_apply_query_sampling_stratified():
    """Test apply_query_sampling with stratified strategy."""
    B, H, W, C = 4, 64, 64, 1
    N = H * W
    fields = {"u": torch.randn(B, N, C), "v": torch.randn(B, N, C)}
    coords = torch.rand(B, N, 2)

    sampled_fields, sampled_coords = apply_query_sampling(
        fields, coords, num_queries=1024, strategy="stratified", grid_shape=(H, W)
    )

    assert sampled_fields["u"].shape == (B, 1024, C)
    assert sampled_fields["v"].shape == (B, 1024, C)
    assert sampled_coords.shape == (B, 1024, 2)


def test_apply_query_sampling_no_sampling():
    """Test apply_query_sampling when num_queries >= N (no sampling)."""
    B, H, W, C = 2, 32, 32, 1
    N = H * W
    fields = {"u": torch.randn(B, N, C)}
    coords = torch.rand(B, N, 2)

    sampled_fields, sampled_coords = apply_query_sampling(
        fields, coords, num_queries=N, strategy="uniform"
    )

    # Should return original tensors (no sampling)
    assert sampled_fields["u"].shape == (B, N, C)
    assert sampled_coords.shape == (B, N, 2)
    assert torch.equal(sampled_fields["u"], fields["u"])
    assert torch.equal(sampled_coords, coords)


def test_apply_query_sampling_invalid_strategy():
    """Test apply_query_sampling with invalid strategy."""
    B, N, C = 2, 1024, 1
    fields = {"u": torch.randn(B, N, C)}
    coords = torch.rand(B, N, 2)

    with pytest.raises(ValueError, match="Unknown sampling strategy"):
        apply_query_sampling(fields, coords, num_queries=256, strategy="invalid")


def test_stratified_sampling_missing_grid_shape():
    """Test that stratified strategy requires grid_shape."""
    B, N, C = 2, 1024, 1
    fields = {"u": torch.randn(B, N, C)}
    coords = torch.rand(B, N, 2)

    with pytest.raises(ValueError, match="grid_shape required"):
        apply_query_sampling(
            fields, coords, num_queries=256, strategy="stratified", grid_shape=None
        )


def test_device_consistency():
    """Test that sampled indices respect device placement."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")
    total_points = 1024
    num_queries = 256

    indices = sample_uniform_queries(total_points, num_queries, device=device)
    assert indices.device.type == "cuda"

    grid_shape = (32, 32)
    indices = sample_stratified_queries(grid_shape, num_queries, device=device)
    assert indices.device.type == "cuda"


def test_sampling_determinism():
    """Test that sampling is deterministic with manual seed."""
    torch.manual_seed(42)
    indices1 = sample_uniform_queries(1000, 100)

    torch.manual_seed(42)
    indices2 = sample_uniform_queries(1000, 100)

    assert torch.equal(indices1, indices2), "Sampling should be deterministic with same seed"


def test_edge_case_single_query():
    """Test sampling with num_queries=1."""
    total_points = 1024
    indices = sample_uniform_queries(total_points, num_queries=1)
    assert indices.shape == (1,)
    assert 0 <= indices.item() < total_points


def test_edge_case_small_grid():
    """Test stratified sampling on very small grid."""
    grid_shape = (4, 4)  # 16 points
    num_queries = 8
    indices = sample_stratified_queries(grid_shape, num_queries)

    assert indices.shape == (num_queries,)
    assert indices.max() < 16
    assert len(indices.unique()) == num_queries  # No duplicates
