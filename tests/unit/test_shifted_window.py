import torch

from ups.core import (
    LogSpacedRelativePositionBias,
    WindowPartitionInfo,
    merge_windows,
    partition_windows,
)


def test_partition_merge_no_shift():
    tensor = torch.randn(2, 8, 8, 4)
    windows, info = partition_windows(tensor, window_size=4)
    assert windows.shape == (2 * 4, 16, 4)
    reconstructed = merge_windows(windows, info)
    assert torch.allclose(reconstructed, tensor)


def test_partition_merge_with_shift():
    tensor = torch.randn(1, 8, 8, 3)
    windows, info = partition_windows(tensor, window_size=4, shift_size=2)
    reconstructed = merge_windows(windows, info)
    assert torch.allclose(reconstructed, tensor)


def test_log_spaced_relative_bias_symmetry():
    bias_module = LogSpacedRelativePositionBias(window_size=4, num_heads=3)
    bias = bias_module()
    assert bias.shape == (3, 16, 16)
    # Distances are symmetric; with zero-initialised weights we expect zeros.
    assert torch.allclose(bias, torch.zeros_like(bias))

    # After a gradient update the bias should change.
    bias_module.weight.data.fill_(0.5)
    updated_bias = bias_module()
    assert not torch.allclose(updated_bias, torch.zeros_like(updated_bias))
