from __future__ import annotations

"""Shifted-window utilities used by the PDE-Transformer core.

This module provides helper functions for slicing tensors into fixed-size
windows (optionally with a cyclic shift) and then stitching them back together.
It mirrors the mechanics introduced by Swin Transformers, which are well-suited
for local attention over spatial grids.

The implementation currently targets 2-D grids represented in ``(B, H, W, C)``
format (batch, height, width, channel). That layout aligns with the rest of the
UPS codebase, where channels are stored last for easier interop with NumPy or
plotting utilities. It can be generalised to higher dimensions if required by
future milestones.
"""

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
from torch import nn


def _to_2tuple(value: Iterable[int] | int) -> Tuple[int, int]:
    if isinstance(value, Iterable):
        value = tuple(value)
        if len(value) != 2:
            raise ValueError("Expected exactly two elements for a 2-D window")
        return int(value[0]), int(value[1])
    return int(value), int(value)


@dataclass(frozen=True)
class WindowPartitionInfo:
    """Metadata needed to reverse a window partition operation."""

    batch: int
    height: int
    width: int
    channels: int
    window_size: Tuple[int, int]
    shift_size: Tuple[int, int]


def partition_windows(
    tensor: torch.Tensor,
    window_size: Iterable[int] | int,
    shift_size: Iterable[int] | int = (0, 0),
) -> Tuple[torch.Tensor, WindowPartitionInfo]:
    """Slice a ``(B, H, W, C)`` tensor into flattened windows.

    Parameters
    ----------
    tensor:
        Input grid-shaped tensor.
    window_size:
        Size of each window (height, width). If a single integer is provided,
        the same value is used for both spatial dimensions.
    shift_size:
        Optional cyclic shift. When non-zero, the tensor is rolled negatively
        before being partitioned; the inverse roll is applied by
        :func:`merge_windows`.
    """

    if tensor.dim() != 4:
        raise ValueError("Expected tensor shape (B, H, W, C)")
    B, H, W, C = tensor.shape
    window_size = _to_2tuple(window_size)
    shift_size = _to_2tuple(shift_size)

    if H % window_size[0] != 0 or W % window_size[1] != 0:
        raise ValueError(
            f"Height {H} and width {W} must be divisible by the window size {window_size}."
        )

    if any(s >= w for s, w in zip(shift_size, window_size)):
        raise ValueError("shift_size must be smaller than window_size in each dimension")

    if shift_size != (0, 0):
        tensor = torch.roll(tensor, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))

    h_windows = H // window_size[0]
    w_windows = W // window_size[1]

    tensor = tensor.view(
        B,
        h_windows,
        window_size[0],
        w_windows,
        window_size[1],
        C,
    )
    # Reorder the axes so each window is contiguous, then flatten spatial pixels.
    windows = (
        tensor.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(B * h_windows * w_windows, window_size[0] * window_size[1], C)
    )

    info = WindowPartitionInfo(
        batch=B,
        height=H,
        width=W,
        channels=C,
        window_size=window_size,
        shift_size=shift_size,
    )
    return windows, info


def merge_windows(windows: torch.Tensor, info: WindowPartitionInfo) -> torch.Tensor:
    """Reconstruct a tensor from flattened windows using stored metadata."""

    B, H, W, C = info.batch, info.height, info.width, info.channels
    window_size = info.window_size
    shift_size = info.shift_size

    if windows.dim() != 3:
        raise ValueError("Expected window tensor with shape (num_windows, window_area, C)")

    window_area = window_size[0] * window_size[1]
    if windows.shape[-1] != C or windows.shape[1] != window_area:
        raise ValueError("Window tensor shape is incompatible with the provided WindowPartitionInfo")

    h_windows = H // window_size[0]
    w_windows = W // window_size[1]

    tensor = (
        windows.view(B, h_windows, w_windows, window_size[0], window_size[1], C)
        .permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(B, H, W, C)
    )

    if shift_size != (0, 0):
        tensor = torch.roll(tensor, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))
    return tensor


class LogSpacedRelativePositionBias(nn.Module):
    """Head-specific relative position bias with log-spaced scaling.

    The module precomputes the signed, logarithmically scaled offsets between
    every pair of positions inside a window. Each attention head learns a small
    set of weights that linearly combine those offsets, producing a bias matrix
    shaped ``(num_heads, window_area, window_area)`` suitable for additive use in
    attention score calculations.
    """

    def __init__(self, window_size: Iterable[int] | int, num_heads: int) -> None:
        super().__init__()
        self.window_size = _to_2tuple(window_size)
        self.num_heads = num_heads

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1)
        coords_flat = coords.view(-1, 2).float()  # (window_area, 2)
        rel = coords_flat[:, None, :] - coords_flat[None, :, :]  # (N, N, 2)
        # Signed log distance keeps directionality but compresses large offsets.
        log_rel = torch.sign(rel) * torch.log1p(rel.abs())
        self.register_buffer("log_relative_positions", log_rel)

        self.weight = nn.Parameter(torch.zeros(num_heads, 2))
        self.bias = nn.Parameter(torch.zeros(num_heads, 1, 1))

    def forward(self) -> torch.Tensor:
        rel = self.log_relative_positions  # (N, N, 2)
        bias = torch.einsum("hd,ijd->hij", self.weight, rel)
        bias = bias + self.bias
        return bias
