"""Drop-path (stochastic depth) regularization.

Reference: Deep Networks with Stochastic Depth (arXiv:1603.09382)
UPT recommendation: 0.1-0.2 for 8-12 layer networks
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DropPath(nn.Module):
    """Drop paths (stochastic depth) per sample.

    When drop_prob > 0, randomly drops entire paths (residual branches)
    during training. At test time, scales by (1 - drop_prob) for expected value.

    Args:
        drop_prob: Probability of dropping a path. Default: 0.0 (disabled).
        scale_by_keep: If True, scale by 1/(1-drop_prob) during training.
                       Default: True (standard practice).

    Example:
        >>> drop_path = DropPath(drop_prob=0.1)
        >>> residual = some_layer(x)
        >>> x = x + drop_path(residual)  # Randomly drop residual 10% of time
    """

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply drop-path to input tensor.

        Args:
            x: Input tensor of shape (B, *, D) where * can be any dims.

        Returns:
            Tensor with same shape, possibly dropped or scaled.
        """
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1.0 - self.drop_prob

        # Create random tensor with shape (B, 1, ..., 1) to broadcast
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(
            shape, dtype=x.dtype, device=x.device
        )
        random_tensor.floor_()  # Binarize: 0 or 1

        if self.scale_by_keep:
            # Scale by 1/keep_prob to maintain expected value
            output = x.div(keep_prob) * random_tensor
        else:
            output = x * random_tensor

        return output

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob}"
