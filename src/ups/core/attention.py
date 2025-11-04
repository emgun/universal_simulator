"""Standard and custom attention mechanisms for UPS.

This module provides multiple attention implementations:
- StandardSelfAttention: Standard multi-head self-attention
- ChannelSeparatedSelfAttention: Existing implementation (re-exported)
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from src.ups.core.blocks_pdet import ChannelSeparatedSelfAttention


class StandardSelfAttention(nn.Module):
    """Standard multi-head self-attention with optional RMSNorm.

    Follows the standard transformer attention mechanism:
    - Linear projections for Q, K, V
    - Scaled dot-product attention with multiple heads
    - Output projection

    Optionally applies RMSNorm to queries and keys for stability
    (recommended by UPT for 8+ layer networks).

    Args:
        dim: Input/output dimension.
        num_heads: Number of attention heads.
        qkv_bias: Add bias to Q, K, V projections. Default: True.
        qk_norm: Apply RMSNorm to Q, K before attention. Default: False.
        attn_drop: Dropout rate for attention weights. Default: 0.0.
        proj_drop: Dropout rate for output projection. Default: 0.0.

    Example:
        >>> attn = StandardSelfAttention(dim=192, num_heads=6)
        >>> x = torch.randn(4, 256, 192)  # (B, T, D)
        >>> out = attn(x)  # (4, 256, 192)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Q, K, V projections (combined for efficiency)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        # Optional Q/K normalization (UPT recommendation for stability)
        self.qk_norm = qk_norm
        if qk_norm:
            from src.ups.core.blocks_pdet import RMSNorm
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, T, D).

        Returns:
            Output tensor of shape (B, T, D).
        """
        B, T, D = x.shape

        # Project to Q, K, V
        qkv = self.qkv(x)  # (B, T, 3*D)
        qkv = qkv.reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, head_dim)
        q, k, v = qkv.unbind(0)  # Each: (B, H, T, head_dim)

        # Optional Q/K normalization
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Scaled dot-product attention (Flash Attention compatible)
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            is_causal=False,
        )  # (B, H, T, head_dim)

        # Reshape and project output
        attn_out = attn_out.transpose(1, 2)  # (B, T, H, head_dim)
        attn_out = attn_out.reshape(B, T, D)  # (B, T, D)

        out = self.proj(attn_out)
        out = self.proj_drop(out)

        return out


__all__ = ["StandardSelfAttention", "ChannelSeparatedSelfAttention"]
