"""Standard and custom attention mechanisms for UPS.

This module provides multiple attention implementations:
- StandardSelfAttention: Standard multi-head self-attention
- FlexSelfAttention: FlexAttention-based self-attention (PyTorch 2.5+)
- ChannelSeparatedSelfAttention: Existing implementation (re-exported)
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ups.core.blocks_pdet import ChannelSeparatedSelfAttention

# Try to import FlexAttention (requires PyTorch 2.5+)
try:
    from torch.nn.attention.flex_attention import flex_attention

    FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    FLEX_ATTENTION_AVAILABLE = False


class FlexSelfAttention(nn.Module):
    """Self-attention using FlexAttention API (PyTorch 2.5+).

    FlexAttention automatically fuses to FlashAttention kernels via torch.compile,
    providing better performance than standard attention implementations.

    Args:
        dim: Input/output dimension.
        num_heads: Number of attention heads.
        qkv_bias: Add bias to Q, K, V projections. Default: False.
        qk_norm: Apply RMSNorm to Q, K before attention. Default: True.
        dropout: Dropout rate (not used, for API compatibility). Default: 0.0.

    Example:
        >>> attn = FlexSelfAttention(dim=192, num_heads=6)
        >>> x = torch.randn(4, 256, 192)  # (B, T, D)
        >>> out = attn(x)  # (4, 256, 192)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = False,
        qk_norm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Q, K, V projections
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        # Optional Q/K normalization
        self.qk_norm = qk_norm
        if qk_norm:
            from ups.core.blocks_pdet import RMSNorm
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

        self.proj = nn.Linear(dim, dim)
        self.dropout_p = dropout

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
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, T, head_dim)
        q, k, v = qkv.unbind(0)  # Each: (B, num_heads, T, head_dim)

        # Apply QK norm
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # FlexAttention (fuses to FlashAttention via compile)
        # Note: FlexAttention expects (B, num_heads, T, head_dim) format
        attn_out = flex_attention(
            q, k, v,
            scale=self.scale,
            enable_gqa=False,  # Not using grouped-query attention
        )  # (B, num_heads, T, head_dim)

        # Reshape and project
        attn_out = attn_out.transpose(1, 2).reshape(B, T, D)  # (B, T, D)
        out = self.proj(attn_out)

        return out


class StandardSelfAttention(nn.Module):
    """Standard multi-head self-attention with optional RMSNorm.

    Follows the standard transformer attention mechanism:
    - Linear projections for Q, K, V
    - Scaled dot-product attention with multiple heads
    - Output projection

    Optionally applies RMSNorm to queries and keys for stability
    (recommended by UPT for 8+ layer networks).

    Can optionally use FlexAttention backend (PyTorch 2.5+) for better performance
    when torch.compile is enabled.

    Args:
        dim: Input/output dimension.
        num_heads: Number of attention heads.
        qkv_bias: Add bias to Q, K, V projections. Default: True.
        qk_norm: Apply RMSNorm to Q, K before attention. Default: False.
        attn_drop: Dropout rate for attention weights. Default: 0.0.
        proj_drop: Dropout rate for output projection. Default: 0.0.
        use_flex: Use FlexAttention backend if available. Default: False.

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
        use_flex: bool = False,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Check if we should use FlexAttention backend
        self.use_flex = use_flex and FLEX_ATTENTION_AVAILABLE
        if use_flex and not FLEX_ATTENTION_AVAILABLE:
            import logging
            logging.warning(
                "FlexAttention requested but not available (requires PyTorch 2.5+). "
                "Falling back to standard attention."
            )

        if self.use_flex:
            # Use FlexAttention backend
            self.attn_impl = FlexSelfAttention(
                dim=dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                dropout=attn_drop,
            )
        else:
            # Standard attention implementation
            # Q, K, V projections (combined for efficiency)
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

            # Optional Q/K normalization (UPT recommendation for stability)
            self.qk_norm = qk_norm
            if qk_norm:
                from ups.core.blocks_pdet import RMSNorm
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
        if self.use_flex:
            # Delegate to FlexAttention backend
            return self.attn_impl(x)

        # Standard attention implementation
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


__all__ = ["StandardSelfAttention", "FlexSelfAttention", "ChannelSeparatedSelfAttention"]
