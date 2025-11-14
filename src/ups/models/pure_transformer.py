"""Pure stacked transformer approximator for UPT.

This module implements a simplified alternative to the U-shaped PDETransformer.
Instead of hierarchical downsampling/upsampling with skip connections, this uses
a simple stack of transformer layers operating on a fixed number of latent tokens.

Recommended for 256-512 latent tokens (UPT guideline).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from ups.core.attention import ChannelSeparatedSelfAttention, StandardSelfAttention
from ups.core.drop_path import DropPath


@dataclass
class PureTransformerConfig:
    """Configuration for pure stacked transformer.

    Args:
        input_dim: Latent dimension (must match latent.dim from config).
        hidden_dim: Hidden dimension for transformer layers.
        depth: Number of transformer layers (e.g., 4, 8, 12).
        num_heads: Number of attention heads per layer.
        attention_type: Type of attention mechanism.
            - "standard": Standard multi-head self-attention
            - "channel_separated": Channel-separated attention (existing)
        group_size: For channel_separated attention, size of channel groups.
                    Ignored for standard attention.
        mlp_ratio: FFN hidden dimension ratio (hidden = dim * mlp_ratio).
        qk_norm: Apply RMSNorm to Q, K in standard attention.
        drop_path: Stochastic depth rate (linearly scaled across depth).
                   0.0 = disabled, 0.1-0.2 = recommended for deep networks.
        dropout: Dropout rate for attention and FFN. Default: 0.0.

    Example configs:
        Small (4 layers, 128 tokens):
            input_dim=192, hidden_dim=192, depth=4, num_heads=4, drop_path=0.0

        Medium (8 layers, 256 tokens):
            input_dim=256, hidden_dim=256, depth=8, num_heads=6, drop_path=0.1

        Large (12 layers, 512 tokens):
            input_dim=384, hidden_dim=384, depth=12, num_heads=8, drop_path=0.15
    """

    input_dim: int
    hidden_dim: int
    depth: int
    num_heads: int
    attention_type: Literal["standard", "channel_separated"] = "standard"
    group_size: int = 32
    mlp_ratio: float = 4.0
    qk_norm: bool = False
    drop_path: float = 0.0
    dropout: float = 0.0
    use_activation_checkpoint: bool = False  # Enable activation checkpointing to save memory


class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm, attention, FFN, and drop-path.

    Architecture:
        x = x + DropPath(Attention(LayerNorm(x)))
        x = x + DropPath(FFN(LayerNorm(x)))

    Args:
        dim: Input/output dimension.
        num_heads: Number of attention heads.
        attention_type: "standard" or "channel_separated".
        group_size: For channel_separated attention.
        mlp_ratio: FFN expansion ratio.
        qk_norm: Apply RMSNorm to Q, K (standard attention only).
        drop_path: Stochastic depth rate.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        attention_type: str,
        group_size: int,
        mlp_ratio: float,
        qk_norm: bool,
        drop_path: float,
        dropout: float,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        # Pre-norm for attention
        self.norm1 = nn.LayerNorm(dim)

        # Attention mechanism (configurable)
        if attention_type == "standard":
            self.attn = StandardSelfAttention(
                dim=dim,
                num_heads=num_heads,
                qkv_bias=True,
                qk_norm=qk_norm,
                attn_drop=dropout,
                proj_drop=dropout,
            )
        elif attention_type == "channel_separated":
            self.attn = ChannelSeparatedSelfAttention(
                dim=dim,
                group_size=group_size,
                num_heads=num_heads,
            )
        else:
            raise ValueError(f"Unknown attention_type: {attention_type}")

        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # Pre-norm for FFN
        self.norm2 = nn.LayerNorm(dim)

        # FFN (MLP)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout),
        )

        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, T, D).

        Returns:
            Output tensor (B, T, D).
        """
        # Only checkpoint during training, not inference
        if self.use_checkpoint and self.training:
            from torch.utils.checkpoint import checkpoint

            # Checkpoint attention sublayer
            x = x + checkpoint(self._attn_forward, x, use_reentrant=False)
            # Checkpoint FFN sublayer
            x = x + checkpoint(self._ffn_forward, x, use_reentrant=False)
        else:
            # Attention block with residual and drop-path
            x = x + self.drop_path1(self.attn(self.norm1(x)))

            # FFN block with residual and drop-path
            x = x + self.drop_path2(self.mlp(self.norm2(x)))

        return x

    def _attn_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Isolated attention forward for checkpointing."""
        return self.drop_path1(self.attn(self.norm1(x)))

    def _ffn_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Isolated FFN forward for checkpointing."""
        return self.drop_path2(self.mlp(self.norm2(x)))


class PureTransformer(nn.Module):
    """Pure stacked transformer for latent space evolution.

    This is a simplified alternative to U-shaped PDETransformer.
    Recommended for 256-512 latent tokens (UPT guideline).

    Architecture:
        Input (B, T, input_dim)
            ↓ input_proj
        (B, T, hidden_dim)
            ↓ depth × TransformerBlock (fixed token count T)
        (B, T, hidden_dim)
            ↓ output_norm + output_proj
        Output (B, T, input_dim)

    Key differences from U-shaped:
    - Fixed token count throughout (no pooling/unpooling)
    - No skip connections (purely sequential processing)
    - Simpler architecture (easier to reason about and scale)
    - Linear drop-path schedule (0 at layer 0, max at layer depth-1)

    Args:
        cfg: Configuration dataclass.

    Example:
        >>> cfg = PureTransformerConfig(
        ...     input_dim=192, hidden_dim=192, depth=8, num_heads=6, drop_path=0.1
        ... )
        >>> model = PureTransformer(cfg)
        >>> x = torch.randn(4, 256, 192)
        >>> out = model(x)  # (4, 256, 192)
    """

    def __init__(self, cfg: PureTransformerConfig):
        super().__init__()
        self.cfg = cfg

        # Dimension validation
        if cfg.input_dim != cfg.hidden_dim:
            # Allow different dims, use projection
            self.input_proj = nn.Linear(cfg.input_dim, cfg.hidden_dim)
            self.output_proj = nn.Linear(cfg.hidden_dim, cfg.input_dim)
        else:
            self.input_proj = nn.Identity()
            self.output_proj = nn.Identity()

        # Stacked transformer blocks with linear drop-path schedule
        drop_path_rates = [
            cfg.drop_path * i / max(cfg.depth - 1, 1) for i in range(cfg.depth)
        ]

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    dim=cfg.hidden_dim,
                    num_heads=cfg.num_heads,
                    attention_type=cfg.attention_type,
                    group_size=cfg.group_size,
                    mlp_ratio=cfg.mlp_ratio,
                    qk_norm=cfg.qk_norm,
                    drop_path=drop_path_rates[i],
                    dropout=cfg.dropout,
                    use_checkpoint=cfg.use_activation_checkpoint,
                )
                for i in range(cfg.depth)
            ]
        )

        # Output normalization
        self.output_norm = nn.LayerNorm(cfg.hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Latent tokens (B, T, input_dim).

        Returns:
            Residual latent tokens (B, T, input_dim).
        """
        # Project to hidden dimension
        x = self.input_proj(x)

        # Apply transformer layers sequentially
        for layer in self.layers:
            x = layer(x)

        # Output normalization and projection
        x = self.output_norm(x)
        x = self.output_proj(x)

        return x
