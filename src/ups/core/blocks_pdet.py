from __future__ import annotations

"""Core building blocks for the PDE-Transformer (PDE-T) stack.

This module provides a compact U-shaped transformer that operates on latent
tokens. The design mirrors the architecture described in the implementation
plan: channel-separated tokens, axial channel attention bridges, and RMSNorm- 
normalised queries/keys. The class is intentionally lightweight so it can be
stacked repeatedly inside the latent operator.
"""

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

try:  # Enable Flash/SDPA kernels when available (PyTorch ≥2.0)
    from torch.backends.cuda import sdp_kernel

    sdp_kernel.enable_math(True)
    sdp_kernel.enable_flash(True)
    sdp_kernel.enable_mem_efficient(True)
except Exception:  # pragma: no cover - best-effort enablement
    pass


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalisation.

    This variant normalises each token independently using the root mean square
    of its features, avoiding the mean subtraction performed by LayerNorm. It is
    numerically stable and a good match for attention architectures that want to
    preserve the mean signal.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * (x / rms)


class ChannelSeparatedSelfAttention(nn.Module):
    """Self-attention applied independently to channel groups.

    The incoming token sequence is split along the feature dimension into
    equally sized groups (channel-separated tokens). Attention is then applied
    within each group, using RMS-normalised queries and keys before projection.
    This matches the "axial channel attention" requirement while keeping the
    implementation efficient (a single multi-head attention call on a reshaped
    batch).
    """

    def __init__(self, dim: int, group_size: int, num_heads: int) -> None:
        super().__init__()
        if dim % group_size != 0:
            raise ValueError("dim must be divisible by group_size for channel separation")
        self.dim = dim
        self.group_size = group_size
        self.groups = dim // group_size
        if group_size % num_heads != 0:
            raise ValueError("group_size must be divisible by num_heads for attention heads")
        self.num_heads = num_heads
        self.head_dim = group_size // num_heads
        self.dropout_p = 0.0

        self.q_proj = nn.Linear(group_size, group_size)
        self.k_proj = nn.Linear(group_size, group_size)
        self.v_proj = nn.Linear(group_size, group_size)
        self.out_proj = nn.Linear(group_size, group_size)
        self.q_norm = RMSNorm(group_size)
        self.k_norm = RMSNorm(group_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        x_groups = x.view(B, T, self.groups, self.group_size)
        x_groups = x_groups.permute(0, 2, 1, 3).contiguous().view(B * self.groups, T, self.group_size)

        q = self.q_proj(self.q_norm(x_groups))
        k = self.k_proj(self.k_norm(x_groups))
        v = self.v_proj(x_groups)

        B_groups = q.shape[0]
        q = q.view(B_groups, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B_groups, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B_groups, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(B_groups, T, self.group_size)
        attn_out = self.out_proj(attn_out)

        attn_out = attn_out.view(B, self.groups, T, self.group_size).permute(0, 2, 1, 3).contiguous()
        attn_out = attn_out.view(B, T, C)
        return attn_out


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerLayer(nn.Module):
    def __init__(self, dim: int, group_size: int, num_heads: int, mlp_ratio: float = 2.0,
                 use_checkpoint: bool = False) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = ChannelSeparatedSelfAttention(dim, group_size=group_size, num_heads=num_heads)
        self.ff = FeedForward(dim, hidden_dim=hidden)
        self.use_checkpoint = use_checkpoint

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Only checkpoint during training, not inference
        if self.use_checkpoint and self.training:
            # Checkpoint attention sublayer
            x = x + checkpoint(self._attn_forward, x, use_reentrant=False)
            # Checkpoint FFN sublayer
            x = x + checkpoint(self._ffn_forward, x, use_reentrant=False)
        else:
            # Original eager execution
            x = x + self.attn(self.attn_norm(x))
            x = x + self.ff(x)
        return x

    def _attn_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Isolated attention forward for checkpointing."""
        return self.attn(self.attn_norm(x))

    def _ffn_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Isolated FFN forward for checkpointing."""
        return self.ff(x)


def _downsample_tokens(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] == 1:
        return x
    if x.shape[1] % 2 == 1:
        pad = x[:, -1:, :]
        x = torch.cat([x, pad], dim=1)
    x_even = x[:, ::2, :]
    x_odd = x[:, 1::2, :]
    return 0.5 * (x_even + x_odd)


def _upsample_tokens(x: torch.Tensor, target_length: int) -> torch.Tensor:
    if x.shape[1] == target_length:
        return x
    x_t = x.transpose(1, 2)
    up = F.interpolate(x_t, size=target_length, mode="linear", align_corners=False)
    return up.transpose(1, 2)


@dataclass
class PDETransformerConfig:
    input_dim: int
    hidden_dim: int
    depths: Sequence[int] = (2, 2, 2)
    group_size: int = 32
    num_heads: int = 4
    mlp_ratio: float = 2.0
    dropout: float = 0.0
    use_activation_checkpoint: bool = False


class PDETransformerBlock(nn.Module):
    """U-shaped transformer backbone for latent evolution."""

    def __init__(self, cfg: PDETransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.input_proj = nn.Linear(cfg.input_dim, cfg.hidden_dim)
        self.output_proj = nn.Linear(cfg.hidden_dim, cfg.input_dim)

        self.down_layers = nn.ModuleList()
        for depth in cfg.depths[:-1]:
            blocks = nn.ModuleList(
                [
                    TransformerLayer(cfg.hidden_dim, cfg.group_size, cfg.num_heads, cfg.mlp_ratio,
                                   use_checkpoint=cfg.use_activation_checkpoint)
                    for _ in range(depth)
                ]
            )
            proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
            self.down_layers.append(nn.ModuleList([blocks, proj]))

        self.bottleneck = nn.ModuleList(
            [
                TransformerLayer(cfg.hidden_dim, cfg.group_size, cfg.num_heads, cfg.mlp_ratio,
                               use_checkpoint=cfg.use_activation_checkpoint)
                for _ in range(cfg.depths[-1])
            ]
        )

        self.up_layers = nn.ModuleList()
        for depth in reversed(cfg.depths[:-1]):
            proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
            blocks = nn.ModuleList(
                [
                    TransformerLayer(cfg.hidden_dim, cfg.group_size, cfg.num_heads, cfg.mlp_ratio,
                                   use_checkpoint=cfg.use_activation_checkpoint)
                    for _ in range(depth)
                ]
            )
            self.up_layers.append(nn.ModuleList([proj, blocks]))

        self.final_norm = nn.LayerNorm(cfg.hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError("Expected tensor shaped (batch, tokens, features)")
        x = self.input_proj(x)
        skips: List[torch.Tensor] = []

        for layer_pack in self.down_layers:
            blocks, proj = layer_pack
            for layer in blocks:
                x = layer(x)
            skips.append(x)
            x = _downsample_tokens(x)
            x = proj(x)

        for layer in self.bottleneck:
            x = layer(x)

        for layer_pack, skip in zip(self.up_layers, reversed(skips)):
            proj, blocks = layer_pack
            x = _upsample_tokens(x, skip.shape[1])
            x = x + proj(skip)
            for layer in blocks:
                x = layer(x)

        x = self.final_norm(x)
        x = self.output_proj(x)
        return x
