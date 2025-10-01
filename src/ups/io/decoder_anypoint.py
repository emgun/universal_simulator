from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F


def _fourier_encode(coords: torch.Tensor, frequencies: Sequence[float]) -> torch.Tensor:
    """Project coordinates onto sin/cos bases for richer positional context.

    Parameters
    ----------
    coords:
        Coordinate tensor shaped ``(batch, num_points, coord_dim)``.
    frequencies:
        Collection of scalar frequencies for the sinusoidal projections. Each
        frequency produces both a sine and cosine channel.
    """

    if not frequencies:
        return coords
    freqs = torch.tensor(frequencies, dtype=coords.dtype, device=coords.device)
    freqs = freqs.view(1, 1, -1, 1)
    coords_expanded = coords.unsqueeze(-2)  # (B, Q, 1, D)
    scaled = 2.0 * torch.pi * coords_expanded * freqs
    sin_feat = torch.sin(scaled)
    cos_feat = torch.cos(scaled)
    encoded = torch.cat([coords, sin_feat.flatten(-2), cos_feat.flatten(-2)], dim=-1)
    return encoded


@dataclass
class AnyPointDecoderConfig:
    latent_dim: int
    query_dim: int
    hidden_dim: int = 128
    num_layers: int = 2
    num_heads: int = 4
    frequencies: Tuple[float, ...] = (1.0, 2.0, 4.0)
    mlp_hidden_dim: int = 128
    output_channels: Mapping[str, int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.output_channels is None or len(self.output_channels) == 0:
            raise ValueError("output_channels must specify at least one field name -> channel count")
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads for multi-head attention")


class AnyPointDecoder(nn.Module):
    """Perceiver-style cross-attention decoder for continuous query points.

    Given a set of latent tokens (the encoder's output) and arbitrary spatial
    query coordinates, the decoder emits predicted field values. The architecture
    mirrors PerceiverIO: queries are embedded, refined via cross-attention
    against the latent tokens, and then passed through lightweight prediction
    heads for each requested output field.
    """

    def __init__(self, cfg: AnyPointDecoderConfig):
        super().__init__()
        self.cfg = cfg
        self.query_embed = nn.Linear(cfg.query_dim + 2 * len(cfg.frequencies) * cfg.query_dim, cfg.hidden_dim)
        self.latent_proj = nn.Linear(cfg.latent_dim, cfg.hidden_dim)

        self.layers = nn.ModuleList()
        for _ in range(cfg.num_layers):
            attn = nn.MultiheadAttention(cfg.hidden_dim, cfg.num_heads, batch_first=True)
            ln_query = nn.LayerNorm(cfg.hidden_dim)
            ln_ff = nn.LayerNorm(cfg.hidden_dim)
            ff = nn.Sequential(
                nn.Linear(cfg.hidden_dim, cfg.mlp_hidden_dim),
                nn.GELU(),
                nn.Linear(cfg.mlp_hidden_dim, cfg.hidden_dim),
            )
            self.layers.append(nn.ModuleList([attn, ln_query, ff, ln_ff]))

        self.heads = nn.ModuleDict()
        for name, out_ch in cfg.output_channels.items():
            self.heads[name] = nn.Sequential(
                nn.Linear(cfg.hidden_dim, cfg.mlp_hidden_dim),
                nn.GELU(),
                nn.Linear(cfg.mlp_hidden_dim, out_ch),
            )

    def forward(
        self,
        points: torch.Tensor,
        latent_tokens: torch.Tensor,
        *,
        conditioning: Optional[Mapping[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Decode latent tokens at arbitrary coordinates.

        Parameters
        ----------
        points:
            Query coordinates with shape ``(batch, num_points, query_dim)``.
        latent_tokens:
            Latent sequence produced by the encoder, shaped ``(batch, num_tokens, latent_dim)``.
        conditioning:
            Optional dictionary of conditioning signals (e.g., boundary condition
            embeddings). If provided, each tensor should be broadcastable to the
            latent shape; the decoder simply concatenates them onto the latent
            token dimension.
        """

        if latent_tokens.dim() != 3 or points.dim() != 3:
            raise ValueError("latents and points must be rank-3 tensors (B, seq, feature)")

        B, Q, _ = points.shape
        latents = latent_tokens
        if conditioning:
            cond_feats = [tensor.expand_as(latent_tokens) for tensor in conditioning.values()]
            latents = torch.cat([latent_tokens, *cond_feats], dim=-1)

        latents = self.latent_proj(latents)

        enriched_points = _fourier_encode(points, self.cfg.frequencies)
        queries = self.query_embed(enriched_points)

        for attn, ln_q, ff, ln_ff in self.layers:
            attn_out, _ = attn(queries, latents, latents)
            queries = ln_q(queries + attn_out)
            ff_out = ff(queries)
            queries = ln_ff(queries + ff_out)

        outputs: Dict[str, torch.Tensor] = {}
        for name, head in self.heads.items():
            outputs[name] = head(queries)
        return outputs

    decode = forward
