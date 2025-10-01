from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class MeshParticleEncoderConfig:
    latent_len: int
    latent_dim: int
    hidden_dim: int
    message_passing_steps: int = 3
    supernodes: int = 2048
    use_coords: bool = True


def _flatten_fields(fields: Mapping[str, torch.Tensor]) -> torch.Tensor:
    flattened = []
    for value in fields.values():
        if value.dim() == 3:
            # Assume (B, N, C)
            flattened.append(value)
        elif value.dim() == 2:
            flattened.append(value.unsqueeze(0))
        elif value.dim() == 4:
            # Treat leading dimension as time/steps, average over it
            flattened.append(value.mean(dim=0, keepdim=True))
        else:
            raise ValueError(f"Unsupported field tensor shape {value.shape}")
    if not flattened:
        raise ValueError("MeshParticleEncoder requires at least one field tensor")
    return torch.cat(flattened, dim=-1)


def _build_adjacency(num_nodes: int, edges: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    if edges.numel() == 0:
        # no edges; fall back to self loops
        rows = torch.arange(num_nodes, device=device)
        return rows, rows
    src = edges[:, 0].to(torch.long)
    dst = edges[:, 1].to(torch.long)
    undirected_src = torch.cat([src, dst])
    undirected_dst = torch.cat([dst, src])
    return undirected_src, undirected_dst


class MeshParticleEncoder(nn.Module):
    """Lightweight encoder for mesh or particle graphs.

    Features are aggregated via residual message passing followed by supernode
    pooling and a Perceiver-style token reduction to the desired latent length.
    """

    def __init__(self, cfg: MeshParticleEncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.node_proj: Optional[nn.Linear] = None
        self.hidden_dim = cfg.hidden_dim
        self.latent_dim = cfg.latent_dim
        self.message_layers = nn.ModuleList(
            [nn.Linear(cfg.hidden_dim, cfg.hidden_dim) for _ in range(cfg.message_passing_steps)]
        )
        if cfg.hidden_dim == cfg.latent_dim:
            self.latent_proj = nn.Identity()
            self.latent_to_hidden = nn.Identity()
        else:
            self.latent_proj = nn.Linear(cfg.hidden_dim, cfg.latent_dim)
            self.latent_to_hidden = nn.Linear(cfg.latent_dim, cfg.hidden_dim)
        self.output_proj: Optional[nn.Linear] = None

    def _ensure_projections(self, input_dim: int) -> None:
        if self.node_proj is None:
            if input_dim == self.hidden_dim:
                self.node_proj = nn.Identity()
            else:
                self.node_proj = nn.Linear(input_dim, self.hidden_dim)
        if self.output_proj is None:
            if input_dim == self.hidden_dim:
                self.output_proj = nn.Identity()
            else:
                self.output_proj = nn.Linear(self.hidden_dim, input_dim)

    def forward(
        self,
        fields: Dict[str, torch.Tensor],
        coords: torch.Tensor,
        *,
        connect: Optional[torch.Tensor] = None,
        meta: Optional[Mapping[str, object]] = None,
    ) -> torch.Tensor:
        device = coords.device
        feat = _flatten_fields(fields).to(device)
        B, N, feat_dim = feat.shape

        if self.cfg.use_coords:
            coord_feat = coords.to(device)
            if coord_feat.dim() == 2:
                coord_feat = coord_feat.unsqueeze(0).expand(B, -1, -1)
            feat = torch.cat([feat, coord_feat], dim=-1)

        self._ensure_projections(feat.size(-1))
        assert self.node_proj is not None
        h = self.node_proj(feat)

        if connect is None:
            connect = torch.arange(N, device=device).unsqueeze(1).repeat(1, 2)
        edges = connect.to(device)
        src_idx, dst_idx = _build_adjacency(N, edges, device)

        for layer in self.message_layers:
            if isinstance(layer, nn.Linear):
                m = torch.zeros_like(h)
                m.index_add_(1, dst_idx, h[:, src_idx, :])
                deg = torch.zeros(N, device=device)
                deg.index_add_(0, dst_idx, torch.ones_like(dst_idx, dtype=deg.dtype))
                deg = deg.clamp_min_(1.0).view(1, N, 1)
                m = m / deg
                m = layer(m)
                h = h + F.gelu(m)

        tokens = h
        S = min(self.cfg.supernodes, N)
        if S < N:
            tokens = self._pool_supernodes(tokens, S)
        pooled_len = tokens.shape[1]
        if pooled_len != self.cfg.latent_len:
            tokens = self._perceiver_pool(tokens, self.cfg.latent_len)

        latent = self.latent_proj(tokens)
        self._cache = {
            "last_tokens": tokens.detach(),
            "node_features": h.detach(),
            "feat_dim": feat_dim,
            "input_dim": feat.size(-1),
            "node_count": N,
        }
        return latent

    def reconstruct(self, latent: torch.Tensor) -> torch.Tensor:
        if not hasattr(self, "_cache"):
            raise RuntimeError("Call forward before reconstructing")
        cache = self._cache
        tokens = self.latent_to_hidden(latent)
        hidden = tokens
        if hidden.shape[1] != cache["node_count"]:
            raise ValueError("Reconstruction only supported when latent tokens match node count")
        assert self.output_proj is not None
        node_feat = self.output_proj(hidden)
        return node_feat

    def _pool_supernodes(self, tokens: torch.Tensor, supernodes: int) -> torch.Tensor:
        B, N, D = tokens.shape
        chunk = (N + supernodes - 1) // supernodes
        pad = chunk * supernodes - N
        if pad > 0:
            pad_tensor = tokens[:, :pad, :]
            tokens = torch.cat([tokens, pad_tensor], dim=1)
        tokens = tokens.view(B, supernodes, chunk, D).mean(dim=2)
        return tokens

    def _perceiver_pool(self, tokens: torch.Tensor, target_len: int) -> torch.Tensor:
        if tokens.shape[1] == target_len:
            return tokens
        tokens_t = tokens.transpose(1, 2)
        pooled = F.adaptive_avg_pool1d(tokens_t, target_len)
        return pooled.transpose(1, 2)
