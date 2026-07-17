from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class CanonicalPointEncoderConfig:
    """Configuration for one discretization-neutral physical-state encoder.

    Every discretization is presented as samples ``(coordinate, field value)``.
    Field names and channel counts are part of the contract: silently changing
    their order or meaning would make paired latent comparisons invalid.
    """

    latent_len: int
    latent_dim: int
    hidden_dim: int
    coord_dim: int
    field_channels: Mapping[str, int]
    supernodes: int = 256
    supernode_neighbors: int = 32
    transformer_layers: int = 2
    num_heads: int = 4
    fourier_frequencies: tuple[float, ...] = (1.0, 2.0, 4.0)

    def __post_init__(self) -> None:
        if not self.field_channels:
            raise ValueError("field_channels must be non-empty")
        if any(channels <= 0 for channels in self.field_channels.values()):
            raise ValueError("every field channel count must be positive")
        for name in ("latent_len", "latent_dim", "hidden_dim", "coord_dim"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.supernodes <= 0 or self.supernode_neighbors <= 0:
            raise ValueError("supernode counts must be positive")
        if self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")


class CanonicalPointEncoder(nn.Module):
    """Encode grids, meshes, or particles through one point-set path.

    The module deliberately has no modality switch. A regular grid is simply a
    point set with regular coordinates; an irregular mesh or particle cloud is
    the same interface with different coordinates. Geometry-aware supernode
    aggregation is followed by transformer processing and learned Perceiver
    queries. The learned queries give latent token slots one shared meaning,
    making direct paired-state alignment a well-defined training diagnostic.

    This is an architecture scaffold, not evidence that a trained canonical
    latent basis exists. Reconstruction and paired-discretization qualification
    are required before using its output with a shared operator.
    """

    def __init__(self, cfg: CanonicalPointEncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.field_order = tuple(cfg.field_channels)
        field_dim = sum(cfg.field_channels.values())
        coord_features = cfg.coord_dim * (1 + 2 * len(cfg.fourier_frequencies))

        self.input_projection = nn.Sequential(
            nn.Linear(field_dim + coord_features, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.num_heads,
            dim_feedforward=4 * cfg.hidden_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.supernode_transformer = nn.TransformerEncoder(
            layer,
            num_layers=cfg.transformer_layers,
            enable_nested_tensor=False,
        )
        self.latent_queries = nn.Parameter(torch.empty(cfg.latent_len, cfg.hidden_dim))
        nn.init.normal_(self.latent_queries, std=cfg.hidden_dim**-0.5)
        self.perceiver_pool = nn.MultiheadAttention(
            cfg.hidden_dim, cfg.num_heads, dropout=0.0, batch_first=True
        )
        self.output_norm = nn.LayerNorm(cfg.hidden_dim)
        self.output_projection = (
            nn.Identity()
            if cfg.hidden_dim == cfg.latent_dim
            else nn.Linear(cfg.hidden_dim, cfg.latent_dim)
        )

    def forward(
        self,
        fields: Mapping[str, torch.Tensor],
        coords: torch.Tensor,
        *,
        connect: torch.Tensor | None = None,
        params: Mapping[str, torch.Tensor] | None = None,
        bc: Mapping[str, torch.Tensor] | None = None,
        geom: Mapping[str, torch.Tensor] | None = None,
        meta: Mapping[str, object] | None = None,
    ) -> torch.Tensor:
        del connect, params, bc, geom, meta
        coords = self._batched_coords(coords)
        values = self._ordered_fields(fields, batch=coords.shape[0], nodes=coords.shape[1])
        if values.device != coords.device:
            values = values.to(coords.device)
        node_tokens = self.input_projection(
            torch.cat([values, self._coordinate_features(coords)], dim=-1)
        )

        supernode_indices = self._farthest_point_indices(coords)
        supernode_coords = self._gather_nodes(coords, supernode_indices)
        supernode_tokens = self._aggregate_supernodes(node_tokens, coords, supernode_coords)
        supernode_tokens = self.supernode_transformer(supernode_tokens)

        queries = self.latent_queries.unsqueeze(0).expand(coords.shape[0], -1, -1)
        pooled, _ = self.perceiver_pool(
            queries, supernode_tokens, supernode_tokens, need_weights=False
        )
        return self.output_projection(self.output_norm(queries + pooled))

    def _ordered_fields(
        self, fields: Mapping[str, torch.Tensor], *, batch: int, nodes: int
    ) -> torch.Tensor:
        expected = set(self.field_order)
        received = set(fields)
        if received != expected:
            missing = sorted(expected - received)
            extra = sorted(received - expected)
            raise ValueError(f"field schema mismatch: missing={missing}, extra={extra}")

        ordered = []
        for name in self.field_order:
            value = fields[name]
            if value.dim() == 2:
                value = value.unsqueeze(0)
            expected_shape = (batch, nodes, self.cfg.field_channels[name])
            if tuple(value.shape) != expected_shape:
                raise ValueError(
                    f"field '{name}' expected shape {expected_shape}, got {tuple(value.shape)}"
                )
            ordered.append(value)
        return torch.cat(ordered, dim=-1)

    def _batched_coords(self, coords: torch.Tensor) -> torch.Tensor:
        if coords.dim() == 2:
            coords = coords.unsqueeze(0)
        if coords.dim() != 3 or coords.shape[-1] != self.cfg.coord_dim:
            raise ValueError(
                "coords must have shape (batch, nodes, coord_dim); " f"got {tuple(coords.shape)}"
            )
        if coords.shape[1] == 0:
            raise ValueError("coords must contain at least one node")
        return coords

    def _coordinate_features(self, coords: torch.Tensor) -> torch.Tensor:
        features = [coords]
        for frequency in self.cfg.fourier_frequencies:
            scaled = 2.0 * torch.pi * float(frequency) * coords
            features.extend((torch.sin(scaled), torch.cos(scaled)))
        return torch.cat(features, dim=-1)

    def _farthest_point_indices(self, coords: torch.Tensor) -> torch.Tensor:
        """Deterministic, geometry-only farthest-point supernode selection."""

        batch, nodes, _ = coords.shape
        count = min(self.cfg.supernodes, nodes)
        if count == nodes:
            return torch.arange(nodes, device=coords.device).expand(batch, -1)

        centroid = coords.mean(dim=1, keepdim=True)
        distances = (coords - centroid).square().sum(dim=-1)
        first = distances.argmax(dim=1)
        selected = [first]
        minimum = torch.full((batch, nodes), torch.inf, device=coords.device, dtype=coords.dtype)
        batch_index = torch.arange(batch, device=coords.device)
        for _ in range(1, count):
            last = coords[batch_index, selected[-1]].unsqueeze(1)
            minimum = torch.minimum(minimum, (coords - last).square().sum(dim=-1))
            selected.append(minimum.argmax(dim=1))
        return torch.stack(selected, dim=1)

    @staticmethod
    def _gather_nodes(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        gather = indices.unsqueeze(-1).expand(-1, -1, values.shape[-1])
        return torch.gather(values, dim=1, index=gather)

    def _aggregate_supernodes(
        self,
        node_tokens: torch.Tensor,
        coords: torch.Tensor,
        supernode_coords: torch.Tensor,
    ) -> torch.Tensor:
        distances = torch.cdist(supernode_coords, coords)
        neighbors = min(self.cfg.supernode_neighbors, coords.shape[1])
        neighbor_distances, neighbor_indices = distances.topk(
            neighbors, dim=-1, largest=False, sorted=False
        )
        batch, supernodes, _ = neighbor_indices.shape
        expanded_tokens = node_tokens.unsqueeze(1).expand(-1, supernodes, -1, -1)
        gather = neighbor_indices.unsqueeze(-1).expand(-1, -1, -1, node_tokens.shape[-1])
        local_tokens = torch.gather(expanded_tokens, dim=2, index=gather)
        scale = (
            neighbor_distances.detach()
            .median(dim=-1, keepdim=True)
            .values.clamp_min(torch.finfo(coords.dtype).eps)
        )
        weights = torch.softmax(-neighbor_distances / scale, dim=-1)
        return (local_tokens * weights.unsqueeze(-1)).sum(dim=2).reshape(batch, supernodes, -1)
