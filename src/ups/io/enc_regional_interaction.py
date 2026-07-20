from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class RegionalInteractionEncoderConfig:
    """Configuration for a compact, discretization-neutral regional encoder."""

    latent_len: int
    latent_dim: int
    hidden_dim: int
    coord_dim: int
    field_channels: Mapping[str, int]
    physical_neighbors: int = 32
    processor_neighbors: tuple[int, ...] = (2, 4, 7)
    fourier_frequencies: tuple[float, ...] = (1.0, 2.0, 4.0)
    require_measure: bool = True

    def __post_init__(self) -> None:
        if not self.field_channels:
            raise ValueError("field_channels must be non-empty")
        if any(channels <= 0 for channels in self.field_channels.values()):
            raise ValueError("every field channel count must be positive")
        for name in (
            "latent_len",
            "latent_dim",
            "hidden_dim",
            "coord_dim",
            "physical_neighbors",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not self.processor_neighbors or any(value <= 0 for value in self.processor_neighbors):
            raise ValueError("processor_neighbors must contain positive neighborhood sizes")


class _RegionalInteractionBlock(nn.Module):
    def __init__(self, hidden_dim: int, coord_dim: int, neighbors: int):
        super().__init__()
        self.neighbors = neighbors
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + coord_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, nodes: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        count = coords.shape[1]
        if count == 1:
            return nodes
        neighbors = min(self.neighbors, count - 1)
        distances = torch.cdist(coords, coords)
        diagonal = torch.eye(count, device=coords.device, dtype=torch.bool).unsqueeze(0)
        distances = distances.masked_fill(diagonal, torch.inf)
        neighbor_distances, neighbor_indices = distances.topk(
            neighbors, dim=-1, largest=False, sorted=True
        )

        expanded_nodes = nodes.unsqueeze(1).expand(-1, count, -1, -1)
        sender_nodes = torch.gather(
            expanded_nodes,
            dim=2,
            index=neighbor_indices.unsqueeze(-1).expand(-1, -1, -1, nodes.shape[-1]),
        )
        receiver_nodes = nodes.unsqueeze(2).expand(-1, -1, neighbors, -1)
        expanded_coords = coords.unsqueeze(1).expand(-1, count, -1, -1)
        sender_coords = torch.gather(
            expanded_coords,
            dim=2,
            index=neighbor_indices.unsqueeze(-1).expand(-1, -1, -1, coords.shape[-1]),
        )
        relative = sender_coords - coords.unsqueeze(2)
        scale = (
            neighbor_distances.detach()
            .amax(dim=-1, keepdim=True)
            .clamp_min(torch.finfo(coords.dtype).eps)
        )
        normalized_distance = neighbor_distances / scale
        messages = self.edge_mlp(
            torch.cat(
                [
                    sender_nodes,
                    receiver_nodes,
                    relative,
                    normalized_distance.unsqueeze(-1),
                ],
                dim=-1,
            )
        )
        aggregate = messages.mean(dim=2)
        update = self.node_mlp(torch.cat([nodes, aggregate], dim=-1))
        return self.norm(nodes + update)


class RegionalInteractionEncoder(nn.Module):
    """Encode a sampled physical field through one regional interaction graph.

    The architecture is a compact encoder-side mechanism test inspired by
    RIGNO. Physical samples send learned relative-coordinate messages to a
    deterministic regional mesh; residual multi-scale graph blocks then
    process those regional states. The regional nodes themselves are the
    latent sequence, so there are no learned latent queries, attention pool,
    modality adapters, or routing inputs.
    """

    def __init__(self, cfg: RegionalInteractionEncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.field_order = tuple(cfg.field_channels)
        self.register_buffer(
            "slot_anchors",
            self._canonical_slot_anchors(cfg.latent_len, cfg.coord_dim),
            persistent=False,
        )
        field_dim = sum(cfg.field_channels.values())
        coord_features = cfg.coord_dim * (1 + 2 * len(cfg.fourier_frequencies))
        self.input_projection = nn.Sequential(
            nn.Linear(field_dim + coord_features, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        self.region_projection = nn.Sequential(
            nn.Linear(coord_features, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        self.physical_edge_mlp = nn.Sequential(
            nn.Linear(cfg.hidden_dim + cfg.coord_dim + 1, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        self.physical_to_region = nn.Sequential(
            nn.Linear(2 * cfg.hidden_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        self.physical_to_region_norm = nn.LayerNorm(cfg.hidden_dim)
        self.processor = nn.ModuleList(
            _RegionalInteractionBlock(cfg.hidden_dim, cfg.coord_dim, neighbors)
            for neighbors in cfg.processor_neighbors
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
        del connect, params, bc, meta
        coords = self._batched_coords(coords)
        values = self._ordered_fields(fields, batch=coords.shape[0], nodes=coords.shape[1])
        measure = self._sample_measure(
            geom, batch=coords.shape[0], nodes=coords.shape[1], reference=coords
        )
        values = values.to(coords.device)

        order = self._canonical_point_order(coords)
        coords = self._gather_nodes(coords, order)
        values = self._gather_nodes(values, order)
        measure = self._gather_nodes(measure, order)
        physical_nodes = self.input_projection(
            torch.cat([values, self._coordinate_features(coords)], dim=-1)
        )

        regional_indices = self._farthest_point_indices(coords, measure)
        regional_coords = self._gather_nodes(coords, regional_indices)
        slot_order = self._assign_region_slots(regional_coords, coords)
        regional_coords = self._gather_nodes(regional_coords, slot_order)
        regional_nodes = self._physical_to_regional(
            physical_nodes, coords, measure, regional_coords
        )
        for block in self.processor:
            regional_nodes = block(regional_nodes, regional_coords)
        return self.output_projection(self.output_norm(regional_nodes))

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
        if coords.shape[1] < self.cfg.latent_len:
            raise ValueError(f"coords must contain at least latent_len={self.cfg.latent_len} nodes")
        return coords

    def _coordinate_features(self, coords: torch.Tensor) -> torch.Tensor:
        features = [coords]
        for frequency in self.cfg.fourier_frequencies:
            scaled = 2.0 * torch.pi * float(frequency) * coords
            features.extend((torch.sin(scaled), torch.cos(scaled)))
        return torch.cat(features, dim=-1)

    def _sample_measure(
        self,
        geom: Mapping[str, torch.Tensor] | None,
        *,
        batch: int,
        nodes: int,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        measure = None if geom is None else geom.get("measure")
        if measure is None:
            if self.cfg.require_measure:
                raise ValueError(
                    "RegionalInteractionEncoder requires geom['measure']; sampling measure "
                    "is part of the discretization-invariant field contract"
                )
            measure = torch.ones(batch, nodes, 1, device=reference.device, dtype=reference.dtype)
        else:
            if measure.dim() == 1:
                measure = measure.view(1, nodes, 1)
            elif measure.dim() == 2:
                measure = measure.unsqueeze(-1)
            expected = (batch, nodes, 1)
            if tuple(measure.shape) != expected:
                raise ValueError(
                    f"geom['measure'] expected shape {expected}, got {tuple(measure.shape)}"
                )
            measure = measure.to(device=reference.device, dtype=reference.dtype)
        if not torch.isfinite(measure).all() or torch.any(measure <= 0):
            raise ValueError("geom['measure'] must be finite and strictly positive")
        return measure / measure.sum(dim=1, keepdim=True)

    @staticmethod
    def _canonical_point_order(coords: torch.Tensor) -> torch.Tensor:
        order = torch.arange(coords.shape[1], device=coords.device).expand(coords.shape[0], -1)
        for dimension in reversed(range(coords.shape[-1])):
            values = torch.gather(coords[..., dimension], dim=1, index=order)
            permutation = torch.argsort(values, dim=1, stable=True)
            order = torch.gather(order, dim=1, index=permutation)
        return order

    @staticmethod
    def _canonical_slot_anchors(count: int, coord_dim: int) -> torch.Tensor:
        resolution = max(2, math.ceil(count ** (1.0 / coord_dim)) + 1)
        axis = (torch.arange(resolution, dtype=torch.float32) + 0.5) / resolution
        candidates = torch.stack(
            torch.meshgrid(*(axis for _ in range(coord_dim)), indexing="ij"), dim=-1
        ).reshape(-1, coord_dim)
        centroid = torch.full((1, coord_dim), 0.5)
        selected = [(candidates - centroid).square().sum(dim=-1).argmax()]
        minimum = torch.full((candidates.shape[0],), torch.inf)
        for _ in range(1, count):
            last = candidates[selected[-1]].unsqueeze(0)
            minimum = torch.minimum(minimum, (candidates - last).square().sum(dim=-1))
            selected.append(minimum.argmax())
        return candidates[torch.stack(selected)]

    def _assign_region_slots(
        self, regional_coords: torch.Tensor, physical_coords: torch.Tensor
    ) -> torch.Tensor:
        lower = physical_coords.amin(dim=1, keepdim=True)
        extent = (physical_coords.amax(dim=1, keepdim=True) - lower).clamp_min(
            torch.finfo(physical_coords.dtype).eps
        )
        normalized = (regional_coords - lower) / extent
        anchors = self.slot_anchors.to(device=regional_coords.device, dtype=regional_coords.dtype)
        costs = torch.cdist(anchors.unsqueeze(0).expand(normalized.shape[0], -1, -1), normalized)
        used = torch.zeros(
            normalized.shape[0], normalized.shape[1], device=normalized.device, dtype=torch.bool
        )
        assignments = []
        for slot in range(self.cfg.latent_len):
            masked_cost = costs[:, slot].masked_fill(used, torch.inf)
            selected = masked_cost.argmin(dim=1)
            assignments.append(selected)
            used.scatter_(1, selected.unsqueeze(1), True)
        return torch.stack(assignments, dim=1)

    def _farthest_point_indices(self, coords: torch.Tensor, measure: torch.Tensor) -> torch.Tensor:
        batch, nodes, _ = coords.shape
        centroid = (coords * measure).sum(dim=1, keepdim=True)
        distances = (coords - centroid).square().sum(dim=-1)
        selected = [distances.argmax(dim=1)]
        minimum = torch.full_like(distances, torch.inf)
        batch_index = torch.arange(batch, device=coords.device)
        for _ in range(1, self.cfg.latent_len):
            last = coords[batch_index, selected[-1]].unsqueeze(1)
            minimum = torch.minimum(minimum, (coords - last).square().sum(dim=-1))
            selected.append(minimum.argmax(dim=1))
        return torch.stack(selected, dim=1)

    @staticmethod
    def _gather_nodes(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        return torch.gather(
            values,
            dim=1,
            index=indices.unsqueeze(-1).expand(-1, -1, values.shape[-1]),
        )

    def _physical_to_regional(
        self,
        physical_nodes: torch.Tensor,
        physical_coords: torch.Tensor,
        measure: torch.Tensor,
        regional_coords: torch.Tensor,
    ) -> torch.Tensor:
        distances = torch.cdist(regional_coords, physical_coords)
        neighbors = min(self.cfg.physical_neighbors, physical_coords.shape[1])
        neighbor_distances, neighbor_indices = distances.topk(
            neighbors, dim=-1, largest=False, sorted=True
        )
        regions = regional_coords.shape[1]
        expanded_nodes = physical_nodes.unsqueeze(1).expand(-1, regions, -1, -1)
        local_nodes = torch.gather(
            expanded_nodes,
            dim=2,
            index=neighbor_indices.unsqueeze(-1).expand(-1, -1, -1, physical_nodes.shape[-1]),
        )
        expanded_coords = physical_coords.unsqueeze(1).expand(-1, regions, -1, -1)
        local_coords = torch.gather(
            expanded_coords,
            dim=2,
            index=neighbor_indices.unsqueeze(-1).expand(-1, -1, -1, physical_coords.shape[-1]),
        )
        relative = local_coords - regional_coords.unsqueeze(2)
        scale = (
            neighbor_distances.detach()
            .median(dim=-1, keepdim=True)
            .values.clamp_min(torch.finfo(physical_coords.dtype).eps)
        )
        messages = self.physical_edge_mlp(
            torch.cat([local_nodes, relative, (neighbor_distances / scale).unsqueeze(-1)], dim=-1)
        )
        expanded_measure = measure.unsqueeze(1).expand(-1, regions, -1, -1)
        local_measure = torch.gather(
            expanded_measure, dim=2, index=neighbor_indices.unsqueeze(-1)
        ).squeeze(-1)
        kernel = torch.exp(-neighbor_distances / scale)
        weights = kernel * local_measure
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(physical_coords.dtype).eps
        )
        aggregate = (messages * weights.unsqueeze(-1)).sum(dim=2)
        regional_base = self.region_projection(self._coordinate_features(regional_coords))
        update = self.physical_to_region(torch.cat([regional_base, aggregate], dim=-1))
        return self.physical_to_region_norm(regional_base + update)
