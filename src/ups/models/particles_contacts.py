from __future__ import annotations

"""Particle neighbor search and symplectic integration primitives."""

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, Tuple

import torch


@dataclass
class NeighborSearchConfig:
    cell_size: float = 0.2
    max_neighbors: int = 128


def _hash_cells(positions: torch.Tensor, cell_size: float) -> Tuple[Dict[Tuple[int, int, int], torch.Tensor], float]:
    inv = 1.0 / cell_size
    cell_coords = torch.floor(positions * inv).to(torch.long)
    buckets: Dict[Tuple[int, int, int], torch.Tensor] = {}
    for idx, cell in enumerate(cell_coords):
        key = tuple(cell.tolist())
        if key not in buckets:
            buckets[key] = torch.tensor([idx], device=positions.device, dtype=torch.long)
        else:
            buckets[key] = torch.cat([buckets[key], torch.tensor([idx], device=positions.device)])
    return buckets, inv


def hierarchical_neighbor_search(
    positions: torch.Tensor,
    radii: torch.Tensor,
    cfg: Optional[NeighborSearchConfig] = None,
) -> torch.Tensor:
    if cfg is None:
        cfg = NeighborSearchConfig()
    if positions.dim() != 2:
        raise ValueError("positions must have shape (N, dim)")
    N = positions.shape[0]
    buckets, inv_cell = _hash_cells(positions, cfg.cell_size)
    edges = []
    offsets = torch.tensor([(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)], device=positions.device)
    for key, idxs in buckets.items():
        cell_center = torch.tensor(key, device=positions.device)
        for offset in offsets:
            neighbor_key = tuple((cell_center + offset).tolist())
            if neighbor_key not in buckets:
                continue
            candidates = buckets[neighbor_key]
            points_a = positions[idxs][:, None, :]
            points_b = positions[candidates][None, :, :]
            dists = torch.linalg.norm(points_a - points_b, dim=-1)
            rad_sum = radii[idxs][:, None] + radii[candidates][None, :]
            mask = (dists <= rad_sum) & (dists > 0)
            a_idx, b_idx = torch.where(mask)
            if a_idx.numel() == 0:
                continue
            pairs = torch.stack([idxs[a_idx], candidates[b_idx]], dim=1)
            edges.append(pairs)
    if not edges:
        return torch.zeros(0, 2, dtype=torch.long, device=positions.device)
    edges_tensor = torch.cat(edges, dim=0)
    edges_tensor = torch.unique(edges_tensor.sort(dim=1).values, dim=0)
    return edges_tensor


ConstraintFn = Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]


class SymplecticIntegrator:
    def __init__(self, constraint: Optional[ConstraintFn] = None) -> None:
        self.constraint = constraint

    def step(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        forces_fn: Callable[[torch.Tensor], torch.Tensor],
        masses: torch.Tensor,
        dt: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        inv_mass = (1.0 / masses).view(-1, 1)
        accel = forces_fn(positions) * inv_mass
        velocities_half = velocities + 0.5 * dt * accel
        new_positions = positions + dt * velocities_half
        if self.constraint is not None:
            new_positions, velocities_half = self.constraint(new_positions, velocities_half)
        new_accel = forces_fn(new_positions) * inv_mass
        new_velocities = velocities_half + 0.5 * dt * new_accel
        return new_positions, new_velocities

