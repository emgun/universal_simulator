"""Query-based sampling for sparse spatial supervision.

This module provides sampling strategies for query-based training, enabling:
1. Training speedup (fewer decoder evaluations)
2. Zero-shot super-resolution (resolution-agnostic training)
3. Better generalization to arbitrary discretizations
"""

from __future__ import annotations

import torch
from typing import Dict, Tuple, Optional


def sample_uniform_queries(
    total_points: int,
    num_queries: int,
    device: torch.device = None,
) -> torch.Tensor:
    """Uniform random sampling of query indices.

    Args:
        total_points: Total number of spatial points (H * W)
        num_queries: Number of query points to sample
        device: Torch device for output tensor

    Returns:
        Query indices (num_queries,) in range [0, total_points)
    """
    if num_queries >= total_points:
        # If requesting >= all points, return all indices (no sampling)
        return torch.arange(total_points, device=device)

    # Uniform random sampling without replacement
    indices = torch.randperm(total_points, device=device)[:num_queries]
    return indices


def sample_stratified_queries(
    grid_shape: Tuple[int, int],
    num_queries: int,
    device: torch.device = None,
) -> torch.Tensor:
    """Stratified sampling ensuring coverage of all grid regions.

    Divides grid into blocks and samples proportionally from each block.
    Ensures no region is under-represented.

    Args:
        grid_shape: (H, W) grid dimensions
        num_queries: Number of query points to sample
        device: Torch device for output tensor

    Returns:
        Query indices (num_queries,) flattened to 1D
    """
    H, W = grid_shape
    total_points = H * W

    if num_queries >= total_points:
        return torch.arange(total_points, device=device)

    # Determine block grid size (aim for sqrt(num_queries) blocks per dim)
    blocks_per_dim = max(1, int(num_queries ** 0.5))
    block_h = H // blocks_per_dim
    block_w = W // blocks_per_dim
    queries_per_block = max(1, num_queries // (blocks_per_dim ** 2))

    indices_list = []
    for i in range(blocks_per_dim):
        for j in range(blocks_per_dim):
            # Define block boundaries
            h_start = i * block_h
            h_end = H if i == blocks_per_dim - 1 else (i + 1) * block_h
            w_start = j * block_w
            w_end = W if j == blocks_per_dim - 1 else (j + 1) * block_w

            # Sample from this block
            block_size = (h_end - h_start) * (w_end - w_start)
            n_samples = min(queries_per_block, block_size)

            # Generate random indices within block
            block_indices = torch.randperm(block_size, device=device)[:n_samples]

            # Convert block-local indices to global flat indices
            h_local = block_indices // (w_end - w_start)
            w_local = block_indices % (w_end - w_start)
            h_global = h_local + h_start
            w_global = w_local + w_start
            global_indices = h_global * W + w_global

            indices_list.append(global_indices)

    # Concatenate all block indices
    all_indices = torch.cat(indices_list)

    # Handle rounding: add random extras if needed
    if len(all_indices) < num_queries:
        extra_needed = num_queries - len(all_indices)
        extra_indices = torch.randperm(total_points, device=device)[:extra_needed]
        all_indices = torch.cat([all_indices, extra_indices])

    return all_indices[:num_queries]


def apply_query_sampling(
    fields: Dict[str, torch.Tensor],
    coords: torch.Tensor,
    num_queries: int,
    strategy: str = "uniform",
    grid_shape: Optional[Tuple[int, int]] = None,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """Apply query sampling to fields and coordinates.

    Args:
        fields: Dict of field tensors {name: (B, N, C)}
        coords: Coordinate tensor (B, N, coord_dim)
        num_queries: Number of query points (if >= N, returns all points)
        strategy: "uniform" or "stratified"
        grid_shape: (H, W) required for stratified sampling

    Returns:
        Tuple of (sampled_fields, sampled_coords)
        - sampled_fields: Dict {name: (B, num_queries, C)}
        - sampled_coords: (B, num_queries, coord_dim)
    """
    B, N, coord_dim = coords.shape
    device = coords.device

    # If num_queries >= N, no sampling (return full dense grid)
    if num_queries >= N:
        return fields, coords

    # Sample query indices (same for all batch elements for consistency)
    if strategy == "uniform":
        query_indices = sample_uniform_queries(N, num_queries, device=device)
    elif strategy == "stratified":
        if grid_shape is None:
            raise ValueError("grid_shape required for stratified sampling")
        query_indices = sample_stratified_queries(grid_shape, num_queries, device=device)
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")

    # Apply sampling to all fields
    sampled_fields = {}
    for name, tensor in fields.items():
        # tensor: (B, N, C) → (B, num_queries, C)
        sampled_fields[name] = tensor[:, query_indices, :]

    # Apply sampling to coordinates
    sampled_coords = coords[:, query_indices, :]  # (B, N, 2) → (B, num_queries, 2)

    return sampled_fields, sampled_coords
