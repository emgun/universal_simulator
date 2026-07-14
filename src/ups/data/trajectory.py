from __future__ import annotations

"""Source-neutral trajectory sample contract for physics datasets.

The canonical layout is time-first and channel-last: ``T x spatial... x C``.
The contract uses plain containers rather than source-specific runtime classes,
so callers can use standard or project-specific PyTorch collators.
"""

from collections.abc import Mapping
from typing import Any, TypedDict

import torch


class TrajectoryIdentity(TypedDict):
    """Stable source identity for a trajectory window when the source exposes it."""

    source: str
    dataset_name: str
    split: str
    sample_index: int
    source_file: str | None
    trajectory_index: int | None
    window_start: int | None


class TrajectoryMetadata(TypedDict):
    """Small, serializable metadata attached to one trajectory window."""

    identity: TrajectoryIdentity
    layout: str
    channel_axis: int
    n_spatial_dims: int
    field_names: tuple[str, ...]
    attributes: dict[str, Any]


class TrajectorySample(TypedDict):
    """Canonical model-facing sample.

    ``fields`` and ``targets`` may have different time lengths, but must share
    spatial and channel shapes. Optional physical context remains named rather
    than flattened so adapters do not silently discard source semantics.
    """

    fields: torch.Tensor
    targets: torch.Tensor
    params: dict[str, torch.Tensor]
    boundary_conditions: dict[str, torch.Tensor]
    auxiliary: dict[str, torch.Tensor]
    space_grid: torch.Tensor | None
    input_time_grid: torch.Tensor | None
    target_time_grid: torch.Tensor | None
    metadata: TrajectoryMetadata


def validate_trajectory_sample(sample: Mapping[str, Any]) -> None:
    """Fail closed when an adapter violates the canonical shape contract."""

    required = {
        "fields",
        "targets",
        "params",
        "boundary_conditions",
        "auxiliary",
        "space_grid",
        "input_time_grid",
        "target_time_grid",
        "metadata",
    }
    missing = sorted(required.difference(sample))
    if missing:
        raise ValueError(f"Trajectory sample is missing keys: {', '.join(missing)}")

    fields = sample["fields"]
    targets = sample["targets"]
    if not isinstance(fields, torch.Tensor) or not isinstance(targets, torch.Tensor):
        raise TypeError("fields and targets must be torch.Tensor values")
    if fields.ndim < 3 or targets.ndim < 3:
        raise ValueError("fields and targets must use T x spatial... x C layout")
    if fields.ndim != targets.ndim or fields.shape[1:] != targets.shape[1:]:
        raise ValueError("fields and targets must share spatial and channel shapes")
    if fields.shape[0] < 1 or targets.shape[0] < 1 or fields.shape[-1] < 1:
        raise ValueError("fields and targets cannot have empty time or channel axes")

    metadata = sample["metadata"]
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping")
    expected_spatial_dims = fields.ndim - 2
    if metadata.get("layout") != "T...C" or metadata.get("channel_axis") != -1:
        raise ValueError("metadata must declare canonical T...C channel-last layout")
    if metadata.get("n_spatial_dims") != expected_spatial_dims:
        raise ValueError("metadata n_spatial_dims does not match field rank")

    field_names = metadata.get("field_names")
    if field_names and len(field_names) != fields.shape[-1]:
        raise ValueError("field_names length must match the channel count")
    for key in ("params", "boundary_conditions", "auxiliary"):
        values = sample[key]
        if not isinstance(values, Mapping):
            raise TypeError(f"{key} must be a mapping")
        if not all(
            isinstance(name, str) and isinstance(value, torch.Tensor)
            for name, value in values.items()
        ):
            raise TypeError(f"{key} must map string names to torch.Tensor values")

    space_grid = sample["space_grid"]
    if space_grid is not None:
        if not isinstance(space_grid, torch.Tensor):
            raise TypeError("space_grid must be a torch.Tensor or None")
        if space_grid.shape[-1] != expected_spatial_dims:
            raise ValueError("space_grid final axis must enumerate spatial dimensions")

    for key, expected_steps in (
        ("input_time_grid", fields.shape[0]),
        ("target_time_grid", targets.shape[0]),
    ):
        grid = sample[key]
        if grid is not None:
            if not isinstance(grid, torch.Tensor):
                raise TypeError(f"{key} must be a torch.Tensor or None")
            if grid.numel() != expected_steps:
                raise ValueError(f"{key} length must match its trajectory time axis")
