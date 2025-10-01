from __future__ import annotations

from typing import Any, Dict, Literal, Optional, TypedDict

import torch


Kind = Literal["grid", "mesh", "particles"]


class Sample(TypedDict):
    """
    Unified physics sample schema.

    Keys
    - kind: one of {grid, mesh, particles}
    - coords: Tensor [N, d] float32/float64
    - connect: Optional Tensor [E, 2] int64 (edges for mesh/graph)
    - fields: Dict[str, Tensor [N, C_f]]
    - bc: Dict[str, Any]
    - params: Dict[str, float]
    - geom: Optional[Dict[str, Any]]
    - time: Tensor scalar
    - dt: Tensor scalar
    - meta: Dict[str, Any]
    """

    kind: Kind
    coords: torch.Tensor
    connect: Optional[torch.Tensor]
    fields: Dict[str, torch.Tensor]
    bc: Dict[str, Any]
    params: Dict[str, float]
    geom: Optional[Dict[str, Any]]
    time: torch.Tensor
    dt: torch.Tensor
    meta: Dict[str, Any]


REQUIRED_KEYS = {
    "kind",
    "coords",
    "connect",
    "fields",
    "bc",
    "params",
    "geom",
    "time",
    "dt",
    "meta",
}


def _is_float_tensor(x: torch.Tensor) -> bool:
    return torch.is_tensor(x) and x.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64)


def _is_int_tensor(x: torch.Tensor) -> bool:
    return torch.is_tensor(x) and x.dtype in (torch.int32, torch.int64)


def validate_sample(sample: Dict[str, Any]) -> None:
    """
    Validates that a dict conforms to the Sample schema.

    Raises ValueError with informative messages on failure.
    """
    missing = REQUIRED_KEYS - set(sample.keys())
    if missing:
        raise ValueError(f"Sample missing required keys: {sorted(missing)}")

    kind = sample["kind"]
    if kind not in ("grid", "mesh", "particles"):
        raise ValueError(f"Invalid kind '{kind}', expected 'grid'|'mesh'|'particles'")

    coords = sample["coords"]
    if not _is_float_tensor(coords) or coords.ndim != 2:
        raise ValueError("coords must be float tensor of shape [N, d]")
    n = coords.shape[0]

    connect = sample["connect"]
    if connect is not None:
        if not _is_int_tensor(connect) or connect.ndim != 2 or connect.shape[1] != 2:
            raise ValueError("connect must be int tensor [E, 2] or None")

    fields = sample["fields"]
    if not isinstance(fields, dict) or not fields:
        raise ValueError("fields must be a non-empty dict[str, Tensor]")
    for k, v in fields.items():
        if not _is_float_tensor(v) or v.ndim != 2 or v.shape[0] != n:
            raise ValueError(f"field '{k}' must be float tensor [N, C_f] with N={n}")

    time = sample["time"]
    dt = sample["dt"]
    if not _is_float_tensor(time) or time.numel() != 1:
        raise ValueError("time must be scalar float tensor")
    if not _is_float_tensor(dt) or dt.numel() != 1:
        raise ValueError("dt must be scalar float tensor")

    params = sample["params"]
    if not isinstance(params, dict):
        raise ValueError("params must be a dict[str, float]")
    for pk, pv in params.items():
        if not isinstance(pv, (float, int)):
            raise ValueError(f"param '{pk}' must be float-like, got {type(pv)}")

    bc = sample["bc"]
    if not isinstance(bc, dict):
        raise ValueError("bc must be a dict[str, Any]")

    geom = sample["geom"]
    if geom is not None and not isinstance(geom, dict):
        raise ValueError("geom must be a dict[str, Any] or None")

    meta = sample["meta"]
    if not isinstance(meta, dict):
        raise ValueError("meta must be a dict[str, Any]")

