from __future__ import annotations

from typing import Any, Dict

import torch


def pi_constants_from_units(units: Dict[str, Any]) -> Dict[str, float]:
    """
    Placeholder for π-group constant derivation.
    For PoC: accept precomputed scales in `units` or default to 1.0.
    """
    scales: Dict[str, float] = {}
    for k, v in units.items():
        try:
            scales[k] = float(v)
        except Exception:
            scales[k] = 1.0
    return scales


def _apply_scale_tensor(x: torch.Tensor, s: float, inverse: bool = False) -> torch.Tensor:
    return x / s if not inverse else x * s


def _apply_scale_fields(fields: Dict[str, torch.Tensor], scales: Dict[str, float], inverse: bool = False) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for name, t in fields.items():
        s = float(scales.get(name, 1.0))
        out[name] = _apply_scale_tensor(t, s, inverse)
    return out


def to_pi_units(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert fields/params to nondimensional π-units using per-key scales.
    Stores scales in `sample['meta']['scale']`.
    """
    s = dict(sample)  # shallow copy
    meta = dict(s.get("meta", {}))
    scale: Dict[str, float] = dict(meta.get("scale", {}))

    # Prefer user-provided units->scales; default to 1.0
    units = meta.get("units", {})
    scale_defaults = pi_constants_from_units(units) if isinstance(units, dict) else {}

    # Build combined scales for fields and params
    fields = s["fields"]
    for k in fields.keys():
        scale.setdefault(k, scale_defaults.get(k, 1.0))
    params = s.get("params", {})
    for k in params.keys():
        scale.setdefault(k, scale_defaults.get(k, 1.0))

    # Apply scaling
    s["fields"] = _apply_scale_fields(fields, scale, inverse=False)
    s["params"] = {k: float(params[k]) / float(scale.get(k, 1.0)) for k in params}

    # Scalars time and dt can optionally be scaled (leave as-is by default)
    meta["scale"] = scale
    s["meta"] = meta
    return s


def from_pi_units(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Inverse transform from π-units using `sample['meta']['scale']`.
    """
    s = dict(sample)
    meta = dict(s.get("meta", {}))
    scale: Dict[str, float] = dict(meta.get("scale", {}))

    fields = s["fields"]
    s["fields"] = _apply_scale_fields(fields, scale, inverse=True)
    params = s.get("params", {})
    s["params"] = {k: float(params[k]) * float(scale.get(k, 1.0)) for k in params}
    s["meta"] = meta
    return s

