from __future__ import annotations

"""Latent state container used throughout the simulator."""

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

import torch


@dataclass
class LatentState:
    """Bundle latent tokens with metadata and helper utilities.

    Attributes
    ----------
    z:
        Latent tensor shaped ``(batch, tokens, dim)``.
    t:
        Optional scalar time value (single-element tensor or float).
    cond:
        Dictionary of conditioning metadata (already moved to the same device
        as ``z``).
    """

    z: torch.Tensor
    t: Optional[torch.Tensor] = None
    cond: Dict[str, torch.Tensor] = field(default_factory=dict)

    def to(self, device: torch.device | str) -> LatentState:
        device_t = torch.device(device)
        z = self.z.to(device_t)
        t = self.t.to(device_t) if self.t is not None and torch.is_tensor(self.t) else self.t
        cond = {k: v.to(device_t) for k, v in self.cond.items()}
        return LatentState(z=z, t=t, cond=cond)

    def detach_clone(self) -> LatentState:
        z = self.z.detach().clone()
        t = self.t.detach().clone() if self.t is not None and torch.is_tensor(self.t) else self.t
        cond = {k: v.detach().clone() for k, v in self.cond.items()}
        return LatentState(z=z, t=t, cond=cond)

    def serialize(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {"z": self.z}
        if self.t is not None:
            data["t"] = self.t
        data["cond"] = self.cond
        return data

    @classmethod
    def deserialize(cls, data: Mapping[str, Any]) -> LatentState:
        if "z" not in data:
            raise KeyError("Serialized latent state must include 'z'")
        z = data["z"]
        t = data.get("t")
        cond = dict(data.get("cond", {}))
        return cls(z=z, t=t, cond=cond)

    def __post_init__(self) -> None:
        if not torch.is_tensor(self.z):
            raise TypeError("LatentState.z must be a torch.Tensor")
        if self.z.dim() != 3:
            raise ValueError("LatentState.z must have shape (batch, tokens, dim)")
        if self.t is not None and not (
            (torch.is_tensor(self.t) and self.t.numel() == 1) or isinstance(self.t, (int, float))
        ):
            raise TypeError("LatentState.t must be None, a scalar tensor, or a float/int")
        for k, v in self.cond.items():
            if not torch.is_tensor(v):
                raise TypeError(f"Condition '{k}' must be a torch.Tensor")
