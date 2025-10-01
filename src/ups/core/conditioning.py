from __future__ import annotations

"""Adaptive Layer Normalisation (AdaLN) conditioning utilities."""

from dataclasses import dataclass
from typing import Mapping, Optional

import torch
from torch import nn


@dataclass
class ConditioningConfig:
    latent_dim: int
    hidden_dim: int = 128
    sources: Mapping[str, int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.sources is None or len(self.sources) == 0:
            raise ValueError("ConditioningConfig.sources must map at least one key to an input dimension")


class AdaLNConditioner(nn.Module):
    """Produce scale/shift/gate signals for Adaptive LayerNorm from metadata."""

    def __init__(self, cfg: ConditioningConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embedders = nn.ModuleDict()
        out_dim = cfg.latent_dim * 3

        for name, in_dim in cfg.sources.items():
            embed = nn.Sequential(
                nn.Linear(in_dim, cfg.hidden_dim),
                nn.GELU(),
                nn.Linear(cfg.hidden_dim, out_dim),
            )
            # Start from neutral modulation (scale≈1, shift≈0, gate≈1).
            nn.init.zeros_(embed[-1].weight)
            nn.init.zeros_(embed[-1].bias)
            self.embedders[name] = embed

        self.register_buffer("_dummy", torch.zeros(1), persistent=False)

    def forward(self, cond: Mapping[str, torch.Tensor]) -> Mapping[str, torch.Tensor]:
        if not cond:
            cond = {}
        total: Optional[torch.Tensor] = None
        batch = None
        for name, embed in self.embedders.items():
            if name not in cond:
                continue
            tensor = cond[name]
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)
            if tensor.dim() != 2:
                raise ValueError(f"Condition tensor for '{name}' must have shape (batch, features)")
            if tensor.size(-1) != embed[0].in_features:
                raise ValueError(
                    f"Condition tensor for '{name}' expected last dim {embed[0].in_features}, got {tensor.size(-1)}"
                )
            batch = tensor.size(0)
            contrib = embed(tensor)
            total = contrib if total is None else total + contrib

        if total is None:
            if batch is None:
                batch = self._dummy.shape[0]
            device = next(self.parameters()).device
            total = torch.zeros(batch, self.cfg.latent_dim * 3, device=device)

        gamma, beta, gate_raw = total.chunk(3, dim=-1)
        scale = 1.0 + gamma
        shift = beta
        gate = torch.sigmoid(gate_raw + 2.0)  # start near 0.88 to preserve signal initially
        return {"scale": scale, "shift": shift, "gate": gate}

    def modulate(self, normed: torch.Tensor, cond: Mapping[str, torch.Tensor]) -> torch.Tensor:
        mods = self(cond)
        scale = mods["scale"].unsqueeze(1)
        shift = mods["shift"].unsqueeze(1)
        gate = mods["gate"].unsqueeze(1)
        return gate * (scale * normed + shift)
