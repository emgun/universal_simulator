from __future__ import annotations

"""Simple latent-space baseline models used for benchmarking."""

from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn


@dataclass
class BaselineConfig:
    latent_dim: int
    tokens: int
    hidden_dim: int = 0


class IdentityBaseline(nn.Module):
    def __init__(self, cfg: BaselineConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def forward(self, z0: torch.Tensor, _: Dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        return z0


class LinearBaseline(nn.Module):
    """Single linear layer applied per-token."""

    def __init__(self, cfg: BaselineConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.linear = nn.Linear(cfg.latent_dim, cfg.latent_dim)

    def forward(self, z0: torch.Tensor, _: Dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        B, T, D = z0.shape
        return self.linear(z0.view(B * T, D)).view(B, T, D)


class MLPBaseline(nn.Module):
    """Two-layer MLP applied per token."""

    def __init__(self, cfg: BaselineConfig) -> None:
        super().__init__()
        hidden = cfg.hidden_dim or max(cfg.latent_dim * 2, 64)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.latent_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, cfg.latent_dim),
        )

    def forward(self, z0: torch.Tensor, _: Dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        B, T, D = z0.shape
        return self.mlp(z0.view(B * T, D)).view(B, T, D)


def build_baseline(name: str, cfg: BaselineConfig) -> nn.Module:
    name = name.lower()
    if name == "identity":
        return IdentityBaseline(cfg)
    if name == "linear":
        return LinearBaseline(cfg)
    if name == "mlp":
        return MLPBaseline(cfg)
    raise ValueError(f"Unknown baseline model '{name}'")
