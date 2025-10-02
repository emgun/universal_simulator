from __future__ import annotations

"""Latent space evolution operator driven by the PDE-Transformer core."""

from dataclasses import dataclass
from typing import Mapping, Optional

import torch
from torch import nn

from ups.core.blocks_pdet import PDETransformerBlock, PDETransformerConfig
from ups.core.conditioning import AdaLNConditioner, ConditioningConfig
from ups.core.latent_state import LatentState


@dataclass
class LatentOperatorConfig:
    latent_dim: int
    pdet: PDETransformerConfig
    conditioning: Optional[ConditioningConfig] = None
    time_embed_dim: int = 64


class TimeEmbedding(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, dt: torch.Tensor) -> torch.Tensor:
        if dt.dim() == 0:
            dt = dt.unsqueeze(0)
        dt = dt.view(-1, 1)
        return self.proj(dt)


class LatentOperator(nn.Module):
    """Advance latent state by one time step using PDE-Transformer backbone."""

    def __init__(self, cfg: LatentOperatorConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.time_embed = TimeEmbedding(cfg.time_embed_dim)
        self.time_to_latent = nn.Linear(cfg.time_embed_dim, cfg.latent_dim)
        pdet_cfg = cfg.pdet
        if pdet_cfg.input_dim != cfg.latent_dim:
            raise ValueError("PDETransformer input_dim must match latent_dim")
        self.core = PDETransformerBlock(pdet_cfg)
        if cfg.conditioning is not None:
            self.conditioner = AdaLNConditioner(cfg.conditioning)
        else:
            self.conditioner = None
        self.output_norm = nn.LayerNorm(cfg.latent_dim)

    def forward(self, state: LatentState, dt: torch.Tensor) -> LatentState:
        residual = self.step(state, dt)
        new_z = state.z + residual
        new_t = None
        if state.t is not None:
            if torch.is_tensor(state.t):
                new_t = state.t + dt
            else:
                new_t = state.t + float(dt.item())
        else:
            new_t = dt
        return LatentState(z=new_z, t=new_t, cond=state.cond)

    def step(self, state: LatentState, dt: torch.Tensor) -> torch.Tensor:
        z = state.z
        if not torch.is_tensor(dt):
            dt = torch.tensor(dt, device=z.device, dtype=z.dtype)
        else:
            dt = dt.to(device=z.device, dtype=z.dtype)
        dt_embed = self.time_embed(dt)
        if dt_embed.size(0) == 1 and z.size(0) > 1:
            dt_embed = dt_embed.expand(z.size(0), -1)
        time_feat = self.time_to_latent(dt_embed).to(z.device)[:, None, :]
        z = z + time_feat
        if self.conditioner is not None:
            z = self.apply_conditioning(z, state.cond)
        residual = self.core(z)
        residual = self.output_norm(residual)
        return residual

    def apply_conditioning(self, tokens: torch.Tensor, cond: Mapping[str, torch.Tensor]) -> torch.Tensor:
        normed = torch.nn.functional.layer_norm(tokens, tokens.shape[-1:])
        assert self.conditioner is not None
        return self.conditioner.modulate(normed, cond)
