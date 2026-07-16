from __future__ import annotations

"""Latent space evolution operator driven by the PDE-Transformer core."""

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import nn

from ups.core.blocks_pdet import PDETransformerBlock, PDETransformerConfig
from ups.core.conditioning import AdaLNConditioner, ConditioningConfig
from ups.core.latent_state import LatentState


@dataclass(frozen=True)
class RoutedAdapterConfig:
    """Task-routed residual adapters around the shared operator trunk."""

    num_experts: int
    bottleneck_dim: int = 16
    route_source: str = "task_id"
    input_enabled: bool = True
    output_enabled: bool = True
    zero_init: bool = True

    def __post_init__(self) -> None:
        if self.num_experts <= 0:
            raise ValueError("RoutedAdapterConfig.num_experts must be positive")
        if self.bottleneck_dim <= 0:
            raise ValueError("RoutedAdapterConfig.bottleneck_dim must be positive")
        if not self.route_source:
            raise ValueError("RoutedAdapterConfig.route_source must not be empty")
        if not self.input_enabled and not self.output_enabled:
            raise ValueError("Routed adapters must enable an input or output adapter")


@dataclass
class LatentOperatorConfig:
    latent_dim: int
    pdet: PDETransformerConfig
    conditioning: ConditioningConfig | None = None
    time_embed_dim: int = 64
    routed_adapters: RoutedAdapterConfig | None = None


class _ResidualAdapter(nn.Module):
    def __init__(self, latent_dim: int, bottleneck_dim: int, *, zero_init: bool) -> None:
        super().__init__()
        self.down = nn.Linear(latent_dim, bottleneck_dim)
        self.activation = nn.SiLU()
        self.up = nn.Linear(bottleneck_dim, latent_dim)
        if zero_init:
            nn.init.zeros_(self.up.weight)
            nn.init.zeros_(self.up.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.up(self.activation(self.down(tokens)))


class _TaskRoutedResidualAdapters(nn.Module):
    def __init__(self, latent_dim: int, cfg: RoutedAdapterConfig) -> None:
        super().__init__()
        self.num_experts = cfg.num_experts
        self.experts = nn.ModuleList(
            _ResidualAdapter(latent_dim, cfg.bottleneck_dim, zero_init=cfg.zero_init)
            for _ in range(cfg.num_experts)
        )

    def forward(self, tokens: torch.Tensor, route: torch.Tensor) -> torch.Tensor:
        if route.dim() != 2:
            raise ValueError("Task route must have shape (batch, num_experts)")
        if route.shape != (tokens.shape[0], self.num_experts):
            raise ValueError(
                "Task route shape must match the token batch and configured expert count"
            )
        if not torch.isfinite(route).all():
            raise ValueError("Task route must contain only finite values")
        route_float = route.to(device=tokens.device, dtype=tokens.dtype)
        zeros = torch.zeros((), device=tokens.device, dtype=tokens.dtype)
        ones = torch.ones((), device=tokens.device, dtype=tokens.dtype)
        if not torch.all((route_float == zeros) | (route_float == ones)):
            raise ValueError("Task route must be one-hot")
        if not torch.all(route_float.sum(dim=-1) == ones):
            raise ValueError("Task route must select exactly one expert per sample")

        expert_deltas = torch.stack([expert(tokens) for expert in self.experts], dim=1)
        return torch.einsum("be,betd->btd", route_float, expert_deltas)


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
        adapter_cfg = cfg.routed_adapters
        self.adapter_route_source = adapter_cfg.route_source if adapter_cfg is not None else None
        self.input_adapters = (
            _TaskRoutedResidualAdapters(cfg.latent_dim, adapter_cfg)
            if adapter_cfg is not None and adapter_cfg.input_enabled
            else None
        )
        self.output_adapters = (
            _TaskRoutedResidualAdapters(cfg.latent_dim, adapter_cfg)
            if adapter_cfg is not None and adapter_cfg.output_enabled
            else None
        )

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
        route = None
        if self.adapter_route_source is not None:
            route = state.cond.get(self.adapter_route_source)
            if route is None:
                raise ValueError(
                    f"Routed adapters require condition source {self.adapter_route_source!r}"
                )
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
        if self.input_adapters is not None:
            assert route is not None
            z = z + self.input_adapters(z, route)
        residual = self.core(z)
        residual = self.output_norm(residual)
        if self.output_adapters is not None:
            assert route is not None
            residual = residual + self.output_adapters(residual, route)
        return residual

    def apply_conditioning(
        self, tokens: torch.Tensor, cond: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        normed = torch.nn.functional.layer_norm(tokens, tokens.shape[-1:])
        assert self.conditioner is not None
        return self.conditioner.modulate(normed, cond)
