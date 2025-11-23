from __future__ import annotations

"""Latent space evolution operator driven by the PDE-Transformer core."""

from dataclasses import dataclass
from typing import Literal, Mapping, Optional, Union

import torch
from torch import nn

from ups.core.blocks_pdet import PDETransformerBlock, PDETransformerConfig
from ups.core.conditioning import AdaLNConditioner, ConditioningConfig
from ups.core.latent_state import LatentState
from ups.models.pure_transformer import PureTransformer, PureTransformerConfig


@dataclass
class LatentOperatorConfig:
    """Configuration for latent space operator.

    Args:
        latent_dim: Latent token dimension.
        pdet: Configuration for approximator (PDETransformerConfig or PureTransformerConfig).
        architecture_type: Type of approximator architecture.
            - "pdet_unet": U-shaped PDE-Transformer (default, current production)
            - "pdet_stack": Pure stacked transformer (new, UPT-style)
        conditioning: Optional adaptive conditioning config.
        time_embed_dim: Time embedding dimension. Default: 64.
    """

    latent_dim: int
    pdet: Union[PDETransformerConfig, PureTransformerConfig]
    architecture_type: Literal["pdet_unet", "pdet_stack"] = "pdet_unet"
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
        # Robustly coerce dt to shape (B, 1) to avoid matmul 1-D errors under FSDP/DDP
        dt = torch.as_tensor(dt, device=dt.device if torch.is_tensor(dt) else None)
        if dt.dim() == 0:
            dt = dt.unsqueeze(0)
        if dt.dim() == 1:
            dt = dt.unsqueeze(-1)
        dt = dt.reshape(-1, 1)
        return self.proj(dt)


class LatentOperator(nn.Module):
    """Advance latent state by one time step using approximator backbone.

    Supports both U-shaped PDE-Transformer and pure stacked transformer architectures.
    """

    def __init__(self, cfg: LatentOperatorConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.time_embed = TimeEmbedding(cfg.time_embed_dim)
        self.time_to_latent = nn.Linear(cfg.time_embed_dim, cfg.latent_dim)

        # Validate dimension match
        pdet_cfg = cfg.pdet
        if pdet_cfg.input_dim != cfg.latent_dim:
            raise ValueError(
                f"Approximator input_dim ({pdet_cfg.input_dim}) "
                f"must match latent_dim ({cfg.latent_dim})"
            )

        # Instantiate core approximator based on architecture type
        if cfg.architecture_type == "pdet_unet":
            self.core = PDETransformerBlock(pdet_cfg)
        elif cfg.architecture_type == "pdet_stack":
            self.core = PureTransformer(pdet_cfg)
        else:
            raise ValueError(f"Unknown architecture_type: {cfg.architecture_type}")

        # Optional conditioning
        if cfg.conditioning is not None:
            self.conditioner = AdaLNConditioner(cfg.conditioning)
        else:
            self.conditioner = None

        self.output_norm = nn.LayerNorm(cfg.latent_dim)

    def forward(self, state: LatentState, dt: torch.Tensor) -> LatentState:
        # NOTE: CUDA graphs disabled for DDP/FSDP compatibility
        # CUDA graphs cache kernel launches and conflict with distributed collectives
        # Only use CUDA graph markers in single-GPU mode
        try:
            import torch.distributed as dist
            if not dist.is_initialized():
                if hasattr(torch, "compiler") and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                    torch.compiler.cudagraph_mark_step_begin()
        except Exception:
            pass
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
        
        # DEBUG: Print shapes to diagnose distributed crash
        # print(f"[DEBUG] Rank {torch.distributed.get_rank() if torch.distributed.is_initialized() else 0} z={z.shape} dt={dt.shape}")

        dt_embed = self.time_embed(dt)
        if dt_embed.size(0) == 1 and z.size(0) > 1:
            dt_embed = dt_embed.expand(z.size(0), -1)
        
        # Debug check for mat2 error
        if dt_embed.dim() != 2:
             print(f"[LatentOperator] Error: dt_embed has shape {dt_embed.shape}, expected 2D")
        
        time_feat = self.time_to_latent(dt_embed).to(z.device)[:, None, :]
        z = z + time_feat
        if self.conditioner is not None:
            z = self.apply_conditioning(z, state.cond)
        residual = self.core(z)
        # NOTE: .clone() removed for DDP/FSDP compatibility
        # Clone breaks gradient flow with distributed wrappers
        residual = self.output_norm(residual)
        return residual

    def apply_conditioning(
        self, tokens: torch.Tensor, cond: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        normed = torch.nn.functional.layer_norm(tokens, tokens.shape[-1:])
        assert self.conditioner is not None
        return self.conditioner.modulate(normed, cond)
