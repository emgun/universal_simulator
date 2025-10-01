from __future__ import annotations

"""Latent-space data assimilation using ensemble Kalman filter (EnKF)."""

from dataclasses import dataclass
from typing import Callable, Dict

import torch

from ups.core.latent_state import LatentState


ObservationFn = Callable[[LatentState], torch.Tensor]


@dataclass
class EnKFConfig:
    inflation: float = 1.01
    jitter: float = 1e-4


def latent_enkf(
    ensemble: Dict[str, LatentState],
    observation: ObservationFn,
    observation_noise: torch.Tensor,
    config: EnKFConfig,
) -> Dict[str, LatentState]:
    keys = list(ensemble.keys())
    stacked = torch.stack([ensemble[k].z for k in keys], dim=0)
    B, T, D = stacked.shape[1:]
    flat = stacked.view(len(keys), -1)
    mean = flat.mean(dim=0, keepdim=True)
    anomalies = (flat - mean) * config.inflation

    obs = torch.stack([observation(ensemble[k]).reshape(-1) for k in keys], dim=0)
    obs_mean = obs.mean(dim=0, keepdim=True)
    obs_anom = obs - obs_mean

    denom = max(len(keys) - 1, 1)
    cov_zz = anomalies.t().mm(anomalies) / denom
    cov_zo = anomalies.t().mm(obs_anom) / denom
    noise = observation_noise + config.jitter * torch.eye(observation_noise.size(-1), device=observation_noise.device)
    gain = torch.linalg.solve(noise, cov_zo.t()).t()
    updated = {}
    obs_mean_flat = obs_mean.flatten()
    for i, key in enumerate(keys):
        innovation = obs_mean_flat + obs_anom[i]
        delta = gain.mm(innovation.unsqueeze(-1)).view(B, T, D)
        updated_z = ensemble[key].z + delta
        updated[key] = LatentState(z=updated_z, t=ensemble[key].t, cond=ensemble[key].cond)
    return updated
