from __future__ import annotations

"""Consistency distillation utilities for few-step diffusion corrector."""

from dataclasses import dataclass
from typing import Callable, Iterable

import torch
from torch import nn

from ups.core.latent_state import LatentState


TeacherFn = Callable[[LatentState, torch.Tensor], LatentState]
StudentFn = Callable[[LatentState, torch.Tensor], LatentState]


@dataclass
class DistillationConfig:
    taus: Iterable[float]
    weight: float = 1.0
    normalize: bool = True


def distillation_loss(
    teacher: TeacherFn,
    student: StudentFn,
    state: LatentState,
    cfg: DistillationConfig,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    total = 0.0
    count = 0
    device = torch.device(device)
    state = state.to(device)
    for tau_val in cfg.taus:
        tau = torch.tensor([tau_val], device=device)
        teacher_state = teacher(state, tau)
        student_state = student(state, tau)
        loss = torch.nn.functional.mse_loss(student_state.z, teacher_state.z.detach())
        total = total + loss
        count += 1
    if cfg.normalize and count > 0:
        total = total / count
    return cfg.weight * total

