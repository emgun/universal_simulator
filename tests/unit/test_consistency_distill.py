import torch

from ups.core.latent_state import LatentState
from ups.training.consistency_distill import DistillationConfig, distillation_loss


def teacher(state: LatentState, tau: torch.Tensor) -> LatentState:
    return LatentState(z=state.z + tau.view(-1, 1, 1), cond=state.cond)


def student(state: LatentState, tau: torch.Tensor) -> LatentState:
    return LatentState(z=state.z + 0.5 * tau.view(-1, 1, 1), cond=state.cond)


def test_distillation_loss_computes_mean():
    z = torch.zeros(2, 4, 3)
    state = LatentState(z=z)
    cfg = DistillationConfig(taus=[0.1, 0.2, 0.3], weight=2.0)
    loss = distillation_loss(teacher, student, state, cfg)
    assert torch.isfinite(loss)
    assert loss > 0


def test_distillation_loss_normalization_toggle():
    z = torch.zeros(1, 3, 2)
    state = LatentState(z=z)
    cfg_norm = DistillationConfig(taus=[0.1, 0.2], normalize=True)
    cfg_no_norm = DistillationConfig(taus=[0.1, 0.2], normalize=False)
    loss_norm = distillation_loss(teacher, student, state, cfg_norm)
    loss_no_norm = distillation_loss(teacher, student, state, cfg_no_norm)
    assert loss_no_norm > loss_norm

