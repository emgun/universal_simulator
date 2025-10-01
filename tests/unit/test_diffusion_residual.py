import torch

from ups.core.latent_state import LatentState
from ups.models.diffusion_residual import DiffusionResidual, DiffusionResidualConfig


def test_diffusion_residual_shapes_and_guidance():
    cfg = DiffusionResidualConfig(latent_dim=24, hidden_dim=48, cond_dim=4, residual_guidance_weight=0.5)
    model = DiffusionResidual(cfg)
    state = LatentState(z=torch.randn(3, 10, 24))
    tau = torch.full((3,), 0.3)
    cond = {"meta": torch.randn(3, 4)}
    decoded_residual = torch.randn(3, 10, 24)
    drift = model(state, tau, cond=cond, decoded_residual=decoded_residual)
    assert drift.shape == state.z.shape
    # ensure decoded residual contributes
    no_guidance = model(state, tau, cond=cond, decoded_residual=None)
    assert not torch.allclose(drift, no_guidance)


def test_diffusion_residual_gradients():
    cfg = DiffusionResidualConfig(latent_dim=16, hidden_dim=32)
    model = DiffusionResidual(cfg)
    state = LatentState(z=torch.randn(2, 6, 16, requires_grad=True))
    tau = torch.tensor([0.1, 0.2])
    drift = model(state, tau)
    loss = drift.pow(2).mean()
    loss.backward()
    assert torch.isfinite(state.z.grad).all()
