import torch

from ups.core.latent_state import LatentState
from ups.inference.da_latent import EnKFConfig, latent_enkf


def observation_fn(state: LatentState) -> torch.Tensor:
    return state.z.mean(dim=1)


def test_ensemble_update_reduces_difference():
    ensemble = {
        "m1": LatentState(z=torch.randn(4, 6, 3)),
        "m2": LatentState(z=torch.randn(4, 6, 3)),
        "m3": LatentState(z=torch.randn(4, 6, 3)),
    }
    obs_noise = torch.eye(4 * 3)
    updated = latent_enkf(ensemble, observation_fn, obs_noise, EnKFConfig())
    assert updated.keys() == ensemble.keys()
