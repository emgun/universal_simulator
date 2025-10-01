import torch

from ups.core.latent_state import LatentState


def test_latent_state_roundtrip():
    z = torch.randn(2, 16, 32)
    t = torch.tensor(1.5)
    cond = {"bc": torch.randn(2, 4)}
    state = LatentState(z=z, t=t, cond=cond)

    moved = state.to("cpu")
    assert moved.z.shape == z.shape
    assert moved.t == t
    assert moved.cond.keys() == cond.keys()

    copy = state.detach_clone()
    assert copy.z is not state.z
    assert torch.allclose(copy.z, state.z)

    payload = state.serialize()
    restored = LatentState.deserialize(payload)
    assert torch.allclose(restored.z, state.z)
    assert torch.allclose(restored.cond["bc"], state.cond["bc"])


def test_latent_state_time_scalar_handling():
    z = torch.randn(1, 8, 16)
    state = LatentState(z=z, t=2.0)
    moved = state.to("cpu")
    assert moved.t == 2.0

    with torch.inference_mode():
        state_tensor = LatentState(z=z, t=torch.tensor(0.0))
        clone = state_tensor.detach_clone()
        assert torch.allclose(clone.t, torch.tensor(0.0))
