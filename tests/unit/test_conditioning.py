import torch

from ups.core.conditioning import AdaLNConditioner, ConditioningConfig


def test_adaln_conditioner_outputs():
    cfg = ConditioningConfig(latent_dim=32, hidden_dim=16, sources={"pde": 4, "params": 6})
    conditioner = AdaLNConditioner(cfg)
    cond = {
        "pde": torch.randn(5, 4),
        "params": torch.randn(5, 6),
    }
    mods = conditioner(cond)
    assert set(mods.keys()) == {"scale", "shift", "gate"}
    for name in ["scale", "shift", "gate"]:
        assert mods[name].shape == (5, 32)
    assert torch.all((mods["gate"] >= 0.0) & (mods["gate"] <= 1.0))


def test_modulate_applies_scale_shift_gate():
    cfg = ConditioningConfig(latent_dim=16, hidden_dim=16, sources={"geom": 3})
    conditioner = AdaLNConditioner(cfg)
    normed = torch.ones(2, 7, 16)
    cond = {"geom": torch.zeros(2, 3)}
    out = conditioner.modulate(normed, cond)
    assert out.shape == normed.shape
    # With zero conditioning, modulation should be a constant scaling factor.
    expected = torch.sigmoid(torch.tensor(2.0)) * torch.ones_like(out)
    assert torch.allclose(out, expected)
