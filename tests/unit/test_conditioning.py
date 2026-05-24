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
    # Zero-initialized projections must make conditioning an exact no-op.
    assert torch.allclose(out, normed)


def test_adaln_conditioner_accepts_set_structured_sources():
    cfg = ConditioningConfig(latent_dim=12, hidden_dim=8, sources={"equation_nodes": 5})
    conditioner = AdaLNConditioner(cfg)
    cond = {"equation_nodes": torch.randn(3, 4, 5)}
    mods = conditioner(cond)
    assert mods["scale"].shape == (3, 12)
    assert mods["shift"].shape == (3, 12)
    assert mods["gate"].shape == (3, 12)
