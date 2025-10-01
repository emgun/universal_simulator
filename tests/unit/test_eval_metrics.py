import torch

from ups.eval.metrics import conservation_gap, nrmse, spectral_energy_error


def test_nrmse_zero_when_identical():
    x = torch.randn(4, 16)
    assert torch.allclose(nrmse(x, x), torch.tensor(0.0))


def test_spectral_energy_error_positive():
    x = torch.randn(2, 32)
    y = x * 1.1
    err = spectral_energy_error(x, y)
    assert err >= 0


def test_conservation_gap_mean_absolute_difference():
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = torch.zeros_like(a)
    gap = conservation_gap(a, b)
    assert torch.allclose(gap, torch.tensor(5.0))
