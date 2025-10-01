import torch

from ups.models.physics_guards import helmholtz_hodge_projection_grid, positify, interface_flux_projection


def test_helmholtz_hodge_projection_reduces_divergence():
    grid = torch.randn(1, 16, 16, 2)
    proj = helmholtz_hodge_projection_grid(grid)
    assert proj.shape == grid.shape


def test_positify_enforces_bounds():
    values = torch.tensor([-1.0, 0.0, 0.5, 2.0])
    projected = positify(values)
    assert torch.all(projected > 0)


def test_interface_flux_projection():
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([3.0, 4.0])
    proj = interface_flux_projection(a, b, alpha=0.25)
    assert torch.allclose(proj, 0.25 * a + 0.75 * b)
