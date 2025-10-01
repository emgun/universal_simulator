import torch

from ups.models.particles_contacts import NeighborSearchConfig, SymplecticIntegrator, hierarchical_neighbor_search


def test_hierarchical_neighbor_search_returns_edges():
    positions = torch.rand(20, 3)
    radii = torch.full((20,), 0.2)
    edges = hierarchical_neighbor_search(positions, radii, NeighborSearchConfig(cell_size=0.3))
    assert edges.shape[1] == 2
    assert (edges[:, 0] != edges[:, 1]).all()


def test_symplectic_integrator_energy():
    def forces(pos: torch.Tensor) -> torch.Tensor:
        return -pos  # harmonic potential

    masses = torch.ones(5)
    integrator = SymplecticIntegrator()
    positions = torch.randn(5, 3)
    velocities = torch.zeros(5, 3)
    initial_energy = (0.5 * (positions**2).sum(dim=-1) + 0.5 * (velocities**2).sum(dim=-1)).sum()
    for _ in range(50):
        positions, velocities = integrator.step(positions, velocities, forces, masses, dt=0.05)
    final_energy = (0.5 * (positions**2).sum(dim=-1) + 0.5 * (velocities**2).sum(dim=-1)).sum()
    assert torch.isfinite(final_energy)
    assert abs(final_energy - initial_energy) < 0.5
