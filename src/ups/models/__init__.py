"""Model-level modules for the Universal Physics Stack."""

from .diffusion_residual import DiffusionResidual, DiffusionResidualConfig
from .latent_operator import LatentOperator, LatentOperatorConfig
from .multiphysics_factor_graph import DomainNode, MultiphysicsFactorGraph, PortEdge
from .particles_contacts import (
    NeighborSearchConfig,
    SymplecticIntegrator,
    hierarchical_neighbor_search,
)
from .physics_guards import helmholtz_hodge_projection_grid, interface_flux_projection, positify
from .steady_prior import SteadyPrior, SteadyPriorConfig, steady_residual_norm

__all__ = [
    "LatentOperator",
    "LatentOperatorConfig",
    "DiffusionResidual",
    "DiffusionResidualConfig",
    "SteadyPrior",
    "SteadyPriorConfig",
    "steady_residual_norm",
    "helmholtz_hodge_projection_grid",
    "positify",
    "interface_flux_projection",
    "MultiphysicsFactorGraph",
    "DomainNode",
    "PortEdge",
    "NeighborSearchConfig",
    "SymplecticIntegrator",
    "hierarchical_neighbor_search",
]
