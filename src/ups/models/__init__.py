"""Model-level modules for the Universal Physics Stack."""

from .latent_operator import LatentOperator, LatentOperatorConfig
from .diffusion_residual import DiffusionResidual, DiffusionResidualConfig
from .steady_prior import SteadyPrior, SteadyPriorConfig, steady_residual_norm
from .physics_guards import helmholtz_hodge_projection_grid, positify, interface_flux_projection
from .multiphysics_factor_graph import MultiphysicsFactorGraph, DomainNode, PortEdge
from .particles_contacts import NeighborSearchConfig, SymplecticIntegrator, hierarchical_neighbor_search

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
