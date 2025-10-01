from __future__ import annotations

"""Multiphysics factor graph for coupling domain nodes and ports."""

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional

import torch
from torch import nn


@dataclass
class DomainNode:
    state: torch.Tensor
    residual: torch.Tensor


@dataclass
class PortEdge:
    node_a: str
    node_b: str
    transfer: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class MultiphysicsFactorGraph(nn.Module):
    def __init__(self, nodes: Dict[str, DomainNode], edges: Iterable[PortEdge], max_iters: int = 8, tol: float = 1e-4) -> None:
        super().__init__()
        self.nodes = nodes
        self.edges = list(edges)
        self.max_iters = max_iters
        self.tol = tol

    def forward(self) -> Dict[str, torch.Tensor]:
        for _ in range(self.max_iters):
            max_res = 0.0
            for edge in self.edges:
                node_a = self.nodes[edge.node_a]
                node_b = self.nodes[edge.node_b]
                delta = edge.transfer(node_a.state, node_b.state)
                node_a.state = node_a.state - 0.5 * delta
                node_b.state = node_b.state + 0.5 * delta
                node_a.residual = delta
                node_b.residual = -delta
                max_res = max(max_res, delta.abs().max().item())
            if max_res < self.tol:
                break
        return {name: node.state for name, node in self.nodes.items()}

