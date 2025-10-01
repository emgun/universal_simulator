import torch

from ups.models.multiphysics_factor_graph import DomainNode, MultiphysicsFactorGraph, PortEdge


def test_factor_graph_converges():
    nodes = {
        "A": DomainNode(state=torch.tensor(1.0), residual=torch.tensor(0.0)),
        "B": DomainNode(state=torch.tensor(-1.0), residual=torch.tensor(0.0)),
    }

    def transfer(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a - b

    graph = MultiphysicsFactorGraph(nodes, [PortEdge("A", "B", transfer)], max_iters=10, tol=1e-5)
    result = graph()
    assert torch.allclose(result["A"], result["B"], atol=1e-3)

