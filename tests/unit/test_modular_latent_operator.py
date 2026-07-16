from __future__ import annotations

from pathlib import Path

import pytest
import torch

from scripts import evaluate as evaluate_script
from scripts import train as train_script
from ups.core.blocks_pdet import PDETransformerConfig
from ups.core.latent_state import LatentState
from ups.models.latent_operator import (
    LatentOperator,
    LatentOperatorConfig,
    RoutedAdapterConfig,
)


def _config(*, adapters: RoutedAdapterConfig | None) -> LatentOperatorConfig:
    latent_dim = 8
    return LatentOperatorConfig(
        latent_dim=latent_dim,
        pdet=PDETransformerConfig(
            input_dim=latent_dim,
            hidden_dim=16,
            depths=(1,),
            group_size=4,
            num_heads=2,
        ),
        time_embed_dim=latent_dim,
        routed_adapters=adapters,
    )


def _adapter_config(**overrides) -> RoutedAdapterConfig:
    values = {
        "num_experts": 3,
        "bottleneck_dim": 4,
        "route_source": "task_id",
        "input_enabled": True,
        "output_enabled": True,
        "zero_init": True,
    }
    values.update(overrides)
    return RoutedAdapterConfig(**values)


def _state(route: torch.Tensor, *, requires_grad: bool = False) -> LatentState:
    return LatentState(
        z=torch.randn(route.shape[0], 5, 8, requires_grad=requires_grad),
        t=torch.tensor(0.0),
        cond={"task_id": route},
    )


def test_disabled_adapters_preserve_legacy_state_dict() -> None:
    legacy = LatentOperator(_config(adapters=None))
    explicitly_disabled = LatentOperator(_config(adapters=None))

    assert legacy.state_dict().keys() == explicitly_disabled.state_dict().keys()
    assert not any("adapter" in key for key in legacy.state_dict())


def test_zero_initialized_adapters_preserve_initial_operator_behavior() -> None:
    torch.manual_seed(7)
    legacy = LatentOperator(_config(adapters=None))
    torch.manual_seed(7)
    modular = LatentOperator(_config(adapters=_adapter_config()))
    route = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    state = _state(route)

    legacy_residual = legacy.step(LatentState(z=state.z, t=state.t, cond={}), torch.tensor(0.1))
    modular_residual = modular.step(state, torch.tensor(0.1))

    assert torch.allclose(modular_residual, legacy_residual)


def test_mixed_batch_routes_each_sample_to_its_selected_expert() -> None:
    operator = LatentOperator(
        _config(adapters=_adapter_config(input_enabled=False, output_enabled=True, zero_init=True))
    )
    assert operator.output_adapters is not None
    for index, expert in enumerate(operator.output_adapters.experts):
        torch.nn.init.zeros_(expert.up.weight)
        torch.nn.init.constant_(expert.up.bias, float(index + 1))

    route = torch.eye(3)
    state = _state(route)
    trunk_residual = operator.output_norm(
        operator.core(
            state.z
            + operator.time_to_latent(operator.time_embed(torch.full((3,), 0.1)))[:, None, :]
        )
    )
    routed_residual = operator.step(state, torch.tensor(0.1))

    expected_offsets = torch.tensor([1.0, 2.0, 3.0]).view(3, 1, 1)
    assert torch.allclose(routed_residual - trunk_residual, expected_offsets)


def test_only_selected_expert_receives_gradient() -> None:
    operator = LatentOperator(
        _config(adapters=_adapter_config(input_enabled=False, output_enabled=True, zero_init=True))
    )
    route = torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    loss = operator.step(_state(route, requires_grad=True), torch.tensor(0.1)).sum()
    loss.backward()

    assert operator.output_adapters is not None
    assert operator.output_adapters.experts[1].up.bias.grad is not None
    assert operator.output_adapters.experts[1].up.bias.grad.abs().sum() > 0
    for index in (0, 2):
        grad = operator.output_adapters.experts[index].up.bias.grad
        assert grad is not None
        assert torch.count_nonzero(grad) == 0


@pytest.mark.parametrize(
    ("route", "match"),
    [
        (torch.tensor([1.0, 0.0, 0.0]), "shape"),
        (torch.tensor([[1.0, 0.0]]), "shape"),
        (torch.tensor([[0.5, 0.5, 0.0]]), "one-hot"),
        (torch.tensor([[1.0, 1.0, 0.0]]), "exactly one"),
        (torch.tensor([[0.0, 0.0, 0.0]]), "exactly one"),
        (torch.tensor([[float("nan"), 0.0, 0.0]]), "finite"),
    ],
)
def test_routed_adapters_reject_invalid_routes(route: torch.Tensor, match: str) -> None:
    operator = LatentOperator(_config(adapters=_adapter_config()))
    batch = route.shape[0] if route.dim() == 2 else 1
    state = LatentState(z=torch.randn(batch, 5, 8), cond={"task_id": route})

    with pytest.raises(ValueError, match=match):
        operator.step(state, torch.tensor(0.1))


def test_routed_adapters_require_configured_condition_source() -> None:
    operator = LatentOperator(_config(adapters=_adapter_config()))
    state = LatentState(z=torch.randn(1, 5, 8), cond={})

    with pytest.raises(ValueError, match="require condition source 'task_id'"):
        operator.step(state, torch.tensor(0.1))


def test_modular_checkpoint_round_trip(tmp_path: Path) -> None:
    config = _config(adapters=_adapter_config(zero_init=False))
    source = LatentOperator(config)
    checkpoint = tmp_path / "operator.pt"
    torch.save(source.state_dict(), checkpoint)

    restored = LatentOperator(config)
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))

    route = torch.tensor([[0.0, 0.0, 1.0]])
    state = _state(route)
    assert torch.allclose(
        source.step(state, torch.tensor(0.1)),
        restored.step(state, torch.tensor(0.1)),
    )


def test_modular_checkpoint_does_not_silently_load_into_legacy_operator() -> None:
    modular = LatentOperator(_config(adapters=_adapter_config()))
    legacy = LatentOperator(_config(adapters=None))

    with pytest.raises(RuntimeError, match="Unexpected key"):
        legacy.load_state_dict(modular.state_dict())


def _factory_config() -> dict:
    return {
        "latent": {"dim": 8, "tokens": 4},
        "data": {
            "conditioning_schema": {
                "task_vocab": ["advection1d", "burgers1d", "darcy2d"],
                "param_vocab": ["beta", "nu"],
            }
        },
        "training": {"auto_conditioning": False, "lambda_semigroup": 0.0},
        "operator": {
            "conditioning": {"sources": {"task_id": 3}, "hidden_dim": 8},
            "pdet": {
                "input_dim": 8,
                "hidden_dim": 16,
                "depths": [1],
                "group_size": 4,
                "num_heads": 2,
            },
            "routed_adapters": {
                "enabled": True,
                "route_source": "task_id",
                "route_vocab": ["advection1d", "burgers1d", "darcy2d"],
                "bottleneck_dim": 4,
                "input_enabled": True,
                "output_enabled": True,
                "zero_init": True,
            },
        },
    }


def test_train_and_evaluate_factories_build_identical_modular_state() -> None:
    train_operator = train_script.make_operator(_factory_config())
    evaluate_operator = evaluate_script.make_operator(_factory_config())

    assert train_operator.state_dict().keys() == evaluate_operator.state_dict().keys()
    evaluate_operator.load_state_dict(train_operator.state_dict())
    assert train_operator.cfg.routed_adapters == evaluate_operator.cfg.routed_adapters


@pytest.mark.parametrize("factory", [train_script.make_operator, evaluate_script.make_operator])
def test_modular_factory_fails_closed_on_route_vocab_or_semigroup(factory) -> None:
    config = _factory_config()
    config["operator"]["routed_adapters"]["route_vocab"] = ["darcy2d"]
    with pytest.raises(ValueError, match="vocabulary"):
        factory(config)

    config = _factory_config()
    config["training"]["lambda_semigroup"] = 0.1
    with pytest.raises(ValueError, match="lambda_semigroup=0"):
        factory(config)
