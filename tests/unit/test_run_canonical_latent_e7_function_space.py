from pathlib import Path

import pytest
import torch

from scripts.run_canonical_latent_e2_benchmark import Representation, representation_points
from scripts.run_canonical_latent_e7_function_space import (
    FunctionSpaceConfig,
    PhysicalFunctionSpace,
    basis_calibration,
    run_function_space,
)


def test_basis_has_frozen_dimension_and_coordinate_semantics() -> None:
    cfg = FunctionSpaceConfig()
    space = PhysicalFunctionSpace(cfg)
    coords = torch.tensor([[0.1, 0.2], [0.7, 0.8]], dtype=torch.float64)

    basis = space.basis(coords)

    assert cfg.basis_dim == 52
    assert basis.shape == (2, 52)
    torch.testing.assert_close(basis[:, 0], torch.ones(2, dtype=torch.float64))
    torch.testing.assert_close(basis[:, -1], 12.0 * (coords[:, 0] - 0.5) * (coords[:, 1] - 0.5))


def test_weighted_projection_recovers_a_field_in_the_common_space() -> None:
    cfg = FunctionSpaceConfig()
    space = PhysicalFunctionSpace(cfg)
    coords, measure = representation_points(Representation("mesh", 14, 0.24, -0.17), batch=3)
    generator = torch.Generator().manual_seed(29)
    expected = torch.randn(3, cfg.basis_dim, 1, generator=generator, dtype=torch.float64)
    values = space.decode(expected, coords)

    projected, design = space.project(values, coords, measure)
    reconstructed = space.decode(projected, coords)

    assert design["rank"] == cfg.basis_dim
    assert design["condition_number"] <= cfg.max_condition_number
    torch.testing.assert_close(projected, expected, rtol=0.0, atol=1e-10)
    torch.testing.assert_close(reconstructed, values, rtol=0.0, atol=1e-10)


def test_projection_is_invariant_to_joint_source_permutation() -> None:
    cfg = FunctionSpaceConfig()
    space = PhysicalFunctionSpace(cfg)
    coords, measure = representation_points(Representation("grid", 10, 0.0, 0.0), batch=2)
    expected = torch.randn(
        2,
        cfg.basis_dim,
        1,
        generator=torch.Generator().manual_seed(41),
        dtype=torch.float64,
    )
    values = space.decode(expected, coords)
    permutation = torch.randperm(coords.shape[1], generator=torch.Generator().manual_seed(7))

    original, _ = space.project(values, coords, measure)
    permuted, _ = space.project(
        values[:, permutation], coords[:, permutation], measure[:, permutation]
    )

    torch.testing.assert_close(original, permuted, rtol=0.0, atol=1e-10)


def test_frozen_basis_calibration_and_geometry_conditioning() -> None:
    cfg = FunctionSpaceConfig()
    report = basis_calibration(PhysicalFunctionSpace(cfg), cfg)

    assert report["maximum_component_nrmse"] == pytest.approx(0.0188519061, abs=1e-9)
    assert report["pass"] is True
    assert report["design"]["rank"] == cfg.basis_dim
    assert report["design"]["condition_number"] <= cfg.max_condition_number


def test_tiny_e7_run_materializes_decision_and_no_bypass_boundary(tmp_path: Path) -> None:
    cfg = FunctionSpaceConfig(validation_states=4, calibration_resolution=32)

    result = run_function_space(cfg, run_dir=tmp_path)

    assert result["experiment"] == "canonical_latent_e7_function_space"
    assert result["basis_dimension"] == 52
    assert set(result["families"]) == {"grid", "mesh"}
    assert result["causal_decision"]["classification"] in {
        "function_space_latent_qualified",
        "function_space_sufficient_projection_unstable",
        "function_space_latent_not_qualified",
    }
    assert result["state_split"]["training_states_read"] == 0
    assert result["boundary"] == {
        "learned_parameters": 0,
        "optimizer_updates": 0,
        "operator_instantiated": False,
        "heldout_reads": 0,
        "representation_label_model_inputs": False,
        "task_label_model_inputs": False,
        "provider_calls": 0,
        "routing_paths": 0,
        "original_source_features_available_after_projection": False,
        "particles_scientifically_qualified": False,
    }
    assert Path(result["result_path"]).is_file()
