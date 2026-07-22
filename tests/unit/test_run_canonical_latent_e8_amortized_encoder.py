from pathlib import Path

import torch

from scripts.run_canonical_latent_e2_benchmark import (
    Representation,
    evaluate_field,
    representation_points,
    state_coefficients,
)
from scripts.run_canonical_latent_e7_function_space import (
    FunctionSpaceConfig,
    PhysicalFunctionSpace,
)
from scripts.run_canonical_latent_e8_amortized_encoder import (
    AmortizedEncoderConfig,
    QuadratureMomentEncoder,
    run_amortized_encoder,
)


def test_encoder_starts_as_the_frozen_quadrature_moment_map() -> None:
    space = PhysicalFunctionSpace(FunctionSpaceConfig(validation_states=4))
    encoder = QuadratureMomentEncoder(space)
    coefficients = state_coefficients(3, seed=29)
    coords, measure = representation_points(Representation("mesh", 10, 0.24, -0.17), batch=3)
    values = evaluate_field(coefficients, coords)

    prediction = encoder(values, coords, measure)
    basis = space.basis(coords)
    expected = basis.transpose(1, 2) @ (measure.double() * values.double())

    torch.testing.assert_close(prediction, expected, rtol=0.0, atol=0.0)


def test_encoder_is_invariant_to_joint_source_permutation() -> None:
    space = PhysicalFunctionSpace(FunctionSpaceConfig(validation_states=4))
    encoder = QuadratureMomentEncoder(space)
    coefficients = state_coefficients(2, seed=31)
    coords, measure = representation_points(Representation("mesh", 14, 0.24, -0.17), batch=2)
    values = evaluate_field(coefficients, coords)
    permutation = torch.randperm(coords.shape[1], generator=torch.Generator().manual_seed(7))

    original = encoder(values, coords, measure)
    permuted = encoder(values[:, permutation], coords[:, permutation], measure[:, permutation])

    torch.testing.assert_close(original, permuted, rtol=0.0, atol=1e-12)


def test_encoder_has_one_fixed_coefficient_output_per_state() -> None:
    space = PhysicalFunctionSpace(FunctionSpaceConfig(validation_states=4))
    encoder = QuadratureMomentEncoder(space)
    coords, measure = representation_points(Representation("grid", 10, 0.0, 0.0), batch=2)
    values = torch.randn(2, coords.shape[1], 1, dtype=torch.float64)

    latent = encoder(values, coords, measure)

    assert latent.shape == (2, 52, 1)
    assert sum(parameter.numel() for parameter in encoder.parameters()) == 6772


def test_tiny_e8_run_materializes_matched_arms_and_boundary(tmp_path: Path) -> None:
    cfg = AmortizedEncoderConfig(
        train_states=8,
        validation_states=4,
        epochs=1,
        batch_size=4,
    )

    result = run_amortized_encoder(cfg, run_dir=tmp_path)

    assert result["experiment"] == "canonical_latent_e8_amortized_universal_encoder"
    assert result["architecture"]["basis_dimension"] == 52
    assert set(result["training"]) == {"shared", "grid_control", "mesh_control"}
    assert all(arm["optimizer_updates"] == 2 for arm in result["training"].values())
    assert all(arm["scheduled_source_examples"] == 16 for arm in result["training"].values())
    assert set(result["evaluation"]["arms"]["grid_control"]) == {
        "grid_high",
        "grid_unseen",
        "mesh_high",
        "mesh_unseen",
    }
    assert result["evaluation"]["particle_mechanics_probe"]["scientific_gate_attached"] is False
    assert result["causal_decision"]["classification"] in {
        "amortized_universal_encoder_qualified",
        "amortized_encoder_capable_without_positive_transfer",
        "amortized_encoder_not_qualified",
    }
    assert result["state_split"]["heldout_states_read"] == 0
    assert result["boundary"] == {
        "operator_instantiated": False,
        "temporal_transitions": 0,
        "heldout_reads": 0,
        "representation_label_model_inputs": False,
        "task_label_model_inputs": False,
        "provider_calls": 0,
        "routing_paths": 0,
        "original_source_features_available_after_encoding": False,
        "exact_e7_solve_used_at_inference": False,
        "particles_scientifically_qualified": False,
    }
    assert Path(result["result_path"]).is_file()
