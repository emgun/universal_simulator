from pathlib import Path

import pytest
import torch

from scripts.run_canonical_latent_e7_function_space import PhysicalFunctionSpace
from scripts.run_canonical_latent_e9_geometry_universal_projection import (
    GeometryProjectionConfig,
    encode_paths,
    geometry_samples,
    run_geometry_projection,
)


def test_particle_geometries_are_nested_positive_and_well_conditioned() -> None:
    cfg = GeometryProjectionConfig(geometry_realizations=2)
    space = PhysicalFunctionSpace(cfg.function_space_config())
    samples = geometry_samples(cfg)

    for family in ("uniform_particles", "warped_particles"):
        for realization in range(cfg.geometry_realizations):
            low = samples[family]["low"][realization]
            high = samples[family]["high"][realization]
            torch.testing.assert_close(
                low.coords, high.coords[: cfg.particle_low_count], rtol=0.0, atol=0.0
            )
            assert torch.all(low.measure > 0)
            assert torch.all(high.measure > 0)
            basis = space.basis(high.coords)
            singular_values = torch.linalg.svdvals(basis * high.measure.sqrt())
            gram = basis.transpose(0, 1) @ (high.measure * basis)
            assert int((singular_values > cfg.rank_tolerance).sum()) == 52
            design_condition = float(singular_values[0] / singular_values[-1])
            gram_condition = float(torch.linalg.cond(gram))
            assert design_condition <= cfg.max_weighted_design_condition_number
            assert gram_condition > cfg.max_condition_number
            assert gram_condition == pytest.approx(design_condition**2, rel=1e-10)


def test_exact_gram_path_recovers_arbitrary_frozen_basis_coefficients() -> None:
    cfg = GeometryProjectionConfig(
        validation_states=4,
        geometry_realizations=1,
        max_condition_number=100.0,
    )
    space = PhysicalFunctionSpace(cfg.function_space_config())
    sample = geometry_samples(cfg)["warped_particles"]["low"][0]
    expected = torch.randn(
        4,
        52,
        1,
        generator=torch.Generator().manual_seed(41),
        dtype=torch.float64,
    )
    values = space.decode(expected, sample.coords)

    encoded, design = encode_paths(space, values, sample)

    assert design["rank"] == 52
    torch.testing.assert_close(encoded["exact_gram_projection"], expected, rtol=0.0, atol=1e-10)
    assert not torch.allclose(encoded["moment_only"], expected, rtol=0.0, atol=1e-4)
    assert not torch.allclose(encoded["diagonal_gram"], expected, rtol=0.0, atol=1e-4)


def test_exact_gram_path_is_invariant_to_joint_source_permutation() -> None:
    cfg = GeometryProjectionConfig(
        validation_states=4,
        geometry_realizations=1,
        max_condition_number=100.0,
    )
    space = PhysicalFunctionSpace(cfg.function_space_config())
    sample = geometry_samples(cfg)["uniform_particles"]["low"][0]
    expected = torch.randn(
        4,
        52,
        1,
        generator=torch.Generator().manual_seed(43),
        dtype=torch.float64,
    )
    values = space.decode(expected, sample.coords)
    permutation = torch.randperm(sample.coords.shape[0], generator=torch.Generator().manual_seed(7))

    original, _ = encode_paths(space, values, sample)
    permuted_sample = type(sample)(
        sample.family,
        sample.budget,
        sample.realization,
        sample.seed,
        sample.coords[permutation],
        sample.measure[permutation],
        sample.warp_a,
        sample.warp_b,
    )
    permuted, _ = encode_paths(space, values[:, permutation], permuted_sample)

    torch.testing.assert_close(
        original["exact_gram_projection"],
        permuted["exact_gram_projection"],
        rtol=0.0,
        atol=1e-10,
    )


def test_tiny_e9_run_materializes_zero_shot_decision_and_boundaries(
    tmp_path: Path,
) -> None:
    cfg = GeometryProjectionConfig(
        validation_states=4,
        calibration_resolution=32,
        geometry_realizations=2,
    )

    result = run_geometry_projection(cfg, run_dir=tmp_path)

    assert result["experiment"] == "canonical_latent_e9_geometry_universal_projection"
    assert result["architecture"]["basis_dimension"] == 52
    assert result["evaluation"]["status"] == "skipped_before_state_read"
    assert result["causal_decision"]["classification"] == (
        "geometry_universal_projection_not_qualified"
    )
    assert result["causal_decision"]["gates"]["design"] is False
    assert result["state_split"]["validation_states_read"] == 0
    assert result["state_split"]["training_states_read"] == 0
    assert result["state_split"]["heldout_states_read"] == 0
    assert result["boundary"]["learned_parameters"] == 0
    assert result["boundary"]["optimizer_updates"] == 0
    assert result["boundary"]["operator_instantiated"] is False
    assert result["boundary"]["routing_paths"] == 0
    assert result["boundary"]["particle_dynamics_qualified"] is False
    assert Path(result["result_path"]).is_file()
