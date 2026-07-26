from pathlib import Path

import pytest
import torch

import scripts.run_canonical_latent_e12_structured_generator as e12
from scripts.run_canonical_latent_e7_function_space import (
    FunctionSpaceConfig,
    PhysicalFunctionSpace,
)
from scripts.run_canonical_latent_e9_geometry_universal_projection import (
    GeometryProjectionConfig,
    geometry_samples,
)
from scripts.run_canonical_latent_e11_coefficient_operator_transfer import (
    canonical_grid,
    sample_coefficients,
    sample_parameters,
    schedule,
)
from scripts.run_canonical_latent_e12_structured_generator import (
    ALL_REGIMES,
    EVALUATION_ARMS,
    EXPECTED_SCHEDULES,
    GEOMETRY_FAMILIES,
    GEOMETRY_REALIZATIONS_BY_FAMILY,
    LEARNED_CHECKPOINTS,
    FixedGenerator,
    RuleAdapter,
    StructuredGenerator,
    frozen_e12_config,
    generator_identification,
    oracle_generators,
    oracle_preflight,
    structure_report,
    validate_evaluation_coverage,
    validate_schedules,
)


def _space() -> PhysicalFunctionSpace:
    return PhysicalFunctionSpace(
        FunctionSpaceConfig(
            seed=23,
            validation_states=64,
            canonical_query_resolution=18,
            calibration_resolution=64,
            max_condition_number=100.0,
        )
    )


def test_e12_structured_generator_has_exact_parameterization() -> None:
    model = StructuredGenerator()
    report = structure_report(model)

    assert sum(parameter.numel() for parameter in model.parameters()) == 2304
    assert report["parameter_count"] == 2304
    assert report["ax_skew_residual"] == 0.0
    assert report["ay_skew_residual"] == 0.0
    assert report["ax_constant_residual"] == 0.0
    assert report["ay_constant_residual"] == 0.0
    assert report["diffusion_off_diagonal_residual"] == 0.0
    assert report["diffusion_maximum_eigenvalue"] <= 0.0


def test_e12_initial_generator_preserves_advection_and_dissipates_diffusion() -> None:
    cfg = frozen_e12_config()
    coefficients = sample_coefficients(4, seed=7)
    advection = sample_parameters(4, "x_advection", seed=11, cfg=cfg)
    diffusion = sample_parameters(4, "diffusion", seed=13, cfg=cfg)
    model = StructuredGenerator()

    advection_prediction = model(coefficients, advection)
    diffusion_prediction = model(coefficients, diffusion)

    torch.testing.assert_close(advection_prediction, coefficients, rtol=0.0, atol=0.0)
    assert torch.all(
        diffusion_prediction[:, :49].square().sum(dim=(1, 2))
        <= coefficients[:, :49].square().sum(dim=(1, 2)) + 1e-12
    )
    torch.testing.assert_close(
        diffusion_prediction[:, 49:],
        coefficients[:, 49:],
        rtol=0.0,
        atol=0.0,
    )


def test_e12_oracle_passes_basis_and_semigroup_preflight() -> None:
    cfg = frozen_e12_config()
    oracle = FixedGenerator(*oracle_generators())

    report = oracle_preflight(cfg, _space(), oracle)

    assert report["active_basis_vectors"] == 49
    assert report["parameter_cases"] == 18
    assert report["e11_projection_closure"]["minimum_projection_rank"] == 52
    assert report["maximum_one_step_decoded_nrmse"] <= 1e-10
    assert report["maximum_eight_step_decoded_nrmse"] <= 1e-10
    assert report["maximum_combined_splitting_decoded_nrmse"] <= 1e-10
    assert report["maximum_combined_semigroup_coefficient_nrmse"] <= 1e-10
    assert report["passed"] is True


def test_e12_oracle_identification_is_numerical_zero() -> None:
    space = _space()
    canonical_coords, _ = canonical_grid(space, 18)
    oracle = FixedGenerator(*oracle_generators())

    report = generator_identification(
        oracle,
        oracle,
        space=space,
        canonical_coords=canonical_coords,
    )

    assert max(report["relative_frobenius"].values()) <= 1e-12
    assert max(report["maximum_supported_entry_relative_error"].values()) <= 1e-12
    assert max(report["off_support_leakage"].values()) <= 1e-12
    assert report["maximum_diffusion_rate_relative_error"] <= 1e-12
    assert max(report["normalized_commutators"].values()) <= 1e-12
    assert report["maximum_basis_action_decoded_nrmse"] <= 1e-12


def test_e12_basis_action_maximum_catches_one_bad_mode() -> None:
    space = _space()
    canonical_coords, _ = canonical_grid(space, 18)
    oracle = FixedGenerator(*oracle_generators())
    ax, ay, diffusion = oracle_generators()
    support = torch.nonzero(ax != 0, as_tuple=False)
    row, column = (int(value) for value in support[0])
    ax[row, column] *= 8.0
    ax[column, row] *= 8.0
    perturbed = FixedGenerator(ax, ay, diffusion)

    report = generator_identification(
        perturbed,
        oracle,
        space=space,
        canonical_coords=canonical_coords,
    )

    assert report["maximum_basis_action_decoded_nrmse"] > 0.5


def test_e12_combined_and_splitting_oracles_are_checkpoint_identical() -> None:
    cfg = frozen_e12_config()
    oracle = FixedGenerator(*oracle_generators())
    coefficients = sample_coefficients(8, seed=17)
    parameters = sample_parameters(8, "composite", seed=19, cfg=cfg)

    combined = RuleAdapter(oracle, "combined")(coefficients, parameters)
    splitting = RuleAdapter(oracle, "splitting")(coefficients, parameters)

    torch.testing.assert_close(combined, splitting, rtol=0.0, atol=2e-12)
    assert not list(oracle.parameters())


def test_e12_schedule_hashes_are_literal_e11_schedules() -> None:
    cfg = frozen_e12_config()
    schedules = {
        regime: schedule(
            cfg.pretrain_updates,
            cfg.pretrain_batch_per_regime,
            cfg.pretrain_trajectories_per_regime * cfg.rollout_steps,
            seed=cfg.schedule_seed + index,
        )
        for index, regime in enumerate(("x_advection", "y_advection", "diffusion"))
    }
    schedules["fine_tune"] = schedule(
        cfg.fine_tune_updates,
        cfg.fine_tune_batch_size,
        cfg.fewshot_trajectories * cfg.rollout_steps,
        seed=cfg.schedule_seed + 10,
    )
    schedules["full_control"] = schedule(
        cfg.full_control_updates,
        cfg.full_control_batch_size,
        cfg.full_control_trajectories * cfg.rollout_steps,
        seed=cfg.schedule_seed + 20,
    )

    report = validate_schedules(schedules)

    assert report["passed"] is True
    assert set(report["records"]) == set(EXPECTED_SCHEDULES)


def test_e12_cross_observation_realization_counts_match_e10_geometry() -> None:
    samples = geometry_samples(
        GeometryProjectionConfig(
            seed=23,
            validation_states=64,
            geometry_realizations=4,
            geometry_seed_start=40_000,
            max_condition_number=100.0,
            max_weighted_design_condition_number=10.0,
        )
    )

    assert {
        family: len(budgets["high"]) for family, budgets in samples.items()
    } == GEOMETRY_REALIZATIONS_BY_FAMILY


def _coverage_fixture() -> dict[str, object]:
    family_pairs = {
        f"{left}__vs__{right}": {
            "realization_pairs": GEOMETRY_REALIZATIONS_BY_FAMILY[left]
            * GEOMETRY_REALIZATIONS_BY_FAMILY[right]
        }
        for left_index, left in enumerate(GEOMETRY_FAMILIES)
        for right in GEOMETRY_FAMILIES[left_index + 1 :]
    }
    evaluation: dict[str, object] = {
        "base": {arm: {regime: {} for regime in ALL_REGIMES} for arm in EVALUATION_ARMS},
        "temporal_extrapolation": {
            arm: {regime: {} for regime in ALL_REGIMES} for arm in EVALUATION_ARMS
        },
        "semigroup": {arm: {regime: {} for regime in ALL_REGIMES} for arm in EVALUATION_ARMS},
        "physics": {
            arm: {
                "by_regime": {regime: {} for regime in ("x_advection", "y_advection", "diffusion")}
            }
            for arm in EVALUATION_ARMS
        },
        "cross_observation": {
            rule: {"geometry_realizations": 4, "pairs": family_pairs}
            for rule in ("combined", "splitting")
        },
        "composition_gap": {
            checkpoint: {regime: {} for regime in ALL_REGIMES} for checkpoint in LEARNED_CHECKPOINTS
        },
    }
    return evaluation


def test_e12_coverage_requires_every_literal_cartesian_cell() -> None:
    evaluation = _coverage_fixture()

    report = validate_evaluation_coverage(evaluation)  # type: ignore[arg-type]

    assert report == {
        "base_cells": 48,
        "temporal_extrapolation_cells": 48,
        "semigroup_cells": 48,
        "physics_cells": 36,
        "cross_observation_rules": 2,
        "composition_gap_cells": 16,
        "passed": True,
    }
    del evaluation["base"]["oracle_splitting"]["diffusion"]  # type: ignore[index]
    with pytest.raises(RuntimeError, match="evaluation coverage is incomplete"):
        validate_evaluation_coverage(evaluation)  # type: ignore[arg-type]


def test_e12_runner_refuses_uncommitted_source_before_state_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        e12,
        "provenance",
        lambda: {
            "source_files_match_git_head": False,
            "git_head_present": True,
            "worktree_clean": False,
            "e11_artifact_hash_matches": True,
            "e11_comparators_match": True,
        },
    )

    with pytest.raises(RuntimeError, match="clean committed Git HEAD"):
        e12.run_e12(frozen_e12_config(), run_dir=tmp_path)


def test_e12_closure_failure_stops_before_sampled_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        e12,
        "provenance",
        lambda: {
            "source_files_match_git_head": True,
            "git_head_present": True,
            "worktree_clean": True,
            "e11_artifact_hash_matches": True,
            "e11_comparators_match": True,
        },
    )
    monkeypatch.setattr(
        e12,
        "closure_preflight",
        lambda *_args, **_kwargs: {
            "passed": False,
            "minimum_projection_rank": 52,
            "all_projected_coefficients_finite": True,
        },
    )

    def forbidden_state_access(*_args: object, **_kwargs: object) -> None:
        pytest.fail("sampled state was accessed after closure failure")

    monkeypatch.setattr(e12, "build_trajectories", forbidden_state_access)

    result = e12._run_once(frozen_e12_config(), run_dir=tmp_path)

    assert result["preflight"]["passed"] is False
    assert result["state_reads"] == {"training": 0, "validation": 0, "heldout": 0}
    assert result["optimizer_updates"] == 0
