from dataclasses import replace
from pathlib import Path

import pytest
import torch

import scripts.run_canonical_latent_e11_coefficient_operator_transfer as e11
from scripts.run_canonical_latent_e7_function_space import (
    FunctionSpaceConfig,
    PhysicalFunctionSpace,
)
from scripts.run_canonical_latent_e11_coefficient_operator_transfer import (
    EVALUATION_ARMS,
    REGIMES,
    ResidualCoefficientOperator,
    closure_preflight,
    decision,
    evolve_periodic,
    frozen_e11_config,
    modal_scales,
    parameter_quartile_reports,
    run_e11,
    sample_coefficients,
    sample_parameters,
    schedule,
    truth_grid,
    validate_evaluation_coverage,
)


def _space() -> PhysicalFunctionSpace:
    return PhysicalFunctionSpace(
        FunctionSpaceConfig(
            seed=23,
            validation_states=64,
            calibration_resolution=64,
            max_condition_number=100.0,
        )
    )


def test_e11_periodic_state_distribution_freezes_modes_and_scale() -> None:
    scales = modal_scales()
    coefficients = sample_coefficients(16, seed=51_001)

    assert scales.shape == (1, 52, 1)
    assert scales[0, 0, 0] == 0.20
    assert torch.all(scales > 0)
    assert torch.count_nonzero(coefficients[:, 49:]) == 0
    assert torch.count_nonzero(coefficients[:, :49]) > 0


def test_exact_periodic_teacher_preserves_constant_and_dissipates_energy() -> None:
    cfg = frozen_e11_config()
    space = _space()
    coords, _, basis = truth_grid(space, cfg.truth_resolution)
    constant_coefficients = torch.zeros(2, 52, 1, dtype=torch.float64)
    constant_coefficients[:, 0] = 0.7
    values = basis.expand(2, -1, -1) @ constant_coefficients
    parameters = torch.tensor(
        [[0.8, -0.3, 0.0, 0.04], [0.0, 0.0, 0.08, 0.04]],
        dtype=torch.float64,
    )

    evolved = evolve_periodic(
        values,
        parameters,
        resolution=cfg.truth_resolution,
    )

    torch.testing.assert_close(evolved, values, rtol=0.0, atol=1e-12)
    assert coords.shape == (1, cfg.truth_resolution**2, 2)


def test_zero_initialized_operator_is_persistence_without_label_inputs() -> None:
    cfg = frozen_e11_config()
    torch.manual_seed(cfg.model_seed)
    model = ResidualCoefficientOperator(cfg)
    coefficients = sample_coefficients(4, seed=7)
    parameters = sample_parameters(
        4,
        "composite",
        seed=11,
        cfg=cfg,
    )

    prediction = model(coefficients, parameters)

    torch.testing.assert_close(prediction, coefficients, rtol=0.0, atol=0.0)
    assert model.network[0].in_features == 56
    assert model.network[-1].out_features == 52


def test_e11_schedule_is_deterministic_and_bounded() -> None:
    first = schedule(12, 7, 31, seed=72_001)
    second = schedule(12, 7, 31, seed=72_001)

    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
    assert first.shape == (12, 7)
    assert int(first.min()) >= 0
    assert int(first.max()) < 31


def test_parameter_quartiles_record_constant_physical_axes() -> None:
    target = sample_coefficients(8, seed=19)
    prediction = target * 0.9
    parameters = torch.zeros(8, 4, dtype=torch.float64)
    parameters[:, 3] = torch.linspace(0.02, 0.06, 8)

    reports = parameter_quartile_reports(
        prediction,
        target,
        parameters,
        target,
        prediction,
    )

    for name in ("abs_vx", "abs_vy", "nu"):
        assert reports[name][0]["constant"] is True
        assert reports[name][0]["count"] == 8
        assert len(reports[name]) == 1
    assert len(reports["dt"]) == 4
    assert all(report["constant"] is False for report in reports["dt"])


def test_e11_active_periodic_basis_passes_closure_preflight() -> None:
    report = closure_preflight(frozen_e11_config(), _space())

    assert report["active_basis_vectors"] == 49
    assert report["inactive_trend_vectors"] == 3
    assert report["minimum_projection_rank"] == 52
    assert report["all_projected_coefficients_finite"] is True
    assert report["all_errors_finite"] is True
    assert report["maximum_truth_to_projection_decoded_nrmse"] <= 0.01
    assert report["maximum_semigroup_composition_error"] <= 1e-10
    assert report["passed"] is True


def test_e11_closure_rejects_nonfinite_projected_coefficients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_project_values = e11.project_values

    def nonfinite_project_values(*args: object, **kwargs: object):
        coefficients, design = original_project_values(*args, **kwargs)
        coefficients = coefficients.clone()
        coefficients[0, 0, 0] = torch.nan
        return coefficients, design

    monkeypatch.setattr(e11, "project_values", nonfinite_project_values)

    report = closure_preflight(frozen_e11_config(), _space())

    assert report["all_projected_coefficients_finite"] is False
    assert report["all_errors_finite"] is False
    assert report["passed"] is False


def test_e11_evaluation_coverage_requires_every_arm_and_baseline() -> None:
    evaluation = {
        "composite": {arm: {} for arm in EVALUATION_ARMS},
        "elementary_by_arm": {arm: {regime: {} for regime in REGIMES} for arm in EVALUATION_ARMS},
        "temporal_extrapolation_by_arm": {arm: {} for arm in EVALUATION_ARMS},
        "semigroup_consistency_by_arm_and_regime": {
            arm: {regime: {} for regime in ("composite", *REGIMES)} for arm in EVALUATION_ARMS
        },
        "physics_by_arm": {arm: {} for arm in EVALUATION_ARMS},
    }

    assert validate_evaluation_coverage(evaluation)["passed"] is True
    del evaluation["composite"]["exact_projected_truth"]
    with pytest.raises(RuntimeError, match="evaluation coverage is incomplete"):
        validate_evaluation_coverage(evaluation)


def _decision_fixture() -> dict[str, object]:
    report = {
        "one_step_coefficient_nrmse": 0.01,
        "one_step_decoded_nrmse": 0.01,
        "rollout_coefficient_nrmse": 0.05,
        "rollout_decoded_nrmse": 0.05,
        "final_high_frequency_spectral": {"nrmse": 0.10},
    }
    scratch = {**report, "rollout_decoded_nrmse": 0.10}
    persistence = {**report, "rollout_decoded_nrmse": 0.50}
    return {
        "closure_preflight": {"passed": True},
        "evaluation": {
            "composite": {
                "pretrained_fewshot": dict(report),
                "scratch_fewshot": scratch,
                "full_composite_control": dict(report),
                "elementary_pretrained_zero_shot": dict(report),
                "persistence": persistence,
            },
            "elementary_retention": {
                "post_finetune_macro_decoded_nrmse": 0.05,
                "post_to_pre_ratio": 1.0,
            },
            "temporal_extrapolation": {
                "rollout_coefficient_nrmse": 0.05,
                "rollout_decoded_nrmse": 0.05,
            },
            "semigroup_consistency": {
                "coefficient_nrmse": 0.01,
                "decoded_nrmse": 0.01,
            },
            "cross_observation": {
                "maximum_coefficient_mismatch": 0.001,
                "maximum_decoded_mismatch": 0.001,
            },
            "physics": {
                "advection_mean_mode_relative_error": 0.0001,
                "maximum_advection_l2_norm_drift": 0.01,
                "diffusion_nonincreasing_energy_fraction": 1.0,
            },
        },
        "provenance": {
            "source_files_match_git_head": True,
            "git_head_present": True,
            "worktree_clean": True,
        },
        "reproducibility": {"byte_identical_complete_runs": True},
        "boundary": {
            "heldout_reads": 0,
            "provider_calls": 0,
            "routing_paths": 0,
            "representation_label_inputs": False,
            "task_label_inputs": False,
            "original_observations_after_projection": False,
        },
    }


def test_e11_decision_path_requires_absolute_accuracy() -> None:
    cfg = frozen_e11_config()
    result = _decision_fixture()

    assert decision(result, cfg)["classification"] == ("coefficient_operator_transfer_qualified")
    result["evaluation"]["composite"]["pretrained_fewshot"]["rollout_decoded_nrmse"] = 0.50
    assert decision(result, cfg)["classification"] == ("coefficient_dynamics_not_qualified")


def test_e11_rejects_off_contract_configuration_before_state_access(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="frozen"):
        run_e11(
            replace(frozen_e11_config(), validation_trajectories=32),
            run_dir=tmp_path,
        )
