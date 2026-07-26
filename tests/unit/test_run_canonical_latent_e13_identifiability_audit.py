import json
import math
from dataclasses import replace

import pytest
import torch

from scripts.run_canonical_latent_e11_coefficient_operator_transfer import (
    TrajectorySet,
    tensor_hash,
)
from scripts.run_canonical_latent_e12_structured_generator import (
    RuleAdapter,
    oracle_generators,
)
from scripts.run_canonical_latent_e13_identifiability_audit import (
    CONTROLS,
    E12_ARTIFACT_PATH,
    E12_LOCK_PATH,
    EXPECTED_E12_ARTIFACT_SHA256,
    EXPECTED_E12_LOCK_SHA256,
    EXPECTED_MASK_HASHES,
    ModeTiedGenerator,
    SupportSparseGenerator,
    _mode_tied_outputs,
    _sha256,
    classify,
    coverage_report,
    frozen_e13_config,
    mask_report,
    parameterization_preflight,
    semantic_masks,
)


def test_e13_replay_inputs_are_hash_locked() -> None:
    lock = json.loads(E12_LOCK_PATH.read_text())

    assert _sha256(E12_LOCK_PATH) == EXPECTED_E12_LOCK_SHA256
    assert _sha256(E12_ARTIFACT_PATH) == EXPECTED_E12_ARTIFACT_SHA256
    assert set(lock["dataset_hashes"]) == {
        "pretrain_x_advection",
        "pretrain_y_advection",
        "pretrain_diffusion",
        "x_advection_validation",
        "y_advection_validation",
        "diffusion_validation",
        "composite_validation",
    }
    assert set(lock["schedule_hashes"]) == {
        "x_advection",
        "y_advection",
        "diffusion",
        "fine_tune",
        "full_control",
    }
    assert len(lock["elementary_checkpoint"]["model_sha256"]) == 64
    assert set(lock["elementary_checkpoint"]["generator_sha256"]) == {
        "A_x",
        "A_y",
        "D",
    }


def test_e13_semantic_masks_are_fixed_before_oracle_values() -> None:
    masks = semantic_masks()
    hashes = {name: tensor_hash(mask.to(torch.uint8)) for name, mask in masks.items()}

    assert hashes == EXPECTED_MASK_HASHES
    assert mask_report()["hashes_match"] is True
    assert int(masks["A_x"].sum()) == 42
    assert int(masks["A_y"].sum()) == 42
    assert int(masks["D"].sum()) == 48
    assert not masks["A_x"][0].any()
    assert not masks["A_y"][:, 0].any()
    assert not masks["D"][0].any()


def test_e13_parameterizations_have_exact_counts_and_structure() -> None:
    sparse = SupportSparseGenerator()
    tied = ModeTiedGenerator()
    report = parameterization_preflight()

    assert sum(parameter.numel() for parameter in sparse.parameters()) == 90
    assert sum(parameter.numel() for parameter in tied.parameters()) == 12
    assert report["expected_parameter_counts"] == {
        "full_skew": 2304,
        "support_sparse": 90,
        "mode_tied": 12,
    }
    assert report["maximum_oracle_phase"] == pytest.approx(0.36 * math.pi)
    assert report["oracle_phase_below_pi"] is True
    assert report["passed"] is True


@pytest.mark.parametrize("model", (SupportSparseGenerator(), ModeTiedGenerator()))
def test_e13_recovery_controls_expose_only_combined_step(model: torch.nn.Module) -> None:
    coefficients = torch.randn(2, 52, 1, dtype=torch.float64)
    parameters = torch.tensor(
        [[0.2, -0.4, 0.03, 0.04], [-0.5, 0.7, 0.06, 0.05]],
        dtype=torch.float64,
    )

    torch.testing.assert_close(
        RuleAdapter(model, "combined")(coefficients, parameters),
        model(coefficients, parameters),
        rtol=0.0,
        atol=0.0,
    )
    with pytest.raises(ValueError, match="combined rule"):
        RuleAdapter(model, "splitting")(coefficients, parameters)


def test_e13_mode_tied_hypothesis_contains_exact_oracle() -> None:
    model = ModeTiedGenerator()
    frequencies = torch.tensor((1.0, 2.0, 3.0), dtype=torch.float64)
    rotations = 2.0 * math.pi * frequencies
    with torch.no_grad():
        model.x_rates.copy_(rotations)
        model.y_rates.copy_(rotations)
        model.x_diffusion_log_rates.copy_(torch.log(rotations.square()))
        model.y_diffusion_log_rates.copy_(torch.log(rotations.square()))

    actual = model.matrices()
    expected = oracle_generators()

    for value, target in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, target, rtol=0.0, atol=5e-13)


def test_e13_support_sparse_hypothesis_contains_exact_oracle() -> None:
    model = SupportSparseGenerator()
    oracle_ax, oracle_ay, oracle_d = oracle_generators()
    x_rates = []
    y_rates = []
    for x_index in (1, 3, 5):
        for y_index in range(7):
            x_rates.append(oracle_ax[7 * x_index + y_index, 7 * (x_index + 1) + y_index])
    for y_index in (1, 3, 5):
        for x_index in range(7):
            y_rates.append(oracle_ay[7 * x_index + y_index, 7 * x_index + y_index + 1])
    with torch.no_grad():
        model.x_rates.copy_(torch.stack(x_rates))
        model.y_rates.copy_(torch.stack(y_rates))
        model.diffusion_log_rates.copy_(torch.log(-torch.diag(oracle_d)[1:]))

    actual = model.matrices()
    expected = (oracle_ax, oracle_ay, oracle_d)

    for value, target in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, target, rtol=0.0, atol=5e-13)


def test_e13_mode_tied_functional_path_retains_vector_derivatives() -> None:
    coefficients = torch.randn(1, 2, 52, 1, dtype=torch.float64)
    coefficients[:, :, 49:] = 0.0
    datasets = {}
    physical = {
        "x_advection": torch.tensor([[0.5, 0.0, 0.0, 0.04]], dtype=torch.float64),
        "y_advection": torch.tensor([[0.0, -0.5, 0.0, 0.04]], dtype=torch.float64),
        "diffusion": torch.tensor([[0.0, 0.0, 0.04, 0.04]], dtype=torch.float64),
    }
    for name, parameters in physical.items():
        datasets[name] = TrajectorySet(name, coefficients.clone(), parameters)
    vector = torch.zeros(12, dtype=torch.float64, requires_grad=True)

    output = _mode_tied_outputs(vector, datasets)
    probe = torch.arange(1, output.numel() + 1, dtype=torch.float64)
    gradient = torch.autograd.grad((output * probe).sum(), vector)[0]

    assert output.numel() == 3 * 49
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) == 12


def _evaluations(*passing: str) -> dict[str, dict[str, bool]]:
    return {
        name: {"recovery_pass": name in passing}
        for name in (
            "full_skew_lbfgs_neutral",
            "full_skew_lbfgs_polish",
            "support_sparse_lbfgs",
            "mode_tied_lbfgs",
        )
    }


@pytest.mark.parametrize(
    ("preflight", "reproduction", "passing", "ranked", "expected"),
    (
        (False, True, (), True, "e13_preflight_failed"),
        (True, False, (), True, "e12_reproduction_failed"),
        (
            True,
            True,
            ("full_skew_lbfgs_neutral",),
            True,
            "full_parameterization_deterministic_recovery_succeeds",
        ),
        (
            True,
            True,
            ("support_sparse_lbfgs",),
            True,
            "support_restriction_required_under_frozen_solvers",
        ),
        (
            True,
            True,
            ("mode_tied_lbfgs",),
            True,
            "mode_tying_required_under_frozen_solvers",
        ),
        (True, True, (), False, "elementary_excitation_rank_deficient"),
        (True, True, (), True, "recovery_controls_not_qualified"),
    ),
)
def test_e13_classification_order_is_literal(
    preflight: bool,
    reproduction: bool,
    passing: tuple[str, ...],
    ranked: bool,
    expected: str,
) -> None:
    excitation = {
        "required_plane_grams_full_rank": ranked,
        "mode_tied_jacobian_full_rank": ranked,
    }

    assert (
        classify(
            preflight_passed=preflight,
            reproduction_passed=reproduction,
            evaluations=_evaluations(*passing),
            excitation=excitation,
        )
        == expected
    )


def test_e13_coverage_requires_literal_cartesian_records() -> None:
    evaluations = {
        control: {"validation": {name: {} for name in ("composite", *("x", "y", "d"))}}
        for control in CONTROLS
    }
    records = [
        {
            "control": control,
            "basis_index": basis,
            "case_name": case,
            "horizon": horizon,
        }
        for control in CONTROLS
        for basis in range(49)
        for case in range(18)
        for horizon in (1, 8)
    ]
    argmax = {
        control: {
            metric: {}
            for metric in (
                "decoded_nrmse",
                "coefficient_nrmse",
                "coefficient_angle_radians",
                "absolute_amplitude_ratio_error",
                "off_target_direction_residual",
            )
        }
        for control in CONTROLS
    }
    recovery_training = {name: {} for name in CONTROLS[:-1]}

    report = coverage_report(evaluations, records, argmax, recovery_training)

    assert report["mode_resolved_records"] == 10584
    assert report["unique_mode_resolved_keys"] == 10584
    assert report["passed"] is True
    records.pop()
    assert coverage_report(evaluations, records, argmax, recovery_training)["passed"] is False


def test_e13_config_rejects_post_registration_solver_drift() -> None:
    cfg = frozen_e13_config()

    with pytest.raises(ValueError, match="deterministic recovery is frozen"):
        replace(cfg, lbfgs_max_iter=251)
