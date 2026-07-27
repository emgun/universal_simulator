from __future__ import annotations

import itertools
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import scripts.run_canonical_latent_e17_quadratic_closure as e17

ROOT = Path(__file__).resolve().parents[2]


def test_config_freezes_budget_and_resolution() -> None:
    cfg = e17.E17Config()
    assert cfg.training_trajectories == 192
    assert cfg.validation_trajectories == 64
    assert cfg.validation_pairs == 32
    assert cfg.truth_resolution == 216
    assert cfg.reference_resolution == 324


def test_mode_order_matches_e7_x_major_y_minor() -> None:
    names = e17.mode_names()
    assert len(names) == 49
    assert names[:8] == (
        "one*one",
        "one*sin1",
        "one*cos1",
        "one*sin2",
        "one*cos2",
        "one*sin3",
        "one*cos3",
        "sin1*one",
    )
    assert names[14] == "cos1*one"
    assert names[-1] == "cos3*cos3"


def test_periodic_mode_frequencies_and_shell_weights() -> None:
    frequencies = e17.periodic_mode_frequencies()
    assert frequencies[0] == (0, 0)
    assert frequencies[1] == (0, 1)
    assert frequencies[7] == (1, 0)
    assert frequencies[48] == (3, 3)
    weights = e17.coefficient_shell_weights()
    assert weights.shape == (48,)
    assert weights[0].item() == 0.5
    assert weights[-1].item() == pytest.approx(1.0 / 19.0)


def test_stratified_uniform_draws_one_value_per_stratum() -> None:
    generator = torch.Generator().manual_seed(101)
    values = e17.stratified_uniform(8, -2.0, 2.0, generator=generator)
    stratum = torch.floor((values + 2.0) / 0.5).to(torch.int64)
    assert sorted(stratum.tolist()) == list(range(8))
    assert torch.all(values >= -2.0)
    assert torch.all(values < 2.0)


def test_small_nonregistered_closure_tail_is_real_normalized_and_unresolved() -> None:
    values, report = e17.closure_tail(seed=7, resolution=24)
    assert values.shape == (24, 24)
    assert report["passed"]
    assert report["half_plane_entries"] > 0
    assert report["rms"] == pytest.approx(0.04, abs=1e-14)
    assert abs(report["mean"]) <= 1e-14
    assert report["active_projection_norm"] <= 1e-12


def _dummy_population_specs() -> tuple[e17.PopulationSpec, e17.ValidationPopulationSpec]:
    training_coefficients = torch.arange(192 * 52, dtype=torch.float64).reshape(192, 52)
    training_parameters = torch.arange(192 * 5, dtype=torch.float64).reshape(192, 5) + 20_000
    training_schedule = torch.arange(192, dtype=torch.int64)
    training = e17.PopulationSpec(
        coefficients=training_coefficients,
        parameters=training_parameters,
        regime_indices=torch.arange(192, dtype=torch.int64) % 4,
        identity_indices=torch.arange(192, dtype=torch.int64),
        identity_keys=tuple(f"training:{index:03d}" for index in range(192)),
        records=(),
        hashes=e17._population_hashes(schedule=training_schedule),
    )
    validation_coefficients = (
        torch.arange(32 * 52, dtype=torch.float64).reshape(32, 52) + 40_000
    ).repeat_interleave(2, dim=0)
    validation_parameters = (
        torch.arange(32 * 5, dtype=torch.float64).reshape(32, 5) + 60_000
    ).repeat_interleave(2, dim=0)
    initial_fields = torch.arange(64 * 4, dtype=torch.float64).reshape(64, 2, 2) + 80_000
    pair_keys = tuple(f"validation_pair:{index:02d}" for index in range(32))
    member_keys = tuple(f"{key}:member:{member}" for key in pair_keys for member in (0, 1))
    validation = e17.ValidationPopulationSpec(
        coefficients=validation_coefficients,
        parameters=validation_parameters,
        initial_fields=initial_fields,
        regime_indices=(torch.arange(32, dtype=torch.int64) % 4).repeat_interleave(2),
        pair_indices=torch.arange(32, dtype=torch.int64).repeat_interleave(2),
        member_indices=torch.tensor((0, 1), dtype=torch.int64).repeat(32),
        pair_identity_keys=pair_keys,
        member_identity_keys=member_keys,
        stress=torch.zeros(64, dtype=torch.bool),
        records=(),
        hashes=e17._population_hashes(pair_schedule=torch.arange(31, -1, -1)),
    )
    return training, validation


def test_population_overlap_report_is_split_qualified_and_fail_closed() -> None:
    training, validation = _dummy_population_specs()
    report = e17.population_overlap_report(training, validation)
    assert report["passed"]
    assert not report["detail"]["identity_overlap"]
    assert not report["detail"]["initial_coefficient_overlap"]
    assert not report["detail"]["parameter_tuple_overlap"]
    validation.coefficients[0] = training.coefficients[0]
    validation.coefficients[1] = training.coefficients[0]
    failed = e17.population_overlap_report(training, validation)
    assert not failed["passed"]
    assert not failed["checks"]["initial_coefficient_disjoint"]


@pytest.mark.parametrize(
    ("resolution", "lower", "upper", "count"),
    ((216, -71, 71, 143), (324, -107, 107, 215)),
)
def test_strict_two_thirds_mask_is_literal(
    resolution: int, lower: int, upper: int, count: int
) -> None:
    retained = e17.retained_wavenumbers(resolution)
    assert sorted(retained) == list(range(lower, upper + 1))
    assert len(retained) == count
    assert int(e17.dealias_mask(resolution).sum().item()) == count * count
    assert resolution // 3 not in retained
    assert -(resolution // 3) not in retained


def test_periodic_projection_appends_zero_trends_and_recovers_coefficients() -> None:
    generator = torch.Generator().manual_seed(17)
    coefficients = torch.randn(3, 49, generator=generator, dtype=torch.float64)
    values = e17.decode_periodic(coefficients, resolution=96)
    recovered = e17.project_periodic(values)
    torch.testing.assert_close(recovered[:, :49], coefficients, atol=1e-12, rtol=1e-12)
    assert torch.equal(recovered[:, 49:], torch.zeros(3, 3, dtype=torch.float64))


def test_spectral_resample_preserves_centered_grid_fourier_field() -> None:
    generator = torch.Generator().manual_seed(18)
    coefficients = torch.randn(2, 49, generator=generator, dtype=torch.float64)
    source = e17.decode_periodic(coefficients, resolution=48)
    resampled = e17.spectral_resample(source, target_resolution=96)
    target = e17.decode_periodic(coefficients, resolution=96)
    torch.testing.assert_close(resampled, target, atol=2e-12, rtol=2e-12)


def test_truth_vector_field_preserves_mean_and_nonlinear_energy() -> None:
    generator = torch.Generator().manual_seed(19)
    coefficients = torch.randn(4, 49, generator=generator, dtype=torch.float64) * 0.1
    values = e17.decode_periodic(coefficients, resolution=48)
    parameters = torch.tensor(
        [
            [0.2, -0.1, 0.0, 0.8, -0.7],
            [-0.2, 0.1, 0.0, -0.8, 0.7],
            [0.0, 0.0, 0.0, 1.0, 1.0],
            [0.1, 0.2, 0.0, -1.0, -1.0],
        ],
        dtype=torch.float64,
    )
    derivative = e17.truth_vector_field(values, parameters)
    assert float(derivative.mean(dim=(-2, -1)).abs().max().item()) <= 1e-12
    residual = (values * derivative).mean(dim=(-2, -1)).abs()
    assert float(residual.max().item()) <= 1e-11


def test_rk4_truth_integrator_shape_and_constant_mode() -> None:
    coefficients, parameters, _ = e17.calibration_cases()
    initial = e17.decode_periodic(coefficients[:2], resolution=48)
    trajectory = e17.integrate_truth(
        initial,
        parameters[:2],
        internal_step=0.002,
        observation_step=0.01,
        transitions=2,
    )
    assert trajectory.shape == (2, 3, 48, 48)
    projected = e17.project_periodic(trajectory)
    torch.testing.assert_close(
        projected[:, :, 0],
        projected[:, :1, 0].expand(-1, 3),
        atol=1e-12,
        rtol=0.0,
    )


@pytest.mark.parametrize("axis", ("x", "y"))
def test_triad_support_and_energy_identity(axis: str) -> None:
    support = e17.triad_coefficients(axis)  # type: ignore[arg-type]
    assert len(support) == 1329
    assert all(output != 0 for output, _, _, _ in support)
    generator = torch.Generator().manual_seed(29)
    coefficients = torch.randn(32, 49, generator=generator, dtype=torch.float64)
    action = e17.apply_sparse_quadratic(coefficients, support)
    residual = torch.einsum("bi,bi->b", coefficients, action)
    assert float(residual.abs().max().item()) <= 1e-10


def test_predecessor_and_sealed_e15_hashes_pass() -> None:
    assert e17.predecessor_report(ROOT)["passed"]
    report = e17.sealed_e15_report(ROOT)
    assert report["passed"]
    assert report["model_sha256"] == e17.E15_MODEL_SHA256
    assert report["generator_sha256"] == e17.E15_GENERATOR_SHA256
    assert len(report["gates"]) == 8


def test_canonical_tensor_record_is_explicit_little_endian() -> None:
    tensor = torch.tensor([[1.0, -2.0], [3.5, 4.25]], dtype=torch.float64)
    record = e17.canonical_tensor_record(tensor)
    assert record == {
        "shape": [2, 2],
        "dtype": "<f8",
        "order": "C",
        "bytes": 32,
        "sha256": "a5a7aeb470a28c49a0ecc1642c552df3c52c5d502df1eba72f217481e183937f",
    }


def test_classification_precedence_exhaustive() -> None:
    names = (
        "preflight_passed",
        "complete",
        "representation_passed",
        "candidate_passed",
        "coverage_passed",
        "finite",
        "boundary_passed",
    )
    for values in itertools.product((False, True), repeat=len(names)):
        state = dict(zip(names, values, strict=True))
        result = e17.classify(**state)
        if not state["preflight_passed"]:
            assert result == "preflight_failed"
        elif not (
            state["complete"]
            and state["coverage_passed"]
            and state["finite"]
            and state["boundary_passed"]
        ):
            assert result == "incomplete"
        elif not state["representation_passed"]:
            assert result == "latent_closure_insufficient"
        elif not state["candidate_passed"]:
            assert result == "quadratic_identification_failed"
        else:
            assert result == "constrained_quadratic_closure_qualified"


def test_prestate_report_reads_no_scientific_state() -> None:
    report = e17.prestate_report(ROOT, e17.E17Config())
    assert report["passed"]
    assert report["state_reads"] == {
        "e15_predecessor": 0,
        "training": 0,
        "validation": 0,
        "heldout": 0,
    }


def test_runtime_is_frozen_in_fresh_process() -> None:
    command = (
        "import json; "
        "import scripts.run_canonical_latent_e17_quadratic_closure as e; "
        "print(json.dumps(e.configure_runtime(), sort_keys=True))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", command],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == {
        "interop_threads": 1,
        "intraop_threads": 1,
        "passed": True,
    }


def test_calibration_fails_before_integration_on_bad_source_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        e17,
        "source_state",
        lambda _root: {
            "head": "bad",
            "clean": False,
            "sources": {},
            "sources_match_head": False,
        },
    )
    monkeypatch.setattr(
        e17,
        "prestate_report",
        lambda _root, _cfg: {
            "passed": True,
            "state_reads": {"training": 0, "validation": 0, "heldout": 0},
        },
    )

    def forbidden(_cfg: e17.E17Config) -> dict[str, object]:
        raise AssertionError("calibration must not run after source preflight failure")

    monkeypatch.setattr(e17, "convergence_calibration", forbidden)
    report = e17.calibration_run_report(
        ROOT,
        e17.E17Config(),
        runtime={"intraop_threads": 1, "interop_threads": 1, "passed": True},
    )
    assert not report["passed"]
    assert report["calibration"] is None
    assert report["state_reads"] == {"training": 0, "validation": 0, "heldout": 0}
