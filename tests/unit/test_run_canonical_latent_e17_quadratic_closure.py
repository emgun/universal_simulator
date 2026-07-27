from __future__ import annotations

import itertools
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
    assert cfg.truth_resolution == 96
    assert cfg.reference_resolution == 144


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


@pytest.mark.parametrize(
    ("resolution", "lower", "upper", "count"),
    ((96, -31, 31, 63), (144, -47, 47, 95)),
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
