import hashlib
import itertools
import json
from pathlib import Path

import pytest
import torch

import scripts.run_canonical_latent_e16_multi_realization_robustness as e16
from scripts.run_canonical_latent_e11_coefficient_operator_transfer import TrajectorySet
from scripts.run_canonical_latent_e15_training_package import (
    _gradient_vector,
    grouped_loss,
    literal_outputs,
    loss_from_outputs,
    source_probe,
)
from scripts.run_canonical_latent_e16_multi_realization_robustness import (
    EXPECTED_E15_CLASSIFICATION,
    PACKAGES,
    PER_REPLICA_TIMEOUT_SECONDS,
    RAW_MEMBERS,
    REALIZATIONS,
    WHOLE_EXPERIMENT_TIMEOUT_SECONDS,
    build_outputs,
    canonical_tensor_record,
    classify,
    deterministic_bundle,
    elementary_schedules,
    frozen_e15_config,
    occurrence_count_report,
    publish_atomic,
    sealed_e15_report,
    stability_from_recovery,
    verify_bundle,
    verify_outputs,
)


def _tiny_dataset() -> TrajectorySet:
    coefficients = torch.arange(3 * 4 * 52, dtype=torch.float64).reshape(3, 4, 52, 1)
    coefficients = coefficients / 1000.0
    parameters = torch.tensor(
        [
            [0.2, 0.0, 0.0, 0.03],
            [0.0, -0.4, 0.0, 0.04],
            [0.0, 0.0, 0.05, 0.02],
        ],
        dtype=torch.float64,
    )
    return TrajectorySet("tiny", coefficients, parameters)


def test_e16_registered_schedule_seeds_are_literal_and_complete() -> None:
    assert REALIZATIONS["r1"]["literal_schedule_seeds"] == {
        "x_advection": 172001,
        "y_advection": 172002,
        "diffusion": 172003,
    }
    assert REALIZATIONS["r2"]["literal_schedule_seeds"] == {
        "x_advection": 272001,
        "y_advection": 272002,
        "diffusion": 272003,
    }
    cfg = e16.realization_config(frozen_e15_config(), REALIZATIONS["r1"])
    generated = elementary_schedules(cfg)
    assert tuple(generated) == ("x_advection", "y_advection", "diffusion")
    assert all(value.shape == (1500, 32) for value in generated.values())


def test_e16_canonical_tensor_serialization_is_little_endian_c_order() -> None:
    value = torch.tensor([[1.25, -2.5], [3.75, 4.0]], dtype=torch.float64).T
    record = canonical_tensor_record(value, kind="float64")
    expected = value.cpu().contiguous().numpy().astype("<f8", copy=True).tobytes(order="C")

    assert record == {
        "shape": [2, 2],
        "dtype": "torch.float64",
        "c_contiguous": True,
        "serialization": "little_endian_float64_c_order",
        "bytes": 32,
        "sha256": hashlib.sha256(expected).hexdigest(),
    }


def test_e16_occurrence_counts_require_complete_48000_draw_population() -> None:
    flattened = torch.arange(48_000, dtype=torch.int64) % 2048
    schedules = {
        name: flattened.reshape(1500, 32).clone()
        for name in ("x_advection", "y_advection", "diffusion")
    }
    counts, report = occurrence_count_report(schedules)

    assert report["passed"] is True
    assert all(value.shape == (2048,) for value in counts.values())
    assert all(record["total"] == 48_000 for record in report["records"].values())
    assert all(record["minimum"] >= 1 for record in report["records"].values())


def test_e16_grouped_weighted_objective_matches_literal_outputs_and_gradients() -> None:
    dataset = _tiny_dataset()
    grouped_model = source_probe()
    literal_model = source_probe()
    inputs, targets, parameters = dataset.transitions
    weights = torch.tensor([1, 3, 2, 4, 1, 5, 2, 1, 3], dtype=torch.float64)

    grouped = grouped_loss(grouped_model, dataset, weights, chunk_trajectories=2)
    literal_prediction = literal_outputs(literal_model, inputs, parameters)
    literal = loss_from_outputs(literal_prediction, targets, weights)

    torch.testing.assert_close(grouped, literal, rtol=1e-14, atol=1e-14)
    torch.testing.assert_close(
        _gradient_vector(grouped, grouped_model),
        _gradient_vector(literal, literal_model),
        rtol=1e-14,
        atol=1e-14,
    )


@pytest.mark.parametrize("bits", range(16))
def test_e16_all_fresh_recovery_vectors_have_one_complete_classification(bits: int) -> None:
    names = (
        ("r1", "deterministic_adamw_restart"),
        ("r1", "componentwise_lbfgs_neutral"),
        ("r2", "deterministic_adamw_restart"),
        ("r2", "componentwise_lbfgs_neutral"),
    )
    fresh = {realization: {package: False for package in PACKAGES} for realization in ("r1", "r2")}
    for index, (realization, package) in enumerate(names):
        fresh[realization][package] = bool(bits & (1 << index))
    vector, stability = stability_from_recovery(
        {package: True for package in PACKAGES},
        fresh,
    )

    adam = fresh["r1"]["deterministic_adamw_restart"] and fresh["r2"][
        "deterministic_adamw_restart"
    ]
    lbfgs = fresh["r1"]["componentwise_lbfgs_neutral"] and fresh["r2"][
        "componentwise_lbfgs_neutral"
    ]
    if adam and lbfgs:
        expected = "both_practical_recovery_packages_stable"
    elif lbfgs:
        expected = "componentwise_lbfgs_neutral_stable_only"
    elif adam:
        expected = "deterministic_adamw_restart_stable_only"
    else:
        expected = "no_practical_recovery_package_stable"

    assert all(vector[package]["r0_sealed_e15"] for package in PACKAGES)
    assert classify(
        preflight_passed=True,
        execution_complete=True,
        stability=stability,
    ) == expected


@pytest.mark.parametrize(
    ("preflight", "complete", "expected"),
    (
        (False, False, "e16_preflight_failed"),
        (False, True, "e16_preflight_failed"),
        (True, False, "e16_execution_incomplete"),
    ),
)
@pytest.mark.parametrize("adam,lbfgs", itertools.product((False, True), repeat=2))
def test_e16_early_failure_precedence_overrides_every_scientific_pattern(
    preflight: bool,
    complete: bool,
    expected: str,
    adam: bool,
    lbfgs: bool,
) -> None:
    assert (
        classify(
            preflight_passed=preflight,
            execution_complete=complete,
            stability={
                "deterministic_adamw_restart": adam,
                "componentwise_lbfgs_neutral": lbfgs,
            },
        )
        == expected
    )


def test_e16_sealed_e15_object_and_gate_mapping_is_exact() -> None:
    report = sealed_e15_report()

    assert report["passed"] is True
    assert report["classification"] == EXPECTED_E15_CLASSIFICATION
    assert all(report["recovery"].values())
    assert all(len(gates) == 8 and all(gates.values()) for gates in report["recovery_gates"].values())


def test_e16_live_inherited_sources_equal_sealed_e15_hashes() -> None:
    paths = {
        "e15_contract": e16.E15_CONTRACT_PATH,
        "e15_runner": e16.E15_RUNNER_PATH,
        "e15_tests": e16.E15_TEST_PATH,
        "e13_runner": e16.E13_RUNNER_PATH,
        "e12_runner": e16.E12_RUNNER_PATH,
        "e11_runner": e16.E11_RUNNER_PATH,
        "e7_runner": e16.E7_RUNNER_PATH,
        "latent_evaluation": e16.LATENT_EVAL_PATH,
    }

    assert {
        name: e16.sha256_path(path) for name, path in paths.items()
    } == e16.EXPECTED_LIVE_INHERITED_HASHES


def _minimal_complete() -> dict:
    return {
        "schema_version": 1,
        "experiment": "canonical_latent_e16_multi_realization_robustness",
        "classification": "both_practical_recovery_packages_stable",
        "config": {},
        "config_sha256": "0" * 64,
        "registered_realizations": REALIZATIONS,
        "provenance": {
            "git_head": "1" * 40,
            "source_files": {},
            "sealed_e15": {"passed": True},
        },
        "oracle_preflight": {"passed": True},
        "realizations": {},
        "recovery_vector": {},
        "stability": {package: True for package in PACKAGES},
        "nonlinear_expansion_authorized": True,
        "state_reads": {"training": 1536, "validation": 256, "heldout": 0},
        "boundary": {"heldout_reads": 0},
        "reproducibility": {"passed": True},
        "replication": {"raw_replicates_byte_identical": True},
    }


def test_e16_bundle_and_atomic_publication_round_trip(tmp_path: Path) -> None:
    first = {"value": 1}
    raw = {name: json.dumps(first, sort_keys=True).encode() for name in RAW_MEMBERS}
    bundle_a, members_a = deterministic_bundle(raw)
    bundle_b, members_b = deterministic_bundle(raw)

    assert bundle_a == bundle_b
    assert members_a == members_b
    verify_bundle(bundle_a, members_a)

    complete = _minimal_complete()
    replica = dict(complete)
    replica.pop("replication")
    outputs = build_outputs(replica, replica, complete)
    verify_outputs(outputs)
    destination = tmp_path / "e16-evidence"
    publish_atomic(destination, outputs)

    assert sorted(path.name for path in destination.iterdir()) == sorted(outputs)
    assert {path.name: path.read_bytes() for path in destination.iterdir()} == outputs


def test_e16_timeout_budget_is_static_and_preregistered() -> None:
    assert PER_REPLICA_TIMEOUT_SECONDS == 28_800
    assert WHOLE_EXPERIMENT_TIMEOUT_SECONDS == 64_800
