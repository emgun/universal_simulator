import io
import json
import tarfile
from pathlib import Path

import pytest
import torch

import scripts.run_canonical_latent_e15_training_package as e15
from scripts.run_canonical_latent_e11_coefficient_operator_transfer import (
    TrajectorySet,
    normalized_loss,
)
from scripts.run_canonical_latent_e12_structured_generator import (
    FixedGenerator,
    StructuredGenerator,
    oracle_generators,
)
from scripts.run_canonical_latent_e13_identifiability_audit import schedules
from scripts.run_canonical_latent_e15_training_package import (
    CANONICAL_OUTPUT_DIR,
    EXPECTED_COUNT_RECORDS,
    OUTPUT_NAMES,
    PER_REPLICA_TIMEOUT_SECONDS,
    RAW_MEMBERS,
    TRACE_STEPS,
    VALIDATION_CASE_NAMES,
    WHOLE_EXPERIMENT_TIMEOUT_SECONDS,
    E15IncompleteError,
    _difference,
    _gradient_vector,
    adamw_state_record,
    build_outputs,
    canonical_bytes,
    classify,
    deterministic_bundle,
    frozen_e15_config,
    frozen_validation_trace_scope,
    grouped_loss,
    grouped_outputs,
    incomplete_status,
    literal_outputs,
    loss_from_outputs,
    occurrence_counts,
    pretty_bytes,
    publish_atomic,
    run_e15,
    sealed_ceiling_integrity,
    source_probe,
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


def test_e15_schedule_counts_are_frozen_before_state_access() -> None:
    cfg = frozen_e15_config()
    count_tensors, report = occurrence_counts(schedules(cfg))

    assert report["passed"] is True
    assert set(report["records"]) == set(EXPECTED_COUNT_RECORDS)
    assert all(value.shape == (2048,) for value in count_tensors.values())
    assert [report["records"][name]["sha256"] for name in EXPECTED_COUNT_RECORDS] == [
        EXPECTED_COUNT_RECORDS[name]["sha256"] for name in EXPECTED_COUNT_RECORDS
    ]


def test_e15_grouped_outputs_equal_literal_outputs_and_gradients() -> None:
    dataset = _tiny_dataset()
    grouped_model = source_probe()
    literal_model = source_probe()
    inputs, targets, parameters = dataset.transitions
    weights = torch.tensor([1, 3, 2, 4, 1, 5, 2, 1, 3], dtype=torch.float64)

    grouped = grouped_outputs(grouped_model, dataset, chunk_trajectories=2)
    literal = literal_model(inputs, parameters)
    grouped_objective = grouped_loss(grouped_model, dataset, weights, chunk_trajectories=2)
    literal_loss = loss_from_outputs(literal, targets, weights)

    torch.testing.assert_close(grouped, literal, rtol=1e-14, atol=1e-14)
    torch.testing.assert_close(grouped_objective, literal_loss, rtol=1e-14, atol=1e-14)
    torch.testing.assert_close(
        _gradient_vector(grouped_objective, grouped_model),
        _gradient_vector(literal_loss, literal_model),
        rtol=1e-14,
        atol=1e-14,
    )


def test_e15_all_ones_loss_is_exact_e13_objective() -> None:
    dataset = _tiny_dataset()
    model = source_probe()
    inputs, targets, parameters = dataset.transitions
    outputs = model(inputs, parameters)
    ones = torch.ones(inputs.shape[0], dtype=torch.float64)

    torch.testing.assert_close(
        loss_from_outputs(outputs, targets, ones),
        normalized_loss(outputs, targets),
        rtol=1e-14,
        atol=1e-14,
    )


def test_e15_source_probe_uses_literal_structured_coordinates() -> None:
    model = source_probe()

    assert model.ax_upper[0].item() == 0.125
    assert model.ay_upper[47].item() == -0.25
    assert model.diffusion_log_rate[7].item() == pytest.approx(
        torch.log(torch.tensor(0.375)).item()
    )
    assert torch.count_nonzero(model.ax_upper) == 1
    assert torch.count_nonzero(model.ay_upper) == 1
    assert torch.count_nonzero(model.diffusion_log_rate) == 1


def test_e15_literal_outputs_supports_fixed_oracle_without_forward() -> None:
    dataset = _tiny_dataset()
    inputs, _, parameters = dataset.transitions
    oracle = FixedGenerator(*oracle_generators())

    actual = literal_outputs(oracle, inputs, parameters)
    expected = oracle.step(inputs, parameters, rule="combined")

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def _recovery(bits: int) -> dict[str, bool]:
    names = (
        "schedule_weighted_adamw_neutral",
        "schedule_weighted_adamw_restart",
        "schedule_weighted_componentwise_lbfgs_neutral",
        "schedule_weighted_componentwise_lbfgs_restart",
    )
    return {name: bool(bits & (1 << index)) for index, name in enumerate(names)}


@pytest.mark.parametrize("bits", range(16))
def test_e15_classification_covers_every_complete_arm_pattern(bits: int) -> None:
    recovery = _recovery(bits)
    if recovery["schedule_weighted_adamw_neutral"]:
        expected = "deterministic_objective_adamw_succeeds_from_neutral"
    elif recovery["schedule_weighted_adamw_restart"]:
        expected = "deterministic_objective_adamw_restart_repairs_e12_checkpoint_only"
    elif recovery["schedule_weighted_componentwise_lbfgs_neutral"]:
        expected = "componentwise_strong_wolfe_lbfgs_package_succeeds_from_neutral"
    elif recovery["schedule_weighted_componentwise_lbfgs_restart"]:
        expected = "componentwise_strong_wolfe_lbfgs_restart_repairs_e12_checkpoint_only"
    else:
        expected = "uniform_population_weighting_required_under_frozen_componentwise_controls"

    assert (
        classify(
            preflight_passed=True,
            reproduction_passed=True,
            integrity_passed=True,
            execution_complete=True,
            recovery=recovery,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("preflight", "reproduction", "integrity", "complete", "expected"),
    (
        (False, False, False, False, "e15_preflight_failed"),
        (True, False, False, False, "e12_reproduction_failed"),
        (True, True, False, False, "frozen_ceiling_or_objective_integrity_failed"),
        (True, True, True, False, "e15_execution_incomplete"),
    ),
)
def test_e15_earlier_failure_precedence_cannot_be_overridden(
    preflight: bool,
    reproduction: bool,
    integrity: bool,
    complete: bool,
    expected: str,
) -> None:
    assert (
        classify(
            preflight_passed=preflight,
            reproduction_passed=reproduction,
            integrity_passed=integrity,
            execution_complete=complete,
            recovery=_recovery(15),
        )
        == expected
    )


def test_e15_trace_environment_and_timeout_contract_is_literal() -> None:
    cfg = frozen_e15_config()

    assert TRACE_STEPS == (0, 1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 1500)
    assert cfg.expected_python == (3, 12, 7)
    assert cfg.expected_torch == "2.7.0"
    assert cfg.intraop_threads == cfg.interop_threads == 1
    assert PER_REPLICA_TIMEOUT_SECONDS == 21600
    assert WHOLE_EXPERIMENT_TIMEOUT_SECONDS == 50400
    assert VALIDATION_CASE_NAMES == (
        "x_-0.20",
        "y_-0.20",
        "x_+0.20",
        "y_+0.20",
        "x_-0.60",
        "y_-0.60",
        "x_+0.60",
        "y_+0.60",
        "x_-1.00",
        "y_-1.00",
        "x_+1.00",
        "y_+1.00",
        "diffusion_0.010",
        "diffusion_0.045",
        "diffusion_0.080",
        "composite_a",
        "composite_b",
        "composite_c",
    )
    assert frozen_validation_trace_scope() == {
        "validation_case_names": list(VALIDATION_CASE_NAMES),
        "validation_horizons": [1, 8],
    }


def test_e15_fresh_adamw_has_no_moment_state_and_exact_constructor() -> None:
    model = StructuredGenerator()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.02,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        amsgrad=False,
        maximize=False,
        foreach=None,
        capturable=False,
        differentiable=False,
        fused=None,
    )
    record = adamw_state_record(optimizer)

    assert record["parameter_state_entries"] == 0
    assert record["complete_state"]["state"] == {}
    assert record["complete_state"]["param_groups"][0]["params"] == [0, 1, 2]
    assert record["constructor"] == {
        "lr": 0.02,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 0.0,
        "amsgrad": False,
        "maximize": False,
        "foreach": None,
        "capturable": False,
        "differentiable": False,
        "fused": None,
    }
    assert len(record["canonical_sha256"]) == 64


def _sample_raw() -> dict[str, bytes]:
    return {
        "replicate_a/result.json": b'{"same":true}\n',
        "replicate_b/result.json": b'{"same":true}\n',
        "complete_result.json": b'{"same":true,"replication":{}}\n',
    }


def test_e15_bundle_has_exact_paths_order_and_metadata() -> None:
    first, members = deterministic_bundle(_sample_raw())
    second, second_members = deterministic_bundle(_sample_raw())

    assert first == second
    assert members == second_members
    verify_bundle(first, members)
    with tarfile.open(fileobj=io.BytesIO(first), mode="r:gz") as archive:
        assert [member.name for member in archive.getmembers()] == list(RAW_MEMBERS)
        for member in archive.getmembers():
            assert member.isfile()
            assert member.mode == 0o644
            assert member.uid == member.gid == member.mtime == 0
            assert member.uname == member.gname == ""
            assert member.pax_headers == {}


def _sample_result() -> dict[str, object]:
    return {
        "schema_version": 1,
        "experiment": "canonical_latent_e15_training_package",
        "classification": "deterministic_objective_adamw_succeeds_from_neutral",
        "config": {"frozen": True},
        "config_sha256": "a" * 64,
        "provenance": {"git_head": "b" * 40, "source_files": {}},
        "boundary": {
            "heldout_reads": 0,
            "provider_calls": 0,
            "routing_paths": 0,
            "encoder_updates": 0,
        },
        "state_reads": {"training": 768, "validation": 256, "heldout": 0},
    }


def test_e15_compact_and_manifest_have_one_way_hash_graph() -> None:
    first = _sample_result()
    second = json.loads(json.dumps(first))
    complete = {
        **first,
        "replication": {
            "raw_replicates_byte_identical": True,
            "replicate_a_raw_sha256": "c" * 64,
            "replicate_b_raw_sha256": "c" * 64,
        },
    }

    outputs = build_outputs(first, second, complete)
    verify_outputs(outputs)
    compact = json.loads(outputs[OUTPUT_NAMES["result"]])
    manifest = json.loads(outputs[OUTPUT_NAMES["manifest"]])

    assert set(outputs) == set(OUTPUT_NAMES.values())
    assert "raw_sha256" not in compact
    assert "bytes" not in compact
    assert "manifest" not in manifest["outputs"]
    assert manifest["compact_result_self_hash_declared"] is False
    assert manifest["manifest_self_hash_declared"] is False


def test_e15_atomic_publication_leaves_exact_top_level_set(tmp_path: Path) -> None:
    first = _sample_result()
    complete = {**first, "replication": {"raw_replicates_byte_identical": True}}
    outputs = build_outputs(first, json.loads(json.dumps(first)), complete)
    destination = tmp_path / "e15"

    publish_atomic(destination, outputs)

    assert {path.name for path in destination.iterdir()} == set(OUTPUT_NAMES.values())
    verify_outputs({path.name: path.read_bytes() for path in destination.iterdir()})
    with pytest.raises(FileExistsError):
        publish_atomic(destination, outputs)
    assert destination.exists()
    verify_outputs({path.name: path.read_bytes() for path in destination.iterdir()})
    assert not (tmp_path / ".e15.lock").exists()


def test_e15_atomic_publication_rolls_back_after_post_rename_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _sample_result()
    complete = {**first, "replication": {"raw_replicates_byte_identical": True}}
    outputs = build_outputs(first, json.loads(json.dumps(first)), complete)
    destination = tmp_path / "e15"
    original_fsync_directory = e15._fsync_directory
    injected = {"raised": False}

    def fail_once_after_rename(path: Path) -> None:
        if path == tmp_path and destination.exists() and not injected["raised"]:
            injected["raised"] = True
            raise OSError("injected post-rename fsync failure")
        original_fsync_directory(path)

    monkeypatch.setattr(e15, "_fsync_directory", fail_once_after_rename)

    with pytest.raises(OSError, match="post-rename"):
        publish_atomic(destination, outputs)

    assert injected["raised"] is True
    assert not destination.exists()
    assert not (tmp_path / ".e15.lock").exists()
    assert not list(tmp_path.glob(".e15.stage-*"))


def test_e15_rejects_noncanonical_destination_before_any_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = {"environment": False}

    def forbidden_environment(_cfg: object) -> dict[str, object]:
        called["environment"] = True
        raise AssertionError("environment preflight must not run")

    monkeypatch.setattr(e15, "configure_environment", forbidden_environment)

    with pytest.raises(ValueError, match="output directory is frozen"):
        run_e15(frozen_e15_config(), output_dir=tmp_path / "wrong")

    assert called["environment"] is False


@pytest.mark.parametrize(
    ("error", "reason"),
    (
        (TimeoutError("injected"), "timeout"),
        (E15IncompleteError("nonfinite_optimizer_state"), "nonfinite_optimizer_state"),
        (E15IncompleteError("publication_failure"), "publication_failure"),
        (MemoryError("injected"), "resource_exhaustion"),
        (OSError("injected"), "resource_or_io_failure"),
    ),
)
def test_e15_runtime_failures_return_bounded_incomplete_status(
    error: BaseException, reason: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(e15, "configure_environment", lambda _cfg: {"passed": True})
    monkeypatch.setattr(e15, "provenance", lambda _environment: {"passed": True})

    def fail_run(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise error

    monkeypatch.setattr(e15, "_run_e15_under_limit", fail_run)

    status = run_e15(frozen_e15_config(), output_dir=CANONICAL_OUTPUT_DIR)

    assert status == incomplete_status(reason)
    assert status["classification"] == "e15_execution_incomplete"
    assert status["scientific_conclusion_recorded"] is False
    assert status["durable_evidence_published"] is False


def test_e15_sealed_uniform_ceilings_bind_every_evaluation_leaf() -> None:
    report = sealed_ceiling_integrity()

    assert report["passed"] is True
    assert all(record["passed"] for record in report["controls"].values())
    assert all(len(record["evaluation_sha256"]) == 64 for record in report["controls"].values())


def test_e15_serializers_reject_nonfinite_and_distinguish_pretty_bytes() -> None:
    value = {"b": [1.0, True], "a": 3}

    assert canonical_bytes(value) == b'{"a":3,"b":[1.0,true]}'
    assert pretty_bytes(value).endswith(b"\n")
    assert canonical_bytes(value) != pretty_bytes(value)
    with pytest.raises(ValueError):
        canonical_bytes({"bad": float("nan")})


def test_e15_difference_uses_registered_absolute_and_relative_tolerance() -> None:
    report = _difference(
        torch.tensor([1.0, 0.0], dtype=torch.float64),
        torch.tensor([1.0 + 5e-15, 5e-15], dtype=torch.float64),
    )

    assert report["passed"] is True
