import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

import scripts.seal_canonical_latent_e14_evidence as e14
from scripts.seal_canonical_latent_e14_evidence import (
    CONTROLS,
    E13_ORIGINAL_EVIDENCE_STATUS,
    EXPECTED_CANONICAL_COMBINED_SHA256,
    EXPECTED_CANONICAL_REPLICATE_SHA256,
    EXPECTED_RAW_HASHES,
    FLOAT64_EPSILON,
    MODE_METRICS,
    OUTPUT_NAMES,
    SEALED_CLASSIFICATION,
    _all_finite,
    _rank_from_record,
    canonical_bytes,
    classification_from_recomputed,
    classification_inputs_from_sections,
    deterministic_bundle,
    independent_recompute,
    pretty_bytes,
    publish_atomic,
    recompute_recovery_pass,
    sha256_bytes,
    validate_frozen_input_bytes,
    verify_bundle,
    verify_seal_files,
)

FROZEN_E13_RESULT = Path(
    "/private/tmp/canonical_latent_e13_identifiability_audit_fe47b93/replicate_a/result.json"
)


def test_e14_hash_labels_are_distinct_and_literal() -> None:
    assert (
        EXPECTED_RAW_HASHES["replicate_a_result.json"]
        == EXPECTED_RAW_HASHES["replicate_b_result.json"]
    )
    assert EXPECTED_RAW_HASHES["replicate_a_result.json"] != EXPECTED_CANONICAL_REPLICATE_SHA256
    assert EXPECTED_RAW_HASHES["complete_result.json"] != EXPECTED_CANONICAL_COMBINED_SHA256
    assert set(OUTPUT_NAMES) == {"bundle", "result", "manifest"}
    assert len(set(OUTPUT_NAMES.values())) == 3


def test_e14_json_serializations_have_unambiguous_hashes() -> None:
    value = {"z": [1.25, True], "a": {"x": 3}}

    assert canonical_bytes(value) == b'{"a":{"x":3},"z":[1.25,true]}'
    assert pretty_bytes(value).endswith(b"\n")
    assert (
        sha256_bytes(canonical_bytes(value)) == hashlib.sha256(canonical_bytes(value)).hexdigest()
    )
    assert sha256_bytes(canonical_bytes(value)) != sha256_bytes(pretty_bytes(value))


def _sample_raw() -> dict[str, bytes]:
    return {
        "replicate_a_result.json": b'{"replicate":"a"}\n',
        "replicate_b_result.json": b'{"replicate":"a"}\n',
        "complete_result.json": b'{"complete":true}\n',
    }


def test_e14_deterministic_bundle_preserves_bytes_header_and_metadata() -> None:
    first, members = deterministic_bundle(_sample_raw())
    second, second_members = deterministic_bundle(_sample_raw())

    assert first == second
    assert members == second_members
    assert first[:3] == b"\x1f\x8b\x08"
    assert first[3] & 0x08 == 0
    assert first[4:8] == b"\x00\x00\x00\x00"
    verify_bundle(first, members)
    with tarfile.open(fileobj=io.BytesIO(first), mode="r:gz") as archive:
        assert [member.name for member in archive.getmembers()] == [
            "replicate_a_result.json",
            "replicate_b_result.json",
            "complete_result.json",
        ]
        for member in archive.getmembers():
            assert member.isfile()
            assert member.pax_headers == {}
            assert member.mode == 0o644
            assert member.uid == member.gid == member.mtime == 0
            assert member.uname == member.gname == ""


def _passing_evaluation(parameter_count: int) -> dict[str, object]:
    identification = {
        "relative_frobenius": {"A_x": 0.01, "A_y": 0.01, "D": 0.01},
        "maximum_supported_entry_relative_error": {"A_x": 0.01, "A_y": 0.01},
        "off_support_leakage": {"A_x": 0.01, "A_y": 0.01},
        "maximum_diffusion_rate_relative_error": 0.01,
        "normalized_commutators": {
            "A_x_A_y": 0.001,
            "A_x_D": 0.001,
            "A_y_D": 0.001,
        },
        "maximum_basis_action_decoded_nrmse": 0.01,
    }
    structure = {
        "parameter_count": parameter_count,
        "ax_skew_residual": 0.0,
        "ay_skew_residual": 0.0,
        "ax_constant_residual": 0.0,
        "ay_constant_residual": 0.0,
        "diffusion_off_diagonal_residual": 0.0,
        "diffusion_constant_residual": 0.0,
        "diffusion_maximum_eigenvalue": 0.0,
        "all_finite": True,
        "constant_mode_structurally_fixed": True,
        "inactive_modes_copied": True,
    }
    validation = {
        regime: {
            "one_step_decoded_nrmse": 0.01,
            "rollout_decoded_nrmse": 0.02,
            "final_high_frequency_spectral": {"nrmse": 0.03},
        }
        for regime in ("composite", "x_advection", "y_advection", "diffusion")
    }
    return {
        "generator_identification": identification,
        "structure": structure,
        "validation": validation,
    }


@pytest.mark.parametrize(
    ("control", "count"),
    (
        ("e12_adamw_replay", 2304),
        ("full_skew_lbfgs_neutral", 2304),
        ("full_skew_lbfgs_polish", 2304),
        ("support_sparse_lbfgs", 90),
        ("mode_tied_lbfgs", 12),
        ("oracle", 0),
    ),
)
def test_e14_recomputes_every_recovery_gate(control: str, count: int) -> None:
    evaluation = _passing_evaluation(count)

    passed, gates = recompute_recovery_pass(control, evaluation)

    assert passed is True
    assert len(gates) == 8
    assert all(gates.values())
    evaluation["generator_identification"]["maximum_basis_action_decoded_nrmse"] = 0.051
    failed, failed_gates = recompute_recovery_pass(control, evaluation)
    assert failed is False
    assert failed_gates["generator_identification"] is False


def test_e14_rank_recomputation_derives_float64_tolerance() -> None:
    singular_values = [10.0, 1.0, 1e-15]
    tolerance = 4 * FLOAT64_EPSILON * singular_values[0]
    record = {
        "shape": [4, 3],
        "rank": 2,
        "condition_number": 1e16,
        "rank_tolerance": tolerance,
        "singular_values": singular_values,
    }

    report = _rank_from_record(record)

    assert report["rank"] == 2
    assert report["condition_number"] == 1e16
    assert report["rank_tolerance"] == tolerance
    assert report["rank_matches"] is True
    assert report["condition_matches"] is True
    assert report["tolerance_matches"] is True
    assert report["full_rank"] is False


@pytest.mark.parametrize(
    ("preflight", "e12", "recovery", "excitation", "expected"),
    (
        (
            False,
            True,
            {"full_skew_lbfgs_neutral": True, "full_skew_lbfgs_polish": False},
            True,
            "e13_preflight_failed",
        ),
        (
            True,
            False,
            {"full_skew_lbfgs_neutral": True, "full_skew_lbfgs_polish": False},
            True,
            "e12_reproduction_failed",
        ),
        (
            True,
            True,
            {"full_skew_lbfgs_neutral": True, "full_skew_lbfgs_polish": False},
            True,
            "full_parameterization_deterministic_recovery_succeeds",
        ),
        (
            True,
            True,
            {
                "full_skew_lbfgs_neutral": False,
                "full_skew_lbfgs_polish": False,
                "support_sparse_lbfgs": True,
            },
            True,
            "support_restriction_required_under_frozen_solvers",
        ),
        (
            True,
            True,
            {
                "full_skew_lbfgs_neutral": False,
                "full_skew_lbfgs_polish": False,
                "support_sparse_lbfgs": False,
                "mode_tied_lbfgs": True,
            },
            True,
            "mode_tying_required_under_frozen_solvers",
        ),
        (
            True,
            True,
            {
                "full_skew_lbfgs_neutral": False,
                "full_skew_lbfgs_polish": False,
                "support_sparse_lbfgs": False,
                "mode_tied_lbfgs": False,
            },
            False,
            "elementary_excitation_rank_deficient",
        ),
        (
            True,
            True,
            {
                "full_skew_lbfgs_neutral": False,
                "full_skew_lbfgs_polish": False,
                "support_sparse_lbfgs": False,
                "mode_tied_lbfgs": False,
            },
            True,
            "recovery_controls_not_qualified",
        ),
    ),
)
def test_e14_classification_precedence_uses_recomputed_sections(
    preflight: bool,
    e12: bool,
    recovery: dict[str, bool],
    excitation: bool,
    expected: str,
) -> None:
    assert classification_from_recomputed(preflight, e12, recovery, excitation) == expected


def _classification_sections() -> dict[str, dict[str, bool]]:
    return {
        "provenance": {"passed": True},
        "parameterization": {"passed": True},
        "oracle": {"passed": True},
        "replay": {"replay_passed": True, "reproduction_passed": True},
        "excitation": {
            "record_integrity_passed": True,
            "required_rank_full": True,
        },
        "coverage": {"passed": True},
    }


def test_e14_wires_replay_failure_to_e12_reproduction_branch() -> None:
    sections = _classification_sections()
    sections["replay"]["replay_passed"] = False

    inputs = classification_inputs_from_sections(**sections)
    classification = classification_from_recomputed(
        inputs["preflight_passed"],
        inputs["e12_reproduction_passed"],
        {
            "full_skew_lbfgs_neutral": True,
            "full_skew_lbfgs_polish": False,
        },
        inputs["excitation_required_rank_full"],
    )

    assert inputs["preflight_passed"] is True
    assert classification == "e12_reproduction_failed"


def test_e14_wires_valid_rank_deficiency_to_excitation_branch() -> None:
    sections = _classification_sections()
    sections["excitation"]["required_rank_full"] = False

    inputs = classification_inputs_from_sections(**sections)
    classification = classification_from_recomputed(
        inputs["preflight_passed"],
        inputs["e12_reproduction_passed"],
        {
            "full_skew_lbfgs_neutral": False,
            "full_skew_lbfgs_polish": False,
            "support_sparse_lbfgs": False,
            "mode_tied_lbfgs": False,
        },
        inputs["excitation_required_rank_full"],
    )

    assert inputs["preflight_passed"] is True
    assert classification == "elementary_excitation_rank_deficient"


def _sample_seal_files() -> tuple[dict[str, bytes], list[dict[str, object]]]:
    bundle, members = deterministic_bundle(_sample_raw())
    sealed_raw_hashes = {member["name"]: member["raw_sha256"] for member in members}
    bundle_record = {
        "filename": OUTPUT_NAMES["bundle"],
        "bytes": len(bundle),
        "raw_sha256": sha256_bytes(bundle),
        "members": members,
    }
    result = pretty_bytes(
        {
            "classification": SEALED_CLASSIFICATION,
            "e13_original_evidence_status": E13_ORIGINAL_EVIDENCE_STATUS,
            "seal_does_not_modify_e13": True,
            "sealed_raw_input_hashes": sealed_raw_hashes,
            "evidence_bundle": bundle_record,
            "e14_boundary": {"seal_does_not_modify_e13": True},
        }
    )
    manifest = {
        "classification": SEALED_CLASSIFICATION,
        "e13_original_evidence_status": E13_ORIGINAL_EVIDENCE_STATUS,
        "seal_does_not_modify_e13": True,
        "sealed_raw_input_hashes": sealed_raw_hashes,
        "manifest_self_hash_declared": False,
        "boundary": {"seal_does_not_modify_e13": True},
        "inputs": {
            "raw_files": {
                member["name"]: {
                    "path": f"/frozen/{member['name']}",
                    "bytes": member["bytes"],
                    "raw_sha256": member["raw_sha256"],
                }
                for member in members
            }
        },
        "outputs": {
            "compact_result": {
                "filename": OUTPUT_NAMES["result"],
                "bytes": len(result),
                "raw_sha256": sha256_bytes(result),
            },
            "evidence_bundle": bundle_record,
        },
    }
    return {
        "bundle": bundle,
        "result": result,
        "manifest": pretty_bytes(manifest),
    }, members


def test_e14_manifest_reopen_checks_every_hash_and_byte_count() -> None:
    files, members = _sample_seal_files()
    verify_seal_files(files, members)

    manifest = json.loads(files["manifest"])
    manifest["inputs"]["raw_files"]["complete_result.json"]["bytes"] += 1
    corrupted = {**files, "manifest": pretty_bytes(manifest)}
    with pytest.raises(RuntimeError, match="input byte count mismatch"):
        verify_seal_files(corrupted, members)


@pytest.mark.parametrize(
    "failure_point",
    (
        "after_write_bundle",
        "after_write_result",
        "after_write_manifest",
        "after_reopen_bundle",
        "after_reopen_result",
        "after_reopen_manifest",
        "after_reopen_verification",
        "before_final_entry_check",
        "after_rename",
        "before_lock_close",
        "after_lock_close",
        "before_lock_unlink",
        "after_lock_unlink",
        "before_lock_parent_fsync",
        "after_lock_parent_fsync",
    ),
)
def test_e14_atomic_publication_cleans_every_injected_failure(
    tmp_path: Path, failure_point: str
) -> None:
    files, members = _sample_seal_files()
    output_dir = tmp_path / "sealed"

    with pytest.raises(RuntimeError, match="injected E14 publication failure"):
        publish_atomic(
            output_dir,
            files,
            members,
            failure_point=failure_point,
        )

    assert not output_dir.exists()
    assert list(tmp_path.iterdir()) == []


def test_e14_atomic_publication_renames_complete_verified_directory(tmp_path: Path) -> None:
    files, members = _sample_seal_files()
    output_dir = tmp_path / "sealed"

    publish_atomic(output_dir, files, members)

    assert output_dir.is_dir()
    assert {path.name for path in output_dir.iterdir()} == set(OUTPUT_NAMES.values())
    reopened = {key: (output_dir / filename).read_bytes() for key, filename in OUTPUT_NAMES.items()}
    assert reopened == files
    verify_seal_files(reopened, members)


def test_e14_refuses_any_existing_final_directory(tmp_path: Path) -> None:
    files, members = _sample_seal_files()
    output_dir = tmp_path / "sealed"
    output_dir.mkdir()

    with pytest.raises(RuntimeError, match="must be absent"):
        publish_atomic(output_dir, files, members)

    assert output_dir.is_dir()
    assert list(output_dir.iterdir()) == []


def test_e14_refuses_broken_symlink_final_entry(tmp_path: Path) -> None:
    files, members = _sample_seal_files()
    output_dir = tmp_path / "sealed"
    output_dir.symlink_to(tmp_path / "missing-target", target_is_directory=True)

    with pytest.raises(RuntimeError, match="entry must be absent"):
        publish_atomic(output_dir, files, members)

    assert output_dir.is_symlink()
    assert not (tmp_path / f".{output_dir.name}.publish.lock").exists()


def test_e14_detects_final_entry_created_before_rename(tmp_path: Path) -> None:
    files, members = _sample_seal_files()
    output_dir = tmp_path / "sealed"

    def create_competing_entry(point: str) -> None:
        if point == "before_final_entry_check":
            output_dir.mkdir()

    with pytest.raises(RuntimeError, match="appeared during publication"):
        publish_atomic(output_dir, files, members, hook=create_competing_entry)

    assert output_dir.is_dir()
    assert list(output_dir.iterdir()) == []
    assert not list(tmp_path.glob(f".{output_dir.name}.staging-*"))
    assert not (tmp_path / f".{output_dir.name}.publish.lock").exists()


def test_e14_source_preflight_runs_before_any_result_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = []

    def failed_source() -> dict[str, object]:
        events.append("source")
        return {"passed": False}

    def forbidden_read(_run_dir: Path) -> None:
        events.append("input")
        raise AssertionError("input read occurred before source preflight")

    monkeypatch.setattr(e14, "source_preflight", failed_source)
    monkeypatch.setattr(e14, "read_frozen_inputs", forbidden_read)

    with pytest.raises(RuntimeError, match="source preflight failed"):
        e14.seal(FROZEN_E13_RESULT.parent.parent, tmp_path / "sealed")

    assert events == ["source"]


def test_e14_rejects_input_bytes_before_parsing_unregistered_content() -> None:
    raw = _sample_raw()

    with pytest.raises(RuntimeError, match="raw E13 input hash mismatch"):
        validate_frozen_input_bytes(raw)


@pytest.mark.skipif(not FROZEN_E13_RESULT.exists(), reason="frozen local E13 bytes unavailable")
def test_e14_full_frozen_e13_recomputation_reaches_registered_classification() -> None:
    result = json.loads(FROZEN_E13_RESULT.read_bytes())

    recomputed = independent_recompute(result)

    assert recomputed["passed"] is True
    assert (
        recomputed["recovery_and_classification"]["recomputed_classification"]
        == "full_parameterization_deterministic_recovery_succeeds"
    )
    assert recomputed["coverage_and_argmax"]["checks"]["exact_cartesian"] is True
    assert recomputed["excitation"]["integrity_checks"]["rank_records_recomputed"] is True


def test_e14_finiteness_and_cartesian_constants_are_literal() -> None:
    assert _all_finite({"values": [1.0, 2.0]}) is True
    assert _all_finite({"values": [1.0, float("inf")]}) is False
    assert len(CONTROLS) == 6
    assert len(MODE_METRICS) == 5
    assert len(CONTROLS) * 49 * 18 * 2 == 10584
