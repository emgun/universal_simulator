#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import os
import shutil
import subprocess
import tarfile
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = Path(__file__).resolve()
CONTRACT_PATH = (
    REPO_ROOT / "docs/research/2026-07-26-canonical-latent-e14-evidence-seal-contract.md"
)
TEST_PATH = REPO_ROOT / "tests/unit/test_seal_canonical_latent_e14_evidence.py"
E13_CONTRACT_PATH = (
    REPO_ROOT / "docs/research/2026-07-26-canonical-latent-e13-identifiability-audit-contract.md"
)
E13_RUNNER_PATH = REPO_ROOT / "scripts/run_canonical_latent_e13_identifiability_audit.py"

EXPECTED_E13_RUN_DIR = Path("/private/tmp/canonical_latent_e13_identifiability_audit_fe47b93")
EXPECTED_E13_EXECUTION_HEAD = "fe47b937205f47a2dba93f0ecbeee83015824c09"
EXPECTED_E13_CONFIG_SHA256 = "08077c555b3476c51e13b861065835f914fdd6c5ae44e1a22dd8d08ec24d12dc"
EXPECTED_E12_CONFIG_SHA256 = "cd428d490ad9d5505f88ead66b41fdb25e25830f45d0eb21f451c5dbea261934"
EXPECTED_RAW_HASHES = {
    "replicate_a_result.json": "e0bc7b0575a8c6982ecc9fbff3899be0792643a57575a2349468a0244e7e6d9c",
    "replicate_b_result.json": "e0bc7b0575a8c6982ecc9fbff3899be0792643a57575a2349468a0244e7e6d9c",
    "complete_result.json": "f65cdc9b6b965b5e77715ae5b00d1ecc21dd683b7ffe09c8ce19235d3283159a",
}
EXPECTED_CANONICAL_REPLICATE_SHA256 = (
    "d04d71e7b528d2332ca9de98bb64d6f065dff4614d82a2416ec5330ce3a41f19"
)
EXPECTED_CANONICAL_COMBINED_SHA256 = (
    "022f8867fab4f9b28f84da501245928ee3ef3f074dc6d1a84ed04c56bad7abbb"
)
EXPECTED_UNDERLYING_CLASSIFICATION = "full_parameterization_deterministic_recovery_succeeds"
SEALED_CLASSIFICATION = "e13_scientific_result_sealed"
E13_ORIGINAL_EVIDENCE_STATUS = "scientifically_recomputed_but_original_seal_nonconforming"
PERSISTENCE_ROLLOUT = 1.5584481380508215
FLOAT64_EPSILON = 2.220446049250313e-16

CONTROLS = (
    "e12_adamw_replay",
    "full_skew_lbfgs_neutral",
    "full_skew_lbfgs_polish",
    "support_sparse_lbfgs",
    "mode_tied_lbfgs",
    "oracle",
)
LEARNED_CONTROLS = CONTROLS[:-1]
REGIMES = ("composite", "x_advection", "y_advection", "diffusion")
ELEMENTARY_REGIMES = REGIMES[1:]
MODE_METRICS = (
    "decoded_nrmse",
    "coefficient_nrmse",
    "coefficient_angle_radians",
    "absolute_amplitude_ratio_error",
    "off_target_direction_residual",
)
EXPECTED_CASE_NAMES = (
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
EXPECTED_PARAMETER_COUNTS = {
    "e12_adamw_replay": 2304,
    "full_skew_lbfgs_neutral": 2304,
    "full_skew_lbfgs_polish": 2304,
    "support_sparse_lbfgs": 90,
    "mode_tied_lbfgs": 12,
    "oracle": 0,
}
EXPECTED_PREFLIGHT_PARAMETER_COUNTS = {
    "full_skew": 2304,
    "support_sparse": 90,
    "mode_tied": 12,
}
EXPECTED_MASK_HASHES = {
    "A_x": "fa83470fcf9853aa4d607ec8935d8b73cffec3735242ef87370661aa67af2806",
    "A_y": "20562976940d1bb28e528ebcbf259fcfb53156b50e606ebffe8a6db322cb3fee",
    "D": "d490fb408871d40fafbf53d94be298ba475a27ec033ef079ff98dccef9ed29b1",
}
EXPECTED_MASK_NONZERO = {"A_x": 42, "A_y": 42, "D": 48}
EXPECTED_E13_SOURCES = {
    "contract": (
        "docs/research/2026-07-26-canonical-latent-e13-identifiability-audit-contract.md",
        "3b130629bfc431f6862ac66ee85930be3b64d4704b75369d967f9c68786b38de",
    ),
    "runner": (
        "scripts/run_canonical_latent_e13_identifiability_audit.py",
        "f95b1e50f409fc939c62120d06f2eafa89a864de148f9309fb38d481c01310c2",
    ),
    "e12_lock": (
        "docs/research/artifacts/canonical_latent_e12_replay_lock.json",
        "0bf8f032daf95415bee401b5a90f4e6ca1598f12748151dd7310b2a6d02f8dfd",
    ),
    "e12_artifact": (
        "docs/research/artifacts/canonical_latent_e12_structured_generator_result.json",
        "d4760ec3d69b4397cc14ffc3bb08edd3f073edcaf1f6d4dd70db070a96cab3b2",
    ),
    "e12_runner": (
        "scripts/run_canonical_latent_e12_structured_generator.py",
        "8edb67652d53e101a63730b9ec4803a69067572a8bab6eee0fb98627785a926a",
    ),
    "e11_runner": (
        "scripts/run_canonical_latent_e11_coefficient_operator_transfer.py",
        "720e2ad33b92faee49fcfbdee84c66c023b40bf1f50427f874f231ab555483eb",
    ),
    "e7_runner": (
        "scripts/run_canonical_latent_e7_function_space.py",
        "cf81597b3909e9693508b62e595eb006a8598d186de062eaf4a8f241d4b07488",
    ),
    "latent_evaluation": (
        "src/ups/eval/latent_qualification.py",
        "e2bb0fb86ac464aa6b96221d706f71aad4fd8fb48992613ead6d5b94e1943994",
    ),
}
EXPECTED_E12_REPRODUCTION = {
    "initial_model_sha256": "64c87294711c68c9bc4a9f56cb3f8a8ca23b1e1eed84493bfc18e18f3a2c9218",
    "elementary_model_sha256": "e9c17bc1871f5b2008d3899da9f59a44cf1209448f50ba576ca63f6281602e7b",
    "generator_sha256": {
        "A_x": "17f70896f48f5651854746c5be9edc7dd42d2fd331c1fbc7d2c6e2edd1a46e53",
        "A_y": "726d18845ad3dc3e462fc16578adf08929594e57dd02818c1e13711eee31fda2",
        "D": "1edb9c7429ebec8a583d000fedf2e754668727fed857b14d3d06f5b1649fa370",
    },
    "training": {
        "first_loss": 0.08166621133143402,
        "final_loss": 0.00032425428259914163,
        "updates": 1500,
        "examples_per_regime": 48000,
        "training_rule": "combined",
    },
    "basis_action_decoded_nrmse": 0.09992145782297766,
    "zero_shot_rollout_decoded_nrmse": 0.00536402045663299,
}
EXPECTED_DATASET_RECORDS = {
    "pretrain_x_advection": {
        "complete_trajectory_sha256": "9e31f0a7a38aad3c2a7e99a0178dac044276cc28e70791259a075ad558f52692",
        "initial_coefficients_sha256": "dc04305d0b1038c902753e3d604c76ce1bc1d10bbbaeb9a40017af33e2518d62",
        "parameters_sha256": "8e7a81b503d74612582901c8feabf4c5d9efc0a4760767299e2c05acab478e38",
        "shape": [256, 9, 52, 1],
    },
    "pretrain_y_advection": {
        "complete_trajectory_sha256": "9742746234e3cafba6b1a178aed6a488fc858099974f8b3b96c5294c261808f1",
        "initial_coefficients_sha256": "dc04305d0b1038c902753e3d604c76ce1bc1d10bbbaeb9a40017af33e2518d62",
        "parameters_sha256": "bbe2918be4233dc8a4b78f3a02c3ed6413f6b78642f1c1cf25cd2b2e0501cfbc",
        "shape": [256, 9, 52, 1],
    },
    "pretrain_diffusion": {
        "complete_trajectory_sha256": "b3ba103ce13ec2b02f3350e0c3a927095fa30f8f37a52890c8cf1862cbbf1609",
        "initial_coefficients_sha256": "dc04305d0b1038c902753e3d604c76ce1bc1d10bbbaeb9a40017af33e2518d62",
        "parameters_sha256": "1a86ac57fd47034074001a54987cc6c0509ce205f8e3196f8a7040eb4fa8d205",
        "shape": [256, 9, 52, 1],
    },
    "composite_validation": {
        "complete_trajectory_sha256": "9b0c5855a41724a777b20caaa565c63ae7bb663b22ac4bba84ec32278eff6f18",
        "initial_coefficients_sha256": "375faef7cc6471e3d78d5d7b83ca137ef8ae463e8655acc262bf9bb9b69c1d37",
        "parameters_sha256": "364f0eed017069d7266f758e5962b984a65fa7bacd334002387a85f76e367003",
        "shape": [64, 9, 52, 1],
    },
    "x_advection_validation": {
        "complete_trajectory_sha256": "946a6a16778ad3eec604a252c0a34b85de218055fcf3adad60d7f2e463227000",
        "initial_coefficients_sha256": "375faef7cc6471e3d78d5d7b83ca137ef8ae463e8655acc262bf9bb9b69c1d37",
        "parameters_sha256": "e8636962b693991c8c9539288c9821dded159dd058d5c59f31a17e23e0cb5135",
        "shape": [64, 9, 52, 1],
    },
    "y_advection_validation": {
        "complete_trajectory_sha256": "c79a8a1be4deb1d02578c8a21190b29e5539e9fc4dfd6a386d2599e0ae25f585",
        "initial_coefficients_sha256": "375faef7cc6471e3d78d5d7b83ca137ef8ae463e8655acc262bf9bb9b69c1d37",
        "parameters_sha256": "218577be06bf309b5fd75d045bfa1676e2a9fd4a35832fbde66d1ca09cb10d7d",
        "shape": [64, 9, 52, 1],
    },
    "diffusion_validation": {
        "complete_trajectory_sha256": "52a33137280bf2b5abf30edfa158679f7ffaa3773172dfab234d2da74bcff72a",
        "initial_coefficients_sha256": "375faef7cc6471e3d78d5d7b83ca137ef8ae463e8655acc262bf9bb9b69c1d37",
        "parameters_sha256": "b9d6fd0e94df19feb54fdd3e322d0e9c603ed45da9ef767a52d1b6f8aeaedf0f",
        "shape": [64, 9, 52, 1],
    },
}
EXPECTED_SCHEDULE_RECORDS = {
    "x_advection": {
        "sha256": "096659f791a0d5e728ccac7aa02801c60856bd0659a080464bb158909be3f6f7",
        "shape": [1500, 32],
    },
    "y_advection": {
        "sha256": "49401932ae228e58f2927a7695d443a17a1a41c8e80375e4346a06973b431507",
        "shape": [1500, 32],
    },
    "diffusion": {
        "sha256": "d86f8d73a236e806ff1988c274af8fd9a2daa30beeada74e8427d526d5b88487",
        "shape": [1500, 32],
    },
    "full_control": {
        "sha256": "602ff2694d9923821782d69627c0b7ece849086abcc431f4ca085de6486519cf",
        "shape": [1500, 96],
    },
    "fine_tune": {
        "sha256": "b0de4bcb3d59866dd05489d7ecf13574dd7763ddf82a6706f72734ec701a7e32",
        "shape": [400, 64],
    },
}
OUTPUT_NAMES = {
    "bundle": "canonical_latent_e14_evidence_bundle.tar.gz",
    "result": "canonical_latent_e14_evidence_seal_result.json",
    "manifest": "canonical_latent_e14_evidence_seal_manifest.json",
}
OUTPUT_ORDER = ("bundle", "result", "manifest")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def pretty_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _git_text(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_bytes_at(commit: str, path: str) -> bytes | None:
    completed = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
    )
    return completed.stdout if completed.returncode == 0 else None


def _committed_bytes(path: Path) -> bytes | None:
    return _git_bytes_at("HEAD", path.relative_to(REPO_ROOT).as_posix())


def source_preflight() -> dict[str, Any]:
    current_sources = {
        "contract": CONTRACT_PATH,
        "runner": RUNNER_PATH,
        "tests": TEST_PATH,
        "e13_contract": E13_CONTRACT_PATH,
        "e13_runner": E13_RUNNER_PATH,
    }
    current_records = {}
    for name, path in current_sources.items():
        working = path.read_bytes()
        committed = _committed_bytes(path)
        current_records[name] = {
            "path": str(path.relative_to(REPO_ROOT)),
            "working_sha256": sha256_bytes(working),
            "committed_sha256": sha256_bytes(committed) if committed is not None else None,
            "matches_git_head": committed is not None and working == committed,
        }
    e13_snapshot_records = {}
    for name, (path, expected_hash) in EXPECTED_E13_SOURCES.items():
        frozen = _git_bytes_at(EXPECTED_E13_EXECUTION_HEAD, path)
        actual_hash = sha256_bytes(frozen) if frozen is not None else None
        e13_snapshot_records[name] = {
            "path": path,
            "expected_sha256": expected_hash,
            "frozen_commit_sha256": actual_hash,
            "matches_frozen_commit": actual_hash == expected_hash,
        }
    git_head = _git_text("rev-parse", "HEAD")
    checks = {
        "git_head_present": bool(git_head),
        "worktree_clean": _git_text("status", "--porcelain") == "",
        "current_sources_match_git_head": all(
            record["matches_git_head"] for record in current_records.values()
        ),
        "e13_execution_commit_present": (
            _git_text("rev-parse", EXPECTED_E13_EXECUTION_HEAD) == EXPECTED_E13_EXECUTION_HEAD
        ),
        "complete_e13_source_snapshot": set(e13_snapshot_records) == set(EXPECTED_E13_SOURCES),
        "e13_source_snapshot_matches": all(
            record["matches_frozen_commit"] for record in e13_snapshot_records.values()
        ),
    }
    return {
        "git_head": git_head,
        "checks": checks,
        "current_sources": current_records,
        "e13_source_snapshot": e13_snapshot_records,
        "passed": all(checks.values()),
    }


def _input_paths(run_dir: Path) -> dict[str, Path]:
    return {
        "replicate_a_result.json": run_dir / "replicate_a/result.json",
        "replicate_b_result.json": run_dir / "replicate_b/result.json",
        "complete_result.json": run_dir / "complete_result.json",
    }


def validate_frozen_input_bytes(raw: dict[str, bytes]) -> dict[str, Any]:
    raw_hashes = {name: sha256_bytes(value) for name, value in raw.items()}
    if raw_hashes != EXPECTED_RAW_HASHES:
        raise RuntimeError(f"E14 raw E13 input hash mismatch: {raw_hashes}")
    if raw["replicate_a_result.json"] != raw["replicate_b_result.json"]:
        raise RuntimeError("E14 E13 raw replicate bytes differ")
    parsed = {name: json.loads(value) for name, value in raw.items()}
    replicate_a = parsed["replicate_a_result.json"]
    replicate_b = parsed["replicate_b_result.json"]
    combined = parsed["complete_result.json"]
    if replicate_a != replicate_b:
        raise RuntimeError("E14 E13 parsed replicate objects differ")
    canonical_hashes = {
        "replicate_payload_sha256": sha256_bytes(canonical_bytes(replicate_a)),
        "combined_payload_sha256": sha256_bytes(canonical_bytes(combined)),
    }
    expected_canonical = {
        "replicate_payload_sha256": EXPECTED_CANONICAL_REPLICATE_SHA256,
        "combined_payload_sha256": EXPECTED_CANONICAL_COMBINED_SHA256,
    }
    if canonical_hashes != expected_canonical:
        raise RuntimeError(f"E14 canonical E13 payload hash mismatch: {canonical_hashes}")
    combined_without_replication = dict(combined)
    embedded_replication = combined_without_replication.pop("replication")
    if combined_without_replication != replicate_a:
        raise RuntimeError("E14 combined E13 payload does not extend the replicate")
    return {
        "parsed": parsed,
        "raw_hashes": raw_hashes,
        "canonical_hashes": canonical_hashes,
        "embedded_mislabeled_replication_hashes": embedded_replication,
        "raw_replicates_byte_identical": True,
        "parsed_replicates_equal": True,
        "combined_extends_replicate": True,
    }


def read_frozen_inputs(run_dir: Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    if run_dir.resolve() != EXPECTED_E13_RUN_DIR:
        raise RuntimeError("E14 input directory is not the frozen E13 run directory")
    raw = {name: path.read_bytes() for name, path in _input_paths(run_dir).items()}
    return raw, validate_frozen_input_bytes(raw)


def _all_finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_all_finite(item) for item in value)
    return True


def _float_equal(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-14, abs_tol=1e-18)


def _rank_from_record(record: dict[str, Any]) -> dict[str, Any]:
    singular_values = record["singular_values"]
    shape = record["shape"]
    largest = max(singular_values)
    tolerance = max(shape) * FLOAT64_EPSILON * largest
    rank = sum(value > tolerance for value in singular_values)
    condition = largest / max(min(singular_values), 1e-300)
    return {
        "rank": rank,
        "condition_number": condition,
        "rank_tolerance": tolerance,
        "rank_matches": rank == record["rank"],
        "condition_matches": _float_equal(condition, record["condition_number"]),
        "tolerance_matches": _float_equal(tolerance, record["rank_tolerance"]),
        "full_rank": rank == min(shape),
    }


def _structure_passes(record: dict[str, Any], expected_count: int) -> bool:
    return all(
        (
            record["parameter_count"] == expected_count,
            record["ax_skew_residual"] <= 1e-12,
            record["ay_skew_residual"] <= 1e-12,
            record["ax_constant_residual"] <= 1e-12,
            record["ay_constant_residual"] <= 1e-12,
            record["diffusion_off_diagonal_residual"] <= 1e-12,
            record["diffusion_constant_residual"] <= 1e-12,
            record["diffusion_maximum_eigenvalue"] <= 1e-12,
            record["all_finite"],
            record["constant_mode_structurally_fixed"],
            record["inactive_modes_copied"],
        )
    )


def verify_provenance(result: dict[str, Any]) -> dict[str, Any]:
    provenance = result["provenance"]
    source_records = provenance["source_files"]
    source_checks = {}
    for name, (path, expected_hash) in EXPECTED_E13_SOURCES.items():
        record = source_records.get(name, {})
        source_checks[name] = all(
            (
                record.get("path") == path,
                record.get("working_sha256") == expected_hash,
                record.get("committed_sha256") == expected_hash,
                record.get("matches_git_head") is True,
            )
        )
    config_hash = sha256_bytes(canonical_bytes(result["config"]))
    zero_boundary = all(
        (
            result["state_reads"]["heldout"] == 0,
            result["boundary"]["heldout_reads"] == 0,
            result["boundary"]["provider_calls"] == 0,
            result["boundary"]["routing_paths"] == 0,
            not result["boundary"]["representation_label_inputs"],
            not result["boundary"]["task_label_inputs"],
            not result["boundary"]["original_observations_after_projection"],
        )
    )
    checks = {
        "complete_source_set": set(source_records) == set(EXPECTED_E13_SOURCES),
        "all_source_records_exact": all(source_checks.values()),
        "execution_head_exact": provenance["git_head"] == EXPECTED_E13_EXECUTION_HEAD,
        "git_head_present": provenance["git_head_present"] is True,
        "execution_worktree_clean": provenance["worktree_clean"] is True,
        "e13_config_hash_recomputed": config_hash == EXPECTED_E13_CONFIG_SHA256,
        "e13_config_hash_stored": result["config_sha256"] == EXPECTED_E13_CONFIG_SHA256,
        "e12_config_hash_exact": provenance["e12_config_sha256"] == EXPECTED_E12_CONFIG_SHA256,
        "deterministic_algorithms": result["reproducibility"]["deterministic_algorithms"] is True,
        "zero_boundary": zero_boundary,
        "all_finite": _all_finite(result),
    }
    recomputed_pass = all(checks.values())
    stored_pass_matches = all(
        (
            provenance["passed"] == recomputed_pass,
            provenance["source_files_match_git_head"] == checks["all_source_records_exact"],
            provenance["e12_lock_hash_matches"] is True,
            provenance["e12_artifact_hash_matches"] is True,
            provenance["e12_config_hash_matches"] is True,
        )
    )
    return {
        "checks": checks,
        "source_checks": source_checks,
        "stored_pass_matches": stored_pass_matches,
        "passed": recomputed_pass,
    }


def verify_parameterization(result: dict[str, Any]) -> dict[str, Any]:
    preflight = result["parameterization_preflight"]
    structures = {
        name: _structure_passes(record, EXPECTED_PREFLIGHT_PARAMETER_COUNTS[name])
        for name, record in preflight["structures"].items()
        if name in EXPECTED_PREFLIGHT_PARAMETER_COUNTS
    }
    phase_ratio = preflight["maximum_oracle_phase"] / math.pi
    checks = {
        "parameter_count_contract": (
            preflight["expected_parameter_counts"] == EXPECTED_PREFLIGHT_PARAMETER_COUNTS
        ),
        "structure_set": set(preflight["structures"]) == set(EXPECTED_PREFLIGHT_PARAMETER_COUNTS),
        "structures": (
            set(structures) == set(EXPECTED_PREFLIGHT_PARAMETER_COUNTS) and all(structures.values())
        ),
        "mask_expected_hashes": preflight["mask"]["expected_hashes"] == EXPECTED_MASK_HASHES,
        "mask_hashes": preflight["mask"]["hashes"] == EXPECTED_MASK_HASHES,
        "mask_nonzero_entries": (preflight["mask"]["nonzero_entries"] == EXPECTED_MASK_NONZERO),
        "oracle_phase_exact": _float_equal(preflight["maximum_oracle_phase"], 1.1309733552923253),
        "oracle_phase_below_pi": preflight["maximum_oracle_phase"] < math.pi,
        "oracle_phase_ratio_recomputed": _float_equal(phase_ratio, preflight["oracle_phase_to_pi"]),
    }
    recomputed_pass = all(checks.values())
    return {
        "checks": checks,
        "structure_checks": structures,
        "stored_pass_matches": preflight["passed"] == recomputed_pass,
        "passed": recomputed_pass,
    }


def verify_oracle(result: dict[str, Any]) -> dict[str, Any]:
    preflight = result["oracle_preflight"]
    records = preflight["records"]
    closure = preflight["e11_projection_closure"]
    closure_records = closure["records"]
    oracle_maxima = {
        "maximum_combined_semigroup_coefficient_nrmse": max(
            record["combined_semigroup_coefficient_nrmse"] for record in records
        ),
        "maximum_combined_splitting_decoded_nrmse": max(
            record["combined_splitting_decoded_nrmse"] for record in records
        ),
        "maximum_one_step_decoded_nrmse": max(
            record["one_step_decoded_nrmse"] for record in records
        ),
        "maximum_eight_step_decoded_nrmse": max(
            record["eight_step_decoded_nrmse"] for record in records
        ),
    }
    closure_maxima = {
        "maximum_semigroup_composition_error": max(
            record["semigroup_composition_error"] for record in closure_records
        ),
        "maximum_truth_to_projection_decoded_nrmse": max(
            max(record["one_step_decoded_nrmse"], record["eight_step_decoded_nrmse"])
            for record in closure_records
        ),
        "minimum_projection_rank": min(record["design"]["rank"] for record in closure_records),
    }
    checks = {
        "oracle_case_order": tuple(record["case"] for record in records) == EXPECTED_CASE_NAMES,
        "closure_case_order": (
            tuple(record["case"] for record in closure_records) == EXPECTED_CASE_NAMES
        ),
        "oracle_record_count": len(records) == preflight["parameter_cases"] == 18,
        "closure_record_count": len(closure_records) == closure["parameter_cases"] == 18,
        "oracle_projection_rank": all(record["projection_rank"] == 52 for record in records),
        "closure_projection_rank": all(
            record["design"]["rank"] == 52 for record in closure_records
        ),
        "oracle_maxima_recomputed": all(
            _float_equal(preflight[name], value) for name, value in oracle_maxima.items()
        ),
        "closure_maxima_recomputed": all(
            _float_equal(closure[name], value) for name, value in closure_maxima.items()
        ),
        "oracle_thresholds": max(oracle_maxima.values()) <= 1e-10,
        "closure_thresholds": (
            closure_maxima["maximum_semigroup_composition_error"] <= 1e-10
            and closure_maxima["maximum_truth_to_projection_decoded_nrmse"] <= 0.01
        ),
        "oracle_structure": _structure_passes(preflight["oracle_structure"], 0),
        "learned_initial_structure": _structure_passes(
            preflight["learned_initial_structure"], 2304
        ),
        "active_basis_vectors": preflight["active_basis_vectors"] == 49,
        "closure_dimensions": (
            closure["active_basis_vectors"] == 49 and closure["inactive_trend_vectors"] == 3
        ),
        "record_finiteness": _all_finite(records) and _all_finite(closure_records),
        "closure_leaf_flags": all(
            record["errors_finite"] and record["projected_coefficients_finite"]
            for record in closure_records
        ),
    }
    recomputed_pass = all(checks.values())
    stored_consistency = all(
        (
            preflight["passed"] == recomputed_pass,
            preflight["all_finite"] is True,
            closure["passed"] is True,
            closure["all_errors_finite"] is True,
            closure["all_projected_coefficients_finite"] is True,
        )
    )
    return {
        "checks": checks,
        "oracle_maxima": oracle_maxima,
        "closure_maxima": closure_maxima,
        "stored_pass_matches": stored_consistency,
        "passed": recomputed_pass,
    }


def verify_replay_and_e12_reproduction(result: dict[str, Any]) -> dict[str, Any]:
    replay = result["replay_lock"]
    reproduction = result["e12_reproduction"]
    reproduction_observed = {
        key: reproduction[key] for key in EXPECTED_E12_REPRODUCTION if key != "training"
    }
    reproduction_observed["training"] = reproduction["training"]
    reproduction_checks = {
        "initial_model_sha256": (
            reproduction["initial_model_sha256"]
            == EXPECTED_E12_REPRODUCTION["initial_model_sha256"]
        ),
        "elementary_model_sha256": (
            reproduction["elementary_model_sha256"]
            == EXPECTED_E12_REPRODUCTION["elementary_model_sha256"]
        ),
        "generator_sha256": (
            reproduction["generator_sha256"] == EXPECTED_E12_REPRODUCTION["generator_sha256"]
        ),
        "first_loss": (
            reproduction["training"]["first_loss"]
            == EXPECTED_E12_REPRODUCTION["training"]["first_loss"]
        ),
        "final_loss": (
            reproduction["training"]["final_loss"]
            == EXPECTED_E12_REPRODUCTION["training"]["final_loss"]
        ),
        "updates": (
            reproduction["training"]["updates"] == EXPECTED_E12_REPRODUCTION["training"]["updates"]
        ),
        "examples_per_regime": (
            reproduction["training"]["examples_per_regime"]
            == EXPECTED_E12_REPRODUCTION["training"]["examples_per_regime"]
        ),
        "training_rule": (
            reproduction["training"]["training_rule"]
            == EXPECTED_E12_REPRODUCTION["training"]["training_rule"]
        ),
        "basis_action": (
            reproduction["basis_action_decoded_nrmse"]
            == EXPECTED_E12_REPRODUCTION["basis_action_decoded_nrmse"]
        ),
        "zero_shot_rollout": (
            reproduction["zero_shot_rollout_decoded_nrmse"]
            == EXPECTED_E12_REPRODUCTION["zero_shot_rollout_decoded_nrmse"]
        ),
    }
    replay_checks = {
        "dataset_records_exact": replay["dataset_records"] == EXPECTED_DATASET_RECORDS,
        "expected_dataset_records_exact": (
            replay["expected_dataset_records"] == EXPECTED_DATASET_RECORDS
        ),
        "schedule_records_exact": replay["schedule_records"] == EXPECTED_SCHEDULE_RECORDS,
        "expected_schedule_records_exact": (
            replay["expected_schedule_records"] == EXPECTED_SCHEDULE_RECORDS
        ),
    }
    recomputed_reproduction_pass = all(reproduction_checks.values())
    recomputed_replay_pass = all(replay_checks.values())
    stored_consistency = all(
        (
            reproduction["checks"]
            == {
                "initial_model_sha256": reproduction_checks["initial_model_sha256"],
                "elementary_model_sha256": reproduction_checks["elementary_model_sha256"],
                "generator_sha256": reproduction_checks["generator_sha256"],
                "first_loss": reproduction_checks["first_loss"],
                "final_loss": reproduction_checks["final_loss"],
                "updates": reproduction_checks["updates"],
                "examples_per_regime": reproduction_checks["examples_per_regime"],
                "basis_action": reproduction_checks["basis_action"],
                "zero_shot_rollout": reproduction_checks["zero_shot_rollout"],
            },
            reproduction["passed"] == recomputed_reproduction_pass,
            replay["datasets_match"]
            == (
                replay_checks["dataset_records_exact"]
                and replay_checks["expected_dataset_records_exact"]
            ),
            replay["schedules_match"]
            == (
                replay_checks["schedule_records_exact"]
                and replay_checks["expected_schedule_records_exact"]
            ),
            replay["passed"] == recomputed_replay_pass,
        )
    )
    return {
        "reproduction_checks": reproduction_checks,
        "replay_checks": replay_checks,
        "observed_reproduction": reproduction_observed,
        "stored_pass_matches": stored_consistency,
        "reproduction_passed": recomputed_reproduction_pass,
        "replay_passed": recomputed_replay_pass,
        "passed": recomputed_reproduction_pass and recomputed_replay_pass,
    }


def verify_excitation(result: dict[str, Any]) -> dict[str, Any]:
    excitation = result["excitation"]
    covariance = {
        name: _rank_from_record(record) for name, record in excitation["input_covariance"].items()
    }
    planes = {
        name: [_rank_from_record(record) for record in records]
        for name, records in excitation["rotation_plane_grams"].items()
    }
    jacobian = _rank_from_record(excitation["mode_tied_oracle_jacobian"])
    all_rank_records = [
        *covariance.values(),
        *(report for records in planes.values() for report in records),
        jacobian,
    ]
    integrity_checks = {
        "covariance_groups": set(covariance) == set(ELEMENTARY_REGIMES),
        "covariance_shapes": all(
            excitation["input_covariance"][name]["shape"] == [48, 48] for name in covariance
        ),
        "rotation_groups": set(planes) == {"A_x", "A_y"},
        "twenty_one_planes_per_group": all(len(records) == 21 for records in planes.values()),
        "plane_shapes": all(
            record["shape"] == [2, 2]
            for records in excitation["rotation_plane_grams"].values()
            for record in records
        ),
        "jacobian_shape": excitation["mode_tied_oracle_jacobian"]["shape"] == [301056, 12],
        "rank_records_recomputed": all(
            report["rank_matches"] and report["condition_matches"] and report["tolerance_matches"]
            for report in all_rank_records
        ),
    }
    rank_checks = {
        "covariance_rank_48": all(report["rank"] == 48 for report in covariance.values()),
        "plane_rank_2": all(
            report["rank"] == 2 for records in planes.values() for report in records
        ),
        "jacobian_rank_12": jacobian["rank"] == 12,
    }
    record_integrity_passed = all(integrity_checks.values())
    required_rank_full = all(rank_checks.values())
    stored_consistency = all(
        (
            excitation["required_plane_grams_full_rank"]
            == (
                rank_checks["covariance_rank_48"]
                and rank_checks["plane_rank_2"]
                and integrity_checks["rank_records_recomputed"]
            ),
            excitation["mode_tied_jacobian_full_rank"]
            == (rank_checks["jacobian_rank_12"] and integrity_checks["rank_records_recomputed"]),
        )
    )
    return {
        "integrity_checks": integrity_checks,
        "rank_checks": rank_checks,
        "covariance": covariance,
        "rotation_planes": planes,
        "mode_tied_jacobian": jacobian,
        "stored_pass_matches": stored_consistency,
        "record_integrity_passed": record_integrity_passed,
        "required_rank_full": required_rank_full,
        "passed": record_integrity_passed,
    }


def verify_coverage_and_argmax(result: dict[str, Any]) -> dict[str, Any]:
    evaluations = result["evaluations"]
    records = result["mode_resolved"]
    argmax = result["mode_argmax"]
    keys = [
        (
            record["control"],
            record["basis_index"],
            record["case_name"],
            record["horizon"],
        )
        for record in records
    ]
    argmax_checks = {}
    for control in CONTROLS:
        subset = [record for record in records if record["control"] == control]
        argmax_checks[control] = {}
        for metric in MODE_METRICS:
            winner = max(subset, key=lambda record: record[metric])
            stored = argmax[control][metric]
            argmax_checks[control][metric] = all(
                (
                    winner[metric] == stored["value"],
                    winner["basis_index"] == stored["key"]["basis_index"],
                    winner["case_name"] == stored["key"]["case_name"],
                    winner["horizon"] == stored["key"]["horizon"],
                )
            )
    expected_keys = {
        (control, basis_index, case_name, horizon)
        for control in CONTROLS
        for basis_index in range(49)
        for case_name in EXPECTED_CASE_NAMES
        for horizon in (1, 8)
    }
    checks = {
        "control_names": set(evaluations) == set(CONTROLS),
        "identification_cells": len(evaluations) == 6,
        "validation_cells": sum(len(value["validation"]) for value in evaluations.values()) == 24,
        "validation_regimes": all(
            set(value["validation"]) == set(REGIMES) for value in evaluations.values()
        ),
        "mode_records": len(records) == 10584,
        "unique_mode_keys": len(set(keys)) == len(keys) == 10584,
        "exact_cartesian": set(keys) == expected_keys,
        "exact_case_names": {record["case_name"] for record in records} == set(EXPECTED_CASE_NAMES),
        "argmax_shape": (
            set(argmax) == set(CONTROLS)
            and all(set(argmax[control]) == set(MODE_METRICS) for control in CONTROLS)
        ),
        "argmax_values_and_identities": all(
            passed
            for control_checks in argmax_checks.values()
            for passed in control_checks.values()
        ),
    }
    recomputed_pass = all(checks.values())
    stored = result["coverage"]
    stored_consistency = all(
        (
            stored["generator_identification_cells"] == 6,
            stored["validation_cells"] == 24,
            stored["mode_resolved_records"] == 10584,
            stored["unique_mode_resolved_keys"] == 10584,
            stored["mode_argmax_cells"] == 30,
            stored["recovery_training_controls"] == 5,
            stored["passed"] == recomputed_pass,
        )
    )
    return {
        "checks": checks,
        "argmax_checks": argmax_checks,
        "stored_pass_matches": stored_consistency,
        "passed": recomputed_pass,
    }


def recompute_recovery_pass(
    control: str, evaluation: dict[str, Any]
) -> tuple[bool, dict[str, bool]]:
    identification = evaluation["generator_identification"]
    validation = evaluation["validation"]
    composite = validation["composite"]
    gates = {
        "structure": _structure_passes(evaluation["structure"], EXPECTED_PARAMETER_COUNTS[control]),
        "finite": _all_finite(evaluation),
        "generator_identification": all(
            (
                max(identification["relative_frobenius"].values()) <= 0.10,
                max(identification["maximum_supported_entry_relative_error"].values()) <= 0.20,
                max(identification["off_support_leakage"].values()) <= 0.10,
                identification["maximum_diffusion_rate_relative_error"] <= 0.20,
                max(identification["normalized_commutators"].values()) <= 0.02,
                identification["maximum_basis_action_decoded_nrmse"] <= 0.05,
            )
        ),
        "high_frequency": composite["final_high_frequency_spectral"]["nrmse"] <= 0.15,
        "elementary_one_step_nonregression": (
            max(validation[regime]["one_step_decoded_nrmse"] for regime in ELEMENTARY_REGIMES)
            <= 0.03
        ),
        "elementary_rollout_nonregression": (
            max(validation[regime]["rollout_decoded_nrmse"] for regime in ELEMENTARY_REGIMES)
            <= 0.08
        ),
        "zero_shot_rollout": composite["rollout_decoded_nrmse"] <= 0.20,
        "zero_shot_to_persistence": (
            composite["rollout_decoded_nrmse"] / PERSISTENCE_ROLLOUT <= 0.75
        ),
    }
    return all(gates.values()), gates


def classification_from_recomputed(
    preflight_passed: bool,
    e12_reproduction_passed: bool,
    recovery: dict[str, bool],
    excitation_passed: bool,
) -> str:
    if not preflight_passed:
        return "e13_preflight_failed"
    if not e12_reproduction_passed:
        return "e12_reproduction_failed"
    if recovery["full_skew_lbfgs_neutral"] or recovery["full_skew_lbfgs_polish"]:
        return "full_parameterization_deterministic_recovery_succeeds"
    if recovery["support_sparse_lbfgs"]:
        return "support_restriction_required_under_frozen_solvers"
    if recovery["mode_tied_lbfgs"]:
        return "mode_tying_required_under_frozen_solvers"
    if not excitation_passed:
        return "elementary_excitation_rank_deficient"
    return "recovery_controls_not_qualified"


def verify_recovery_and_classification(
    result: dict[str, Any],
    *,
    preflight_passed: bool,
    e12_reproduction_passed: bool,
    excitation_passed: bool,
) -> dict[str, Any]:
    recovery = {}
    for control in CONTROLS:
        evaluation = result["evaluations"][control]
        passed, gates = recompute_recovery_pass(control, evaluation)
        recovery[control] = {
            "recomputed_pass": passed,
            "stored_pass": evaluation["recovery_pass"],
            "gates": gates,
            "stored_gates_match": gates == evaluation["gates"],
        }
    classification = classification_from_recomputed(
        preflight_passed,
        e12_reproduction_passed,
        {control: record["recomputed_pass"] for control, record in recovery.items()},
        excitation_passed,
    )
    checks = {
        "stored_recovery_passes": all(
            record["recomputed_pass"] == record["stored_pass"] for record in recovery.values()
        ),
        "stored_recovery_gates": all(record["stored_gates_match"] for record in recovery.values()),
        "classification_matches_stored": classification == result["classification"],
        "expected_classification": classification == EXPECTED_UNDERLYING_CLASSIFICATION,
    }
    return {
        "controls": recovery,
        "recomputed_classification": classification,
        "checks": checks,
        "passed": all(checks.values()),
    }


def verify_eight_step_caveat(result: dict[str, Any]) -> dict[str, Any]:
    expected = {
        "full_skew_lbfgs_neutral": {
            "basis_index": 41,
            "case_name": "composite_c",
            "horizon": 8,
            "value": 0.3595701757249822,
        },
        "full_skew_lbfgs_polish": {
            "basis_index": 41,
            "case_name": "composite_c",
            "horizon": 8,
            "value": 0.37147804128286543,
        },
    }
    records = {}
    for control, target in expected.items():
        stored = result["mode_argmax"][control]["decoded_nrmse"]
        records[control] = {
            "basis_index": stored["key"]["basis_index"],
            "case_name": stored["key"]["case_name"],
            "horizon": stored["key"]["horizon"],
            "decoded_nrmse": stored["value"],
            "matches": (
                stored["key"]["basis_index"] == target["basis_index"]
                and stored["key"]["case_name"] == target["case_name"]
                and stored["key"]["horizon"] == target["horizon"]
                and stored["value"] == target["value"]
            ),
        }
    return {
        "records": records,
        "interpretation": (
            "relative error on a strongly attenuated maximum-diffusion target; "
            "recorded but not gated by the frozen E13 contract"
        ),
        "passed": all(record["matches"] for record in records.values()),
    }


def classification_inputs_from_sections(
    *,
    provenance: dict[str, Any],
    parameterization: dict[str, Any],
    oracle: dict[str, Any],
    replay: dict[str, Any],
    excitation: dict[str, Any],
    coverage: dict[str, Any],
) -> dict[str, bool]:
    return {
        "preflight_passed": all(
            (
                provenance["passed"],
                parameterization["passed"],
                oracle["passed"],
                excitation["record_integrity_passed"],
                coverage["passed"],
            )
        ),
        "e12_reproduction_passed": (replay["replay_passed"] and replay["reproduction_passed"]),
        "excitation_required_rank_full": excitation["required_rank_full"],
    }


def independent_recompute(result: dict[str, Any]) -> dict[str, Any]:
    provenance = verify_provenance(result)
    parameterization = verify_parameterization(result)
    oracle = verify_oracle(result)
    replay = verify_replay_and_e12_reproduction(result)
    excitation = verify_excitation(result)
    coverage = verify_coverage_and_argmax(result)
    classification_inputs = classification_inputs_from_sections(
        provenance=provenance,
        parameterization=parameterization,
        oracle=oracle,
        replay=replay,
        excitation=excitation,
        coverage=coverage,
    )
    recovery = verify_recovery_and_classification(
        result,
        preflight_passed=classification_inputs["preflight_passed"],
        e12_reproduction_passed=classification_inputs["e12_reproduction_passed"],
        excitation_passed=classification_inputs["excitation_required_rank_full"],
    )
    caveat = verify_eight_step_caveat(result)
    sections = {
        "provenance": provenance,
        "parameterization": parameterization,
        "oracle": oracle,
        "replay_and_e12_reproduction": replay,
        "excitation": excitation,
        "coverage_and_argmax": coverage,
        "recovery_and_classification": recovery,
        "eight_step_caveat": caveat,
    }
    return {
        **sections,
        "classification_inputs": classification_inputs,
        "recomputed_preflight_passed": classification_inputs["preflight_passed"],
        "passed": all(section["passed"] for section in sections.values()),
    }


def deterministic_bundle(raw: dict[str, bytes]) -> tuple[bytes, list[dict[str, Any]]]:
    member_sources = tuple((name, raw[name]) for name in EXPECTED_RAW_HASHES)
    output = io.BytesIO()
    with gzip.GzipFile(
        filename="",
        mode="wb",
        compresslevel=9,
        fileobj=output,
        mtime=0,
    ) as compressed:
        with tarfile.open(fileobj=compressed, mode="w", format=tarfile.GNU_FORMAT) as archive:
            for name, value in member_sources:
                info = tarfile.TarInfo(name)
                info.size = len(value)
                info.mode = 0o644
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                info.mtime = 0
                info.pax_headers = {}
                archive.addfile(info, io.BytesIO(value))
    members = [
        {"name": name, "bytes": len(value), "raw_sha256": sha256_bytes(value)}
        for name, value in member_sources
    ]
    return output.getvalue(), members


def verify_bundle(bundle: bytes, expected_members: list[dict[str, Any]]) -> None:
    if bundle[:3] != b"\x1f\x8b\x08":
        raise RuntimeError("E14 bundle is not gzip")
    if bundle[3] & 0x08:
        raise RuntimeError("E14 gzip unexpectedly stores an original filename")
    if bundle[4:8] != b"\x00\x00\x00\x00":
        raise RuntimeError("E14 gzip mtime is not zero")
    with tarfile.open(fileobj=io.BytesIO(bundle), mode="r:gz") as archive:
        members = archive.getmembers()
        if [member.name for member in members] != [record["name"] for record in expected_members]:
            raise RuntimeError("E14 bundle member order or names changed")
        for member, expected in zip(members, expected_members, strict=True):
            if not all(
                (
                    member.isfile(),
                    member.pax_headers == {},
                    member.mode == 0o644,
                    member.uid == 0,
                    member.gid == 0,
                    member.uname == "",
                    member.gname == "",
                    member.mtime == 0,
                    member.size == expected["bytes"],
                )
            ):
                raise RuntimeError(f"E14 bundle metadata changed for {member.name}")
            extracted = archive.extractfile(member)
            if extracted is None:
                raise RuntimeError(f"E14 bundle member unreadable: {member.name}")
            member_bytes = extracted.read()
            if len(member_bytes) != expected["bytes"]:
                raise RuntimeError(f"E14 bundle member byte count changed: {member.name}")
            if sha256_bytes(member_bytes) != expected["raw_sha256"]:
                raise RuntimeError(f"E14 bundle member hash changed: {member.name}")


def _key_metrics(result: dict[str, Any], recompute: dict[str, Any]) -> dict[str, Any]:
    records = {}
    for control in CONTROLS:
        evaluation = result["evaluations"][control]
        records[control] = {
            "recomputed_recovery_pass": recompute["recovery_and_classification"]["controls"][
                control
            ]["recomputed_pass"],
            "maximum_basis_action_decoded_nrmse": evaluation["generator_identification"][
                "maximum_basis_action_decoded_nrmse"
            ],
            "composite_rollout_decoded_nrmse": evaluation["validation"]["composite"][
                "rollout_decoded_nrmse"
            ],
            "composite_high_frequency_nrmse": evaluation["validation"]["composite"][
                "final_high_frequency_spectral"
            ]["nrmse"],
        }
    return records


def build_output_bytes(
    *,
    run_dir: Path,
    raw: dict[str, bytes],
    inputs: dict[str, Any],
    result: dict[str, Any],
    recompute: dict[str, Any],
    source: dict[str, Any],
) -> tuple[dict[str, bytes], list[dict[str, Any]]]:
    bundle, members = deterministic_bundle(raw)
    verify_bundle(bundle, members)
    bundle_record = {
        "filename": OUTPUT_NAMES["bundle"],
        "bytes": len(bundle),
        "raw_sha256": sha256_bytes(bundle),
        "members": members,
    }
    boundary = {
        "state_reads": 0,
        "optimizer_updates": 0,
        "heldout_reads": 0,
        "provider_calls": 0,
        "routing_paths": 0,
        "seal_does_not_modify_e13": True,
        "synthetic_periodic_scalar_linear_only": True,
        "nonlinear_dynamics_qualified": False,
        "particle_dynamics_qualified": False,
        "public_or_claim_grade": False,
    }
    compact = {
        "schema_version": 1,
        "experiment": "canonical_latent_e14_evidence_seal",
        "classification": SEALED_CLASSIFICATION,
        "underlying_e13_classification": result["classification"],
        "e13_original_evidence_status": E13_ORIGINAL_EVIDENCE_STATUS,
        "seal_does_not_modify_e13": True,
        "e13_execution_head": result["provenance"]["git_head"],
        "e14_sealing_head": source["git_head"],
        "sealed_raw_input_hashes": inputs["raw_hashes"],
        "inputs": {
            "run_directory": str(run_dir),
            "raw_file_sha256": inputs["raw_hashes"],
            "canonical_payload_sha256": inputs["canonical_hashes"],
            "raw_replicates_byte_identical": inputs["raw_replicates_byte_identical"],
            "parsed_replicates_equal": inputs["parsed_replicates_equal"],
            "combined_extends_replicate": inputs["combined_extends_replicate"],
            "known_mislabeled_e13_replication_fields": inputs[
                "embedded_mislabeled_replication_hashes"
            ],
        },
        "independent_recompute": recompute,
        "key_metrics": _key_metrics(result, recompute),
        "evidence_bundle": bundle_record,
        "e14_boundary": boundary,
        "interpretation_boundary": (
            "E14 separately seals immutable E13 bytes; it does not retroactively make "
            "the original E13 output contract-complete or qualify routing, nonlinear, "
            "particle, public, or claim-grade behavior."
        ),
    }
    compact_bytes = pretty_bytes(compact)
    raw_file_records = {
        name: {
            "path": str(path),
            "bytes": len(raw[name]),
            "raw_sha256": sha256_bytes(raw[name]),
        }
        for name, path in _input_paths(run_dir).items()
    }
    manifest = {
        "schema_version": 1,
        "experiment": "canonical_latent_e14_evidence_seal",
        "classification": SEALED_CLASSIFICATION,
        "underlying_e13_classification": result["classification"],
        "e13_original_evidence_status": E13_ORIGINAL_EVIDENCE_STATUS,
        "seal_does_not_modify_e13": True,
        "sealed_raw_input_hashes": inputs["raw_hashes"],
        "source_provenance": source,
        "inputs": {
            "run_directory": str(run_dir),
            "raw_files": raw_file_records,
            "canonical_payload_sha256": inputs["canonical_hashes"],
        },
        "outputs": {
            "compact_result": {
                "filename": OUTPUT_NAMES["result"],
                "bytes": len(compact_bytes),
                "raw_sha256": sha256_bytes(compact_bytes),
            },
            "evidence_bundle": bundle_record,
        },
        "boundary": boundary,
        "manifest_self_hash_declared": False,
    }
    manifest_bytes = pretty_bytes(manifest)
    return {
        "bundle": bundle,
        "result": compact_bytes,
        "manifest": manifest_bytes,
    }, members


def verify_seal_files(
    file_bytes: dict[str, bytes],
    expected_members: list[dict[str, Any]],
) -> None:
    if set(file_bytes) != set(OUTPUT_NAMES):
        raise RuntimeError("E14 output byte set changed")
    manifest = json.loads(file_bytes["manifest"])
    compact = json.loads(file_bytes["result"])
    if manifest["manifest_self_hash_declared"] is not False:
        raise RuntimeError("E14 manifest unexpectedly declares a self hash")
    if manifest["classification"] != SEALED_CLASSIFICATION:
        raise RuntimeError("E14 manifest classification changed")
    if compact["classification"] != SEALED_CLASSIFICATION:
        raise RuntimeError("E14 compact-result classification changed")
    expected_raw_hashes = {member["name"]: member["raw_sha256"] for member in expected_members}
    if manifest["sealed_raw_input_hashes"] != expected_raw_hashes:
        raise RuntimeError("E14 manifest sealed raw input hashes changed")
    if compact["sealed_raw_input_hashes"] != expected_raw_hashes:
        raise RuntimeError("E14 compact-result sealed raw input hashes changed")
    for record in (manifest, compact):
        if record["e13_original_evidence_status"] != E13_ORIGINAL_EVIDENCE_STATUS:
            raise RuntimeError("E14 non-retroactive E13 evidence status changed")
        if record["seal_does_not_modify_e13"] is not True:
            raise RuntimeError("E14 immutable-E13 declaration changed")
    if manifest["boundary"]["seal_does_not_modify_e13"] is not True:
        raise RuntimeError("E14 manifest boundary changed")
    if compact["e14_boundary"]["seal_does_not_modify_e13"] is not True:
        raise RuntimeError("E14 compact-result boundary changed")
    for key, filename in OUTPUT_NAMES.items():
        if key == "manifest":
            continue
        manifest_key = "evidence_bundle" if key == "bundle" else "compact_result"
        record = manifest["outputs"][manifest_key]
        if record["filename"] != filename:
            raise RuntimeError(f"E14 manifest filename mismatch for {key}")
        if record["bytes"] != len(file_bytes[key]):
            raise RuntimeError(f"E14 manifest byte count mismatch for {key}")
        if record["raw_sha256"] != sha256_bytes(file_bytes[key]):
            raise RuntimeError(f"E14 manifest hash mismatch for {key}")
    if manifest["outputs"]["evidence_bundle"]["members"] != expected_members:
        raise RuntimeError("E14 manifest archive member records changed")
    if compact["evidence_bundle"] != manifest["outputs"]["evidence_bundle"]:
        raise RuntimeError("E14 compact and manifest bundle declarations differ")
    raw_files = manifest["inputs"]["raw_files"]
    if set(raw_files) != {record["name"] for record in expected_members}:
        raise RuntimeError("E14 manifest raw input set changed")
    for member in expected_members:
        record = raw_files[member["name"]]
        if record["bytes"] != member["bytes"]:
            raise RuntimeError(f"E14 manifest input byte count mismatch for {member['name']}")
        if record["raw_sha256"] != member["raw_sha256"]:
            raise RuntimeError(f"E14 manifest input hash mismatch for {member['name']}")
    verify_bundle(file_bytes["bundle"], expected_members)


def _write_fsync(path: Path, value: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _inject(
    failure_point: str | None,
    point: str,
    hook: Callable[[str], None] | None = None,
) -> None:
    if hook is not None:
        hook(point)
    if failure_point == point:
        raise RuntimeError(f"injected E14 publication failure: {point}")


def _entry_exists(path: Path) -> bool:
    return os.path.lexists(path)


def publish_atomic(
    output_dir: Path,
    file_bytes: dict[str, bytes],
    expected_members: list[dict[str, Any]],
    *,
    failure_point: str | None = None,
    hook: Callable[[str], None] | None = None,
) -> None:
    parent = output_dir.parent
    if not parent.is_dir():
        raise RuntimeError("E14 output parent directory must already exist")
    lock_path = parent / f".{output_dir.name}.publish.lock"
    try:
        lock_descriptor = os.open(
            lock_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
    except FileExistsError as error:
        raise RuntimeError("E14 publication lock already exists") from error
    stage: Path | None = None
    renamed = False
    errors: list[BaseException] = []
    lock_closed = False
    try:
        _fsync_directory(parent)
        if _entry_exists(output_dir):
            raise RuntimeError("E14 final output directory entry must be absent")
        stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=parent))
        for key in OUTPUT_ORDER:
            _write_fsync(stage / OUTPUT_NAMES[key], file_bytes[key])
            _inject(failure_point, f"after_write_{key}", hook)
        staged_names = {path.name for path in stage.iterdir()}
        if staged_names != set(OUTPUT_NAMES.values()):
            raise RuntimeError("E14 staged output set changed")
        reopened = {}
        for key in OUTPUT_ORDER:
            reopened[key] = (stage / OUTPUT_NAMES[key]).read_bytes()
            if reopened[key] != file_bytes[key]:
                raise RuntimeError(f"E14 staged reopen mismatch for {key}")
            _inject(failure_point, f"after_reopen_{key}", hook)
        verify_seal_files(reopened, expected_members)
        _inject(failure_point, "after_reopen_verification", hook)
        _fsync_directory(stage)
        _inject(failure_point, "before_final_entry_check", hook)
        if _entry_exists(output_dir):
            raise RuntimeError("E14 final output directory entry appeared during publication")
        os.rename(stage, output_dir)
        renamed = True
        _inject(failure_point, "after_rename", hook)
        _fsync_directory(parent)
    except BaseException as error:
        errors.append(error)

    try:
        _inject(failure_point, "before_lock_close", hook)
    except BaseException as error:
        errors.append(error)
    try:
        os.close(lock_descriptor)
        lock_closed = True
    except BaseException as error:
        errors.append(error)
    try:
        _inject(failure_point, "after_lock_close", hook)
    except BaseException as error:
        errors.append(error)
    try:
        _inject(failure_point, "before_lock_unlink", hook)
    except BaseException as error:
        errors.append(error)
    try:
        if _entry_exists(lock_path):
            lock_path.unlink()
    except BaseException as error:
        errors.append(error)
    try:
        _inject(failure_point, "after_lock_unlink", hook)
    except BaseException as error:
        errors.append(error)
    try:
        _inject(failure_point, "before_lock_parent_fsync", hook)
    except BaseException as error:
        errors.append(error)
    try:
        _fsync_directory(parent)
    except BaseException as error:
        errors.append(error)
    try:
        _inject(failure_point, "after_lock_parent_fsync", hook)
    except BaseException as error:
        errors.append(error)

    if errors:
        primary = errors[0]
        cleanup = output_dir if renamed else stage
        try:
            if cleanup is not None and _entry_exists(cleanup):
                shutil.rmtree(cleanup)
        except BaseException as cleanup_error:
            primary.add_note(f"E14 cleanup error: {cleanup_error!r}")
        if not lock_closed:
            try:
                os.close(lock_descriptor)
            except BaseException as cleanup_error:
                primary.add_note(f"E14 lock-close cleanup error: {cleanup_error!r}")
        try:
            if _entry_exists(lock_path):
                lock_path.unlink()
        except BaseException as cleanup_error:
            primary.add_note(f"E14 lock-unlink cleanup error: {cleanup_error!r}")
        try:
            _fsync_directory(parent)
        except BaseException as cleanup_error:
            primary.add_note(f"E14 parent-fsync cleanup error: {cleanup_error!r}")
        raise primary


def _path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def seal(
    run_dir: Path,
    output_dir: Path,
    *,
    failure_point: str | None = None,
    hook: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    source = source_preflight()
    if not source["passed"]:
        raise RuntimeError(f"E14 source preflight failed: {source}")
    resolved_run_dir = run_dir.resolve()
    resolved_output_dir = output_dir.resolve()
    if resolved_output_dir == resolved_run_dir or _path_is_within(
        resolved_output_dir, resolved_run_dir
    ):
        raise RuntimeError("E14 output directory must be outside the E13 input directory")
    if _entry_exists(output_dir):
        raise RuntimeError("E14 final output directory must be absent")

    raw, inputs = read_frozen_inputs(run_dir)
    result = inputs["parsed"]["replicate_a_result.json"]
    recompute = independent_recompute(result)
    if not recompute["passed"]:
        raise RuntimeError("E14 independent E13 recomputation failed")
    files, members = build_output_bytes(
        run_dir=run_dir,
        raw=raw,
        inputs=inputs,
        result=result,
        recompute=recompute,
        source=source,
    )
    verify_seal_files(files, members)
    publish_atomic(
        output_dir,
        files,
        members,
        failure_point=failure_point,
        hook=hook,
    )
    return {
        "classification": SEALED_CLASSIFICATION,
        "e13_original_evidence_status": E13_ORIGINAL_EVIDENCE_STATUS,
        "seal_does_not_modify_e13": True,
        "output_directory": str(output_dir),
        "compact_result": str(output_dir / OUTPUT_NAMES["result"]),
        "manifest": str(output_dir / OUTPUT_NAMES["manifest"]),
        "bundle": str(output_dir / OUTPUT_NAMES["bundle"]),
        "compact_result_sha256": sha256_bytes(files["result"]),
        "manifest_sha256": sha256_bytes(files["manifest"]),
        "bundle_sha256": sha256_bytes(files["bundle"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e13-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sealed = seal(args.e13_run_dir, args.output_dir)
    print(json.dumps(sealed, sort_keys=True))


if __name__ == "__main__":
    main()
