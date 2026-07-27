#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import gzip
import io
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch

from scripts.run_canonical_latent_e7_function_space import (
    FunctionSpaceConfig,
    PhysicalFunctionSpace,
)
from scripts.run_canonical_latent_e11_coefficient_operator_transfer import (
    REGIMES,
    TrajectorySet,
    build_trajectories,
    canonical_grid,
    model_hash,
    schedule,
)
from scripts.run_canonical_latent_e12_structured_generator import (
    FixedGenerator,
    StructuredGenerator,
    _all_finite,
    _matrix_hashes,
    oracle_generators,
    oracle_preflight,
    train_elementary,
)
from scripts.run_canonical_latent_e13_identifiability_audit import (
    evaluate_control,
    excitation_report,
    mode_resolved_records,
)
from scripts.run_canonical_latent_e15_training_package import (
    E15Config,
    canonical_bytes,
    configure_environment,
    frozen_e15_config,
    objective_integrity,
    optimize_components_weighted,
    pretty_bytes,
    sha256_bytes,
    sha256_path,
    time_limit,
    train_adamw,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = Path(__file__).resolve()
CONTRACT_PATH = (
    REPO_ROOT
    / "docs/research/2026-07-27-canonical-latent-e16-multi-realization-robustness-contract.md"
)
TEST_PATH = REPO_ROOT / "tests/unit/test_run_canonical_latent_e16_multi_realization_robustness.py"
E15_RUNNER_PATH = REPO_ROOT / "scripts/run_canonical_latent_e15_training_package.py"
E15_TEST_PATH = REPO_ROOT / "tests/unit/test_run_canonical_latent_e15_training_package.py"
E15_CONTRACT_PATH = (
    REPO_ROOT / "docs/research/2026-07-26-canonical-latent-e15-training-package-contract.md"
)
E15_RESULT_RECORD_PATH = (
    REPO_ROOT / "docs/research/2026-07-26-canonical-latent-e15-training-package-result.md"
)
E15_ARTIFACT_DIR = REPO_ROOT / "docs/research/artifacts/canonical_latent_e15_training_package"
E15_BUNDLE_PATH = E15_ARTIFACT_DIR / "canonical_latent_e15_training_package_evidence_bundle.tar.gz"
E15_RESULT_PATH = E15_ARTIFACT_DIR / "canonical_latent_e15_training_package_result.json"
E15_MANIFEST_PATH = E15_ARTIFACT_DIR / "canonical_latent_e15_training_package_manifest.json"
E13_RUNNER_PATH = REPO_ROOT / "scripts/run_canonical_latent_e13_identifiability_audit.py"
E12_RUNNER_PATH = REPO_ROOT / "scripts/run_canonical_latent_e12_structured_generator.py"
E11_RUNNER_PATH = REPO_ROOT / "scripts/run_canonical_latent_e11_coefficient_operator_transfer.py"
E7_RUNNER_PATH = REPO_ROOT / "scripts/run_canonical_latent_e7_function_space.py"
LATENT_EVAL_PATH = REPO_ROOT / "src/ups/eval/latent_qualification.py"

EXPECTED_E15_HASHES = {
    "bundle": "3347ec66843ed51e30a36996335915221407c979b64afa13b96f9ee0d76b618a",
    "result": "e3b91ecc792085f45e6b80bd970cb6da15fb869a7a49e8fec4feb782b919768d",
    "manifest": "1208b5e5158f9c2ff0ae0dd5ab310ec5967cfdc7bc5d0ab131e8c0387effd311",
}
EXPECTED_E15_SOURCE_HASHES = {
    "contract": "5a3c826d29d65de549fbcfb4a186f6dcced7fd39c51e5169c4d5217d04bdbece",
    "runner": "943558c42d2e8a13879fc3fe6f1301142efe7c7949f51e7e4ff509a6af6ae9ca",
    "tests": "191640579e037c1e0165c1a736b0991ebbab762f59ee0d2c4c1ac0ef6cc2c8d2",
}
EXPECTED_LIVE_INHERITED_HASHES = {
    "e15_contract": EXPECTED_E15_SOURCE_HASHES["contract"],
    "e15_runner": EXPECTED_E15_SOURCE_HASHES["runner"],
    "e15_tests": EXPECTED_E15_SOURCE_HASHES["tests"],
    "e13_runner": "f95b1e50f409fc939c62120d06f2eafa89a864de148f9309fb38d481c01310c2",
    "e12_runner": "8edb67652d53e101a63730b9ec4803a69067572a8bab6eee0fb98627785a926a",
    "e11_runner": "720e2ad33b92faee49fcfbdee84c66c023b40bf1f50427f874f231ab555483eb",
    "e7_runner": "cf81597b3909e9693508b62e595eb006a8598d186de062eaf4a8f241d4b07488",
    "latent_evaluation": "e2bb0fb86ac464aa6b96221d706f71aad4fd8fb48992613ead6d5b94e1943994",
}
EXPECTED_E15_MEMBERS = (
    {
        "name": "replicate_a/result.json",
        "bytes": 7_037_400,
        "raw_sha256": "f2cc65ecea260f67adce89413cf148b8cae9ee51899e5adba0661c980d30ceed",
    },
    {
        "name": "replicate_b/result.json",
        "bytes": 7_037_400,
        "raw_sha256": "f2cc65ecea260f67adce89413cf148b8cae9ee51899e5adba0661c980d30ceed",
    },
    {
        "name": "complete_result.json",
        "bytes": 7_038_050,
        "raw_sha256": "ba2294304bdb8f97add96e4a6d39869f044a90067df41c4d682608a5d3820429",
    },
)
EXPECTED_E15_CANONICAL_REPLICATE_SHA256 = (
    "eec9fbc5bca20fc7c94600217dd13cc55d18fc779cae2b4990e0aed2be191758"
)
EXPECTED_E15_CLASSIFICATION = "deterministic_objective_adamw_restart_repairs_e12_checkpoint_only"
PACKAGES = ("deterministic_adamw_restart", "componentwise_lbfgs_neutral")
E15_CONTROL_BY_PACKAGE = {
    "deterministic_adamw_restart": "schedule_weighted_adamw_restart",
    "componentwise_lbfgs_neutral": "schedule_weighted_componentwise_lbfgs_neutral",
}
REALIZATIONS = {
    "r1": {
        "pretrain_state_seed": 151_001,
        "pretrain_parameter_seed": 151_101,
        "schedule_seed": 172_001,
        "literal_schedule_seeds": {
            "x_advection": 172_001,
            "y_advection": 172_002,
            "diffusion": 172_003,
        },
    },
    "r2": {
        "pretrain_state_seed": 251_001,
        "pretrain_parameter_seed": 251_101,
        "schedule_seed": 272_001,
        "literal_schedule_seeds": {
            "x_advection": 272_001,
            "y_advection": 272_002,
            "diffusion": 272_003,
        },
    },
}
RAW_MEMBERS = (
    "replicate_a/result.json",
    "replicate_b/result.json",
    "complete_result.json",
)
OUTPUT_NAMES = {
    "bundle": "canonical_latent_e16_multi_realization_robustness_evidence_bundle.tar.gz",
    "result": "canonical_latent_e16_multi_realization_robustness_result.json",
    "manifest": "canonical_latent_e16_multi_realization_robustness_manifest.json",
}
CANONICAL_OUTPUT_DIR = (
    REPO_ROOT / "docs/research/artifacts/canonical_latent_e16_multi_realization_robustness"
)
PER_REPLICA_TIMEOUT_SECONDS = 8 * 60 * 60
WHOLE_EXPERIMENT_TIMEOUT_SECONDS = 18 * 60 * 60


class E16IncompleteError(RuntimeError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _git_text(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _committed_sha256(path: Path) -> str | None:
    relative = path.relative_to(REPO_ROOT).as_posix()
    completed = subprocess.run(
        ["git", "show", f"HEAD:{relative}"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
    )
    return sha256_bytes(completed.stdout) if completed.returncode == 0 else None


def sealed_e15_report() -> dict[str, Any]:
    hashes = {
        "bundle": sha256_path(E15_BUNDLE_PATH),
        "result": sha256_path(E15_RESULT_PATH),
        "manifest": sha256_path(E15_MANIFEST_PATH),
    }
    result = json.loads(E15_RESULT_PATH.read_text())
    manifest = json.loads(E15_MANIFEST_PATH.read_text())
    source_hashes = {
        name: result["source_files"][name]["working_sha256"] for name in EXPECTED_E15_SOURCE_HASHES
    }
    recovery = {
        package: bool(result["evaluations"][control]["recovery_pass"])
        for package, control in E15_CONTROL_BY_PACKAGE.items()
    }
    recovery_gates = {
        package: result["evaluations"][control]["gates"]
        for package, control in E15_CONTROL_BY_PACKAGE.items()
    }
    bundle_record = manifest["outputs"]["evidence_bundle"]
    compact_record = manifest["outputs"]["compact_result"]
    with tarfile.open(E15_BUNDLE_PATH, mode="r:gz") as archive:
        members = archive.getmembers()
        observed_members = []
        for member in members:
            stream = archive.extractfile(member)
            if stream is None:
                raise RuntimeError(f"E15 archive member unreadable: {member.name}")
            value = stream.read()
            observed_members.append(
                {
                    "name": member.name,
                    "bytes": len(value),
                    "raw_sha256": sha256_bytes(value),
                }
            )
    replication = result["raw_and_canonical_hashes"]
    checks = {
        "artifact_hashes": hashes == EXPECTED_E15_HASHES,
        "source_hashes": source_hashes == EXPECTED_E15_SOURCE_HASHES,
        "classification": result["classification"] == EXPECTED_E15_CLASSIFICATION,
        "recovery_bits": all(recovery.values()),
        "recovery_gate_vectors": all(
            len(gates) == 8 and all(gates.values()) for gates in recovery_gates.values()
        ),
        "bundle_members": observed_members == list(EXPECTED_E15_MEMBERS),
        "raw_replicates_identical": (
            replication["raw_replicates_byte_identical"]
            and replication["replicate_a_raw_sha256"] == EXPECTED_E15_MEMBERS[0]["raw_sha256"]
            and replication["replicate_b_raw_sha256"] == EXPECTED_E15_MEMBERS[1]["raw_sha256"]
        ),
        "canonical_replicates_identical": (
            replication["replicate_a_canonical_sha256"] == EXPECTED_E15_CANONICAL_REPLICATE_SHA256
            and replication["replicate_b_canonical_sha256"]
            == EXPECTED_E15_CANONICAL_REPLICATE_SHA256
            and replication["complete_canonical_without_replication_sha256"]
            == EXPECTED_E15_CANONICAL_REPLICATE_SHA256
        ),
        "manifest_classification": manifest["classification"] == result["classification"],
        "manifest_bundle_hash": bundle_record["raw_sha256"] == hashes["bundle"],
        "manifest_bundle_bytes": bundle_record["bytes"] == E15_BUNDLE_PATH.stat().st_size,
        "manifest_result_hash": compact_record["raw_sha256"] == hashes["result"],
        "manifest_result_bytes": compact_record["bytes"] == E15_RESULT_PATH.stat().st_size,
        "heldout_zero": result["state_reads"]["heldout"] == 0,
    }
    return {
        "hashes": hashes,
        "expected_hashes": EXPECTED_E15_HASHES,
        "source_hashes": source_hashes,
        "expected_source_hashes": EXPECTED_E15_SOURCE_HASHES,
        "members": observed_members,
        "expected_members": EXPECTED_E15_MEMBERS,
        "classification": result["classification"],
        "expected_classification": EXPECTED_E15_CLASSIFICATION,
        "recovery": recovery,
        "recovery_gates": recovery_gates,
        "replication": replication,
        "checks": checks,
        "passed": all(checks.values()),
    }


def provenance(environment: dict[str, Any]) -> dict[str, Any]:
    sources = {
        "contract": CONTRACT_PATH,
        "runner": RUNNER_PATH,
        "tests": TEST_PATH,
        "e15_contract": E15_CONTRACT_PATH,
        "e15_runner": E15_RUNNER_PATH,
        "e15_tests": E15_TEST_PATH,
        "e15_result_record": E15_RESULT_RECORD_PATH,
        "e15_bundle": E15_BUNDLE_PATH,
        "e15_result": E15_RESULT_PATH,
        "e15_manifest": E15_MANIFEST_PATH,
        "e13_runner": E13_RUNNER_PATH,
        "e12_runner": E12_RUNNER_PATH,
        "e11_runner": E11_RUNNER_PATH,
        "e7_runner": E7_RUNNER_PATH,
        "latent_evaluation": LATENT_EVAL_PATH,
    }
    records = {}
    for name, path in sources.items():
        working = sha256_path(path)
        committed = _committed_sha256(path)
        records[name] = {
            "path": path.relative_to(REPO_ROOT).as_posix(),
            "working_sha256": working,
            "committed_sha256": committed,
            "matches_git_head": committed is not None and working == committed,
            "expected_sealed_sha256": EXPECTED_LIVE_INHERITED_HASHES.get(name),
            "matches_sealed_source": (
                working == EXPECTED_LIVE_INHERITED_HASHES[name]
                if name in EXPECTED_LIVE_INHERITED_HASHES
                else None
            ),
        }
    seal = sealed_e15_report()
    report = {
        "git_head": _git_text("rev-parse", "HEAD"),
        "worktree_clean": _git_text("status", "--porcelain") == "",
        "source_files": records,
        "source_files_match_git_head": all(
            record["matches_git_head"] for record in records.values()
        ),
        "sealed_inherited_sources_match": all(
            records[name]["matches_sealed_source"] for name in EXPECTED_LIVE_INHERITED_HASHES
        ),
        "sealed_e15": seal,
        "environment": environment,
    }
    report["passed"] = all(
        (
            len(report["git_head"]) == 40,
            report["worktree_clean"],
            report["source_files_match_git_head"],
            report["sealed_inherited_sources_match"],
            seal["passed"],
            environment["passed"],
        )
    )
    return report


def realization_config(base: E15Config, descriptor: dict[str, Any]) -> E15Config:
    return replace(
        base,
        pretrain_state_seed=descriptor["pretrain_state_seed"],
        pretrain_parameter_seed=descriptor["pretrain_parameter_seed"],
        schedule_seed=descriptor["schedule_seed"],
    )


def build_training(
    cfg: E15Config,
    space: PhysicalFunctionSpace,
) -> dict[str, TrajectorySet]:
    return {
        regime: build_trajectories(
            regime,
            count=cfg.pretrain_trajectories_per_regime,
            state_seed=cfg.pretrain_state_seed,
            parameter_seed=cfg.pretrain_parameter_seed,
            regime=regime,
            cfg=cfg,
            space=space,
        )
        for regime in REGIMES
    }


def build_validation(
    cfg: E15Config,
    space: PhysicalFunctionSpace,
) -> dict[str, TrajectorySet]:
    parameter_seeds = {
        "x_advection": cfg.x_validation_parameter_seed,
        "y_advection": cfg.y_validation_parameter_seed,
        "diffusion": cfg.diffusion_validation_parameter_seed,
    }
    validation = {
        regime: build_trajectories(
            f"{regime}_validation",
            count=cfg.validation_trajectories,
            state_seed=cfg.validation_state_seed,
            parameter_seed=seed,
            regime=regime,
            cfg=cfg,
            space=space,
        )
        for regime, seed in parameter_seeds.items()
    }
    validation["composite"] = build_trajectories(
        "composite_validation",
        count=cfg.validation_trajectories,
        state_seed=cfg.validation_state_seed,
        parameter_seed=cfg.validation_parameter_seed,
        regime="composite",
        cfg=cfg,
        space=space,
    )
    return validation


def elementary_schedules(cfg: E15Config) -> dict[str, torch.Tensor]:
    return {
        regime: schedule(
            cfg.pretrain_updates,
            cfg.pretrain_batch_per_regime,
            cfg.pretrain_trajectories_per_regime * cfg.rollout_steps,
            seed=cfg.schedule_seed + index,
        )
        for index, regime in enumerate(REGIMES)
    }


def canonical_tensor_record(tensor: torch.Tensor, *, kind: str) -> dict[str, Any]:
    value = tensor.detach().cpu().contiguous()
    if kind == "float64":
        if value.dtype != torch.float64:
            raise TypeError("E16 float tensor serialization requires torch.float64")
        raw = value.numpy().astype("<f8", copy=True).tobytes(order="C")
        byte_order = "little_endian_float64"
    elif kind == "int64":
        if value.dtype != torch.int64:
            raise TypeError("E16 integer tensor serialization requires torch.int64")
        raw = value.numpy().astype("<i8", copy=True).tobytes(order="C")
        byte_order = "little_endian_int64"
    else:
        raise ValueError(f"unknown E16 tensor serialization kind: {kind}")
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "c_contiguous": value.is_contiguous(),
        "serialization": f"{byte_order}_c_order",
        "bytes": len(raw),
        "sha256": sha256_bytes(raw),
    }


def canonical_dataset_record(dataset: TrajectorySet) -> dict[str, Any]:
    return {
        "complete_trajectory": canonical_tensor_record(dataset.coefficients, kind="float64"),
        "initial_coefficients": canonical_tensor_record(dataset.coefficients[:, 0], kind="float64"),
        "parameters": canonical_tensor_record(dataset.parameters, kind="float64"),
        "all_finite": bool(
            torch.isfinite(dataset.coefficients).all() and torch.isfinite(dataset.parameters).all()
        ),
    }


def schedule_record(tensor: torch.Tensor) -> dict[str, Any]:
    record = canonical_tensor_record(tensor, kind="int64")
    record["passed"] = all(
        (
            record["shape"] == [1500, 32],
            record["dtype"] == "torch.int64",
            record["c_contiguous"],
            record["bytes"] == 48_000 * 8,
        )
    )
    return record


def occurrence_count_report(
    schedule_tensors: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    counts: dict[str, torch.Tensor] = {}
    records = {}
    for regime in REGIMES:
        value = torch.bincount(schedule_tensors[regime].reshape(-1), minlength=2048).to(torch.int64)
        probability = value.to(torch.float64) / float(value.sum())
        uniform = torch.full_like(probability, 1.0 / value.numel())
        record = {
            **canonical_tensor_record(value, kind="int64"),
            "total": int(value.sum().item()),
            "minimum": int(value.min().item()),
            "maximum": int(value.max().item()),
            "zeros": int((value == 0).sum().item()),
            "l1_from_uniform": float((probability - uniform).abs().sum().item()),
            "coefficient_of_variation": float(
                value.to(torch.float64).std(unbiased=False).item()
                / value.to(torch.float64).mean().item()
            ),
        }
        record["passed"] = (
            record["shape"] == [2048]
            and record["dtype"] == "torch.int64"
            and record["c_contiguous"]
            and record["total"] == 48_000
            and record["minimum"] >= 1
            and record["zeros"] == 0
        )
        counts[regime] = value
        records[regime] = record
    return counts, {
        "records": records,
        "passed": all(record["passed"] for record in records.values()),
    }


def _parameter_ranges_pass(dataset: TrajectorySet, regime: str, cfg: E15Config) -> bool:
    parameters = dataset.parameters
    if parameters.shape != (cfg.pretrain_trajectories_per_regime, 4):
        return False
    if not torch.isfinite(parameters).all():
        return False
    vx, vy, nu, dt = parameters.unbind(dim=1)
    common = bool(((dt >= cfg.minimum_dt) & (dt <= cfg.maximum_dt)).all())
    if regime == "x_advection":
        active = vx.abs()
        return common and bool(
            (vy == 0).all()
            and (nu == 0).all()
            and (active >= cfg.minimum_speed).all()
            and (active <= cfg.maximum_speed).all()
        )
    if regime == "y_advection":
        active = vy.abs()
        return common and bool(
            (vx == 0).all()
            and (nu == 0).all()
            and (active >= cfg.minimum_speed).all()
            and (active <= cfg.maximum_speed).all()
        )
    return common and bool(
        (vx == 0).all()
        and (vy == 0).all()
        and (nu >= cfg.minimum_diffusivity).all()
        and (nu <= cfg.maximum_diffusivity).all()
    )


def realization_preflight(
    name: str,
    descriptor: dict[str, Any],
    cfg: E15Config,
    training: dict[str, TrajectorySet],
    schedule_tensors: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    counts, count_report = occurrence_count_report(schedule_tensors)
    schedule_records = {regime: schedule_record(schedule_tensors[regime]) for regime in REGIMES}
    literal_schedule_seeds = descriptor["literal_schedule_seeds"]
    expected_literal_schedule_seeds = {
        regime: descriptor["schedule_seed"] + index for index, regime in enumerate(REGIMES)
    }
    schedule_seed_binding_passed = literal_schedule_seeds == expected_literal_schedule_seeds
    dataset_records = {
        regime: canonical_dataset_record(dataset) for regime, dataset in training.items()
    }
    shapes_pass = all(
        dataset.coefficients.shape == (256, 9, 52, 1)
        and dataset.transitions[0].shape == (2048, 52, 1)
        and dataset.transitions[1].shape == (2048, 52, 1)
        and dataset.transitions[2].shape == (2048, 4)
        for dataset in training.values()
    )
    parameter_ranges = {
        regime: _parameter_ranges_pass(dataset, regime, cfg) for regime, dataset in training.items()
    }
    excitation = excitation_report(training)
    excitation_passed = bool(
        excitation["required_plane_grams_full_rank"]
        and excitation["mode_tied_jacobian_full_rank"]
        and all(record["full_rank"] for record in excitation["input_covariance"].values())
    )
    report = {
        "name": name,
        "descriptor": descriptor,
        "descriptor_sha256": sha256_bytes(canonical_bytes(descriptor)),
        "generator_independence": {
            "state_sampler": "local_torch_generator_manual_seed",
            "parameter_sampler": "local_torch_generator_manual_seed",
            "schedule_sampler": "local_torch_generator_manual_seed",
            "shared_generator_state": False,
        },
        "dataset_records": dataset_records,
        "schedule_records": schedule_records,
        "literal_schedule_seeds": literal_schedule_seeds,
        "expected_literal_schedule_seeds": expected_literal_schedule_seeds,
        "schedule_seed_binding_passed": schedule_seed_binding_passed,
        "occurrence_counts": count_report,
        "shapes_pass": shapes_pass,
        "parameter_ranges": parameter_ranges,
        "excitation": excitation,
    }
    report["passed"] = all(
        (
            shapes_pass,
            all(record["all_finite"] for record in dataset_records.values()),
            all(parameter_ranges.values()),
            all(record["passed"] for record in schedule_records.values()),
            schedule_seed_binding_passed,
            count_report["passed"],
            excitation_passed,
        )
    )
    return counts, report


def coverage_report(
    evaluations: dict[str, Any],
    records: list[dict[str, Any]],
    argmax: dict[str, Any],
    training: dict[str, Any],
) -> dict[str, Any]:
    keys = {
        (
            record["control"],
            record["basis_index"],
            record["case_name"],
            record["horizon"],
        )
        for record in records
    }
    report = {
        "generator_identification_cells": len(evaluations),
        "validation_cells": sum(
            len(control_report["validation"]) for control_report in evaluations.values()
        ),
        "mode_resolved_records": len(records),
        "unique_mode_resolved_keys": len(keys),
        "mode_argmax_cells": sum(len(values) for values in argmax.values()),
        "training_records": len(training),
    }
    expected = {
        "generator_identification_cells": 4,
        "validation_cells": 16,
        "mode_resolved_records": 7056,
        "unique_mode_resolved_keys": 7056,
        "mode_argmax_cells": 20,
        "training_records": 3,
    }
    report["expected"] = expected
    report["passed"] = all(report[key] == value for key, value in expected.items())
    return report


def classify(
    *,
    preflight_passed: bool,
    execution_complete: bool,
    stability: dict[str, bool],
) -> str:
    if not preflight_passed:
        return "e16_preflight_failed"
    if not execution_complete:
        return "e16_execution_incomplete"
    adam = stability["deterministic_adamw_restart"]
    lbfgs = stability["componentwise_lbfgs_neutral"]
    if adam and lbfgs:
        return "both_practical_recovery_packages_stable"
    if lbfgs:
        return "componentwise_lbfgs_neutral_stable_only"
    if adam:
        return "deterministic_adamw_restart_stable_only"
    return "no_practical_recovery_package_stable"


def stability_from_recovery(
    sealed: dict[str, bool],
    fresh: dict[str, dict[str, bool]],
) -> tuple[dict[str, dict[str, bool]], dict[str, bool]]:
    vector = {
        package: {
            "r0_sealed_e15": bool(sealed[package]),
            **{realization: bool(bits[package]) for realization, bits in fresh.items()},
        }
        for package in PACKAGES
    }
    return vector, {package: all(bits.values()) for package, bits in vector.items()}


def _boundary() -> dict[str, Any]:
    return {
        "synthetic_periodic_scalar_linear_only": True,
        "nonlinear_dynamics_qualified": False,
        "particle_dynamics_qualified": False,
        "heldout_reads": 0,
        "provider_calls": 0,
        "routing_paths": 0,
        "representation_label_inputs": False,
        "task_label_inputs": False,
        "source_bypasses": 0,
        "encoder_updates": 0,
        "public_or_claim_grade": False,
    }


def prepare_realization(
    name: str,
    descriptor: dict[str, Any],
    *,
    base_cfg: E15Config,
    space: PhysicalFunctionSpace,
    validation: dict[str, TrajectorySet],
) -> dict[str, Any]:
    cfg = realization_config(base_cfg, descriptor)
    training = build_training(cfg, space)
    schedule_tensors = elementary_schedules(cfg)
    counts, preflight = realization_preflight(
        name,
        descriptor,
        cfg,
        training,
        schedule_tensors,
    )
    validation_seed_binding = {
        "validation_state_seed": cfg.validation_state_seed,
        "validation_parameter_seed": cfg.validation_parameter_seed,
        "x_validation_parameter_seed": cfg.x_validation_parameter_seed,
        "y_validation_parameter_seed": cfg.y_validation_parameter_seed,
        "diffusion_validation_parameter_seed": cfg.diffusion_validation_parameter_seed,
    }
    expected_validation_seed_binding = {
        "validation_state_seed": 61_001,
        "validation_parameter_seed": 61_101,
        "x_validation_parameter_seed": 61_102,
        "y_validation_parameter_seed": 61_103,
        "diffusion_validation_parameter_seed": 61_104,
    }
    preflight["validation_seed_binding"] = validation_seed_binding
    preflight["expected_validation_seed_binding"] = expected_validation_seed_binding
    preflight["validation_records"] = {
        regime: canonical_dataset_record(dataset) for regime, dataset in validation.items()
    }
    preflight["passed"] = (
        preflight["passed"] and validation_seed_binding == expected_validation_seed_binding
    )
    return {
        "name": name,
        "descriptor": descriptor,
        "cfg": cfg,
        "training": training,
        "schedule_tensors": schedule_tensors,
        "counts": counts,
        "preflight": preflight,
    }


def prepare_checkpoint(
    prepared: dict[str, Any],
    *,
    oracle: FixedGenerator,
) -> dict[str, Any]:
    name = prepared["name"]
    descriptor = prepared["descriptor"]
    cfg = prepared["cfg"]
    training = prepared["training"]
    schedule_tensors = prepared["schedule_tensors"]
    counts = prepared["counts"]
    preflight = prepared["preflight"]
    neutral = StructuredGenerator()
    checkpoint = StructuredGenerator()
    replay_training = train_elementary(
        checkpoint,
        training,
        {regime: schedule_tensors[regime] for regime in REGIMES},
        cfg,
    )
    integrity = objective_integrity(
        neutral,
        checkpoint,
        oracle,
        training,
        counts,
        schedule_tensors,
        chunk_trajectories=cfg.group_chunk_trajectories,
    )
    preflight["objective_integrity"] = integrity
    preflight["passed"] = preflight["passed"] and integrity["passed"]
    checkpoint_record = {
        "model_sha256": model_hash(checkpoint),
        "generator_sha256": _matrix_hashes(checkpoint),
        "training": replay_training,
    }
    if not preflight["passed"]:
        return {
            "name": name,
            "descriptor": descriptor,
            "cfg": cfg,
            "training": training,
            "schedule_tensors": schedule_tensors,
            "counts": counts,
            "preflight": preflight,
            "checkpoint": checkpoint_record,
            "execution_complete": False,
        }
    return {
        "name": name,
        "descriptor": descriptor,
        "cfg": cfg,
        "training": training,
        "schedule_tensors": schedule_tensors,
        "counts": counts,
        "preflight": preflight,
        "neutral": neutral,
        "checkpoint_model": checkpoint,
        "checkpoint": checkpoint_record,
    }


def run_realization(
    prepared: dict[str, Any],
    *,
    space: PhysicalFunctionSpace,
    validation: dict[str, TrajectorySet],
    oracle: FixedGenerator,
    canonical_coords: torch.Tensor,
) -> dict[str, Any]:
    name = prepared["name"]
    descriptor = prepared["descriptor"]
    cfg = prepared["cfg"]
    training = prepared["training"]
    counts = prepared["counts"]
    preflight = prepared["preflight"]
    neutral = prepared["neutral"]
    checkpoint = prepared["checkpoint_model"]
    checkpoint_record = prepared["checkpoint"]
    replay_training = checkpoint_record["training"]
    adam_restart = copy.deepcopy(checkpoint)
    lbfgs_neutral = neutral
    training_reports = {
        "e12_checkpoint_replay": replay_training,
        "deterministic_adamw_restart": train_adamw(
            adam_restart,
            training,
            counts,
            validation,
            oracle=oracle,
            space=space,
            canonical_coords=canonical_coords,
            cfg=cfg,
        ),
        "componentwise_lbfgs_neutral": optimize_components_weighted(
            lbfgs_neutral,
            training,
            counts,
            validation=validation,
            oracle=oracle,
            space=space,
            canonical_coords=canonical_coords,
            cfg=cfg,
        ),
    }
    controls = {
        "e12_checkpoint": checkpoint,
        "deterministic_adamw_restart": adam_restart,
        "componentwise_lbfgs_neutral": lbfgs_neutral,
        "oracle": oracle,
    }
    evaluations = {
        control: evaluate_control(
            model,
            validation,
            oracle=oracle,
            space=space,
            canonical_coords=canonical_coords,
            cfg=cfg,
        )
        for control, model in controls.items()
    }
    mode_records, mode_argmax = mode_resolved_records(
        controls,
        oracle=oracle,
        space=space,
        canonical_coords=canonical_coords,
    )
    coverage = coverage_report(evaluations, mode_records, mode_argmax, training_reports)
    recovery = {package: bool(evaluations[package]["recovery_pass"]) for package in PACKAGES}
    execution_complete = coverage["passed"] and _all_finite(
        {
            "training": training_reports,
            "evaluations": evaluations,
            "mode_records": mode_records,
            "mode_argmax": mode_argmax,
        }
    )
    return {
        "name": name,
        "descriptor": descriptor,
        "config": asdict(cfg),
        "config_sha256": sha256_bytes(canonical_bytes(asdict(cfg))),
        "preflight": preflight,
        "checkpoint": {
            **checkpoint_record,
            "evaluation": evaluations["e12_checkpoint"],
        },
        "training": training_reports,
        "evaluations": evaluations,
        "mode_resolved": mode_records,
        "mode_argmax": mode_argmax,
        "coverage": coverage,
        "recovery": recovery,
        "execution_complete": execution_complete,
    }


def run_replicate(
    cfg: E15Config,
    *,
    environment: dict[str, Any],
    provenance_report: dict[str, Any],
) -> dict[str, Any]:
    if asdict(cfg) != asdict(frozen_e15_config()):
        raise ValueError("E16 requires the exact frozen E15 base configuration")
    if not provenance_report["passed"]:
        return {
            "schema_version": 1,
            "experiment": "canonical_latent_e16_multi_realization_robustness",
            "config": asdict(cfg),
            "provenance": provenance_report,
            "classification": "e16_preflight_failed",
            "state_reads": {"training": 0, "validation": 0, "heldout": 0},
            "boundary": _boundary(),
        }

    torch.manual_seed(cfg.model_seed)
    space = PhysicalFunctionSpace(
        FunctionSpaceConfig(
            seed=23,
            validation_states=cfg.validation_trajectories,
            canonical_query_resolution=cfg.canonical_query_resolution,
            calibration_resolution=cfg.truth_resolution,
            max_condition_number=100.0,
        )
    )
    oracle = FixedGenerator(*oracle_generators())
    oracle_report = oracle_preflight(cfg, space, oracle)
    if not oracle_report["passed"]:
        return {
            "schema_version": 1,
            "experiment": "canonical_latent_e16_multi_realization_robustness",
            "config": asdict(cfg),
            "config_sha256": sha256_bytes(canonical_bytes(asdict(cfg))),
            "registered_realizations": REALIZATIONS,
            "provenance": provenance_report,
            "oracle_preflight": oracle_report,
            "realizations": {},
            "recovery_vector": {},
            "stability": {package: False for package in PACKAGES},
            "classification": "e16_preflight_failed",
            "nonlinear_expansion_authorized": False,
            "state_reads": {"training": 0, "validation": 0, "heldout": 0},
            "boundary": _boundary(),
            "reproducibility": environment,
        }
    validation = build_validation(cfg, space)
    canonical_coords, _ = canonical_grid(space, cfg.canonical_query_resolution)
    prepared = {
        name: prepare_realization(
            name,
            descriptor,
            base_cfg=cfg,
            space=space,
            validation=validation,
        )
        for name, descriptor in REALIZATIONS.items()
    }
    sampled_preflight_passed = all(value["preflight"]["passed"] for value in prepared.values())
    if sampled_preflight_passed:
        checkpoint_prepared = {
            name: prepare_checkpoint(
                value,
                oracle=oracle,
            )
            for name, value in prepared.items()
        }
        package_preflight_passed = all(
            value["preflight"]["passed"] for value in checkpoint_prepared.values()
        )
        if package_preflight_passed:
            realization_results = {
                name: run_realization(
                    value,
                    space=space,
                    validation=validation,
                    oracle=oracle,
                    canonical_coords=canonical_coords,
                )
                for name, value in checkpoint_prepared.items()
            }
        else:
            realization_results = {
                name: {
                    "name": name,
                    "descriptor": value["descriptor"],
                    "preflight": value["preflight"],
                    "checkpoint": value["checkpoint"],
                    "execution_complete": False,
                }
                for name, value in checkpoint_prepared.items()
            }
    else:
        realization_results = {
            name: {
                "name": name,
                "descriptor": value["descriptor"],
                "preflight": value["preflight"],
                "execution_complete": False,
            }
            for name, value in prepared.items()
        }
    preflight_passed = sampled_preflight_passed and all(
        result["preflight"]["passed"] for result in realization_results.values()
    )
    sealed_recovery = provenance_report["sealed_e15"]["recovery"]
    recovery_vector, stability = stability_from_recovery(
        sealed_recovery,
        {
            name: {
                package: bool(result.get("recovery", {}).get(package, False))
                for package in PACKAGES
            }
            for name, result in realization_results.items()
        },
    )
    execution_complete = preflight_passed and all(
        result.get("execution_complete", False) for result in realization_results.values()
    )
    classification = classify(
        preflight_passed=preflight_passed,
        execution_complete=execution_complete,
        stability=stability,
    )
    result = {
        "schema_version": 1,
        "experiment": "canonical_latent_e16_multi_realization_robustness",
        "config": asdict(cfg),
        "config_sha256": sha256_bytes(canonical_bytes(asdict(cfg))),
        "registered_realizations": REALIZATIONS,
        "provenance": provenance_report,
        "oracle_preflight": oracle_report,
        "realizations": realization_results,
        "recovery_vector": recovery_vector,
        "stability": stability,
        "classification": classification,
        "nonlinear_expansion_authorized": classification
        in {
            "both_practical_recovery_packages_stable",
            "componentwise_lbfgs_neutral_stable_only",
            "deterministic_adamw_restart_stable_only",
        },
        "state_reads": {"training": 1536, "validation": 256, "heldout": 0},
        "boundary": _boundary(),
        "reproducibility": environment,
    }
    if not _all_finite(result):
        raise E16IncompleteError("nonfinite_result")
    return result


def deterministic_bundle(raw: dict[str, bytes]) -> tuple[bytes, list[dict[str, Any]]]:
    output = io.BytesIO()
    with gzip.GzipFile(
        filename="", mode="wb", compresslevel=9, fileobj=output, mtime=0
    ) as compressed:
        with tarfile.open(fileobj=compressed, mode="w", format=tarfile.GNU_FORMAT) as archive:
            for name in RAW_MEMBERS:
                value = raw[name]
                info = tarfile.TarInfo(name)
                info.size = len(value)
                info.mode = 0o644
                info.uid = info.gid = info.mtime = 0
                info.uname = info.gname = ""
                info.pax_headers = {}
                archive.addfile(info, io.BytesIO(value))
    members = [
        {"name": name, "bytes": len(raw[name]), "raw_sha256": sha256_bytes(raw[name])}
        for name in RAW_MEMBERS
    ]
    return output.getvalue(), members


def verify_bundle(bundle: bytes, expected: list[dict[str, Any]]) -> None:
    if bundle[:3] != b"\x1f\x8b\x08" or bundle[3] & 0x08 or bundle[4:8] != b"\0\0\0\0":
        raise RuntimeError("E16 gzip header is not deterministic")
    with tarfile.open(fileobj=io.BytesIO(bundle), mode="r:gz") as archive:
        members = archive.getmembers()
        if [member.name for member in members] != list(RAW_MEMBERS):
            raise RuntimeError("E16 archive member set or order changed")
        for member, record in zip(members, expected, strict=True):
            stream = archive.extractfile(member)
            if stream is None:
                raise RuntimeError(f"E16 archive member unreadable: {member.name}")
            value = stream.read()
            if not all(
                (
                    member.isfile(),
                    member.pax_headers == {},
                    member.mode == 0o644,
                    member.uid == member.gid == member.mtime == 0,
                    member.uname == member.gname == "",
                    len(value) == record["bytes"],
                    sha256_bytes(value) == record["raw_sha256"],
                )
            ):
                raise RuntimeError(f"E16 archive metadata/hash mismatch: {member.name}")


def build_outputs(
    first: dict[str, Any],
    second: dict[str, Any],
    complete: dict[str, Any],
) -> dict[str, bytes]:
    raw = {
        RAW_MEMBERS[0]: pretty_bytes(first),
        RAW_MEMBERS[1]: pretty_bytes(second),
        RAW_MEMBERS[2]: pretty_bytes(complete),
    }
    bundle, members = deterministic_bundle(raw)
    verify_bundle(bundle, members)
    bundle_record = {
        "filename": OUTPUT_NAMES["bundle"],
        "bytes": len(bundle),
        "raw_sha256": sha256_bytes(bundle),
        "members": members,
    }
    compact = {
        "schema_version": 1,
        "experiment": complete["experiment"],
        "classification": complete["classification"],
        "execution_head": complete["provenance"]["git_head"],
        "config": complete["config"],
        "config_sha256": complete["config_sha256"],
        "registered_realizations": complete["registered_realizations"],
        "source_files": complete["provenance"]["source_files"],
        "sealed_e15": complete["provenance"]["sealed_e15"],
        "evidence_bundle": bundle_record,
        "raw_and_canonical_hashes": complete["replication"],
        "oracle_preflight": complete["oracle_preflight"],
        "realizations": complete["realizations"],
        "recovery_vector": complete["recovery_vector"],
        "stability": complete["stability"],
        "nonlinear_expansion_authorized": complete["nonlinear_expansion_authorized"],
        "state_reads": complete["state_reads"],
        "boundary": complete["boundary"],
        "reproducibility": complete["reproducibility"],
    }
    compact_bytes = pretty_bytes(compact)
    manifest = {
        "schema_version": 1,
        "experiment": complete["experiment"],
        "classification": complete["classification"],
        "execution_head": complete["provenance"]["git_head"],
        "outputs": {
            "evidence_bundle": bundle_record,
            "compact_result": {
                "filename": OUTPUT_NAMES["result"],
                "bytes": len(compact_bytes),
                "raw_sha256": sha256_bytes(compact_bytes),
            },
        },
        "compact_result_self_hash_declared": False,
        "manifest_self_hash_declared": False,
        "boundary": complete["boundary"],
    }
    return {
        OUTPUT_NAMES["bundle"]: bundle,
        OUTPUT_NAMES["result"]: compact_bytes,
        OUTPUT_NAMES["manifest"]: pretty_bytes(manifest),
    }


def verify_outputs(outputs: dict[str, bytes]) -> None:
    if set(outputs) != set(OUTPUT_NAMES.values()):
        raise RuntimeError("E16 output filename set changed")
    manifest = json.loads(outputs[OUTPUT_NAMES["manifest"]])
    compact = json.loads(outputs[OUTPUT_NAMES["result"]])
    result_record = manifest["outputs"]["compact_result"]
    bundle_record = manifest["outputs"]["evidence_bundle"]
    if result_record["raw_sha256"] != sha256_bytes(outputs[OUTPUT_NAMES["result"]]):
        raise RuntimeError("E16 compact hash mismatch")
    if result_record["bytes"] != len(outputs[OUTPUT_NAMES["result"]]):
        raise RuntimeError("E16 compact byte count mismatch")
    if bundle_record["raw_sha256"] != sha256_bytes(outputs[OUTPUT_NAMES["bundle"]]):
        raise RuntimeError("E16 bundle hash mismatch")
    if bundle_record["bytes"] != len(outputs[OUTPUT_NAMES["bundle"]]):
        raise RuntimeError("E16 bundle byte count mismatch")
    if "raw_sha256" in compact or "bytes" in compact:
        raise RuntimeError("E16 compact self hash detected")
    if "manifest" in manifest.get("outputs", {}):
        raise RuntimeError("E16 manifest self hash detected")
    verify_bundle(outputs[OUTPUT_NAMES["bundle"]], bundle_record["members"])


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _remove_path(path: Path) -> None:
    if not os.path.lexists(path):
        return
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def publish_atomic(output_dir: Path, outputs: dict[str, bytes]) -> None:
    verify_outputs(outputs)
    parent = output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    lock = output_dir.with_name(f".{output_dir.name}.lock")
    stage: Path | None = None
    descriptor: int | None = None
    published = False
    failure: BaseException | None = None
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.write(descriptor, b"e16-publication-lock\n")
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        if os.path.lexists(output_dir):
            raise FileExistsError(f"E16 output already exists: {output_dir}")
        stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.stage-", dir=parent))
        for name, value in outputs.items():
            path = stage / name
            with path.open("xb") as stream:
                stream.write(value)
                stream.flush()
                os.fsync(stream.fileno())
        _fsync_directory(stage)
        reopened = {path.name: path.read_bytes() for path in stage.iterdir()}
        verify_outputs(reopened)
        if os.path.lexists(output_dir):
            raise FileExistsError(f"E16 output raced before publication: {output_dir}")
        os.rename(stage, output_dir)
        stage = None
        published = True
        _fsync_directory(parent)
    except BaseException as error:
        failure = error
    try:
        if descriptor is not None:
            os.close(descriptor)
        if stage is not None:
            _remove_path(stage)
        if os.path.lexists(lock):
            lock.unlink()
            _fsync_directory(parent)
    except BaseException as cleanup_error:
        if failure is None:
            failure = cleanup_error
    if failure is not None:
        try:
            if published:
                _remove_path(output_dir)
                _fsync_directory(parent)
            if os.path.lexists(lock):
                lock.unlink()
                _fsync_directory(parent)
        except BaseException as rollback_error:
            raise RuntimeError("E16 publication rollback failed") from rollback_error
        raise failure


def incomplete_status(reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment": "canonical_latent_e16_multi_realization_robustness",
        "classification": "e16_execution_incomplete",
        "incomplete_reason": reason,
        "scientific_conclusion_recorded": False,
        "durable_evidence_published": False,
        "boundary": _boundary(),
    }


def _run_under_limit(
    cfg: E15Config,
    *,
    output_dir: Path,
    environment: dict[str, Any],
    provenance_report: dict[str, Any],
) -> dict[str, Any]:
    with time_limit(PER_REPLICA_TIMEOUT_SECONDS, "E16 replicate A"):
        first = run_replicate(
            cfg,
            environment=environment,
            provenance_report=provenance_report,
        )
    with time_limit(PER_REPLICA_TIMEOUT_SECONDS, "E16 replicate B"):
        second = run_replicate(
            cfg,
            environment=environment,
            provenance_report=provenance_report,
        )
    first_canonical = canonical_bytes(first)
    second_canonical = canonical_bytes(second)
    if first_canonical != second_canonical:
        raise E16IncompleteError("replication_payload_mismatch")
    first_raw = pretty_bytes(first)
    second_raw = pretty_bytes(second)
    if first_raw != second_raw:
        raise E16IncompleteError("replication_raw_bytes_mismatch")
    complete = {
        **first,
        "replication": {
            "raw_replicates_byte_identical": True,
            "replicate_a_raw_sha256": sha256_bytes(first_raw),
            "replicate_b_raw_sha256": sha256_bytes(second_raw),
            "replicate_a_canonical_sha256": sha256_bytes(first_canonical),
            "replicate_b_canonical_sha256": sha256_bytes(second_canonical),
        },
    }
    without_replication = dict(complete)
    without_replication.pop("replication")
    if without_replication != first:
        raise E16IncompleteError("combined_result_mismatch")
    complete["replication"]["combined_without_replication_equals_replicate"] = True
    complete["replication"]["complete_canonical_without_replication_sha256"] = sha256_bytes(
        canonical_bytes(without_replication)
    )
    outputs = build_outputs(first, second, complete)
    try:
        publish_atomic(output_dir, outputs)
    except BaseException as error:
        raise E16IncompleteError("publication_failure") from error
    return complete


def run_e16(cfg: E15Config, *, output_dir: Path) -> dict[str, Any]:
    if output_dir.resolve(strict=False) != CANONICAL_OUTPUT_DIR.resolve(strict=False):
        raise ValueError(
            f"E16 output directory is frozen to {CANONICAL_OUTPUT_DIR.relative_to(REPO_ROOT)}"
        )
    environment = configure_environment(cfg)
    provenance_report = provenance(environment)
    try:
        with time_limit(WHOLE_EXPERIMENT_TIMEOUT_SECONDS, "E16 whole experiment"):
            return _run_under_limit(
                cfg,
                output_dir=output_dir,
                environment=environment,
                provenance_report=provenance_report,
            )
    except E16IncompleteError as error:
        return incomplete_status(error.reason)
    except TimeoutError:
        return incomplete_status("timeout")
    except MemoryError:
        return incomplete_status("resource_exhaustion")
    except KeyboardInterrupt:
        return incomplete_status("interrupted")
    except OSError:
        return incomplete_status("resource_or_io_failure")
    except RuntimeError:
        return incomplete_status("runtime_failure")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=CANONICAL_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_e16(frozen_e15_config(), output_dir=args.output_dir)
    if result["classification"] == "e16_execution_incomplete":
        print(json.dumps(result, sort_keys=True))
        raise SystemExit(2)
    print(json.dumps({"classification": result["classification"]}, sort_keys=True))


if __name__ == "__main__":
    main()
