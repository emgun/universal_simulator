#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import io
import json
import math
import os
import platform
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from scripts.run_canonical_latent_e7_function_space import (
    FunctionSpaceConfig,
    PhysicalFunctionSpace,
)
from scripts.run_canonical_latent_e11_coefficient_operator_transfer import (
    REGIMES,
    TrajectorySet,
    canonical_grid,
    modal_scales,
    model_hash,
    normalized_loss,
)
from scripts.run_canonical_latent_e12_structured_generator import (
    FixedGenerator,
    RuleAdapter,
    StructuredGenerator,
    _all_finite,
    _matrix_hashes,
    closure_parameter_cases,
    oracle_generators,
    oracle_preflight,
    train_elementary,
)
from scripts.run_canonical_latent_e13_identifiability_audit import (
    E7_RUNNER_PATH,
    E11_RUNNER_PATH,
    E12_ARTIFACT_PATH,
    E12_LOCK_PATH,
    E12_RUNNER_PATH,
    EXPECTED_E12_BASIS_ACTION,
    EXPECTED_E12_ELEMENTARY_SHA256,
    EXPECTED_E12_FINAL_LOSS,
    EXPECTED_E12_FIRST_LOSS,
    EXPECTED_E12_GENERATOR_SHA256,
    EXPECTED_E12_INITIAL_SHA256,
    EXPECTED_E12_ZERO_SHOT_ROLLOUT,
    LATENT_EVAL_PATH,
    E13Config,
    build_frozen_datasets,
    component_parameters,
    coverage_report,
    evaluate_control,
    mode_resolved_records,
    replay_lock_report,
    schedules,
)
from scripts.seal_canonical_latent_e14_evidence import recompute_recovery_pass

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = Path(__file__).resolve()
CONTRACT_PATH = (
    REPO_ROOT / "docs/research/2026-07-26-canonical-latent-e15-training-package-contract.md"
)
TEST_PATH = REPO_ROOT / "tests/unit/test_run_canonical_latent_e15_training_package.py"
E13_CONTRACT_PATH = (
    REPO_ROOT / "docs/research/2026-07-26-canonical-latent-e13-identifiability-audit-contract.md"
)
E13_RUNNER_PATH = REPO_ROOT / "scripts/run_canonical_latent_e13_identifiability_audit.py"
E14_CONTRACT_PATH = (
    REPO_ROOT / "docs/research/2026-07-26-canonical-latent-e14-evidence-seal-contract.md"
)
E14_SEALER_PATH = REPO_ROOT / "scripts/seal_canonical_latent_e14_evidence.py"
E14_RESULT_PATH = (
    REPO_ROOT
    / "docs/research/artifacts/canonical_latent_e14_evidence_seal"
    / "canonical_latent_e14_evidence_seal_result.json"
)
E14_MANIFEST_PATH = (
    REPO_ROOT
    / "docs/research/artifacts/canonical_latent_e14_evidence_seal"
    / "canonical_latent_e14_evidence_seal_manifest.json"
)
E14_BUNDLE_PATH = (
    REPO_ROOT
    / "docs/research/artifacts/canonical_latent_e14_evidence_seal"
    / "canonical_latent_e14_evidence_bundle.tar.gz"
)
E14_RESULT_RECORD_PATH = (
    REPO_ROOT / "docs/research/2026-07-26-canonical-latent-e14-evidence-seal-result.md"
)
E14_HANDOFF_PATH = (
    REPO_ROOT / "docs/steward/2026-07-26-canonical-latent-e14-evidence-seal-handoff.md"
)

EXPECTED_E14_HASHES = {
    "bundle": "a4886e3b3a8c678abe7b8f44907b8655af4b0ac68fb47ca9353ae2bfca677b5c",
    "result": "a26b4948a0db7fd7aa1a0c067bf29eaacd3007e4c93e1bb309a5c610cad04413",
    "manifest": "88f45edfa0c456ec19eef8da8167ed0f2bf6213f13f00ef6f32d934c4cf9257e",
}
EXPECTED_E13_EVALUATION_HASHES = {
    "full_skew_lbfgs_neutral": ("087dcd799fa909129d7b8848dffd3073eb6818d6b3fb5a4bcb6e9fa9a73df06b"),
    "full_skew_lbfgs_polish": ("1511fb6e6a5a7b450a5f235a2548a62a709f73d59198968ce99e312218cd76ef"),
}
EXPECTED_E13_MODELS = {
    "full_skew_lbfgs_neutral": {
        "model_sha256": "b1faad552a12d9e71a2ec9788cf5b9e46547cfb8ac1fc8e7cdd706af7e46208f",
        "generator_sha256": {
            "A_x": "6e9cf3c8f18b4a6613909f8b305b6cd24638a4f005322ca685fb54028b4b506c",
            "A_y": "e051059bd996fb29d07addc914346c439e62ac6061edf3d4fc80046742e145a6",
            "D": "8bd2aa03a087dffafdce8669d9fe2200a60c45229d6dc4e510718b6ed1b8a8d6",
        },
    },
    "full_skew_lbfgs_polish": {
        "model_sha256": "2044abed51748bcda79173f0a99d84c38391d3af5553aabe102e94812c5392db",
        "generator_sha256": {
            "A_x": "86bb2d1c7834cfe641973007e940f9024309bcd720424fc7c98a82b44aea3226",
            "A_y": "476163d49bde39e561e5ae28a84a0e2880bcf5e3d31aa684ee7b84fe4239b506",
            "D": "1edb9c7429ebec8a583d000fedf2e754668727fed857b14d3d06f5b1649fa370",
        },
    },
}
EXPECTED_COUNT_RECORDS = {
    "x_advection": {
        "sha256": "1048116699e92f3de058114f01552707497a7fcd31e2a1bb288dbed2fdd0e5b5",
        "total": 48000,
        "minimum": 10,
        "maximum": 46,
        "zeros": 0,
    },
    "y_advection": {
        "sha256": "c2208664ec391862db76668b1084b7ff8b2f08dc39a43259527f38cbcbf75b93",
        "total": 48000,
        "minimum": 9,
        "maximum": 41,
        "zeros": 0,
    },
    "diffusion": {
        "sha256": "3c62df055b01246b7a88ba8fa187f642b441645f9ce056c77f91272fa6c8a1a3",
        "total": 48000,
        "minimum": 8,
        "maximum": 40,
        "zeros": 0,
    },
}

CONTROLS = (
    "e12_adamw_replay",
    "schedule_weighted_adamw_neutral",
    "schedule_weighted_adamw_restart",
    "schedule_weighted_componentwise_lbfgs_neutral",
    "schedule_weighted_componentwise_lbfgs_restart",
    "oracle",
)
NEW_CONTROLS = CONTROLS[1:-1]
TRACE_STEPS = (0, 1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 1500)
SEMANTIC_FREQUENCIES = (0, 1, 1, 2, 2, 3, 3)
RAW_MEMBERS = (
    "replicate_a/result.json",
    "replicate_b/result.json",
    "complete_result.json",
)
OUTPUT_NAMES = {
    "bundle": "canonical_latent_e15_training_package_evidence_bundle.tar.gz",
    "result": "canonical_latent_e15_training_package_result.json",
    "manifest": "canonical_latent_e15_training_package_manifest.json",
}
CANONICAL_OUTPUT_DIR = REPO_ROOT / "docs/research/artifacts/canonical_latent_e15_training_package"
VALIDATION_CASE_NAMES = tuple(record[0] for record in closure_parameter_cases())
PER_REPLICA_TIMEOUT_SECONDS = 6 * 60 * 60
WHOLE_EXPERIMENT_TIMEOUT_SECONDS = 14 * 60 * 60


class E15IncompleteError(RuntimeError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class E15Config(E13Config):
    group_chunk_trajectories: int = 32
    expected_python: tuple[int, int, int] = (3, 12, 7)
    expected_torch: str = "2.7.0"
    intraop_threads: int = 1
    interop_threads: int = 1

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            self.group_chunk_trajectories != 32
            or self.expected_python != (3, 12, 7)
            or self.expected_torch != "2.7.0"
            or self.intraop_threads != 1
            or self.interop_threads != 1
        ):
            raise ValueError("E15 execution environment is frozen")


def frozen_e15_config() -> E15Config:
    return E15Config()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def pretty_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_path(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _committed_sha256(path: Path) -> str | None:
    relative = path.relative_to(REPO_ROOT).as_posix()
    completed = subprocess.run(
        ["git", "show", f"HEAD:{relative}"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
    )
    return sha256_bytes(completed.stdout) if completed.returncode == 0 else None


def _git_text(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@contextmanager
def time_limit(seconds: int, label: str):
    if seconds <= 0:
        raise ValueError("E15 time limit must be positive")
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_remaining, previous_interval = signal.setitimer(signal.ITIMER_REAL, 0.0)
    started = time.monotonic()

    def _raise_timeout(_signum: int, _frame: Any) -> None:
        raise TimeoutError(f"{label} exceeded its frozen time limit")

    signal.signal(signal.SIGALRM, _raise_timeout)
    effective = (
        min(float(seconds), previous_remaining) if previous_remaining > 0 else float(seconds)
    )
    signal.setitimer(signal.ITIMER_REAL, effective)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_remaining > 0:
            elapsed = time.monotonic() - started
            signal.setitimer(
                signal.ITIMER_REAL,
                max(previous_remaining - elapsed, 1e-6),
                previous_interval,
            )


def configure_environment(cfg: E15Config) -> dict[str, Any]:
    torch.set_num_threads(cfg.intraop_threads)
    torch.set_num_interop_threads(cfg.interop_threads)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(cfg.model_seed)
    report = {
        "python": list(sys.version_info[:3]),
        "python_full": platform.python_version(),
        "torch": torch.__version__,
        "device": "cpu",
        "dtype": "torch.float64",
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "intraop_threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
    }
    report["passed"] = all(
        (
            tuple(report["python"]) == cfg.expected_python,
            report["torch"] == cfg.expected_torch,
            report["device"] == "cpu",
            report["dtype"] == "torch.float64",
            report["deterministic_algorithms"] is True,
            report["intraop_threads"] == 1,
            report["interop_threads"] == 1,
        )
    )
    return report


def provenance(environment: dict[str, Any]) -> dict[str, Any]:
    sources = {
        "contract": CONTRACT_PATH,
        "runner": RUNNER_PATH,
        "tests": TEST_PATH,
        "e13_contract": E13_CONTRACT_PATH,
        "e13_runner": E13_RUNNER_PATH,
        "e12_lock": E12_LOCK_PATH,
        "e12_artifact": E12_ARTIFACT_PATH,
        "e12_runner": E12_RUNNER_PATH,
        "e11_runner": E11_RUNNER_PATH,
        "e7_runner": E7_RUNNER_PATH,
        "latent_evaluation": LATENT_EVAL_PATH,
        "e14_contract": E14_CONTRACT_PATH,
        "e14_sealer": E14_SEALER_PATH,
        "e14_result": E14_RESULT_PATH,
        "e14_manifest": E14_MANIFEST_PATH,
        "e14_bundle": E14_BUNDLE_PATH,
        "e14_result_record": E14_RESULT_RECORD_PATH,
        "e14_handoff": E14_HANDOFF_PATH,
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
        }
    e14_hashes = {
        "bundle": sha256_path(E14_BUNDLE_PATH),
        "result": sha256_path(E14_RESULT_PATH),
        "manifest": sha256_path(E14_MANIFEST_PATH),
    }
    report = {
        "git_head": _git_text("rev-parse", "HEAD"),
        "worktree_clean": _git_text("status", "--porcelain") == "",
        "source_files": records,
        "source_files_match_git_head": all(
            record["matches_git_head"] for record in records.values()
        ),
        "e14_hashes": e14_hashes,
        "e14_hashes_match": e14_hashes == EXPECTED_E14_HASHES,
        "environment": environment,
    }
    report["passed"] = all(
        (
            len(report["git_head"]) == 40,
            report["worktree_clean"],
            report["source_files_match_git_head"],
            report["e14_hashes_match"],
            environment["passed"],
        )
    )
    return report


def occurrence_counts(
    schedule_tensors: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    counts = {}
    records = {}
    for regime in REGIMES:
        value = torch.bincount(schedule_tensors[regime].reshape(-1), minlength=2048).to(torch.int64)
        raw = value.cpu().contiguous().numpy().astype("<i8", copy=True).tobytes()
        probability = value.to(torch.float64) / float(value.sum())
        uniform = torch.full_like(probability, 1.0 / value.numel())
        record = {
            "sha256": sha256_bytes(raw),
            "total": int(value.sum()),
            "minimum": int(value.min()),
            "maximum": int(value.max()),
            "zeros": int((value == 0).sum()),
            "l1_from_uniform": float((probability - uniform).abs().sum()),
            "coefficient_of_variation": float(
                value.to(torch.float64).std(unbiased=False) / value.to(torch.float64).mean()
            ),
        }
        record["passed"] = all(
            record[key] == expected for key, expected in EXPECTED_COUNT_RECORDS[regime].items()
        )
        counts[regime] = value
        records[regime] = record
    return counts, {
        "records": records,
        "passed": all(record["passed"] for record in records.values()),
    }


def grouped_outputs(
    model: StructuredGenerator | FixedGenerator,
    dataset: TrajectorySet,
    *,
    chunk_trajectories: int,
) -> torch.Tensor:
    ax, ay, diffusion = model.matrices()
    outputs = []
    for start in range(0, dataset.coefficients.shape[0], chunk_trajectories):
        stop = min(start + chunk_trajectories, dataset.coefficients.shape[0])
        parameters = dataset.parameters[start:stop]
        generator = (
            parameters[:, 0, None, None] * ax
            + parameters[:, 1, None, None] * ay
            + parameters[:, 2, None, None] * diffusion
        )
        transition = torch.matrix_exp(parameters[:, 3, None, None] * generator)
        active = dataset.coefficients[start:stop, :-1, :49]
        evolved = transition[:, None] @ active
        inactive = dataset.coefficients[start:stop, :-1, 49:].clone()
        outputs.append(torch.cat((evolved, inactive), dim=2))
    return torch.cat(outputs, dim=0).reshape(-1, 52, 1)


def loss_from_outputs(
    outputs: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    per_transition = ((outputs - targets) / modal_scales()).square().mean(dim=(1, 2))
    return (per_transition * weights).sum() / weights.sum()


def grouped_loss(
    model: StructuredGenerator,
    dataset: TrajectorySet,
    weights: torch.Tensor,
    *,
    chunk_trajectories: int,
) -> torch.Tensor:
    ax, ay, diffusion = model.matrices()
    grouped_weights = weights.reshape(dataset.coefficients.shape[0], -1)
    numerator = model.ax_upper.new_zeros(())
    for start in range(0, dataset.coefficients.shape[0], chunk_trajectories):
        stop = min(start + chunk_trajectories, dataset.coefficients.shape[0])
        parameters = dataset.parameters[start:stop]
        generator = (
            parameters[:, 0, None, None] * ax
            + parameters[:, 1, None, None] * ay
            + parameters[:, 2, None, None] * diffusion
        )
        transition = torch.matrix_exp(parameters[:, 3, None, None] * generator)
        inputs = dataset.coefficients[start:stop, :-1]
        targets = dataset.coefficients[start:stop, 1:]
        evolved = transition[:, None] @ inputs[:, :, :49]
        outputs = torch.cat((evolved, inputs[:, :, 49:].clone()), dim=2)
        per_transition = (
            ((outputs - targets) / modal_scales().view(1, 1, 52, 1)).square().mean(dim=(2, 3))
        )
        numerator = numerator + (per_transition * grouped_weights[start:stop]).sum()
    return numerator / weights.sum()


def _gradient_vector(loss: torch.Tensor, model: nn.Module) -> torch.Tensor:
    parameters = list(model.parameters())
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    return torch.cat(
        [
            (torch.zeros_like(parameter) if gradient is None else gradient).reshape(-1)
            for parameter, gradient in zip(parameters, gradients, strict=True)
        ]
    )


def _difference(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    absolute = (left - right).abs()
    denominator = torch.maximum(left.abs(), right.abs()).clamp_min(1e-300)
    report = {
        "maximum_absolute": float(absolute.max().item()) if absolute.numel() else 0.0,
        "maximum_relative": (
            float((absolute / denominator).max().item()) if absolute.numel() else 0.0
        ),
    }
    report["passed"] = bool(torch.allclose(left, right, rtol=1e-14, atol=1e-14))
    return report


def source_probe() -> StructuredGenerator:
    model = StructuredGenerator()
    with torch.no_grad():
        model.ax_upper.zero_()
        model.ay_upper.zero_()
        model.diffusion_log_rate.zero_()
        model.ax_upper[0] = 0.125
        model.ay_upper[47] = -0.25
        model.diffusion_log_rate[7] = math.log(0.375)
    return model


def equivalence_probe(
    model: StructuredGenerator | FixedGenerator,
    datasets: dict[str, TrajectorySet],
    counts: dict[str, torch.Tensor],
    schedule_tensors: dict[str, torch.Tensor],
    *,
    chunk_trajectories: int,
) -> dict[str, Any]:
    trainable = any(True for _ in model.parameters())
    regimes = {}
    for regime in REGIMES:
        dataset = datasets[regime]
        literal_inputs, targets, parameters = dataset.transitions
        grouped = grouped_outputs(model, dataset, chunk_trajectories=chunk_trajectories)
        literal = model(literal_inputs, parameters)
        ones = torch.ones(2048, dtype=torch.float64)
        weighted_grouped = loss_from_outputs(grouped, targets, counts[regime])
        weighted_literal = loss_from_outputs(literal, targets, counts[regime])
        per_transition = ((literal - targets) / modal_scales()).square().mean(dim=(1, 2))
        literal_schedule = per_transition[schedule_tensors[regime].reshape(-1)].mean()
        uniform_grouped = loss_from_outputs(grouped, targets, ones)
        uniform_e13 = normalized_loss(literal, targets)
        record = {
            "outputs": _difference(grouped, literal),
            "weighted_losses": _difference(
                weighted_grouped.reshape(1), weighted_literal.reshape(1)
            ),
            "literal_schedule_losses": _difference(
                weighted_grouped.reshape(1), literal_schedule.reshape(1)
            ),
            "all_ones_e13_losses": _difference(uniform_grouped.reshape(1), uniform_e13.reshape(1)),
        }
        if trainable:
            record["weighted_gradients"] = _difference(
                _gradient_vector(weighted_grouped, model),
                _gradient_vector(literal_schedule, model),
            )
            record["all_ones_e13_gradients"] = _difference(
                _gradient_vector(uniform_grouped, model),
                _gradient_vector(uniform_e13, model),
            )
        record["passed"] = all(value["passed"] for key, value in record.items() if key != "passed")
        regimes[regime] = record
    return {
        "trainable": trainable,
        "regimes": regimes,
        "passed": all(record["passed"] for record in regimes.values()),
    }


def objective_integrity(
    neutral: StructuredGenerator,
    replay: StructuredGenerator,
    oracle: FixedGenerator,
    datasets: dict[str, TrajectorySet],
    counts: dict[str, torch.Tensor],
    schedule_tensors: dict[str, torch.Tensor],
    *,
    chunk_trajectories: int,
) -> dict[str, Any]:
    probes = {
        "neutral": equivalence_probe(
            copy.deepcopy(neutral),
            datasets,
            counts,
            schedule_tensors,
            chunk_trajectories=chunk_trajectories,
        ),
        "e12_checkpoint": equivalence_probe(
            copy.deepcopy(replay),
            datasets,
            counts,
            schedule_tensors,
            chunk_trajectories=chunk_trajectories,
        ),
        "oracle": equivalence_probe(
            oracle,
            datasets,
            counts,
            schedule_tensors,
            chunk_trajectories=chunk_trajectories,
        ),
        "source_probe": equivalence_probe(
            source_probe(),
            datasets,
            counts,
            schedule_tensors,
            chunk_trajectories=chunk_trajectories,
        ),
    }
    separability = {}
    for name, base in (("neutral", neutral), ("e12_checkpoint", replay)):
        model = copy.deepcopy(base)
        cross = {}
        regime_losses = []
        literal_losses = []
        for regime in REGIMES:
            loss = grouped_loss(
                model,
                datasets[regime],
                counts[regime].to(torch.float64),
                chunk_trajectories=chunk_trajectories,
            )
            gradients = torch.autograd.grad(loss, tuple(model.parameters()), retain_graph=True)
            names = ("A_x", "A_y", "D")
            active = {"x_advection": "A_x", "y_advection": "A_y", "diffusion": "D"}[regime]
            cross[regime] = {
                block: float(gradient.abs().max().item())
                for block, gradient in zip(names, gradients, strict=True)
                if block != active
            }
            regime_losses.append(loss / 3.0)
            inputs, targets, parameters = datasets[regime].transitions
            literal_losses.append(
                loss_from_outputs(
                    model(inputs, parameters),
                    targets,
                    counts[regime].to(torch.float64),
                )
                / 3.0
            )
        grouped_joint = sum(regime_losses)
        literal_joint = sum(literal_losses)
        gradient_match = _difference(
            _gradient_vector(grouped_joint, model),
            _gradient_vector(literal_joint, model),
        )
        cross_zero = all(value == 0.0 for values in cross.values() for value in values.values())
        separability[name] = {
            "cross_block_maximum_absolute_gradients": cross,
            "cross_block_exact_zero": cross_zero,
            "joint_gradient_match": gradient_match,
            "passed": cross_zero and gradient_match["passed"],
        }
    return {
        "probes": probes,
        "separability": separability,
        "passed": all(value["passed"] for value in probes.values())
        and all(value["passed"] for value in separability.values()),
    }


def sealed_ceiling_integrity() -> dict[str, Any]:
    manifest = json.loads(E14_MANIFEST_PATH.read_text())
    compact = json.loads(E14_RESULT_PATH.read_text())
    raw_members = {}
    metadata_checks = []
    with tarfile.open(E14_BUNDLE_PATH, mode="r:gz") as archive:
        members = archive.getmembers()
        expected = manifest["outputs"]["evidence_bundle"]["members"]
        metadata_checks.append(
            [member.name for member in members] == [record["name"] for record in expected]
        )
        for member, record in zip(members, expected, strict=True):
            stream = archive.extractfile(member)
            if stream is None:
                raise RuntimeError(f"unreadable E14 member: {member.name}")
            value = stream.read()
            raw_members[member.name] = value
            metadata_checks.append(
                all(
                    (
                        member.isfile(),
                        member.pax_headers == {},
                        member.mode == 0o644,
                        member.uid == member.gid == member.mtime == 0,
                        member.uname == member.gname == "",
                        len(value) == record["bytes"],
                        sha256_bytes(value) == record["raw_sha256"],
                    )
                )
            )
    raw = json.loads(raw_members["replicate_a_result.json"])
    controls = {}
    for control in EXPECTED_E13_EVALUATION_HASHES:
        evaluation = raw["evaluations"][control]
        evaluation_hash = sha256_bytes(canonical_bytes(evaluation))
        recomputed_pass, recomputed_gates = recompute_recovery_pass(control, evaluation)
        expected_model = EXPECTED_E13_MODELS[control]
        controls[control] = {
            "evaluation_sha256": evaluation_hash,
            "expected_evaluation_sha256": EXPECTED_E13_EVALUATION_HASHES[control],
            "model_sha256": evaluation["model_sha256"],
            "generator_sha256": evaluation["generator_sha256"],
            "recomputed_gates": recomputed_gates,
            "recomputed_pass": recomputed_pass,
            "stored_gates_match": recomputed_gates == evaluation["gates"],
            "stored_pass_matches": recomputed_pass == evaluation["recovery_pass"],
            "passed": all(
                (
                    evaluation_hash == EXPECTED_E13_EVALUATION_HASHES[control],
                    evaluation["model_sha256"] == expected_model["model_sha256"],
                    evaluation["generator_sha256"] == expected_model["generator_sha256"],
                    recomputed_gates == evaluation["gates"],
                    recomputed_pass is True,
                    evaluation["recovery_pass"] is True,
                )
            ),
        }
    boundary = compact["e14_boundary"]
    zero_boundary = all(
        boundary[key] == 0
        for key in (
            "state_reads",
            "optimizer_updates",
            "heldout_reads",
            "provider_calls",
            "routing_paths",
        )
    )
    checks = {
        "bundle_hash": sha256_path(E14_BUNDLE_PATH) == EXPECTED_E14_HASHES["bundle"],
        "result_hash": sha256_path(E14_RESULT_PATH) == EXPECTED_E14_HASHES["result"],
        "manifest_hash": sha256_path(E14_MANIFEST_PATH) == EXPECTED_E14_HASHES["manifest"],
        "member_metadata_and_hashes": all(metadata_checks),
        "replicates_identical": (
            raw_members["replicate_a_result.json"] == raw_members["replicate_b_result.json"]
        ),
        "seal_classification": compact["classification"] == "e13_scientific_result_sealed",
        "underlying_classification": (
            compact["underlying_e13_classification"]
            == "full_parameterization_deterministic_recovery_succeeds"
        ),
        "sealing_head": (compact["e14_sealing_head"] == "63e23d50f2eef3ca2644100674c10c4e2aa3a5ba"),
        "independent_sections": compact["independent_recompute"]["passed"] is True,
        "zero_state_boundary": zero_boundary,
        "matched_controls": all(record["passed"] for record in controls.values()),
    }
    return {"checks": checks, "controls": controls, "passed": all(checks.values())}


def _block_gradient_norms(model: StructuredGenerator) -> dict[str, float]:
    values = {}
    for name, parameter in (
        ("A_x", model.ax_upper),
        ("A_y", model.ay_upper),
        ("D", model.diffusion_log_rate),
    ):
        values[name] = (
            0.0
            if parameter.grad is None
            else float(torch.linalg.vector_norm(parameter.grad).item())
        )
    return values


def _block_update_norms(
    model: StructuredGenerator, before: dict[str, torch.Tensor]
) -> dict[str, float]:
    return {
        "A_x": float(torch.linalg.vector_norm(model.ax_upper.detach() - before["A_x"]).item()),
        "A_y": float(torch.linalg.vector_norm(model.ay_upper.detach() - before["A_y"]).item()),
        "D": float(
            torch.linalg.vector_norm(model.diffusion_log_rate.detach() - before["D"]).item()
        ),
    }


def _clone_blocks(model: StructuredGenerator) -> dict[str, torch.Tensor]:
    return {
        "A_x": model.ax_upper.detach().clone(),
        "A_y": model.ay_upper.detach().clone(),
        "D": model.diffusion_log_rate.detach().clone(),
    }


def _objective_backward(
    model: StructuredGenerator,
    datasets: dict[str, TrajectorySet],
    counts: dict[str, torch.Tensor],
    *,
    chunk_trajectories: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    model.zero_grad(set_to_none=True)
    losses = {}
    total = model.ax_upper.new_zeros(())
    for regime in REGIMES:
        loss = grouped_loss(
            model,
            datasets[regime],
            counts[regime].to(torch.float64),
            chunk_trajectories=chunk_trajectories,
        )
        (loss / 3.0).backward()
        losses[regime] = float(loss.item())
        total = total + loss.detach() / 3.0
    return total, losses


def _semantic_frequency_table(
    model: StructuredGenerator,
    oracle: FixedGenerator,
    space: PhysicalFunctionSpace,
    canonical_coords: torch.Tensor,
) -> list[dict[str, Any]]:
    records, _ = mode_resolved_records(
        {"trace": model},
        oracle=oracle,
        space=space,
        canonical_coords=canonical_coords,
    )
    one_step = [record for record in records if record["horizon"] == 1]
    table = []
    for fx in range(4):
        for fy in range(4):
            subset = [
                record
                for record in one_step
                if SEMANTIC_FREQUENCIES[record["basis_index"] // 7] == fx
                and SEMANTIC_FREQUENCIES[record["basis_index"] % 7] == fy
            ]
            winner = max(subset, key=lambda record: record["decoded_nrmse"])
            table.append(
                {
                    "f_x": fx,
                    "f_y": fy,
                    "maximum_decoded_basis_action_nrmse": winner["decoded_nrmse"],
                    "basis_index": winner["basis_index"],
                    "case_name": winner["case_name"],
                    "horizon": 1,
                }
            )
    return table


def frozen_validation_trace_scope() -> dict[str, Any]:
    return {
        "validation_case_names": list(VALIDATION_CASE_NAMES),
        "validation_horizons": [1, 8],
    }


def trace_diagnostic(
    model: StructuredGenerator,
    *,
    step: int,
    weighted_loss: float,
    weighted_regime_losses: dict[str, float],
    pre_gradient_norms: dict[str, float],
    post_gradient_norms: dict[str, float],
    update_norms: dict[str, float],
    datasets: dict[str, TrajectorySet],
    counts: dict[str, torch.Tensor],
    validation: dict[str, TrajectorySet],
    oracle: FixedGenerator,
    space: PhysicalFunctionSpace,
    canonical_coords: torch.Tensor,
    cfg: E15Config,
) -> dict[str, Any]:
    uniform_losses = {
        regime: float(
            grouped_loss(
                model,
                datasets[regime],
                torch.ones(2048, dtype=torch.float64),
                chunk_trajectories=cfg.group_chunk_trajectories,
            ).item()
        )
        for regime in REGIMES
    }
    evaluation = evaluate_control(
        model,
        validation,
        oracle=oracle,
        space=space,
        canonical_coords=canonical_coords,
        cfg=cfg,
    )
    return {
        "step": step,
        "schedule_weighted_loss": weighted_loss,
        "uniform_loss": sum(uniform_losses.values()) / 3.0,
        "weighted_regime_losses": weighted_regime_losses,
        "uniform_regime_losses": uniform_losses,
        "pre_update_gradient_norms": pre_gradient_norms,
        "post_update_gradient_norms": post_gradient_norms,
        "update_norms": update_norms,
        "relative_generator_frobenius": evaluation["generator_identification"][
            "relative_frobenius"
        ],
        "maximum_decoded_basis_action_nrmse": evaluation["generator_identification"][
            "maximum_basis_action_decoded_nrmse"
        ],
        "semantic_frequency_table": _semantic_frequency_table(
            model, oracle, space, canonical_coords
        ),
        "composite_rollout_decoded_nrmse": evaluation["validation"]["composite"][
            "rollout_decoded_nrmse"
        ],
        "composite_final_high_frequency_nrmse": evaluation["validation"]["composite"][
            "final_high_frequency_spectral"
        ]["nrmse"],
        **frozen_validation_trace_scope(),
    }


def adamw_state_record(optimizer: torch.optim.AdamW) -> dict[str, Any]:
    state = optimizer.state_dict()
    params = state["param_groups"][0]
    constructor = {
        key: params[key]
        for key in (
            "lr",
            "betas",
            "eps",
            "weight_decay",
            "amsgrad",
            "maximize",
            "foreach",
            "capturable",
            "differentiable",
            "fused",
        )
    }
    record = {
        "constructor": constructor,
        "parameter_state_entries": len(state["state"]),
        "complete_state": state,
    }
    record["canonical_sha256"] = sha256_bytes(canonical_bytes(state))
    return record


def train_adamw(
    model: StructuredGenerator,
    datasets: dict[str, TrajectorySet],
    counts: dict[str, torch.Tensor],
    validation: dict[str, TrajectorySet],
    *,
    oracle: FixedGenerator,
    space: PhysicalFunctionSpace,
    canonical_coords: torch.Tensor,
    cfg: E15Config,
) -> dict[str, Any]:
    initial_model_sha256 = model_hash(model)
    initial_generator_sha256 = _matrix_hashes(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.pretrain_learning_rate,
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
    pre_state = adamw_state_record(optimizer)
    if pre_state["parameter_state_entries"] != 0:
        raise RuntimeError("E15 fresh AdamW unexpectedly has moment state")
    traces = []
    initial_loss, initial_regime = _objective_backward(
        model,
        datasets,
        counts,
        chunk_trajectories=cfg.group_chunk_trajectories,
    )
    initial_gradients = _block_gradient_norms(model)
    traces.append(
        trace_diagnostic(
            model,
            step=0,
            weighted_loss=float(initial_loss.item()),
            weighted_regime_losses=initial_regime,
            pre_gradient_norms=initial_gradients,
            post_gradient_norms=initial_gradients,
            update_norms={"A_x": 0.0, "A_y": 0.0, "D": 0.0},
            datasets=datasets,
            counts=counts,
            validation=validation,
            oracle=oracle,
            space=space,
            canonical_coords=canonical_coords,
            cfg=cfg,
        )
    )
    for step in range(1, cfg.pretrain_updates + 1):
        before = _clone_blocks(model)
        loss, regime_losses = _objective_backward(
            model,
            datasets,
            counts,
            chunk_trajectories=cfg.group_chunk_trajectories,
        )
        pre_gradients = _block_gradient_norms(model)
        optimizer.step()
        update_norms = _block_update_norms(model, before)
        if step in TRACE_STEPS:
            post_loss, post_regime = _objective_backward(
                model,
                datasets,
                counts,
                chunk_trajectories=cfg.group_chunk_trajectories,
            )
            post_gradients = _block_gradient_norms(model)
            traces.append(
                trace_diagnostic(
                    model,
                    step=step,
                    weighted_loss=float(post_loss.item()),
                    weighted_regime_losses=post_regime,
                    pre_gradient_norms=pre_gradients,
                    post_gradient_norms=post_gradients,
                    update_norms=update_norms,
                    datasets=datasets,
                    counts=counts,
                    validation=validation,
                    oracle=oracle,
                    space=space,
                    canonical_coords=canonical_coords,
                    cfg=cfg,
                )
            )
        if not _all_finite(
            {
                "loss": float(loss.item()),
                "regime_losses": regime_losses,
                "pre_gradients": pre_gradients,
                "updates": update_norms,
            }
        ):
            raise E15IncompleteError("nonfinite_optimizer_state")
    return {
        "updates": cfg.pretrain_updates,
        "initial_model_sha256": initial_model_sha256,
        "initial_generator_sha256": initial_generator_sha256,
        "initial_optimizer_state": pre_state,
        "traces": traces,
        "trace_steps": [record["step"] for record in traces],
    }


def optimize_components_weighted(
    model: StructuredGenerator,
    datasets: dict[str, TrajectorySet],
    counts: dict[str, torch.Tensor],
    *,
    validation: dict[str, TrajectorySet],
    oracle: FixedGenerator,
    space: PhysicalFunctionSpace,
    canonical_coords: torch.Tensor,
    cfg: E15Config,
) -> dict[str, Any]:
    def diagnostic() -> dict[str, Any]:
        weighted_tensors = {
            name: grouped_loss(
                model,
                datasets[name],
                counts[name].to(torch.float64),
                chunk_trajectories=cfg.group_chunk_trajectories,
            )
            for name in REGIMES
        }
        weighted_objective = sum(weighted_tensors.values()) / 3.0
        gradients = torch.autograd.grad(weighted_objective, tuple(model.parameters()))
        gradient_norms = {
            name: float(torch.linalg.vector_norm(gradient).item())
            for name, gradient in zip(("A_x", "A_y", "D"), gradients, strict=True)
        }
        weighted_losses = {name: float(value.item()) for name, value in weighted_tensors.items()}
        uniform_losses = {
            name: float(
                grouped_loss(
                    model,
                    datasets[name],
                    torch.ones(2048, dtype=torch.float64),
                    chunk_trajectories=cfg.group_chunk_trajectories,
                ).item()
            )
            for name in REGIMES
        }
        evaluation = evaluate_control(
            model,
            validation,
            oracle=oracle,
            space=space,
            canonical_coords=canonical_coords,
            cfg=cfg,
        )
        return {
            "schedule_weighted_loss": sum(weighted_losses.values()) / 3.0,
            "uniform_loss": sum(uniform_losses.values()) / 3.0,
            "weighted_regime_losses": weighted_losses,
            "uniform_regime_losses": uniform_losses,
            "gradient_norms": gradient_norms,
            "relative_generator_frobenius": evaluation["generator_identification"][
                "relative_frobenius"
            ],
            "maximum_decoded_basis_action_nrmse": evaluation["generator_identification"][
                "maximum_basis_action_decoded_nrmse"
            ],
            "semantic_frequency_table": _semantic_frequency_table(
                model, oracle, space, canonical_coords
            ),
            "composite_rollout_decoded_nrmse": evaluation["validation"]["composite"][
                "rollout_decoded_nrmse"
            ],
            "composite_final_high_frequency_nrmse": evaluation["validation"]["composite"][
                "final_high_frequency_spectral"
            ]["nrmse"],
            **frozen_validation_trace_scope(),
        }

    reports = {}
    for regime in REGIMES:
        component = {"x_advection": "A_x", "y_advection": "A_y", "diffusion": "D"}[regime]
        optimized = component_parameters(model, component)
        optimizer = torch.optim.LBFGS(
            optimized,
            lr=cfg.lbfgs_learning_rate,
            max_iter=cfg.lbfgs_max_iter,
            max_eval=cfg.lbfgs_max_eval,
            tolerance_grad=cfg.lbfgs_tolerance_grad,
            tolerance_change=cfg.lbfgs_tolerance_change,
            history_size=cfg.lbfgs_history_size,
            line_search_fn="strong_wolfe",
        )
        history = []

        def closure(
            optimizer: torch.optim.LBFGS = optimizer,
            regime: str = regime,
            optimized: list[nn.Parameter] = optimized,
            history: list[dict[str, Any]] = history,
        ) -> torch.Tensor:
            optimizer.zero_grad(set_to_none=True)
            loss = grouped_loss(
                model,
                datasets[regime],
                counts[regime].to(torch.float64),
                chunk_trajectories=cfg.group_chunk_trajectories,
            )
            loss.backward()
            gradient = torch.cat([parameter.grad.reshape(-1) for parameter in optimized])
            history.append(
                {
                    "closure": len(history),
                    "loss": float(loss.item()),
                    "gradient_norm": float(torch.linalg.vector_norm(gradient).item()),
                }
            )
            return loss

        block_before = _clone_blocks(model)
        before = {
            "model_sha256": model_hash(model),
            "generator_sha256": _matrix_hashes(model),
            "diagnostic": diagnostic(),
            "update_norms": {"A_x": 0.0, "A_y": 0.0, "D": 0.0},
        }
        optimizer.step(closure)
        state = optimizer.state[optimized[0]]
        reports[component] = {
            "optimizer_constructor": {
                "lr": cfg.lbfgs_learning_rate,
                "max_iter": cfg.lbfgs_max_iter,
                "max_eval": cfg.lbfgs_max_eval,
                "tolerance_grad": cfg.lbfgs_tolerance_grad,
                "tolerance_change": cfg.lbfgs_tolerance_change,
                "history_size": cfg.lbfgs_history_size,
                "line_search_fn": "strong_wolfe",
            },
            "before": before,
            "after": {
                "model_sha256": model_hash(model),
                "generator_sha256": _matrix_hashes(model),
                "update_norms": _block_update_norms(model, block_before),
                "diagnostic": diagnostic(),
            },
            "n_iter": int(state["n_iter"]),
            "func_evals": int(state["func_evals"]),
            "closure_count": len(history),
            "history": history,
        }
    return reports


def classify(
    *,
    preflight_passed: bool,
    reproduction_passed: bool,
    integrity_passed: bool,
    execution_complete: bool,
    recovery: dict[str, bool],
) -> str:
    if not preflight_passed:
        return "e15_preflight_failed"
    if not reproduction_passed:
        return "e12_reproduction_failed"
    if not integrity_passed:
        return "frozen_ceiling_or_objective_integrity_failed"
    if not execution_complete:
        return "e15_execution_incomplete"
    if recovery["schedule_weighted_adamw_neutral"]:
        return "deterministic_objective_adamw_succeeds_from_neutral"
    if recovery["schedule_weighted_adamw_restart"]:
        return "deterministic_objective_adamw_restart_repairs_e12_checkpoint_only"
    if recovery["schedule_weighted_componentwise_lbfgs_neutral"]:
        return "componentwise_strong_wolfe_lbfgs_package_succeeds_from_neutral"
    if recovery["schedule_weighted_componentwise_lbfgs_restart"]:
        return "componentwise_strong_wolfe_lbfgs_restart_repairs_e12_checkpoint_only"
    return "uniform_population_weighting_required_under_frozen_componentwise_controls"


def _reproduction(
    initial: StructuredGenerator,
    replay: StructuredGenerator,
    replay_training: dict[str, Any],
    validation: dict[str, TrajectorySet],
    oracle: FixedGenerator,
    space: PhysicalFunctionSpace,
    canonical_coords: torch.Tensor,
    cfg: E15Config,
) -> dict[str, Any]:
    from scripts.run_canonical_latent_e11_coefficient_operator_transfer import rollout
    from scripts.run_canonical_latent_e12_structured_generator import (
        generator_identification,
    )

    identification = generator_identification(
        replay, oracle, space=space, canonical_coords=canonical_coords
    )
    composite, _ = rollout(
        RuleAdapter(replay, "combined"),
        validation["composite"],
        space=space,
        canonical_coords=canonical_coords,
        cfg=cfg,
    )
    checks = {
        "initial_model_sha256": model_hash(initial) == EXPECTED_E12_INITIAL_SHA256,
        "elementary_model_sha256": model_hash(replay) == EXPECTED_E12_ELEMENTARY_SHA256,
        "generator_sha256": _matrix_hashes(replay) == EXPECTED_E12_GENERATOR_SHA256,
        "first_loss": replay_training["first_loss"] == EXPECTED_E12_FIRST_LOSS,
        "final_loss": replay_training["final_loss"] == EXPECTED_E12_FINAL_LOSS,
        "updates": replay_training["updates"] == 1500,
        "examples_per_regime": replay_training["examples_per_regime"] == 48000,
        "basis_action": abs(
            identification["maximum_basis_action_decoded_nrmse"] - EXPECTED_E12_BASIS_ACTION
        )
        <= 1e-12,
        "zero_shot_rollout": abs(
            composite["rollout_decoded_nrmse"] - EXPECTED_E12_ZERO_SHOT_ROLLOUT
        )
        <= 1e-12,
    }
    return {
        "checks": checks,
        "training": replay_training,
        "initial_model_sha256": model_hash(initial),
        "elementary_model_sha256": model_hash(replay),
        "generator_sha256": _matrix_hashes(replay),
        "basis_action_decoded_nrmse": identification["maximum_basis_action_decoded_nrmse"],
        "zero_shot_rollout_decoded_nrmse": composite["rollout_decoded_nrmse"],
        "passed": all(checks.values()),
    }


def run_replicate(
    cfg: E15Config,
    *,
    environment: dict[str, Any],
    provenance_report: dict[str, Any],
) -> dict[str, Any]:
    if asdict(cfg) != asdict(frozen_e15_config()):
        raise ValueError("E15 requires the exact frozen configuration")
    if not provenance_report["passed"]:
        return {
            "schema_version": 1,
            "experiment": "canonical_latent_e15_training_package",
            "config": asdict(cfg),
            "provenance": provenance_report,
            "classification": "e15_preflight_failed",
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
            "experiment": "canonical_latent_e15_training_package",
            "config": asdict(cfg),
            "provenance": provenance_report,
            "oracle_preflight": oracle_report,
            "classification": "e15_preflight_failed",
            "state_reads": {"training": 0, "validation": 0, "heldout": 0},
            "boundary": _boundary(),
        }
    training, validation = build_frozen_datasets(cfg, space)
    schedule_tensors = schedules(cfg)
    replay_lock = replay_lock_report(training, validation, schedule_tensors)
    counts, count_report = occurrence_counts(schedule_tensors)
    ceiling = sealed_ceiling_integrity()
    if not replay_lock["passed"]:
        return {
            "schema_version": 1,
            "experiment": "canonical_latent_e15_training_package",
            "config": asdict(cfg),
            "provenance": provenance_report,
            "oracle_preflight": oracle_report,
            "replay_lock": replay_lock,
            "occurrence_counts": count_report,
            "sealed_e13_ceilings": ceiling,
            "classification": "e12_reproduction_failed",
            "state_reads": {"training": 768, "validation": 256, "heldout": 0},
            "boundary": _boundary(),
        }
    initial = StructuredGenerator()
    replay = copy.deepcopy(initial)
    replay_training = train_elementary(
        replay,
        training,
        {name: schedule_tensors[name] for name in REGIMES},
        cfg,
    )
    canonical_coords, _ = canonical_grid(space, cfg.canonical_query_resolution)
    reproduction = _reproduction(
        initial,
        replay,
        replay_training,
        validation,
        oracle,
        space,
        canonical_coords,
        cfg,
    )
    integrity = objective_integrity(
        initial,
        replay,
        oracle,
        training,
        counts,
        schedule_tensors,
        chunk_trajectories=cfg.group_chunk_trajectories,
    )
    integrity_passed = all(
        (
            oracle_report["passed"],
            replay_lock["passed"],
            count_report["passed"],
            ceiling["passed"],
            integrity["passed"],
        )
    )
    if not reproduction["passed"] or not integrity_passed:
        classification = classify(
            preflight_passed=True,
            reproduction_passed=reproduction["passed"],
            integrity_passed=integrity_passed,
            execution_complete=False,
            recovery={name: False for name in NEW_CONTROLS},
        )
        return {
            "schema_version": 1,
            "experiment": "canonical_latent_e15_training_package",
            "config": asdict(cfg),
            "provenance": provenance_report,
            "oracle_preflight": oracle_report,
            "replay_lock": replay_lock,
            "occurrence_counts": count_report,
            "e12_reproduction": reproduction,
            "sealed_e13_ceilings": ceiling,
            "objective_integrity": integrity,
            "classification": classification,
            "state_reads": {"training": 768, "validation": 256, "heldout": 0},
            "boundary": _boundary(),
        }

    adam_neutral = StructuredGenerator()
    adam_restart = copy.deepcopy(replay)
    lbfgs_neutral = StructuredGenerator()
    lbfgs_restart = copy.deepcopy(replay)
    training_reports = {
        "e12_adamw_replay": replay_training,
        "schedule_weighted_adamw_neutral": train_adamw(
            adam_neutral,
            training,
            counts,
            validation,
            oracle=oracle,
            space=space,
            canonical_coords=canonical_coords,
            cfg=cfg,
        ),
        "schedule_weighted_adamw_restart": train_adamw(
            adam_restart,
            training,
            counts,
            validation,
            oracle=oracle,
            space=space,
            canonical_coords=canonical_coords,
            cfg=cfg,
        ),
        "schedule_weighted_componentwise_lbfgs_neutral": optimize_components_weighted(
            lbfgs_neutral,
            training,
            counts,
            validation=validation,
            oracle=oracle,
            space=space,
            canonical_coords=canonical_coords,
            cfg=cfg,
        ),
        "schedule_weighted_componentwise_lbfgs_restart": optimize_components_weighted(
            lbfgs_restart,
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
        "e12_adamw_replay": replay,
        "schedule_weighted_adamw_neutral": adam_neutral,
        "schedule_weighted_adamw_restart": adam_restart,
        "schedule_weighted_componentwise_lbfgs_neutral": lbfgs_neutral,
        "schedule_weighted_componentwise_lbfgs_restart": lbfgs_restart,
        "oracle": oracle,
    }
    evaluations = {
        name: evaluate_control(
            model,
            validation,
            oracle=oracle,
            space=space,
            canonical_coords=canonical_coords,
            cfg=cfg,
        )
        for name, model in controls.items()
    }
    mode_records, mode_argmax = mode_resolved_records(
        controls,
        oracle=oracle,
        space=space,
        canonical_coords=canonical_coords,
    )
    coverage = coverage_report(evaluations, mode_records, mode_argmax, training_reports)
    recovery = {name: evaluations[name]["recovery_pass"] for name in NEW_CONTROLS}
    execution_complete = coverage["passed"] and _all_finite(
        {
            "training": training_reports,
            "evaluations": evaluations,
            "mode_records": mode_records,
            "mode_argmax": mode_argmax,
        }
    )
    classification = classify(
        preflight_passed=True,
        reproduction_passed=True,
        integrity_passed=True,
        execution_complete=execution_complete,
        recovery=recovery,
    )
    result = {
        "schema_version": 1,
        "experiment": "canonical_latent_e15_training_package",
        "config": asdict(cfg),
        "config_sha256": sha256_bytes(canonical_bytes(asdict(cfg))),
        "provenance": provenance_report,
        "oracle_preflight": oracle_report,
        "replay_lock": replay_lock,
        "occurrence_counts": count_report,
        "e12_reproduction": reproduction,
        "sealed_e13_ceilings": ceiling,
        "objective_integrity": integrity,
        "training": training_reports,
        "evaluations": evaluations,
        "mode_resolved": mode_records,
        "mode_argmax": mode_argmax,
        "coverage": coverage,
        "classification": classification,
        "state_reads": {"training": 768, "validation": 256, "heldout": 0},
        "boundary": _boundary(),
        "reproducibility": environment,
    }
    if not _all_finite(result):
        raise E15IncompleteError("nonfinite_result")
    return result


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
        {
            "name": name,
            "bytes": len(raw[name]),
            "raw_sha256": sha256_bytes(raw[name]),
        }
        for name in RAW_MEMBERS
    ]
    return output.getvalue(), members


def verify_bundle(bundle: bytes, expected: list[dict[str, Any]]) -> None:
    if bundle[:3] != b"\x1f\x8b\x08" or bundle[3] & 0x08 or bundle[4:8] != b"\0\0\0\0":
        raise RuntimeError("E15 gzip header is not deterministic")
    with tarfile.open(fileobj=io.BytesIO(bundle), mode="r:gz") as archive:
        members = archive.getmembers()
        if [member.name for member in members] != list(RAW_MEMBERS):
            raise RuntimeError("E15 archive member set or order changed")
        for member, record in zip(members, expected, strict=True):
            stream = archive.extractfile(member)
            if stream is None:
                raise RuntimeError(f"E15 archive member unreadable: {member.name}")
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
                raise RuntimeError(f"E15 archive metadata/hash mismatch: {member.name}")


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
        "config_sha256": complete.get("config_sha256"),
        "source_files": complete["provenance"]["source_files"],
        "evidence_bundle": bundle_record,
        "raw_and_canonical_hashes": complete["replication"],
        "replay_lock": complete.get("replay_lock"),
        "occurrence_counts": complete.get("occurrence_counts"),
        "e12_reproduction": complete.get("e12_reproduction"),
        "sealed_e13_ceilings": complete.get("sealed_e13_ceilings"),
        "objective_integrity": complete.get("objective_integrity"),
        "training": complete.get("training"),
        "evaluations": complete.get("evaluations"),
        "mode_argmax": complete.get("mode_argmax"),
        "coverage": complete.get("coverage"),
        "boundary": complete["boundary"],
        "state_reads": complete["state_reads"],
        "reproducibility": complete.get("reproducibility"),
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
    outputs = {
        OUTPUT_NAMES["bundle"]: bundle,
        OUTPUT_NAMES["result"]: compact_bytes,
        OUTPUT_NAMES["manifest"]: pretty_bytes(manifest),
    }
    if "raw_sha256" in compact or "bytes" in compact:
        raise RuntimeError("E15 compact result declares a self hash or byte count")
    return outputs


def verify_outputs(outputs: dict[str, bytes]) -> None:
    if set(outputs) != set(OUTPUT_NAMES.values()):
        raise RuntimeError("E15 output filename set changed")
    manifest = json.loads(outputs[OUTPUT_NAMES["manifest"]])
    compact = json.loads(outputs[OUTPUT_NAMES["result"]])
    result_record = manifest["outputs"]["compact_result"]
    bundle_record = manifest["outputs"]["evidence_bundle"]
    if result_record["raw_sha256"] != sha256_bytes(outputs[OUTPUT_NAMES["result"]]):
        raise RuntimeError("E15 compact hash mismatch")
    if result_record["bytes"] != len(outputs[OUTPUT_NAMES["result"]]):
        raise RuntimeError("E15 compact byte count mismatch")
    if bundle_record["raw_sha256"] != sha256_bytes(outputs[OUTPUT_NAMES["bundle"]]):
        raise RuntimeError("E15 bundle hash mismatch")
    if bundle_record["bytes"] != len(outputs[OUTPUT_NAMES["bundle"]]):
        raise RuntimeError("E15 bundle byte count mismatch")
    if "raw_sha256" in compact or "bytes" in compact:
        raise RuntimeError("E15 compact self hash detected")
    if "manifest" in manifest.get("outputs", {}):
        raise RuntimeError("E15 manifest self hash detected")
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
        os.write(descriptor, b"e15-publication-lock\n")
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        if os.path.lexists(output_dir):
            raise FileExistsError(f"E15 output already exists: {output_dir}")
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
            raise FileExistsError(f"E15 output raced before publication: {output_dir}")
        os.rename(stage, output_dir)
        stage = None
        published = True
        _fsync_directory(parent)
    except BaseException as error:
        failure = error
    try:
        if descriptor is not None:
            os.close(descriptor)
            descriptor = None
        if stage is not None:
            _remove_path(stage)
            stage = None
        if os.path.lexists(lock):
            lock.unlink()
            _fsync_directory(parent)
    except BaseException as cleanup_error:
        if failure is None:
            failure = cleanup_error
    if failure is not None:
        rollback_error: BaseException | None = None
        try:
            if stage is not None and os.path.lexists(stage):
                _remove_path(stage)
                _fsync_directory(parent)
            if published:
                _remove_path(output_dir)
                _fsync_directory(parent)
            if os.path.lexists(lock):
                lock.unlink()
                _fsync_directory(parent)
        except BaseException as error:
            rollback_error = error
        if rollback_error is not None:
            raise RuntimeError("E15 publication rollback failed") from rollback_error
        raise failure


def _run_e15_under_limit(
    cfg: E15Config,
    *,
    output_dir: Path,
    environment: dict[str, Any],
    provenance_report: dict[str, Any],
) -> dict[str, Any]:
    with time_limit(PER_REPLICA_TIMEOUT_SECONDS, "E15 replicate A"):
        first = run_replicate(cfg, environment=environment, provenance_report=provenance_report)
    with time_limit(PER_REPLICA_TIMEOUT_SECONDS, "E15 replicate B"):
        second = run_replicate(cfg, environment=environment, provenance_report=provenance_report)
    first_canonical = canonical_bytes(first)
    second_canonical = canonical_bytes(second)
    if first_canonical != second_canonical:
        raise E15IncompleteError("replication_payload_mismatch")
    first_raw = pretty_bytes(first)
    second_raw = pretty_bytes(second)
    if first_raw != second_raw:
        raise E15IncompleteError("replication_raw_bytes_mismatch")
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
        raise E15IncompleteError("combined_result_mismatch")
    complete["replication"]["combined_without_replication_equals_replicate"] = True
    complete["replication"]["complete_canonical_without_replication_sha256"] = sha256_bytes(
        canonical_bytes(without_replication)
    )
    outputs = build_outputs(first, second, complete)
    try:
        publish_atomic(output_dir, outputs)
    except BaseException as error:
        raise E15IncompleteError("publication_failure") from error
    return complete


def incomplete_status(reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment": "canonical_latent_e15_training_package",
        "classification": "e15_execution_incomplete",
        "incomplete_reason": reason,
        "scientific_conclusion_recorded": False,
        "durable_evidence_published": False,
        "boundary": _boundary(),
    }


def run_e15(cfg: E15Config, *, output_dir: Path) -> dict[str, Any]:
    if output_dir.resolve(strict=False) != CANONICAL_OUTPUT_DIR.resolve(strict=False):
        raise ValueError(
            f"E15 output directory is frozen to {CANONICAL_OUTPUT_DIR.relative_to(REPO_ROOT)}"
        )
    environment = configure_environment(cfg)
    provenance_report = provenance(environment)
    try:
        with time_limit(WHOLE_EXPERIMENT_TIMEOUT_SECONDS, "E15 whole experiment"):
            return _run_e15_under_limit(
                cfg,
                output_dir=output_dir,
                environment=environment,
                provenance_report=provenance_report,
            )
    except E15IncompleteError as error:
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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CANONICAL_OUTPUT_DIR,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_e15(frozen_e15_config(), output_dir=args.output_dir)
    if result["classification"] == "e15_execution_incomplete":
        print(json.dumps(result, sort_keys=True))
        raise SystemExit(2)
    print(json.dumps({"classification": result["classification"]}, sort_keys=True))


if __name__ == "__main__":
    main()
