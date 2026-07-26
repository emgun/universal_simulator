#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from scripts.run_canonical_latent_e7_function_space import (
    FunctionSpaceConfig,
    PhysicalFunctionSpace,
    _decoded_mismatch,
    _relative_mismatch,
)
from scripts.run_canonical_latent_e11_coefficient_operator_transfer import (
    REGIMES,
    E11Config,
    TrajectorySet,
    build_trajectories,
    canonical_grid,
    closure_preflight,
    cross_observation_report,
    evolve_periodic,
    exact_semigroup_consistency,
    model_hash,
    normalized_loss,
    physics_report,
    project_values,
    rollout,
    schedule,
    semigroup_consistency,
    tensor_hash,
    truth_grid,
)
from ups.eval.latent_qualification import global_nrmse

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    REPO_ROOT / "docs/research/2026-07-25-canonical-latent-e12-structured-generator-contract.md"
)
RUNNER_PATH = Path(__file__).resolve()
E11_ARTIFACT_PATH = (
    REPO_ROOT
    / "docs/research/artifacts/canonical_latent_e11_coefficient_operator_transfer_result.json"
)
E11_ARTIFACT_SHA256 = "d9142f94c87d5ffb0b44ff67d665b46b50e529a6d50acee703d15d20e8445f15"
E11_COMPARATORS = {
    "elementary_pretrained_zero_shot_rollout_decoded_nrmse": 2.456423109290437,
    "pretrained_fewshot_rollout_decoded_nrmse": 2.4541693117065217,
    "scratch_fewshot_rollout_decoded_nrmse": 1.5126324830652857,
    "full_composite_control_rollout_decoded_nrmse": 0.9381449992978823,
    "persistence_rollout_decoded_nrmse": 1.5584481380508215,
}
EXPECTED_SCHEDULES = {
    "x_advection": {
        "shape": [1500, 32],
        "sha256": "096659f791a0d5e728ccac7aa02801c60856bd0659a080464bb158909be3f6f7",
    },
    "y_advection": {
        "shape": [1500, 32],
        "sha256": "49401932ae228e58f2927a7695d443a17a1a41c8e80375e4346a06973b431507",
    },
    "diffusion": {
        "shape": [1500, 32],
        "sha256": "d86f8d73a236e806ff1988c274af8fd9a2daa30beeada74e8427d526d5b88487",
    },
    "fine_tune": {
        "shape": [400, 64],
        "sha256": "b0de4bcb3d59866dd05489d7ecf13574dd7763ddf82a6706f72734ec701a7e32",
    },
    "full_control": {
        "shape": [1500, 96],
        "sha256": "602ff2694d9923821782d69627c0b7ece849086abcc431f4ca085de6486519cf",
    },
}
LEARNED_CHECKPOINTS = (
    "structured_elementary_pretrained",
    "structured_pretrained_fewshot",
    "structured_scratch_fewshot",
    "structured_full_composite_control",
)
RULES = ("combined", "splitting")
EVALUATION_ARMS = tuple(
    f"{checkpoint}_{rule}" for checkpoint in LEARNED_CHECKPOINTS for rule in RULES
) + (
    "persistence",
    "exact_projected_truth",
    "oracle_combined",
    "oracle_splitting",
)
ALL_REGIMES = ("composite", *REGIMES)
GEOMETRY_FAMILIES = (
    "grid",
    "warped_mesh",
    "uniform_particles",
    "warped_particles",
)


@dataclass(frozen=True)
class E12Config(E11Config):
    pretrain_learning_rate: float = 2e-2
    fine_tune_learning_rate: float = 5e-3
    weight_decay: float = 0.0
    max_generator_relative_frobenius_error: float = 0.10
    max_supported_entry_relative_error: float = 0.20
    max_off_support_leakage: float = 0.10
    max_diffusion_rate_relative_error: float = 0.20
    max_normalized_commutator: float = 0.02
    max_basis_action_decoded_nrmse: float = 0.05
    max_e11_dense_ratio: float = 0.25
    max_combined_splitting_decoded_mismatch: float = 0.05

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            self.pretrain_learning_rate != 2e-2
            or self.fine_tune_learning_rate != 5e-3
            or self.weight_decay != 0.0
        ):
            raise ValueError("E12 optimization is frozen")


def frozen_e12_config() -> E12Config:
    return E12Config()


class StructuredGenerator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        upper = torch.triu_indices(48, 48, offset=1)
        self.register_buffer("upper_row", upper[0])
        self.register_buffer("upper_col", upper[1])
        self.ax_upper = nn.Parameter(torch.zeros(1128, dtype=torch.float64))
        self.ay_upper = nn.Parameter(torch.zeros(1128, dtype=torch.float64))
        self.diffusion_log_rate = nn.Parameter(torch.zeros(48, dtype=torch.float64))

    def matrices(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        def skew(values: torch.Tensor) -> torch.Tensor:
            result = values.new_zeros((49, 49))
            rows = self.upper_row + 1
            cols = self.upper_col + 1
            result[rows, cols] = values
            result[cols, rows] = -values
            return result

        ax = skew(self.ax_upper)
        ay = skew(self.ay_upper)
        diffusion = self.diffusion_log_rate.new_zeros((49, 49))
        diffusion[torch.arange(1, 49), torch.arange(1, 49)] = -torch.exp(self.diffusion_log_rate)
        return ax, ay, diffusion

    def step(
        self,
        coefficients: torch.Tensor,
        parameters: torch.Tensor,
        *,
        rule: str,
    ) -> torch.Tensor:
        ax, ay, diffusion = self.matrices()
        active = coefficients[:, :49]
        vx = parameters[:, 0].view(-1, 1, 1)
        vy = parameters[:, 1].view(-1, 1, 1)
        nu = parameters[:, 2].view(-1, 1, 1)
        dt = parameters[:, 3].view(-1, 1, 1)
        if rule == "combined":
            generator = vx * ax + vy * ay + nu * diffusion
            evolved = torch.matrix_exp(dt * generator) @ active
        elif rule == "splitting":
            evolved = active
            factors = (
                (0.5 * nu, diffusion),
                (0.5 * vy, ay),
                (vx, ax),
                (0.5 * vy, ay),
                (0.5 * nu, diffusion),
            )
            for scale, matrix in factors:
                evolved = torch.matrix_exp(dt * scale * matrix) @ evolved
        else:
            raise ValueError(f"unknown E12 composition rule: {rule}")
        return torch.cat((evolved, coefficients[:, 49:].clone()), dim=1)

    def forward(self, coefficients: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
        return self.step(coefficients, parameters, rule="combined")


class FixedGenerator(nn.Module):
    def __init__(self, ax: torch.Tensor, ay: torch.Tensor, diffusion: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("ax", ax.clone())
        self.register_buffer("ay", ay.clone())
        self.register_buffer("diffusion", diffusion.clone())

    def matrices(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.ax, self.ay, self.diffusion

    def step(
        self,
        coefficients: torch.Tensor,
        parameters: torch.Tensor,
        *,
        rule: str,
    ) -> torch.Tensor:
        active = coefficients[:, :49]
        vx = parameters[:, 0].view(-1, 1, 1)
        vy = parameters[:, 1].view(-1, 1, 1)
        nu = parameters[:, 2].view(-1, 1, 1)
        dt = parameters[:, 3].view(-1, 1, 1)
        if rule == "combined":
            generator = vx * self.ax + vy * self.ay + nu * self.diffusion
            evolved = torch.matrix_exp(dt * generator) @ active
        elif rule == "splitting":
            evolved = active
            for scale, matrix in (
                (0.5 * nu, self.diffusion),
                (0.5 * vy, self.ay),
                (vx, self.ax),
                (0.5 * vy, self.ay),
                (0.5 * nu, self.diffusion),
            ):
                evolved = torch.matrix_exp(dt * scale * matrix) @ evolved
        else:
            raise ValueError(f"unknown E12 composition rule: {rule}")
        return torch.cat((evolved, coefficients[:, 49:].clone()), dim=1)


class RuleAdapter(nn.Module):
    def __init__(self, generator: StructuredGenerator | FixedGenerator, rule: str) -> None:
        super().__init__()
        self.generator = generator
        self.rule = rule

    def forward(self, coefficients: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
        return self.generator.step(coefficients, parameters, rule=self.rule)


def oracle_generators() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ax = torch.zeros(49, 49, dtype=torch.float64)
    ay = torch.zeros(49, 49, dtype=torch.float64)
    diffusion = torch.zeros(49, 49, dtype=torch.float64)
    one_dimensional_frequencies = (0, 1, 1, 2, 2, 3, 3)
    for x_index, x_frequency in enumerate(one_dimensional_frequencies):
        for y_index, y_frequency in enumerate(one_dimensional_frequencies):
            index = 7 * x_index + y_index
            diffusion[index, index] = -((2.0 * math.pi) ** 2) * (x_frequency**2 + y_frequency**2)
            if x_index in (1, 3, 5):
                cosine_index = 7 * (x_index + 1) + y_index
                omega = 2.0 * math.pi * x_frequency
                ax[index, cosine_index] = omega
                ax[cosine_index, index] = -omega
            if y_index in (1, 3, 5):
                cosine_index = 7 * x_index + y_index + 1
                omega = 2.0 * math.pi * y_frequency
                ay[index, cosine_index] = omega
                ay[cosine_index, index] = -omega
    return ax, ay, diffusion


def closure_parameter_cases() -> tuple[tuple[str, float, float, float], ...]:
    cases: list[tuple[str, float, float, float]] = []
    for speed in (0.20, 0.60, 1.00):
        for sign in (-1.0, 1.0):
            cases.append((f"x_{sign * speed:+.2f}", sign * speed, 0.0, 0.0))
            cases.append((f"y_{sign * speed:+.2f}", 0.0, sign * speed, 0.0))
    for diffusivity in (0.01, 0.045, 0.08):
        cases.append((f"diffusion_{diffusivity:.3f}", 0.0, 0.0, diffusivity))
    cases.extend(
        (
            ("composite_a", -1.0, 0.2, 0.01),
            ("composite_b", 0.6, -0.6, 0.045),
            ("composite_c", 1.0, 1.0, 0.08),
        )
    )
    return tuple(cases)


def structure_report(
    generator: StructuredGenerator | FixedGenerator,
) -> dict[str, float | int | bool]:
    ax, ay, diffusion = generator.matrices()
    parameters = sum(parameter.numel() for parameter in generator.parameters())
    return {
        "parameter_count": parameters,
        "ax_skew_residual": float((ax + ax.T).abs().max().item()),
        "ay_skew_residual": float((ay + ay.T).abs().max().item()),
        "ax_constant_residual": float(torch.cat((ax[0], ax[:, 0])).abs().max().item()),
        "ay_constant_residual": float(torch.cat((ay[0], ay[:, 0])).abs().max().item()),
        "diffusion_off_diagonal_residual": float(
            (diffusion - torch.diag(torch.diag(diffusion))).abs().max().item()
        ),
        "diffusion_constant_residual": float(diffusion[0, 0].abs().item()),
        "diffusion_maximum_eigenvalue": float(torch.linalg.eigvalsh(diffusion).max().item()),
        "all_finite": bool(
            torch.isfinite(ax).all()
            and torch.isfinite(ay).all()
            and torch.isfinite(diffusion).all()
        ),
        "constant_mode_structurally_fixed": True,
        "inactive_modes_copied": True,
    }


def _matrix_hashes(
    generator: StructuredGenerator | FixedGenerator,
) -> dict[str, str]:
    return {
        name: tensor_hash(tensor)
        for name, tensor in zip(("A_x", "A_y", "D"), generator.matrices(), strict=True)
    }


def generator_identification(
    generator: StructuredGenerator | FixedGenerator,
    oracle: FixedGenerator,
    *,
    space: PhysicalFunctionSpace,
    canonical_coords: torch.Tensor,
) -> dict[str, Any]:
    ax, ay, diffusion = generator.matrices()
    oracle_ax, oracle_ay, oracle_diffusion = oracle.matrices()

    def relative_frobenius(value: torch.Tensor, target: torch.Tensor) -> float:
        return float((torch.linalg.norm(value - target) / torch.linalg.norm(target)).item())

    def supported_error(value: torch.Tensor, target: torch.Tensor) -> float:
        support = target != 0
        return float(
            ((value[support] - target[support]).abs() / target[support].abs()).max().item()
        )

    def leakage(value: torch.Tensor, target: torch.Tensor) -> float:
        support = target != 0
        return float(
            (torch.linalg.norm(value.masked_fill(support, 0.0)) / torch.linalg.norm(target)).item()
        )

    def normalized_commutator(left: torch.Tensor, right: torch.Tensor) -> float:
        denominator = torch.linalg.norm(left) * torch.linalg.norm(right)
        return float(
            (torch.linalg.norm(left @ right - right @ left) / denominator.clamp_min(1e-24)).item()
        )

    active_basis = torch.eye(52, dtype=torch.float64)[:49].unsqueeze(-1)
    maximum_basis_action = 0.0
    model = RuleAdapter(generator, "combined")
    oracle_model = RuleAdapter(oracle, "combined")
    for _, vx, vy, nu in closure_parameter_cases():
        parameters = torch.tensor([[vx, vy, nu, 0.04]], dtype=torch.float64).expand(49, -1)
        prediction = model(active_basis, parameters)
        target = oracle_model(active_basis, parameters)
        prediction_decoded = space.decode(prediction, canonical_coords)
        target_decoded = space.decode(target, canonical_coords)
        per_basis_nrmse = torch.sqrt(
            (prediction_decoded - target_decoded).square().sum(dim=(1, 2))
            / target_decoded.square().sum(dim=(1, 2)).clamp_min(1e-24)
        )
        maximum_basis_action = max(
            maximum_basis_action,
            float(per_basis_nrmse.max().item()),
        )

    diffusion_rates = torch.diag(diffusion)[1:]
    oracle_rates = torch.diag(oracle_diffusion)[1:]
    return {
        "relative_frobenius": {
            "A_x": relative_frobenius(ax, oracle_ax),
            "A_y": relative_frobenius(ay, oracle_ay),
            "D": relative_frobenius(diffusion, oracle_diffusion),
        },
        "maximum_supported_entry_relative_error": {
            "A_x": supported_error(ax, oracle_ax),
            "A_y": supported_error(ay, oracle_ay),
        },
        "off_support_leakage": {
            "A_x": leakage(ax, oracle_ax),
            "A_y": leakage(ay, oracle_ay),
        },
        "maximum_diffusion_rate_relative_error": float(
            ((diffusion_rates - oracle_rates).abs() / oracle_rates.abs()).max().item()
        ),
        "normalized_commutators": {
            "A_x_A_y": normalized_commutator(ax, ay),
            "A_x_D": normalized_commutator(ax, diffusion),
            "A_y_D": normalized_commutator(ay, diffusion),
        },
        "maximum_basis_action_decoded_nrmse": maximum_basis_action,
    }


def oracle_preflight(
    cfg: E12Config,
    space: PhysicalFunctionSpace,
    oracle: FixedGenerator,
) -> dict[str, Any]:
    e11_closure = closure_preflight(cfg, space)
    combined = RuleAdapter(oracle, "combined")
    splitting = RuleAdapter(oracle, "splitting")
    coords, measure, basis = truth_grid(space, cfg.truth_resolution)
    active_basis = torch.eye(52, dtype=torch.float64)[:49].unsqueeze(-1)
    initial_values = basis.expand(49, -1, -1) @ active_basis
    maximum_one_step_error = 0.0
    maximum_eight_step_error = 0.0
    maximum_combined_splitting_mismatch = 0.0
    maximum_semigroup_error = 0.0
    all_finite = True
    records = []
    for name, vx, vy, nu in closure_parameter_cases():
        parameters = torch.tensor([[vx, vy, nu, 0.04]], dtype=torch.float64).expand(49, -1)
        one_values = evolve_periodic(
            initial_values,
            parameters,
            resolution=cfg.truth_resolution,
        )
        _, design = project_values(space, one_values, coords, measure)
        combined_one = combined(active_basis, parameters)
        splitting_one = splitting(active_basis, parameters)
        combined_eight = active_basis
        splitting_eight = active_basis
        repeated_values = initial_values
        for _ in range(cfg.rollout_steps):
            combined_eight = combined(combined_eight, parameters)
            splitting_eight = splitting(splitting_eight, parameters)
            repeated_values = evolve_periodic(
                repeated_values,
                parameters,
                resolution=cfg.truth_resolution,
            )
        direct_parameters = parameters.clone()
        direct_parameters[:, 3] *= cfg.rollout_steps
        combined_direct = combined(active_basis, direct_parameters)
        one_error = max(
            global_nrmse(space.decode(combined_one, coords), one_values),
            global_nrmse(space.decode(splitting_one, coords), one_values),
        )
        eight_error = max(
            global_nrmse(space.decode(combined_eight, coords), repeated_values),
            global_nrmse(space.decode(splitting_eight, coords), repeated_values),
        )
        rule_mismatch = global_nrmse(
            space.decode(splitting_eight, coords),
            space.decode(combined_eight, coords),
        )
        semigroup_error = global_nrmse(combined_eight, combined_direct)
        values = (one_error, eight_error, rule_mismatch, semigroup_error)
        all_finite &= all(math.isfinite(value) for value in values)
        maximum_one_step_error = max(maximum_one_step_error, one_error)
        maximum_eight_step_error = max(maximum_eight_step_error, eight_error)
        maximum_combined_splitting_mismatch = max(
            maximum_combined_splitting_mismatch, rule_mismatch
        )
        maximum_semigroup_error = max(maximum_semigroup_error, semigroup_error)
        records.append(
            {
                "case": name,
                "projection_rank": int(design["rank"]),
                "one_step_decoded_nrmse": one_error,
                "eight_step_decoded_nrmse": eight_error,
                "combined_splitting_decoded_nrmse": rule_mismatch,
                "combined_semigroup_coefficient_nrmse": semigroup_error,
            }
        )
    oracle_structure = structure_report(oracle)
    learned_structure = structure_report(StructuredGenerator())
    passed = all(
        (
            e11_closure["passed"],
            e11_closure["minimum_projection_rank"] == 52,
            e11_closure["all_projected_coefficients_finite"],
            all_finite,
            learned_structure["parameter_count"] == 2304,
            learned_structure["ax_skew_residual"] <= 1e-12,
            learned_structure["ay_skew_residual"] <= 1e-12,
            learned_structure["diffusion_off_diagonal_residual"] <= 1e-12,
            oracle_structure["ax_skew_residual"] <= 1e-12,
            oracle_structure["ay_skew_residual"] <= 1e-12,
            oracle_structure["diffusion_off_diagonal_residual"] <= 1e-12,
            oracle_structure["diffusion_maximum_eigenvalue"] <= 1e-12,
            maximum_one_step_error <= 1e-10,
            maximum_eight_step_error <= 1e-10,
            maximum_combined_splitting_mismatch <= 1e-10,
            maximum_semigroup_error <= 1e-10,
        )
    )
    return {
        "e11_projection_closure": e11_closure,
        "learned_initial_structure": learned_structure,
        "oracle_structure": oracle_structure,
        "parameter_cases": len(records),
        "active_basis_vectors": 49,
        "maximum_one_step_decoded_nrmse": maximum_one_step_error,
        "maximum_eight_step_decoded_nrmse": maximum_eight_step_error,
        "maximum_combined_splitting_decoded_nrmse": maximum_combined_splitting_mismatch,
        "maximum_combined_semigroup_coefficient_nrmse": maximum_semigroup_error,
        "all_finite": all_finite,
        "records": records,
        "passed": passed,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _committed_sha256(path: Path) -> str | None:
    relative = path.relative_to(REPO_ROOT).as_posix()
    completed = subprocess.run(
        ["git", "show", f"HEAD:{relative}"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        return None
    return hashlib.sha256(completed.stdout).hexdigest()


def provenance() -> dict[str, Any]:
    files = {"contract": CONTRACT_PATH, "runner": RUNNER_PATH}
    source_hashes = {name: _sha256(path) for name, path in files.items()}
    committed_hashes = {name: _committed_sha256(path) for name, path in files.items()}
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    worktree_clean = not subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    artifact_hash = _sha256(E11_ARTIFACT_PATH)
    artifact = json.loads(E11_ARTIFACT_PATH.read_text(encoding="utf-8"))
    observed_comparators = {
        f"{name}_rollout_decoded_nrmse": report["rollout_decoded_nrmse"]
        for name, report in artifact["composite_validation"].items()
        if name
        in {
            "elementary_pretrained_zero_shot",
            "pretrained_fewshot",
            "scratch_fewshot",
            "full_composite_control",
            "persistence",
        }
    }
    return {
        "source_sha256": source_hashes,
        "committed_source_sha256": committed_hashes,
        "source_files_match_git_head": source_hashes == committed_hashes,
        "git_head": git_head,
        "git_head_present": len(git_head) == 40,
        "worktree_clean": worktree_clean,
        "e11_artifact_sha256": artifact_hash,
        "e11_artifact_hash_matches": artifact_hash == E11_ARTIFACT_SHA256,
        "e11_comparators": observed_comparators,
        "e11_comparators_match": observed_comparators == E11_COMPARATORS,
    }


def _dataset_hash(dataset: TrajectorySet) -> dict[str, Any]:
    return {
        "shape": list(dataset.coefficients.shape),
        "initial_coefficients_sha256": tensor_hash(dataset.coefficients[:, 0]),
        "parameters_sha256": tensor_hash(dataset.parameters),
        "complete_trajectory_sha256": tensor_hash(dataset.coefficients),
        "all_finite": bool(
            torch.isfinite(dataset.coefficients).all() and torch.isfinite(dataset.parameters).all()
        ),
    }


def validate_schedules(schedules: dict[str, torch.Tensor]) -> dict[str, Any]:
    records = {}
    passed = True
    for name, expected in EXPECTED_SCHEDULES.items():
        tensor = schedules[name]
        little_endian_bytes = (
            tensor.cpu().contiguous().numpy().astype("<i8", copy=True).tobytes(order="C")
        )
        record = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "sha256": hashlib.sha256(little_endian_bytes).hexdigest(),
            "c_contiguous": tensor.is_contiguous(),
            "little_endian_int64_raw_bytes": True,
        }
        record["passed"] = (
            record["shape"] == expected["shape"]
            and record["dtype"] == "torch.int64"
            and record["sha256"] == expected["sha256"]
            and record["c_contiguous"]
        )
        passed &= bool(record["passed"])
        records[name] = record
    return {"records": records, "passed": passed}


def train_elementary(
    model: StructuredGenerator,
    datasets: dict[str, TrajectorySet],
    schedules: dict[str, torch.Tensor],
    cfg: E12Config,
) -> dict[str, Any]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.pretrain_learning_rate,
        weight_decay=cfg.weight_decay,
    )
    transitions = {name: dataset.transitions for name, dataset in datasets.items()}
    losses = []
    for update in range(cfg.pretrain_updates):
        batch_inputs = []
        batch_targets = []
        batch_parameters = []
        for name in REGIMES:
            inputs, targets, parameters = transitions[name]
            indices = schedules[name][update]
            batch_inputs.append(inputs[indices])
            batch_targets.append(targets[indices])
            batch_parameters.append(parameters[indices])
        inputs = torch.cat(batch_inputs)
        targets = torch.cat(batch_targets)
        parameters = torch.cat(batch_parameters)
        optimizer.zero_grad(set_to_none=True)
        loss = normalized_loss(model(inputs, parameters), targets)
        loss.backward()
        optimizer.step()
        if update in (0, cfg.pretrain_updates - 1):
            losses.append(float(loss.item()))
    return {
        "updates": cfg.pretrain_updates,
        "examples_per_regime": cfg.pretrain_updates * cfg.pretrain_batch_per_regime,
        "first_loss": losses[0],
        "final_loss": losses[-1],
        "training_rule": "combined",
    }


def train_single(
    model: StructuredGenerator,
    dataset: TrajectorySet,
    indices: torch.Tensor,
    *,
    learning_rate: float,
) -> dict[str, Any]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
    inputs, targets, parameters = dataset.transitions
    losses = []
    for update, batch_indices in enumerate(indices):
        optimizer.zero_grad(set_to_none=True)
        loss = normalized_loss(
            model(inputs[batch_indices], parameters[batch_indices]),
            targets[batch_indices],
        )
        loss.backward()
        optimizer.step()
        if update in (0, indices.shape[0] - 1):
            losses.append(float(loss.item()))
    return {
        "updates": int(indices.shape[0]),
        "examples": int(indices.numel()),
        "first_loss": losses[0],
        "final_loss": losses[-1],
        "training_rule": "combined",
    }


def validate_evaluation_coverage(evaluation: dict[str, Any]) -> dict[str, Any]:
    required_arms = set(EVALUATION_ARMS)
    required_regimes = set(ALL_REGIMES)
    failures = []
    for section in ("base", "temporal_extrapolation", "semigroup", "physics"):
        present = set(evaluation.get(section, {}))
        if present != required_arms:
            failures.append(
                {
                    "section": section,
                    "missing_arms": sorted(required_arms - present),
                    "unexpected_arms": sorted(present - required_arms),
                }
            )
    for arm in EVALUATION_ARMS:
        for section in ("base", "temporal_extrapolation", "semigroup"):
            present = set(evaluation[section].get(arm, {}))
            if present != required_regimes:
                failures.append(
                    {
                        "section": f"{section}.{arm}",
                        "missing_regimes": sorted(required_regimes - present),
                        "unexpected_regimes": sorted(present - required_regimes),
                    }
                )
        physics = set(evaluation["physics"].get(arm, {}).get("by_regime", {}))
        if physics != set(REGIMES):
            failures.append(
                {
                    "section": f"physics.{arm}.by_regime",
                    "missing_regimes": sorted(set(REGIMES) - physics),
                    "unexpected_regimes": sorted(physics - set(REGIMES)),
                }
            )
    if set(evaluation.get("cross_observation", {})) != set(RULES):
        failures.append({"section": "cross_observation", "expected": list(RULES)})
    expected_family_pairs = {
        f"{left}__vs__{right}"
        for left_index, left in enumerate(GEOMETRY_FAMILIES)
        for right in GEOMETRY_FAMILIES[left_index + 1 :]
    }
    for rule in RULES:
        cross = evaluation.get("cross_observation", {}).get(rule, {})
        if (
            cross.get("geometry_realizations") != 4
            or set(cross.get("pairs", {})) != expected_family_pairs
            or any(
                record.get("realization_pairs") != 16 for record in cross.get("pairs", {}).values()
            )
        ):
            failures.append(
                {
                    "section": f"cross_observation.{rule}",
                    "expected_family_pairs": sorted(expected_family_pairs),
                    "expected_realization_pairs_per_family_pair": 16,
                }
            )
    expected_gap = set(LEARNED_CHECKPOINTS)
    if set(evaluation.get("composition_gap", {})) != expected_gap:
        failures.append({"section": "composition_gap", "expected": list(expected_gap)})
    for checkpoint in LEARNED_CHECKPOINTS:
        if set(evaluation["composition_gap"].get(checkpoint, {})) != required_regimes:
            failures.append(
                {
                    "section": f"composition_gap.{checkpoint}",
                    "expected": list(ALL_REGIMES),
                }
            )
    if failures:
        raise RuntimeError(f"E12 evaluation coverage is incomplete: {failures}")
    return {
        "base_cells": len(EVALUATION_ARMS) * len(ALL_REGIMES),
        "temporal_extrapolation_cells": len(EVALUATION_ARMS) * len(ALL_REGIMES),
        "semigroup_cells": len(EVALUATION_ARMS) * len(ALL_REGIMES),
        "physics_cells": len(EVALUATION_ARMS) * len(REGIMES),
        "cross_observation_rules": len(RULES),
        "composition_gap_cells": len(LEARNED_CHECKPOINTS) * len(ALL_REGIMES),
        "passed": True,
    }


def _all_finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    return True


def identification_passes(report: dict[str, Any], cfg: E12Config) -> bool:
    return all(
        (
            max(report["relative_frobenius"].values())
            <= cfg.max_generator_relative_frobenius_error,
            max(report["maximum_supported_entry_relative_error"].values())
            <= cfg.max_supported_entry_relative_error,
            max(report["off_support_leakage"].values()) <= cfg.max_off_support_leakage,
            report["maximum_diffusion_rate_relative_error"]
            <= cfg.max_diffusion_rate_relative_error,
            max(report["normalized_commutators"].values()) <= cfg.max_normalized_commutator,
            report["maximum_basis_action_decoded_nrmse"] <= cfg.max_basis_action_decoded_nrmse,
        )
    )


def _rule_gates(result: dict[str, Any], cfg: E12Config, rule: str) -> dict[str, bool]:
    evaluation = result["evaluation"]
    suffix = f"_{rule}"
    candidate = evaluation["base"]["structured_pretrained_fewshot" + suffix]["composite"]
    scratch = evaluation["base"]["structured_scratch_fewshot" + suffix]["composite"]
    full = evaluation["base"]["structured_full_composite_control" + suffix]["composite"]
    zero_shot = evaluation["base"]["structured_elementary_pretrained" + suffix]["composite"]
    persistence = evaluation["base"]["persistence"]["composite"]
    elementary_before = evaluation["base"]["structured_elementary_pretrained" + suffix]
    elementary_after = evaluation["base"]["structured_pretrained_fewshot" + suffix]
    pre_macro = sum(elementary_before[regime]["rollout_decoded_nrmse"] for regime in REGIMES) / len(
        REGIMES
    )
    post_macro = sum(elementary_after[regime]["rollout_decoded_nrmse"] for regime in REGIMES) / len(
        REGIMES
    )
    temporal = evaluation["temporal_extrapolation"]["structured_pretrained_fewshot" + suffix]
    semigroup = evaluation["semigroup"]["structured_pretrained_fewshot" + suffix]
    cross = evaluation["cross_observation"][rule]
    physics = evaluation["physics"]["structured_pretrained_fewshot" + suffix]
    return {
        "one_step_accuracy": (
            candidate["one_step_coefficient_nrmse"] <= cfg.max_one_step_nrmse
            and candidate["one_step_decoded_nrmse"] <= cfg.max_one_step_nrmse
        ),
        "rollout_accuracy": (
            candidate["rollout_coefficient_nrmse"] <= cfg.max_rollout_nrmse
            and candidate["rollout_decoded_nrmse"] <= cfg.max_rollout_nrmse
        ),
        "high_frequency": candidate["final_high_frequency_spectral"]["nrmse"]
        <= cfg.max_high_frequency_nrmse,
        "fewshot_transfer": candidate["rollout_decoded_nrmse"]
        / max(scratch["rollout_decoded_nrmse"], 1e-24)
        <= cfg.max_pretrained_to_scratch_ratio,
        "full_data_parity": candidate["rollout_decoded_nrmse"]
        / max(full["rollout_decoded_nrmse"], 1e-24)
        <= cfg.max_pretrained_to_full_ratio,
        "zero_shot_composition": (
            zero_shot["rollout_decoded_nrmse"] <= cfg.max_zero_shot_rollout_nrmse
            and zero_shot["rollout_decoded_nrmse"]
            / max(persistence["rollout_decoded_nrmse"], 1e-24)
            <= cfg.max_zero_shot_to_persistence_ratio
        ),
        "elementary_retention": (
            post_macro <= cfg.max_elementary_retention_nrmse
            and post_macro / max(pre_macro, 1e-24) <= cfg.max_elementary_retention_ratio
        ),
        "temporal_extrapolation": max(
            max(report["rollout_coefficient_nrmse"], report["rollout_decoded_nrmse"])
            for report in temporal.values()
        )
        <= cfg.max_extrapolation_nrmse,
        "semigroup_measurement": max(
            max(report["coefficient_nrmse"], report["decoded_nrmse"])
            for report in semigroup.values()
        )
        <= cfg.max_semigroup_consistency_nrmse,
        "cross_observation_invariance": (
            cross["maximum_coefficient_mismatch"] <= cfg.max_cross_observation_mismatch
            and cross["maximum_decoded_mismatch"] <= cfg.max_cross_observation_mismatch
        ),
        "physics": (
            physics["advection_mean_mode_relative_error"] <= cfg.max_mean_mode_relative_error
            and physics["maximum_advection_l2_norm_drift"] <= cfg.max_advection_l2_drift
            and physics["diffusion_nonincreasing_energy_fraction"]
            >= cfg.minimum_diffusion_energy_monotonic_fraction
        ),
        "e11_improvement": (
            candidate["rollout_decoded_nrmse"]
            <= cfg.max_e11_dense_ratio * E11_COMPARATORS["pretrained_fewshot_rollout_decoded_nrmse"]
            and zero_shot["rollout_decoded_nrmse"]
            <= cfg.max_e11_dense_ratio
            * E11_COMPARATORS["elementary_pretrained_zero_shot_rollout_decoded_nrmse"]
        ),
        "combined_splitting_mismatch": max(
            record["decoded_nrmse"]
            for record in evaluation["composition_gap"]["structured_pretrained_fewshot"].values()
        )
        <= cfg.max_combined_splitting_decoded_mismatch,
    }


def _rule_metric_values(result: dict[str, Any], rule: str) -> dict[str, float]:
    evaluation = result["evaluation"]
    suffix = f"_{rule}"
    candidate = evaluation["base"]["structured_pretrained_fewshot" + suffix]["composite"]
    scratch = evaluation["base"]["structured_scratch_fewshot" + suffix]["composite"]
    full = evaluation["base"]["structured_full_composite_control" + suffix]["composite"]
    zero_shot = evaluation["base"]["structured_elementary_pretrained" + suffix]["composite"]
    persistence = evaluation["base"]["persistence"]["composite"]
    before = evaluation["base"]["structured_elementary_pretrained" + suffix]
    after = evaluation["base"]["structured_pretrained_fewshot" + suffix]
    pre_macro = sum(before[regime]["rollout_decoded_nrmse"] for regime in REGIMES) / len(REGIMES)
    post_macro = sum(after[regime]["rollout_decoded_nrmse"] for regime in REGIMES) / len(REGIMES)
    temporal = evaluation["temporal_extrapolation"]["structured_pretrained_fewshot" + suffix]
    semigroup = evaluation["semigroup"]["structured_pretrained_fewshot" + suffix]
    cross = evaluation["cross_observation"][rule]
    physics = evaluation["physics"]["structured_pretrained_fewshot" + suffix]
    return {
        "one_step_coefficient_nrmse": candidate["one_step_coefficient_nrmse"],
        "one_step_decoded_nrmse": candidate["one_step_decoded_nrmse"],
        "rollout_coefficient_nrmse": candidate["rollout_coefficient_nrmse"],
        "rollout_decoded_nrmse": candidate["rollout_decoded_nrmse"],
        "final_high_frequency_nrmse": candidate["final_high_frequency_spectral"]["nrmse"],
        "pretrained_to_scratch_ratio": candidate["rollout_decoded_nrmse"]
        / max(scratch["rollout_decoded_nrmse"], 1e-24),
        "pretrained_to_full_ratio": candidate["rollout_decoded_nrmse"]
        / max(full["rollout_decoded_nrmse"], 1e-24),
        "zero_shot_rollout_decoded_nrmse": zero_shot["rollout_decoded_nrmse"],
        "zero_shot_to_persistence_ratio": zero_shot["rollout_decoded_nrmse"]
        / max(persistence["rollout_decoded_nrmse"], 1e-24),
        "post_finetune_elementary_macro_decoded_nrmse": post_macro,
        "post_to_pre_retention_ratio": post_macro / max(pre_macro, 1e-24),
        "worst_temporal_nrmse": max(
            max(report["rollout_coefficient_nrmse"], report["rollout_decoded_nrmse"])
            for report in temporal.values()
        ),
        "worst_semigroup_nrmse": max(
            max(report["coefficient_nrmse"], report["decoded_nrmse"])
            for report in semigroup.values()
        ),
        "cross_observation_coefficient_mismatch": cross["maximum_coefficient_mismatch"],
        "cross_observation_decoded_mismatch": cross["maximum_decoded_mismatch"],
        "advection_mean_mode_relative_error": physics["advection_mean_mode_relative_error"],
        "maximum_advection_l2_norm_drift": physics["maximum_advection_l2_norm_drift"],
        "diffusion_nonincreasing_energy_fraction": physics[
            "diffusion_nonincreasing_energy_fraction"
        ],
        "candidate_to_e11_dense_ratio": candidate["rollout_decoded_nrmse"]
        / E11_COMPARATORS["pretrained_fewshot_rollout_decoded_nrmse"],
        "zero_shot_to_e11_dense_ratio": zero_shot["rollout_decoded_nrmse"]
        / E11_COMPARATORS["elementary_pretrained_zero_shot_rollout_decoded_nrmse"],
        "worst_combined_splitting_decoded_mismatch": max(
            record["decoded_nrmse"]
            for record in evaluation["composition_gap"]["structured_pretrained_fewshot"].values()
        ),
    }


def decision(result: dict[str, Any], cfg: E12Config) -> dict[str, Any]:
    identifications = result["generator_identification"]
    structures = result["checkpoint_structure"]
    shared = {
        "closure": result["preflight"]["e11_projection_closure"]["passed"],
        "oracle_correctness": result["preflight"]["passed"],
        "generator_identification": (
            identification_passes(identifications["structured_elementary_pretrained"], cfg)
            and identification_passes(identifications["structured_pretrained_fewshot"], cfg)
            and max(
                max(identifications["oracle"]["relative_frobenius"].values()),
                max(identifications["oracle"]["maximum_supported_entry_relative_error"].values()),
                max(identifications["oracle"]["off_support_leakage"].values()),
                identifications["oracle"]["maximum_diffusion_rate_relative_error"],
                max(identifications["oracle"]["normalized_commutators"].values()),
                identifications["oracle"]["maximum_basis_action_decoded_nrmse"],
            )
            <= 1e-12
        ),
        "parameter_structure_and_count": all(
            report["parameter_count"] == 2304
            and report["ax_skew_residual"] <= 1e-12
            and report["ay_skew_residual"] <= 1e-12
            and report["ax_constant_residual"] <= 1e-12
            and report["ay_constant_residual"] <= 1e-12
            and report["diffusion_off_diagonal_residual"] <= 1e-12
            and report["diffusion_constant_residual"] <= 1e-12
            and report["diffusion_maximum_eigenvalue"] <= 1e-12
            and report["all_finite"]
            for report in structures.values()
        ),
        "source_data_schedule_provenance": all(
            (
                result["provenance"]["source_files_match_git_head"],
                result["provenance"]["git_head_present"],
                result["provenance"]["worktree_clean"],
                result["provenance"]["e11_artifact_hash_matches"],
                result["provenance"]["e11_comparators_match"],
                result["schedules"]["passed"],
                all(record["all_finite"] for record in result["dataset_hashes"].values()),
                _all_finite(result),
            )
        ),
        "cartesian_coverage": result["evaluation"]["coverage"]["passed"],
        "complete_result_replication": result["reproducibility"]["byte_identical_complete_runs"],
        "boundary": all(
            (
                result["boundary"]["heldout_reads"] == 0,
                result["boundary"]["provider_calls"] == 0,
                result["boundary"]["routing_paths"] == 0,
                not result["boundary"]["representation_label_inputs"],
                not result["boundary"]["task_label_inputs"],
                not result["boundary"]["original_observations_after_projection"],
            )
        ),
    }
    combined = _rule_gates(result, cfg, "combined")
    splitting = _rule_gates(result, cfg, "splitting")
    combined_metrics = _rule_metric_values(result, "combined")
    splitting_metrics = _rule_metric_values(result, "splitting")
    if all(shared.values()) and all(combined.values()):
        classification = "structured_additive_generator_qualified"
        next_move = "preregister the first nonlinear coefficient-dynamics expansion"
    else:
        without_transfer = {
            name: passed
            for name, passed in combined.items()
            if name not in {"fewshot_transfer", "full_data_parity"}
        }
        if all(shared.values()) and all(without_transfer.values()):
            classification = "structured_generator_capable_without_transfer"
            next_move = "audit elementary-to-composite transfer without changing the representation"
        else:
            classification = "structured_generator_not_qualified"
            next_move = "audit generator identifiability or optimization before broader physics"
    improvements = {
        name: {
            "combined_passed": combined[name],
            "splitting_passed": splitting[name],
            "splitting_improves_failed_combined": bool(not combined[name] and splitting[name]),
        }
        for name in combined
    }
    return {
        "classification": classification,
        "shared_gates": shared,
        "combined_gates": combined,
        "splitting_diagnostic_gates": splitting,
        "splitting_diagnostic_improves_failed_combined": any(
            record["splitting_improves_failed_combined"] for record in improvements.values()
        ),
        "splitting_gate_comparison": improvements,
        "combined_metrics": combined_metrics,
        "splitting_metrics": splitting_metrics,
        "splitting_minus_combined_metric_deltas": {
            name: splitting_metrics[name] - combined_metrics[name] for name in combined_metrics
        },
        "next_move": next_move,
    }


def _evaluate(
    cfg: E12Config,
    *,
    space: PhysicalFunctionSpace,
    models: dict[str, StructuredGenerator],
    oracle: FixedGenerator,
    base_datasets: dict[str, TrajectorySet],
    temporal_datasets: dict[str, TrajectorySet],
) -> dict[str, Any]:
    canonical_coords, _ = canonical_grid(space, cfg.canonical_query_resolution)
    arms: dict[str, tuple[nn.Module | None, bool]] = {}
    for checkpoint, model in models.items():
        for rule in RULES:
            arms[f"{checkpoint}_{rule}"] = (RuleAdapter(model, rule), False)
    arms.update(
        {
            "persistence": (None, False),
            "exact_projected_truth": (None, True),
            "oracle_combined": (RuleAdapter(oracle, "combined"), False),
            "oracle_splitting": (RuleAdapter(oracle, "splitting"), False),
        }
    )
    evaluation: dict[str, Any] = {
        "base": {},
        "temporal_extrapolation": {},
        "semigroup": {},
        "physics": {},
        "cross_observation": {},
        "composition_gap": {},
    }
    base_predictions: dict[str, dict[str, torch.Tensor]] = {}
    for arm, (model, exact_truth) in arms.items():
        evaluation["base"][arm] = {}
        evaluation["temporal_extrapolation"][arm] = {}
        evaluation["semigroup"][arm] = {}
        base_predictions[arm] = {}
        for regime, dataset in base_datasets.items():
            report, prediction = rollout(
                model,  # type: ignore[arg-type]
                dataset,
                space=space,
                canonical_coords=canonical_coords,
                cfg=cfg,
                exact_truth=exact_truth,
            )
            evaluation["base"][arm][regime] = report
            report["all_finite"] = _all_finite(report)
            base_predictions[arm][regime] = prediction
            if exact_truth:
                semigroup_report = exact_semigroup_consistency(
                    dataset.coefficients[:, 0],
                    dataset.parameters,
                    space=space,
                    canonical_coords=canonical_coords,
                    cfg=cfg,
                )
            else:
                semigroup_report = semigroup_consistency(
                    model,  # type: ignore[arg-type]
                    dataset.coefficients[:, 0],
                    dataset.parameters,
                    space=space,
                    canonical_coords=canonical_coords,
                )
            semigroup_report["all_finite"] = _all_finite(semigroup_report)
            evaluation["semigroup"][arm][regime] = semigroup_report
        for regime, dataset in temporal_datasets.items():
            report, _ = rollout(
                model,  # type: ignore[arg-type]
                dataset,
                space=space,
                canonical_coords=canonical_coords,
                cfg=cfg,
                exact_truth=exact_truth,
            )
            report["all_finite"] = _all_finite(report)
            evaluation["temporal_extrapolation"][arm][regime] = report
        evaluation["physics"][arm] = physics_report(
            base_predictions[arm]["x_advection"],
            base_predictions[arm]["y_advection"],
            base_predictions[arm]["diffusion"],
            base_datasets["x_advection"],
            base_datasets["y_advection"],
        )
    validation = base_datasets["composite"]
    for rule in RULES:
        evaluation["cross_observation"][rule] = cross_observation_report(
            RuleAdapter(models["structured_pretrained_fewshot"], rule),  # type: ignore[arg-type]
            validation,
            space=space,
            canonical_coords=canonical_coords,
            cfg=cfg,
        )
    for checkpoint in LEARNED_CHECKPOINTS:
        evaluation["composition_gap"][checkpoint] = {}
        combined_arm = f"{checkpoint}_combined"
        splitting_arm = f"{checkpoint}_splitting"
        for regime in ALL_REGIMES:
            combined_prediction = base_predictions[combined_arm][regime]
            splitting_prediction = base_predictions[splitting_arm][regime]
            target_decoded = space.decode(
                base_datasets[regime].coefficients[:, -1], canonical_coords
            )
            evaluation["composition_gap"][checkpoint][regime] = {
                "coefficient_nrmse": _relative_mismatch(
                    splitting_prediction[:, -1],
                    combined_prediction[:, -1],
                ),
                "decoded_nrmse": _decoded_mismatch(
                    space.decode(splitting_prediction[:, -1], canonical_coords),
                    space.decode(combined_prediction[:, -1], canonical_coords),
                    target_decoded,
                ),
            }
    evaluation["coverage"] = validate_evaluation_coverage(evaluation)
    return evaluation


def _run_once(cfg: E12Config, *, run_dir: Path) -> dict[str, Any]:
    if asdict(cfg) != asdict(frozen_e12_config()):
        raise ValueError("E12 requires the exact frozen configuration")
    provenance_report = provenance()
    if not all(
        (
            provenance_report["source_files_match_git_head"],
            provenance_report["git_head_present"],
            provenance_report["worktree_clean"],
            provenance_report["e11_artifact_hash_matches"],
            provenance_report["e11_comparators_match"],
        )
    ):
        raise RuntimeError(
            "E12 provenance must match a clean committed Git HEAD and locked E11 artifact "
            "before sampled state access"
        )

    torch.manual_seed(cfg.model_seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    run_dir.mkdir(parents=True, exist_ok=True)
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
    preflight = oracle_preflight(cfg, space, oracle)
    if not preflight["passed"]:
        result = {
            "schema_version": 1,
            "experiment": "canonical_latent_e12_structured_generator",
            "config": asdict(cfg),
            "provenance": provenance_report,
            "preflight": preflight,
            "state_reads": {"training": 0, "validation": 0, "heldout": 0},
            "optimizer_updates": 0,
        }
        result_path = run_dir / "result.json"
        result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        result["result_path"] = str(result_path)
        return result

    pretrain_datasets = {
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
    fewshot = build_trajectories(
        "composite_fewshot",
        count=cfg.fewshot_trajectories,
        state_seed=cfg.fewshot_state_seed,
        parameter_seed=cfg.fewshot_parameter_seed,
        regime="composite",
        cfg=cfg,
        space=space,
    )
    full_control = build_trajectories(
        "composite_full",
        count=cfg.full_control_trajectories,
        state_seed=cfg.full_state_seed,
        parameter_seed=cfg.full_parameter_seed,
        regime="composite",
        cfg=cfg,
        space=space,
    )
    validation = build_trajectories(
        "composite_validation",
        count=cfg.validation_trajectories,
        state_seed=cfg.validation_state_seed,
        parameter_seed=cfg.validation_parameter_seed,
        regime="composite",
        cfg=cfg,
        space=space,
    )
    extrapolation = build_trajectories(
        "composite_temporal_extrapolation",
        count=cfg.validation_trajectories,
        state_seed=cfg.validation_state_seed,
        parameter_seed=cfg.validation_parameter_seed,
        regime="composite",
        cfg=cfg,
        space=space,
        dt_override=cfg.extrapolation_dt,
    )
    parameter_seeds = {
        "x_advection": cfg.x_validation_parameter_seed,
        "y_advection": cfg.y_validation_parameter_seed,
        "diffusion": cfg.diffusion_validation_parameter_seed,
    }
    elementary_validation = {
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
    elementary_extrapolation = {
        regime: build_trajectories(
            f"{regime}_temporal_extrapolation",
            count=cfg.validation_trajectories,
            state_seed=cfg.validation_state_seed,
            parameter_seed=seed,
            regime=regime,
            cfg=cfg,
            space=space,
            dt_override=cfg.extrapolation_dt,
        )
        for regime, seed in parameter_seeds.items()
    }
    datasets = {
        **{f"pretrain_{name}": dataset for name, dataset in pretrain_datasets.items()},
        "composite_fewshot": fewshot,
        "composite_full_control": full_control,
        "composite_validation": validation,
        "composite_temporal_extrapolation": extrapolation,
        **{f"{name}_validation": dataset for name, dataset in elementary_validation.items()},
        **{
            f"{name}_temporal_extrapolation": dataset
            for name, dataset in elementary_extrapolation.items()
        },
    }
    dataset_hashes = {name: _dataset_hash(dataset) for name, dataset in datasets.items()}

    pretrain_schedules = {
        regime: schedule(
            cfg.pretrain_updates,
            cfg.pretrain_batch_per_regime,
            pretrain_datasets[regime].transitions[0].shape[0],
            seed=cfg.schedule_seed + index,
        )
        for index, regime in enumerate(REGIMES)
    }
    fine_schedule = schedule(
        cfg.fine_tune_updates,
        cfg.fine_tune_batch_size,
        fewshot.transitions[0].shape[0],
        seed=cfg.schedule_seed + 10,
    )
    full_schedule = schedule(
        cfg.full_control_updates,
        cfg.full_control_batch_size,
        full_control.transitions[0].shape[0],
        seed=cfg.schedule_seed + 20,
    )
    schedule_report = validate_schedules(
        {
            **pretrain_schedules,
            "fine_tune": fine_schedule,
            "full_control": full_schedule,
        }
    )
    if not schedule_report["passed"]:
        raise RuntimeError("E12 schedule hashes do not match the frozen E11 schedules")

    initial_model = StructuredGenerator()
    initial_state = copy.deepcopy(initial_model.state_dict())
    elementary = StructuredGenerator()
    elementary.load_state_dict(initial_state)
    elementary_training = train_elementary(elementary, pretrain_datasets, pretrain_schedules, cfg)
    pretrained_fewshot = copy.deepcopy(elementary)
    pretrained_training = train_single(
        pretrained_fewshot,
        fewshot,
        fine_schedule,
        learning_rate=cfg.fine_tune_learning_rate,
    )
    scratch = StructuredGenerator()
    scratch.load_state_dict(initial_state)
    scratch_training = train_single(
        scratch,
        fewshot,
        fine_schedule,
        learning_rate=cfg.fine_tune_learning_rate,
    )
    full = StructuredGenerator()
    full.load_state_dict(initial_state)
    full_training = train_single(
        full,
        full_control,
        full_schedule,
        learning_rate=cfg.pretrain_learning_rate,
    )
    models = {
        "structured_elementary_pretrained": elementary,
        "structured_pretrained_fewshot": pretrained_fewshot,
        "structured_scratch_fewshot": scratch,
        "structured_full_composite_control": full,
    }
    base_datasets = {"composite": validation, **elementary_validation}
    temporal_datasets = {"composite": extrapolation, **elementary_extrapolation}
    evaluation = _evaluate(
        cfg,
        space=space,
        models=models,
        oracle=oracle,
        base_datasets=base_datasets,
        temporal_datasets=temporal_datasets,
    )
    canonical_coords, _ = canonical_grid(space, cfg.canonical_query_resolution)
    identification = {
        name: generator_identification(
            model, oracle, space=space, canonical_coords=canonical_coords
        )
        for name, model in models.items()
    }
    identification["oracle"] = generator_identification(
        oracle, oracle, space=space, canonical_coords=canonical_coords
    )
    checkpoint_structure = {name: structure_report(model) for name, model in models.items()}
    result = {
        "schema_version": 1,
        "experiment": "canonical_latent_e12_structured_generator",
        "config": asdict(cfg),
        "config_sha256": hashlib.sha256(
            json.dumps(asdict(cfg), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "provenance": provenance_report,
        "preflight": preflight,
        "architecture": {
            "kind": "additive_continuous_time_coefficient_generator",
            "basis_dimension": 52,
            "active_periodic_modes": 49,
            "parameter_count": 2304,
            "training_rule": "combined",
            "splitting_role": "checkpoint_identical_evaluation_only",
            "representation_label_inputs": False,
            "task_label_inputs": False,
            "routing_paths": 0,
        },
        "state_splits": {
            "elementary_pretrain_trajectories": 3 * cfg.pretrain_trajectories_per_regime,
            "composite_fewshot_trajectories": cfg.fewshot_trajectories,
            "composite_full_control_trajectories": cfg.full_control_trajectories,
            "validation_trajectories": cfg.validation_trajectories,
            "heldout_reads": 0,
        },
        "dataset_hashes": dataset_hashes,
        "schedules": schedule_report,
        "training": {
            "structured_elementary_pretrained": elementary_training,
            "structured_pretrained_fewshot": pretrained_training,
            "structured_scratch_fewshot": scratch_training,
            "structured_full_composite_control": full_training,
        },
        "checkpoints": {
            "initial_sha256": model_hash(initial_model),
            **{name: model_hash(model) for name, model in models.items()},
        },
        "generator_sha256": {
            **{name: _matrix_hashes(model) for name, model in models.items()},
            "oracle": _matrix_hashes(oracle),
        },
        "checkpoint_structure": checkpoint_structure,
        "generator_identification": identification,
        "evaluation": evaluation,
        "reproducibility": {"deterministic_algorithms": True},
        "boundary": {
            "synthetic_periodic_scalar_only": True,
            "linear_advection_diffusion_only": True,
            "nonlinear_dynamics_qualified": False,
            "particle_dynamics_qualified": False,
            "heldout_reads": 0,
            "provider_calls": 0,
            "routing_paths": 0,
            "representation_label_inputs": False,
            "task_label_inputs": False,
            "original_observations_after_projection": False,
            "public_or_claim_grade": False,
        },
    }
    if not _all_finite(result):
        raise RuntimeError("E12 result contains a nonfinite numeric value")
    result_path = run_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    result["result_path"] = str(result_path)
    return result


def run_e12(cfg: E12Config, *, run_dir: Path) -> dict[str, Any]:
    if asdict(cfg) != asdict(frozen_e12_config()):
        raise ValueError("E12 requires the exact frozen configuration")
    first = _run_once(cfg, run_dir=run_dir / "replicate_a")
    second = _run_once(cfg, run_dir=run_dir / "replicate_b")
    first_bytes = Path(first["result_path"]).read_bytes()
    second_bytes = Path(second["result_path"]).read_bytes()
    raw_identical = first_bytes == second_bytes
    first_hash = hashlib.sha256(first_bytes).hexdigest()
    second_hash = hashlib.sha256(second_bytes).hexdigest()
    complete_results = []
    for raw_bytes in (first_bytes, second_bytes):
        complete = json.loads(raw_bytes)
        complete["reproducibility"] = {
            "deterministic_algorithms": True,
            "replicate_a_raw_sha256": first_hash,
            "replicate_b_raw_sha256": second_hash,
            "raw_runs_byte_identical": raw_identical,
            "byte_identical_complete_runs": raw_identical,
        }
        if complete["preflight"]["passed"]:
            complete["causal_decision"] = decision(complete, cfg)
        else:
            complete["causal_decision"] = {
                "classification": "structured_generator_not_qualified",
                "shared_gates": {"oracle_correctness": False},
                "next_move": "repair the structured-generator oracle implementation",
            }
        complete_results.append((json.dumps(complete, indent=2, sort_keys=True) + "\n").encode())
    first_complete = run_dir / "replicate_a" / "complete_result.json"
    second_complete = run_dir / "replicate_b" / "complete_result.json"
    first_complete.write_bytes(complete_results[0])
    second_complete.write_bytes(complete_results[1])
    complete_identical = first_complete.read_bytes() == second_complete.read_bytes()
    if complete_identical != raw_identical:
        raise RuntimeError("E12 raw and complete replication identities disagree")
    result_path = run_dir / "result.json"
    result_path.write_bytes(complete_results[0])
    result_hash = hashlib.sha256(complete_results[0]).hexdigest()
    manifest = {
        "schema_version": 1,
        "experiment": "canonical_latent_e12_structured_generator",
        "result_sha256": result_hash,
        "replicate_a_complete_sha256": hashlib.sha256(first_complete.read_bytes()).hexdigest(),
        "replicate_b_complete_sha256": hashlib.sha256(second_complete.read_bytes()).hexdigest(),
        "byte_identical_complete_runs": complete_identical,
        "git_head": json.loads(complete_results[0])["provenance"]["git_head"],
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    result = json.loads(complete_results[0])
    result["result_path"] = str(result_path)
    result["manifest_path"] = str(manifest_path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen E12 structured coefficient-generator gate"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--print-json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_e12(frozen_e12_config(), run_dir=args.run_dir)
    summary = {
        "causal_decision": result["causal_decision"],
        "result_path": result["result_path"],
        "manifest_path": result["manifest_path"],
    }
    if args.print_json:
        print(json.dumps(summary, sort_keys=True))
    else:
        print(
            f"classification={summary['causal_decision']['classification']} "
            f"result={summary['result_path']}"
        )


if __name__ == "__main__":
    main()
