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
)
from scripts.run_canonical_latent_e11_coefficient_operator_transfer import (
    REGIMES,
    TrajectorySet,
    build_trajectories,
    canonical_grid,
    model_hash,
    normalized_loss,
    rollout,
    schedule,
    tensor_hash,
)
from scripts.run_canonical_latent_e12_structured_generator import (
    E12Config,
    FixedGenerator,
    RuleAdapter,
    StructuredGenerator,
    _all_finite,
    _dataset_hash,
    _matrix_hashes,
    closure_parameter_cases,
    frozen_e12_config,
    generator_identification,
    identification_passes,
    oracle_generators,
    oracle_preflight,
    structure_report,
    train_elementary,
    validate_schedules,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = Path(__file__).resolve()
CONTRACT_PATH = (
    REPO_ROOT / "docs/research/2026-07-26-canonical-latent-e13-identifiability-audit-contract.md"
)
E12_LOCK_PATH = REPO_ROOT / "docs/research/artifacts/canonical_latent_e12_replay_lock.json"
E12_ARTIFACT_PATH = (
    REPO_ROOT / "docs/research/artifacts/canonical_latent_e12_structured_generator_result.json"
)
E12_RUNNER_PATH = REPO_ROOT / "scripts/run_canonical_latent_e12_structured_generator.py"
E11_RUNNER_PATH = REPO_ROOT / "scripts/run_canonical_latent_e11_coefficient_operator_transfer.py"
E7_RUNNER_PATH = REPO_ROOT / "scripts/run_canonical_latent_e7_function_space.py"
LATENT_EVAL_PATH = REPO_ROOT / "src/ups/eval/latent_qualification.py"

EXPECTED_E12_LOCK_SHA256 = "0bf8f032daf95415bee401b5a90f4e6ca1598f12748151dd7310b2a6d02f8dfd"
EXPECTED_E12_ARTIFACT_SHA256 = "d4760ec3d69b4397cc14ffc3bb08edd3f073edcaf1f6d4dd70db070a96cab3b2"
EXPECTED_E12_CONFIG_SHA256 = "cd428d490ad9d5505f88ead66b41fdb25e25830f45d0eb21f451c5dbea261934"
EXPECTED_E12_INITIAL_SHA256 = "64c87294711c68c9bc4a9f56cb3f8a8ca23b1e1eed84493bfc18e18f3a2c9218"
EXPECTED_E12_ELEMENTARY_SHA256 = "e9c17bc1871f5b2008d3899da9f59a44cf1209448f50ba576ca63f6281602e7b"
EXPECTED_E12_GENERATOR_SHA256 = {
    "A_x": "17f70896f48f5651854746c5be9edc7dd42d2fd331c1fbc7d2c6e2edd1a46e53",
    "A_y": "726d18845ad3dc3e462fc16578adf08929594e57dd02818c1e13711eee31fda2",
    "D": "1edb9c7429ebec8a583d000fedf2e754668727fed857b14d3d06f5b1649fa370",
}
EXPECTED_E12_FIRST_LOSS = 0.08166621133143402
EXPECTED_E12_FINAL_LOSS = 0.00032425428259914163
EXPECTED_E12_BASIS_ACTION = 0.09992145782297766
EXPECTED_E12_ZERO_SHOT_ROLLOUT = 0.00536402045663299
E12_PERSISTENCE_ROLLOUT = 1.5584481380508215

SEMANTIC_COMPONENTS = (
    "constant",
    "sin1",
    "cos1",
    "sin2",
    "cos2",
    "sin3",
    "cos3",
)
CONTROLS = (
    "e12_adamw_replay",
    "full_skew_lbfgs_neutral",
    "full_skew_lbfgs_polish",
    "support_sparse_lbfgs",
    "mode_tied_lbfgs",
    "oracle",
)
LEARNED_CONTROLS = CONTROLS[:-1]
REGIME_TO_COMPONENT = {
    "x_advection": "A_x",
    "y_advection": "A_y",
    "diffusion": "D",
}
EXPECTED_MASK_HASHES = {
    "A_x": "fa83470fcf9853aa4d607ec8935d8b73cffec3735242ef87370661aa67af2806",
    "A_y": "20562976940d1bb28e528ebcbf259fcfb53156b50e606ebffe8a6db322cb3fee",
    "D": "d490fb408871d40fafbf53d94be298ba475a27ec033ef079ff98dccef9ed29b1",
}


@dataclass(frozen=True)
class E13Config(E12Config):
    lbfgs_learning_rate: float = 1.0
    lbfgs_max_iter: int = 250
    lbfgs_max_eval: int = 300
    lbfgs_history_size: int = 100
    lbfgs_tolerance_grad: float = 1e-12
    lbfgs_tolerance_change: float = 1e-15

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            self.lbfgs_learning_rate != 1.0
            or self.lbfgs_max_iter != 250
            or self.lbfgs_max_eval != 300
            or self.lbfgs_history_size != 100
            or self.lbfgs_tolerance_grad != 1e-12
            or self.lbfgs_tolerance_change != 1e-15
        ):
            raise ValueError("E13 deterministic recovery is frozen")


def frozen_e13_config() -> E13Config:
    return E13Config()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _committed_sha256(path: Path) -> str | None:
    relative = path.relative_to(REPO_ROOT)
    completed = subprocess.run(
        ["git", "show", f"HEAD:{relative.as_posix()}"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        return None
    return hashlib.sha256(completed.stdout).hexdigest()


def _git_text(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def provenance() -> dict[str, Any]:
    sources = {
        "contract": CONTRACT_PATH,
        "runner": RUNNER_PATH,
        "e12_lock": E12_LOCK_PATH,
        "e12_artifact": E12_ARTIFACT_PATH,
        "e12_runner": E12_RUNNER_PATH,
        "e11_runner": E11_RUNNER_PATH,
        "e7_runner": E7_RUNNER_PATH,
        "latent_evaluation": LATENT_EVAL_PATH,
    }
    source_records = {}
    source_match = True
    for name, path in sources.items():
        working = _sha256(path)
        committed = _committed_sha256(path)
        matches = committed is not None and working == committed
        source_records[name] = {
            "path": str(path.relative_to(REPO_ROOT)),
            "working_sha256": working,
            "committed_sha256": committed,
            "matches_git_head": matches,
        }
        source_match &= matches
    e12_cfg_sha = hashlib.sha256(
        json.dumps(asdict(frozen_e12_config()), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    report = {
        "git_head": _git_text("rev-parse", "HEAD"),
        "git_head_present": bool(_git_text("rev-parse", "HEAD")),
        "worktree_clean": _git_text("status", "--porcelain") == "",
        "source_files": source_records,
        "source_files_match_git_head": source_match,
        "e12_lock_hash_matches": _sha256(E12_LOCK_PATH) == EXPECTED_E12_LOCK_SHA256,
        "e12_artifact_hash_matches": _sha256(E12_ARTIFACT_PATH) == EXPECTED_E12_ARTIFACT_SHA256,
        "e12_config_sha256": e12_cfg_sha,
        "e12_config_hash_matches": e12_cfg_sha == EXPECTED_E12_CONFIG_SHA256,
    }
    report["passed"] = all(
        (
            report["git_head_present"],
            report["worktree_clean"],
            report["source_files_match_git_head"],
            report["e12_lock_hash_matches"],
            report["e12_artifact_hash_matches"],
            report["e12_config_hash_matches"],
        )
    )
    return report


def semantic_masks() -> dict[str, torch.Tensor]:
    ax = torch.zeros(49, 49, dtype=torch.bool)
    ay = torch.zeros(49, 49, dtype=torch.bool)
    diffusion = torch.zeros(49, 49, dtype=torch.bool)
    for x_index in (1, 3, 5):
        for y_index in range(7):
            left = 7 * x_index + y_index
            right = 7 * (x_index + 1) + y_index
            ax[left, right] = True
            ax[right, left] = True
    for y_index in (1, 3, 5):
        for x_index in range(7):
            left = 7 * x_index + y_index
            right = left + 1
            ay[left, right] = True
            ay[right, left] = True
    diffusion[torch.arange(1, 49), torch.arange(1, 49)] = True
    return {"A_x": ax, "A_y": ay, "D": diffusion}


def mask_report() -> dict[str, Any]:
    masks = semantic_masks()
    hashes = {name: tensor_hash(mask.to(torch.uint8)) for name, mask in masks.items()}
    return {
        "hashes": hashes,
        "expected_hashes": EXPECTED_MASK_HASHES,
        "hashes_match": hashes == EXPECTED_MASK_HASHES,
        "nonzero_entries": {name: int(mask.sum().item()) for name, mask in masks.items()},
    }


class SupportSparseGenerator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.x_rates = nn.Parameter(torch.zeros(21, dtype=torch.float64))
        self.y_rates = nn.Parameter(torch.zeros(21, dtype=torch.float64))
        self.diffusion_log_rates = nn.Parameter(torch.zeros(48, dtype=torch.float64))

    def matrices(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ax = self.x_rates.new_zeros((49, 49))
        ay = self.y_rates.new_zeros((49, 49))
        diffusion = self.diffusion_log_rates.new_zeros((49, 49))
        cursor = 0
        for x_index in (1, 3, 5):
            for y_index in range(7):
                left = 7 * x_index + y_index
                right = 7 * (x_index + 1) + y_index
                ax[left, right] = self.x_rates[cursor]
                ax[right, left] = -self.x_rates[cursor]
                cursor += 1
        cursor = 0
        for y_index in (1, 3, 5):
            for x_index in range(7):
                left = 7 * x_index + y_index
                right = left + 1
                ay[left, right] = self.y_rates[cursor]
                ay[right, left] = -self.y_rates[cursor]
                cursor += 1
        diffusion[torch.arange(1, 49), torch.arange(1, 49)] = -torch.exp(self.diffusion_log_rates)
        return ax, ay, diffusion

    def step(
        self,
        coefficients: torch.Tensor,
        parameters: torch.Tensor,
        *,
        rule: str,
    ) -> torch.Tensor:
        if rule != "combined":
            raise ValueError("E13 recovery controls freeze the combined rule")
        return _combined_step(self, coefficients, parameters)

    def forward(self, coefficients: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
        return self.step(coefficients, parameters, rule="combined")


class ModeTiedGenerator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.x_rates = nn.Parameter(torch.zeros(3, dtype=torch.float64))
        self.y_rates = nn.Parameter(torch.zeros(3, dtype=torch.float64))
        self.x_diffusion_log_rates = nn.Parameter(torch.zeros(3, dtype=torch.float64))
        self.y_diffusion_log_rates = nn.Parameter(torch.zeros(3, dtype=torch.float64))

    def matrices(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ax = self.x_rates.new_zeros((49, 49))
        ay = self.y_rates.new_zeros((49, 49))
        diffusion = self.x_rates.new_zeros((49, 49))
        for harmonic, x_index in enumerate((1, 3, 5)):
            for y_index in range(7):
                left = 7 * x_index + y_index
                right = 7 * (x_index + 1) + y_index
                ax[left, right] = self.x_rates[harmonic]
                ax[right, left] = -self.x_rates[harmonic]
        for harmonic, y_index in enumerate((1, 3, 5)):
            for x_index in range(7):
                left = 7 * x_index + y_index
                right = left + 1
                ay[left, right] = self.y_rates[harmonic]
                ay[right, left] = -self.y_rates[harmonic]
        x_rates = torch.exp(self.x_diffusion_log_rates)
        y_rates = torch.exp(self.y_diffusion_log_rates)
        frequency_group = (None, 0, 0, 1, 1, 2, 2)
        for x_index in range(7):
            for y_index in range(7):
                index = 7 * x_index + y_index
                value = self.x_rates.new_zeros(())
                x_group = frequency_group[x_index]
                y_group = frequency_group[y_index]
                if x_group is not None:
                    value = value + x_rates[x_group]
                if y_group is not None:
                    value = value + y_rates[y_group]
                diffusion[index, index] = -value
        return ax, ay, diffusion

    def step(
        self,
        coefficients: torch.Tensor,
        parameters: torch.Tensor,
        *,
        rule: str,
    ) -> torch.Tensor:
        if rule != "combined":
            raise ValueError("E13 recovery controls freeze the combined rule")
        return _combined_step(self, coefficients, parameters)

    def forward(self, coefficients: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
        return self.step(coefficients, parameters, rule="combined")


Generator = StructuredGenerator | SupportSparseGenerator | ModeTiedGenerator | FixedGenerator


def _combined_step(
    generator: StructuredGenerator | SupportSparseGenerator | ModeTiedGenerator,
    coefficients: torch.Tensor,
    parameters: torch.Tensor,
) -> torch.Tensor:
    ax, ay, diffusion = generator.matrices()
    vx = parameters[:, 0].view(-1, 1, 1)
    vy = parameters[:, 1].view(-1, 1, 1)
    nu = parameters[:, 2].view(-1, 1, 1)
    dt = parameters[:, 3].view(-1, 1, 1)
    matrix = vx * ax + vy * ay + nu * diffusion
    evolved = torch.matrix_exp(dt * matrix) @ coefficients[:, :49]
    return torch.cat((evolved, coefficients[:, 49:].clone()), dim=1)


def component_parameters(
    model: StructuredGenerator | SupportSparseGenerator | ModeTiedGenerator,
    component: str,
) -> list[nn.Parameter]:
    if isinstance(model, StructuredGenerator):
        return {
            "A_x": [model.ax_upper],
            "A_y": [model.ay_upper],
            "D": [model.diffusion_log_rate],
        }[component]
    if isinstance(model, SupportSparseGenerator):
        return {
            "A_x": [model.x_rates],
            "A_y": [model.y_rates],
            "D": [model.diffusion_log_rates],
        }[component]
    return {
        "A_x": [model.x_rates],
        "A_y": [model.y_rates],
        "D": [model.x_diffusion_log_rates, model.y_diffusion_log_rates],
    }[component]


def optimize_components(
    model: StructuredGenerator | SupportSparseGenerator | ModeTiedGenerator,
    datasets: dict[str, TrajectorySet],
    cfg: E13Config,
) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    for regime in REGIMES:
        component = REGIME_TO_COMPONENT[regime]
        inputs, targets, parameters = (
            tensor.contiguous() for tensor in datasets[regime].transitions
        )
        if inputs.shape != (2048, 52, 1) or targets.shape != (2048, 52, 1):
            raise RuntimeError(f"E13 complete population shape drift for {regime}")
        if parameters.shape != (2048, 4):
            raise RuntimeError(f"E13 parameter population shape drift for {regime}")
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
            inputs: torch.Tensor = inputs,
            targets: torch.Tensor = targets,
            parameters: torch.Tensor = parameters,
            optimized: list[nn.Parameter] = optimized,
            history: list[dict[str, float | int]] = history,
        ) -> torch.Tensor:
            optimizer.zero_grad(set_to_none=True)
            loss = normalized_loss(model(inputs, parameters), targets)
            loss.backward()
            gradients = [parameter.grad.reshape(-1) for parameter in optimized]
            gradient_norm = torch.linalg.vector_norm(torch.cat(gradients))
            history.append(
                {
                    "closure": len(history),
                    "loss": float(loss.item()),
                    "gradient_norm": float(gradient_norm.item()),
                }
            )
            return loss

        optimizer.step(closure)
        state = optimizer.state[optimized[0]]
        reports[component] = {
            "population_shape": list(inputs.shape),
            "parameter_shape": list(parameters.shape),
            "n_iter": int(state["n_iter"]),
            "func_evals": int(state["func_evals"]),
            "closure_count": len(history),
            "first_loss": history[0]["loss"],
            "final_loss": history[-1]["loss"],
            "first_gradient_norm": history[0]["gradient_norm"],
            "final_gradient_norm": history[-1]["gradient_norm"],
            "history": history,
        }
    return reports


def _structure_passes(report: dict[str, Any]) -> bool:
    return all(
        (
            report["ax_skew_residual"] <= 1e-12,
            report["ay_skew_residual"] <= 1e-12,
            report["ax_constant_residual"] <= 1e-12,
            report["ay_constant_residual"] <= 1e-12,
            report["diffusion_off_diagonal_residual"] <= 1e-12,
            report["diffusion_constant_residual"] <= 1e-12,
            report["diffusion_maximum_eigenvalue"] <= 1e-12,
            report["all_finite"],
            report["constant_mode_structurally_fixed"],
            report["inactive_modes_copied"],
        )
    )


def parameterization_preflight() -> dict[str, Any]:
    mask = mask_report()
    models = {
        "full_skew": StructuredGenerator(),
        "support_sparse": SupportSparseGenerator(),
        "mode_tied": ModeTiedGenerator(),
    }
    structures = {name: structure_report(model) for name, model in models.items()}
    expected_counts = {"full_skew": 2304, "support_sparse": 90, "mode_tied": 12}
    passed = mask["hashes_match"] and all(
        structures[name]["parameter_count"] == expected_counts[name]
        and _structure_passes(structures[name])
        for name in models
    )
    oracle_phase = (
        frozen_e13_config().maximum_dt * frozen_e13_config().maximum_speed * 2.0 * math.pi * 3.0
    )
    passed &= oracle_phase < math.pi
    return {
        "mask": mask,
        "structures": structures,
        "expected_parameter_counts": expected_counts,
        "maximum_oracle_phase": oracle_phase,
        "oracle_phase_to_pi": oracle_phase / math.pi,
        "oracle_phase_below_pi": oracle_phase < math.pi,
        "passed": passed,
    }


def _rank_report(matrix: torch.Tensor) -> dict[str, Any]:
    singular_values = torch.linalg.svdvals(matrix)
    tolerance = (
        max(matrix.shape) * torch.finfo(matrix.dtype).eps * singular_values.max().clamp_min(1e-300)
    )
    rank = int((singular_values > tolerance).sum().item())
    condition = float((singular_values.max() / singular_values.min().clamp_min(1e-300)).item())
    return {
        "shape": list(matrix.shape),
        "rank": rank,
        "condition_number": condition,
        "rank_tolerance": float(tolerance.item()),
        "singular_values": [float(value) for value in singular_values],
        "full_rank": rank == min(matrix.shape),
    }


def excitation_report(datasets: dict[str, TrajectorySet]) -> dict[str, Any]:
    covariance = {}
    for regime, dataset in datasets.items():
        inputs = dataset.transitions[0][:, 1:49, 0]
        covariance[regime] = _rank_report(inputs.T @ inputs)
    plane_grams = {"A_x": [], "A_y": []}
    x_inputs = datasets["x_advection"].transitions[0][:, :, 0]
    y_inputs = datasets["y_advection"].transitions[0][:, :, 0]
    for x_index in (1, 3, 5):
        for y_index in range(7):
            indices = (7 * x_index + y_index, 7 * (x_index + 1) + y_index)
            plane_grams["A_x"].append(
                {
                    "indices": list(indices),
                    **_rank_report(x_inputs[:, indices].T @ x_inputs[:, indices]),
                }
            )
    for y_index in (1, 3, 5):
        for x_index in range(7):
            indices = (7 * x_index + y_index, 7 * x_index + y_index + 1)
            plane_grams["A_y"].append(
                {
                    "indices": list(indices),
                    **_rank_report(y_inputs[:, indices].T @ y_inputs[:, indices]),
                }
            )
    jacobian = mode_tied_oracle_jacobian_report(datasets)
    return {
        "input_covariance": covariance,
        "rotation_plane_grams": plane_grams,
        "mode_tied_oracle_jacobian": jacobian,
        "required_plane_grams_full_rank": all(
            record["full_rank"] for records in plane_grams.values() for record in records
        ),
        "mode_tied_jacobian_full_rank": jacobian["full_rank"],
    }


def _mode_tied_outputs(vector: torch.Tensor, datasets: dict[str, TrajectorySet]) -> torch.Tensor:
    model = ModeTiedGenerator()
    parameter_values = {
        "x_rates": vector[0:3],
        "y_rates": vector[3:6],
        "x_diffusion_log_rates": vector[6:9],
        "y_diffusion_log_rates": vector[9:12],
    }
    outputs = []
    for regime in REGIMES:
        inputs, _, parameters = datasets[regime].transitions
        prediction = torch.func.functional_call(
            model,
            parameter_values,
            (inputs, parameters),
        )
        outputs.append(prediction[:, :49].reshape(-1))
    return torch.cat(outputs)


def mode_tied_oracle_jacobian_report(
    datasets: dict[str, TrajectorySet],
) -> dict[str, Any]:
    frequencies = torch.tensor((1.0, 2.0, 3.0), dtype=torch.float64)
    rotation = 2.0 * math.pi * frequencies
    diffusion_log = torch.log(rotation.square())
    oracle_vector = torch.cat((rotation, rotation, diffusion_log, diffusion_log))
    columns = []
    for index in range(12):
        direction = torch.zeros(12, dtype=torch.float64)
        direction[index] = 1.0
        _, derivative = torch.autograd.functional.jvp(
            lambda value: _mode_tied_outputs(value, datasets),
            (oracle_vector,),
            (direction,),
            create_graph=False,
            strict=True,
        )
        columns.append(derivative)
    jacobian = torch.stack(columns, dim=1)
    report = _rank_report(jacobian)
    report["oracle_vector_sha256"] = tensor_hash(oracle_vector)
    return report


def build_frozen_datasets(
    cfg: E13Config, space: PhysicalFunctionSpace
) -> tuple[dict[str, TrajectorySet], dict[str, TrajectorySet]]:
    training = {
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
    return training, validation


def schedules(cfg: E13Config) -> dict[str, torch.Tensor]:
    records = {
        regime: schedule(
            cfg.pretrain_updates,
            cfg.pretrain_batch_per_regime,
            cfg.pretrain_trajectories_per_regime * cfg.rollout_steps,
            seed=cfg.schedule_seed + index,
        )
        for index, regime in enumerate(REGIMES)
    }
    records["fine_tune"] = schedule(
        cfg.fine_tune_updates,
        cfg.fine_tune_batch_size,
        cfg.fewshot_trajectories * cfg.rollout_steps,
        seed=cfg.schedule_seed + 10,
    )
    records["full_control"] = schedule(
        cfg.full_control_updates,
        cfg.full_control_batch_size,
        cfg.full_control_trajectories * cfg.rollout_steps,
        seed=cfg.schedule_seed + 20,
    )
    return records


def replay_lock_report(
    training: dict[str, TrajectorySet],
    validation: dict[str, TrajectorySet],
    schedule_tensors: dict[str, torch.Tensor],
) -> dict[str, Any]:
    lock = json.loads(E12_LOCK_PATH.read_text())
    generated_datasets = {
        **{f"pretrain_{name}": value for name, value in training.items()},
        **{f"{name}_validation": value for name, value in validation.items()},
    }
    dataset_records = {name: _dataset_hash(dataset) for name, dataset in generated_datasets.items()}
    for record in dataset_records.values():
        record.pop("all_finite")
    schedules_report = validate_schedules(schedule_tensors)
    generated_schedule_records = {
        name: {
            "shape": record["shape"],
            "sha256": record["sha256"],
        }
        for name, record in schedules_report["records"].items()
    }
    return {
        "dataset_records": dataset_records,
        "expected_dataset_records": lock["dataset_hashes"],
        "datasets_match": dataset_records == lock["dataset_hashes"],
        "schedule_records": generated_schedule_records,
        "expected_schedule_records": lock["schedule_hashes"],
        "schedules_match": generated_schedule_records == lock["schedule_hashes"],
        "passed": (
            dataset_records == lock["dataset_hashes"]
            and generated_schedule_records == lock["schedule_hashes"]
            and schedules_report["passed"]
        ),
    }


def evaluate_control(
    model: Generator,
    validation: dict[str, TrajectorySet],
    *,
    oracle: FixedGenerator,
    space: PhysicalFunctionSpace,
    canonical_coords: torch.Tensor,
    cfg: E13Config,
) -> dict[str, Any]:
    model_adapter = RuleAdapter(model, "combined")  # type: ignore[arg-type]
    validation_reports = {}
    for regime, dataset in validation.items():
        report, _ = rollout(
            model_adapter,  # type: ignore[arg-type]
            dataset,
            space=space,
            canonical_coords=canonical_coords,
            cfg=cfg,
        )
        report["all_finite"] = _all_finite(report)
        validation_reports[regime] = report
    identification = generator_identification(
        model, oracle, space=space, canonical_coords=canonical_coords  # type: ignore[arg-type]
    )
    structure = structure_report(model)  # type: ignore[arg-type]
    elementary_one = max(validation_reports[name]["one_step_decoded_nrmse"] for name in REGIMES)
    elementary_rollout = max(validation_reports[name]["rollout_decoded_nrmse"] for name in REGIMES)
    composite = validation_reports["composite"]
    gates = {
        "structure": _structure_passes(structure),
        "generator_identification": identification_passes(identification, cfg),
        "high_frequency": (
            composite["final_high_frequency_spectral"]["nrmse"] <= cfg.max_high_frequency_nrmse
        ),
        "elementary_one_step_nonregression": elementary_one <= cfg.max_one_step_nrmse,
        "elementary_rollout_nonregression": elementary_rollout <= cfg.max_rollout_nrmse,
        "zero_shot_rollout": composite["rollout_decoded_nrmse"] <= cfg.max_zero_shot_rollout_nrmse,
        "zero_shot_to_persistence": (
            composite["rollout_decoded_nrmse"] / E12_PERSISTENCE_ROLLOUT
            <= cfg.max_zero_shot_to_persistence_ratio
        ),
        "finite": _all_finite(
            {
                "structure": structure,
                "identification": identification,
                "validation": validation_reports,
            }
        ),
    }
    return {
        "structure": structure,
        "model_sha256": model_hash(model),
        "generator_sha256": _matrix_hashes(model),  # type: ignore[arg-type]
        "generator_identification": identification,
        "validation": validation_reports,
        "maximum_elementary_one_step_decoded_nrmse": elementary_one,
        "maximum_elementary_rollout_decoded_nrmse": elementary_rollout,
        "gates": gates,
        "recovery_pass": all(gates.values()),
    }


def _apply_steps(
    model: nn.Module, values: torch.Tensor, parameters: torch.Tensor, steps: int
) -> torch.Tensor:
    prediction = values
    for _ in range(steps):
        prediction = model(prediction, parameters)
    return prediction


def mode_resolved_records(
    controls: dict[str, Generator],
    *,
    oracle: FixedGenerator,
    space: PhysicalFunctionSpace,
    canonical_coords: torch.Tensor,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, dict[str, Any]]]]:
    basis = torch.eye(52, dtype=torch.float64)[:49].unsqueeze(-1)
    oracle_adapter = RuleAdapter(oracle, "combined")
    records = []
    argmax: dict[str, dict[str, dict[str, Any]]] = {}
    metric_names = (
        "decoded_nrmse",
        "coefficient_nrmse",
        "coefficient_angle_radians",
        "absolute_amplitude_ratio_error",
        "off_target_direction_residual",
    )
    for control_name, generator in controls.items():
        adapter = RuleAdapter(generator, "combined")  # type: ignore[arg-type]
        argmax[control_name] = {}
        for case_name, vx, vy, nu in closure_parameter_cases():
            parameters = torch.tensor([[vx, vy, nu, 0.04]], dtype=torch.float64).expand(49, -1)
            for horizon in (1, 8):
                prediction = _apply_steps(adapter, basis, parameters, horizon)
                target = _apply_steps(oracle_adapter, basis, parameters, horizon)
                predicted_decoded = space.decode(prediction, canonical_coords)
                target_decoded = space.decode(target, canonical_coords)
                decoded = torch.sqrt(
                    (predicted_decoded - target_decoded).square().sum(dim=(1, 2))
                    / target_decoded.square().sum(dim=(1, 2)).clamp_min(1e-300)
                )
                difference = (prediction - target).reshape(49, -1)
                target_flat = target.reshape(49, -1)
                prediction_flat = prediction.reshape(49, -1)
                target_norm = torch.linalg.vector_norm(target_flat, dim=1)
                prediction_norm = torch.linalg.vector_norm(prediction_flat, dim=1)
                coefficient = torch.linalg.vector_norm(difference, dim=1) / target_norm
                cosine = (
                    (prediction_flat * target_flat).sum(dim=1)
                    / (prediction_norm * target_norm).clamp_min(1e-300)
                ).clamp(-1.0, 1.0)
                angle = torch.arccos(cosine)
                amplitude = prediction_norm / target_norm
                projection = (
                    (prediction_flat * target_flat).sum(dim=1)
                    / target_norm.square().clamp_min(1e-300)
                ).unsqueeze(1) * target_flat
                off_target = (
                    torch.linalg.vector_norm(prediction_flat - projection, dim=1) / target_norm
                )
                for basis_index in range(49):
                    record = {
                        "control": control_name,
                        "basis_index": basis_index,
                        "x_component": SEMANTIC_COMPONENTS[basis_index // 7],
                        "y_component": SEMANTIC_COMPONENTS[basis_index % 7],
                        "case_name": case_name,
                        "parameters": {
                            "v_x": vx,
                            "v_y": vy,
                            "nu": nu,
                            "dt": 0.04,
                        },
                        "horizon": horizon,
                        "decoded_nrmse": float(decoded[basis_index].item()),
                        "coefficient_nrmse": float(coefficient[basis_index].item()),
                        "coefficient_angle_radians": float(angle[basis_index].item()),
                        "amplitude_ratio": float(amplitude[basis_index].item()),
                        "absolute_amplitude_ratio_error": float(
                            abs(amplitude[basis_index].item() - 1.0)
                        ),
                        "off_target_direction_residual": float(off_target[basis_index].item()),
                    }
                    records.append(record)
                    for metric in metric_names:
                        current = argmax[control_name].get(metric)
                        if current is None or record[metric] > current["value"]:
                            argmax[control_name][metric] = {
                                "value": record[metric],
                                "key": {
                                    "basis_index": basis_index,
                                    "case_name": case_name,
                                    "horizon": horizon,
                                },
                            }
    return records, argmax


def coverage_report(
    evaluations: dict[str, Any],
    records: list[dict[str, Any]],
    argmax: dict[str, Any],
    recovery_training: dict[str, Any],
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
    validation_cells = sum(
        len(control_report["validation"]) for control_report in evaluations.values()
    )
    argmax_cells = sum(len(values) for values in argmax.values())
    report = {
        "generator_identification_cells": len(evaluations),
        "validation_cells": validation_cells,
        "mode_resolved_records": len(records),
        "unique_mode_resolved_keys": len(keys),
        "mode_argmax_cells": argmax_cells,
        "recovery_training_controls": len(recovery_training),
    }
    report["passed"] = report == {
        **report,
        "generator_identification_cells": 6,
        "validation_cells": 24,
        "mode_resolved_records": 10584,
        "unique_mode_resolved_keys": 10584,
        "mode_argmax_cells": 30,
        "recovery_training_controls": 5,
    }
    return report


def classify(
    *,
    preflight_passed: bool,
    reproduction_passed: bool,
    evaluations: dict[str, Any],
    excitation: dict[str, Any],
) -> str:
    if not preflight_passed:
        return "e13_preflight_failed"
    if not reproduction_passed:
        return "e12_reproduction_failed"
    if (
        evaluations["full_skew_lbfgs_neutral"]["recovery_pass"]
        or evaluations["full_skew_lbfgs_polish"]["recovery_pass"]
    ):
        return "full_parameterization_deterministic_recovery_succeeds"
    if evaluations["support_sparse_lbfgs"]["recovery_pass"]:
        return "support_restriction_required_under_frozen_solvers"
    if evaluations["mode_tied_lbfgs"]["recovery_pass"]:
        return "mode_tying_required_under_frozen_solvers"
    if (
        not excitation["required_plane_grams_full_rank"]
        or not excitation["mode_tied_jacobian_full_rank"]
    ):
        return "elementary_excitation_rank_deficient"
    return "recovery_controls_not_qualified"


def _early_failure(
    cfg: E13Config,
    run_dir: Path,
    provenance_report: dict[str, Any],
    parameterization: dict[str, Any],
) -> dict[str, Any]:
    result = {
        "schema_version": 1,
        "experiment": "canonical_latent_e13_identifiability_audit",
        "config": asdict(cfg),
        "provenance": provenance_report,
        "parameterization_preflight": parameterization,
        "classification": "e13_preflight_failed",
        "state_reads": {"training": 0, "validation": 0, "heldout": 0},
        "optimizer_updates": 0,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def _run_once(cfg: E13Config, *, run_dir: Path) -> dict[str, Any]:
    if asdict(cfg) != asdict(frozen_e13_config()):
        raise ValueError("E13 requires the exact frozen configuration")
    provenance_report = provenance()
    parameterization = parameterization_preflight()
    if not provenance_report["passed"] or not parameterization["passed"]:
        return _early_failure(cfg, run_dir, provenance_report, parameterization)

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
    oracle_report = oracle_preflight(cfg, space, oracle)
    if not oracle_report["passed"]:
        parameterization["oracle_preflight"] = oracle_report
        parameterization["passed"] = False
        return _early_failure(cfg, run_dir, provenance_report, parameterization)

    training, validation = build_frozen_datasets(cfg, space)
    schedule_tensors = schedules(cfg)
    lock_report = replay_lock_report(training, validation, schedule_tensors)
    if not lock_report["passed"]:
        result = {
            "schema_version": 1,
            "experiment": "canonical_latent_e13_identifiability_audit",
            "config": asdict(cfg),
            "provenance": provenance_report,
            "parameterization_preflight": parameterization,
            "oracle_preflight": oracle_report,
            "replay_lock": lock_report,
            "classification": "e12_reproduction_failed",
            "state_reads": {
                "training": 768,
                "validation": 256,
                "heldout": 0,
            },
            "optimizer_updates": 0,
        }
        (run_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        return result

    initial = StructuredGenerator()
    replay = copy.deepcopy(initial)
    replay_training = train_elementary(
        replay,
        training,
        {name: schedule_tensors[name] for name in REGIMES},
        cfg,
    )
    canonical_coords, _ = canonical_grid(space, cfg.canonical_query_resolution)
    replay_identification = generator_identification(
        replay, oracle, space=space, canonical_coords=canonical_coords
    )
    replay_composite, _ = rollout(
        RuleAdapter(replay, "combined"),
        validation["composite"],
        space=space,
        canonical_coords=canonical_coords,
        cfg=cfg,
    )
    reproduction_checks = {
        "initial_model_sha256": model_hash(initial) == EXPECTED_E12_INITIAL_SHA256,
        "elementary_model_sha256": model_hash(replay) == EXPECTED_E12_ELEMENTARY_SHA256,
        "generator_sha256": _matrix_hashes(replay) == EXPECTED_E12_GENERATOR_SHA256,
        "first_loss": replay_training["first_loss"] == EXPECTED_E12_FIRST_LOSS,
        "final_loss": replay_training["final_loss"] == EXPECTED_E12_FINAL_LOSS,
        "updates": replay_training["updates"] == 1500,
        "examples_per_regime": replay_training["examples_per_regime"] == 48000,
        "basis_action": abs(
            replay_identification["maximum_basis_action_decoded_nrmse"] - EXPECTED_E12_BASIS_ACTION
        )
        <= 1e-12,
        "zero_shot_rollout": abs(
            replay_composite["rollout_decoded_nrmse"] - EXPECTED_E12_ZERO_SHOT_ROLLOUT
        )
        <= 1e-12,
    }
    reproduction = {
        "checks": reproduction_checks,
        "training": replay_training,
        "initial_model_sha256": model_hash(initial),
        "elementary_model_sha256": model_hash(replay),
        "generator_sha256": _matrix_hashes(replay),
        "basis_action_decoded_nrmse": replay_identification["maximum_basis_action_decoded_nrmse"],
        "zero_shot_rollout_decoded_nrmse": replay_composite["rollout_decoded_nrmse"],
        "passed": all(reproduction_checks.values()),
    }
    if not reproduction["passed"]:
        result = {
            "schema_version": 1,
            "experiment": "canonical_latent_e13_identifiability_audit",
            "config": asdict(cfg),
            "provenance": provenance_report,
            "parameterization_preflight": parameterization,
            "oracle_preflight": oracle_report,
            "replay_lock": lock_report,
            "e12_reproduction": reproduction,
            "classification": "e12_reproduction_failed",
            "state_reads": {"training": 768, "validation": 256, "heldout": 0},
            "optimizer_updates": 1500,
        }
        (run_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        return result

    excitation = excitation_report(training)
    neutral = StructuredGenerator()
    polish = copy.deepcopy(replay)
    sparse = SupportSparseGenerator()
    tied = ModeTiedGenerator()
    recovery_training = {
        "e12_adamw_replay": replay_training,
        "full_skew_lbfgs_neutral": optimize_components(neutral, training, cfg),
        "full_skew_lbfgs_polish": optimize_components(polish, training, cfg),
        "support_sparse_lbfgs": optimize_components(sparse, training, cfg),
        "mode_tied_lbfgs": optimize_components(tied, training, cfg),
    }
    controls: dict[str, Generator] = {
        "e12_adamw_replay": replay,
        "full_skew_lbfgs_neutral": neutral,
        "full_skew_lbfgs_polish": polish,
        "support_sparse_lbfgs": sparse,
        "mode_tied_lbfgs": tied,
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
        controls, oracle=oracle, space=space, canonical_coords=canonical_coords
    )
    coverage = coverage_report(evaluations, mode_records, mode_argmax, recovery_training)
    if not coverage["passed"]:
        raise RuntimeError(f"E13 evidence coverage failure: {coverage}")
    classification = classify(
        preflight_passed=True,
        reproduction_passed=True,
        evaluations=evaluations,
        excitation=excitation,
    )
    result = {
        "schema_version": 1,
        "experiment": "canonical_latent_e13_identifiability_audit",
        "config": asdict(cfg),
        "config_sha256": hashlib.sha256(
            json.dumps(asdict(cfg), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "provenance": provenance_report,
        "parameterization_preflight": parameterization,
        "oracle_preflight": oracle_report,
        "replay_lock": lock_report,
        "e12_reproduction": reproduction,
        "excitation": excitation,
        "recovery_training": recovery_training,
        "evaluations": evaluations,
        "mode_resolved": mode_records,
        "mode_argmax": mode_argmax,
        "coverage": coverage,
        "classification": classification,
        "state_reads": {"training": 768, "validation": 256, "heldout": 0},
        "optimizer_step_calls": 1512,
        "optimizer_iterations": 1500
        + sum(
            component["n_iter"]
            for control in recovery_training.values()
            if isinstance(control, dict) and "A_x" in control
            for component in control.values()
        ),
        "optimizer_closure_evaluations": 1500
        + sum(
            component["closure_count"]
            for control in recovery_training.values()
            if isinstance(control, dict) and "A_x" in control
            for component in control.values()
        ),
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
        "reproducibility": {"deterministic_algorithms": True},
    }
    if not _all_finite(result):
        raise RuntimeError("E13 result contains a nonfinite numeric value")
    (run_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def run_e13(cfg: E13Config, *, run_dir: Path) -> dict[str, Any]:
    if asdict(cfg) != asdict(frozen_e13_config()):
        raise ValueError("E13 requires the exact frozen configuration")
    first = _run_once(cfg, run_dir=run_dir / "replicate_a")
    second = _run_once(cfg, run_dir=run_dir / "replicate_b")
    first_bytes = json.dumps(first, sort_keys=True, separators=(",", ":")).encode()
    second_bytes = json.dumps(second, sort_keys=True, separators=(",", ":")).encode()
    if first_bytes != second_bytes:
        raise RuntimeError("E13 complete replicates are not byte-identical")
    complete_sha256 = hashlib.sha256(first_bytes).hexdigest()
    complete = {
        **first,
        "replication": {
            "byte_identical_complete_runs": True,
            "replicate_a_sha256": complete_sha256,
            "replicate_b_sha256": hashlib.sha256(second_bytes).hexdigest(),
            "complete_result_sha256": complete_sha256,
        },
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "complete_result.json").write_text(
        json.dumps(complete, indent=2, sort_keys=True) + "\n"
    )
    return complete


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("/private/tmp/canonical_latent_e13_identifiability_audit"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_e13(frozen_e13_config(), run_dir=args.run_dir)
    print(json.dumps({"classification": result["classification"]}, sort_keys=True))


if __name__ == "__main__":
    main()
