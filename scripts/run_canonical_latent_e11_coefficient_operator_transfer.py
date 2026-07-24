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

if __package__:
    from scripts.run_canonical_latent_e4_capacity_ladder import (
        high_frequency_spectral_report,
    )
    from scripts.run_canonical_latent_e7_function_space import (
        FunctionSpaceConfig,
        PhysicalFunctionSpace,
        _decoded_mismatch,
        _relative_mismatch,
    )
    from scripts.run_canonical_latent_e9_geometry_universal_projection import (
        GeometryProjectionConfig,
        encode_paths,
        geometry_samples,
    )
else:
    from run_canonical_latent_e4_capacity_ladder import (  # type: ignore[no-redef]
        high_frequency_spectral_report,
    )
    from run_canonical_latent_e7_function_space import (  # type: ignore[no-redef]
        FunctionSpaceConfig,
        PhysicalFunctionSpace,
        _decoded_mismatch,
        _relative_mismatch,
    )
    from run_canonical_latent_e9_geometry_universal_projection import (  # type: ignore[no-redef]
        GeometryProjectionConfig,
        encode_paths,
        geometry_samples,
    )
from ups.eval.latent_qualification import effective_rank, global_nrmse

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    REPO_ROOT
    / "docs/research/2026-07-24-canonical-latent-e11-coefficient-operator-transfer-contract.md"
)
RUNNER_PATH = Path(__file__).resolve()
REGIMES = ("x_advection", "y_advection", "diffusion")


@dataclass(frozen=True)
class E11Config:
    truth_resolution: int = 64
    canonical_query_resolution: int = 18
    rollout_steps: int = 8
    pretrain_trajectories_per_regime: int = 256
    fewshot_trajectories: int = 8
    full_control_trajectories: int = 256
    validation_trajectories: int = 64
    pretrain_state_seed: int = 51_001
    pretrain_parameter_seed: int = 51_101
    fewshot_state_seed: int = 52_001
    fewshot_parameter_seed: int = 52_101
    full_state_seed: int = 53_001
    full_parameter_seed: int = 53_101
    validation_state_seed: int = 61_001
    validation_parameter_seed: int = 61_101
    x_validation_parameter_seed: int = 61_102
    y_validation_parameter_seed: int = 61_103
    diffusion_validation_parameter_seed: int = 61_104
    model_seed: int = 71_001
    schedule_seed: int = 72_001
    geometry_seed_start: int = 40_000
    geometry_realizations: int = 4
    hidden_width: int = 96
    pretrain_updates: int = 1_500
    pretrain_batch_per_regime: int = 32
    fine_tune_updates: int = 400
    fine_tune_batch_size: int = 64
    full_control_updates: int = 1_500
    full_control_batch_size: int = 96
    pretrain_learning_rate: float = 2e-3
    fine_tune_learning_rate: float = 1e-3
    weight_decay: float = 1e-6
    minimum_speed: float = 0.20
    maximum_speed: float = 1.00
    minimum_diffusivity: float = 0.01
    maximum_diffusivity: float = 0.08
    minimum_dt: float = 0.02
    maximum_dt: float = 0.06
    extrapolation_dt: float = 0.075
    closure_max_decoded_nrmse: float = 0.01
    closure_max_composition_error: float = 1e-10
    max_one_step_nrmse: float = 0.03
    max_rollout_nrmse: float = 0.08
    max_high_frequency_nrmse: float = 0.15
    max_pretrained_to_scratch_ratio: float = 0.80
    max_pretrained_to_full_ratio: float = 1.25
    max_zero_shot_rollout_nrmse: float = 0.20
    max_zero_shot_to_persistence_ratio: float = 0.75
    max_elementary_retention_nrmse: float = 0.08
    max_elementary_retention_ratio: float = 1.25
    max_extrapolation_nrmse: float = 0.12
    max_semigroup_consistency_nrmse: float = 0.05
    max_cross_observation_mismatch: float = 0.01
    max_mean_mode_relative_error: float = 1e-3
    max_advection_l2_drift: float = 0.05
    minimum_diffusion_energy_monotonic_fraction: float = 0.99

    def __post_init__(self) -> None:
        if self.truth_resolution != 64 or self.rollout_steps != 8:
            raise ValueError("E11 truth resolution and rollout length are frozen")
        if self.fewshot_trajectories != 8 or self.validation_trajectories != 64:
            raise ValueError("E11 few-shot and validation counts are frozen")
        if self.geometry_realizations != 4:
            raise ValueError("E11 freezes four stochastic geometry realizations")


@dataclass(frozen=True)
class TrajectorySet:
    name: str
    coefficients: torch.Tensor
    parameters: torch.Tensor

    @property
    def transitions(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs = self.coefficients[:, :-1].reshape(-1, 52, 1)
        targets = self.coefficients[:, 1:].reshape(-1, 52, 1)
        repeated_parameters = (
            self.parameters[:, None, :]
            .expand(-1, self.coefficients.shape[1] - 1, -1)
            .reshape(-1, 4)
        )
        return inputs, targets, repeated_parameters


def frozen_e11_config() -> E11Config:
    return E11Config()


def modal_scales() -> torch.Tensor:
    one_dimensional_frequencies = (0, 1, 1, 2, 2, 3, 3)
    scales = []
    for x_frequency in one_dimensional_frequencies:
        for y_frequency in one_dimensional_frequencies:
            if x_frequency == 0 and y_frequency == 0:
                scales.append(0.20)
            else:
                scales.append(0.7 / (1.0 + x_frequency**2 + y_frequency**2) ** 1.5)
    scales.extend((1.0, 1.0, 1.0))
    return torch.tensor(scales, dtype=torch.float64).view(1, 52, 1)


def sample_coefficients(count: int, *, seed: int) -> torch.Tensor:
    coefficients = torch.randn(
        count,
        52,
        1,
        generator=torch.Generator().manual_seed(seed),
        dtype=torch.float64,
    )
    coefficients = coefficients * modal_scales()
    coefficients[:, 49:] = 0.0
    return coefficients


def sample_parameters(
    count: int,
    regime: str,
    *,
    seed: int,
    cfg: E11Config,
    dt_override: float | None = None,
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    speed_magnitude = cfg.minimum_speed + (cfg.maximum_speed - cfg.minimum_speed) * torch.rand(
        count, generator=generator, dtype=torch.float64
    )
    signs = torch.where(
        torch.rand(count, generator=generator, dtype=torch.float64) < 0.5,
        -torch.ones(count, dtype=torch.float64),
        torch.ones(count, dtype=torch.float64),
    )
    second_speed_magnitude = cfg.minimum_speed + (
        cfg.maximum_speed - cfg.minimum_speed
    ) * torch.rand(count, generator=generator, dtype=torch.float64)
    second_signs = torch.where(
        torch.rand(count, generator=generator, dtype=torch.float64) < 0.5,
        -torch.ones(count, dtype=torch.float64),
        torch.ones(count, dtype=torch.float64),
    )
    diffusivity = cfg.minimum_diffusivity + (
        cfg.maximum_diffusivity - cfg.minimum_diffusivity
    ) * torch.rand(count, generator=generator, dtype=torch.float64)
    dt = cfg.minimum_dt + (cfg.maximum_dt - cfg.minimum_dt) * torch.rand(
        count, generator=generator, dtype=torch.float64
    )
    if dt_override is not None:
        dt.fill_(dt_override)

    zeros = torch.zeros(count, dtype=torch.float64)
    if regime == "x_advection":
        values = (signs * speed_magnitude, zeros, zeros, dt)
    elif regime == "y_advection":
        values = (zeros, signs * speed_magnitude, zeros, dt)
    elif regime == "diffusion":
        values = (zeros, zeros, diffusivity, dt)
    elif regime == "composite":
        values = (
            signs * speed_magnitude,
            second_signs * second_speed_magnitude,
            diffusivity,
            dt,
        )
    else:
        raise ValueError(f"unknown E11 regime: {regime}")
    return torch.stack(values, dim=1)


def truth_grid(
    space: PhysicalFunctionSpace, resolution: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    axis = torch.arange(resolution, dtype=torch.float64) / resolution
    x, y = torch.meshgrid(axis, axis, indexing="ij")
    coords = torch.stack((x, y), dim=-1).reshape(1, -1, 2)
    measure = torch.full(
        (1, resolution * resolution, 1),
        1.0 / (resolution * resolution),
        dtype=torch.float64,
    )
    basis = space.basis(coords)
    return coords, measure, basis


def canonical_grid(
    space: PhysicalFunctionSpace, resolution: int
) -> tuple[torch.Tensor, torch.Tensor]:
    axis = (torch.arange(resolution, dtype=torch.float64) + 0.5) / resolution
    x, y = torch.meshgrid(axis, axis, indexing="ij")
    coords = torch.stack((x, y), dim=-1).reshape(1, -1, 2)
    return coords, space.basis(coords)


def evolve_periodic(
    values: torch.Tensor,
    parameters: torch.Tensor,
    *,
    resolution: int,
    dt_multiplier: float = 1.0,
) -> torch.Tensor:
    fields = values.reshape(values.shape[0], resolution, resolution)
    frequencies = torch.fft.fftfreq(resolution, d=1.0 / resolution, dtype=torch.float64)
    k_x, k_y = torch.meshgrid(frequencies, frequencies, indexing="ij")
    velocity_x = parameters[:, 0].view(-1, 1, 1)
    velocity_y = parameters[:, 1].view(-1, 1, 1)
    diffusivity = parameters[:, 2].view(-1, 1, 1)
    dt = parameters[:, 3].view(-1, 1, 1) * dt_multiplier
    generator = -diffusivity * (2.0 * math.pi) ** 2 * (
        k_x.square() + k_y.square()
    ) - 1j * 2.0 * math.pi * (velocity_x * k_x + velocity_y * k_y)
    multiplier = torch.exp(dt * generator)
    evolved = torch.fft.ifft2(torch.fft.fft2(fields) * multiplier).real
    return evolved.reshape(values.shape[0], resolution * resolution, 1)


def project_values(
    space: PhysicalFunctionSpace,
    values: torch.Tensor,
    coords: torch.Tensor,
    measure: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    return space.project(
        values,
        coords.expand(values.shape[0], -1, -1),
        measure.expand(values.shape[0], -1, -1),
    )


def build_trajectories(
    name: str,
    *,
    count: int,
    state_seed: int,
    parameter_seed: int,
    regime: str,
    cfg: E11Config,
    space: PhysicalFunctionSpace,
    dt_override: float | None = None,
) -> TrajectorySet:
    coefficients = sample_coefficients(count, seed=state_seed)
    parameters = sample_parameters(
        count,
        regime,
        seed=parameter_seed,
        cfg=cfg,
        dt_override=dt_override,
    )
    coords, measure, basis = truth_grid(space, cfg.truth_resolution)
    values = basis.expand(count, -1, -1) @ coefficients
    trajectory = [coefficients]
    for _ in range(cfg.rollout_steps):
        values = evolve_periodic(
            values,
            parameters,
            resolution=cfg.truth_resolution,
        )
        projected, _ = project_values(space, values, coords, measure)
        trajectory.append(projected)
    return TrajectorySet(name, torch.stack(trajectory, dim=1), parameters)


def closure_preflight(
    cfg: E11Config,
    space: PhysicalFunctionSpace,
) -> dict[str, Any]:
    parameter_cases = []
    for speed in (0.20, 0.60, 1.00):
        for sign in (-1.0, 1.0):
            parameter_cases.append((f"x_{sign * speed:+.2f}", sign * speed, 0.0, 0.0))
            parameter_cases.append((f"y_{sign * speed:+.2f}", 0.0, sign * speed, 0.0))
    for diffusivity in (0.01, 0.045, 0.08):
        parameter_cases.append((f"diffusion_{diffusivity:.3f}", 0.0, 0.0, diffusivity))
    parameter_cases.extend(
        (
            ("composite_a", -1.0, 0.2, 0.01),
            ("composite_b", 0.6, -0.6, 0.045),
            ("composite_c", 1.0, 1.0, 0.08),
        )
    )

    coords, measure, basis = truth_grid(space, cfg.truth_resolution)
    active_basis_coefficients = torch.eye(52, dtype=torch.float64)[:49].unsqueeze(-1)
    initial_values = basis.expand(49, -1, -1) @ active_basis_coefficients
    records = []
    maximum_projection_error = 0.0
    maximum_composition_error = 0.0
    minimum_rank = 52
    for name, velocity_x, velocity_y, diffusivity in parameter_cases:
        parameters = torch.tensor(
            [[velocity_x, velocity_y, diffusivity, 0.04]],
            dtype=torch.float64,
        ).expand(49, -1)
        one_step_values = evolve_periodic(
            initial_values,
            parameters,
            resolution=cfg.truth_resolution,
        )
        one_step_coefficients, design = project_values(space, one_step_values, coords, measure)
        one_step_decoded = basis.expand(49, -1, -1) @ one_step_coefficients
        one_step_error = global_nrmse(one_step_decoded, one_step_values)

        repeated_values = initial_values
        for _ in range(cfg.rollout_steps):
            repeated_values = evolve_periodic(
                repeated_values,
                parameters,
                resolution=cfg.truth_resolution,
            )
        repeated_coefficients, _ = project_values(space, repeated_values, coords, measure)
        repeated_decoded = basis.expand(49, -1, -1) @ repeated_coefficients
        repeated_projection_error = global_nrmse(repeated_decoded, repeated_values)
        direct_values = evolve_periodic(
            initial_values,
            parameters,
            resolution=cfg.truth_resolution,
            dt_multiplier=cfg.rollout_steps,
        )
        direct_coefficients, _ = project_values(space, direct_values, coords, measure)
        composition_error = _relative_mismatch(repeated_coefficients, direct_coefficients)
        maximum_projection_error = max(
            maximum_projection_error, one_step_error, repeated_projection_error
        )
        maximum_composition_error = max(maximum_composition_error, composition_error)
        minimum_rank = min(minimum_rank, int(design["rank"]))
        records.append(
            {
                "case": name,
                "one_step_decoded_nrmse": one_step_error,
                "eight_step_decoded_nrmse": repeated_projection_error,
                "semigroup_composition_error": composition_error,
                "design": design,
            }
        )
    passed = (
        minimum_rank == 52
        and maximum_projection_error <= cfg.closure_max_decoded_nrmse
        and maximum_composition_error <= cfg.closure_max_composition_error
    )
    return {
        "active_basis_vectors": 49,
        "inactive_trend_vectors": 3,
        "parameter_cases": len(parameter_cases),
        "minimum_projection_rank": minimum_rank,
        "maximum_truth_to_projection_decoded_nrmse": maximum_projection_error,
        "maximum_semigroup_composition_error": maximum_composition_error,
        "records": records,
        "passed": passed,
    }


class ResidualCoefficientOperator(nn.Module):
    def __init__(self, cfg: E11Config):
        super().__init__()
        self.register_buffer("coefficient_scale", modal_scales())
        self.register_buffer(
            "parameter_scale",
            torch.tensor(
                [1.0, 1.0, cfg.maximum_diffusivity, cfg.extrapolation_dt],
                dtype=torch.float64,
            ).view(1, 4),
        )
        self.network = nn.Sequential(
            nn.Linear(56, cfg.hidden_width, dtype=torch.float64),
            nn.GELU(),
            nn.Linear(cfg.hidden_width, cfg.hidden_width, dtype=torch.float64),
            nn.GELU(),
            nn.Linear(cfg.hidden_width, 52, dtype=torch.float64),
        )
        output = self.network[-1]
        assert isinstance(output, nn.Linear)
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)

    def forward(self, coefficients: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
        normalized_coefficients = (coefficients / self.coefficient_scale).squeeze(-1)
        normalized_parameters = parameters / self.parameter_scale
        normalized_increment = self.network(
            torch.cat((normalized_coefficients, normalized_parameters), dim=1)
        ).unsqueeze(-1)
        return coefficients + normalized_increment * self.coefficient_scale


def model_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def schedule(
    updates: int,
    batch_size: int,
    population: int,
    *,
    seed: int,
) -> torch.Tensor:
    return torch.randint(
        population,
        (updates, batch_size),
        generator=torch.Generator().manual_seed(seed),
    )


def tensor_hash(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.cpu().contiguous().numpy().tobytes()).hexdigest()


def normalized_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return ((prediction - target) / modal_scales()).square().mean()


def train_elementary(
    model: ResidualCoefficientOperator,
    datasets: dict[str, TrajectorySet],
    schedules: dict[str, torch.Tensor],
    cfg: E11Config,
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
    }


def train_single(
    model: ResidualCoefficientOperator,
    dataset: TrajectorySet,
    indices: torch.Tensor,
    *,
    learning_rate: float,
    weight_decay: float,
) -> dict[str, Any]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
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
        "updates": indices.shape[0],
        "examples": indices.numel(),
        "first_loss": losses[0],
        "final_loss": losses[-1],
    }


def parameter_quartile_reports(
    prediction: torch.Tensor,
    target: torch.Tensor,
    parameters: torch.Tensor,
    target_decoded: torch.Tensor,
    prediction_decoded: torch.Tensor,
) -> dict[str, list[dict[str, float | int | bool]]]:
    reports = {}
    for name, values in (
        ("abs_vx", parameters[:, 0].abs()),
        ("abs_vy", parameters[:, 1].abs()),
        ("nu", parameters[:, 2]),
        ("dt", parameters[:, 3]),
    ):
        if torch.equal(values.min(), values.max()):
            reports[name] = [
                {
                    "count": int(values.numel()),
                    "lower": float(values[0].item()),
                    "upper": float(values[0].item()),
                    "constant": True,
                    "coefficient_nrmse": global_nrmse(prediction, target),
                    "decoded_nrmse": global_nrmse(prediction_decoded, target_decoded),
                }
            ]
            continue
        boundaries = torch.quantile(
            values, torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float64)
        )
        quartiles = []
        for index in range(4):
            if index == 3:
                mask = (values >= boundaries[index]) & (values <= boundaries[index + 1])
            else:
                mask = (values >= boundaries[index]) & (values < boundaries[index + 1])
            quartiles.append(
                {
                    "count": int(mask.sum().item()),
                    "lower": float(boundaries[index].item()),
                    "upper": float(boundaries[index + 1].item()),
                    "constant": False,
                    "coefficient_nrmse": global_nrmse(prediction[mask], target[mask]),
                    "decoded_nrmse": global_nrmse(prediction_decoded[mask], target_decoded[mask]),
                }
            )
        reports[name] = quartiles
    return reports


def rollout(
    model: ResidualCoefficientOperator | None,
    dataset: TrajectorySet,
    *,
    space: PhysicalFunctionSpace,
    canonical_coords: torch.Tensor,
    cfg: E11Config,
) -> tuple[dict[str, Any], torch.Tensor]:
    prediction = dataset.coefficients[:, 0]
    predictions = [prediction]
    for _ in range(cfg.rollout_steps):
        if model is None:
            prediction = prediction
        else:
            prediction = model(prediction, dataset.parameters)
        predictions.append(prediction)
    predicted_sequence = torch.stack(predictions, dim=1)
    coefficient_by_step = []
    decoded_by_step = []
    for step in range(1, cfg.rollout_steps + 1):
        target = dataset.coefficients[:, step]
        predicted = predicted_sequence[:, step]
        coefficient_by_step.append(global_nrmse(predicted, target))
        decoded_by_step.append(
            global_nrmse(
                space.decode(predicted, canonical_coords),
                space.decode(target, canonical_coords),
            )
        )
    final_target = dataset.coefficients[:, -1]
    final_prediction = predicted_sequence[:, -1]
    final_target_decoded = space.decode(final_target, canonical_coords)
    final_prediction_decoded = space.decode(final_prediction, canonical_coords)
    target_mean = final_target[:, 0]
    prediction_mean = final_prediction[:, 0]
    mean_mode_error = global_nrmse(prediction_mean, target_mean)
    return (
        {
            "one_step_coefficient_nrmse": coefficient_by_step[0],
            "one_step_decoded_nrmse": decoded_by_step[0],
            "rollout_coefficient_nrmse": coefficient_by_step[-1],
            "rollout_decoded_nrmse": decoded_by_step[-1],
            "coefficient_nrmse_by_step": coefficient_by_step,
            "decoded_nrmse_by_step": decoded_by_step,
            "final_high_frequency_spectral": high_frequency_spectral_report(
                final_prediction_decoded,
                final_target_decoded,
                resolution=cfg.canonical_query_resolution,
                minimum_radius=3.0,
            ),
            "final_effective_coefficient_rank": effective_rank(final_prediction),
            "maximum_absolute_coefficient_error": float(
                (final_prediction - final_target).abs().max().item()
            ),
            "mean_mode_relative_error": mean_mode_error,
            "parameter_quartiles": parameter_quartile_reports(
                final_prediction,
                final_target,
                dataset.parameters,
                final_target_decoded,
                final_prediction_decoded,
            ),
        },
        predicted_sequence,
    )


def semigroup_consistency(
    model: ResidualCoefficientOperator,
    coefficients: torch.Tensor,
    parameters: torch.Tensor,
    *,
    space: PhysicalFunctionSpace,
    canonical_coords: torch.Tensor,
) -> dict[str, float]:
    one_parameters = parameters.clone()
    one_parameters[:, 3] = 0.04
    half_parameters = parameters.clone()
    half_parameters[:, 3] = 0.02
    one_step = model(coefficients, one_parameters)
    two_steps = model(model(coefficients, half_parameters), half_parameters)
    one_decoded = space.decode(one_step, canonical_coords)
    two_decoded = space.decode(two_steps, canonical_coords)
    return {
        "coefficient_nrmse": global_nrmse(two_steps, one_step),
        "decoded_nrmse": global_nrmse(two_decoded, one_decoded),
    }


def physics_report(
    x_prediction: torch.Tensor,
    y_prediction: torch.Tensor,
    diffusion_prediction: torch.Tensor,
    x_dataset: TrajectorySet,
    y_dataset: TrajectorySet,
) -> dict[str, float]:
    advection_mean_error = max(
        global_nrmse(x_prediction[:, -1, 0], x_dataset.coefficients[:, -1, 0]),
        global_nrmse(y_prediction[:, -1, 0], y_dataset.coefficients[:, -1, 0]),
    )
    advection_l2_drift = 0.0
    for prediction in (x_prediction, y_prediction):
        initial_norm = prediction[:, 0, :49].square().sum(dim=(1, 2)).sqrt()
        final_norm = prediction[:, -1, :49].square().sum(dim=(1, 2)).sqrt()
        drift = ((final_norm - initial_norm).abs() / initial_norm.clamp_min(1e-12)).max()
        advection_l2_drift = max(advection_l2_drift, float(drift.item()))
    diffusion_energy = diffusion_prediction[:, :, :49].square().sum(dim=(2, 3))
    monotonic = diffusion_energy[:, 1:] <= diffusion_energy[:, :-1] + 1e-12
    return {
        "advection_mean_mode_relative_error": advection_mean_error,
        "maximum_advection_l2_norm_drift": advection_l2_drift,
        "diffusion_nonincreasing_energy_fraction": float(monotonic.double().mean().item()),
    }


def cross_observation_report(
    model: ResidualCoefficientOperator,
    dataset: TrajectorySet,
    *,
    space: PhysicalFunctionSpace,
    canonical_coords: torch.Tensor,
    cfg: E11Config,
) -> dict[str, Any]:
    geometry_cfg = GeometryProjectionConfig(
        seed=23,
        validation_states=cfg.validation_trajectories,
        geometry_realizations=cfg.geometry_realizations,
        geometry_seed_start=cfg.geometry_seed_start,
        max_condition_number=100.0,
        max_weighted_design_condition_number=10.0,
    )
    samples = geometry_samples(geometry_cfg)
    initial_coefficients = dataset.coefficients[:, 0]
    predictions: dict[str, torch.Tensor] = {}
    decoded_predictions: dict[str, torch.Tensor] = {}
    encoding_errors: dict[str, list[float]] = {}
    for family, budgets in samples.items():
        family_predictions = []
        family_decoded = []
        encoding_errors[family] = []
        for sample in budgets["high"]:
            values = space.decode(initial_coefficients, sample.coords)
            paths, _ = encode_paths(space, values, sample)
            encoded = paths["exact_gram_projection"]
            encoding_errors[family].append(global_nrmse(encoded, initial_coefficients))
            prediction = model(encoded, dataset.parameters)
            family_predictions.append(prediction)
            family_decoded.append(space.decode(prediction, canonical_coords))
        predictions[family] = torch.stack(family_predictions)
        decoded_predictions[family] = torch.stack(family_decoded)

    family_pairs = {}
    maximum_coefficient_mismatch = 0.0
    maximum_decoded_mismatch = 0.0
    families = tuple(predictions)
    canonical_target = space.decode(dataset.coefficients[:, 1], canonical_coords)
    for left_index, left in enumerate(families):
        for right in families[left_index + 1 :]:
            coefficient_records = []
            decoded_records = []
            for left_realization in range(predictions[left].shape[0]):
                for right_realization in range(predictions[right].shape[0]):
                    coefficient_records.append(
                        _relative_mismatch(
                            predictions[left][left_realization],
                            predictions[right][right_realization],
                        )
                    )
                    decoded_records.append(
                        _decoded_mismatch(
                            decoded_predictions[left][left_realization],
                            decoded_predictions[right][right_realization],
                            canonical_target,
                        )
                    )
            coefficient_maximum = max(coefficient_records)
            decoded_maximum = max(decoded_records)
            maximum_coefficient_mismatch = max(maximum_coefficient_mismatch, coefficient_maximum)
            maximum_decoded_mismatch = max(maximum_decoded_mismatch, decoded_maximum)
            family_pairs[f"{left}__vs__{right}"] = {
                "realization_pairs": len(coefficient_records),
                "coefficient_mean": sum(coefficient_records) / len(coefficient_records),
                "coefficient_maximum": coefficient_maximum,
                "decoded_mean": sum(decoded_records) / len(decoded_records),
                "decoded_maximum": decoded_maximum,
            }
    return {
        "geometry_seed_start": cfg.geometry_seed_start,
        "geometry_realizations": cfg.geometry_realizations,
        "encoding_nrmse": encoding_errors,
        "pairs": family_pairs,
        "maximum_coefficient_mismatch": maximum_coefficient_mismatch,
        "maximum_decoded_mismatch": maximum_decoded_mismatch,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _committed_sha256(path: Path) -> str | None:
    relative_path = path.relative_to(REPO_ROOT).as_posix()
    completed = subprocess.run(
        ["git", "show", f"HEAD:{relative_path}"],
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
    return {
        "source_sha256": source_hashes,
        "committed_source_sha256": committed_hashes,
        "source_files_match_git_head": source_hashes == committed_hashes,
        "git_head": git_head,
        "git_head_present": len(git_head) == 40,
        "worktree_clean": worktree_clean,
    }


def decision(result: dict[str, Any], cfg: E11Config) -> dict[str, Any]:
    reports = result["evaluation"]["composite"]
    candidate = reports["pretrained_fewshot"]
    scratch = reports["scratch_fewshot"]
    full = reports["full_composite_control"]
    zero_shot = reports["elementary_pretrained_zero_shot"]
    persistence = reports["persistence"]
    retention = result["evaluation"]["elementary_retention"]
    extrapolation = result["evaluation"]["temporal_extrapolation"]
    consistency = result["evaluation"]["semigroup_consistency"]
    cross_observation = result["evaluation"]["cross_observation"]
    physics = result["evaluation"]["physics"]
    gates = {
        "closure": result["closure_preflight"]["passed"],
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
        "fewshot_transfer": candidate["rollout_decoded_nrmse"] / scratch["rollout_decoded_nrmse"]
        <= cfg.max_pretrained_to_scratch_ratio,
        "full_data_parity": candidate["rollout_decoded_nrmse"] / full["rollout_decoded_nrmse"]
        <= cfg.max_pretrained_to_full_ratio,
        "zero_shot_composition": (
            zero_shot["rollout_decoded_nrmse"] <= cfg.max_zero_shot_rollout_nrmse
            and zero_shot["rollout_decoded_nrmse"] / persistence["rollout_decoded_nrmse"]
            <= cfg.max_zero_shot_to_persistence_ratio
        ),
        "elementary_retention": (
            retention["post_finetune_macro_decoded_nrmse"] <= cfg.max_elementary_retention_nrmse
            and retention["post_to_pre_ratio"] <= cfg.max_elementary_retention_ratio
        ),
        "temporal_extrapolation": (
            extrapolation["rollout_coefficient_nrmse"] <= cfg.max_extrapolation_nrmse
            and extrapolation["rollout_decoded_nrmse"] <= cfg.max_extrapolation_nrmse
        ),
        "semigroup_consistency": (
            consistency["coefficient_nrmse"] <= cfg.max_semigroup_consistency_nrmse
            and consistency["decoded_nrmse"] <= cfg.max_semigroup_consistency_nrmse
        ),
        "cross_observation": (
            cross_observation["maximum_coefficient_mismatch"] <= cfg.max_cross_observation_mismatch
            and cross_observation["maximum_decoded_mismatch"] <= cfg.max_cross_observation_mismatch
        ),
        "physics": (
            physics["advection_mean_mode_relative_error"] <= cfg.max_mean_mode_relative_error
            and physics["maximum_advection_l2_norm_drift"] <= cfg.max_advection_l2_drift
            and physics["diffusion_nonincreasing_energy_fraction"]
            >= cfg.minimum_diffusion_energy_monotonic_fraction
        ),
        "provenance": all(
            (
                result["provenance"]["source_files_match_git_head"],
                result["provenance"]["git_head_present"],
                result["provenance"]["worktree_clean"],
                result["reproducibility"]["byte_identical_complete_runs"],
            )
        ),
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
    capable_without_transfer_gates = (
        "closure",
        "one_step_accuracy",
        "rollout_accuracy",
        "high_frequency",
        "temporal_extrapolation",
        "semigroup_consistency",
        "cross_observation",
        "physics",
        "provenance",
        "boundary",
    )
    if all(gates.values()):
        classification = "coefficient_operator_transfer_qualified"
        next_move = "preregister the first nonlinear coefficient-dynamics expansion"
    elif all(gates[name] for name in capable_without_transfer_gates):
        classification = "coefficient_operator_capable_without_transfer"
        next_move = "test an explicit additive-generator or Strang-splitting operator"
    else:
        classification = "coefficient_dynamics_not_qualified"
        if not gates["closure"]:
            next_move = "repair or expand the coefficient basis before learned dynamics"
        else:
            next_move = "test an explicit additive-generator or Strang-splitting operator"
    return {
        "classification": classification,
        "gates": gates,
        "ratios": {
            "pretrained_to_scratch_rollout_decoded": candidate["rollout_decoded_nrmse"]
            / scratch["rollout_decoded_nrmse"],
            "pretrained_to_full_rollout_decoded": candidate["rollout_decoded_nrmse"]
            / full["rollout_decoded_nrmse"],
            "zero_shot_to_persistence_rollout_decoded": zero_shot["rollout_decoded_nrmse"]
            / persistence["rollout_decoded_nrmse"],
        },
        "next_move": next_move,
    }


def _run_e11_once(cfg: E11Config, *, run_dir: Path) -> dict[str, Any]:
    if asdict(cfg) != asdict(frozen_e11_config()):
        raise ValueError("E11 requires the exact frozen configuration")
    provenance_report = provenance()
    if not all(
        (
            provenance_report["source_files_match_git_head"],
            provenance_report["git_head_present"],
            provenance_report["worktree_clean"],
        )
    ):
        raise RuntimeError(
            "E11 provenance must match a clean committed Git HEAD before state access"
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
    closure = closure_preflight(cfg, space)
    config_payload = asdict(cfg)
    if not closure["passed"]:
        result = {
            "schema_version": 1,
            "experiment": "canonical_latent_e11_coefficient_operator_transfer",
            "config": config_payload,
            "config_sha256": hashlib.sha256(
                json.dumps(config_payload, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "provenance": provenance_report,
            "closure_preflight": closure,
            "state_reads": {"training": 0, "validation": 0, "heldout": 0},
            "optimizer_updates": 0,
            "causal_decision": {
                "classification": "coefficient_dynamics_not_qualified",
                "gates": {"closure": False},
                "next_move": "repair or expand the coefficient basis before learned dynamics",
            },
        }
        result_path = run_dir / "result.json"
        result_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
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
    elementary_validation_seeds = {
        "x_advection": cfg.x_validation_parameter_seed,
        "y_advection": cfg.y_validation_parameter_seed,
        "diffusion": cfg.diffusion_validation_parameter_seed,
    }
    elementary_validation = {
        regime: build_trajectories(
            f"{regime}_validation",
            count=cfg.validation_trajectories,
            state_seed=cfg.validation_state_seed,
            parameter_seed=parameter_seed,
            regime=regime,
            cfg=cfg,
            space=space,
        )
        for regime, parameter_seed in elementary_validation_seeds.items()
    }

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

    torch.manual_seed(cfg.model_seed)
    initial_model = ResidualCoefficientOperator(cfg)
    initial_state = copy.deepcopy(initial_model.state_dict())
    initial_hash = model_hash(initial_model)

    pretrained = ResidualCoefficientOperator(cfg)
    pretrained.load_state_dict(initial_state)
    pretrain_training = train_elementary(pretrained, pretrain_datasets, pretrain_schedules, cfg)
    pretrained_hash = model_hash(pretrained)

    pretrained_fewshot = copy.deepcopy(pretrained)
    pretrained_fine_training = train_single(
        pretrained_fewshot,
        fewshot,
        fine_schedule,
        learning_rate=cfg.fine_tune_learning_rate,
        weight_decay=cfg.weight_decay,
    )

    scratch_fewshot = ResidualCoefficientOperator(cfg)
    scratch_fewshot.load_state_dict(initial_state)
    scratch_fine_training = train_single(
        scratch_fewshot,
        fewshot,
        fine_schedule,
        learning_rate=cfg.fine_tune_learning_rate,
        weight_decay=cfg.weight_decay,
    )

    full_model = ResidualCoefficientOperator(cfg)
    full_model.load_state_dict(initial_state)
    full_training = train_single(
        full_model,
        full_control,
        full_schedule,
        learning_rate=cfg.pretrain_learning_rate,
        weight_decay=cfg.weight_decay,
    )

    canonical_coords, _ = canonical_grid(space, cfg.canonical_query_resolution)
    evaluation: dict[str, Any] = {"composite": {}}
    composite_models = {
        "elementary_pretrained_zero_shot": pretrained,
        "pretrained_fewshot": pretrained_fewshot,
        "scratch_fewshot": scratch_fewshot,
        "full_composite_control": full_model,
        "persistence": None,
    }
    for name, model in composite_models.items():
        report, _ = rollout(
            model,
            validation,
            space=space,
            canonical_coords=canonical_coords,
            cfg=cfg,
        )
        evaluation["composite"][name] = report

    pre_retention = {}
    post_retention = {}
    post_predictions = {}
    for regime, dataset in elementary_validation.items():
        pre_report, _ = rollout(
            pretrained,
            dataset,
            space=space,
            canonical_coords=canonical_coords,
            cfg=cfg,
        )
        post_report, prediction = rollout(
            pretrained_fewshot,
            dataset,
            space=space,
            canonical_coords=canonical_coords,
            cfg=cfg,
        )
        pre_retention[regime] = pre_report
        post_retention[regime] = post_report
        post_predictions[regime] = prediction
    pre_macro = sum(report["rollout_decoded_nrmse"] for report in pre_retention.values()) / len(
        REGIMES
    )
    post_macro = sum(report["rollout_decoded_nrmse"] for report in post_retention.values()) / len(
        REGIMES
    )
    evaluation["elementary_retention"] = {
        "before_finetune": pre_retention,
        "after_finetune": post_retention,
        "pre_finetune_macro_decoded_nrmse": pre_macro,
        "post_finetune_macro_decoded_nrmse": post_macro,
        "post_to_pre_ratio": post_macro / max(pre_macro, 1e-24),
    }
    evaluation["temporal_extrapolation"], _ = rollout(
        pretrained_fewshot,
        extrapolation,
        space=space,
        canonical_coords=canonical_coords,
        cfg=cfg,
    )
    evaluation["semigroup_consistency"] = semigroup_consistency(
        pretrained_fewshot,
        validation.coefficients[:, 0],
        validation.parameters,
        space=space,
        canonical_coords=canonical_coords,
    )
    evaluation["cross_observation"] = cross_observation_report(
        pretrained_fewshot,
        validation,
        space=space,
        canonical_coords=canonical_coords,
        cfg=cfg,
    )
    evaluation["physics"] = physics_report(
        post_predictions["x_advection"],
        post_predictions["y_advection"],
        post_predictions["diffusion"],
        elementary_validation["x_advection"],
        elementary_validation["y_advection"],
    )

    parameter_count = sum(parameter.numel() for parameter in initial_model.parameters())
    result = {
        "schema_version": 1,
        "experiment": "canonical_latent_e11_coefficient_operator_transfer",
        "config": config_payload,
        "config_sha256": hashlib.sha256(
            json.dumps(config_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "provenance": provenance_report,
        "closure_preflight": closure,
        "architecture": {
            "kind": "residual_coefficient_mlp",
            "basis_dimension": 52,
            "active_periodic_modes": 49,
            "hidden_width": cfg.hidden_width,
            "parameter_count": parameter_count,
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
        "schedules": {
            "pretrain_sha256": {
                regime: tensor_hash(indices) for regime, indices in pretrain_schedules.items()
            },
            "fine_tune_sha256": tensor_hash(fine_schedule),
            "full_control_sha256": tensor_hash(full_schedule),
            "fine_tune_schedule_shared_exactly": True,
        },
        "training": {
            "elementary_pretrained": pretrain_training,
            "pretrained_fewshot": pretrained_fine_training,
            "scratch_fewshot": scratch_fine_training,
            "full_composite_control": full_training,
        },
        "checkpoints": {
            "initial_sha256": initial_hash,
            "elementary_pretrained_sha256": pretrained_hash,
            "pretrained_fewshot_sha256": model_hash(pretrained_fewshot),
            "scratch_fewshot_sha256": model_hash(scratch_fewshot),
            "full_composite_control_sha256": model_hash(full_model),
        },
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
    result_path = run_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result["result_path"] = str(result_path)
    return result


def run_e11(cfg: E11Config, *, run_dir: Path) -> dict[str, Any]:
    if asdict(cfg) != asdict(frozen_e11_config()):
        raise ValueError("E11 requires the exact frozen configuration")

    first = _run_e11_once(cfg, run_dir=run_dir / "replicate_a")
    second = _run_e11_once(cfg, run_dir=run_dir / "replicate_b")
    first_path = Path(first["result_path"])
    second_path = Path(second["result_path"])
    first_bytes = first_path.read_bytes()
    second_bytes = second_path.read_bytes()
    byte_identical = first_bytes == second_bytes

    result = json.loads(first_bytes)
    result["reproducibility"] = {
        "deterministic_algorithms": True,
        "replicate_a_sha256": hashlib.sha256(first_bytes).hexdigest(),
        "replicate_b_sha256": hashlib.sha256(second_bytes).hexdigest(),
        "byte_identical_complete_runs": byte_identical,
    }
    if result["closure_preflight"]["passed"]:
        result["causal_decision"] = decision(result, cfg)
    else:
        result["causal_decision"] = {
            "classification": "coefficient_dynamics_not_qualified",
            "gates": {
                "closure": False,
                "provenance": byte_identical,
            },
            "next_move": "repair or expand the coefficient basis before learned dynamics",
        }
    result_path = run_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result["result_path"] = str(result_path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen E11 coefficient-operator transfer gate"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--print-json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_e11(frozen_e11_config(), run_dir=args.run_dir)
    summary = {
        "causal_decision": result["causal_decision"],
        "result_path": result["result_path"],
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
