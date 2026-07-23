#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

if __package__:
    from scripts.run_canonical_latent_e2_benchmark import (
        Representation,
        evaluate_field,
        representation_points,
        state_coefficients,
    )
    from scripts.run_canonical_latent_e4_capacity_ladder import (
        high_frequency_spectral_report,
    )
    from scripts.run_canonical_latent_e7_function_space import (
        FunctionSpaceConfig,
        PhysicalFunctionSpace,
        _decoded_mismatch,
        _relative_mismatch,
    )
else:
    from run_canonical_latent_e2_benchmark import (  # type: ignore[no-redef]
        Representation,
        evaluate_field,
        representation_points,
        state_coefficients,
    )
    from run_canonical_latent_e4_capacity_ladder import (  # type: ignore[no-redef]
        high_frequency_spectral_report,
    )
    from run_canonical_latent_e7_function_space import (  # type: ignore[no-redef]
        FunctionSpaceConfig,
        PhysicalFunctionSpace,
        _decoded_mismatch,
        _relative_mismatch,
    )
from ups.eval.latent_qualification import effective_rank, global_nrmse


@dataclass(frozen=True)
class GeometryProjectionConfig:
    seed: int = 17
    validation_states: int = 24
    calibration_resolution: int = 128
    canonical_query_resolution: int = 18
    grid_low_resolution: int = 10
    grid_high_resolution: int = 18
    particle_low_count: int = 256
    particle_high_count: int = 576
    geometry_realizations: int = 8
    geometry_seed_start: int = 20_000
    warp_limit: float = 0.28
    warp_l1_limit: float = 0.42
    max_condition_number: float = 10.0
    max_weighted_design_condition_number: float = 10.0
    rank_tolerance: float = 1e-10
    max_coefficient_nrmse: float = 0.10
    max_decoded_nrmse: float = 0.10
    high_frequency_radius: float = 3.0
    max_high_frequency_nrmse: float = 0.25
    max_cross_family_coefficient_mismatch: float = 0.10
    max_cross_family_decoded_mismatch: float = 0.10
    max_particle_convergence_ratio: float = 0.90
    max_structured_convergence_ratio: float = 1.10
    max_realization_dispersion: float = 0.10
    invariance_atol: float = 1e-10
    max_ablation_ratio: float = 0.50

    def __post_init__(self) -> None:
        if self.validation_states < 4:
            raise ValueError("E9 requires at least four validation states")
        if self.calibration_resolution < self.canonical_query_resolution:
            raise ValueError("calibration resolution must cover canonical queries")
        if self.grid_low_resolution < 8:
            raise ValueError("E9 grid resolution is too small for the frozen basis")
        if self.particle_low_count <= 52 or self.particle_high_count <= self.particle_low_count:
            raise ValueError("E9 particle counts must identify and refine the 52-mode basis")
        if self.geometry_realizations < 1:
            raise ValueError("E9 requires at least one geometry realization")

    def function_space_config(self) -> FunctionSpaceConfig:
        return FunctionSpaceConfig(
            seed=self.seed,
            validation_states=self.validation_states,
            train_low_resolution=self.grid_low_resolution,
            train_high_resolution=14,
            validation_resolution=self.grid_high_resolution,
            canonical_query_resolution=self.canonical_query_resolution,
            calibration_resolution=self.calibration_resolution,
            max_condition_number=self.max_condition_number,
            rank_tolerance=self.rank_tolerance,
        )


@dataclass(frozen=True)
class GeometrySample:
    family: str
    budget: str
    realization: int
    seed: int
    coords: torch.Tensor
    measure: torch.Tensor
    warp_a: float = 0.0
    warp_b: float = 0.0


def _warp_parameters(cfg: GeometryProjectionConfig, seed: int) -> tuple[float, float]:
    generator = torch.Generator().manual_seed(seed + 911)
    draws = torch.rand(2, generator=generator, dtype=torch.float64)
    a = float((2.0 * draws[0] - 1.0) * cfg.warp_limit)
    b = float((2.0 * draws[1] - 1.0) * cfg.warp_limit)
    total = abs(a) + abs(b)
    if total > cfg.warp_l1_limit:
        scale = cfg.warp_l1_limit / total
        a *= scale
        b *= scale
    return a, b


def _warp_coordinates_and_measure(
    uv: torch.Tensor, *, warp_a: float, warp_b: float
) -> tuple[torch.Tensor, torch.Tensor]:
    u = uv[..., 0]
    v = uv[..., 1]
    sin_u = torch.sin(2.0 * math.pi * u)
    cos_u = torch.cos(2.0 * math.pi * u)
    sin_v = torch.sin(2.0 * math.pi * v)
    cos_v = torch.cos(2.0 * math.pi * v)
    scale = 1.0 / (2.0 * math.pi)
    x = u + warp_a * scale * sin_u * sin_v
    y = v + warp_b * scale * sin_u * sin_v
    jacobian = 1.0 + warp_a * cos_u * sin_v + warp_b * sin_u * cos_v
    if torch.any(jacobian <= 0):
        raise ValueError("E9 warp produced a non-positive Jacobian")
    coords = torch.stack((x, y), dim=-1)
    measure = jacobian.unsqueeze(-1)
    measure = measure / measure.sum()
    return coords, measure


def _midpoint_parameter_grid(resolution: int) -> torch.Tensor:
    axis = (torch.arange(resolution, dtype=torch.float64) + 0.5) / resolution
    u, v = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((u, v), dim=-1).reshape(-1, 2)


def geometry_samples(cfg: GeometryProjectionConfig) -> dict[str, dict[str, list[GeometrySample]]]:
    samples: dict[str, dict[str, list[GeometrySample]]] = {
        family: {"low": [], "high": []}
        for family in ("grid", "warped_mesh", "uniform_particles", "warped_particles")
    }
    for budget, resolution in (
        ("low", cfg.grid_low_resolution),
        ("high", cfg.grid_high_resolution),
    ):
        uv = _midpoint_parameter_grid(resolution)
        measure = torch.full((uv.shape[0], 1), 1.0 / uv.shape[0], dtype=torch.float64)
        samples["grid"][budget].append(
            GeometrySample("grid", budget, 0, cfg.geometry_seed_start, uv, measure)
        )

    for realization in range(cfg.geometry_realizations):
        seed = cfg.geometry_seed_start + realization
        warp_a, warp_b = _warp_parameters(cfg, seed)
        high_uv = torch.rand(
            cfg.particle_high_count,
            2,
            generator=torch.Generator().manual_seed(seed),
            dtype=torch.float64,
        )
        for budget, resolution, point_count in (
            ("low", cfg.grid_low_resolution, cfg.particle_low_count),
            ("high", cfg.grid_high_resolution, cfg.particle_high_count),
        ):
            mesh_uv = _midpoint_parameter_grid(resolution)
            mesh_coords, mesh_measure = _warp_coordinates_and_measure(
                mesh_uv, warp_a=warp_a, warp_b=warp_b
            )
            samples["warped_mesh"][budget].append(
                GeometrySample(
                    "warped_mesh",
                    budget,
                    realization,
                    seed,
                    mesh_coords,
                    mesh_measure,
                    warp_a,
                    warp_b,
                )
            )

            particle_uv = high_uv[:point_count]
            uniform_measure = torch.full((point_count, 1), 1.0 / point_count, dtype=torch.float64)
            samples["uniform_particles"][budget].append(
                GeometrySample(
                    "uniform_particles",
                    budget,
                    realization,
                    seed,
                    particle_uv,
                    uniform_measure,
                )
            )
            warped_coords, warped_measure = _warp_coordinates_and_measure(
                particle_uv, warp_a=warp_a, warp_b=warp_b
            )
            samples["warped_particles"][budget].append(
                GeometrySample(
                    "warped_particles",
                    budget,
                    realization,
                    seed,
                    warped_coords,
                    warped_measure,
                    warp_a,
                    warp_b,
                )
            )
    return samples


def _design(
    space: PhysicalFunctionSpace,
    sample: GeometrySample,
    *,
    enforce_condition: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | int]]:
    basis = space.basis(sample.coords)
    measure = sample.measure.double()
    if (
        torch.any(measure <= 0)
        or not torch.isfinite(measure).all()
        or not torch.isclose(measure.sum(), torch.tensor(1.0, dtype=torch.float64))
    ):
        raise ValueError("E9 requires positive finite normalized quadrature masses")
    weighted_design = basis * measure.sqrt()
    singular_values = torch.linalg.svdvals(weighted_design)
    rank = int((singular_values > space.cfg.rank_tolerance).sum().item())
    weighted_design_condition_number = float((singular_values[0] / singular_values[-1]).item())
    if rank != space.cfg.basis_dim:
        raise ValueError(f"E9 geometry is rank deficient: {rank} != {space.cfg.basis_dim}")
    gram = basis.transpose(0, 1) @ (measure * basis)
    gram_condition_number = float(torch.linalg.cond(gram).item())
    if enforce_condition and gram_condition_number > space.cfg.max_condition_number:
        raise ValueError(
            f"Gram condition number {gram_condition_number} exceeds "
            f"{space.cfg.max_condition_number}"
        )
    return (
        basis,
        gram,
        {
            "rank": rank,
            "mass_admissible": True,
            "mass_sum": float(measure.sum().item()),
            "condition_number": gram_condition_number,
            "gram_condition_number": gram_condition_number,
            "weighted_design_condition_number": weighted_design_condition_number,
            "minimum_singular_value": float(singular_values[-1].item()),
            "maximum_singular_value": float(singular_values[0].item()),
        },
    )


def encode_paths(
    space: PhysicalFunctionSpace,
    values: torch.Tensor,
    sample: GeometrySample,
) -> tuple[dict[str, torch.Tensor], dict[str, float | int]]:
    if values.dim() != 3 or values.shape[1] != sample.coords.shape[0]:
        raise ValueError("values must match the E9 geometry node count")
    basis, gram, design = _design(space, sample)
    right_hand_side = basis.transpose(0, 1).unsqueeze(0) @ (
        sample.measure.double().unsqueeze(0) * values.double()
    )
    exact = torch.linalg.solve(gram.unsqueeze(0), right_hand_side)
    diagonal = gram.diagonal().view(1, -1, 1)
    return {
        "exact_gram_projection": exact,
        "moment_only": right_hand_side,
        "diagonal_gram": right_hand_side / diagonal,
    }, design


def _tensor_nrmse(prediction: torch.Tensor, target: torch.Tensor) -> float:
    numerator = (prediction.double() - target.double()).square().sum()
    denominator = target.double().square().sum().clamp_min(1e-24)
    return float(torch.sqrt(numerator / denominator).item())


def _realization_dispersion(coefficients: torch.Tensor, canonical_target: torch.Tensor) -> float:
    if coefficients.shape[0] == 1:
        return 0.0
    centered = coefficients - coefficients.mean(dim=0, keepdim=True)
    numerator = centered.square().sum()
    denominator = (canonical_target.double().square().sum() * coefficients.shape[0]).clamp_min(
        1e-24
    )
    return float(torch.sqrt(numerator / denominator).item())


def _source_order_error(
    space: PhysicalFunctionSpace,
    state_parameters: torch.Tensor,
    sample: GeometrySample,
    canonical_query: torch.Tensor,
    *,
    seed: int,
) -> tuple[float, float, list[float], list[float]]:
    values = evaluate_field(state_parameters, sample.coords).double()
    original, _ = encode_paths(space, values, sample)
    permutation = torch.randperm(
        sample.coords.shape[0], generator=torch.Generator().manual_seed(seed)
    )
    permuted_sample = GeometrySample(
        sample.family,
        sample.budget,
        sample.realization,
        sample.seed,
        sample.coords[permutation],
        sample.measure[permutation],
        sample.warp_a,
        sample.warp_b,
    )
    permuted_values = values[:, permutation]
    permuted, _ = encode_paths(space, permuted_values, permuted_sample)
    original_coefficients = original["exact_gram_projection"]
    permuted_coefficients = permuted["exact_gram_projection"]
    original_decoded = space.decode(original_coefficients, canonical_query)
    permuted_decoded = space.decode(permuted_coefficients, canonical_query)
    coefficient_errors = (
        (original_coefficients - permuted_coefficients).abs().flatten(start_dim=1).amax(dim=1)
    )
    decoded_errors = (original_decoded - permuted_decoded).abs().flatten(start_dim=1).amax(dim=1)
    return (
        float(coefficient_errors.max().item()),
        float(decoded_errors.max().item()),
        [float(value) for value in coefficient_errors.tolist()],
        [float(value) for value in decoded_errors.tolist()],
    )


def _evaluate_budget(
    space: PhysicalFunctionSpace,
    state_parameters: torch.Tensor,
    samples: list[GeometrySample],
    canonical_coefficients: torch.Tensor,
    canonical_query: torch.Tensor,
    canonical_target: torch.Tensor,
    cfg: GeometryProjectionConfig,
) -> tuple[dict[str, dict[str, Any]], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    all_coefficients: dict[str, list[torch.Tensor]] = {
        "exact_gram_projection": [],
        "moment_only": [],
        "diagonal_gram": [],
    }
    all_decoded: dict[str, list[torch.Tensor]] = {name: [] for name in all_coefficients}
    realization_records = []
    maximum_coefficient_order_error = 0.0
    maximum_decoded_order_error = 0.0
    for index, sample in enumerate(samples):
        values = evaluate_field(state_parameters, sample.coords).double()
        encoded, design = encode_paths(space, values, sample)
        record: dict[str, Any] = {
            "realization": sample.realization,
            "seed": sample.seed,
            "source_node_count": sample.coords.shape[0],
            "compression_ratio": sample.coords.shape[0] / space.cfg.basis_dim,
            "warp_a": sample.warp_a,
            "warp_b": sample.warp_b,
            "design": design,
            "paths": {},
        }
        for name, coefficients in encoded.items():
            decoded = space.decode(coefficients, canonical_query)
            all_coefficients[name].append(coefficients)
            all_decoded[name].append(decoded)
            record["paths"][name] = {
                "coefficient_nrmse_to_canonical": _tensor_nrmse(
                    coefficients, canonical_coefficients
                ),
                "canonical_query_nrmse": global_nrmse(decoded, canonical_target),
                "states": [
                    {
                        "state_index": state_index,
                        "coefficient_nrmse_to_canonical": _tensor_nrmse(
                            coefficients[state_index : state_index + 1],
                            canonical_coefficients[state_index : state_index + 1],
                        ),
                        "canonical_query_nrmse": global_nrmse(
                            decoded[state_index : state_index + 1],
                            canonical_target[state_index : state_index + 1],
                        ),
                        "high_frequency_spectral": high_frequency_spectral_report(
                            decoded[state_index : state_index + 1],
                            canonical_target[state_index : state_index + 1],
                            resolution=cfg.canonical_query_resolution,
                            minimum_radius=cfg.high_frequency_radius,
                        ),
                        "design_rank": design["rank"],
                    }
                    for state_index in range(state_parameters.shape[0])
                ],
            }
        (
            coefficient_order_error,
            decoded_order_error,
            state_coefficient_order_errors,
            state_decoded_order_errors,
        ) = _source_order_error(
            space,
            state_parameters,
            sample,
            canonical_query,
            seed=cfg.seed + 3100 + index,
        )
        record["source_order_coefficient_max_abs_error"] = coefficient_order_error
        record["source_order_decoded_max_abs_error"] = decoded_order_error
        for state_index, state_record in enumerate(
            record["paths"]["exact_gram_projection"]["states"]
        ):
            state_record["source_order_coefficient_max_abs_error"] = state_coefficient_order_errors[
                state_index
            ]
            state_record["source_order_decoded_max_abs_error"] = state_decoded_order_errors[
                state_index
            ]
        maximum_coefficient_order_error = max(
            maximum_coefficient_order_error, coefficient_order_error
        )
        maximum_decoded_order_error = max(maximum_decoded_order_error, decoded_order_error)
        realization_records.append(record)

    reports = {}
    stacked_coefficients = {}
    stacked_decoded = {}
    repeated_coefficients = canonical_coefficients.repeat(len(samples), 1, 1)
    repeated_target = canonical_target.repeat(len(samples), 1, 1)
    for name in all_coefficients:
        coefficients = torch.cat(all_coefficients[name], dim=0)
        decoded = torch.cat(all_decoded[name], dim=0)
        stacked_coefficients[name] = torch.stack(all_coefficients[name], dim=0)
        stacked_decoded[name] = torch.stack(all_decoded[name], dim=0)
        reports[name] = {
            "coefficient_nrmse_to_canonical": _tensor_nrmse(coefficients, repeated_coefficients),
            "canonical_query_nrmse": global_nrmse(decoded, repeated_target),
            "high_frequency_spectral": high_frequency_spectral_report(
                decoded,
                repeated_target,
                resolution=cfg.canonical_query_resolution,
                minimum_radius=cfg.high_frequency_radius,
            ),
            "effective_coefficient_rank": effective_rank(coefficients),
            "realization_coefficient_dispersion": _realization_dispersion(
                stacked_coefficients[name], canonical_coefficients
            ),
        }
    reports["exact_gram_projection"][
        "source_order_coefficient_max_abs_error"
    ] = maximum_coefficient_order_error
    reports["exact_gram_projection"][
        "source_order_decoded_max_abs_error"
    ] = maximum_decoded_order_error
    reports["exact_gram_projection"]["realizations"] = realization_records
    return reports, stacked_coefficients, stacked_decoded


def _decision(result: dict[str, Any], cfg: GeometryProjectionConfig) -> dict[str, Any]:
    candidate = result["evaluation"]["paths"]["exact_gram_projection"]["families"]
    designs = [
        realization["design"]
        for family in candidate.values()
        for budget in family.values()
        for realization in budget["realizations"]
    ]
    family_high = {family: budgets["high"] for family, budgets in candidate.items()}
    semantics = result["evaluation"]["exact_semantics"]
    convergence = result["evaluation"]["exact_convergence"]
    causal = result["evaluation"]["causal_ablation"]
    gates = {
        "design": all(
            design["rank"] == 52
            and design["mass_admissible"]
            and design["weighted_design_condition_number"]
            <= cfg.max_weighted_design_condition_number
            and design["gram_condition_number"] <= cfg.max_condition_number
            for design in designs
        ),
        "coefficient_accuracy": all(
            report["coefficient_nrmse_to_canonical"] <= cfg.max_coefficient_nrmse
            for report in family_high.values()
        ),
        "decoded_accuracy": all(
            report["canonical_query_nrmse"] <= cfg.max_decoded_nrmse
            for report in family_high.values()
        ),
        "high_frequency": all(
            report["high_frequency_spectral"]["nrmse"] <= cfg.max_high_frequency_nrmse
            for report in family_high.values()
        ),
        "cross_family_semantics": all(
            pair["coefficient_relative_mismatch"] <= cfg.max_cross_family_coefficient_mismatch
            and pair["decoded_mismatch_nrmse"] <= cfg.max_cross_family_decoded_mismatch
            for pair in semantics.values()
        ),
        "convergence": (
            convergence["uniform_particles"] <= cfg.max_particle_convergence_ratio
            and convergence["warped_particles"] <= cfg.max_particle_convergence_ratio
            and convergence["grid"] <= cfg.max_structured_convergence_ratio
            and convergence["warped_mesh"] <= cfg.max_structured_convergence_ratio
        ),
        "realization_stability": all(
            report["realization_coefficient_dispersion"] <= cfg.max_realization_dispersion
            for report in family_high.values()
        ),
        "source_order_invariance": all(
            report["source_order_coefficient_max_abs_error"] <= cfg.invariance_atol
            and report["source_order_decoded_max_abs_error"] <= cfg.invariance_atol
            for report in family_high.values()
        ),
        "full_gram_causal_advantage": (
            causal["exact_to_moment_only_ratio"] <= cfg.max_ablation_ratio
            and causal["exact_to_diagonal_gram_ratio"] <= cfg.max_ablation_ratio
        ),
        "boundary": all(
            (
                result["boundary"]["learned_parameters"] == 0,
                result["boundary"]["optimizer_updates"] == 0,
                not result["boundary"]["operator_instantiated"],
                result["boundary"]["heldout_reads"] == 0,
                result["boundary"]["routing_paths"] == 0,
                not result["boundary"]["original_source_features_available_after_projection"],
            )
        ),
    }
    if all(gates.values()):
        classification = "geometry_universal_projection_qualified"
        next_move = "preregister the first coefficient-space latent operator gate"
    elif gates["design"]:
        classification = "projection_identifiable_but_sampling_unstable"
        next_move = "repair only the basis, quadrature process, or sampling budget"
    else:
        classification = "geometry_universal_projection_not_qualified"
        next_move = "reconsider the basis or regularized projection before dynamics"
    return {
        "classification": classification,
        "gates": gates,
        "next_move": next_move,
    }


def _worst_realization_pair(
    left_coefficients: torch.Tensor,
    right_coefficients: torch.Tensor,
    left_decoded: torch.Tensor,
    right_decoded: torch.Tensor,
    canonical_target: torch.Tensor,
) -> dict[str, Any]:
    coefficient_records = []
    decoded_records = []
    for left_index in range(left_coefficients.shape[0]):
        for right_index in range(right_coefficients.shape[0]):
            coefficient_records.append(
                {
                    "left_realization": left_index,
                    "right_realization": right_index,
                    "value": _relative_mismatch(
                        left_coefficients[left_index],
                        right_coefficients[right_index],
                    ),
                }
            )
            decoded_records.append(
                {
                    "left_realization": left_index,
                    "right_realization": right_index,
                    "value": _decoded_mismatch(
                        left_decoded[left_index],
                        right_decoded[right_index],
                        canonical_target,
                    ),
                }
            )
    worst_coefficient = max(coefficient_records, key=lambda record: record["value"])
    worst_decoded = max(decoded_records, key=lambda record: record["value"])
    return {
        "coefficient_relative_mismatch": worst_coefficient["value"],
        "coefficient_mean_relative_mismatch": sum(record["value"] for record in coefficient_records)
        / len(coefficient_records),
        "coefficient_worst_pair": {
            key: value for key, value in worst_coefficient.items() if key != "value"
        },
        "decoded_mismatch_nrmse": worst_decoded["value"],
        "decoded_mean_mismatch_nrmse": sum(record["value"] for record in decoded_records)
        / len(decoded_records),
        "decoded_worst_pair": {
            key: value for key, value in worst_decoded.items() if key != "value"
        },
        "realization_pairs_evaluated": len(coefficient_records),
    }


def run_geometry_projection(cfg: GeometryProjectionConfig, *, run_dir: Path) -> dict[str, Any]:
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(1)
    run_dir.mkdir(parents=True, exist_ok=True)
    space = PhysicalFunctionSpace(cfg.function_space_config())
    samples = geometry_samples(cfg)
    geometry_preflight = {
        family: {
            budget: [
                {
                    "realization": sample.realization,
                    "seed": sample.seed,
                    "source_node_count": sample.coords.shape[0],
                    "warp_a": sample.warp_a,
                    "warp_b": sample.warp_b,
                    "design": _design(space, sample, enforce_condition=False)[2],
                }
                for sample in family_samples
            ]
            for budget, family_samples in budgets.items()
        }
        for family, budgets in samples.items()
    }
    preflight_pass = all(
        record["design"]["rank"] == space.cfg.basis_dim
        and record["design"]["mass_admissible"]
        and record["design"]["weighted_design_condition_number"]
        <= cfg.max_weighted_design_condition_number
        and record["design"]["gram_condition_number"] <= cfg.max_condition_number
        for family in geometry_preflight.values()
        for budget in family.values()
        for record in budget
    )
    config_payload = asdict(cfg)
    if not preflight_pass:
        result: dict[str, Any] = {
            "schema_version": 1,
            "experiment": "canonical_latent_e9_geometry_universal_projection",
            "config": config_payload,
            "config_sha256": hashlib.sha256(
                json.dumps(config_payload, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "architecture": {
                "kind": "exact_quadrature_gram_projection",
                "basis_dimension": space.cfg.basis_dim,
                "learned_parameters": 0,
                "geometry_factorization_cacheable": True,
                "representation_labels_used": False,
            },
            "state_split": {
                "validation_count_configured": cfg.validation_states,
                "validation_states_read": 0,
                "training_states_read": 0,
                "heldout_states_read": 0,
            },
            "geometry_preflight": geometry_preflight,
            "evaluation": {
                "status": "skipped_before_state_read",
                "reason": "frozen Gram condition-number gate failed",
            },
            "boundary": {
                "learned_parameters": 0,
                "optimizer_updates": 0,
                "operator_instantiated": False,
                "temporal_transitions": 0,
                "training_state_reads": 0,
                "validation_state_reads": 0,
                "heldout_reads": 0,
                "representation_label_model_inputs": False,
                "task_label_model_inputs": False,
                "provider_calls": 0,
                "routing_paths": 0,
                "original_source_features_available_after_projection": False,
                "particle_dynamics_qualified": False,
                "arbitrary_particle_distributions_qualified": False,
            },
            "causal_decision": {
                "classification": "geometry_universal_projection_not_qualified",
                "gates": {
                    "design": False,
                    "state_evaluation_skipped": True,
                },
                "next_move": (
                    "preregister an E7-equivalent weighted-design condition gate "
                    "on fresh geometries and states"
                ),
            },
        }
        result_path = run_dir / "result.json"
        result_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        result["result_path"] = str(result_path)
        return result

    validation_parameters = state_coefficients(cfg.validation_states, seed=cfg.seed + 10_000)
    canonical_representation = Representation(
        "canonical_queries", cfg.canonical_query_resolution, 0.0, 0.0
    )
    canonical_query, _ = representation_points(canonical_representation, batch=1)
    canonical_query = canonical_query.double().expand(cfg.validation_states, -1, -1)
    canonical_target = evaluate_field(validation_parameters, canonical_query).double()
    calibration_uv = _midpoint_parameter_grid(cfg.calibration_resolution)
    calibration_measure = torch.full(
        (1, calibration_uv.shape[0], 1),
        1.0 / calibration_uv.shape[0],
        dtype=torch.float64,
    )
    calibration_coords = calibration_uv.unsqueeze(0).expand(cfg.validation_states, -1, -1)
    calibration_values = evaluate_field(validation_parameters, calibration_coords).double()
    canonical_coefficients, canonical_design = space.project(
        calibration_values,
        calibration_coords,
        calibration_measure.expand(cfg.validation_states, -1, -1),
    )

    evaluation_paths: dict[str, Any] = {
        name: {"families": {}} for name in ("exact_gram_projection", "moment_only", "diagonal_gram")
    }
    exact_coefficients: dict[str, dict[str, torch.Tensor]] = {}
    exact_decoded: dict[str, dict[str, torch.Tensor]] = {}
    for family, budgets in samples.items():
        exact_coefficients[family] = {}
        exact_decoded[family] = {}
        for budget, family_samples in budgets.items():
            reports, coefficients, decoded = _evaluate_budget(
                space,
                validation_parameters,
                family_samples,
                canonical_coefficients,
                canonical_query,
                canonical_target,
                cfg,
            )
            for path, report in reports.items():
                evaluation_paths[path]["families"].setdefault(family, {})[budget] = report
            exact_coefficients[family][budget] = coefficients["exact_gram_projection"]
            exact_decoded[family][budget] = decoded["exact_gram_projection"]

    exact_semantics = {}
    families = tuple(samples)
    for left_index, left in enumerate(families):
        for right in families[left_index + 1 :]:
            exact_semantics[f"{left}__vs__{right}"] = _worst_realization_pair(
                exact_coefficients[left]["high"],
                exact_coefficients[right]["high"],
                exact_decoded[left]["high"],
                exact_decoded[right]["high"],
                canonical_target,
            )
    exact_convergence = {
        family: evaluation_paths["exact_gram_projection"]["families"][family]["high"][
            "coefficient_nrmse_to_canonical"
        ]
        / evaluation_paths["exact_gram_projection"]["families"][family]["low"][
            "coefficient_nrmse_to_canonical"
        ]
        for family in families
    }
    macro_errors = {
        path: sum(
            evaluation_paths[path]["families"][family]["high"]["coefficient_nrmse_to_canonical"]
            for family in families
        )
        / len(families)
        for path in evaluation_paths
    }
    causal_ablation = {
        "high_budget_macro_coefficient_nrmse": macro_errors,
        "exact_to_moment_only_ratio": (
            macro_errors["exact_gram_projection"] / macro_errors["moment_only"]
        ),
        "exact_to_diagonal_gram_ratio": (
            macro_errors["exact_gram_projection"] / macro_errors["diagonal_gram"]
        ),
    }

    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "canonical_latent_e9_geometry_universal_projection",
        "config": config_payload,
        "config_sha256": hashlib.sha256(
            json.dumps(config_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "architecture": {
            "kind": "exact_quadrature_gram_projection",
            "basis_dimension": space.cfg.basis_dim,
            "learned_parameters": 0,
            "geometry_factorization_cacheable": True,
            "representation_labels_used": False,
        },
        "state_split": {
            "validation_count": cfg.validation_states,
            "validation_seed": cfg.seed + 10_000,
            "training_states_read": 0,
            "heldout_states_read": 0,
        },
        "canonical_target": {
            "calibration_resolution": cfg.calibration_resolution,
            "design": canonical_design,
        },
        "geometry_families": {
            family: {
                budget: [
                    {
                        "realization": sample.realization,
                        "seed": sample.seed,
                        "source_node_count": sample.coords.shape[0],
                        "warp_a": sample.warp_a,
                        "warp_b": sample.warp_b,
                    }
                    for sample in family_samples
                ]
                for budget, family_samples in budgets.items()
            }
            for family, budgets in samples.items()
        },
        "evaluation": {
            "paths": evaluation_paths,
            "exact_semantics": exact_semantics,
            "exact_convergence": exact_convergence,
            "causal_ablation": causal_ablation,
        },
        "boundary": {
            "learned_parameters": 0,
            "optimizer_updates": 0,
            "operator_instantiated": False,
            "temporal_transitions": 0,
            "training_state_reads": 0,
            "heldout_reads": 0,
            "representation_label_model_inputs": False,
            "task_label_model_inputs": False,
            "provider_calls": 0,
            "routing_paths": 0,
            "original_source_features_available_after_projection": False,
            "particle_dynamics_qualified": False,
            "arbitrary_particle_distributions_qualified": False,
        },
    }
    result["causal_decision"] = _decision(result, cfg)
    result_path = run_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result["result_path"] = str(result_path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the E9 zero-shot geometry-universal E7 projection test"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--validation-states", type=int, default=GeometryProjectionConfig.validation_states
    )
    parser.add_argument(
        "--geometry-realizations",
        type=int,
        default=GeometryProjectionConfig.geometry_realizations,
    )
    parser.add_argument("--print-json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = GeometryProjectionConfig(
        validation_states=args.validation_states,
        geometry_realizations=args.geometry_realizations,
    )
    result = run_geometry_projection(cfg, run_dir=args.run_dir)
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
