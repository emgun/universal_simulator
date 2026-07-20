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
        BenchmarkConfig,
        Representation,
        _representations,
        evaluate_field,
        inverse_distance_interpolate,
        representation_points,
        state_coefficients,
    )
    from scripts.run_canonical_latent_e4_capacity_ladder import (
        high_frequency_spectral_report,
    )
else:
    from run_canonical_latent_e2_benchmark import (  # type: ignore[no-redef]
        BenchmarkConfig,
        Representation,
        _representations,
        evaluate_field,
        inverse_distance_interpolate,
        representation_points,
        state_coefficients,
    )
    from run_canonical_latent_e4_capacity_ladder import (  # type: ignore[no-redef]
        high_frequency_spectral_report,
    )
from ups.eval.latent_qualification import (
    discretization_mismatch_report,
    effective_rank,
    global_nrmse,
    paired_latent_report,
)


@dataclass(frozen=True)
class FunctionSpaceConfig:
    seed: int = 17
    validation_states: int = 24
    train_low_resolution: int = 10
    train_high_resolution: int = 14
    validation_resolution: int = 18
    canonical_query_resolution: int = 18
    fourier_cutoff: int = 3
    trend_modes: tuple[str, ...] = ("x", "y", "xy")
    calibration_resolution: int = 128
    max_calibration_component_nrmse: float = 0.02
    max_condition_number: float = 10.0
    rank_tolerance: float = 1e-10
    max_interpolation_baseline_ratio: float = 2.0
    high_frequency_radius: float = 3.0
    max_high_frequency_nrmse: float = 0.25
    max_paired_coefficient_mismatch: float = 0.10
    max_paired_decoded_mismatch: float = 0.10
    max_refinement_coefficient_mismatch: float = 0.15
    invariance_atol: float = 1e-10
    minimum_high_resolution_compression: float = 2.0

    def __post_init__(self) -> None:
        if self.validation_states < 4:
            raise ValueError("E7 requires at least four paired validation states")
        if self.fourier_cutoff != 3 or self.trend_modes != ("x", "y", "xy"):
            raise ValueError("E7 freezes cutoff three plus x, y, and xy trend modes")
        if self.basis_dim != 52:
            raise ValueError("E7 freezes a 52-coefficient physical function space")

    @property
    def basis_dim(self) -> int:
        one_dimensional = 1 + 2 * self.fourier_cutoff
        return one_dimensional * one_dimensional + len(self.trend_modes)

    def benchmark_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            seed=self.seed,
            train_states=128,
            validation_states=self.validation_states,
            train_low_resolution=self.train_low_resolution,
            train_high_resolution=self.train_high_resolution,
            validation_resolution=self.validation_resolution,
            canonical_query_resolution=self.canonical_query_resolution,
        )


class PhysicalFunctionSpace:
    """A deterministic coefficient latent evaluated from physical coordinates."""

    def __init__(self, cfg: FunctionSpaceConfig):
        self.cfg = cfg

    def _one_dimensional(self, values: torch.Tensor) -> torch.Tensor:
        features = [torch.ones_like(values)]
        for frequency in range(1, self.cfg.fourier_cutoff + 1):
            phase = 2.0 * math.pi * frequency * values
            features.extend((math.sqrt(2.0) * torch.sin(phase), math.sqrt(2.0) * torch.cos(phase)))
        return torch.stack(features, dim=-1)

    def basis(self, coords: torch.Tensor) -> torch.Tensor:
        if coords.dim() not in (2, 3) or coords.shape[-1] != 2:
            raise ValueError("coordinates must have shape [nodes, 2] or [batch, nodes, 2]")
        coords = coords.double()
        x = coords[..., 0]
        y = coords[..., 1]
        x_modes = self._one_dimensional(x)
        y_modes = self._one_dimensional(y)
        spectral = (x_modes.unsqueeze(-1) * y_modes.unsqueeze(-2)).flatten(-2)
        trends = torch.stack(
            (
                math.sqrt(12.0) * (x - 0.5),
                math.sqrt(12.0) * (y - 0.5),
                12.0 * (x - 0.5) * (y - 0.5),
            ),
            dim=-1,
        )
        basis = torch.cat((spectral, trends), dim=-1)
        if basis.shape[-1] != self.cfg.basis_dim:
            raise RuntimeError("function-space basis dimension drifted")
        return basis

    def project(
        self, values: torch.Tensor, coords: torch.Tensor, measure: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float | int]]:
        if values.dim() != 3 or values.shape[-1] != 1:
            raise ValueError("E7 supports one scalar field with shape [batch, nodes, 1]")
        if coords.dim() == 2:
            coords = coords.unsqueeze(0).expand(values.shape[0], -1, -1)
        if measure.dim() == 2:
            measure = measure.unsqueeze(0).expand(values.shape[0], -1, -1)
        if values.shape[:2] != coords.shape[:2] or values.shape[:2] != measure.shape[:2]:
            raise ValueError("values, coordinates, and measures must share batch and nodes")
        if torch.any(measure <= 0) or not torch.isfinite(measure).all():
            raise ValueError("projection requires positive finite quadrature masses")

        basis = self.basis(coords)
        weights = measure.double()
        weighted_design = basis * weights.sqrt()
        singular_values = torch.linalg.svdvals(weighted_design[0])
        rank = int((singular_values > self.cfg.rank_tolerance).sum().item())
        condition_number = float((singular_values[0] / singular_values[-1]).item())
        if rank != self.cfg.basis_dim:
            raise ValueError(f"projection design is rank deficient: {rank} != {self.cfg.basis_dim}")
        if condition_number > self.cfg.max_condition_number:
            raise ValueError(
                f"projection condition number {condition_number} exceeds "
                f"{self.cfg.max_condition_number}"
            )

        gram = basis.transpose(1, 2) @ (weights * basis)
        right_hand_side = basis.transpose(1, 2) @ (weights * values.double())
        coefficients = torch.linalg.solve(gram, right_hand_side)
        return coefficients, {
            "rank": rank,
            "condition_number": condition_number,
            "minimum_singular_value": float(singular_values[-1].item()),
            "maximum_singular_value": float(singular_values[0].item()),
        }

    def decode(self, coefficients: torch.Tensor, query_coords: torch.Tensor) -> torch.Tensor:
        if coefficients.dim() != 3 or coefficients.shape[1:] != (self.cfg.basis_dim, 1):
            raise ValueError("coefficients must have shape [batch, 52, 1]")
        if query_coords.dim() == 2:
            query_coords = query_coords.unsqueeze(0).expand(coefficients.shape[0], -1, -1)
        if query_coords.shape[0] != coefficients.shape[0]:
            if query_coords.shape[0] != 1:
                raise ValueError("query batch must equal coefficient batch or be one")
            query_coords = query_coords.expand(coefficients.shape[0], -1, -1)
        return self.basis(query_coords) @ coefficients


def _relative_mismatch(left: torch.Tensor, right: torch.Tensor) -> float:
    numerator = (left.double() - right.double()).square().sum()
    denominator = ((left.double() + right.double()) * 0.5).square().sum().clamp_min(1e-24)
    return float(torch.sqrt(numerator / denominator).item())


def _decoded_mismatch(left: torch.Tensor, right: torch.Tensor, target: torch.Tensor) -> float:
    numerator = (left.double() - right.double()).square().sum()
    denominator = target.double().square().sum().clamp_min(1e-24)
    return float(torch.sqrt(numerator / denominator).item())


def _weighted_projection_residual(
    prediction: torch.Tensor, target: torch.Tensor, measure: torch.Tensor
) -> float:
    numerator = (measure.double() * (prediction.double() - target.double()).square()).sum()
    denominator = (measure.double() * target.double().square()).sum().clamp_min(1e-24)
    return float(torch.sqrt(numerator / denominator).item())


def basis_calibration(space: PhysicalFunctionSpace, cfg: FunctionSpaceConfig) -> dict[str, Any]:
    axis = (
        torch.arange(cfg.calibration_resolution, dtype=torch.float64) + 0.5
    ) / cfg.calibration_resolution
    x, y = torch.meshgrid(axis, axis, indexing="ij")
    coords = torch.stack((x, y), dim=-1).reshape(1, -1, 2)
    measure = torch.full((1, coords.shape[1], 1), 1.0 / coords.shape[1], dtype=torch.float64)
    components = []
    for index in range(7):
        coefficients = torch.zeros(1, 7, dtype=torch.float64)
        coefficients[0, index] = 1.0
        components.append(evaluate_field(coefficients, coords).double())
    values = torch.cat(components, dim=0)
    repeated_coords = coords.expand(7, -1, -1)
    repeated_measure = measure.expand(7, -1, -1)
    latent, design = space.project(values, repeated_coords, repeated_measure)
    reconstruction = space.decode(latent, repeated_coords)
    errors = [
        global_nrmse(reconstruction[index : index + 1], values[index : index + 1])
        for index in range(7)
    ]
    return {
        "resolution": cfg.calibration_resolution,
        "component_nrmse": errors,
        "maximum_component_nrmse": max(errors),
        "pass": max(errors) <= cfg.max_calibration_component_nrmse,
        "design": design,
    }


def _source_order_error(
    space: PhysicalFunctionSpace,
    values: torch.Tensor,
    coords: torch.Tensor,
    measure: torch.Tensor,
    query: torch.Tensor,
    *,
    seed: int,
) -> float:
    coefficients, _ = space.project(values, coords, measure)
    original = space.decode(coefficients, query)
    permutation = torch.randperm(coords.shape[1], generator=torch.Generator().manual_seed(seed))
    permuted_coefficients, _ = space.project(
        values[:, permutation], coords[:, permutation], measure[:, permutation]
    )
    permuted = space.decode(permuted_coefficients, query)
    return float((original - permuted).abs().max().item())


def _evaluate_representation(
    space: PhysicalFunctionSpace,
    state_coefficients_: torch.Tensor,
    representation: Representation,
    canonical_query: torch.Tensor,
    canonical_target: torch.Tensor,
    cfg: FunctionSpaceConfig,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    coords, measure = representation_points(representation, batch=state_coefficients_.shape[0])
    values = evaluate_field(state_coefficients_, coords)
    latent, design = space.project(values, coords, measure)
    prediction = space.decode(latent, canonical_query)
    native_prediction = space.decode(latent, coords)
    interpolation = inverse_distance_interpolate(values, coords, canonical_query)
    interpolation_error = global_nrmse(interpolation, canonical_target)
    error = global_nrmse(prediction, canonical_target)
    target_scale = canonical_target.double().square().mean().sqrt().clamp_min(1e-12)
    return (
        {
            "canonical_query_nrmse": error,
            "inverse_distance_interpolation_nrmse": interpolation_error,
            "to_interpolation_ratio": error / interpolation_error,
            "weighted_source_projection_nrmse": _weighted_projection_residual(
                native_prediction, values, measure
            ),
            "high_frequency_spectral": high_frequency_spectral_report(
                prediction,
                canonical_target,
                resolution=cfg.canonical_query_resolution,
                minimum_radius=cfg.high_frequency_radius,
            ),
            "prediction_to_target_std_ratio": float(
                prediction.std() / canonical_target.double().std().clamp_min(1e-12)
            ),
            "normalized_mean_bias": float(
                (prediction - canonical_target.double()).mean().abs() / target_scale
            ),
            "effective_coefficient_rank": effective_rank(latent),
            "coefficient_norm_mean": float(torch.linalg.vector_norm(latent, dim=1).mean()),
            "source_node_count": coords.shape[1],
            "coefficient_count": cfg.basis_dim,
            "compression_ratio": coords.shape[1] / cfg.basis_dim,
            "design": design,
            "source_order_max_abs_error": _source_order_error(
                space,
                values,
                coords,
                measure,
                canonical_query,
                seed=cfg.seed + 1900,
            ),
        },
        latent,
        prediction,
    )


def _decision(result: dict[str, Any], cfg: FunctionSpaceConfig) -> dict[str, Any]:
    family_gates = {}
    for family in ("grid", "mesh"):
        summary = result["families"][family]
        high = summary["representations"]["high"]
        unseen = summary["representations"]["validation"]
        family_gates[family] = {
            "absolute_reconstruction_pass": (
                high["canonical_query_nrmse"]
                <= cfg.max_interpolation_baseline_ratio
                * high["inverse_distance_interpolation_nrmse"]
            ),
            "unseen_resolution_stability_pass": (
                unseen["canonical_query_nrmse"] <= 1.10 * high["canonical_query_nrmse"]
            ),
            "high_frequency_pass": (
                high["high_frequency_spectral"]["nrmse"] <= cfg.max_high_frequency_nrmse
            ),
            "refinement_coefficient_pass": (
                summary["coefficient_mismatch"]["high_vs_validation"]
                <= cfg.max_refinement_coefficient_mismatch
            ),
            "source_order_invariance_pass": (
                high["source_order_max_abs_error"] <= cfg.invariance_atol
            ),
            "design_pass": (
                high["design"]["rank"] == cfg.basis_dim
                and high["design"]["condition_number"] <= cfg.max_condition_number
            ),
            "compression_pass": (
                high["compression_ratio"] >= cfg.minimum_high_resolution_compression
            ),
        }
    paired = result["paired_grid_mesh"]
    semantic_gates = {
        "paired_coefficient_pass": (
            paired["high"]["coefficient_relative_mismatch"] <= cfg.max_paired_coefficient_mismatch
        ),
        "paired_decoded_pass": (
            paired["high"]["decoded_mismatch_nrmse"] <= cfg.max_paired_decoded_mismatch
        ),
        "basis_calibration_pass": result["basis_calibration"]["pass"],
        "all_designs_pass": all(
            representation["design"]["rank"] == cfg.basis_dim
            and representation["design"]["condition_number"] <= cfg.max_condition_number
            for family in result["families"].values()
            for representation in family["representations"].values()
        ),
    }
    absolute = all(gates["absolute_reconstruction_pass"] for gates in family_gates.values())
    qualified = (
        absolute
        and all(all(gates.values()) for gates in family_gates.values())
        and all(semantic_gates.values())
    )
    if qualified:
        classification = "function_space_latent_qualified"
        next_move = (
            "train one universal coordinate-quadrature encoder to infer the frozen coefficients"
        )
    elif absolute:
        classification = "function_space_sufficient_projection_unstable"
        next_move = "repair only deterministic projection or basis normalization"
    else:
        classification = "function_space_latent_not_qualified"
        next_move = "close the spectral-polynomial space and reconsider the common basis family"
    return {
        "classification": classification,
        "family_gates": family_gates,
        "semantic_gates": semantic_gates,
        "next_move": next_move,
    }


def run_function_space(cfg: FunctionSpaceConfig, *, run_dir: Path) -> dict[str, Any]:
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(1)
    run_dir.mkdir(parents=True, exist_ok=True)
    benchmark_cfg = cfg.benchmark_config()
    representations = _representations(benchmark_cfg)
    validation_coefficients = state_coefficients(cfg.validation_states, seed=cfg.seed + 10_000)
    canonical = Representation("canonical_queries", cfg.canonical_query_resolution, 0.0, 0.0)
    canonical_coords, _ = representation_points(canonical, batch=1)
    canonical_query = canonical_coords.expand(cfg.validation_states, -1, -1)
    canonical_target = evaluate_field(validation_coefficients, canonical_query).double()
    space = PhysicalFunctionSpace(cfg)

    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "canonical_latent_e7_function_space",
        "config": asdict(cfg),
        "basis_dimension": cfg.basis_dim,
        "basis_calibration": basis_calibration(space, cfg),
        "state_split": {
            "validation_count": cfg.validation_states,
            "validation_seed": cfg.seed + 10_000,
            "training_states_read": 0,
            "heldout_states_read": 0,
        },
        "families": {},
        "boundary": {
            "learned_parameters": 0,
            "optimizer_updates": 0,
            "operator_instantiated": False,
            "heldout_reads": 0,
            "representation_label_model_inputs": False,
            "task_label_model_inputs": False,
            "provider_calls": 0,
            "routing_paths": 0,
            "original_source_features_available_after_projection": False,
            "particles_scientifically_qualified": False,
        },
    }
    result["config_sha256"] = hashlib.sha256(
        json.dumps(result["config"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    all_latents: dict[str, torch.Tensor] = {}
    all_predictions: dict[str, torch.Tensor] = {}
    family_names = {
        "grid": {
            "low": f"grid_{cfg.train_low_resolution}",
            "high": f"grid_{cfg.train_high_resolution}",
            "validation": f"grid_{cfg.validation_resolution}",
        },
        "mesh": {
            "low": f"mesh_{cfg.train_low_resolution}_a",
            "high": f"mesh_{cfg.train_high_resolution}_a",
            "validation": f"mesh_{cfg.validation_resolution}_a",
        },
    }
    for family, labels in family_names.items():
        family_result: dict[str, Any] = {"representations": {}}
        for label, name in labels.items():
            evaluation, latent, prediction = _evaluate_representation(
                space,
                validation_coefficients,
                representations[name],
                canonical_query,
                canonical_target,
                cfg,
            )
            family_result["representations"][label] = evaluation
            all_latents[name] = latent
            all_predictions[name] = prediction
        family_result["coefficient_mismatch"] = {
            "low_vs_high": _relative_mismatch(
                all_latents[labels["low"]], all_latents[labels["high"]]
            ),
            "high_vs_validation": _relative_mismatch(
                all_latents[labels["high"]], all_latents[labels["validation"]]
            ),
        }
        family_result["decoded_mismatch"] = discretization_mismatch_report(
            {label: all_predictions[name] for label, name in labels.items()},
            canonical_target,
        )
        result["families"][family] = family_result

    paired = {}
    for label in ("low", "high", "validation"):
        grid_name = family_names["grid"][label]
        mesh_name = family_names["mesh"][label]
        paired[label] = {
            "coefficient_relative_mismatch": _relative_mismatch(
                all_latents[grid_name], all_latents[mesh_name]
            ),
            "decoded_mismatch_nrmse": _decoded_mismatch(
                all_predictions[grid_name], all_predictions[mesh_name], canonical_target
            ),
            "latent_report": paired_latent_report(all_latents[grid_name], all_latents[mesh_name]),
        }
    remesh_name = f"mesh_{cfg.train_high_resolution}_b"
    remesh_evaluation, remesh_latent, remesh_prediction = _evaluate_representation(
        space,
        validation_coefficients,
        representations[remesh_name],
        canonical_query,
        canonical_target,
        cfg,
    )
    high_mesh_name = family_names["mesh"]["high"]
    paired["mesh_high_remesh"] = {
        "coefficient_relative_mismatch": _relative_mismatch(
            all_latents[high_mesh_name], remesh_latent
        ),
        "decoded_mismatch_nrmse": _decoded_mismatch(
            all_predictions[high_mesh_name], remesh_prediction, canonical_target
        ),
        "remesh_evaluation": remesh_evaluation,
    }
    result["paired_grid_mesh"] = paired
    result["causal_decision"] = _decision(result, cfg)
    result_path = run_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result["result_path"] = str(result_path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the validation-only E7 deterministic function-space sufficiency test"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--validation-states", type=int, default=FunctionSpaceConfig.validation_states
    )
    parser.add_argument(
        "--calibration-resolution",
        type=int,
        default=FunctionSpaceConfig.calibration_resolution,
    )
    parser.add_argument("--print-json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = FunctionSpaceConfig(
        validation_states=args.validation_states,
        calibration_resolution=args.calibration_resolution,
    )
    result = run_function_space(cfg, run_dir=args.run_dir)
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
