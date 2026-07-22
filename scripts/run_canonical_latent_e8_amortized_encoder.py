#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

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
    from scripts.run_canonical_latent_e7_function_space import (
        FunctionSpaceConfig,
        PhysicalFunctionSpace,
        _decoded_mismatch,
        _relative_mismatch,
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
    from run_canonical_latent_e7_function_space import (  # type: ignore[no-redef]
        FunctionSpaceConfig,
        PhysicalFunctionSpace,
        _decoded_mismatch,
        _relative_mismatch,
    )
from ups.eval.latent_qualification import effective_rank, global_nrmse


@dataclass(frozen=True)
class AmortizedEncoderConfig:
    seed: int = 17
    train_states: int = 128
    validation_states: int = 24
    epochs: int = 120
    batch_size: int = 16
    learning_rate: float = 2e-3
    weight_decay: float = 1e-6
    hidden_dim: int = 64
    train_low_resolution: int = 10
    train_high_resolution: int = 14
    validation_resolution: int = 18
    canonical_query_resolution: int = 18
    particle_probe_points: int = 196
    coefficient_loss_weight: float = 0.5
    decoded_loss_weight: float = 0.5
    max_coefficient_nrmse: float = 0.10
    max_interpolation_baseline_ratio: float = 2.0
    high_frequency_radius: float = 3.0
    max_high_frequency_nrmse: float = 0.25
    max_unseen_resolution_ratio: float = 1.10
    max_paired_coefficient_mismatch: float = 0.10
    max_paired_decoded_mismatch: float = 0.10
    max_refinement_coefficient_mismatch: float = 0.15
    max_remesh_ratio: float = 1.10
    max_remesh_coefficient_mismatch: float = 0.15
    invariance_atol: float = 1e-8
    max_native_control_ratio: float = 1.10
    max_shared_macro_control_ratio: float = 0.98

    def __post_init__(self) -> None:
        if self.train_states < self.batch_size or self.train_states % self.batch_size:
            raise ValueError("train_states must contain complete batches")
        if self.validation_states < 4:
            raise ValueError("E8 requires at least four validation states")
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if self.hidden_dim != 64:
            raise ValueError("E8 freezes residual hidden width at 64")
        if self.coefficient_loss_weight + self.decoded_loss_weight != 1.0:
            raise ValueError("E8 loss weights must sum to one")

    def benchmark_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            seed=self.seed,
            train_states=self.train_states,
            validation_states=self.validation_states,
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            train_low_resolution=self.train_low_resolution,
            train_high_resolution=self.train_high_resolution,
            validation_resolution=self.validation_resolution,
            canonical_query_resolution=self.canonical_query_resolution,
        )

    def function_space_config(self) -> FunctionSpaceConfig:
        return FunctionSpaceConfig(
            seed=self.seed,
            validation_states=self.validation_states,
            train_low_resolution=self.train_low_resolution,
            train_high_resolution=self.train_high_resolution,
            validation_resolution=self.validation_resolution,
            canonical_query_resolution=self.canonical_query_resolution,
        )


class QuadratureMomentEncoder(nn.Module):
    """Amortize E7 from discretization-consistent physical basis moments."""

    def __init__(self, space: PhysicalFunctionSpace, *, hidden_dim: int = 64):
        super().__init__()
        self.space = space
        self.residual = nn.Sequential(
            nn.Linear(space.cfg.basis_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, space.cfg.basis_dim),
        ).double()
        output = self.residual[-1]
        assert isinstance(output, nn.Linear)
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)

    def forward(
        self, values: torch.Tensor, coords: torch.Tensor, measure: torch.Tensor
    ) -> torch.Tensor:
        if values.dim() != 3 or values.shape[-1] != 1:
            raise ValueError("values must have shape [batch, nodes, 1]")
        if coords.dim() == 2:
            coords = coords.unsqueeze(0).expand(values.shape[0], -1, -1)
        if measure.dim() == 2:
            measure = measure.unsqueeze(0).expand(values.shape[0], -1, -1)
        if values.shape[:2] != coords.shape[:2] or values.shape[:2] != measure.shape[:2]:
            raise ValueError("values, coordinates, and measures must share batch and nodes")
        if torch.any(measure <= 0) or not torch.isfinite(measure).all():
            raise ValueError("encoder requires positive finite quadrature masses")
        basis = self.space.basis(coords)
        moment = basis.transpose(1, 2) @ (measure.double() * values.double())
        vector = moment.squeeze(-1)
        return (vector + self.residual(vector)).unsqueeze(-1)


def _state_dict_sha256(module: nn.Module) -> str:
    buffer = io.BytesIO()
    torch.save(module.state_dict(), buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _tensor_nrmse(prediction: torch.Tensor, target: torch.Tensor) -> float:
    numerator = (prediction.double() - target.double()).square().sum()
    denominator = target.double().square().sum().clamp_min(1e-24)
    return float(torch.sqrt(numerator / denominator).item())


def _source_tensors(
    coefficients: torch.Tensor,
    representation: Representation,
    space: PhysicalFunctionSpace,
) -> dict[str, torch.Tensor]:
    coords, measure = representation_points(representation, batch=coefficients.shape[0])
    values = evaluate_field(coefficients, coords)
    teacher, _ = space.project(values, coords, measure)
    return {
        "coords": coords.double(),
        "measure": measure.double(),
        "values": values.double(),
        "teacher": teacher.double(),
    }


def _training_loss(
    encoder: QuadratureMomentEncoder,
    source: dict[str, torch.Tensor],
    indices: torch.Tensor,
    canonical_coords: torch.Tensor,
    canonical_target: torch.Tensor,
    cfg: AmortizedEncoderConfig,
) -> torch.Tensor:
    prediction = encoder(
        source["values"][indices],
        source["coords"][indices],
        source["measure"][indices],
    )
    teacher = source["teacher"][indices]
    coefficient_scale = teacher.square().mean().clamp_min(1e-12)
    coefficient_loss = F.mse_loss(prediction, teacher) / coefficient_scale
    decoded = encoder.space.decode(prediction, canonical_coords[indices])
    target = canonical_target[indices]
    decoded_scale = target.square().mean().clamp_min(1e-12)
    decoded_loss = F.mse_loss(decoded, target) / decoded_scale
    return cfg.coefficient_loss_weight * coefficient_loss + cfg.decoded_loss_weight * decoded_loss


def train_arm(
    encoder: QuadratureMomentEncoder,
    sources: dict[str, dict[str, torch.Tensor]],
    schedules: tuple[tuple[str, str], ...],
    canonical_coords: torch.Tensor,
    canonical_target: torch.Tensor,
    cfg: AmortizedEncoderConfig,
) -> dict[str, Any]:
    encoder.train()
    optimizer = torch.optim.AdamW(
        encoder.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    generator = torch.Generator().manual_seed(cfg.seed)
    history = []
    for epoch in range(cfg.epochs):
        source_names = schedules[epoch % len(schedules)]
        permutation = torch.randperm(cfg.train_states, generator=generator)
        epoch_loss = 0.0
        for start in range(0, cfg.train_states, cfg.batch_size):
            indices = permutation[start : start + cfg.batch_size]
            loss = sum(
                _training_loss(
                    encoder,
                    sources[name],
                    indices,
                    canonical_coords,
                    canonical_target,
                    cfg,
                )
                for name in source_names
            ) / len(source_names)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer.step()
            epoch_loss += float(loss.detach()) * indices.numel()
        history.append(epoch_loss / cfg.train_states)
    return {
        "epochs": cfg.epochs,
        "optimizer_updates": cfg.epochs * cfg.train_states // cfg.batch_size,
        "scheduled_source_examples": cfg.epochs * cfg.train_states * 2,
        "initial_loss": history[0],
        "final_loss": history[-1],
        "minimum_loss": min(history),
    }


@torch.no_grad()
def _evaluate_source(
    encoder: QuadratureMomentEncoder,
    coefficients: torch.Tensor,
    representation: Representation,
    canonical_query: torch.Tensor,
    canonical_target: torch.Tensor,
    cfg: AmortizedEncoderConfig,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    encoder.eval()
    coords, measure = representation_points(representation, batch=coefficients.shape[0])
    values = evaluate_field(coefficients, coords).double()
    teacher, design = encoder.space.project(values, coords, measure)
    prediction = encoder(values, coords, measure)
    decoded = encoder.space.decode(prediction, canonical_query)
    interpolation = inverse_distance_interpolate(
        values.float(), coords, canonical_query.float()
    ).double()
    coefficient_error = _tensor_nrmse(prediction, teacher)
    decoded_error = global_nrmse(decoded, canonical_target)
    interpolation_error = global_nrmse(interpolation, canonical_target)
    permutation = torch.randperm(
        coords.shape[1], generator=torch.Generator().manual_seed(cfg.seed + 1900)
    )
    permuted_prediction = encoder(
        values[:, permutation], coords[:, permutation], measure[:, permutation]
    )
    permuted_decoded = encoder.space.decode(permuted_prediction, canonical_query)
    return (
        {
            "coefficient_nrmse_to_e7_teacher": coefficient_error,
            "canonical_query_nrmse": decoded_error,
            "inverse_distance_interpolation_nrmse": interpolation_error,
            "to_interpolation_ratio": decoded_error / interpolation_error,
            "high_frequency_spectral": high_frequency_spectral_report(
                decoded,
                canonical_target,
                resolution=cfg.canonical_query_resolution,
                minimum_radius=cfg.high_frequency_radius,
            ),
            "effective_coefficient_rank": effective_rank(prediction),
            "coefficient_norm_mean": float(
                torch.linalg.vector_norm(prediction, dim=1).mean().item()
            ),
            "source_node_count": coords.shape[1],
            "teacher_design": design,
            "source_order_coefficient_max_abs_error": float(
                (prediction - permuted_prediction).abs().max().item()
            ),
            "source_order_decoded_max_abs_error": float(
                (decoded - permuted_decoded).abs().max().item()
            ),
        },
        prediction,
        decoded,
    )


@torch.no_grad()
def _particle_probe(
    encoder: QuadratureMomentEncoder,
    coefficients: torch.Tensor,
    canonical_query: torch.Tensor,
    canonical_target: torch.Tensor,
    cfg: AmortizedEncoderConfig,
) -> dict[str, Any]:
    coords = torch.rand(
        1,
        cfg.particle_probe_points,
        2,
        generator=torch.Generator().manual_seed(cfg.seed + 2401),
        dtype=torch.float64,
    ).expand(coefficients.shape[0], -1, -1)
    measure = torch.full(
        (coefficients.shape[0], cfg.particle_probe_points, 1),
        1.0 / cfg.particle_probe_points,
        dtype=torch.float64,
    )
    values = evaluate_field(coefficients, coords).double()
    teacher, design = encoder.space.project(values, coords, measure)
    prediction = encoder(values, coords, measure)
    decoded = encoder.space.decode(prediction, canonical_query)
    return {
        "scientific_gate_attached": False,
        "scientifically_qualified": False,
        "sampling": "deterministic uniform Monte Carlo coordinates with equal masses",
        "coordinate_seed": cfg.seed + 2401,
        "point_count": cfg.particle_probe_points,
        "coefficient_nrmse_to_e7_teacher": _tensor_nrmse(prediction, teacher),
        "canonical_query_nrmse": global_nrmse(decoded, canonical_target),
        "teacher_design": design,
    }


def _decision(result: dict[str, Any], cfg: AmortizedEncoderConfig) -> dict[str, Any]:
    shared = result["evaluation"]["arms"]["shared"]
    grid_high = shared["grid_high"]
    mesh_high = shared["mesh_high"]
    grid_unseen = shared["grid_unseen"]
    mesh_unseen = shared["mesh_unseen"]
    remesh = shared["mesh_remesh"]
    paired = result["evaluation"]["shared_semantics"]
    semantic_gates = {
        "coefficient_accuracy": (
            grid_high["coefficient_nrmse_to_e7_teacher"] <= cfg.max_coefficient_nrmse
            and mesh_high["coefficient_nrmse_to_e7_teacher"] <= cfg.max_coefficient_nrmse
        ),
        "absolute_reconstruction": (
            grid_high["to_interpolation_ratio"] <= cfg.max_interpolation_baseline_ratio
            and mesh_high["to_interpolation_ratio"] <= cfg.max_interpolation_baseline_ratio
        ),
        "high_frequency": (
            grid_high["high_frequency_spectral"]["nrmse"] <= cfg.max_high_frequency_nrmse
            and mesh_high["high_frequency_spectral"]["nrmse"] <= cfg.max_high_frequency_nrmse
        ),
        "unseen_resolution_stability": (
            grid_unseen["canonical_query_nrmse"]
            <= cfg.max_unseen_resolution_ratio * grid_high["canonical_query_nrmse"]
            and mesh_unseen["canonical_query_nrmse"]
            <= cfg.max_unseen_resolution_ratio * mesh_high["canonical_query_nrmse"]
        ),
        "paired_semantics": (
            paired["high_grid_mesh_coefficient_mismatch"] <= cfg.max_paired_coefficient_mismatch
            and paired["high_grid_mesh_decoded_mismatch"] <= cfg.max_paired_decoded_mismatch
        ),
        "refinement": (
            paired["grid_high_unseen_coefficient_mismatch"]
            <= cfg.max_refinement_coefficient_mismatch
            and paired["mesh_high_unseen_coefficient_mismatch"]
            <= cfg.max_refinement_coefficient_mismatch
        ),
        "remeshing": (
            remesh["canonical_query_nrmse"]
            <= cfg.max_remesh_ratio * mesh_high["canonical_query_nrmse"]
            and paired["mesh_high_remesh_coefficient_mismatch"]
            <= cfg.max_remesh_coefficient_mismatch
        ),
        "source_order_invariance": all(
            source["source_order_coefficient_max_abs_error"] <= cfg.invariance_atol
            and source["source_order_decoded_max_abs_error"] <= cfg.invariance_atol
            for source in shared.values()
        ),
        "boundary": all(
            (
                not result["boundary"]["operator_instantiated"],
                result["boundary"]["heldout_reads"] == 0,
                not result["boundary"]["representation_label_model_inputs"],
                not result["boundary"]["routing_paths"],
                not result["boundary"]["original_source_features_available_after_encoding"],
            )
        ),
    }
    transfer = result["evaluation"]["positive_transfer"]
    transfer_gates = {
        "matched_native_noninferiority": (
            transfer["shared_to_matched_grid_control_ratio"] <= cfg.max_native_control_ratio
            and transfer["shared_to_matched_mesh_control_ratio"] <= cfg.max_native_control_ratio
        ),
        "cross_family_macro_advantage": (
            transfer["shared_to_grid_control_macro_ratio"] <= cfg.max_shared_macro_control_ratio
            and transfer["shared_to_mesh_control_macro_ratio"] <= cfg.max_shared_macro_control_ratio
        ),
    }
    semantic_pass = all(semantic_gates.values())
    transfer_pass = all(transfer_gates.values())
    if semantic_pass and transfer_pass:
        classification = "amortized_universal_encoder_qualified"
        next_move = "freeze the learned encoder and preregister a coefficient-space operator gate"
    elif semantic_pass:
        classification = "amortized_encoder_capable_without_positive_transfer"
        next_move = "establish a meaningful cross-representation advantage before dynamics"
    else:
        classification = "amortized_encoder_not_qualified"
        next_move = "diagnose only the learned correction or objective with the E7 basis frozen"
    return {
        "classification": classification,
        "semantic_gates": semantic_gates,
        "positive_transfer_gates": transfer_gates,
        "next_move": next_move,
    }


def run_amortized_encoder(cfg: AmortizedEncoderConfig, *, run_dir: Path) -> dict[str, Any]:
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(1)
    run_dir.mkdir(parents=True, exist_ok=True)
    benchmark_cfg = cfg.benchmark_config()
    representations = _representations(benchmark_cfg)
    space = PhysicalFunctionSpace(cfg.function_space_config())
    train_coefficients = state_coefficients(cfg.train_states, seed=cfg.seed)
    validation_coefficients = state_coefficients(cfg.validation_states, seed=cfg.seed + 10_000)
    canonical_representation = Representation(
        "canonical_queries", cfg.canonical_query_resolution, 0.0, 0.0
    )
    canonical_coords, _ = representation_points(canonical_representation, batch=1)
    canonical_coords = canonical_coords.double()
    train_query = canonical_coords.expand(cfg.train_states, -1, -1)
    validation_query = canonical_coords.expand(cfg.validation_states, -1, -1)
    train_target = evaluate_field(train_coefficients, train_query).double()
    validation_target = evaluate_field(validation_coefficients, validation_query).double()

    low = cfg.train_low_resolution
    high = cfg.train_high_resolution
    unseen = cfg.validation_resolution
    source_names = {
        "grid_low": f"grid_{low}",
        "grid_high": f"grid_{high}",
        "grid_unseen": f"grid_{unseen}",
        "mesh_low": f"mesh_{low}_a",
        "mesh_high": f"mesh_{high}_a",
        "mesh_unseen": f"mesh_{unseen}_a",
        "mesh_remesh": f"mesh_{high}_b",
    }
    train_sources = {
        name: _source_tensors(train_coefficients, representations[name], space)
        for name in (
            source_names["grid_low"],
            source_names["grid_high"],
            source_names["mesh_low"],
            source_names["mesh_high"],
        )
    }

    initial = QuadratureMomentEncoder(space, hidden_dim=cfg.hidden_dim)
    arms = {
        "shared": copy.deepcopy(initial),
        "grid_control": copy.deepcopy(initial),
        "mesh_control": copy.deepcopy(initial),
    }
    initial_sha = _state_dict_sha256(initial)
    schedules = {
        "shared": (
            (source_names["grid_low"], source_names["mesh_high"]),
            (source_names["mesh_low"], source_names["grid_high"]),
        ),
        "grid_control": ((source_names["grid_low"], source_names["grid_high"]),),
        "mesh_control": ((source_names["mesh_low"], source_names["mesh_high"]),),
    }
    training = {}
    for name, encoder in arms.items():
        training[name] = train_arm(
            encoder,
            train_sources,
            schedules[name],
            train_query,
            train_target,
            cfg,
        )
        training[name]["checkpoint_sha256"] = _state_dict_sha256(encoder)

    evaluation: dict[str, Any] = {"arms": {}}
    tensors: dict[str, dict[str, tuple[torch.Tensor, torch.Tensor]]] = {}
    evaluated_labels = ("grid_high", "grid_unseen", "mesh_high", "mesh_unseen")
    for arm_name, encoder in arms.items():
        arm_result = {}
        tensors[arm_name] = {}
        labels = evaluated_labels if arm_name != "shared" else tuple(source_names)
        for label in labels:
            report, latent, decoded = _evaluate_source(
                encoder,
                validation_coefficients,
                representations[source_names[label]],
                validation_query,
                validation_target,
                cfg,
            )
            arm_result[label] = report
            tensors[arm_name][label] = (latent, decoded)
        evaluation["arms"][arm_name] = arm_result

    shared_tensors = tensors["shared"]
    evaluation["shared_semantics"] = {
        "high_grid_mesh_coefficient_mismatch": _relative_mismatch(
            shared_tensors["grid_high"][0], shared_tensors["mesh_high"][0]
        ),
        "high_grid_mesh_decoded_mismatch": _decoded_mismatch(
            shared_tensors["grid_high"][1],
            shared_tensors["mesh_high"][1],
            validation_target,
        ),
        "grid_high_unseen_coefficient_mismatch": _relative_mismatch(
            shared_tensors["grid_high"][0], shared_tensors["grid_unseen"][0]
        ),
        "mesh_high_unseen_coefficient_mismatch": _relative_mismatch(
            shared_tensors["mesh_high"][0], shared_tensors["mesh_unseen"][0]
        ),
        "mesh_high_remesh_coefficient_mismatch": _relative_mismatch(
            shared_tensors["mesh_high"][0], shared_tensors["mesh_remesh"][0]
        ),
    }
    macros = {}
    for arm_name in arms:
        errors = [
            evaluation["arms"][arm_name][label]["coefficient_nrmse_to_e7_teacher"]
            for label in evaluated_labels
        ]
        macros[arm_name] = sum(errors) / len(errors)
    shared_grid = evaluation["arms"]["shared"]["grid_high"]["coefficient_nrmse_to_e7_teacher"]
    shared_mesh = evaluation["arms"]["shared"]["mesh_high"]["coefficient_nrmse_to_e7_teacher"]
    grid_control = evaluation["arms"]["grid_control"]["grid_high"][
        "coefficient_nrmse_to_e7_teacher"
    ]
    mesh_control = evaluation["arms"]["mesh_control"]["mesh_high"][
        "coefficient_nrmse_to_e7_teacher"
    ]
    evaluation["positive_transfer"] = {
        "macro_coefficient_nrmse": macros,
        "shared_to_matched_grid_control_ratio": shared_grid / grid_control,
        "shared_to_matched_mesh_control_ratio": shared_mesh / mesh_control,
        "shared_to_grid_control_macro_ratio": macros["shared"] / macros["grid_control"],
        "shared_to_mesh_control_macro_ratio": macros["shared"] / macros["mesh_control"],
    }
    evaluation["particle_mechanics_probe"] = _particle_probe(
        arms["shared"],
        validation_coefficients,
        validation_query,
        validation_target,
        cfg,
    )

    config_payload = asdict(cfg)
    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "canonical_latent_e8_amortized_universal_encoder",
        "config": config_payload,
        "config_sha256": hashlib.sha256(
            json.dumps(config_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "initial_checkpoint_sha256": initial_sha,
        "architecture": {
            "kind": "quadrature_basis_moment_residual",
            "basis_dimension": space.cfg.basis_dim,
            "hidden_dimension": cfg.hidden_dim,
            "learned_parameters": sum(parameter.numel() for parameter in initial.parameters()),
            "frozen_basis": "E7 cutoff-three tensor Fourier plus x, y, xy trends",
            "representation_blind": True,
        },
        "state_split": {
            "train_count": cfg.train_states,
            "validation_count": cfg.validation_states,
            "train_seed": cfg.seed,
            "validation_seed": cfg.seed + 10_000,
            "heldout_states_read": 0,
        },
        "training": training,
        "evaluation": evaluation,
        "boundary": {
            "operator_instantiated": False,
            "temporal_transitions": 0,
            "heldout_reads": 0,
            "representation_label_model_inputs": False,
            "task_label_model_inputs": False,
            "provider_calls": 0,
            "routing_paths": 0,
            "original_source_features_available_after_encoding": False,
            "exact_e7_solve_used_at_inference": False,
            "particles_scientifically_qualified": False,
        },
    }
    result["causal_decision"] = _decision(result, cfg)
    result_path = run_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result["result_path"] = str(result_path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the E8 amortized universal coefficient encoder experiment"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=AmortizedEncoderConfig.epochs)
    parser.add_argument("--train-states", type=int, default=AmortizedEncoderConfig.train_states)
    parser.add_argument(
        "--validation-states", type=int, default=AmortizedEncoderConfig.validation_states
    )
    parser.add_argument("--batch-size", type=int, default=AmortizedEncoderConfig.batch_size)
    parser.add_argument("--print-json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = AmortizedEncoderConfig(
        epochs=args.epochs,
        train_states=args.train_states,
        validation_states=args.validation_states,
        batch_size=args.batch_size,
    )
    result = run_amortized_encoder(cfg, run_dir=args.run_dir)
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
