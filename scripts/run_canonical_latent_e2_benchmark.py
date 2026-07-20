#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from ups.eval.latent_qualification import (
    cross_discretization_codec_report,
    discretization_mismatch_report,
    effective_rank,
    global_nrmse,
    paired_latent_report,
)
from ups.io import (
    AnyPointDecoder,
    AnyPointDecoderConfig,
    CanonicalPointEncoder,
    CanonicalPointEncoderConfig,
)


@dataclass(frozen=True)
class BenchmarkConfig:
    seed: int = 17
    train_states: int = 128
    validation_states: int = 24
    epochs: int = 120
    batch_size: int = 16
    learning_rate: float = 2e-3
    latent_len: int = 8
    latent_dim: int = 32
    hidden_dim: int = 32
    supernodes: int = 24
    supernode_neighbors: int = 16
    transformer_layers: int = 1
    alignment_weight: float = 0.10
    train_low_resolution: int = 10
    train_high_resolution: int = 14
    validation_resolution: int = 18
    canonical_query_resolution: int = 18
    permutation_trials: int = 199
    max_interpolation_baseline_ratio: float = 2.0

    def __post_init__(self) -> None:
        if self.train_states < self.batch_size or self.validation_states < 4:
            raise ValueError(
                "benchmark requires at least one train batch and four validation states"
            )
        if self.train_states % self.batch_size:
            raise ValueError("train_states must be divisible by batch_size")
        if self.permutation_trials < 99:
            raise ValueError("permutation_trials must be at least 99")


@dataclass(frozen=True)
class Representation:
    name: str
    resolution: int
    warp_a: float
    warp_b: float


class CanonicalCodec(nn.Module):
    def __init__(self, cfg: BenchmarkConfig):
        super().__init__()
        self.encoder = CanonicalPointEncoder(
            CanonicalPointEncoderConfig(
                latent_len=cfg.latent_len,
                latent_dim=cfg.latent_dim,
                hidden_dim=cfg.hidden_dim,
                coord_dim=2,
                field_channels={"u": 1},
                supernodes=cfg.supernodes,
                supernode_neighbors=cfg.supernode_neighbors,
                transformer_layers=cfg.transformer_layers,
                num_heads=4,
                require_measure=True,
            )
        )
        self.decoder = AnyPointDecoder(
            AnyPointDecoderConfig(
                latent_dim=cfg.latent_dim,
                query_dim=2,
                hidden_dim=cfg.hidden_dim,
                num_layers=1,
                num_heads=4,
                frequencies=(1.0, 2.0, 3.0),
                mlp_hidden_dim=cfg.hidden_dim,
                output_channels={"u": 1},
            )
        )

    def encode(
        self, values: torch.Tensor, coords: torch.Tensor, measure: torch.Tensor
    ) -> torch.Tensor:
        return self.encoder({"u": values}, coords, geom={"measure": measure})

    def decode(self, latent: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        return self.decoder(query, latent)["u"]


def state_coefficients(count: int, *, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    coefficients = torch.randn(count, 7, generator=generator)
    coefficients[:, 0] *= 0.25
    coefficients[:, 1:] *= torch.tensor([0.9, 0.8, 0.6, 0.5, 0.35, 0.25])
    return coefficients


def evaluate_field(coefficients: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    if coords.dim() == 2:
        coords = coords.unsqueeze(0).expand(coefficients.shape[0], -1, -1)
    x = coords[..., 0]
    y = coords[..., 1]
    c = coefficients
    gaussian = torch.exp(-24.0 * ((x - 0.31).square() + (y - 0.68).square()))
    field = (
        c[:, 0:1]
        + c[:, 1:2] * torch.sin(2.0 * math.pi * x)
        + c[:, 2:3] * torch.cos(2.0 * math.pi * y)
        + c[:, 3:4] * torch.sin(2.0 * math.pi * (x + y))
        + c[:, 4:5] * gaussian
        + c[:, 5:6] * torch.sin(6.0 * math.pi * x) * torch.cos(4.0 * math.pi * y)
        + c[:, 6:7] * (x - 0.5) * (y - 0.5)
    )
    return field.unsqueeze(-1)


def representation_points(
    representation: Representation, *, batch: int
) -> tuple[torch.Tensor, torch.Tensor]:
    resolution = representation.resolution
    axis = (torch.arange(resolution, dtype=torch.float32) + 0.5) / resolution
    u, v = torch.meshgrid(axis, axis, indexing="ij")
    sin_u = torch.sin(2.0 * math.pi * u)
    cos_u = torch.cos(2.0 * math.pi * u)
    sin_v = torch.sin(2.0 * math.pi * v)
    cos_v = torch.cos(2.0 * math.pi * v)
    scale = 1.0 / (2.0 * math.pi)
    x = u + representation.warp_a * scale * sin_u * sin_v
    y = v + representation.warp_b * scale * sin_u * sin_v
    jacobian = 1.0 + representation.warp_a * cos_u * sin_v + representation.warp_b * sin_u * cos_v
    if torch.any(jacobian <= 0):
        raise ValueError(f"representation '{representation.name}' has non-positive Jacobian")
    coords = torch.stack([x, y], dim=-1).reshape(1, resolution * resolution, 2)
    measure = jacobian.reshape(1, resolution * resolution, 1)
    measure = measure / measure.sum(dim=1, keepdim=True)
    return coords.expand(batch, -1, -1), measure.expand(batch, -1, -1)


def _representations(cfg: BenchmarkConfig) -> dict[str, Representation]:
    low = cfg.train_low_resolution
    high = cfg.train_high_resolution
    validation = cfg.validation_resolution
    return {
        f"grid_{low}": Representation(f"grid_{low}", low, 0.0, 0.0),
        f"grid_{high}": Representation(f"grid_{high}", high, 0.0, 0.0),
        f"grid_{validation}": Representation(f"grid_{validation}", validation, 0.0, 0.0),
        f"mesh_{low}_a": Representation(f"mesh_{low}_a", low, 0.24, -0.17),
        f"mesh_{high}_a": Representation(f"mesh_{high}_a", high, 0.24, -0.17),
        f"mesh_{high}_b": Representation(f"mesh_{high}_b", high, -0.19, 0.22),
        f"mesh_{validation}_a": Representation(f"mesh_{validation}_a", validation, 0.24, -0.17),
    }


def _codec_loss(
    codec: CanonicalCodec,
    coefficients: torch.Tensor,
    representations: tuple[Representation, Representation],
    canonical_coords: torch.Tensor,
    cfg: BenchmarkConfig,
) -> torch.Tensor:
    latents = []
    reconstruction_losses = []
    canonical_target = evaluate_field(coefficients, canonical_coords)
    for representation in representations:
        coords, measure = representation_points(representation, batch=coefficients.shape[0])
        values = evaluate_field(coefficients, coords)
        latent = codec.encode(values, coords, measure)
        latents.append(latent)
        native_prediction = codec.decode(latent, coords)
        canonical_prediction = codec.decode(latent, canonical_coords)
        native_scale = values.square().mean().clamp_min(1e-8)
        canonical_scale = canonical_target.square().mean().clamp_min(1e-8)
        reconstruction_losses.append(
            0.5 * F.mse_loss(native_prediction, values) / native_scale
            + 0.5 * F.mse_loss(canonical_prediction, canonical_target) / canonical_scale
        )
    left = F.layer_norm(latents[0].flatten(1), (latents[0][0].numel(),))
    right = F.layer_norm(latents[1].flatten(1), (latents[1][0].numel(),))
    alignment = F.mse_loss(left, right)
    return sum(reconstruction_losses) / 2.0 + cfg.alignment_weight * alignment


def train_arm(
    codec: CanonicalCodec,
    coefficients: torch.Tensor,
    representation_pairs: tuple[tuple[Representation, Representation], ...],
    canonical_coords: torch.Tensor,
    cfg: BenchmarkConfig,
) -> dict[str, Any]:
    codec.train()
    optimizer = torch.optim.AdamW(codec.parameters(), lr=cfg.learning_rate, weight_decay=1e-6)
    generator = torch.Generator().manual_seed(cfg.seed)
    history = []
    for epoch in range(cfg.epochs):
        representations = representation_pairs[epoch % len(representation_pairs)]
        permutation = torch.randperm(coefficients.shape[0], generator=generator)
        epoch_loss = 0.0
        for start in range(0, coefficients.shape[0], cfg.batch_size):
            batch = coefficients[permutation[start : start + cfg.batch_size]]
            query = canonical_coords.expand(batch.shape[0], -1, -1)
            loss = _codec_loss(codec, batch, representations, query, cfg)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(codec.parameters(), 1.0)
            optimizer.step()
            epoch_loss += float(loss.detach()) * batch.shape[0]
        history.append(epoch_loss / coefficients.shape[0])
    return {
        "epochs": cfg.epochs,
        "optimizer_updates": cfg.epochs * coefficients.shape[0] // cfg.batch_size,
        "scheduled_source_examples": cfg.epochs * coefficients.shape[0] * 2,
        "initial_loss": history[0],
        "final_loss": history[-1],
        "minimum_loss": min(history),
    }


def _state_dict_sha256(module: nn.Module) -> str:
    buffer = io.BytesIO()
    torch.save(module.state_dict(), buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _encode_representation(
    codec: CanonicalCodec, coefficients: torch.Tensor, representation: Representation
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    coords, measure = representation_points(representation, batch=coefficients.shape[0])
    values = evaluate_field(coefficients, coords)
    latent = codec.encode(values, coords, measure)
    return latent, coords, measure, values


def _permutation_p_value(
    left: torch.Tensor, right: torch.Tensor, *, trials: int, seed: int
) -> float:
    observed = paired_latent_report(left, right)["retrieval"]["symmetric_top1"]
    generator = torch.Generator().manual_seed(seed)
    exceedances = 0
    for _ in range(trials):
        permutation = torch.randperm(right.shape[0], generator=generator)
        score = paired_latent_report(left, right[permutation])["retrieval"]["symmetric_top1"]
        exceedances += int(score >= observed)
    return (exceedances + 1) / (trials + 1)


def inverse_distance_interpolate(
    values: torch.Tensor,
    source_coords: torch.Tensor,
    query_coords: torch.Tensor,
    *,
    neighbors: int = 4,
) -> torch.Tensor:
    distances = torch.cdist(query_coords, source_coords)
    neighbor_distances, neighbor_indices = distances.topk(
        min(neighbors, source_coords.shape[1]), dim=-1, largest=False
    )
    expanded = values.unsqueeze(1).expand(-1, query_coords.shape[1], -1, -1)
    gathered = torch.gather(expanded, dim=2, index=neighbor_indices.unsqueeze(-1))
    weights = neighbor_distances.clamp_min(1e-6).reciprocal()
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return (gathered * weights.unsqueeze(-1)).sum(dim=2)


@torch.no_grad()
def evaluate_arms(
    arms: dict[str, CanonicalCodec],
    coefficients: torch.Tensor,
    representations: dict[str, Representation],
    canonical_coords: torch.Tensor,
    cfg: BenchmarkConfig,
) -> dict[str, Any]:
    for codec in arms.values():
        codec.eval()
    canonical_query = canonical_coords.expand(coefficients.shape[0], -1, -1)
    canonical_target = evaluate_field(coefficients, canonical_query)
    shared = arms["shared"]
    latents: dict[str, torch.Tensor] = {}
    predictions: dict[str, torch.Tensor] = {}
    samples: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for name, representation in representations.items():
        latent, coords, measure, values = _encode_representation(
            shared, coefficients, representation
        )
        latents[name] = latent
        predictions[name] = shared.decode(latent, canonical_query)
        samples[name] = (coords, measure, values)

    high = cfg.train_high_resolution
    validation = cfg.validation_resolution
    grid_name = f"grid_{high}"
    mesh_name = f"mesh_{high}_a"
    remesh_name = f"mesh_{high}_b"
    grid_latent = latents[grid_name]
    mesh_latent = latents[mesh_name]
    grid_coords, _, grid_values = samples[grid_name]
    mesh_coords, mesh_measure, mesh_values = samples[mesh_name]
    codec_matrix = cross_discretization_codec_report(
        {
            "grid": {
                "grid": shared.decode(grid_latent, grid_coords),
                "mesh": shared.decode(grid_latent, mesh_coords),
            },
            "mesh": {
                "grid": shared.decode(mesh_latent, grid_coords),
                "mesh": shared.decode(mesh_latent, mesh_coords),
            },
        },
        {"grid": grid_values, "mesh": mesh_values},
    )
    paired = paired_latent_report(grid_latent, mesh_latent)
    negative_pair_rmse = paired_latent_report(grid_latent, mesh_latent.roll(1, 0))[
        "standardized_pair_rmse"
    ]
    permutation_p = _permutation_p_value(
        grid_latent, mesh_latent, trials=cfg.permutation_trials, seed=cfg.seed + 700
    )

    grid_control_latent, _, _, _ = _encode_representation(
        arms["grid_control"], coefficients, representations[grid_name]
    )
    mesh_control_latent, _, _, _ = _encode_representation(
        arms["mesh_control"], coefficients, representations[mesh_name]
    )
    grid_control_prediction = arms["grid_control"].decode(grid_control_latent, canonical_query)
    mesh_control_prediction = arms["mesh_control"].decode(mesh_control_latent, canonical_query)
    shared_grid_error = global_nrmse(predictions[grid_name], canonical_target)
    shared_mesh_error = global_nrmse(predictions[mesh_name], canonical_target)
    grid_control_error = global_nrmse(grid_control_prediction, canonical_target)
    mesh_control_error = global_nrmse(mesh_control_prediction, canonical_target)
    grid_interpolation_error = global_nrmse(
        inverse_distance_interpolate(grid_values, grid_coords, canonical_query),
        canonical_target,
    )
    mesh_interpolation_error = global_nrmse(
        inverse_distance_interpolate(mesh_values, mesh_coords, canonical_query),
        canonical_target,
    )

    generator = torch.Generator().manual_seed(cfg.seed + 900)
    permutation = torch.randperm(mesh_coords.shape[1], generator=generator)
    permuted_latent = shared.encode(
        mesh_values[:, permutation],
        mesh_coords[:, permutation],
        mesh_measure[:, permutation],
    )
    permutation_latent_max_abs = float((mesh_latent - permuted_latent).abs().max())
    permutation_decoded_max_abs = float(
        (
            shared.decode(mesh_latent, canonical_query)
            - shared.decode(permuted_latent, canonical_query)
        )
        .abs()
        .max()
    )

    diagonal = codec_matrix["global_nrmse_matrix"]
    worst_within = max(diagonal["grid"]["grid"], diagonal["mesh"]["mesh"])
    off_diagonal = [diagonal["grid"]["mesh"], diagonal["mesh"]["grid"]]
    better_native = min(diagonal["grid"]["grid"], diagonal["mesh"]["mesh"])
    retrieval = paired["retrieval"]["symmetric_top1"]
    chance = paired["retrieval"]["chance_top1"]
    shared_grid_rank = paired["effective_rank"]["left"]
    shared_mesh_rank = paired["effective_rank"]["right"]
    grid_control_rank = effective_rank(grid_control_latent)
    mesh_control_rank = effective_rank(mesh_control_latent)
    physical_state_rank = effective_rank(canonical_target)
    remesh_error = global_nrmse(predictions[remesh_name], canonical_target)
    mismatch = discretization_mismatch_report(
        {
            name: predictions[name]
            for name in (
                f"grid_{cfg.train_low_resolution}",
                f"grid_{high}",
                f"grid_{validation}",
                f"mesh_{cfg.train_low_resolution}_a",
                mesh_name,
                remesh_name,
                f"mesh_{validation}_a",
            )
        },
        canonical_target,
    )
    mismatch_pairs = mismatch["pairwise_output_mismatch_nrmse"]
    low_mismatch = mismatch_pairs[
        f"grid_{cfg.train_low_resolution}__vs__mesh_{cfg.train_low_resolution}_a"
    ]
    high_mismatch = mismatch_pairs[f"grid_{validation}__vs__mesh_{validation}_a"]

    gates = {
        "identity": permutation_latent_max_abs <= 1e-5 and permutation_decoded_max_abs <= 1e-5,
        "within_codec": shared_grid_error <= 1.10 * grid_control_error
        and shared_mesh_error <= 1.10 * mesh_control_error,
        "absolute_reconstruction": shared_grid_error
        <= cfg.max_interpolation_baseline_ratio * grid_interpolation_error
        and shared_mesh_error <= cfg.max_interpolation_baseline_ratio * mesh_interpolation_error,
        "cross_codec": all(error <= 1.10 * worst_within for error in off_diagonal),
        "canonical_queries": shared_grid_error <= 1.10 * better_native
        and shared_mesh_error <= 1.10 * better_native,
        "paired_identity": retrieval >= 0.90
        and retrieval >= 10.0 * chance
        and permutation_p < 0.01,
        "alignment_margin": paired["standardized_pair_rmse"] <= 0.50 * negative_pair_rmse,
        "rank": shared_grid_rank >= 0.80 * grid_control_rank
        and shared_mesh_rank >= 0.80 * mesh_control_rank
        and shared_grid_rank >= 0.80 * physical_state_rank
        and shared_mesh_rank >= 0.80 * physical_state_rank
        and shared_grid_rank > 0.0
        and shared_mesh_rank > 0.0,
        "remeshing": remesh_error <= 1.10 * shared_mesh_error,
        "resolution_convergence": high_mismatch <= 1.10 * low_mismatch,
        "boundary": True,
    }

    return {
        "status": "qualified" if all(gates.values()) else "not_qualified",
        "gates": gates,
        "paired_latent": {
            **paired,
            "fixed_negative_pair_rmse": negative_pair_rmse,
            "permutation_p_value": permutation_p,
        },
        "codec_matrix": codec_matrix,
        "canonical_query_nrmse": {
            "shared_grid": shared_grid_error,
            "shared_mesh": shared_mesh_error,
            "shared_remesh": remesh_error,
            "grid_control": grid_control_error,
            "mesh_control": mesh_control_error,
            "grid_inverse_distance_interpolation": grid_interpolation_error,
            "mesh_inverse_distance_interpolation": mesh_interpolation_error,
        },
        "effective_rank": {
            "shared_grid": shared_grid_rank,
            "shared_mesh": shared_mesh_rank,
            "grid_control": grid_control_rank,
            "mesh_control": mesh_control_rank,
            "physical_state": physical_state_rank,
        },
        "identity_permutation": {
            "latent_max_abs": permutation_latent_max_abs,
            "decoded_max_abs": permutation_decoded_max_abs,
        },
        "discretization_mismatch": {
            **mismatch,
            "low_grid_mesh_mismatch_nrmse": low_mismatch,
            "high_grid_mesh_mismatch_nrmse": high_mismatch,
        },
        "boundary": {
            "operator_instantiated": False,
            "heldout_reads": 0,
            "representation_label_model_inputs": False,
            "task_label_model_inputs": False,
            "provider_calls": 0,
        },
    }


def run_benchmark(cfg: BenchmarkConfig, *, run_dir: Path) -> dict[str, Any]:
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    torch.set_num_threads(1)
    run_dir.mkdir(parents=True, exist_ok=True)
    representations = _representations(cfg)
    train_coefficients = state_coefficients(cfg.train_states, seed=cfg.seed)
    validation_coefficients = state_coefficients(cfg.validation_states, seed=cfg.seed + 10_000)
    canonical_representation = Representation(
        "canonical_queries", cfg.canonical_query_resolution, 0.0, 0.0
    )
    canonical_coords, _ = representation_points(canonical_representation, batch=1)

    initial = CanonicalCodec(cfg)
    arms = {
        "shared": copy.deepcopy(initial),
        "grid_control": copy.deepcopy(initial),
        "mesh_control": copy.deepcopy(initial),
    }
    initial_sha = _state_dict_sha256(initial)
    torch.save(initial.state_dict(), run_dir / "initial_codec.pt")
    low = cfg.train_low_resolution
    high = cfg.train_high_resolution
    pairs = {
        "shared": (
            (representations[f"grid_{low}"], representations[f"mesh_{low}_a"]),
            (representations[f"grid_{high}"], representations[f"mesh_{high}_a"]),
        ),
        "grid_control": ((representations[f"grid_{low}"], representations[f"grid_{high}"]),),
        "mesh_control": (
            (
                representations[f"mesh_{low}_a"],
                representations[f"mesh_{high}_a"],
            ),
        ),
    }
    training = {}
    for name, codec in arms.items():
        training[name] = train_arm(codec, train_coefficients, pairs[name], canonical_coords, cfg)
        torch.save(codec.state_dict(), run_dir / f"{name}_codec.pt")
        training[name]["checkpoint_sha256"] = _state_dict_sha256(codec)

    evaluation = evaluate_arms(
        arms, validation_coefficients, representations, canonical_coords, cfg
    )
    config_payload = asdict(cfg)
    config_sha = hashlib.sha256(
        json.dumps(config_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    result = {
        "schema_version": 1,
        "experiment": "canonical_latent_e2_measure_aware_analytic",
        "config": config_payload,
        "config_sha256": config_sha,
        "initial_checkpoint_sha256": initial_sha,
        "state_split": {
            "train_count": cfg.train_states,
            "validation_count": cfg.validation_states,
            "coefficient_generator": "independent fixed torch generators",
            "cross_split_identity_overlap": 0,
        },
        "representations": {name: asdict(value) for name, value in representations.items()},
        "training": training,
        "evaluation": evaluation,
    }
    result_path = run_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result["result_path"] = str(result_path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the validation-only, no-operator canonical latent E2 benchmark"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=BenchmarkConfig.epochs)
    parser.add_argument("--train-states", type=int, default=BenchmarkConfig.train_states)
    parser.add_argument("--validation-states", type=int, default=BenchmarkConfig.validation_states)
    parser.add_argument("--batch-size", type=int, default=BenchmarkConfig.batch_size)
    parser.add_argument("--print-json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = BenchmarkConfig(
        epochs=args.epochs,
        train_states=args.train_states,
        validation_states=args.validation_states,
        batch_size=args.batch_size,
    )
    result = run_benchmark(cfg, run_dir=args.run_dir)
    summary = {
        "status": result["evaluation"]["status"],
        "gates": result["evaluation"]["gates"],
        "result_path": result["result_path"],
    }
    if args.print_json:
        print(json.dumps(summary, sort_keys=True))
    else:
        print(f"status={summary['status']} result={summary['result_path']}")


if __name__ == "__main__":
    main()
