#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

if __package__:
    from scripts.run_canonical_latent_e2_benchmark import (
        BenchmarkConfig,
        CanonicalCodec,
        Representation,
        _representations,
        evaluate_field,
        inverse_distance_interpolate,
        representation_points,
        state_coefficients,
    )
else:
    from run_canonical_latent_e2_benchmark import (  # type: ignore[no-redef]
        BenchmarkConfig,
        CanonicalCodec,
        Representation,
        _representations,
        evaluate_field,
        inverse_distance_interpolate,
        representation_points,
        state_coefficients,
    )
from ups.eval.latent_qualification import (
    discretization_mismatch_report,
    effective_rank,
    global_nrmse,
)
from ups.io import AnyPointDecoder, AnyPointDecoderConfig


@dataclass(frozen=True)
class CapacityRung:
    name: str
    latent_len: int
    supernodes: int


@dataclass(frozen=True)
class CapacityLadderConfig:
    seed: int = 17
    train_states: int = 128
    validation_states: int = 24
    epochs: int = 120
    batch_size: int = 16
    learning_rate: float = 2e-3
    latent_dim: int = 32
    hidden_dim: int = 32
    supernode_neighbors: int = 16
    transformer_layers: int = 1
    alignment_weight: float = 0.10
    train_low_resolution: int = 10
    train_high_resolution: int = 14
    validation_resolution: int = 18
    canonical_query_resolution: int = 18
    max_interpolation_baseline_ratio: float = 2.0
    high_frequency_radius: float = 3.0
    rungs: tuple[CapacityRung, ...] = (
        CapacityRung("C8", 8, 24),
        CapacityRung("C16", 16, 48),
        CapacityRung("C32", 32, 96),
    )

    def __post_init__(self) -> None:
        if self.train_states < self.batch_size or self.validation_states < 4:
            raise ValueError("capacity ladder requires one train batch and four validation states")
        if self.train_states % self.batch_size:
            raise ValueError("train_states must be divisible by batch_size")
        if not self.rungs:
            raise ValueError("at least one capacity rung is required")
        names = [rung.name for rung in self.rungs]
        if len(names) != len(set(names)):
            raise ValueError("capacity rung names must be unique")
        if any(rung.latent_len <= 0 or rung.supernodes <= 0 for rung in self.rungs):
            raise ValueError("capacity rung sizes must be positive")

    def benchmark_config(self, rung: CapacityRung) -> BenchmarkConfig:
        return BenchmarkConfig(
            encoder_kind="perceiver",
            seed=self.seed,
            train_states=self.train_states,
            validation_states=self.validation_states,
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            latent_len=rung.latent_len,
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            supernodes=rung.supernodes,
            supernode_neighbors=self.supernode_neighbors,
            transformer_layers=self.transformer_layers,
            alignment_weight=self.alignment_weight,
            train_low_resolution=self.train_low_resolution,
            train_high_resolution=self.train_high_resolution,
            validation_resolution=self.validation_resolution,
            canonical_query_resolution=self.canonical_query_resolution,
            max_interpolation_baseline_ratio=self.max_interpolation_baseline_ratio,
        )


def _coordinate_features(
    coords: torch.Tensor, frequencies: tuple[float, ...] = (1.0, 2.0, 4.0)
) -> torch.Tensor:
    features = [coords]
    for frequency in frequencies:
        scaled = 2.0 * torch.pi * frequency * coords
        features.extend((torch.sin(scaled), torch.cos(scaled)))
    return torch.cat(features, dim=-1)


class DirectPointCodec(nn.Module):
    """No-compression learned ceiling using every physical sample as a token."""

    def __init__(self, cfg: CapacityLadderConfig):
        super().__init__()
        coord_features = 2 * (1 + 2 * 3)
        self.encoder = nn.Sequential(
            nn.Linear(1 + coord_features + 1, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.latent_dim),
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
        if values.shape[:2] != coords.shape[:2] or measure.shape != (*coords.shape[:2], 1):
            raise ValueError("direct-point values, coordinates, and measure must share nodes")
        if not torch.isfinite(measure).all() or torch.any(measure <= 0):
            raise ValueError("direct-point measure must be finite and strictly positive")
        normalized_measure = measure / measure.sum(dim=1, keepdim=True)
        relative_measure = normalized_measure * coords.shape[1]
        return self.encoder(
            torch.cat([values, _coordinate_features(coords), relative_measure], dim=-1)
        )

    def decode(self, latent: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        return self.decoder(query, latent)["u"]


def _state_dict_sha256(module: nn.Module) -> str:
    buffer = io.BytesIO()
    torch.save(module.state_dict(), buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _codec_loss(
    codec: nn.Module,
    coefficients: torch.Tensor,
    representations: tuple[Representation, Representation],
    canonical_coords: torch.Tensor,
    *,
    alignment_weight: float,
) -> torch.Tensor:
    latents = []
    reconstruction_losses = []
    canonical_target = evaluate_field(coefficients, canonical_coords)
    for representation in representations:
        coords, measure = representation_points(representation, batch=coefficients.shape[0])
        values = evaluate_field(coefficients, coords)
        latent = codec.encode(values, coords, measure)  # type: ignore[attr-defined]
        latents.append(latent)
        native_prediction = codec.decode(latent, coords)  # type: ignore[attr-defined]
        canonical_prediction = codec.decode(latent, canonical_coords)  # type: ignore[attr-defined]
        native_scale = values.square().mean().clamp_min(1e-8)
        canonical_scale = canonical_target.square().mean().clamp_min(1e-8)
        reconstruction_losses.append(
            0.5 * F.mse_loss(native_prediction, values) / native_scale
            + 0.5 * F.mse_loss(canonical_prediction, canonical_target) / canonical_scale
        )
    loss = sum(reconstruction_losses) / 2.0
    if alignment_weight:
        if latents[0].shape != latents[1].shape:
            raise ValueError("latent alignment requires the same token shape")
        flattened = [latent.flatten(1) for latent in latents]
        normalized = [F.layer_norm(latent, (latent.shape[1],)) for latent in flattened]
        loss = loss + alignment_weight * F.mse_loss(normalized[0], normalized[1])
    return loss


def _linear_slope(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    tensor = torch.tensor(values, dtype=torch.float64)
    x = torch.arange(tensor.numel(), dtype=torch.float64)
    centered_x = x - x.mean()
    return float((centered_x * (tensor - tensor.mean())).sum() / centered_x.square().sum())


def train_specialist(
    codec: nn.Module,
    coefficients: torch.Tensor,
    representations: tuple[Representation, Representation],
    canonical_coords: torch.Tensor,
    cfg: CapacityLadderConfig,
    *,
    alignment_weight: float,
) -> dict[str, Any]:
    codec.train()
    optimizer = torch.optim.AdamW(codec.parameters(), lr=cfg.learning_rate, weight_decay=1e-6)
    generator = torch.Generator().manual_seed(cfg.seed)
    history = []
    for _ in range(cfg.epochs):
        permutation = torch.randperm(coefficients.shape[0], generator=generator)
        epoch_loss = 0.0
        for start in range(0, coefficients.shape[0], cfg.batch_size):
            batch = coefficients[permutation[start : start + cfg.batch_size]]
            query = canonical_coords.expand(batch.shape[0], -1, -1)
            loss = _codec_loss(
                codec,
                batch,
                representations,
                query,
                alignment_weight=alignment_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(codec.parameters(), 1.0)
            optimizer.step()
            epoch_loss += float(loss.detach()) * batch.shape[0]
        history.append(epoch_loss / coefficients.shape[0])
    trailing = history[-min(10, len(history)) :]
    minimum = min(history)
    return {
        "epochs": cfg.epochs,
        "optimizer_updates": cfg.epochs * coefficients.shape[0] // cfg.batch_size,
        "scheduled_source_examples": cfg.epochs * coefficients.shape[0] * 2,
        "initial_loss": history[0],
        "final_loss": history[-1],
        "minimum_loss": minimum,
        "final_to_minimum_loss_ratio": history[-1] / minimum,
        "trailing_10_epoch_loss_slope": _linear_slope(trailing),
    }


def high_frequency_spectral_report(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    resolution: int,
    minimum_radius: float,
) -> dict[str, float]:
    prediction_grid = prediction[..., 0].reshape(prediction.shape[0], resolution, resolution)
    target_grid = target[..., 0].reshape(target.shape[0], resolution, resolution)
    prediction_fft = torch.fft.rfft2(prediction_grid, norm="ortho")
    target_fft = torch.fft.rfft2(target_grid, norm="ortho")
    frequency_x = torch.fft.fftfreq(resolution, d=1.0 / resolution)
    frequency_y = torch.fft.rfftfreq(resolution, d=1.0 / resolution)
    radius = torch.sqrt(frequency_x[:, None].square() + frequency_y[None, :].square())
    mask = radius >= minimum_radius
    error_energy = (prediction_fft - target_fft).abs().square()[:, mask].sum()
    target_energy = target_fft.abs().square()[:, mask].sum().clamp_min(1e-12)
    prediction_amplitude = prediction_fft.abs()[:, mask].sum()
    target_amplitude = target_fft.abs()[:, mask].sum().clamp_min(1e-12)
    return {
        "nrmse": float(torch.sqrt(error_energy / target_energy)),
        "amplitude_ratio": float(prediction_amplitude / target_amplitude),
    }


@torch.no_grad()
def evaluate_specialist(
    codec: nn.Module,
    coefficients: torch.Tensor,
    family: str,
    representations: dict[str, Representation],
    canonical_coords: torch.Tensor,
    cfg: CapacityLadderConfig,
) -> dict[str, Any]:
    codec.eval()
    low = cfg.train_low_resolution
    high = cfg.train_high_resolution
    validation = cfg.validation_resolution
    names = (
        (f"grid_{low}", f"grid_{high}", f"grid_{validation}")
        if family == "grid"
        else (f"mesh_{low}_a", f"mesh_{high}_a", f"mesh_{validation}_a")
    )
    query = canonical_coords.expand(coefficients.shape[0], -1, -1)
    target = evaluate_field(coefficients, query)
    predictions = {}
    latents = {}
    samples = {}
    for label, name in zip(("low", "high", "validation"), names):
        coords, measure = representation_points(representations[name], batch=coefficients.shape[0])
        values = evaluate_field(coefficients, coords)
        latent = codec.encode(values, coords, measure)  # type: ignore[attr-defined]
        latents[label] = latent
        predictions[label] = codec.decode(latent, query)  # type: ignore[attr-defined]
        samples[label] = (coords, values)

    high_coords, high_values = samples["high"]
    interpolation = inverse_distance_interpolate(high_values, high_coords, query)
    errors = {label: global_nrmse(prediction, target) for label, prediction in predictions.items()}
    interpolation_error = global_nrmse(interpolation, target)
    mismatch = discretization_mismatch_report(predictions, target)
    prediction = predictions["high"]
    target_scale = target.square().mean().sqrt().clamp_min(1e-12)
    spectral = high_frequency_spectral_report(
        prediction,
        target,
        resolution=cfg.canonical_query_resolution,
        minimum_radius=cfg.high_frequency_radius,
    )
    return {
        "canonical_query_nrmse": errors,
        "inverse_distance_interpolation_nrmse": interpolation_error,
        "high_to_interpolation_ratio": errors["high"] / interpolation_error,
        "absolute_reconstruction_pass": (
            errors["high"] <= cfg.max_interpolation_baseline_ratio * interpolation_error
        ),
        "validation_resolution_stability_pass": errors["validation"] <= 1.10 * errors["high"],
        "discretization_mismatch": mismatch,
        "high_frequency_spectral": spectral,
        "prediction_to_target_std_ratio": float(prediction.std() / target.std().clamp_min(1e-12)),
        "normalized_mean_bias": float((prediction - target).mean().abs() / target_scale),
        "effective_latent_rank": effective_rank(latents["high"]),
        "high_source_token_count": latents["high"].shape[1],
    }


def _family_pairs(
    representations: dict[str, Representation], cfg: CapacityLadderConfig
) -> dict[str, tuple[Representation, Representation]]:
    low = cfg.train_low_resolution
    high = cfg.train_high_resolution
    return {
        "grid": (representations[f"grid_{low}"], representations[f"grid_{high}"]),
        "mesh": (representations[f"mesh_{low}_a"], representations[f"mesh_{high}_a"]),
    }


def run_capacity_ladder(cfg: CapacityLadderConfig, *, run_dir: Path) -> dict[str, Any]:
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    torch.set_num_threads(1)
    run_dir.mkdir(parents=True, exist_ok=True)
    base = cfg.benchmark_config(cfg.rungs[0])
    representations = _representations(base)
    pairs = _family_pairs(representations, cfg)
    train_coefficients = state_coefficients(cfg.train_states, seed=cfg.seed)
    validation_coefficients = state_coefficients(cfg.validation_states, seed=cfg.seed + 10_000)
    canonical = Representation("canonical_queries", cfg.canonical_query_resolution, 0.0, 0.0)
    canonical_coords, _ = representation_points(canonical, batch=1)

    compressed_results = {}
    for rung in cfg.rungs:
        torch.manual_seed(cfg.seed)
        initial = CanonicalCodec(cfg.benchmark_config(rung))
        arms = {family: copy.deepcopy(initial) for family in ("grid", "mesh")}
        rung_result = {
            "rung": asdict(rung),
            "initial_checkpoint_sha256": _state_dict_sha256(initial),
            "architecture": {
                "encoder_parameters": sum(
                    parameter.numel() for parameter in initial.encoder.parameters()
                ),
                "decoder_parameters": sum(
                    parameter.numel() for parameter in initial.decoder.parameters()
                ),
                "total_parameters": sum(parameter.numel() for parameter in initial.parameters()),
            },
            "families": {},
        }
        for family, codec in arms.items():
            training = train_specialist(
                codec,
                train_coefficients,
                pairs[family],
                canonical_coords,
                cfg,
                alignment_weight=cfg.alignment_weight,
            )
            checkpoint = run_dir / f"{rung.name}_{family}_codec.pt"
            torch.save(codec.state_dict(), checkpoint)
            training["checkpoint_sha256"] = _state_dict_sha256(codec)
            rung_result["families"][family] = {
                "training": training,
                "evaluation": evaluate_specialist(
                    codec,
                    validation_coefficients,
                    family,
                    representations,
                    canonical_coords,
                    cfg,
                ),
            }
        compressed_results[rung.name] = rung_result

    torch.manual_seed(cfg.seed)
    direct_initial = DirectPointCodec(cfg)
    direct_results = {
        "initial_checkpoint_sha256": _state_dict_sha256(direct_initial),
        "architecture": {
            "encoder_parameters": sum(
                parameter.numel() for parameter in direct_initial.encoder.parameters()
            ),
            "decoder_parameters": sum(
                parameter.numel() for parameter in direct_initial.decoder.parameters()
            ),
            "total_parameters": sum(parameter.numel() for parameter in direct_initial.parameters()),
            "compression": "none",
        },
        "families": {},
    }
    for family in ("grid", "mesh"):
        codec = copy.deepcopy(direct_initial)
        training = train_specialist(
            codec,
            train_coefficients,
            pairs[family],
            canonical_coords,
            cfg,
            alignment_weight=0.0,
        )
        checkpoint = run_dir / f"direct_{family}_codec.pt"
        torch.save(codec.state_dict(), checkpoint)
        training["checkpoint_sha256"] = _state_dict_sha256(codec)
        direct_results["families"][family] = {
            "training": training,
            "evaluation": evaluate_specialist(
                codec,
                validation_coefficients,
                family,
                representations,
                canonical_coords,
                cfg,
            ),
        }

    passing_rungs = [
        rung.name
        for rung in cfg.rungs
        if all(
            compressed_results[rung.name]["families"][family]["evaluation"][
                "absolute_reconstruction_pass"
            ]
            for family in ("grid", "mesh")
        )
    ]
    direct_pass = all(
        direct_results["families"][family]["evaluation"]["absolute_reconstruction_pass"]
        for family in ("grid", "mesh")
    )
    if passing_rungs:
        causal_decision = {
            "classification": "fixed_latent_capacity_causal",
            "smallest_passing_rung": passing_rungs[0],
            "next_move": "retest one shared codec at the smallest passing capacity",
        }
    elif direct_pass:
        causal_decision = {
            "classification": "compression_tokenization_causal",
            "smallest_passing_rung": None,
            "next_move": "test one high-fidelity scientific tokenizer under the same ceiling",
        }
    else:
        causal_decision = {
            "classification": "decoder_objective_or_schedule_blocker",
            "smallest_passing_rung": None,
            "next_move": "pause encoders and isolate decoder, objective, and schedule",
        }

    config_payload = asdict(cfg)
    config_sha = hashlib.sha256(
        json.dumps(config_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    result = {
        "schema_version": 1,
        "experiment": "canonical_latent_e4_capacity_identifiability",
        "config": config_payload,
        "config_sha256": config_sha,
        "state_split": {
            "train_count": cfg.train_states,
            "validation_count": cfg.validation_states,
            "cross_split_identity_overlap": 0,
        },
        "compressed_ladder": compressed_results,
        "direct_point_ceiling": direct_results,
        "causal_decision": causal_decision,
        "boundary": {
            "operator_instantiated": False,
            "heldout_reads": 0,
            "representation_label_model_inputs": False,
            "task_label_model_inputs": False,
            "provider_calls": 0,
        },
    }
    result_path = run_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result["result_path"] = str(result_path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the validation-only canonical latent E4 capacity ladder"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=CapacityLadderConfig.epochs)
    parser.add_argument("--train-states", type=int, default=CapacityLadderConfig.train_states)
    parser.add_argument(
        "--validation-states", type=int, default=CapacityLadderConfig.validation_states
    )
    parser.add_argument("--batch-size", type=int, default=CapacityLadderConfig.batch_size)
    parser.add_argument("--print-json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = CapacityLadderConfig(
        epochs=args.epochs,
        train_states=args.train_states,
        validation_states=args.validation_states,
        batch_size=args.batch_size,
    )
    result = run_capacity_ladder(cfg, run_dir=args.run_dir)
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
