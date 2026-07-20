#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
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
        train_arm,
    )
    from scripts.run_canonical_latent_e4_capacity_ladder import (
        CapacityLadderConfig,
        _family_pairs,
        _linear_slope,
        _state_dict_sha256,
        evaluate_specialist,
        high_frequency_spectral_report,
    )
    from scripts.run_canonical_latent_e5_decoder_locality import (
        DecoderLocalityConfig,
        LocalIntegralDecoder,
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
        train_arm,
    )
    from run_canonical_latent_e4_capacity_ladder import (  # type: ignore[no-redef]
        CapacityLadderConfig,
        _family_pairs,
        _linear_slope,
        _state_dict_sha256,
        evaluate_specialist,
        high_frequency_spectral_report,
    )
    from run_canonical_latent_e5_decoder_locality import (  # type: ignore[no-redef]
        DecoderLocalityConfig,
        LocalIntegralDecoder,
    )
from ups.eval.latent_qualification import (
    discretization_mismatch_report,
    effective_rank,
    global_nrmse,
)
from ups.io import RegionalInteractionEncoder, RegionalInteractionEncoderConfig


@dataclass(frozen=True)
class CompressedLocalityConfig:
    seed: int = 17
    train_states: int = 128
    validation_states: int = 24
    epochs: int = 120
    batch_size: int = 16
    learning_rate: float = 2e-3
    latent_len: int = 8
    latent_dim: int = 32
    hidden_dim: int = 32
    physical_neighbors: int = 16
    processor_neighbors: tuple[int, ...] = (2, 4, 7)
    alignment_weight: float = 0.10
    train_low_resolution: int = 10
    train_high_resolution: int = 14
    validation_resolution: int = 18
    canonical_query_resolution: int = 18
    max_interpolation_baseline_ratio: float = 2.0
    high_frequency_radius: float = 3.0
    local_support_radius: float = 0.47
    local_max_neighbors: int = 8
    covering_radius_margin: float = 1.05
    locality_spectral_improvement: float = 0.25
    helpful_improvement: float = 0.10
    invariance_atol: float = 1e-6

    def __post_init__(self) -> None:
        if self.train_states < self.batch_size or self.validation_states < 4:
            raise ValueError("E6 requires one train batch and four validation states")
        if self.train_states % self.batch_size:
            raise ValueError("train_states must be divisible by batch_size")
        if self.latent_len != 8:
            raise ValueError("E6 freezes the smallest eight-token regional latent")
        if self.local_max_neighbors != self.latent_len:
            raise ValueError("E6 must make all compressed tokens eligible for physical support")

    def benchmark_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            encoder_kind="regional_interaction",
            seed=self.seed,
            train_states=self.train_states,
            validation_states=self.validation_states,
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            latent_len=self.latent_len,
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            supernodes=24,
            supernode_neighbors=self.physical_neighbors,
            transformer_layers=1,
            alignment_weight=self.alignment_weight,
            train_low_resolution=self.train_low_resolution,
            train_high_resolution=self.train_high_resolution,
            validation_resolution=self.validation_resolution,
            canonical_query_resolution=self.canonical_query_resolution,
            max_interpolation_baseline_ratio=self.max_interpolation_baseline_ratio,
        )

    def capacity_config(self) -> CapacityLadderConfig:
        return CapacityLadderConfig(
            seed=self.seed,
            train_states=self.train_states,
            validation_states=self.validation_states,
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            alignment_weight=self.alignment_weight,
            train_low_resolution=self.train_low_resolution,
            train_high_resolution=self.train_high_resolution,
            validation_resolution=self.validation_resolution,
            canonical_query_resolution=self.canonical_query_resolution,
            max_interpolation_baseline_ratio=self.max_interpolation_baseline_ratio,
            high_frequency_radius=self.high_frequency_radius,
        )

    def local_decoder_config(self) -> DecoderLocalityConfig:
        return DecoderLocalityConfig(
            seed=self.seed,
            train_states=self.train_states,
            validation_states=self.validation_states,
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            train_low_resolution=self.train_low_resolution,
            train_high_resolution=self.train_high_resolution,
            validation_resolution=self.validation_resolution,
            canonical_query_resolution=self.canonical_query_resolution,
            max_interpolation_baseline_ratio=self.max_interpolation_baseline_ratio,
            high_frequency_radius=self.high_frequency_radius,
            local_support_radius=self.local_support_radius,
            local_max_neighbors=self.local_max_neighbors,
            locality_spectral_improvement=self.locality_spectral_improvement,
            helpful_improvement=self.helpful_improvement,
            invariance_atol=self.invariance_atol,
        )


class RegionalLocalCodec(nn.Module):
    """Eight spatial regional tokens decoded without original-source access."""

    def __init__(self, cfg: CompressedLocalityConfig):
        super().__init__()
        self.encoder = RegionalInteractionEncoder(
            RegionalInteractionEncoderConfig(
                latent_len=cfg.latent_len,
                latent_dim=cfg.latent_dim,
                hidden_dim=cfg.hidden_dim,
                coord_dim=2,
                field_channels={"u": 1},
                physical_neighbors=cfg.physical_neighbors,
                processor_neighbors=cfg.processor_neighbors,
                require_measure=True,
            )
        )
        self.decoder = LocalIntegralDecoder(cfg.local_decoder_config())

    def encode(
        self, values: torch.Tensor, coords: torch.Tensor, measure: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.encoder.forward_with_geometry({"u": values}, coords, geom={"measure": measure})

    def decode(
        self,
        latent: torch.Tensor,
        query: torch.Tensor,
        latent_coords: torch.Tensor,
        latent_measure: torch.Tensor,
    ) -> torch.Tensor:
        return self.decoder(latent, latent_coords, latent_measure, query)


def materialize_covering_radius(
    encoder: RegionalInteractionEncoder,
    representations: dict[str, Representation],
    canonical_coords: torch.Tensor,
) -> dict[str, Any]:
    by_representation = {}
    for name, representation in representations.items():
        coords, measure = representation_points(representation, batch=1)
        order = encoder._canonical_point_order(coords)
        ordered_coords = encoder._gather_nodes(coords, order)
        ordered_measure = encoder._gather_nodes(measure, order)
        regional_indices = encoder._farthest_point_indices(ordered_coords, ordered_measure)
        regional_coords = encoder._gather_nodes(ordered_coords, regional_indices)
        slot_order = encoder._assign_region_slots(regional_coords, ordered_coords)
        regional_coords = encoder._gather_nodes(regional_coords, slot_order)
        nearest = torch.cdist(canonical_coords, regional_coords).amin(dim=-1)
        by_representation[name] = {
            "maximum": float(nearest.max()),
            "p99": float(torch.quantile(nearest, 0.99)),
            "mean": float(nearest.mean()),
        }
    maximum = max(value["maximum"] for value in by_representation.values())
    return {"by_representation": by_representation, "maximum": maximum}


def _local_loss(
    codec: RegionalLocalCodec,
    coefficients: torch.Tensor,
    representations: tuple[Representation, Representation],
    canonical_coords: torch.Tensor,
    *,
    alignment_weight: float,
) -> torch.Tensor:
    canonical_target = evaluate_field(coefficients, canonical_coords)
    losses = []
    latents = []
    for representation in representations:
        coords, measure = representation_points(representation, batch=coefficients.shape[0])
        values = evaluate_field(coefficients, coords)
        latent, latent_coords, latent_measure = codec.encode(values, coords, measure)
        latents.append(latent)
        native_prediction = codec.decode(latent, coords, latent_coords, latent_measure)
        canonical_prediction = codec.decode(latent, canonical_coords, latent_coords, latent_measure)
        native_scale = values.square().mean().clamp_min(1e-8)
        canonical_scale = canonical_target.square().mean().clamp_min(1e-8)
        losses.append(
            0.5 * F.mse_loss(native_prediction, values) / native_scale
            + 0.5 * F.mse_loss(canonical_prediction, canonical_target) / canonical_scale
        )
    loss = sum(losses) / 2.0
    if alignment_weight:
        flattened = [latent.flatten(1) for latent in latents]
        normalized = [F.layer_norm(value, (value.shape[1],)) for value in flattened]
        loss = loss + alignment_weight * F.mse_loss(normalized[0], normalized[1])
    return loss


def train_local_specialist(
    codec: RegionalLocalCodec,
    coefficients: torch.Tensor,
    representations: tuple[Representation, Representation],
    canonical_coords: torch.Tensor,
    cfg: CompressedLocalityConfig,
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
            loss = _local_loss(
                codec,
                batch,
                representations,
                query,
                alignment_weight=cfg.alignment_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(codec.parameters(), 1.0)
            optimizer.step()
            epoch_loss += float(loss.detach()) * batch.shape[0]
        history.append(epoch_loss / coefficients.shape[0])
    minimum = min(history)
    return {
        "epochs": cfg.epochs,
        "optimizer_updates": cfg.epochs * coefficients.shape[0] // cfg.batch_size,
        "scheduled_source_examples": cfg.epochs * coefficients.shape[0] * 2,
        "initial_loss": history[0],
        "final_loss": history[-1],
        "minimum_loss": minimum,
        "final_to_minimum_loss_ratio": history[-1] / minimum,
        "trailing_10_epoch_loss_slope": _linear_slope(history[-min(10, len(history)) :]),
    }


@torch.no_grad()
def _source_order_error(
    codec: RegionalLocalCodec,
    latent: torch.Tensor,
    latent_coords: torch.Tensor,
    latent_measure: torch.Tensor,
    query: torch.Tensor,
    *,
    seed: int,
) -> float:
    original = codec.decode(latent, query, latent_coords, latent_measure)
    permutation = torch.randperm(latent.shape[1], generator=torch.Generator().manual_seed(seed))
    permuted = codec.decode(
        latent[:, permutation],
        query,
        latent_coords[:, permutation],
        latent_measure[:, permutation],
    )
    return float((original - permuted).abs().max())


@torch.no_grad()
def evaluate_local_specialist(
    codec: RegionalLocalCodec,
    coefficients: torch.Tensor,
    family: str,
    representations: dict[str, Representation],
    canonical_coords: torch.Tensor,
    cfg: CompressedLocalityConfig,
) -> dict[str, Any]:
    codec.eval()
    names = (
        (
            f"grid_{cfg.train_low_resolution}",
            f"grid_{cfg.train_high_resolution}",
            f"grid_{cfg.validation_resolution}",
        )
        if family == "grid"
        else (
            f"mesh_{cfg.train_low_resolution}_a",
            f"mesh_{cfg.train_high_resolution}_a",
            f"mesh_{cfg.validation_resolution}_a",
        )
    )
    query = canonical_coords.expand(coefficients.shape[0], -1, -1)
    target = evaluate_field(coefficients, query)
    predictions: dict[str, torch.Tensor] = {}
    latents: dict[str, torch.Tensor] = {}
    geometry: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    samples: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for label, name in zip(("low", "high", "validation"), names):
        coords, measure = representation_points(representations[name], batch=coefficients.shape[0])
        values = evaluate_field(coefficients, coords)
        latent, latent_coords, latent_measure = codec.encode(values, coords, measure)
        latents[label] = latent
        geometry[label] = (latent_coords, latent_measure)
        predictions[label] = codec.decode(latent, query, latent_coords, latent_measure)
        samples[label] = (coords, measure, values)

    high_coords, _, high_values = samples["high"]
    high_latent_coords, high_latent_measure = geometry["high"]
    interpolation = inverse_distance_interpolate(high_values, high_coords, query)
    errors = {label: global_nrmse(prediction, target) for label, prediction in predictions.items()}
    interpolation_error = global_nrmse(interpolation, target)
    prediction = predictions["high"]
    target_scale = target.square().mean().sqrt().clamp_min(1e-12)
    mass_sum_error = float((high_latent_measure.sum(dim=1) - 1.0).abs().max())
    return {
        "canonical_query_nrmse": errors,
        "inverse_distance_interpolation_nrmse": interpolation_error,
        "high_to_interpolation_ratio": errors["high"] / interpolation_error,
        "absolute_reconstruction_pass": (
            errors["high"] <= cfg.max_interpolation_baseline_ratio * interpolation_error
        ),
        "validation_resolution_stability_pass": errors["validation"] <= 1.10 * errors["high"],
        "discretization_mismatch": discretization_mismatch_report(predictions, target),
        "high_frequency_spectral": high_frequency_spectral_report(
            prediction,
            target,
            resolution=cfg.canonical_query_resolution,
            minimum_radius=cfg.high_frequency_radius,
        ),
        "prediction_to_target_std_ratio": float(prediction.std() / target.std().clamp_min(1e-12)),
        "normalized_mean_bias": float((prediction - target).mean().abs() / target_scale),
        "effective_latent_rank": effective_rank(latents["high"]),
        "high_source_node_count": high_coords.shape[1],
        "compressed_token_count": latents["high"].shape[1],
        "compression_ratio": high_coords.shape[1] / latents["high"].shape[1],
        "regional_measure": {
            "minimum": float(high_latent_measure.min()),
            "maximum": float(high_latent_measure.max()),
            "sum_max_abs_error": mass_sum_error,
        },
        "neighbor_count": codec.decoder.neighbor_report(high_latent_coords, query),
        "source_order_max_abs_error": _source_order_error(
            codec,
            latents["high"],
            high_latent_coords,
            high_latent_measure,
            query,
            seed=cfg.seed + 1100,
        ),
    }


@torch.no_grad()
def paired_anchor_report(
    codec: RegionalLocalCodec,
    coefficients: torch.Tensor,
    representations: dict[str, Representation],
    cfg: CompressedLocalityConfig,
) -> dict[str, float]:
    coordinates = {}
    for family, name in (
        ("grid", f"grid_{cfg.train_high_resolution}"),
        ("mesh", f"mesh_{cfg.train_high_resolution}_a"),
    ):
        coords, measure = representation_points(representations[name], batch=coefficients.shape[0])
        values = evaluate_field(coefficients, coords)
        _, latent_coords, _ = codec.encode(values, coords, measure)
        coordinates[family] = latent_coords
    distances = (coordinates["grid"] - coordinates["mesh"]).square().sum(-1).sqrt()
    return {
        "matched_slot_mean_distance": float(distances.mean()),
        "maximum": float(distances.max()),
    }


def _decision(result: dict[str, Any], cfg: CompressedLocalityConfig) -> dict[str, Any]:
    comparisons = {}
    for family in ("grid", "mesh"):
        control = result["arms"]["global_control"]["families"][family]["evaluation"]
        local = result["arms"]["local_integral"]["families"][family]["evaluation"]
        comparisons[family] = {
            "high_nrmse_relative_improvement": 1.0
            - local["canonical_query_nrmse"]["high"] / control["canonical_query_nrmse"]["high"],
            "high_frequency_nrmse_relative_improvement": 1.0
            - local["high_frequency_spectral"]["nrmse"]
            / control["high_frequency_spectral"]["nrmse"],
            "absolute_reconstruction_pass": local["absolute_reconstruction_pass"],
            "validation_resolution_stability_pass": local["validation_resolution_stability_pass"],
            "source_order_invariance_pass": (
                local["source_order_max_abs_error"] <= cfg.invariance_atol
            ),
            "regional_measure_pass": (
                local["regional_measure"]["minimum"] > 0
                and local["regional_measure"]["sum_max_abs_error"] <= 1e-6
            ),
            "physical_support_pass": local["neighbor_count"]["minimum"] >= 1,
        }
    parameter_gate = (
        result["arms"]["local_integral"]["architecture"]["decoder_parameters"]
        <= result["arms"]["global_control"]["architecture"]["decoder_parameters"]
    )
    causal = parameter_gate and all(
        comparison["absolute_reconstruction_pass"]
        and comparison["validation_resolution_stability_pass"]
        and comparison["source_order_invariance_pass"]
        and comparison["regional_measure_pass"]
        and comparison["physical_support_pass"]
        and comparison["high_frequency_nrmse_relative_improvement"]
        >= cfg.locality_spectral_improvement
        for comparison in comparisons.values()
    )
    helpful = all(
        comparison["high_nrmse_relative_improvement"] >= cfg.helpful_improvement
        and comparison["high_frequency_nrmse_relative_improvement"] >= cfg.helpful_improvement
        and comparison["validation_resolution_stability_pass"]
        and comparison["source_order_invariance_pass"]
        and comparison["regional_measure_pass"]
        and comparison["physical_support_pass"]
        for comparison in comparisons.values()
    )
    if causal:
        classification = "compressed_spatial_latent_qualified"
        next_move = "run one shared grid-mesh codec qualification with the frozen spatial tokens"
    elif helpful:
        classification = "compressed_locality_helpful_but_insufficient"
        next_move = "run one preregistered anchor-count identifiability comparison"
    else:
        classification = "compressed_locality_not_qualified"
        next_move = "close the compact regional codec and reconsider the representation contract"
    return {
        "classification": classification,
        "parameter_budget_pass": parameter_gate,
        "family_comparisons": comparisons,
        "next_move": next_move,
    }


def run_compressed_locality(cfg: CompressedLocalityConfig, *, run_dir: Path) -> dict[str, Any]:
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    torch.set_num_threads(1)
    run_dir.mkdir(parents=True, exist_ok=True)
    benchmark_cfg = cfg.benchmark_config()
    capacity_cfg = cfg.capacity_config()
    representations = _representations(benchmark_cfg)
    pairs = _family_pairs(representations, capacity_cfg)
    train_coefficients = state_coefficients(cfg.train_states, seed=cfg.seed)
    validation_coefficients = state_coefficients(cfg.validation_states, seed=cfg.seed + 10_000)
    canonical = Representation("canonical_queries", cfg.canonical_query_resolution, 0.0, 0.0)
    canonical_coords, _ = representation_points(canonical, batch=1)

    torch.manual_seed(cfg.seed)
    global_initial = CanonicalCodec(benchmark_cfg)
    torch.manual_seed(cfg.seed)
    local_initial = RegionalLocalCodec(cfg)
    global_encoder_hash = _state_dict_sha256(global_initial.encoder)
    local_encoder_hash = _state_dict_sha256(local_initial.encoder)
    if global_encoder_hash != local_encoder_hash:
        raise RuntimeError("E6 arms must begin with identical regional encoders")

    covering = materialize_covering_radius(
        global_initial.encoder, representations, canonical_coords
    )
    if cfg.local_support_radius < cfg.covering_radius_margin * covering["maximum"]:
        raise ValueError("frozen local radius does not satisfy preregistered anchor coverage")

    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "canonical_latent_e6_compressed_locality",
        "config": asdict(cfg),
        "state_split": {
            "train_count": cfg.train_states,
            "validation_count": cfg.validation_states,
            "cross_split_identity_overlap": 0,
        },
        "initial_encoder_sha256": global_encoder_hash,
        "anchor_covering_radius": covering,
        "arms": {},
        "boundary": {
            "operator_instantiated": False,
            "heldout_reads": 0,
            "representation_label_model_inputs": False,
            "task_label_model_inputs": False,
            "provider_calls": 0,
            "routing_paths": 0,
            "original_source_features_available_to_decoder": False,
        },
    }
    result["config_sha256"] = hashlib.sha256(
        json.dumps(result["config"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    for arm_name, initial in (
        ("global_control", global_initial),
        ("local_integral", local_initial),
    ):
        arm = {
            "initial_checkpoint_sha256": _state_dict_sha256(initial),
            "architecture": {
                "encoder_parameters": sum(p.numel() for p in initial.encoder.parameters()),
                "decoder_parameters": sum(p.numel() for p in initial.decoder.parameters()),
                "total_parameters": sum(p.numel() for p in initial.parameters()),
                "latent_tokens": cfg.latent_len,
                "latent_dim": cfg.latent_dim,
                "hidden_dim": cfg.hidden_dim,
                "source_bypass": False,
            },
            "families": {},
        }
        for family in ("grid", "mesh"):
            codec = copy.deepcopy(initial)
            if arm_name == "global_control":
                training = train_arm(
                    codec,
                    train_coefficients,
                    (pairs[family],),
                    canonical_coords,
                    benchmark_cfg,
                )
                evaluation = evaluate_specialist(
                    codec,
                    validation_coefficients,
                    family,
                    representations,
                    canonical_coords,
                    capacity_cfg,
                )
            else:
                training = train_local_specialist(
                    codec,
                    train_coefficients,
                    pairs[family],
                    canonical_coords,
                    cfg,
                )
                evaluation = evaluate_local_specialist(
                    codec,
                    validation_coefficients,
                    family,
                    representations,
                    canonical_coords,
                    cfg,
                )
            checkpoint = run_dir / f"{arm_name}_{family}_codec.pt"
            torch.save(codec.state_dict(), checkpoint)
            training["checkpoint_sha256"] = _state_dict_sha256(codec)
            arm["families"][family] = {"training": training, "evaluation": evaluation}
        result["arms"][arm_name] = arm

    result["paired_anchor_geometry"] = paired_anchor_report(
        local_initial, validation_coefficients[:1], representations, cfg
    )
    result["causal_decision"] = _decision(result, cfg)
    result_path = run_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result["result_path"] = str(result_path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the validation-only E6 compressed spatial-locality ablation"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=CompressedLocalityConfig.epochs)
    parser.add_argument("--train-states", type=int, default=CompressedLocalityConfig.train_states)
    parser.add_argument(
        "--validation-states", type=int, default=CompressedLocalityConfig.validation_states
    )
    parser.add_argument("--batch-size", type=int, default=CompressedLocalityConfig.batch_size)
    parser.add_argument("--print-json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = CompressedLocalityConfig(
        epochs=args.epochs,
        train_states=args.train_states,
        validation_states=args.validation_states,
        batch_size=args.batch_size,
    )
    result = run_compressed_locality(cfg, run_dir=args.run_dir)
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
