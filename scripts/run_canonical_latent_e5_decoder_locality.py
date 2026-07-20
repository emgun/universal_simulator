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
        Representation,
        _representations,
        evaluate_field,
        inverse_distance_interpolate,
        representation_points,
        state_coefficients,
    )
    from scripts.run_canonical_latent_e4_capacity_ladder import (
        CapacityLadderConfig,
        DirectPointCodec,
        _coordinate_features,
        _family_pairs,
        _linear_slope,
        _state_dict_sha256,
        evaluate_specialist,
        high_frequency_spectral_report,
        train_specialist,
    )
else:
    from run_canonical_latent_e2_benchmark import (  # type: ignore[no-redef]
        Representation,
        _representations,
        evaluate_field,
        inverse_distance_interpolate,
        representation_points,
        state_coefficients,
    )
    from run_canonical_latent_e4_capacity_ladder import (  # type: ignore[no-redef]
        CapacityLadderConfig,
        DirectPointCodec,
        _coordinate_features,
        _family_pairs,
        _linear_slope,
        _state_dict_sha256,
        evaluate_specialist,
        high_frequency_spectral_report,
        train_specialist,
    )
from ups.eval.latent_qualification import (
    discretization_mismatch_report,
    effective_rank,
    global_nrmse,
)


@dataclass(frozen=True)
class DecoderLocalityConfig:
    seed: int = 17
    train_states: int = 128
    validation_states: int = 24
    epochs: int = 120
    batch_size: int = 16
    learning_rate: float = 2e-3
    latent_dim: int = 32
    hidden_dim: int = 32
    train_low_resolution: int = 10
    train_high_resolution: int = 14
    validation_resolution: int = 18
    canonical_query_resolution: int = 18
    max_interpolation_baseline_ratio: float = 2.0
    high_frequency_radius: float = 3.0
    local_support_radius: float = 0.20
    local_max_neighbors: int = 32
    locality_spectral_improvement: float = 0.25
    helpful_improvement: float = 0.10
    invariance_atol: float = 1e-6

    def __post_init__(self) -> None:
        if self.train_states < self.batch_size or self.validation_states < 4:
            raise ValueError("E5 requires one train batch and four validation states")
        if self.train_states % self.batch_size:
            raise ValueError("train_states must be divisible by batch_size")
        if self.local_support_radius <= 0 or self.local_max_neighbors <= 0:
            raise ValueError("local support radius and neighbor cap must be positive")

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
            train_low_resolution=self.train_low_resolution,
            train_high_resolution=self.train_high_resolution,
            validation_resolution=self.validation_resolution,
            canonical_query_resolution=self.canonical_query_resolution,
            max_interpolation_baseline_ratio=self.max_interpolation_baseline_ratio,
            high_frequency_radius=self.high_frequency_radius,
        )


class LocalIntegralDecoder(nn.Module):
    """Quadrature-aware fixed-radius source-to-query integral decoder."""

    def __init__(self, cfg: DecoderLocalityConfig):
        super().__init__()
        self.cfg = cfg
        coordinate_features = 2 * (1 + 2 * 3)
        self.source_projection = nn.Linear(cfg.latent_dim, cfg.hidden_dim)
        self.message_projection = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.query_projection = nn.Linear(coordinate_features, cfg.hidden_dim)
        self.kernel = nn.Sequential(
            nn.Linear(cfg.hidden_dim + 3, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, 1),
        )
        self.fusion = nn.Sequential(
            nn.Linear(2 * cfg.hidden_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
        )
        self.output = nn.Linear(cfg.hidden_dim, 1)

    def _neighbors(
        self, source_coords: torch.Tensor, query_coords: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distances = torch.cdist(query_coords, source_coords)
        count = min(self.cfg.local_max_neighbors, source_coords.shape[1])
        neighbor_distances, indices = distances.topk(count, dim=-1, largest=False, sorted=True)
        support = neighbor_distances <= self.cfg.local_support_radius
        if not bool(support.any(dim=-1).all()):
            raise ValueError("every query must have a source point inside local support")
        return neighbor_distances, indices, support

    @staticmethod
    def _gather(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        expanded = values.unsqueeze(1).expand(-1, indices.shape[1], -1, -1)
        return torch.gather(
            expanded,
            dim=2,
            index=indices.unsqueeze(-1).expand(-1, -1, -1, values.shape[-1]),
        )

    def neighbor_report(
        self, source_coords: torch.Tensor, query_coords: torch.Tensor
    ) -> dict[str, float]:
        distances, _, support = self._neighbors(source_coords, query_coords)
        counts = support.sum(dim=-1).to(torch.float32)
        distances_all = torch.cdist(query_coords, source_coords)
        within_radius = (distances_all <= self.cfg.local_support_radius).sum(dim=-1)
        truncated = within_radius > self.cfg.local_max_neighbors
        return {
            "minimum": float(counts.min()),
            "mean": float(counts.mean()),
            "maximum": float(counts.max()),
            "truncation_fraction": float(truncated.to(torch.float32).mean()),
            "furthest_retained_distance": float(distances.masked_fill(~support, 0.0).max()),
        }

    def forward(
        self,
        source_tokens: torch.Tensor,
        source_coords: torch.Tensor,
        source_measure: torch.Tensor,
        query_coords: torch.Tensor,
    ) -> torch.Tensor:
        if source_tokens.dim() != 3 or source_coords.dim() != 3 or query_coords.dim() != 3:
            raise ValueError("tokens, source coordinates, and queries must be rank-3")
        if source_tokens.shape[:2] != source_coords.shape[:2]:
            raise ValueError("source tokens and coordinates must share nodes")
        if source_measure.shape != (*source_coords.shape[:2], 1):
            raise ValueError("source measure must have shape (batch, nodes, 1)")
        if not torch.isfinite(source_measure).all() or torch.any(source_measure <= 0):
            raise ValueError("source measure must be finite and strictly positive")

        measure = source_measure / source_measure.sum(dim=1, keepdim=True)
        distances, indices, support = self._neighbors(source_coords, query_coords)
        neighbor_tokens = self._gather(source_tokens, indices)
        neighbor_coords = self._gather(source_coords, indices)
        neighbor_measure = self._gather(measure, indices)
        relative = (neighbor_coords - query_coords.unsqueeze(2)) / self.cfg.local_support_radius
        relative_features = torch.cat(
            [relative, (distances / self.cfg.local_support_radius).unsqueeze(-1)], dim=-1
        )

        source_features = self.source_projection(neighbor_tokens)
        logits = self.kernel(torch.cat([source_features, relative_features], dim=-1))
        logits = logits + neighbor_measure.clamp_min(1e-12).log()
        weights = torch.softmax(logits.masked_fill(~support.unsqueeze(-1), -torch.inf), dim=2)
        messages = self.message_projection(source_features)
        aggregate = (weights * messages).sum(dim=2)
        query_features = self.query_projection(_coordinate_features(query_coords))
        fused = self.fusion(torch.cat([query_features, aggregate], dim=-1))
        return self.output(fused)


class LocalIntegralDirectPointCodec(nn.Module):
    """The E4 direct-point encoder paired with the E5 local integral decoder."""

    def __init__(self, cfg: DecoderLocalityConfig):
        super().__init__()
        base = DirectPointCodec(cfg.capacity_config())
        self.encoder = base.encoder
        self.decoder = LocalIntegralDecoder(cfg)

    def encode(
        self, values: torch.Tensor, coords: torch.Tensor, measure: torch.Tensor
    ) -> torch.Tensor:
        normalized_measure = measure / measure.sum(dim=1, keepdim=True)
        relative_measure = normalized_measure * coords.shape[1]
        return self.encoder(
            torch.cat([values, _coordinate_features(coords), relative_measure], dim=-1)
        )

    def decode(
        self,
        latent: torch.Tensor,
        query: torch.Tensor,
        source_coords: torch.Tensor,
        source_measure: torch.Tensor,
    ) -> torch.Tensor:
        return self.decoder(latent, source_coords, source_measure, query)


def _local_loss(
    codec: LocalIntegralDirectPointCodec,
    coefficients: torch.Tensor,
    representations: tuple[Representation, Representation],
    canonical_coords: torch.Tensor,
) -> torch.Tensor:
    canonical_target = evaluate_field(coefficients, canonical_coords)
    losses = []
    for representation in representations:
        coords, measure = representation_points(representation, batch=coefficients.shape[0])
        values = evaluate_field(coefficients, coords)
        latent = codec.encode(values, coords, measure)
        native_prediction = codec.decode(latent, coords, coords, measure)
        canonical_prediction = codec.decode(latent, canonical_coords, coords, measure)
        native_scale = values.square().mean().clamp_min(1e-8)
        canonical_scale = canonical_target.square().mean().clamp_min(1e-8)
        losses.append(
            0.5 * F.mse_loss(native_prediction, values) / native_scale
            + 0.5 * F.mse_loss(canonical_prediction, canonical_target) / canonical_scale
        )
    return sum(losses) / 2.0


def train_local_specialist(
    codec: LocalIntegralDirectPointCodec,
    coefficients: torch.Tensor,
    representations: tuple[Representation, Representation],
    canonical_coords: torch.Tensor,
    cfg: DecoderLocalityConfig,
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
            loss = _local_loss(codec, batch, representations, query)
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
    codec: nn.Module,
    latent: torch.Tensor,
    source_coords: torch.Tensor,
    source_measure: torch.Tensor,
    query: torch.Tensor,
    *,
    local: bool,
    seed: int,
) -> float:
    if local:
        original = codec.decode(latent, query, source_coords, source_measure)  # type: ignore[attr-defined]
    else:
        original = codec.decode(latent, query)  # type: ignore[attr-defined]
    permutation = torch.randperm(
        source_coords.shape[1], generator=torch.Generator().manual_seed(seed)
    )
    if local:
        permuted = codec.decode(  # type: ignore[attr-defined]
            latent[:, permutation],
            query,
            source_coords[:, permutation],
            source_measure[:, permutation],
        )
    else:
        permuted = codec.decode(latent[:, permutation], query)  # type: ignore[attr-defined]
    return float((original - permuted).abs().max())


@torch.no_grad()
def evaluate_local_specialist(
    codec: LocalIntegralDirectPointCodec,
    coefficients: torch.Tensor,
    family: str,
    representations: dict[str, Representation],
    canonical_coords: torch.Tensor,
    cfg: DecoderLocalityConfig,
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
    samples: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for label, name in zip(("low", "high", "validation"), names):
        coords, measure = representation_points(representations[name], batch=coefficients.shape[0])
        values = evaluate_field(coefficients, coords)
        latent = codec.encode(values, coords, measure)
        latents[label] = latent
        predictions[label] = codec.decode(latent, query, coords, measure)
        samples[label] = (coords, measure, values)

    high_coords, high_measure, high_values = samples["high"]
    interpolation = inverse_distance_interpolate(high_values, high_coords, query)
    errors = {label: global_nrmse(prediction, target) for label, prediction in predictions.items()}
    interpolation_error = global_nrmse(interpolation, target)
    prediction = predictions["high"]
    target_scale = target.square().mean().sqrt().clamp_min(1e-12)
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
        "high_source_token_count": latents["high"].shape[1],
        "neighbor_count": codec.decoder.neighbor_report(high_coords, query),
        "source_order_max_abs_error": _source_order_error(
            codec,
            latents["high"],
            high_coords,
            high_measure,
            query,
            local=True,
            seed=cfg.seed + 900,
        ),
    }


def _decision(result: dict[str, Any], cfg: DecoderLocalityConfig) -> dict[str, Any]:
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
        }
    parameter_gate = (
        result["arms"]["local_integral"]["architecture"]["decoder_parameters"]
        <= result["arms"]["global_control"]["architecture"]["decoder_parameters"]
    )
    causal = parameter_gate and all(
        comparison["absolute_reconstruction_pass"]
        and comparison["validation_resolution_stability_pass"]
        and comparison["source_order_invariance_pass"]
        and comparison["high_frequency_nrmse_relative_improvement"]
        >= cfg.locality_spectral_improvement
        for comparison in comparisons.values()
    )
    helpful = all(
        comparison["high_nrmse_relative_improvement"] >= cfg.helpful_improvement
        and comparison["high_frequency_nrmse_relative_improvement"] >= cfg.helpful_improvement
        and comparison["validation_resolution_stability_pass"]
        and comparison["source_order_invariance_pass"]
        for comparison in comparisons.values()
    )
    if causal:
        classification = "decoder_locality_causal"
        next_move = "freeze the local decoder and retest only the smallest C8 specialist codec"
    elif helpful:
        classification = "decoder_locality_helpful_but_insufficient"
        next_move = "keep all-point tokens and isolate objective versus schedule"
    else:
        classification = "decoder_locality_not_causal"
        next_move = "close decoder architecture work and isolate objective versus schedule"
    return {
        "classification": classification,
        "parameter_budget_pass": parameter_gate,
        "family_comparisons": comparisons,
        "next_move": next_move,
    }


def run_decoder_locality(cfg: DecoderLocalityConfig, *, run_dir: Path) -> dict[str, Any]:
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    torch.set_num_threads(1)
    run_dir.mkdir(parents=True, exist_ok=True)
    capacity_cfg = cfg.capacity_config()
    representations = _representations(capacity_cfg.benchmark_config(capacity_cfg.rungs[0]))
    pairs = _family_pairs(representations, capacity_cfg)
    train_coefficients = state_coefficients(cfg.train_states, seed=cfg.seed)
    validation_coefficients = state_coefficients(cfg.validation_states, seed=cfg.seed + 10_000)
    canonical = Representation("canonical_queries", cfg.canonical_query_resolution, 0.0, 0.0)
    canonical_coords, _ = representation_points(canonical, batch=1)

    torch.manual_seed(cfg.seed)
    global_initial = DirectPointCodec(capacity_cfg)
    torch.manual_seed(cfg.seed)
    local_initial = LocalIntegralDirectPointCodec(cfg)
    global_encoder_hash = _state_dict_sha256(global_initial.encoder)
    local_encoder_hash = _state_dict_sha256(local_initial.encoder)
    if global_encoder_hash != local_encoder_hash:
        raise RuntimeError("E5 arms must begin with identical direct-point encoders")

    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "canonical_latent_e5_decoder_locality",
        "config": asdict(cfg),
        "state_split": {
            "train_count": cfg.train_states,
            "validation_count": cfg.validation_states,
            "cross_split_identity_overlap": 0,
        },
        "initial_encoder_sha256": global_encoder_hash,
        "arms": {},
        "boundary": {
            "operator_instantiated": False,
            "heldout_reads": 0,
            "representation_label_model_inputs": False,
            "task_label_model_inputs": False,
            "provider_calls": 0,
            "routing_paths": 0,
        },
    }
    config_payload = json.dumps(result["config"], sort_keys=True, separators=(",", ":")).encode()
    result["config_sha256"] = hashlib.sha256(config_payload).hexdigest()

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
                "compression": "none",
                "hidden_dim": cfg.hidden_dim,
            },
            "families": {},
        }
        for family in ("grid", "mesh"):
            codec = copy.deepcopy(initial)
            if arm_name == "global_control":
                training = train_specialist(
                    codec,
                    train_coefficients,
                    pairs[family],
                    canonical_coords,
                    capacity_cfg,
                    alignment_weight=0.0,
                )
                evaluation = evaluate_specialist(
                    codec,
                    validation_coefficients,
                    family,
                    representations,
                    canonical_coords,
                    capacity_cfg,
                )
                high_name = (
                    f"grid_{cfg.train_high_resolution}"
                    if family == "grid"
                    else f"mesh_{cfg.train_high_resolution}_a"
                )
                high_coords, high_measure = representation_points(
                    representations[high_name], batch=validation_coefficients.shape[0]
                )
                high_values = evaluate_field(validation_coefficients, high_coords)
                high_latent = codec.encode(high_values, high_coords, high_measure)
                query = canonical_coords.expand(validation_coefficients.shape[0], -1, -1)
                evaluation["source_order_max_abs_error"] = _source_order_error(
                    codec,
                    high_latent,
                    high_coords,
                    high_measure,
                    query,
                    local=False,
                    seed=cfg.seed + 900,
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
            arm["families"][family] = {
                "training": training,
                "evaluation": evaluation,
            }
        result["arms"][arm_name] = arm

    result["causal_decision"] = _decision(result, cfg)
    result_path = run_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result["result_path"] = str(result_path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the validation-only canonical latent E5 decoder-locality ablation"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=DecoderLocalityConfig.epochs)
    parser.add_argument("--train-states", type=int, default=DecoderLocalityConfig.train_states)
    parser.add_argument(
        "--validation-states", type=int, default=DecoderLocalityConfig.validation_states
    )
    parser.add_argument("--batch-size", type=int, default=DecoderLocalityConfig.batch_size)
    parser.add_argument("--print-json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = DecoderLocalityConfig(
        epochs=args.epochs,
        train_states=args.train_states,
        validation_states=args.validation_states,
        batch_size=args.batch_size,
    )
    result = run_decoder_locality(cfg, run_dir=args.run_dir)
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
