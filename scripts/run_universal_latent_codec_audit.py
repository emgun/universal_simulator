#!/usr/bin/env python3
"""Run the D6 E1 codec-only and latent-geometry audit.

The latent operator is intentionally absent. The audit evaluates the 2x2
joint/matched encoder-decoder swap matrix on the exact locked validation data,
then records latent geometry and a deterministic within-validation task probe.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
import yaml

import scripts.train as train_runtime
from ups.data.latent_pairs import infer_grid_shape, make_grid_coords
from ups.data.manifests import load_data_lock
from ups.data.pdebench import PDEBenchConfig, PDEBenchDataset
from ups.data.staging import file_checksum

TASKS = ("advection1d", "burgers1d", "darcy2d")
MATCHED_ARMS = {task: f"ablation-{task}" for task in TASKS}
SCHEMA_VERSION = "ups.universal-latent-codec-audit.v1"


def _sha256(path: Path) -> str:
    return file_checksum(path, "sha256")


def _load_config(path: Path, *, data_root: Path, lock_path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    data = cfg.setdefault("data", {})
    data["root"] = str(data_root)
    data["task_roots"] = {task: str(data_root) for task in TASKS}
    data["split"] = "val"
    data["data_lock_path"] = str(lock_path)
    return cfg


def _load_models(
    *, config_path: Path, checkpoint_dir: Path, data_root: Path, lock_path: Path
) -> tuple[torch.nn.Module, torch.nn.Module, dict[str, Any]]:
    cfg = _load_config(config_path, data_root=data_root, lock_path=lock_path)
    encoder = train_runtime.make_encoder(cfg)
    decoder = train_runtime.make_decoder(cfg)
    encoder.load_state_dict(
        torch.load(checkpoint_dir / "encoder_joint.pt", map_location="cpu", weights_only=False)
    )
    decoder.load_state_dict(
        torch.load(checkpoint_dir / "decoder_joint.pt", map_location="cpu", weights_only=False)
    )
    encoder.eval()
    decoder.eval()
    return encoder, decoder, cfg


def _relative_error(prediction: torch.Tensor, target: torch.Tensor) -> float:
    numerator, denominator = _relative_components(prediction, target)
    return float(torch.sqrt(numerator / denominator))


def _relative_components(
    prediction: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    numerator = torch.sum((prediction.double() - target.double()) ** 2)
    denominator = torch.sum(target.double() ** 2).clamp_min(torch.finfo(torch.float64).eps)
    return numerator, denominator


def _spectral_error(
    prediction: torch.Tensor, target: torch.Tensor, grid_shape: tuple[int, int]
) -> float:
    height, width = grid_shape
    pred = prediction.double().reshape(height, width, -1)
    truth = target.double().reshape(height, width, -1)
    if height == 1:
        pred_spectrum = torch.fft.rfft(pred[0], dim=0).abs()
        truth_spectrum = torch.fft.rfft(truth[0], dim=0).abs()
    else:
        pred_spectrum = torch.fft.rfft2(pred, dim=(0, 1)).abs()
        truth_spectrum = torch.fft.rfft2(truth, dim=(0, 1)).abs()
    return _relative_error(pred_spectrum, truth_spectrum)


def summarize_errors(
    relative: list[float],
    spectral: list[float],
    numerators: list[float] | None = None,
    denominators: list[float] | None = None,
) -> dict[str, float | int]:
    rel = torch.tensor(relative, dtype=torch.float64)
    spec = torch.tensor(spectral, dtype=torch.float64)
    summary: dict[str, float | int] = {
        "sample_count": len(relative),
        "sample_mean_nrmse": float(rel.mean()),
        "sample_median_nrmse": float(rel.median()),
        "sample_p95_nrmse": float(torch.quantile(rel, 0.95)),
        "sample_mean_spectral_nrmse": float(spec.mean()),
    }
    if numerators is not None and denominators is not None:
        summary["global_nrmse"] = math.sqrt(sum(numerators) / max(sum(denominators), 1e-30))
    return summary


def latent_geometry(latents: torch.Tensor) -> dict[str, float | int]:
    matrix = latents.double().reshape(latents.shape[0], -1)
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    energy = singular.square()
    total = energy.sum()
    if float(total) <= 0.0:
        effective_rank = 0.0
        stable_rank = 0.0
        condition = math.inf
    else:
        probabilities = energy / total
        positive = probabilities > 0
        effective_rank = float(
            torch.exp(-(probabilities[positive] * probabilities[positive].log()).sum())
        )
        stable_rank = float(total / energy.max())
        retained = singular[singular > singular.max() * 1e-12]
        condition = float(singular.max() / retained.min()) if retained.numel() else math.inf
    variance = centered.var(dim=0, unbiased=False)
    max_variance = variance.max().clamp_min(torch.finfo(torch.float64).eps)
    return {
        "sample_count": int(matrix.shape[0]),
        "flattened_dimension": int(matrix.shape[1]),
        "effective_rank": effective_rank,
        "stable_rank": stable_rank,
        "centered_condition_number": condition,
        "latent_rms": float(torch.sqrt(matrix.square().mean())),
        "mean_sample_l2_norm": float(torch.linalg.vector_norm(matrix, dim=1).mean()),
        "mean_dimension_variance": float(variance.mean()),
        "fraction_dimensions_below_1e_6_max_variance": float(
            (variance <= max_variance * 1e-6).double().mean()
        ),
    }


def linear_cka(left: torch.Tensor, right: torch.Tensor) -> float:
    x = left.double().reshape(left.shape[0], -1)
    y = right.double().reshape(right.shape[0], -1)
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    cross = torch.linalg.matrix_norm(x.T @ y).square()
    denominator = torch.linalg.matrix_norm(x.T @ x) * torch.linalg.matrix_norm(y.T @ y)
    return float(cross / denominator.clamp_min(torch.finfo(torch.float64).eps))


def task_probe(task_latents: dict[str, torch.Tensor]) -> dict[str, Any]:
    train_features = []
    train_labels = []
    test_features = []
    test_labels = []
    for label, task in enumerate(TASKS):
        features = task_latents[task].double().reshape(task_latents[task].shape[0], -1)
        for index, feature in enumerate(features):
            if index % 2 == 0:
                train_features.append(feature)
                train_labels.append(label)
            else:
                test_features.append(feature)
                test_labels.append(label)
    train = torch.stack(train_features)
    test = torch.stack(test_features)
    labels = torch.tensor(train_labels)
    truth = torch.tensor(test_labels)
    mean = train.mean(dim=0, keepdim=True)
    scale = train.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-12)
    train = (train - mean) / scale
    test = (test - mean) / scale
    train = torch.nn.functional.normalize(train, dim=1)
    test = torch.nn.functional.normalize(test, dim=1)
    centroids = torch.stack([train[labels == label].mean(dim=0) for label in range(len(TASKS))])
    centroids = torch.nn.functional.normalize(centroids, dim=1)
    predicted = torch.argmax(test @ centroids.T, dim=1)
    per_task = {
        task: float((predicted[truth == label] == label).double().mean())
        for label, task in enumerate(TASKS)
    }
    return {
        "method": "deterministic_even_train_odd_test_standardized_cosine_nearest_centroid",
        "train_count": len(train_features),
        "test_count": len(test_features),
        "accuracy": float((predicted == truth).double().mean()),
        "balanced_accuracy": float(sum(per_task.values()) / len(per_task)),
        "per_task_accuracy": per_task,
        "chance_accuracy": 1.0 / len(TASKS),
        "interpretation": "diagnostic task separability; neither a pass nor a failure gate",
    }


def _checkpoint_alias_evidence(base_path: Path, joint_path: Path) -> dict[str, Any]:
    base = torch.load(base_path, map_location="cpu", weights_only=False)
    joint = torch.load(joint_path, map_location="cpu", weights_only=False)
    elements = 0
    tensors = 0
    all_equal = True
    for name, joint_value in joint.items():
        base_value = base.get(name)
        try:
            if base_value is None or base_value.shape != joint_value.shape:
                all_equal = False
                continue
        except RuntimeError:
            continue
        all_equal = all_equal and torch.equal(base_value, joint_value)
        elements += int(joint_value.numel())
        tensors += 1
    return {
        "compared_tensors": tensors,
        "compared_elements": elements,
        "tensor_values_equal": all_equal,
        "base_file_sha256": _sha256(base_path),
        "joint_file_sha256": _sha256(joint_path),
        "pre_joint_state_recoverable_from_base_checkpoint": not all_equal,
        "interpretation": (
            "train_joint_codec_operator overwrites the base checkpoint with the selected joint "
            "state; equal tensors mean the pre-joint state was not retained"
        ),
    }


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    lock = load_data_lock(args.lock)
    if lock.purpose != "training" or "valid" not in lock.requested_roles:
        raise ValueError("E1 requires the training lock with its validation role")
    if any(item.role in {"test", "heldout", "held_out"} for item in lock.objects):
        raise PermissionError("E1 refuses a lock containing held-out objects")

    valid_objects = {item.path: item for item in lock.objects if item.role == "valid"}
    for task in TASKS:
        path = args.data_root / f"{task}_val.h5"
        item = valid_objects.get(path.name)
        if item is None or not path.is_file():
            raise FileNotFoundError(f"Missing locked validation object: {path}")
        if path.stat().st_size != item.size_bytes or _sha256(path) != item.checksums["sha256"]:
            raise ValueError(f"Locked validation object failed verification: {path}")

    arm_names = ("joint-modular", *MATCHED_ARMS.values())
    models = {}
    configs = {}
    for arm in arm_names:
        config_path = args.config_root / f"{arm}.eval.yaml"
        checkpoint_dir = args.arms_root / arm / "checkpoints"
        encoder, decoder, cfg = _load_models(
            config_path=config_path,
            checkpoint_dir=checkpoint_dir,
            data_root=args.data_root,
            lock_path=args.lock,
        )
        models[arm] = {"encoder": encoder, "decoder": decoder}
        configs[arm] = {
            "path": str(config_path),
            "sha256": _sha256(config_path),
            "encoder_checkpoint_alias": _checkpoint_alias_evidence(
                checkpoint_dir / "encoder.pt", checkpoint_dir / "encoder_joint.pt"
            ),
            "decoder_checkpoint_alias": _checkpoint_alias_evidence(
                checkpoint_dir / "decoder.pt", checkpoint_dir / "decoder_joint.pt"
            ),
        }

    results: dict[str, Any] = {}
    joint_task_latents = {}
    for task in TASKS:
        dataset = PDEBenchDataset(
            PDEBenchConfig(
                task=task,
                split="val",
                root=str(args.data_root),
                data_lock_path=str(args.lock),
                max_samples=args.max_samples,
            )
        )
        matched_arm = MATCHED_ARMS[task]
        encoders = {
            "joint": models["joint-modular"]["encoder"],
            "matched": models[matched_arm]["encoder"],
        }
        decoders = {
            "joint": models["joint-modular"]["decoder"],
            "matched": models[matched_arm]["decoder"],
        }
        combo_errors = {
            f"{encoder_name}_encoder__{decoder_name}_decoder": {
                "relative": [],
                "spectral": [],
                "numerator": [],
                "denominator": [],
            }
            for encoder_name in encoders
            for decoder_name in decoders
        }
        target_combo_errors = (
            {
                f"{encoder_name}_encoder__{decoder_name}_decoder": {
                    "relative": [],
                    "spectral": [],
                    "numerator": [],
                    "denominator": [],
                }
                for encoder_name in encoders
                for decoder_name in decoders
            }
            if task == "darcy2d"
            else None
        )
        task_latents: dict[str, list[torch.Tensor]] = {"joint": [], "matched": []}
        source_states = []
        target_latents: dict[str, list[torch.Tensor]] = {"joint": [], "matched": []}
        target_states = []
        with torch.no_grad():
            for index in range(len(dataset)):
                sample = dataset[index]
                fields = sample["fields"].float()
                grid_shape = infer_grid_shape(fields)
                source_state = train_runtime._flatten_field_step(fields[0], grid_shape)
                source_states.append(source_state.squeeze(0).cpu())
                coords = make_grid_coords(grid_shape, torch.device("cpu"))
                encoded = {
                    name: encoder({"u": source_state}, coords, meta={"grid_shape": grid_shape})
                    for name, encoder in encoders.items()
                }
                for name, latent in encoded.items():
                    task_latents[name].append(latent.squeeze(0).cpu())
                for encoder_name, latent in encoded.items():
                    for decoder_name, decoder in decoders.items():
                        prediction = decoder(coords, latent, conditioning={})["u"]
                        key = f"{encoder_name}_encoder__{decoder_name}_decoder"
                        combo_errors[key]["relative"].append(
                            _relative_error(prediction, source_state)
                        )
                        numerator, denominator = _relative_components(prediction, source_state)
                        combo_errors[key]["numerator"].append(float(numerator))
                        combo_errors[key]["denominator"].append(float(denominator))
                        combo_errors[key]["spectral"].append(
                            _spectral_error(prediction, source_state, grid_shape)
                        )
                if target_combo_errors is not None:
                    target_state = train_runtime._flatten_field_step(
                        sample["targets"].float()[0], grid_shape
                    )
                    target_states.append(target_state.squeeze(0).cpu())
                    target_encoded = {
                        name: encoder({"u": target_state}, coords, meta={"grid_shape": grid_shape})
                        for name, encoder in encoders.items()
                    }
                    for name, latent in target_encoded.items():
                        target_latents[name].append(latent.squeeze(0).cpu())
                    for encoder_name, latent in target_encoded.items():
                        for decoder_name, decoder in decoders.items():
                            prediction = decoder(coords, latent, conditioning={})["u"]
                            key = f"{encoder_name}_encoder__{decoder_name}_decoder"
                            target_combo_errors[key]["relative"].append(
                                _relative_error(prediction, target_state)
                            )
                            numerator, denominator = _relative_components(prediction, target_state)
                            target_combo_errors[key]["numerator"].append(float(numerator))
                            target_combo_errors[key]["denominator"].append(float(denominator))
                            target_combo_errors[key]["spectral"].append(
                                _spectral_error(prediction, target_state, grid_shape)
                            )
        stacked = {name: torch.stack(values) for name, values in task_latents.items()}
        joint_task_latents[task] = stacked["joint"]
        results[task] = {
            "source_state": "first physical input state; Darcy uses coefficient, not solution",
            "codec_swap_matrix": {
                key: summarize_errors(
                    value["relative"],
                    value["spectral"],
                    value["numerator"],
                    value["denominator"],
                )
                for key, value in combo_errors.items()
            },
            "latent_geometry": {name: latent_geometry(value) for name, value in stacked.items()},
            "source_state_geometry": latent_geometry(torch.stack(source_states)),
            "joint_vs_matched_linear_cka": linear_cka(stacked["joint"], stacked["matched"]),
        }
        if target_combo_errors is not None:
            stacked_targets = {name: torch.stack(values) for name, values in target_latents.items()}
            results[task]["target_solution"] = {
                "codec_swap_matrix": {
                    key: summarize_errors(
                        value["relative"],
                        value["spectral"],
                        value["numerator"],
                        value["denominator"],
                    )
                    for key, value in target_combo_errors.items()
                },
                "latent_geometry": {
                    name: latent_geometry(value) for name, value in stacked_targets.items()
                },
                "physical_state_geometry": latent_geometry(torch.stack(target_states)),
                "joint_vs_matched_linear_cka": linear_cka(
                    stacked_targets["joint"], stacked_targets["matched"]
                ),
                "training_contract_note": (
                    "train_decoder reconstructs fields only; Darcy targets enter decoder "
                    "supervision only through the coupled operator rollout stages"
                ),
            }

    worse_tasks = []
    ratios = {}
    global_ratios = {}
    for task in TASKS:
        matrix = results[task]["codec_swap_matrix"]
        joint = matrix["joint_encoder__joint_decoder"]["sample_mean_nrmse"]
        matched = matrix["matched_encoder__matched_decoder"]["sample_mean_nrmse"]
        ratios[task] = joint / matched if matched else math.inf
        joint_global = matrix["joint_encoder__joint_decoder"]["global_nrmse"]
        matched_global = matrix["matched_encoder__matched_decoder"]["global_nrmse"]
        global_ratios[task] = joint_global / matched_global if matched_global else math.inf
        if joint > matched:
            worse_tasks.append(task)

    return {
        "schema_version": SCHEMA_VERSION,
        "implementation": {
            "audit_script_sha256": _sha256(Path(__file__).resolve()),
            "training_runtime_path": str(Path(train_runtime.__file__).resolve()),
            "training_runtime_sha256": _sha256(Path(train_runtime.__file__).resolve()),
        },
        "boundary": {
            "operator_instantiated": False,
            "operator_called": False,
            "training_or_parameter_updates": False,
            "heldout_reads": 0,
            "source_state_index": 0,
            "post_hoc_diagnostic_not_preregistered": True,
        },
        "data": {
            "lock_path": str(args.lock),
            "lock_sha256": lock.lock_sha256,
            "root": str(args.data_root),
            "validation_objects": {
                name: {
                    "size_bytes": item.size_bytes,
                    "sha256": item.checksums["sha256"],
                }
                for name, item in valid_objects.items()
            },
        },
        "checkpoint_bindings": configs,
        "tasks": results,
        "joint_latent_task_probe": task_probe(joint_task_latents),
        "comparison": {
            "joint_to_matched_codec_sample_mean_nrmse_ratio": ratios,
            "joint_to_matched_codec_global_nrmse_ratio": global_ratios,
            "tasks_where_joint_codec_is_worse": worse_tasks,
            "darcy_target_solution_joint_to_matched_codec_sample_mean_nrmse_ratio": (
                results["darcy2d"]["target_solution"]["codec_swap_matrix"][
                    "joint_encoder__joint_decoder"
                ]["sample_mean_nrmse"]
                / results["darcy2d"]["target_solution"]["codec_swap_matrix"][
                    "matched_encoder__matched_decoder"
                ]["sample_mean_nrmse"]
            ),
            "darcy_target_solution_joint_to_matched_codec_global_nrmse_ratio": (
                results["darcy2d"]["target_solution"]["codec_swap_matrix"][
                    "joint_encoder__joint_decoder"
                ]["global_nrmse"]
                / results["darcy2d"]["target_solution"]["codec_swap_matrix"][
                    "matched_encoder__matched_decoder"
                ]["global_nrmse"]
            ),
        },
        "interpretation_boundary": {
            "can_localize_error_to_codec_path": True,
            "can_separate_encoder_from_decoder_without_coadaptation_caveat": False,
            "can_establish_cross_modality_universality": False,
            "can_authorize_family_routing": False,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--arms-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.lock = args.lock.resolve()
    report = run_audit(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
