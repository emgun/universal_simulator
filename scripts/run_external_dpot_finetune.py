#!/usr/bin/env python
from __future__ import annotations

"""Finetune DPOT channel-lift adapter layers under the light-v1 protocol."""

import argparse
import hashlib
import importlib
import json
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts import run_external_neuraloperator_fno_baseline as fno_runner
from scripts.run_external_poseidon_transfer_adapter import (
    light_step_to_poseidon_pixels,
    poseidon_pixels_to_repo_flat,
)
from scripts.run_physical_conv_baseline import _add_rollout_metrics
from ups.data.latent_pairs import infer_channel_count, infer_grid_shape
from ups.data.pdebench import get_pdebench_spec
from ups.eval.pdebench_runner import _aggregate_chunk_metrics, _flatten_field_step

MEASUREMENT_TYPE = "dpot_finetune_validation_measurement"
ALLOWED_STATUSES = {"validation_finetune_measurement_complete", "invalid"}
CHANNEL_LIFT_ADAPTER_MODE = "channel_lift"
ALLOWED_ADAPTER_MODES = (CHANNEL_LIFT_ADAPTER_MODE,)
HISTORY_INIT_REPEAT_CURRENT = "repeat_current"
ALLOWED_HISTORY_INITS = (HISTORY_INIT_REPEAT_CURRENT,)
DPOT_SOURCE_URL = "https://github.com/HaoZhongkai/DPOT"
DPOT_MODEL_IMPORT = "models.dpot.DPOTNet"
DEFAULT_CHECKPOINT_FILE = "model_Ti.pth"
DEFAULT_EXPECTED_CHECKPOINT_SHA256 = (
    "074c337f9b3a3c70253f8022ce6be7e7dfb809a91a7b00e46fbfedf9611d767f"
)
DEFAULT_DPOT_SOURCE_COMMIT = "dcd2f9a9359765e19ad63e2f3f879a2a8ce1aa17"
CHANNEL_LIFT_PARAMETER_PREFIXES = ("lift.", "readout.")


class MissingDPOTDependencyError(RuntimeError):
    """Raised when live DPOT validation is requested without official source."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(path: Path) -> str:
    try:
        proc = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            capture_output=True,
            check=True,
        )
    except Exception:
        return "unknown"
    return proc.stdout.strip()


def _field_step_count(fields: torch.Tensor) -> int:
    if fields.dim() >= 3 and fields.shape[0] > 1:
        return int(fields.shape[0])
    return 1


def _insert_dpot_path(dpot_repo: Path | None) -> str:
    if dpot_repo is None:
        raise MissingDPOTDependencyError("Pass --dpot-repo pointing at the official DPOT checkout")
    if not dpot_repo.exists():
        raise MissingDPOTDependencyError(f"DPOT repo does not exist: {dpot_repo}")
    repo_path = str(dpot_repo)
    sys.path.insert(0, repo_path)
    return repo_path


def _remove_dpot_path(path: str) -> None:
    if not path:
        return
    try:
        sys.path.remove(path)
    except ValueError:
        pass


def dpot_source_snapshot(dpot_repo: Path | None) -> dict[str, Any]:
    exists = dpot_repo is not None and dpot_repo.exists()
    required_files = [
        "README.md",
        "models/dpot.py",
    ]
    files: dict[str, dict[str, Any]] = {}
    for relative in required_files:
        path = dpot_repo / relative if exists and dpot_repo is not None else Path(relative)
        files[relative] = {
            "exists": path.exists(),
            "bytes": path.stat().st_size if path.exists() else 0,
        }
    return {
        "source_id": "dpot_official_repo",
        "source_url": DPOT_SOURCE_URL,
        "repo_path": str(dpot_repo) if dpot_repo is not None else "",
        "available": bool(exists),
        "commit": _git_commit(dpot_repo) if exists and dpot_repo is not None else "missing",
        "required_files": files,
    }


def resolve_dpot_checkpoint_file(
    *,
    dpot_repo: Path | None,
    filename: str,
    expected_sha256: str,
) -> dict[str, Any]:
    candidates = [Path(filename)]
    if dpot_repo is not None:
        candidates.insert(0, dpot_repo / filename)
    checkpoint_path = next((path for path in candidates if path.exists()), None)
    if checkpoint_path is None:
        searched = ", ".join(str(path) for path in candidates)
        raise MissingDPOTDependencyError(f"DPOT checkpoint not found; searched: {searched}")
    sha256 = _sha256_file(checkpoint_path)
    if expected_sha256 and sha256 != expected_sha256:
        raise RuntimeError(
            f"DPOT checkpoint SHA256 mismatch for {checkpoint_path}: "
            f"expected {expected_sha256}, got {sha256}"
        )
    return {
        "source": "hzk17/DPOT",
        "filename": filename,
        "path": str(checkpoint_path),
        "bytes": checkpoint_path.stat().st_size,
        "sha256": sha256,
        "expected_sha256": expected_sha256,
        "sha256_status": "matched" if expected_sha256 else "recorded",
    }


def _checkpoint_state_dict(payload: Any) -> Mapping[str, torch.Tensor]:
    if isinstance(payload, Mapping):
        for key in ("model", "state_dict", "model_state_dict"):
            state = payload.get(key)
            if isinstance(state, Mapping):
                return state
        return payload
    raise TypeError("DPOT checkpoint payload must be a state dict or mapping containing one")


def load_dpot_model(
    *,
    dpot_repo: Path | None,
    checkpoint_file: str,
    expected_checkpoint_sha256: str,
    dpot_source_commit: str,
    device: str | torch.device,
) -> tuple[nn.Module, dict[str, Any], dict[str, Any]]:
    source = dpot_source_snapshot(dpot_repo)
    if dpot_source_commit and source["commit"] != dpot_source_commit:
        raise RuntimeError(
            f"DPOT source commit mismatch: expected {dpot_source_commit}, got {source['commit']}"
        )
    checkpoint = resolve_dpot_checkpoint_file(
        dpot_repo=dpot_repo,
        filename=checkpoint_file,
        expected_sha256=expected_checkpoint_sha256,
    )
    inserted_path = _insert_dpot_path(dpot_repo)
    try:
        module = importlib.import_module("models.dpot")
        dpot_cls = module.DPOTNet
        model = dpot_cls(
            img_size=128,
            patch_size=8,
            mixing_type="afno",
            in_channels=4,
            in_timesteps=10,
            out_timesteps=1,
            out_channels=4,
            normalize=False,
            embed_dim=512,
            modes=32,
            depth=4,
            n_blocks=4,
            mlp_ratio=1,
            out_layer_dim=32,
            n_cls=12,
        )
        payload = torch.load(checkpoint["path"], map_location="cpu")
        model.load_state_dict(_checkpoint_state_dict(payload), strict=True)
    except Exception as exc:
        raise MissingDPOTDependencyError(
            "Official DPOT source and Tiny checkpoint are required for live validation. "
            "Pass --dpot-repo pointing at the official checkout and --checkpoint-file "
            "pointing at model_Ti.pth."
        ) from exc
    finally:
        _remove_dpot_path(inserted_path)
    model.to(torch.device(device)).eval()
    model_info = {
        "implementation": DPOT_MODEL_IMPORT,
        "effective_config": {
            "image_size": 128,
            "patch_size": 8,
            "mixing_type": "afno",
            "in_channels": 4,
            "in_timesteps": 10,
            "out_timesteps": 1,
            "out_channels": 4,
            "normalize": False,
            "embed_dim": 512,
            "modes": 32,
            "depth": 4,
            "n_blocks": 4,
            "mlp_ratio": 1,
            "out_layer_dim": 32,
            "n_cls": 12,
        },
        "embedding_recovery_replaced": False,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
    }
    return model, model_info, checkpoint


def _extract_model_output(output: Any) -> torch.Tensor:
    if hasattr(output, "output"):
        return output.output
    if isinstance(output, tuple):
        return output[0]
    return output


class ChannelLiftDPOT(nn.Module):
    """Scalar adapter over a frozen DPOT backbone.

    DPOT consumes ``(B, H, W, T, C)`` tensors. This wrapper accepts scalar
    history windows shaped ``(B, T, 1, H, W)``, lifts each frame to the native
    four-channel DPOT contract, and reads the one-step four-channel prediction
    back to scalar pixels.
    """

    def __init__(
        self,
        backbone: nn.Module,
        *,
        history_steps: int,
        backbone_channels: int = 4,
    ) -> None:
        super().__init__()
        if int(history_steps) < 1:
            raise ValueError("history_steps must be positive")
        if int(backbone_channels) < 1:
            raise ValueError("backbone_channels must be positive")
        self.backbone = backbone
        self.history_steps = int(history_steps)
        self.backbone_channels = int(backbone_channels)
        self.lift = nn.Conv2d(1, self.backbone_channels, kernel_size=1, bias=True)
        self.readout = nn.Conv2d(self.backbone_channels, 1, kernel_size=1, bias=True)
        with torch.no_grad():
            self.lift.weight.fill_(1.0)
            self.lift.bias.zero_()
            self.readout.weight.fill_(1.0 / float(self.backbone_channels))
            self.readout.bias.zero_()

    def forward(self, history_pixels: torch.Tensor) -> torch.Tensor:
        if history_pixels.dim() != 5:
            raise ValueError(
                "history_pixels must have shape (B,T,1,H,W); "
                f"got {tuple(history_pixels.shape)}"
            )
        batch, steps, channels, height, width = history_pixels.shape
        if int(steps) != self.history_steps:
            raise ValueError(f"Expected {self.history_steps} history steps, got {steps}")
        if int(channels) != 1:
            raise ValueError(f"Expected scalar history channel count 1, got {channels}")
        lifted = self.lift(history_pixels.reshape(batch * steps, channels, height, width))
        lifted = lifted.reshape(batch, steps, self.backbone_channels, height, width)
        dpot_input = lifted.permute(0, 3, 4, 1, 2).contiguous()
        output = _extract_model_output(self.backbone(dpot_input))
        if output.dim() != 5:
            raise ValueError(f"DPOT output must have shape (B,H,W,T,C), got {tuple(output.shape)}")
        last = output[:, :, :, -1, :].permute(0, 3, 1, 2).contiguous()
        return self.readout(last)


def build_repeat_current_history(pixels: torch.Tensor, *, history_steps: int) -> torch.Tensor:
    if pixels.dim() != 4:
        raise ValueError(f"pixels must have shape (B,1,H,W), got {tuple(pixels.shape)}")
    if int(pixels.shape[1]) != 1:
        raise ValueError(f"pixels must be scalar-channel, got {pixels.shape[1]}")
    return pixels.unsqueeze(1).repeat(1, int(history_steps), 1, 1, 1).contiguous()


def append_prediction_to_history(history: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    if history.dim() != 5:
        raise ValueError(f"history must have shape (B,T,1,H,W), got {tuple(history.shape)}")
    if prediction.dim() != 4:
        raise ValueError(f"prediction must have shape (B,1,H,W), got {tuple(prediction.shape)}")
    return torch.cat((history[:, 1:], prediction.unsqueeze(1)), dim=1).contiguous()


def configure_trainable_dpot_parameters(
    model: nn.Module,
    *,
    adapter_mode: str = CHANNEL_LIFT_ADAPTER_MODE,
) -> dict[str, Any]:
    if adapter_mode not in ALLOWED_ADAPTER_MODES:
        raise ValueError(f"adapter_mode must be one of {list(ALLOWED_ADAPTER_MODES)}")
    trainable_names: list[str] = []
    frozen_names: list[str] = []
    total_parameter_count = 0
    trainable_parameter_count = 0
    for name, parameter in model.named_parameters():
        total_parameter_count += int(parameter.numel())
        trainable = name.startswith(CHANNEL_LIFT_PARAMETER_PREFIXES)
        parameter.requires_grad_(trainable)
        if trainable:
            trainable_names.append(name)
            trainable_parameter_count += int(parameter.numel())
        else:
            frozen_names.append(name)
    if not trainable_names:
        raise ValueError(
            "No DPOT channel-lift adapter parameters matched; expected names "
            f"starting with {list(CHANNEL_LIFT_PARAMETER_PREFIXES)}"
        )
    return {
        "adapter_mode": adapter_mode,
        "trainable_parameter_names": trainable_names,
        "frozen_parameter_count": total_parameter_count - trainable_parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "total_parameter_count": total_parameter_count,
        "frozen_parameter_names_sample": frozen_names[:20],
    }


def collect_dpot_training_pairs(
    cfg: Mapping[str, Any],
    *,
    tasks: Sequence[str],
    split: str,
    data_root: str | None,
    max_train_samples: int,
    rollout_steps: int,
    image_size: int,
    history_steps: int,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    histories: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    records: list[dict[str, Any]] = []

    for task in tasks:
        dataset = fno_runner._dataset(
            cfg,
            task=task,
            split=split,
            data_root=data_root,
            max_samples=max_train_samples,
        )
        family = get_pdebench_spec(task).family
        task_pairs = 0
        task_channels: int | None = None
        task_grid_shape: tuple[int, int] | None = None
        for sample_idx in range(len(dataset)):
            fields = dataset[sample_idx]["fields"].float()
            grid_shape = infer_grid_shape(fields)
            channels = infer_channel_count(fields, grid_shape)
            if channels != 1:
                raise ValueError(
                    f"DPOT channel-lift finetuning currently expects one channel; "
                    f"task={task} has {channels}"
                )
            task_channels = channels
            task_grid_shape = grid_shape
            max_steps = min(_field_step_count(fields) - 1, int(rollout_steps))
            for step in range(max_steps):
                current = light_step_to_poseidon_pixels(
                    fields[step],
                    grid_shape,
                    image_size=image_size,
                )
                target = light_step_to_poseidon_pixels(
                    fields[step + 1],
                    grid_shape,
                    image_size=image_size,
                )
                histories.append(
                    build_repeat_current_history(current, history_steps=history_steps)
                )
                targets.append(target)
                task_pairs += 1
        records.append(
            {
                "task": task,
                "split": split,
                "family": family,
                "sample_count": len(dataset),
                "pairs_collected": task_pairs,
                "repo_inferred_grid_shape": list(task_grid_shape or (0, 0)),
                "repo_inferred_channels": task_channels,
                "dpot_image_size": int(image_size),
                "history_steps": int(history_steps),
                "history_init": HISTORY_INIT_REPEAT_CURRENT,
                "teacher_forced_steps": True,
            }
        )

    if not histories:
        raise RuntimeError("DPOT finetuning received no train pairs")
    return torch.cat(histories, dim=0), torch.cat(targets, dim=0), records


def train_dpot_adapter(
    cfg: Mapping[str, Any],
    model: nn.Module,
    *,
    tasks: Sequence[str],
    split: str,
    data_root: str | None,
    max_train_samples: int,
    rollout_steps: int,
    image_size: int,
    history_steps: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    grad_clip_norm: float,
    seed: int,
    device: str | torch.device,
) -> dict[str, Any]:
    histories, targets, records = collect_dpot_training_pairs(
        cfg,
        tasks=tasks,
        split=split,
        data_root=data_root,
        max_train_samples=max_train_samples,
        rollout_steps=rollout_steps,
        image_size=image_size,
        history_steps=history_steps,
    )
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise ValueError("No trainable DPOT parameters configured")

    torch.manual_seed(int(seed))
    device = torch.device(device)
    model.to(device).train()
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    generator = torch.Generator().manual_seed(int(seed))
    best_loss = float("inf")
    best_state = fno_runner._clone_tensor_state_dict(model)
    epoch_losses: list[float] = []

    for _ in range(max(int(epochs), 1)):
        order = torch.randperm(int(histories.shape[0]), generator=generator)
        total_loss = 0.0
        batches = 0
        for start in range(0, int(histories.shape[0]), max(int(batch_size), 1)):
            index = order[start : start + max(int(batch_size), 1)]
            history = histories.index_select(0, index).to(device)
            target = targets.index_select(0, index).to(device)
            pred = model(history)
            if not torch.isfinite(pred).all():
                raise RuntimeError(
                    "Non-finite DPOT finetune prediction during training; "
                    "reduce learning rate or adapter scope"
                )
            loss = torch.mean((pred - target) ** 2)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    "Non-finite DPOT finetune loss during training; "
                    "reduce learning rate or adapter scope"
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if float(grad_clip_norm) > 0.0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_parameters,
                    max_norm=float(grad_clip_norm),
                )
                if not torch.isfinite(torch.as_tensor(grad_norm)):
                    raise RuntimeError(
                        "Non-finite DPOT finetune gradient norm during training; "
                        "reduce learning rate or adapter scope"
                    )
            optimizer.step()
            for name, parameter in model.named_parameters():
                if parameter.requires_grad and not torch.isfinite(parameter).all():
                    raise RuntimeError(
                        f"Non-finite DPOT finetune parameter after optimizer step: {name}"
                    )
            total_loss += float(loss.detach().cpu().item())
            batches += 1
        mean_loss = total_loss / max(batches, 1)
        epoch_losses.append(mean_loss)
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_state = fno_runner._clone_tensor_state_dict(model)

    model.load_state_dict(best_state)
    model.to("cpu").eval()
    return {
        "train_split": split,
        "train_pairs": int(histories.shape[0]),
        "epochs": int(max(int(epochs), 1)),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "batch_size": int(batch_size),
        "grad_clip_norm": float(grad_clip_norm),
        "seed": int(seed),
        "best_train_mse": best_loss,
        "epoch_train_mse": epoch_losses,
        "training_records": records,
    }


def _model_predict_history(
    model: nn.Module,
    history: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    with torch.no_grad():
        prediction = model(history.to(device))
    return prediction.detach().cpu()


def evaluate_dpot_validation(
    cfg: Mapping[str, Any],
    model: nn.Module,
    *,
    tasks: Sequence[str],
    split: str,
    data_root: str | None,
    max_eval_samples: int,
    rollout_steps: int,
    image_size: int,
    history_steps: int,
    device: str | torch.device,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    device = torch.device(device)
    total_pred: list[torch.Tensor] = []
    total_target: list[torch.Tensor] = []
    per_task_pred: dict[str, list[torch.Tensor]] = {}
    per_task_target: dict[str, list[torch.Tensor]] = {}
    per_family_pred: dict[str, list[torch.Tensor]] = {}
    per_family_target: dict[str, list[torch.Tensor]] = {}
    records: list[dict[str, Any]] = []

    for task in tasks:
        dataset = fno_runner._dataset(
            cfg,
            task=task,
            split=split,
            data_root=data_root,
            max_samples=max_eval_samples,
        )
        family = get_pdebench_spec(task).family
        task_pairs = 0
        task_channels: int | None = None
        task_grid_shape: tuple[int, int] | None = None
        for sample_idx in range(len(dataset)):
            fields = dataset[sample_idx]["fields"].float()
            grid_shape = infer_grid_shape(fields)
            channels = infer_channel_count(fields, grid_shape)
            if channels != 1:
                raise ValueError(
                    f"DPOT scalar validation currently expects one channel; "
                    f"task={task} has {channels}"
                )
            task_channels = channels
            task_grid_shape = grid_shape
            max_steps = min(_field_step_count(fields) - 1, int(rollout_steps))
            current_pixels = light_step_to_poseidon_pixels(
                fields[0],
                grid_shape,
                image_size=image_size,
            )
            history = build_repeat_current_history(
                current_pixels,
                history_steps=history_steps,
            )
            for step in range(max_steps):
                pred_pixels = _model_predict_history(model, history, device=device)
                if not torch.isfinite(pred_pixels).all():
                    raise RuntimeError(f"Non-finite DPOT prediction for task={task}")
                pred = poseidon_pixels_to_repo_flat(pred_pixels, grid_shape)
                target = _flatten_field_step(fields[step + 1].float(), grid_shape).cpu()
                total_pred.append(pred)
                total_target.append(target)
                per_task_pred.setdefault(task, []).append(pred)
                per_task_target.setdefault(task, []).append(target)
                per_family_pred.setdefault(family, []).append(pred)
                per_family_target.setdefault(family, []).append(target)
                history = append_prediction_to_history(history, pred_pixels)
                task_pairs += 1
        records.append(
            {
                "task": task,
                "split": split,
                "family": family,
                "sample_count": len(dataset),
                "pairs_evaluated": task_pairs,
                "repo_inferred_grid_shape": list(task_grid_shape or (0, 0)),
                "repo_inferred_channels": task_channels,
                "dpot_image_size": int(image_size),
                "history_steps": int(history_steps),
                "history_init": HISTORY_INIT_REPEAT_CURRENT,
                "autoregressive_rollout": True,
            }
        )

    if not total_pred:
        raise RuntimeError("DPOT validation received no eval pairs")
    stats = _aggregate_chunk_metrics(total_pred, total_target)
    metrics = {
        "decoded_mse": stats["mse"],
        "decoded_mae": stats["mae"],
        "decoded_nrmse": stats["nrmse"],
        "decoded_rrmse": stats["rrmse"],
        "decoded_spectral_energy_error": stats["spectral_energy_error"],
        "decoded_rollout_mse": stats["mse"],
        "decoded_rollout_mae": stats["mae"],
        "decoded_rollout_nrmse": stats["nrmse"],
        "decoded_rollout_rrmse": stats["rrmse"],
        "decoded_rollout_spectral_energy_error": stats["spectral_energy_error"],
        "mse": stats["mse"],
        "mae": stats["mae"],
        "rmse": stats["mse"] ** 0.5,
    }
    for task, pred_chunks in per_task_pred.items():
        _add_rollout_metrics(
            metrics,
            prefix=f"task_{task}_",
            pred_chunks=pred_chunks,
            target_chunks=per_task_target[task],
        )
    for family, pred_chunks in per_family_pred.items():
        _add_rollout_metrics(
            metrics,
            prefix=f"family_{family}_",
            pred_chunks=pred_chunks,
            target_chunks=per_family_target[family],
        )
    return metrics, records


def _command_record(args: argparse.Namespace) -> list[str]:
    command = [
        "python",
        "scripts/run_external_dpot_finetune.py",
        "--config",
        args.config,
        "--name",
        args.name,
        "--output-root",
        args.output_root,
        "--train-split",
        args.train_split,
        "--eval-split",
        args.eval_split,
        "--max-train-samples",
        str(args.max_train_samples),
        "--max-eval-samples",
        str(args.max_eval_samples),
        "--rollout-steps",
        str(args.rollout_steps),
        "--dpot-repo",
        args.dpot_repo or "",
        "--dpot-source-commit",
        args.dpot_source_commit,
        "--checkpoint-file",
        args.checkpoint_file,
        "--expected-checkpoint-sha256",
        args.expected_checkpoint_sha256,
        "--device",
        args.device,
        "--epochs",
        str(args.epochs),
        "--learning-rate",
        str(args.learning_rate),
        "--weight-decay",
        str(args.weight_decay),
        "--batch-size",
        str(args.batch_size),
        "--grad-clip-norm",
        str(args.grad_clip_norm),
        "--adapter-mode",
        args.adapter_mode,
        "--history-steps",
        str(args.history_steps),
        "--history-init",
        args.history_init,
        "--image-size",
        str(args.image_size),
        "--seed",
        str(args.seed),
    ]
    if args.data_root:
        command.extend(["--data-root", args.data_root])
    tasks = list(args.tasks or args.task)
    if tasks:
        command.append("--tasks")
        command.extend(str(task) for task in tasks)
    return command


def validate_dpot_finetune_summary(summary: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if summary.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if summary.get("status") not in ALLOWED_STATUSES:
        errors.append(f"status must be one of {sorted(ALLOWED_STATUSES)}")
    if summary.get("measurement_type") != MEASUREMENT_TYPE:
        errors.append(f"measurement_type must be {MEASUREMENT_TYPE}")
    if summary.get("train_split") == "test":
        errors.append("train_split must not be test")
    if summary.get("split") == "test":
        errors.append("DPOT readiness summaries must not use split=test")
    if summary.get("held_out_test_used") is not False:
        errors.append("validation summaries must mark held_out_test_used false")
    if summary.get("held_out_test_data_read") is not False:
        errors.append("validation summaries must mark held_out_test_data_read false")
    if summary.get("claim_comparable") is not False:
        errors.append("DPOT finetune validation measurements are not claim comparable")
    if summary.get("published_numbers_directly_comparable") is not False:
        errors.append("published_numbers_directly_comparable must be false")
    metrics = summary.get("metrics")
    if not isinstance(metrics, Mapping) or "decoded_rollout_nrmse" not in metrics:
        errors.append("metrics.decoded_rollout_nrmse is required")
    details = summary.get("details", {})
    checkpoint = details.get("pretrained_checkpoint") if isinstance(details, Mapping) else None
    if not isinstance(checkpoint, Mapping) or not checkpoint.get("sha256"):
        errors.append("details.pretrained_checkpoint.sha256 is required")
    source = details.get("dpot_source") if isinstance(details, Mapping) else None
    if not isinstance(source, Mapping) or not source.get("commit"):
        errors.append("details.dpot_source.commit is required")
    trainable = details.get("trainable_parameters") if isinstance(details, Mapping) else None
    if (
        not isinstance(trainable, Mapping)
        or int(trainable.get("trainable_parameter_count", 0)) <= 0
    ):
        errors.append("details.trainable_parameters.trainable_parameter_count must be positive")
    adapter_mode = details.get("adapter_mode") if isinstance(details, Mapping) else None
    if adapter_mode not in ALLOWED_ADAPTER_MODES:
        errors.append(f"details.adapter_mode must be one of {list(ALLOWED_ADAPTER_MODES)}")
    history_steps = details.get("history_steps") if isinstance(details, Mapping) else None
    if int(history_steps or 0) <= 0:
        errors.append("details.history_steps must be positive")
    history_init = details.get("history_init") if isinstance(details, Mapping) else None
    if history_init not in ALLOWED_HISTORY_INITS:
        errors.append(f"details.history_init must be one of {list(ALLOWED_HISTORY_INITS)}")
    return errors


def run_dpot_finetune(args: argparse.Namespace) -> Path:
    if args.train_split == "test":
        raise RuntimeError("DPOT readiness finetuning must not train on split=test")
    if args.eval_split == "test":
        raise RuntimeError(
            "DPOT readiness runner does not allow split=test. Use --eval-split val; "
            "a future held-out path needs a separate pre-test contract."
        )
    if args.history_init != HISTORY_INIT_REPEAT_CURRENT:
        raise RuntimeError("Only --history-init repeat_current is implemented for DPOT readiness")

    cfg = fno_runner._load_cfg(args.config)
    tasks = fno_runner._as_task_names(cfg, args.tasks or args.task)
    backbone, model_info, checkpoint = load_dpot_model(
        dpot_repo=Path(args.dpot_repo) if args.dpot_repo else None,
        checkpoint_file=args.checkpoint_file,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        dpot_source_commit=args.dpot_source_commit,
        device=args.device,
    )
    model = ChannelLiftDPOT(
        backbone,
        history_steps=args.history_steps,
        backbone_channels=4,
    ).to(torch.device(args.device))
    trainable_info = configure_trainable_dpot_parameters(
        model,
        adapter_mode=args.adapter_mode,
    )

    started = time.time()
    train_info = train_dpot_adapter(
        cfg,
        model,
        tasks=tasks,
        split=args.train_split,
        data_root=args.data_root,
        max_train_samples=args.max_train_samples,
        rollout_steps=args.rollout_steps,
        image_size=args.image_size,
        history_steps=args.history_steps,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        grad_clip_norm=args.grad_clip_norm,
        seed=args.seed,
        device=args.device,
    )
    model.to(torch.device(args.device)).eval()
    metrics, eval_records = evaluate_dpot_validation(
        cfg,
        model,
        tasks=tasks,
        split=args.eval_split,
        data_root=args.data_root,
        max_eval_samples=args.max_eval_samples,
        rollout_steps=args.rollout_steps,
        image_size=args.image_size,
        history_steps=args.history_steps,
        device=args.device,
    )
    finished = time.time()

    output_root = Path(args.output_root)
    run_dir = output_root / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"
    summary: dict[str, Any] = {
        "schema_version": 1,
        "status": "validation_finetune_measurement_complete",
        "measurement_type": MEASUREMENT_TYPE,
        "run_name": args.name,
        "config": args.config,
        "eval_config": args.config,
        "train_split": args.train_split,
        "split": args.eval_split,
        "metrics": metrics,
        "claim_comparable": False,
        "published_numbers_directly_comparable": False,
        "held_out_test_used": False,
        "held_out_test_data_read": False,
        "stages": ["external_dpot_channel_lift_adapter_finetune"],
        "extra": {
            "baseline": "external_dpot_finetune",
            "implementation": DPOT_MODEL_IMPORT,
            "source_url": DPOT_SOURCE_URL,
            "task": tasks[0] if len(tasks) == 1 else list(tasks),
            "train_split": args.train_split,
            "split": args.eval_split,
            "max_train_samples": args.max_train_samples,
            "max_eval_samples": args.max_eval_samples,
            "rollout_steps": args.rollout_steps,
            "image_size": args.image_size,
            "device": args.device,
            "metric": "decoded_rollout_nrmse",
            "command": _command_record(args),
        },
        "details": {
            "dpot_source": dpot_source_snapshot(Path(args.dpot_repo) if args.dpot_repo else None),
            "pretrained_checkpoint": checkpoint,
            "model": model_info,
            "adapter_mode": args.adapter_mode,
            "history_steps": int(args.history_steps),
            "history_init": args.history_init,
            "trainable_parameters": trainable_info,
            "training": train_info,
            "evaluation_records": eval_records,
            "contract": {
                "validation_split_only": True,
                "train_split_only": args.train_split != "test",
                "repeat_current_history": args.history_init == HISTORY_INIT_REPEAT_CURRENT,
                "autoregressive_rollout": True,
                "frozen_backbone_channel_lift_finetune": True,
                "published_numbers_directly_comparable": False,
            },
        },
        "duration_sec": finished - started,
    }
    errors = validate_dpot_finetune_summary(summary)
    if errors:
        summary["status"] = "invalid"
        summary["validation_errors"] = errors
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "summary": str(summary_path),
                "status": summary["status"],
                "main_metric": {
                    "decoded_rollout_nrmse": metrics["decoded_rollout_nrmse"],
                },
            },
            indent=2,
        )
    )
    return summary_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_multitask_heterogeneous_light_best.yaml")
    parser.add_argument("--name", default="dpot_tiny_channel_lift_smoke_val_light_v1")
    parser.add_argument("--output-root", default="reports/research/sota_loop/external_baselines")
    parser.add_argument("--data-root")
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument("--tasks", nargs="+", default=[])
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="val")
    parser.add_argument("--max-train-samples", type=int, default=2)
    parser.add_argument("--max-eval-samples", type=int, default=2)
    parser.add_argument("--rollout-steps", type=int, default=4)
    parser.add_argument("--dpot-repo")
    parser.add_argument("--dpot-source-commit", default=DEFAULT_DPOT_SOURCE_COMMIT)
    parser.add_argument("--checkpoint-file", default=DEFAULT_CHECKPOINT_FILE)
    parser.add_argument(
        "--expected-checkpoint-sha256",
        default=DEFAULT_EXPECTED_CHECKPOINT_SHA256,
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument(
        "--adapter-mode",
        default=CHANNEL_LIFT_ADAPTER_MODE,
        choices=list(ALLOWED_ADAPTER_MODES),
    )
    parser.add_argument("--history-steps", type=int, default=10)
    parser.add_argument(
        "--history-init",
        default=HISTORY_INIT_REPEAT_CURRENT,
        choices=list(ALLOWED_HISTORY_INITS),
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=17)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_dpot_finetune(args)


if __name__ == "__main__":
    main()
