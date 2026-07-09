#!/usr/bin/env python
from __future__ import annotations

"""Finetune Poseidon ScOT scalar adapter layers under the light-v1 protocol."""

import argparse
import hashlib
import json
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
from scripts.run_external_poseidon_scot_validation import (
    DEFAULT_CHECKPOINT_FILE,
    evaluate_poseidon_scot_validation,
    load_poseidon_scot_model,
    resolve_checkpoint_file,
)
from scripts.run_external_poseidon_transfer_adapter import (
    POSEIDON_MODEL_IMPORT,
    POSEIDON_SOURCE_URL,
    light_step_to_poseidon_pixels,
    poseidon_checkpoint_handle,
    poseidon_pixels_to_repo_flat,
    poseidon_source_snapshot,
)
from scripts.run_physical_conv_baseline import _add_rollout_metrics
from ups.data.latent_pairs import infer_channel_count, infer_grid_shape
from ups.data.pdebench import get_pdebench_spec
from ups.eval.pdebench_runner import _aggregate_chunk_metrics, _flatten_field_step

MEASUREMENT_TYPE = "poseidon_scot_finetune_validation_measurement"
ALLOWED_STATUSES = {"validation_finetune_measurement_complete", "invalid"}
SCALAR_ADAPTER_MODE = "scalar_layers"
CHANNEL_LIFT_ADAPTER_MODE = "channel_lift"
TASK_MODULATED_CHANNEL_LIFT_ADAPTER_MODE = "channel_lift_task_modulated"
ALLOWED_ADAPTER_MODES = (
    SCALAR_ADAPTER_MODE,
    CHANNEL_LIFT_ADAPTER_MODE,
    TASK_MODULATED_CHANNEL_LIFT_ADAPTER_MODE,
)
SCALAR_LAYER_PARAMETER_MARKERS = (
    "embeddings.patch_embeddings.projection",
    "patch_recovery.projection",
    "patch_recovery.mixup",
)
CHANNEL_LIFT_PARAMETER_PREFIXES = ("lift.", "readout.")
TASK_MODULATION_PARAMETER_PREFIXES = (
    "task_lift_gain.",
    "task_lift_bias.",
    "task_readout_gain.",
    "task_readout_bias.",
)


class ChannelLiftScOT(nn.Module):
    """Scalar adapter that keeps the pretrained multi-channel interface intact.

    The backbone (including its pretrained patch embedding and recovery) stays
    frozen and untouched. A 1x1 lift initialized to channel replication maps
    the scalar field into the backbone's native channel space, and a 1x1
    readout initialized to the channel mean maps the native output back to a
    scalar field, so at initialization the wrapper computes exactly the
    pretrained operator applied to a replicated scalar field.
    """

    def __init__(self, backbone: nn.Module, *, backbone_channels: int) -> None:
        super().__init__()
        if int(backbone_channels) < 1:
            raise ValueError("backbone_channels must be positive")
        self.backbone = backbone
        self.backbone_channels = int(backbone_channels)
        self.lift = nn.Conv2d(1, self.backbone_channels, kernel_size=1, bias=True)
        self.readout = nn.Conv2d(self.backbone_channels, 1, kernel_size=1, bias=True)
        with torch.no_grad():
            self.lift.weight.fill_(1.0)
            self.lift.bias.zero_()
            self.readout.weight.fill_(1.0 / float(self.backbone_channels))
            self.readout.bias.zero_()

    def forward(self, *, pixel_values: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        lifted = self.lift(pixel_values)
        output = self.backbone(pixel_values=lifted, time=time)
        return self.readout(_extract_model_output(output))


class TaskModulatedChannelLiftScOT(nn.Module):
    """Task-conditioned Option B adapter over the frozen native-channel ScOT.

    The pretrained backbone interface remains intact. The base scalar lift and
    readout are initialized exactly like ``ChannelLiftScOT``. Per-task affine
    gains/biases are initialized to identity, so Option B starts equivalent to
    Option A and only learns small task-specific calibration around the frozen
    pretrained operator.
    """

    def __init__(
        self,
        backbone: nn.Module,
        *,
        backbone_channels: int,
        task_names: Sequence[str],
    ) -> None:
        super().__init__()
        if int(backbone_channels) < 1:
            raise ValueError("backbone_channels must be positive")
        if not task_names:
            raise ValueError("task_names must not be empty")
        self.backbone = backbone
        self.backbone_channels = int(backbone_channels)
        self.task_names = tuple(str(task) for task in task_names)
        self.task_to_index = {task: idx for idx, task in enumerate(self.task_names)}
        if len(self.task_to_index) != len(self.task_names):
            raise ValueError(f"task_names must be unique, got {self.task_names}")
        task_count = len(self.task_names)
        self.lift = nn.Conv2d(1, self.backbone_channels, kernel_size=1, bias=True)
        self.readout = nn.Conv2d(self.backbone_channels, 1, kernel_size=1, bias=True)
        self.task_lift_gain = nn.Embedding(task_count, self.backbone_channels)
        self.task_lift_bias = nn.Embedding(task_count, self.backbone_channels)
        self.task_readout_gain = nn.Embedding(task_count, 1)
        self.task_readout_bias = nn.Embedding(task_count, 1)
        with torch.no_grad():
            self.lift.weight.fill_(1.0)
            self.lift.bias.zero_()
            self.readout.weight.fill_(1.0 / float(self.backbone_channels))
            self.readout.bias.zero_()
            self.task_lift_gain.weight.fill_(1.0)
            self.task_lift_bias.weight.zero_()
            self.task_readout_gain.weight.fill_(1.0)
            self.task_readout_bias.weight.zero_()

    @staticmethod
    def _affine_view(values: torch.Tensor) -> torch.Tensor:
        return values.unsqueeze(-1).unsqueeze(-1)

    def forward(
        self,
        *,
        pixel_values: torch.Tensor,
        time: torch.Tensor,
        task_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if task_ids is None:
            raise ValueError("TaskModulatedChannelLiftScOT requires task_ids")
        task_ids = task_ids.to(device=pixel_values.device, dtype=torch.long)
        if task_ids.shape != (int(pixel_values.shape[0]),):
            raise ValueError(
                f"task_ids must have shape ({int(pixel_values.shape[0])},), "
                f"got {tuple(task_ids.shape)}"
            )
        lifted = self.lift(pixel_values)
        lifted = lifted * self._affine_view(self.task_lift_gain(task_ids))
        lifted = lifted + self._affine_view(self.task_lift_bias(task_ids))
        output = self.backbone(pixel_values=lifted, time=time)
        scalar = self.readout(_extract_model_output(output))
        scalar = scalar * self._affine_view(self.task_readout_gain(task_ids))
        scalar = scalar + self._affine_view(self.task_readout_bias(task_ids))
        return scalar


def _field_step_count(fields: torch.Tensor) -> int:
    if fields.dim() >= 3 and fields.shape[0] > 1:
        return int(fields.shape[0])
    return 1


def _extract_model_output(output: Any) -> torch.Tensor:
    if hasattr(output, "output"):
        return output.output
    if isinstance(output, tuple):
        return output[0]
    return output


def _forward_poseidon_pixels(
    model: nn.Module,
    pixels: torch.Tensor,
    *,
    time_value: float,
    device: torch.device,
    task_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    pixels = pixels.to(device)
    time_tensor = torch.full(
        (int(pixels.shape[0]),),
        float(time_value),
        dtype=pixels.dtype,
        device=device,
    )
    if task_ids is not None:
        task_ids = task_ids.to(device=device, dtype=torch.long)
        return _extract_model_output(
            model(pixel_values=pixels, time=time_tensor, task_ids=task_ids)
        )
    return _extract_model_output(model(pixel_values=pixels, time=time_tensor))


def configure_trainable_poseidon_parameters(
    model: nn.Module,
    *,
    adapter_mode: str = SCALAR_ADAPTER_MODE,
) -> dict[str, Any]:
    """Freeze the backbone and unfreeze only the configured adapter layers."""

    if adapter_mode not in ALLOWED_ADAPTER_MODES:
        raise ValueError(f"adapter_mode must be one of {list(ALLOWED_ADAPTER_MODES)}")

    def _is_trainable(name: str) -> bool:
        if adapter_mode == CHANNEL_LIFT_ADAPTER_MODE:
            return name.startswith(CHANNEL_LIFT_PARAMETER_PREFIXES)
        if adapter_mode == TASK_MODULATED_CHANNEL_LIFT_ADAPTER_MODE:
            return name.startswith(
                CHANNEL_LIFT_PARAMETER_PREFIXES + TASK_MODULATION_PARAMETER_PREFIXES
            )
        return any(marker in name for marker in SCALAR_LAYER_PARAMETER_MARKERS)

    trainable_names: list[str] = []
    frozen_names: list[str] = []
    total_parameter_count = 0
    trainable_parameter_count = 0
    for name, parameter in model.named_parameters():
        total_parameter_count += int(parameter.numel())
        trainable = _is_trainable(name)
        parameter.requires_grad_(trainable)
        if trainable:
            trainable_names.append(name)
            trainable_parameter_count += int(parameter.numel())
        else:
            frozen_names.append(name)

    if not trainable_names:
        expected = (
            list(CHANNEL_LIFT_PARAMETER_PREFIXES)
            if adapter_mode in (CHANNEL_LIFT_ADAPTER_MODE, TASK_MODULATED_CHANNEL_LIFT_ADAPTER_MODE)
            else list(SCALAR_LAYER_PARAMETER_MARKERS)
        )
        raise ValueError(
            f"No Poseidon adapter parameters matched for mode {adapter_mode!r}; "
            f"expected names matching {expected}"
        )

    return {
        "adapter_mode": adapter_mode,
        "trainable_parameter_names": trainable_names,
        "frozen_parameter_count": total_parameter_count - trainable_parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "total_parameter_count": total_parameter_count,
        "frozen_parameter_names_sample": frozen_names[:20],
    }


def collect_poseidon_training_pairs(
    cfg: Mapping[str, Any],
    *,
    tasks: Sequence[str],
    split: str,
    data_root: str | None,
    max_train_samples: int,
    rollout_steps: int,
    image_size: int,
    task_to_index: Mapping[str, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, list[dict[str, Any]]]:
    current_pixels: list[torch.Tensor] = []
    target_pixels: list[torch.Tensor] = []
    task_ids: list[torch.Tensor] = []
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
                    f"Poseidon scalar finetuning currently expects one channel; "
                    f"task={task} has {channels}"
                )
            task_channels = channels
            task_grid_shape = grid_shape
            max_steps = min(_field_step_count(fields) - 1, int(rollout_steps))
            for step in range(max_steps):
                current_pixels.append(
                    light_step_to_poseidon_pixels(
                        fields[step],
                        grid_shape,
                        image_size=image_size,
                    )
                )
                target_pixels.append(
                    light_step_to_poseidon_pixels(
                        fields[step + 1],
                        grid_shape,
                        image_size=image_size,
                    )
                )
                if task_to_index is not None:
                    task_ids.append(torch.tensor([int(task_to_index[task])], dtype=torch.long))
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
                "poseidon_image_size": int(image_size),
                "teacher_forced_steps": True,
            }
        )

    if not current_pixels:
        raise RuntimeError("Poseidon finetuning received no train pairs")
    return (
        torch.cat(current_pixels, dim=0),
        torch.cat(target_pixels, dim=0),
        torch.cat(task_ids, dim=0) if task_ids else None,
        records,
    )


def collect_poseidon_training_sequences(
    cfg: Mapping[str, Any],
    *,
    tasks: Sequence[str],
    split: str,
    data_root: str | None,
    max_train_samples: int,
    sequence_steps: int,
    image_size: int,
) -> torch.Tensor | None:
    """Collect (N, K+1, 1, H, W) pixel windows for composed rollout loss."""

    if int(sequence_steps) < 2:
        return None
    window = int(sequence_steps) + 1
    sequences: list[torch.Tensor] = []
    for task in tasks:
        dataset = fno_runner._dataset(
            cfg,
            task=task,
            split=split,
            data_root=data_root,
            max_samples=max_train_samples,
        )
        for sample_idx in range(len(dataset)):
            fields = dataset[sample_idx]["fields"].float()
            grid_shape = infer_grid_shape(fields)
            steps = _field_step_count(fields)
            for start in range(0, steps - window + 1, int(sequence_steps)):
                frames = [
                    light_step_to_poseidon_pixels(
                        fields[start + offset],
                        grid_shape,
                        image_size=image_size,
                    )
                    for offset in range(window)
                ]
                sequences.append(torch.cat(frames, dim=0))
    if not sequences:
        return None
    return torch.stack(sequences, dim=0)


def collect_poseidon_training_sequences_with_task_ids(
    cfg: Mapping[str, Any],
    *,
    tasks: Sequence[str],
    split: str,
    data_root: str | None,
    max_train_samples: int,
    sequence_steps: int,
    image_size: int,
    task_to_index: Mapping[str, int],
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Collect pixel rollout windows plus one task id per window."""

    if int(sequence_steps) < 2:
        return None
    window = int(sequence_steps) + 1
    sequences: list[torch.Tensor] = []
    task_ids: list[torch.Tensor] = []
    for task in tasks:
        dataset = fno_runner._dataset(
            cfg,
            task=task,
            split=split,
            data_root=data_root,
            max_samples=max_train_samples,
        )
        for sample_idx in range(len(dataset)):
            fields = dataset[sample_idx]["fields"].float()
            grid_shape = infer_grid_shape(fields)
            steps = _field_step_count(fields)
            for start in range(0, steps - window + 1, int(sequence_steps)):
                frames = [
                    light_step_to_poseidon_pixels(
                        fields[start + offset],
                        grid_shape,
                        image_size=image_size,
                    )
                    for offset in range(window)
                ]
                sequences.append(torch.cat(frames, dim=0))
                task_ids.append(torch.tensor(int(task_to_index[task]), dtype=torch.long))
    if not sequences:
        return None
    return torch.stack(sequences, dim=0), torch.stack(task_ids, dim=0)


def train_poseidon_scot_adapter(
    cfg: Mapping[str, Any],
    model: nn.Module,
    *,
    tasks: Sequence[str],
    split: str,
    data_root: str | None,
    max_train_samples: int,
    rollout_steps: int,
    image_size: int,
    time_value: float,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    grad_clip_norm: float = 1.0,
    rollout_loss_steps: int = 0,
    rollout_loss_weight: float = 1.0,
    seed: int,
    device: str | torch.device,
    task_to_index: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    currents, targets, pair_task_ids, records = collect_poseidon_training_pairs(
        cfg,
        tasks=tasks,
        split=split,
        data_root=data_root,
        max_train_samples=max_train_samples,
        rollout_steps=rollout_steps,
        image_size=image_size,
        task_to_index=task_to_index,
    )
    sequences = None
    sequence_task_ids = None
    if int(rollout_loss_steps) >= 2 and float(rollout_loss_weight) > 0.0:
        if task_to_index is None:
            sequences = collect_poseidon_training_sequences(
                cfg,
                tasks=tasks,
                split=split,
                data_root=data_root,
                max_train_samples=max_train_samples,
                sequence_steps=rollout_loss_steps,
                image_size=image_size,
            )
        else:
            sequence_payload = collect_poseidon_training_sequences_with_task_ids(
                cfg,
                tasks=tasks,
                split=split,
                data_root=data_root,
                max_train_samples=max_train_samples,
                sequence_steps=rollout_loss_steps,
                image_size=image_size,
                task_to_index=task_to_index,
            )
            if sequence_payload is not None:
                sequences, sequence_task_ids = sequence_payload
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise ValueError("No trainable Poseidon parameters configured")

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
        order = torch.randperm(int(currents.shape[0]), generator=generator)
        total_loss = 0.0
        batches = 0
        for start in range(0, int(currents.shape[0]), max(int(batch_size), 1)):
            index = order[start : start + max(int(batch_size), 1)]
            current = currents.index_select(0, index)
            target = targets.index_select(0, index).to(device)
            task_batch = pair_task_ids.index_select(0, index) if pair_task_ids is not None else None
            pred = _forward_poseidon_pixels(
                model,
                current,
                time_value=time_value,
                device=device,
                task_ids=task_batch,
            )
            if not torch.isfinite(pred).all():
                raise RuntimeError(
                    "Non-finite Poseidon finetune prediction during training; "
                    "reduce learning rate or adapter scope"
                )
            loss = torch.mean((pred - target) ** 2)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    "Non-finite Poseidon finetune loss during training; "
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
                        "Non-finite Poseidon finetune gradient norm during training; "
                        "reduce learning rate or adapter scope"
                    )
            optimizer.step()
            for name, parameter in model.named_parameters():
                if parameter.requires_grad and not torch.isfinite(parameter).all():
                    raise RuntimeError(
                        f"Non-finite Poseidon finetune parameter after optimizer step: {name}"
                    )
            total_loss += float(loss.detach().cpu().item())
            batches += 1
        if sequences is not None:
            sequence_order = torch.randperm(int(sequences.shape[0]), generator=generator)
            for start in range(0, int(sequences.shape[0]), max(int(batch_size), 1)):
                index = sequence_order[start : start + max(int(batch_size), 1)]
                window = sequences.index_select(0, index).to(device)
                task_batch = (
                    sequence_task_ids.index_select(0, index)
                    if sequence_task_ids is not None
                    else None
                )
                state = window[:, 0]
                rollout_loss = torch.zeros((), device=device)
                for offset in range(1, int(window.shape[1])):
                    state = _forward_poseidon_pixels(
                        model,
                        state,
                        time_value=time_value,
                        device=device,
                        task_ids=task_batch,
                    )
                    rollout_loss = rollout_loss + torch.mean((state - window[:, offset]) ** 2)
                rollout_loss = (
                    float(rollout_loss_weight) * rollout_loss / float(window.shape[1] - 1)
                )
                if not torch.isfinite(rollout_loss):
                    raise RuntimeError(
                        "Non-finite Poseidon rollout loss during training; "
                        "reduce learning rate or rollout_loss_steps"
                    )
                optimizer.zero_grad(set_to_none=True)
                rollout_loss.backward()
                if float(grad_clip_norm) > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        trainable_parameters,
                        max_norm=float(grad_clip_norm),
                    )
                optimizer.step()
                total_loss += float(rollout_loss.detach().cpu().item())
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
        "train_pairs": int(currents.shape[0]),
        "task_conditioned_batches": pair_task_ids is not None,
        "rollout_loss_steps": int(rollout_loss_steps),
        "rollout_loss_weight": float(rollout_loss_weight),
        "rollout_sequences": int(sequences.shape[0]) if sequences is not None else 0,
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


def evaluate_poseidon_task_modulated_validation(
    cfg: Mapping[str, Any],
    model: nn.Module,
    *,
    tasks: Sequence[str],
    split: str,
    data_root: str | None,
    max_eval_samples: int,
    rollout_steps: int,
    image_size: int,
    time_value: float,
    task_to_index: Mapping[str, int],
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
        task_index = int(task_to_index[task])
        for sample_idx in range(len(dataset)):
            fields = dataset[sample_idx]["fields"].float()
            grid_shape = infer_grid_shape(fields)
            channels = infer_channel_count(fields, grid_shape)
            if channels != 1:
                raise ValueError(
                    f"Poseidon task-modulated validation currently expects one channel; "
                    f"task={task} has {channels}"
                )
            task_channels = channels
            task_grid_shape = grid_shape
            max_steps = min(_field_step_count(fields) - 1, int(rollout_steps))
            for step in range(max_steps):
                pixels = light_step_to_poseidon_pixels(
                    fields[step],
                    grid_shape,
                    image_size=image_size,
                )
                task_ids = torch.full(
                    (int(pixels.shape[0]),),
                    task_index,
                    dtype=torch.long,
                )
                with torch.no_grad():
                    pred_pixels = (
                        _forward_poseidon_pixels(
                            model,
                            pixels,
                            time_value=time_value,
                            device=device,
                            task_ids=task_ids,
                        )
                        .detach()
                        .cpu()
                    )
                pred = poseidon_pixels_to_repo_flat(pred_pixels, grid_shape)
                target = _flatten_field_step(fields[step + 1].float(), grid_shape).cpu()
                if not torch.isfinite(pred).all():
                    raise RuntimeError(f"Non-finite Poseidon task-modulated prediction for {task}")
                total_pred.append(pred)
                total_target.append(target)
                per_task_pred.setdefault(task, []).append(pred)
                per_task_target.setdefault(task, []).append(target)
                per_family_pred.setdefault(family, []).append(pred)
                per_family_target.setdefault(family, []).append(target)
                task_pairs += 1
        records.append(
            {
                "task": task,
                "task_index": task_index,
                "split": split,
                "family": family,
                "sample_count": len(dataset),
                "pairs_evaluated": task_pairs,
                "repo_inferred_grid_shape": list(task_grid_shape or (0, 0)),
                "repo_inferred_channels": task_channels,
                "poseidon_image_size": int(image_size),
                "time_value": float(time_value),
                "teacher_forced_steps": True,
            }
        )

    if not total_pred:
        raise RuntimeError("Poseidon task-modulated validation received no eval pairs")
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
        "scripts/run_external_poseidon_scot_finetune.py",
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
        "--poseidon-model-size",
        args.poseidon_model_size,
        "--checkpoint-file",
        args.checkpoint_file,
        "--device",
        args.device,
        "--time-value",
        str(args.time_value),
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
        "--rollout-loss-steps",
        str(args.rollout_loss_steps),
        "--rollout-loss-weight",
        str(args.rollout_loss_weight),
        "--seed",
        str(args.seed),
        "--held-out-ledger-json",
        args.held_out_ledger_json,
    ]
    if args.data_root:
        command.extend(["--data-root", args.data_root])
    if args.poseidon_repo:
        command.extend(["--poseidon-repo", args.poseidon_repo])
    if args.image_size:
        command.extend(["--image-size", str(args.image_size)])
    if args.expected_checkpoint_sha256:
        command.extend(["--expected-checkpoint-sha256", args.expected_checkpoint_sha256])
    if args.allow_held_out_test_eval:
        command.append("--allow-held-out-test-eval")
    if args.allow_repeat_test:
        command.append("--allow-repeat-test")
    tasks = list(args.tasks or args.task)
    if tasks:
        command.append("--tasks")
        command.extend(str(task) for task in tasks)
    return command


def _poseidon_test_measurement_key(
    *,
    args: argparse.Namespace,
    tasks: Sequence[str],
) -> str:
    payload = {
        "adapter": "external_poseidon_scot_finetune",
        "adapter_mode": args.adapter_mode,
        "batch_size": args.batch_size,
        "checkpoint_file": args.checkpoint_file,
        "config": args.config,
        "data_root": args.data_root,
        "device": args.device,
        "epochs": args.epochs,
        "eval_split": args.eval_split,
        "expected_checkpoint_sha256": args.expected_checkpoint_sha256,
        "image_size": args.image_size,
        "learning_rate": args.learning_rate,
        "max_eval_samples": args.max_eval_samples,
        "max_train_samples": args.max_train_samples,
        "metric": "decoded_rollout_nrmse",
        "poseidon_model_size": args.poseidon_model_size,
        "rollout_steps": args.rollout_steps,
        "seed": args.seed,
        "tasks": list(tasks),
        "time_value": args.time_value,
        "train_split": args.train_split,
        "weight_decay": args.weight_decay,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _guard_poseidon_test_measurement(
    *,
    args: argparse.Namespace,
    tasks: Sequence[str],
) -> dict[str, Any]:
    measurement_key = _poseidon_test_measurement_key(args=args, tasks=tasks)
    ledger = fno_runner._load_test_ledger(args.held_out_ledger_json)
    existing_keys = {
        str(entry.get("measurement_key"))
        for entry in ledger.get("measurements", [])
        if isinstance(entry, dict)
    }
    already_recorded = measurement_key in existing_keys
    if already_recorded and not args.allow_repeat_test:
        raise RuntimeError(
            "held-out external Poseidon ScOT finetune test measurement already recorded; "
            "set --allow-repeat-test only for explicit debugging repeats"
        )
    return {
        "enabled": True,
        "allow_repeat_test": bool(args.allow_repeat_test),
        "already_recorded": already_recorded,
        "ledger_path": args.held_out_ledger_json,
        "measurement_key": measurement_key,
        "recorded": False,
    }


def _record_poseidon_test_measurement(
    *,
    args: argparse.Namespace,
    tasks: Sequence[str],
    policy: dict[str, Any],
    metrics: dict[str, float],
    summary_path: Path,
) -> bool:
    if not policy.get("enabled") or policy.get("allow_repeat_test"):
        return False
    ledger = fno_runner._load_test_ledger(args.held_out_ledger_json)
    ledger.setdefault("measurements", []).append(
        {
            "measurement_key": policy["measurement_key"],
            "metric": "decoded_rollout_nrmse",
            "run_name": args.name,
            "summary": str(summary_path),
            "test_metric_value": float(metrics["decoded_rollout_nrmse"]),
            "test_split": args.eval_split,
            "tasks": list(tasks),
        }
    )
    fno_runner._write_test_ledger(args.held_out_ledger_json, ledger)
    return True


def validate_poseidon_finetune_summary(summary: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if summary.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if summary.get("status") not in ALLOWED_STATUSES:
        errors.append(f"status must be one of {sorted(ALLOWED_STATUSES)}")
    if summary.get("measurement_type") != MEASUREMENT_TYPE:
        errors.append(f"measurement_type must be {MEASUREMENT_TYPE}")
    if summary.get("train_split") == "test":
        errors.append("train_split must not be test")
    if summary.get("split") == "test" and summary.get("held_out_test_used") is not True:
        errors.append("test split summaries must mark held_out_test_used true")
    if summary.get("split") != "test" and summary.get("held_out_test_used") is not False:
        errors.append("validation summaries must mark held_out_test_used false")
    if summary.get("claim_comparable") is not False:
        errors.append("Poseidon finetune validation measurements are not claim comparable")
    if summary.get("published_numbers_directly_comparable") is not False:
        errors.append("published_numbers_directly_comparable must be false")
    metrics = summary.get("metrics")
    if not isinstance(metrics, Mapping) or "decoded_rollout_nrmse" not in metrics:
        errors.append("metrics.decoded_rollout_nrmse is required")
    checkpoint = summary.get("details", {}).get("pretrained_checkpoint")
    if not isinstance(checkpoint, Mapping) or not checkpoint.get("sha256"):
        errors.append("details.pretrained_checkpoint.sha256 is required")
    trainable = summary.get("details", {}).get("trainable_parameters")
    if (
        not isinstance(trainable, Mapping)
        or int(trainable.get("trainable_parameter_count", 0)) <= 0
    ):
        errors.append("details.trainable_parameters.trainable_parameter_count must be positive")
    adapter_mode = summary.get("details", {}).get("adapter_mode")
    if adapter_mode not in ALLOWED_ADAPTER_MODES:
        errors.append(f"details.adapter_mode must be one of {list(ALLOWED_ADAPTER_MODES)}")
    if adapter_mode in (CHANNEL_LIFT_ADAPTER_MODE, TASK_MODULATED_CHANNEL_LIFT_ADAPTER_MODE):
        model_details = summary.get("details", {}).get("model", {})
        if (
            isinstance(model_details, Mapping)
            and model_details.get("embedding_recovery_replaced") is not False
        ):
            errors.append(
                "channel_lift summaries must keep the pretrained embedding/recovery "
                "(details.model.embedding_recovery_replaced must be false)"
            )
    if adapter_mode == TASK_MODULATED_CHANNEL_LIFT_ADAPTER_MODE:
        task_modulation = summary.get("details", {}).get("task_modulation")
        if not isinstance(task_modulation, Mapping) or not task_modulation.get("task_to_index"):
            errors.append(
                "task-modulated summaries must record details.task_modulation.task_to_index"
            )
    return errors


def run_poseidon_scot_finetune(args: argparse.Namespace) -> Path:
    if args.train_split == "test":
        raise RuntimeError("Poseidon finetuning must not train on split=test")
    if args.eval_split == "test" and not args.allow_held_out_test_eval:
        raise RuntimeError(
            "Live Poseidon ScOT finetuning evaluation on split=test requires "
            "--allow-held-out-test-eval. Use --eval-split val while debugging transfer behavior."
        )

    cfg = fno_runner._load_cfg(args.config)
    tasks = fno_runner._as_task_names(cfg, args.tasks or args.task)
    held_out_test_policy = {"enabled": False, "recorded": False}
    if args.eval_split == "test":
        held_out_test_policy = _guard_poseidon_test_measurement(args=args, tasks=tasks)

    checkpoint_handle = poseidon_checkpoint_handle(args.poseidon_model_size)
    checkpoint = resolve_checkpoint_file(
        checkpoint_handle=checkpoint_handle,
        filename=args.checkpoint_file,
        expected_sha256=args.expected_checkpoint_sha256,
    )
    channel_lift = args.adapter_mode in (
        CHANNEL_LIFT_ADAPTER_MODE,
        TASK_MODULATED_CHANNEL_LIFT_ADAPTER_MODE,
    )
    model, model_info = load_poseidon_scot_model(
        poseidon_repo=Path(args.poseidon_repo) if args.poseidon_repo else None,
        checkpoint_handle=checkpoint_handle,
        image_size=args.image_size if args.image_size else None,
        channels=None if channel_lift else 1,
        device=args.device,
    )
    image_size = int(model_info["effective_config"]["image_size"])
    task_to_index: dict[str, int] | None = None
    if args.adapter_mode == CHANNEL_LIFT_ADAPTER_MODE:
        backbone_channels = int(model_info["effective_config"]["num_channels"])
        model = ChannelLiftScOT(model, backbone_channels=backbone_channels).to(
            torch.device(args.device)
        )
        model_info = dict(model_info)
        model_info["channel_lift_backbone_channels"] = backbone_channels
    elif args.adapter_mode == TASK_MODULATED_CHANNEL_LIFT_ADAPTER_MODE:
        backbone_channels = int(model_info["effective_config"]["num_channels"])
        model = TaskModulatedChannelLiftScOT(
            model,
            backbone_channels=backbone_channels,
            task_names=tasks,
        ).to(torch.device(args.device))
        task_to_index = dict(model.task_to_index)
        model_info = dict(model_info)
        model_info["channel_lift_backbone_channels"] = backbone_channels
        model_info["task_modulated_channel_lift"] = True
        model_info["task_modulation_task_names"] = list(tasks)
    trainable_info = configure_trainable_poseidon_parameters(
        model,
        adapter_mode=args.adapter_mode,
    )

    started = time.time()
    train_info = train_poseidon_scot_adapter(
        cfg,
        model,
        tasks=tasks,
        split=args.train_split,
        data_root=args.data_root,
        max_train_samples=args.max_train_samples,
        rollout_steps=args.rollout_steps,
        image_size=image_size,
        time_value=args.time_value,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        grad_clip_norm=args.grad_clip_norm,
        rollout_loss_steps=args.rollout_loss_steps,
        rollout_loss_weight=args.rollout_loss_weight,
        seed=args.seed,
        device=args.device,
        task_to_index=task_to_index,
    )
    model.to(torch.device(args.device)).eval()
    if task_to_index is None:
        metrics, eval_records = evaluate_poseidon_scot_validation(
            cfg,
            model,
            tasks=tasks,
            split=args.eval_split,
            data_root=args.data_root,
            max_eval_samples=args.max_eval_samples,
            rollout_steps=args.rollout_steps,
            image_size=image_size,
            time_value=args.time_value,
            device=args.device,
        )
    else:
        metrics, eval_records = evaluate_poseidon_task_modulated_validation(
            cfg,
            model,
            tasks=tasks,
            split=args.eval_split,
            data_root=args.data_root,
            max_eval_samples=args.max_eval_samples,
            rollout_steps=args.rollout_steps,
            image_size=image_size,
            time_value=args.time_value,
            task_to_index=task_to_index,
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
        "held_out_test_used": args.eval_split == "test",
        "held_out_test_data_read": args.eval_split == "test",
        "stages": ["external_poseidon_scot_scalar_adapter_finetune"],
        "extra": {
            "baseline": "external_poseidon_scot_finetune",
            "implementation": POSEIDON_MODEL_IMPORT,
            "source_url": POSEIDON_SOURCE_URL,
            "task": tasks[0] if len(tasks) == 1 else list(tasks),
            "train_split": args.train_split,
            "split": args.eval_split,
            "max_train_samples": args.max_train_samples,
            "max_eval_samples": args.max_eval_samples,
            "rollout_steps": args.rollout_steps,
            "image_size": image_size,
            "time_value": args.time_value,
            "device": args.device,
            "metric": "decoded_rollout_nrmse",
            "allow_held_out_test_eval": bool(args.allow_held_out_test_eval),
            "held_out_ledger_reference": args.held_out_ledger_json,
            "command": _command_record(args),
        },
        "details": {
            "poseidon_source": poseidon_source_snapshot(
                Path(args.poseidon_repo) if args.poseidon_repo else None
            ),
            "pretrained_checkpoint": checkpoint,
            "model": model_info,
            "adapter_mode": args.adapter_mode,
            "trainable_parameters": trainable_info,
            "task_modulation": {
                "enabled": task_to_index is not None,
                "task_to_index": task_to_index or {},
                "parameterization": (
                    "per-task affine gain/bias before frozen native-channel backbone "
                    "and after scalar readout"
                    if task_to_index is not None
                    else ""
                ),
            },
            "training": train_info,
            "evaluation_records": eval_records,
            "contract": {
                "validation_split_only": args.eval_split != "test",
                "train_split_only": args.train_split != "test",
                "teacher_forced_light_v1_steps": True,
                "frozen_backbone_scalar_adapter_finetune": True,
                "pretrained_embedding_recovery_intact": args.adapter_mode
                in (CHANNEL_LIFT_ADAPTER_MODE, TASK_MODULATED_CHANNEL_LIFT_ADAPTER_MODE),
                "task_modulated_channel_lift": task_to_index is not None,
                "rollout_loss_steps": int(args.rollout_loss_steps),
                "published_numbers_directly_comparable": False,
            },
            "held_out_test_policy": held_out_test_policy,
        },
        "duration_sec": finished - started,
    }
    errors = validate_poseidon_finetune_summary(summary)
    if errors:
        summary["status"] = "invalid"
        summary["validation_errors"] = errors
    elif args.eval_split == "test":
        held_out_test_policy["recorded"] = _record_poseidon_test_measurement(
            args=args,
            tasks=tasks,
            policy=held_out_test_policy,
            metrics=metrics,
            summary_path=summary_path,
        )
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
    parser.add_argument("--name", default="poseidon_scot_scalar_ft_val_light_v1")
    parser.add_argument("--output-root", default="reports/research/sota_loop/external_baselines")
    parser.add_argument("--data-root")
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument("--tasks", nargs="+", default=[])
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="val")
    parser.add_argument("--max-train-samples", type=int, default=2)
    parser.add_argument("--max-eval-samples", type=int, default=2)
    parser.add_argument("--rollout-steps", type=int, default=4)
    parser.add_argument("--poseidon-model-size", default="T")
    parser.add_argument("--checkpoint-file", default=DEFAULT_CHECKPOINT_FILE)
    parser.add_argument("--expected-checkpoint-sha256", default="")
    parser.add_argument("--poseidon-repo")
    parser.add_argument("--image-size", type=int, default=0)
    parser.add_argument("--time-value", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--adapter-mode",
        default=SCALAR_ADAPTER_MODE,
        choices=list(ALLOWED_ADAPTER_MODES),
    )
    parser.add_argument("--rollout-loss-steps", type=int, default=0)
    parser.add_argument("--rollout-loss-weight", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--held-out-ledger-json",
        default="reports/research/sota_loop/external_baselines/test_ledger.json",
    )
    parser.add_argument("--allow-held-out-test-eval", action="store_true")
    parser.add_argument("--allow-repeat-test", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_poseidon_scot_finetune(args)


if __name__ == "__main__":
    main()
