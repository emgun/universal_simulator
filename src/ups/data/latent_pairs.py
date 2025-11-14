from __future__ import annotations

"""Reusable latent pair datasets for grid, mesh, and particle sources."""

import io
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from ups.data.datasets import GridZarrDataset, MeshZarrDataset, ParticleZarrDataset
from ups.data.pdebench import (
    PDEBenchConfig,
    PDEBenchDataset,
    get_pdebench_spec,
    resolve_pdebench_root,
)
from ups.io.enc_grid import GridEncoder, GridEncoderConfig
from ups.io.enc_mesh_particle import MeshParticleEncoder, MeshParticleEncoderConfig


def infer_grid_shape(fields: torch.Tensor) -> tuple[int, int]:
    """Return (H, W) for a PDEBench sample tensor."""

    if fields.dim() == 4:  # (T, H, W, C) or (T, C, H, W)
        shape = fields.shape
        if shape[-1] <= 8:
            return int(shape[-3]), int(shape[-2])
        return int(shape[-2]), int(shape[-1])
    if fields.dim() == 3:  # (T, X, C) or (T, C, X)
        shape = fields.shape
        if shape[-1] <= 8:
            return 1, int(shape[-2])
        return 1, int(shape[-1])
    if fields.dim() == 2:  # (T, X)
        return 1, int(fields.shape[-1])
    raise ValueError(f"Unsupported PDEBench field shape {tuple(fields.shape)}")


def infer_channel_count(fields: torch.Tensor, grid_shape: tuple[int, int]) -> int:
    """Derive channel count from a sample tensor and grid shape."""

    if fields.dim() < 2:
        raise ValueError("Expected at least 2D field tensor to infer channels")
    H, W = grid_shape
    example = fields[0]
    total = example.numel()
    area = H * W
    if total % area != 0:
        raise ValueError(
            f"Field tensor with elements {total} not divisible by grid area {area}; cannot infer channels"
        )
    return int(total // area)


def make_grid_coords(grid_shape: tuple[int, int], device: torch.device) -> torch.Tensor:
    """Create normalised grid coordinates shaped (1, N, 2)."""

    H, W = grid_shape
    ys = torch.linspace(0.0, 1.0, H, dtype=torch.float32, device=device)
    xs = torch.linspace(0.0, 1.0, W, dtype=torch.float32, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    coords = torch.stack([grid_x, grid_y], dim=-1).reshape(1, H * W, 2)
    return coords


def _broadcast_condition_tensor(tensor: torch.Tensor, seq_len: int) -> torch.Tensor:
    tensor = tensor.float()
    if tensor.dim() == 0:
        return tensor.view(1, 1).expand(seq_len, 1).contiguous()

    lead = tensor.shape[0]
    if lead == seq_len:
        data = tensor
    elif lead == seq_len + 1:
        data = tensor[:-1]
    elif lead == 1:
        data = tensor.expand(seq_len, *tensor.shape[1:])
    else:
        data = tensor.view(1, -1).expand(seq_len, -1)

    if data.dim() == 1:
        data = data.unsqueeze(-1)
    else:
        data = data.reshape(seq_len, -1)
    return data.contiguous()


def _to_float_tensor(value: Any) -> torch.Tensor | None:
    if torch.is_tensor(value):
        return value.float()
    try:
        tensor = torch.as_tensor(value, dtype=torch.float32)
    except (TypeError, ValueError):
        return None
    if tensor.numel() == 0:
        return None
    return tensor.float()


def prepare_conditioning(
    params: dict[str, Any] | None,
    bc: dict[str, Any] | None,
    seq_len: int,
    *,
    time: Any | None = None,
    dt: Any | None = None,
    extras: dict[str, Any] | None = None,
) -> dict[str, torch.Tensor]:
    cond: dict[str, torch.Tensor] = {}

    def ingest(prefix: str, mapping: dict[str, Any] | None) -> None:
        if not mapping:
            return
        for key, value in mapping.items():
            tensor = _to_float_tensor(value)
            if tensor is None:
                continue
            cond[f"{prefix}_{key}"] = _broadcast_condition_tensor(tensor, seq_len)

    ingest("param", params)
    ingest("bc", bc)

    if time is not None:
        tensor = _to_float_tensor(time)
        if tensor is not None:
            cond["time"] = _broadcast_condition_tensor(tensor, seq_len)

    if dt is not None:
        tensor = _to_float_tensor(dt)
        if tensor is not None:
            cond["dt"] = _broadcast_condition_tensor(tensor, seq_len)

    if extras:
        for key, value in extras.items():
            tensor = _to_float_tensor(value)
            if tensor is None:
                continue
            cond[key] = _broadcast_condition_tensor(tensor, seq_len)
    return cond


def collate_conditions(conditions: Iterable[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = set()
    for cond in conditions:
        keys.update(cond.keys())
    collated: dict[str, torch.Tensor] = {}
    for key in keys:
        tensors = [cond[key] for cond in conditions if key in cond]
        if not tensors:
            continue
        collated[key] = torch.cat(tensors, dim=0)
    return collated


def _fields_to_latent_batch(
    encoder: GridEncoder,
    fields: torch.Tensor,
    coords: torch.Tensor,
    grid_shape: tuple[int, int],
    *,
    field_name: str,
    params: dict[str, torch.Tensor] | None = None,
    bc: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    T = fields.shape[0]

    if fields.dim() == 4:
        if fields.shape[-1] <= 8:
            data = fields.permute(0, 3, 1, 2)
        else:
            data = fields
    elif fields.dim() == 3:
        if fields.shape[-1] <= 8:
            data = fields.permute(0, 2, 1).unsqueeze(2)
        else:
            data = fields.unsqueeze(2)
    elif fields.dim() == 2:
        data = fields.unsqueeze(1).unsqueeze(2)
    else:
        raise ValueError(f"Unsupported PDEBench field shape {tuple(fields.shape)}")

    H, W = grid_shape
    data = data.contiguous().view(T, data.shape[1], H, W)
    B, C, H, W = data.shape
    flattened = data.view(B, C, H * W).transpose(1, 2)

    first_param = next(encoder.parameters(), None)
    encoder_device = first_param.device if first_param is not None else fields.device
    if flattened.device != encoder_device:
        flattened = flattened.to(encoder_device, non_blocking=True)

    coord_batch = coords.expand(B, -1, -1)
    if coord_batch.device != encoder_device:
        coord_batch = coord_batch.to(encoder_device, non_blocking=True)

    params_device: dict[str, torch.Tensor] | None = None
    bc_device: dict[str, torch.Tensor] | None = None
    if params is not None:
        params_device = {k: v.to(encoder_device, non_blocking=True) for k, v in params.items()}
    if bc is not None:
        bc_device = {k: v.to(encoder_device, non_blocking=True) for k, v in bc.items()}

    meta = {"grid_shape": grid_shape}
    field_inputs = {field_name: flattened}

    with torch.no_grad():
        latent = encoder(field_inputs, coord_batch, params=params_device, bc=bc_device, meta=meta)
    return latent.cpu()


def _encode_grid_sample(
    encoder: GridEncoder, sample: dict[str, Any], device: torch.device | None = None
) -> torch.Tensor:
    meta = dict(sample.get("meta", {}))
    if "grid_shape" not in meta:
        raise ValueError("Grid sample meta must include 'grid_shape'")
    H, W = map(int, meta["grid_shape"])

    field_batches: dict[str, torch.Tensor] = {}
    fields = sample.get("fields", {})
    if not fields:
        raise ValueError("Grid sample requires non-empty fields for encoding")
    if device is None:
        params = list(encoder.parameters())
        device = params[0].device if params else torch.device("cpu")
    for name, tensor in fields.items():
        if tensor.dim() != 2:
            raise ValueError(f"Field '{name}' expected shape (N, C); got {tensor.shape}")
        field_batches[name] = tensor.unsqueeze(0).float().to(device, non_blocking=True)

    coords = sample["coords"].float()
    if coords.dim() == 2:
        coords_batch = coords.unsqueeze(0)
    elif coords.dim() == 3:
        coords_batch = coords
    else:
        raise ValueError("Coords tensor must have shape (N, d) or (B, N, d)")
    coords_batch = coords_batch.to(device, non_blocking=True)

    meta.setdefault("grid_shape", (H, W))

    with torch.no_grad():
        latent = encoder(field_batches, coords_batch, meta=meta)
    return latent.squeeze(0).cpu()


@dataclass
class LatentPair:
    z0: torch.Tensor
    z1: torch.Tensor
    cond: dict[str, torch.Tensor]
    future: torch.Tensor | None = None
    # Optional fields for UPT inverse losses (Phase 1.5)
    input_fields: dict[str, torch.Tensor] | None = None
    coords: torch.Tensor | None = None
    meta: dict | None = None
    task_name: str | None = None  # Track source task for multi-task training


class GridLatentPairDataset(Dataset):
    """Wrap a PDEBench dataset and emit latent (t, t+1) token pairs."""

    def __init__(
        self,
        base: PDEBenchDataset,
        encoder: GridEncoder,
        coords: torch.Tensor,
        grid_shape: tuple[int, int],
        field_name: str = "u",
        *,
        task: str | None = None,
        device: torch.device | None = None,
        cache_dir: Path | None = None,
        cache_dtype: torch.dtype | None = torch.float16,
        time_stride: int = 1,
        rollout_horizon: int = 1,
        use_inverse_losses: bool = False,
    ) -> None:
        super().__init__()
        self.base = base
        self.encoder = encoder
        self.device = device or torch.device("cpu")
        self.coords = coords.to(self.device)
        self.grid_shape = grid_shape
        self.field_name = field_name
        self.task_name = task  # Store task name for multi-task training
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dtype = cache_dtype
        self.time_stride = max(1, int(time_stride))
        self.rollout_horizon = max(1, int(rollout_horizon))
        self.use_inverse_losses = use_inverse_losses

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> LatentPair:
        cache_hit = False
        latent_seq: torch.Tensor | None = None
        params_cpu = None
        bc_cpu = None
        fields_cpu = None  # For inverse losses

        if self.cache_dir:
            cache_path = self.cache_dir / f"sample_{idx:05d}.pt"
            if cache_path.exists():
                try:
                    data = torch.load(cache_path, map_location="cpu")
                except (RuntimeError, EOFError):  # corrupted file
                    cache_path.unlink(missing_ok=True)
                else:
                    # Keep cached latents on CPU so DataLoader pinning stays valid.
                    latent_seq = data["latent"].float()
                    params_cpu = data.get("params")
                    bc_cpu = data.get("bc")
                    # For inverse losses, also load physical fields if cached
                    if self.use_inverse_losses:
                        fields_cpu = data.get("fields")
                    cache_hit = True

        if latent_seq is None:
            sample = self.base[idx]
            fields_cpu = sample["fields"].float()
            params_cpu = sample.get("params")
            bc_cpu = sample.get("bc")

            # For DDP safety: temporarily move encoder to device for encoding
            encoder_was_on = next(self.encoder.parameters()).device
            if encoder_was_on != self.device:
                self.encoder.to(self.device)

            fields = fields_cpu.to(self.device, non_blocking=True)
            params_device = None
            if params_cpu is not None:
                params_device = {
                    k: v.to(self.device, non_blocking=True) for k, v in params_cpu.items()
                }
            bc_device = None
            if bc_cpu is not None:
                bc_device = {k: v.to(self.device, non_blocking=True) for k, v in bc_cpu.items()}

            # Move coords to device for encoding
            coords_device = self.coords.to(self.device, non_blocking=True)

            latent_seq = _fields_to_latent_batch(
                self.encoder,
                fields,
                coords_device,
                self.grid_shape,
                params=params_device,
                bc=bc_device,
                field_name=self.field_name,
            )

            # Move encoder back to original device to avoid CUDA IPC issues in DDP
            if encoder_was_on != self.device:
                self.encoder.to(encoder_was_on)
            if self.cache_dir and not cache_hit:
                to_store = (
                    latent_seq.to(self.cache_dtype) if self.cache_dtype is not None else latent_seq
                )
                tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
                tmp_path.unlink(missing_ok=True)
                payload = {
                    "latent": to_store.cpu(),
                    "params": params_cpu,
                    "bc": bc_cpu,
                }
                # Do not persist physical fields to reduce cache size and inode pressure.
                # When inverse losses are enabled, fields will be loaded on-demand from the base dataset.
                buffer = io.BytesIO()
                torch.save(payload, buffer)
                tmp_path.write_bytes(buffer.getvalue())
                tmp_path.replace(cache_path)

        # If cache hit but inverse losses needed and fields not cached, reload from base
        if self.use_inverse_losses and fields_cpu is None:
            sample = self.base[idx]
            fields_cpu = sample["fields"].float()

        # Optionally downsample the time dimension to accelerate epochs
        if self.time_stride > 1:
            latent_seq = latent_seq[:: self.time_stride]
            if fields_cpu is not None and self.time_stride > 1:
                fields_cpu = fields_cpu[:: self.time_stride]

        if latent_seq.shape[0] <= self.rollout_horizon:
            raise ValueError("Need more time steps than rollout horizon to form latent pairs")

        base_len = latent_seq.shape[0] - self.rollout_horizon
        z0 = latent_seq[:base_len]
        targets = []
        for step in range(1, self.rollout_horizon + 1):
            targets.append(latent_seq[step : step + base_len])
        target_stack = torch.stack(targets, dim=1)
        z1 = target_stack[:, 0]
        future = target_stack[:, 1:] if self.rollout_horizon > 1 else None
        cond = prepare_conditioning(params_cpu, bc_cpu, base_len)

        # Prepare optional inverse loss inputs
        input_fields = None
        coords = None
        meta = None
        if self.use_inverse_losses and fields_cpu is not None:
            # Extract fields for ALL pairs (not just first timestep)
            # Each z0 in the pairs corresponds to a different timestep
            # Shape: (base_len, ...) to match z0 shape (base_len, tokens, dim)
            fields_for_pairs = fields_cpu[:base_len]  # (base_len, ...)

            # Convert to dict format expected by loss functions
            # Process all timesteps together
            if fields_for_pairs.dim() == 2:  # 1D spatial: (base_len, N)
                # Add channel dimension
                formatted = fields_for_pairs.unsqueeze(-1)  # (base_len, N, 1)
            elif fields_for_pairs.dim() == 3:  # (base_len, N, C) or (base_len, H, W)
                if fields_for_pairs.shape[-1] <= 8:  # Likely (base_len, N, C)
                    formatted = fields_for_pairs
                else:  # Likely (base_len, H, W) - flatten spatial
                    base_len_actual, H, W = fields_for_pairs.shape
                    formatted = fields_for_pairs.reshape(
                        base_len_actual, H * W, 1
                    )  # (base_len, H*W, 1)
            elif fields_for_pairs.dim() == 4:  # (base_len, H, W, C)
                base_len_actual, H, W, C = fields_for_pairs.shape
                formatted = fields_for_pairs.reshape(
                    base_len_actual, H * W, C
                )  # (base_len, H*W, C)
            else:
                # Flatten spatial dims, keep channel dim
                formatted = fields_for_pairs.reshape(
                    fields_for_pairs.shape[0], -1, fields_for_pairs.shape[-1]
                )

            input_fields = {self.field_name: formatted}

            # Replicate coords for each pair in the trajectory
            coords = self.coords.cpu().expand(base_len, -1, -1)  # (base_len, N, 2)
            meta = {"grid_shape": self.grid_shape}

        return LatentPair(
            z0,
            z1,
            cond,
            future=future,
            input_fields=input_fields,
            coords=coords,
            meta=meta,
            task_name=self.task_name,
        )


class GridZarrLatentPairDataset(Dataset):
    """Encode consecutive time steps from a GridZarrDataset into latent pairs."""

    def __init__(
        self, base: GridZarrDataset, encoder: GridEncoder, rollout_horizon: int = 1
    ) -> None:
        super().__init__()
        self.base = base
        self.encoder = encoder
        params = list(encoder.parameters())
        self.device = params[0].device if params else torch.device("cpu")
        self.rollout_horizon = max(1, int(rollout_horizon))

    def __len__(self) -> int:
        return max(len(self.base) - self.rollout_horizon, 0)

    def __getitem__(self, idx: int) -> LatentPair:
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        latents = []
        samples = []
        for step in range(self.rollout_horizon + 1):
            sample = self.base[idx + step]
            samples.append(sample)
            latents.append(_encode_grid_sample(self.encoder, sample, self.device).squeeze(0))

        latent_stack = torch.stack(latents, dim=0)
        cond = prepare_conditioning(
            samples[0].get("params"),
            samples[0].get("bc"),
            1,
            time=samples[0].get("time"),
            dt=samples[0].get("dt"),
        )
        z0 = latent_stack[0].unsqueeze(0)
        z1 = latent_stack[1].unsqueeze(0)
        future = latent_stack[2:].unsqueeze(0) if self.rollout_horizon > 1 else None
        return LatentPair(z0, z1, cond, future=future)


def _graph_coords_at_time(
    sample: dict[str, Any],
    step_fields: dict[str, torch.Tensor],
    step: int,
) -> torch.Tensor:
    if "positions" in step_fields:
        coords = step_fields["positions"][step]
    else:
        coords = sample["coords"].float()
    coords = coords.float()
    if coords.dim() == 1:
        coords = coords.view(1, -1, 1)
    elif coords.dim() == 2:
        coords = coords.unsqueeze(0)
    elif coords.dim() == 3 and coords.shape[0] != 1:
        coords = coords[:1]
    return coords


class GraphLatentPairDataset(Dataset):
    """Encode mesh or particle samples into latent time-step pairs."""

    def __init__(
        self,
        base: Dataset,
        encoder: MeshParticleEncoder,
        kind: str = "mesh",
        rollout_horizon: int = 1,
        *,
        task: str | None = None,
        cache_dir: Path | None = None,
        cache_dtype: torch.dtype | None = torch.float16,
        time_stride: int = 1,
    ) -> None:
        super().__init__()
        self.base = base
        self.encoder = encoder
        self.kind = kind
        self.task_name = task  # Store task name for multi-task training
        params = list(encoder.parameters())
        self.device = params[0].device if params else torch.device("cpu")
        self.rollout_horizon = max(1, int(rollout_horizon))
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dtype = cache_dtype
        self.time_stride = max(1, int(time_stride))

    def __len__(self) -> int:
        return len(self.base)

    def _load_cached_latent(self, idx: int) -> dict[str, Any] | None:
        if not self.cache_dir:
            return None
        cache_path = self.cache_dir / f"sample_{idx:05d}.pt"
        if not cache_path.exists():
            return None
        try:
            data = torch.load(cache_path, map_location="cpu")
        except (RuntimeError, EOFError):
            cache_path.unlink(missing_ok=True)
            return None
        data["cache_path"] = cache_path
        return data

    def _store_cached_latent(
        self,
        idx: int,
        latent_seq: torch.Tensor,
        *,
        params: dict[str, torch.Tensor] | None,
        bc: dict[str, torch.Tensor] | None,
        time: torch.Tensor | None,
        dt: torch.Tensor | None,
    ) -> None:
        if not self.cache_dir:
            return
        cache_path = self.cache_dir / f"sample_{idx:05d}.pt"
        tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        tmp_path.unlink(missing_ok=True)
        to_store = latent_seq.to(self.cache_dtype) if self.cache_dtype is not None else latent_seq
        payload = {
            "latent": to_store.cpu(),
            "params": params,
            "bc": bc,
            "time": time,
            "dt": dt,
        }
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        tmp_path.write_bytes(buffer.getvalue())
        tmp_path.replace(cache_path)

    def __getitem__(self, idx: int) -> LatentPair:
        cached = self._load_cached_latent(idx)
        sample = None
        params_cpu = None
        bc_cpu = None
        time_cpu = None
        dt_cpu = None
        if cached is not None:
            latent_seq = cached["latent"].float()
            params_cpu = cached.get("params")
            bc_cpu = cached.get("bc")
            time_cpu = cached.get("time")
            dt_cpu = cached.get("dt")
        else:
            sample = self.base[idx]
            fields = sample.get("fields", {})
            if not fields:
                raise ValueError("Graph samples must provide fields with time dimension")

            step_fields: dict[str, torch.Tensor] = {}
            time_dim: int | None = None
            for name, tensor in fields.items():
                if tensor.dim() == 2:
                    tensor = tensor.unsqueeze(0)
                if tensor.dim() != 3:
                    raise ValueError(
                        f"Field '{name}' must have shape (T, N, C); got {tensor.shape}"
                    )
                if time_dim is None:
                    time_dim = tensor.shape[0]
                elif tensor.shape[0] != time_dim:
                    raise ValueError("All fields must share the same time dimension")
                step_fields[name] = tensor.float().to(self.device, non_blocking=True)

            if time_dim is None or time_dim <= self.rollout_horizon:
                raise ValueError("Need more time steps than rollout horizon to form latent pairs")

            latents: list[torch.Tensor] = []
            for step in range(time_dim):
                field_batch = {
                    name: tensor[step].unsqueeze(0) for name, tensor in step_fields.items()
                }
                coords = _graph_coords_at_time(sample, step_fields, step).to(
                    self.device, non_blocking=True
                )
                connect = sample.get("connect")
                if connect is not None:
                    connect = connect.to(self.device, dtype=torch.long, non_blocking=True)
                meta = sample.get("meta")
                with torch.no_grad():
                    latent_step = self.encoder(field_batch, coords, connect=connect, meta=meta)
                latents.append(latent_step.squeeze(0).cpu())

            latent_seq = torch.stack(latents, dim=0)
            params_cpu = None
            sample_params = sample.get("params")
            if sample_params:
                params_cpu = {}
                for k, v in sample_params.items():
                    if torch.is_tensor(v):
                        params_cpu[k] = v.detach().cpu()
                    else:
                        tensor = _to_float_tensor(v)
                        if tensor is not None:
                            params_cpu[k] = tensor.detach().cpu()
                if not params_cpu:
                    params_cpu = None
            bc_cpu = None
            sample_bc = sample.get("bc")
            if sample_bc:
                bc_cpu = {}
                for k, v in sample_bc.items():
                    if torch.is_tensor(v):
                        bc_cpu[k] = v.detach().cpu()
                    else:
                        tensor = _to_float_tensor(v)
                        if tensor is not None:
                            bc_cpu[k] = tensor.detach().cpu()
                if not bc_cpu:
                    bc_cpu = None
            time_val = sample.get("time")
            if torch.is_tensor(time_val):
                time_cpu = time_val.detach().cpu()
            elif time_val is not None:
                time_cpu = torch.as_tensor(time_val, dtype=torch.float32)
            else:
                time_cpu = None
            dt_val = sample.get("dt")
            if torch.is_tensor(dt_val):
                dt_cpu = dt_val.detach().cpu()
            elif dt_val is not None:
                dt_cpu = torch.as_tensor(dt_val, dtype=torch.float32)
            else:
                dt_cpu = None

            if self.cache_dir:
                self._store_cached_latent(
                    idx, latent_seq, params=params_cpu, bc=bc_cpu, time=time_cpu, dt=dt_cpu
                )

        if self.time_stride > 1:
            latent_seq = latent_seq[:: self.time_stride]
        time_dim = latent_seq.shape[0]
        if time_dim <= self.rollout_horizon:
            raise ValueError("Need more time steps than rollout horizon to form latent pairs")
        base_len = time_dim - self.rollout_horizon
        cond = prepare_conditioning(
            params_cpu,
            bc_cpu,
            base_len,
            time=time_cpu,
            dt=dt_cpu,
        )
        z0 = latent_seq[:base_len]
        targets = []
        for step in range(1, self.rollout_horizon + 1):
            targets.append(latent_seq[step : step + base_len])
        target_stack = torch.stack(targets, dim=1)
        z1 = target_stack[:, 0]
        future = target_stack[:, 1:] if self.rollout_horizon > 1 else None
        return LatentPair(z0, z1, cond, future=future, task_name=self.task_name)


def collate_latent_pairs(
    batch_items: Iterable[LatentPair],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    z0_chunks = []
    z1_chunks = []
    cond_list = []
    future_chunks: list[torch.Tensor] = []
    future_available = True
    for item in batch_items:
        z0_chunks.append(item.z0)
        z1_chunks.append(item.z1)
        cond_list.append(item.cond)
        if item.future is None:
            future_available = False
        else:
            future_chunks.append(item.future)
    z0 = torch.cat(z0_chunks, dim=0)
    z1 = torch.cat(z1_chunks, dim=0)
    cond = collate_conditions(cond_list)
    if future_available and future_chunks:
        future = torch.cat(future_chunks, dim=0)
        return z0, z1, cond, future
    return z0, z1, cond


def _build_pdebench_dataset(
    data_cfg: dict[str, Any],
) -> tuple[Dataset, GridEncoder, tuple[int, int], str]:
    dataset = PDEBenchDataset(
        PDEBenchConfig(
            task=data_cfg["task"], split=data_cfg.get("split", "train"), root=data_cfg.get("root")
        ),
    )

    sample_fields = dataset.fields[0]
    grid_shape = infer_grid_shape(sample_fields)
    channels = infer_channel_count(sample_fields, grid_shape)
    field_name = data_cfg.get("field_name", "u")
    encoder_cfg = GridEncoderConfig(
        patch_size=data_cfg.get("patch_size", 4),
        latent_dim=data_cfg.get("latent_dim", 32),
        latent_len=data_cfg.get("latent_len", 16),
        field_channels={field_name: channels},
    )
    grid_encoder = GridEncoder(encoder_cfg).eval()
    return dataset, grid_encoder, grid_shape, field_name


def build_latent_pair_loader(cfg: dict[str, Any]) -> DataLoader:
    """Construct a DataLoader that yields latent pairs from data config."""

    data_cfg = cfg.get("data", {})
    latent_cfg = cfg.get("latent", {})
    train_cfg = cfg.get("training", {})
    use_inverse_losses = train_cfg.get("use_inverse_losses", False)
    batch = train_cfg.get("batch_size", 16)
    # Keep dataset on CPU for DDP safety - DataLoader will move batches to GPU
    device = torch.device("cpu")
    default_workers = max(1, (os.cpu_count() or 4) // 4)
    num_workers = int(train_cfg.get("num_workers", default_workers))
    pin_memory = bool(train_cfg.get("pin_memory", torch.cuda.is_available()))
    prefetch_factor = train_cfg.get("prefetch_factor")
    use_parallel_encoding = bool(train_cfg.get("use_parallel_encoding", False))
    cache_dir_cfg = train_cfg.get("latent_cache_dir")
    cache_root: Path | None = Path(cache_dir_cfg) if cache_dir_cfg else None
    cache_dtype_str = train_cfg.get("latent_cache_dtype", "float16") or None
    cache_dtype = (
        getattr(torch, cache_dtype_str)
        if cache_dtype_str and hasattr(torch, cache_dtype_str)
        else None
    )
    split_name = data_cfg.get("split", "train")
    time_stride = int(train_cfg.get("time_stride", 1))
    rollout_horizon = max(1, int(train_cfg.get("rollout_horizon", 1)))

    loader_kwargs: dict[str, Any] = {
        "batch_size": batch,
        "shuffle": True,
        "collate_fn": latent_pair_collate,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }
    if prefetch_factor is not None and num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    # Support single task or multi-task mixing
    tasks = data_cfg.get("task")
    if tasks:
        from ups.data.parallel_cache import (
            PreloadedCacheDataset,
            check_cache_complete,
            check_sufficient_ram,
            estimate_cache_size_mb,
        )

        task_list = list(tasks) if isinstance(tasks, (list, tuple)) else [tasks]
        latent_datasets: list[Dataset] = []

        def _make_grid_latent_dataset(task_name: str) -> Dataset:
            dataset, encoder, grid_shape, field_name = _build_pdebench_dataset(
                {
                    **data_cfg,
                    "task": task_name,
                    "latent_dim": latent_cfg.get("dim", 32),
                    "latent_len": latent_cfg.get("tokens", 16),
                }
            )
            # Keep encoder on CPU to avoid CUDA IPC issues in DDP
            # Encoder will be moved to device during encoding in __getitem__
            encoder = encoder.to("cpu")
            coords = make_grid_coords(grid_shape, torch.device("cpu"))
            ds_cache = cache_root / f"{task_name}_{split_name}" if cache_root else None

            use_preloaded = False
            if ds_cache and ds_cache.exists():
                num_samples = len(dataset)
                cache_complete, num_cached = check_cache_complete(ds_cache, num_samples)
                if cache_complete:
                    cache_size_mb = estimate_cache_size_mb(
                        ds_cache, num_samples=min(10, num_samples)
                    )
                    if check_sufficient_ram(cache_size_mb):
                        print(f"✅ Using PreloadedCacheDataset for {task_name}_{split_name}")
                        print(f"   Cache: {num_cached} samples, ~{cache_size_mb:.0f} MB")
                        use_preloaded = True
                    else:
                        print(
                            f"⚠️  Insufficient RAM for PreloadedCacheDataset ({cache_size_mb:.0f} MB required)"
                        )
                        print("   Falling back to disk I/O mode (slower but memory-efficient)")
                else:
                    print(
                        f"⚠️  Cache incomplete for {task_name}_{split_name} ({num_cached}/{num_samples} samples)"
                    )
                    print("   Falling back to on-demand encoding (will populate cache)")

            if use_preloaded and ds_cache:
                return PreloadedCacheDataset(
                    cache_dir=ds_cache,
                    num_samples=len(dataset),
                    time_stride=time_stride,
                    rollout_horizon=rollout_horizon,
                )

            return GridLatentPairDataset(
                dataset,
                encoder,
                coords,
                grid_shape,
                field_name=field_name,
                task=task_name,
                device=device,
                cache_dir=ds_cache,
                cache_dtype=cache_dtype,
                time_stride=time_stride,
                rollout_horizon=rollout_horizon,
                use_inverse_losses=use_inverse_losses,
            )

        def _make_graph_latent_dataset(task_name: str, spec_kind: str) -> Dataset:
            data_root = resolve_pdebench_root(data_cfg.get("root"))
            zarr_path = data_root / f"{task_name}_{split_name}.zarr"
            if not zarr_path.exists():
                raise FileNotFoundError(f"Expected dataset at {zarr_path} for task '{task_name}'")

            if spec_kind == "mesh":
                base_dataset: Dataset = MeshZarrDataset(str(zarr_path), group=task_name)
            elif spec_kind == "particles":
                base_dataset = ParticleZarrDataset(str(zarr_path), group=task_name)
            else:
                raise ValueError(f"Unsupported spec kind '{spec_kind}' for graph dataset")

            hidden_dim = data_cfg.get("hidden_dim", max(latent_cfg.get("dim", 32) * 2, 64))
            encoder_cfg = MeshParticleEncoderConfig(
                latent_len=latent_cfg.get("tokens", 16),
                latent_dim=latent_cfg.get("dim", 32),
                hidden_dim=hidden_dim,
                message_passing_steps=data_cfg.get("message_passing_steps", 3),
                supernodes=data_cfg.get("supernodes", 2048),
                use_coords=data_cfg.get("use_coords", True),
            )
            graph_encoder = MeshParticleEncoder(encoder_cfg).eval().to(device)
            return GraphLatentPairDataset(
                base_dataset,
                graph_encoder,
                kind=spec_kind,
                rollout_horizon=rollout_horizon,
                task=task_name,
            )

        warned_parallel_mismatch = False

        for task_name in task_list:
            spec = get_pdebench_spec(task_name)
            if spec.kind == "grid":
                latent_datasets.append(_make_grid_latent_dataset(task_name))
            else:
                latent_datasets.append(_make_graph_latent_dataset(task_name, spec.kind))

        if use_parallel_encoding and len(task_list) > 1 and not warned_parallel_mismatch:
            print(
                "⚠️  use_parallel_encoding is enabled but cannot be applied when mixing multiple tasks."
            )
            warned_parallel_mismatch = True

        mixed = latent_datasets[0] if len(latent_datasets) == 1 else ConcatDataset(latent_datasets)
        loader_kwargs["collate_fn"] = latent_pair_collate

        # Check if distributed training is active
        import torch.distributed as dist

        is_distributed = dist.is_initialized()

        # For multi-task, use task-aware sampler in distributed mode
        if len(task_list) > 1 and is_distributed:
            from ups.data.task_samplers import MultiTaskDistributedSampler

            # Extract per-task sizes
            task_sizes = [len(ds) for ds in latent_datasets]
            sampler = MultiTaskDistributedSampler(
                task_sizes=task_sizes,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                shuffle=True,
                seed=train_cfg.get("seed", 0),
                drop_last=False,
            )
            loader_kwargs["shuffle"] = False  # Sampler handles shuffling
            loader_kwargs["sampler"] = sampler
        elif is_distributed:
            # Single-task distributed training
            from torch.utils.data.distributed import DistributedSampler

            sampler = DistributedSampler(
                mixed,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                shuffle=True,
                seed=train_cfg.get("seed", 0),
            )
            loader_kwargs["shuffle"] = False
            loader_kwargs["sampler"] = sampler
        # else: Single-GPU mode, use default shuffle=True

        if (
            use_parallel_encoding
            and len(task_list) == 1
            and isinstance(mixed, GridLatentPairDataset)
        ):
            maybe_loader = _maybe_build_parallel_loader(
                mixed, batch, num_workers, pin_memory, prefetch_factor
            )
            if maybe_loader is not None:
                return maybe_loader

        return DataLoader(mixed, **loader_kwargs)

    kind = data_cfg.get("kind")
    if kind == "grid":
        dataset = GridZarrDataset(data_cfg["path"], group=data_cfg.get("group"))
        if len(dataset) < 2:
            raise ValueError(
                "GridZarrDataset must contain at least two time steps to form latent pairs"
            )
        sample0 = dataset[0]
        field_channels = {name: tensor.shape[1] for name, tensor in sample0["fields"].items()}
        encoder_cfg = GridEncoderConfig(
            patch_size=data_cfg.get("patch_size", 4),
            latent_dim=latent_cfg.get("dim", 32),
            latent_len=latent_cfg.get("tokens", 16),
            field_channels=field_channels,
        )
        grid_encoder = GridEncoder(encoder_cfg).eval().to(device)
        latent_dataset = GridZarrLatentPairDataset(
            dataset, grid_encoder, rollout_horizon=rollout_horizon
        )
        if use_parallel_encoding:
            maybe_loader = _maybe_build_parallel_loader(
                latent_dataset, batch, num_workers, pin_memory, prefetch_factor
            )
            if maybe_loader is not None:
                return maybe_loader
        return DataLoader(latent_dataset, **loader_kwargs)

    if kind in {"mesh", "particles"}:
        path = data_cfg["path"]
        group = data_cfg.get("group")
        if kind == "mesh":
            base_dataset: Dataset = MeshZarrDataset(path, group=group or "mesh_poisson")
        else:
            base_dataset = ParticleZarrDataset(path, group=group or "particles_advect")

        if len(base_dataset) == 0:
            raise ValueError(f"{kind.capitalize()} dataset at {path} is empty")
        sample = base_dataset[0]
        if not sample.get("fields"):
            raise ValueError(
                f"{kind.capitalize()} dataset at {path} does not provide time-series fields; "
                "augment the dataset with 'fields' tensors shaped (T, N, C) before training."
            )

        hidden_dim = data_cfg.get("hidden_dim", max(latent_cfg.get("dim", 32) * 2, 64))
        encoder_cfg = MeshParticleEncoderConfig(
            latent_len=latent_cfg.get("tokens", 16),
            latent_dim=latent_cfg.get("dim", 32),
            hidden_dim=hidden_dim,
            message_passing_steps=data_cfg.get("message_passing_steps", 3),
            supernodes=data_cfg.get("supernodes", 2048),
            use_coords=data_cfg.get("use_coords", True),
        )
        graph_encoder = MeshParticleEncoder(encoder_cfg).eval().to(device)
        latent_dataset = GraphLatentPairDataset(
            base_dataset, graph_encoder, kind=kind, rollout_horizon=rollout_horizon
        )
        return DataLoader(latent_dataset, **loader_kwargs)

    raise ValueError("Data configuration must specify either 'task' or 'kind'")


def _infer_parallel_device() -> torch.device:
    if torch.cuda.is_available():
        local_rank = os.environ.get("LOCAL_RANK")
        try:
            idx = int(local_rank) if local_rank is not None else torch.cuda.current_device()
        except (ValueError, RuntimeError):
            idx = torch.cuda.current_device()
        idx = max(0, min(idx, torch.cuda.device_count() - 1))
        return torch.device(f"cuda:{idx}")
    return torch.device("cpu")


def _maybe_build_parallel_loader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int | None,
) -> DataLoader | None:
    if not isinstance(dataset, GridLatentPairDataset):
        return None
    if num_workers <= 0:
        return None
    try:
        from ups.data.parallel_cache import build_parallel_latent_loader
    except ImportError:
        return None

    device = _infer_parallel_device()
    coords = dataset.coords.to(device)
    return build_parallel_latent_loader(
        dataset.base,
        dataset.encoder,
        coords,
        dataset.grid_shape,
        dataset.field_name,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        cache_dir=dataset.cache_dir,
        cache_dtype=dataset.cache_dtype,
        time_stride=dataset.time_stride,
        rollout_horizon=dataset.rollout_horizon,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
    )


def latent_pair_collate(batch):
    """Custom collate function for LatentPair instances with optional fields.

    Handles the case where some fields (input_fields, coords, meta) may be None
    for some or all samples in the batch.

    Note: Each LatentPair may contain multiple pairs (num_pairs, tokens, dim) from
    a single trajectory. We concatenate along dim=0 to flatten into (batch*num_pairs, tokens, dim).
    """
    if not batch:
        raise ValueError("Empty batch")

    # Concatenate pairs from all samples (flattens num_pairs into batch dimension)
    z0 = torch.cat([item.z0 for item in batch], dim=0)
    z1 = torch.cat([item.z1 for item in batch], dim=0)

    # Collate conditioning dicts
    cond = collate_conditions([item.cond for item in batch])

    # Handle optional future field (concatenate to match z0/z1)
    futures = [item.future for item in batch if item.future is not None]
    future = torch.cat(futures, dim=0) if futures else None

    # Handle optional inverse loss fields
    # Only include in batch if at least one sample has them
    input_fields = None
    coords = None
    meta = None

    fields_list = [item.input_fields for item in batch if item.input_fields is not None]
    if fields_list:
        # Collate input_fields dicts
        # For multi-task batches, tasks may have different spatial dimensions
        # Skip collation if dimension mismatch (input_fields only used for inverse losses)
        try:
            field_names = set()
            for f in fields_list:
                field_names.update(f.keys())
            input_fields = {}
            for name in field_names:
                tensors = [f[name] for f in fields_list if name in f]
                if tensors:
                    input_fields[name] = torch.cat(tensors, dim=0)
        except RuntimeError as e:
            if "Sizes of tensors must match" in str(e):
                # Mixed task batch with different spatial dims - skip input_fields
                input_fields = None
            else:
                raise

    coords_list = [item.coords for item in batch if item.coords is not None]
    if coords_list:
        try:
            coords = torch.cat(coords_list, dim=0)
        except RuntimeError as e:
            if "Sizes of tensors must match" in str(e):
                # Mixed task batch with different spatial dims - skip coords
                coords = None
            else:
                raise

    # Meta is typically the same for all samples, just take the first
    meta_list = [item.meta for item in batch if item.meta is not None]
    if meta_list:
        meta = meta_list[0]

    # Collect task names per sample (replicate for each pair in trajectory)
    # Each item may have multiple pairs (z0 shape: (num_pairs, tokens, dim))
    # We need task_name repeated for each pair to match concatenated z0
    task_names = []
    for item in batch:
        if item.task_name is not None:
            num_pairs = item.z0.shape[0]  # Number of pairs in this trajectory
            task_names.extend([item.task_name] * num_pairs)
        else:
            # Backward compatibility: if task_name is None, add None for each pair
            num_pairs = item.z0.shape[0]
            task_names.extend([None] * num_pairs)

    # If all task names are None, set to None for backward compatibility
    if all(name is None for name in task_names):
        task_names = None

    # Return as dict for easy unpacking in training loop
    return {
        "z0": z0,
        "z1": z1,
        "cond": cond,
        "future": future,
        "input_fields": input_fields,
        "coords": coords,
        "meta": meta,
        "task_names": task_names,
    }


def unpack_batch(batch):
    """Unpack batch from DataLoader.

    Supports both legacy tuple format and new dict format from latent_pair_collate.
    """
    # New dict format (with optional inverse loss fields)
    if isinstance(batch, dict):
        # Return as dict to preserve optional fields for training loop
        return batch

    # Legacy tuple format (backward compatibility)
    if isinstance(batch, (list, tuple)):
        if len(batch) == 4 and isinstance(batch[2], dict):
            z0, z1, cond, future = batch
            return z0, z1, cond, future
        if len(batch) == 3 and isinstance(batch[2], dict):
            z0, z1, cond = batch
            return z0, z1, cond
        if len(batch) == 2:
            z0, z1 = batch
            return z0, z1, {}

    raise ValueError("Unexpected batch structure returned by DataLoader")
