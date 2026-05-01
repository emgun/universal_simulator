from __future__ import annotations

"""Reusable latent pair datasets for grid, mesh, and particle sources."""

import io
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import ConcatDataset

from ups.data.datasets import GridZarrDataset, MeshZarrDataset, ParticleZarrDataset
from ups.data.pdebench import (
    PDEBenchConfig,
    PDEBenchDataset,
    pdebench_equation_nodes,
    pdebench_equation_signature,
    pdebench_task_semantics,
)
from ups.io.enc_grid import GridEncoder, GridEncoderConfig
from ups.io.enc_mesh_particle import MeshParticleEncoder, MeshParticleEncoderConfig


def infer_grid_shape(fields: torch.Tensor) -> Tuple[int, int]:
    """Return (H, W) for a PDEBench sample tensor."""

    if fields.dim() == 4:  # (T, H, W, C) or (T, C, H, W)
        shape = fields.shape
        if shape[1] <= 4 and (shape[1] < shape[2] or shape[1] < shape[3]):
            return int(shape[-2]), int(shape[-1])
        if shape[-1] <= 4 and (shape[-1] < shape[1] or shape[-1] < shape[2]):
            return int(shape[1]), int(shape[2])
        if shape[-1] <= 8:
            return int(shape[-3]), int(shape[-2])
        return int(shape[-2]), int(shape[-1])
    if fields.dim() == 3:  # (T, X, C) or (T, C, X)
        shape = fields.shape
        if shape[1] <= 4 and shape[1] < shape[2]:
            return 1, int(shape[-1])
        if shape[-1] <= 4 and shape[-1] < shape[1]:
            return 1, int(shape[-2])
        if shape[-1] <= 8:
            return 1, int(shape[-2])
        return 1, int(shape[-1])
    if fields.dim() == 2:  # (T, X)
        return 1, int(fields.shape[-1])
    raise ValueError(f"Unsupported PDEBench field shape {tuple(fields.shape)}")


def infer_channel_count(fields: torch.Tensor, grid_shape: Tuple[int, int]) -> int:
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


def make_grid_coords(grid_shape: Tuple[int, int], device: torch.device) -> torch.Tensor:
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


def _broadcast_static_condition_tensor(tensor: torch.Tensor, seq_len: int) -> torch.Tensor:
    tensor = tensor.float()
    if tensor.dim() == 0:
        return tensor.view(1, 1).expand(seq_len, 1).contiguous()
    if tensor.dim() == 1:
        return tensor.reshape(1, -1).expand(seq_len, -1).contiguous()
    return tensor.unsqueeze(0).expand(seq_len, *tensor.shape).contiguous()


def _to_float_tensor(value: Any) -> Optional[torch.Tensor]:
    if torch.is_tensor(value):
        return value.float()
    try:
        tensor = torch.as_tensor(value, dtype=torch.float32)
    except (TypeError, ValueError):
        return None
    if tensor.numel() == 0:
        return None
    return tensor.float()


def _presence_signature(vocab: Sequence[str], mapping: Optional[Mapping[str, Any]]) -> torch.Tensor:
    presence = torch.tensor(
        [1.0 if mapping is not None and key in mapping else 0.0 for key in vocab],
        dtype=torch.float32,
    )
    density = presence.sum().view(1) / max(float(len(vocab)), 1.0)
    return torch.cat([presence, density], dim=0)


def _presence_nodes(vocab: Sequence[str], mapping: Optional[Mapping[str, Any]]) -> torch.Tensor:
    size = len(vocab)
    if size == 0:
        return torch.zeros((0, 0), dtype=torch.float32)
    eye = torch.eye(size, dtype=torch.float32)
    presence = torch.tensor(
        [1.0 if mapping is not None and key in mapping else 0.0 for key in vocab],
        dtype=torch.float32,
    ).view(size, 1)
    return torch.cat([eye, presence], dim=-1)


def _expand_batch_static_condition_tensor(tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
    tensor = tensor.float()
    if tensor.dim() == 0:
        return tensor.view(1, 1).expand(batch_size, 1).contiguous()
    if tensor.dim() == 1:
        return tensor.view(1, -1).expand(batch_size, -1).contiguous()
    return tensor.unsqueeze(0).expand(batch_size, *tensor.shape).contiguous()


def prepare_conditioning(
    params: Optional[Dict[str, Any]],
    bc: Optional[Dict[str, Any]],
    seq_len: int,
    *,
    time: Optional[Any] = None,
    dt: Optional[Any] = None,
    extras: Optional[Dict[str, Any]] = None,
    param_vocab: Optional[Sequence[str]] = None,
    bc_vocab: Optional[Sequence[str]] = None,
) -> Dict[str, torch.Tensor]:
    cond: Dict[str, torch.Tensor] = {}

    def ingest(prefix: str, mapping: Optional[Dict[str, Any]]) -> None:
        if not mapping:
            return
        for key, value in mapping.items():
            tensor = _to_float_tensor(value)
            if tensor is None:
                continue
            cond[f"{prefix}_{key}"] = _broadcast_condition_tensor(tensor, seq_len)

    ingest("param", params)
    ingest("bc", bc)

    if param_vocab:
        param_presence = _presence_signature(param_vocab, params)[:-1]
        cond["param_presence"] = _broadcast_static_condition_tensor(param_presence, seq_len)
        cond["parameter_signature"] = _broadcast_static_condition_tensor(
            _presence_signature(param_vocab, params),
            seq_len,
        )
        cond["parameter_nodes"] = _broadcast_static_condition_tensor(_presence_nodes(param_vocab, params), seq_len)

    if bc_vocab:
        bc_presence = _presence_signature(bc_vocab, bc)[:-1]
        cond["bc_presence"] = _broadcast_static_condition_tensor(bc_presence, seq_len)
        cond["boundary_signature"] = _broadcast_static_condition_tensor(
            _presence_signature(bc_vocab, bc),
            seq_len,
        )
        cond["boundary_nodes"] = _broadcast_static_condition_tensor(_presence_nodes(bc_vocab, bc), seq_len)

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
            cond[key] = _broadcast_static_condition_tensor(tensor, seq_len)
    return cond


def collate_conditions(
    conditions: Iterable[Dict[str, torch.Tensor]],
    *,
    lengths: Optional[Sequence[int]] = None,
) -> Dict[str, torch.Tensor]:
    conditions = list(conditions)
    keys = set()
    for cond in conditions:
        keys.update(cond.keys())
    if lengths is None:
        inferred: List[int] = []
        for cond in conditions:
            if cond:
                exemplar = next(iter(cond.values()))
                inferred.append(int(exemplar.shape[0]))
            else:
                inferred.append(0)
        lengths = inferred
    collated: Dict[str, torch.Tensor] = {}
    for key in keys:
        exemplar = next((cond[key] for cond in conditions if key in cond), None)
        if exemplar is None:
            continue
        chunks: List[torch.Tensor] = []
        for idx, cond in enumerate(conditions):
            if key in cond:
                chunks.append(cond[key])
                continue
            pad_len = int(lengths[idx]) if lengths is not None else 0
            if pad_len <= 0:
                continue
            chunks.append(exemplar.new_zeros((pad_len, *exemplar.shape[1:])))
        if chunks:
            collated[key] = torch.cat(chunks, dim=0)
    return collated


def pdebench_conditioning_extras(
    *,
    task_name: Optional[str],
    grid_shape: Tuple[int, int],
    task_vocab: Optional[Sequence[str]] = None,
) -> Dict[str, torch.Tensor]:
    H, W = grid_shape
    extras: Dict[str, torch.Tensor] = {
        "resolution": torch.tensor([float(H), float(W)], dtype=torch.float32),
        "spatial_dims": torch.tensor([2.0 if H > 1 else 1.0], dtype=torch.float32),
    }
    if task_name is not None:
        extras.update(pdebench_task_semantics(task_name, task_vocab=task_vocab))
        extras["equation_signature"] = pdebench_equation_signature(task_name)
        extras["equation_nodes"] = pdebench_equation_nodes(task_name)
    return extras


def conditioning_source_dims_from_sample(
    sample: Mapping[str, Any],
    *,
    grid_shape: Tuple[int, int],
    task_name: Optional[str] = None,
    task_vocab: Optional[Sequence[str]] = None,
    param_vocab: Optional[Sequence[str]] = None,
    bc_vocab: Optional[Sequence[str]] = None,
) -> Dict[str, int]:
    fields = sample["fields"]
    seq_len = max(int(fields.shape[0]) - 1, 1)
    extras = pdebench_conditioning_extras(task_name=task_name, grid_shape=grid_shape, task_vocab=task_vocab)
    cond = prepare_conditioning(
        sample.get("params"),
        sample.get("bc"),
        seq_len,
        extras=extras,
        param_vocab=param_vocab,
        bc_vocab=bc_vocab,
    )
    return {key: int(value.shape[-1]) for key, value in cond.items()}


def _condition_step_tensor(tensor: torch.Tensor, *, batch_size: int, step: int) -> torch.Tensor:
    tensor = tensor.float()
    if tensor.dim() == 0:
        return tensor.view(1, 1).expand(batch_size, 1).contiguous()
    if tensor.shape[0] == batch_size:
        if tensor.dim() == 1:
            return tensor.view(batch_size, 1).contiguous()
        if tensor.dim() >= 2 and tensor.shape[1] > step:
            selected = tensor[:, step]
            if selected.dim() == 1:
                return selected.unsqueeze(-1).contiguous()
            return selected.reshape(batch_size, -1).contiguous()
        return tensor.reshape(batch_size, -1).contiguous()
    if batch_size == 1 and tensor.dim() >= 1 and tensor.shape[0] > step:
        selected = tensor[step]
        if selected.dim() == 0:
            return selected.view(1, 1).contiguous()
        return selected.reshape(1, -1).contiguous()
    return tensor.reshape(1, -1).expand(batch_size, -1).contiguous()


def pdebench_condition_step(
    params: Optional[Mapping[str, Any]],
    bc: Optional[Mapping[str, Any]],
    *,
    batch_size: int,
    step: int,
    extras: Optional[Mapping[str, Any]] = None,
    param_vocab: Optional[Sequence[str]] = None,
    bc_vocab: Optional[Sequence[str]] = None,
) -> Dict[str, torch.Tensor]:
    cond: Dict[str, torch.Tensor] = {}
    if params:
        for key, value in params.items():
            tensor = _to_float_tensor(value)
            if tensor is None:
                continue
            cond[f"param_{key}"] = _condition_step_tensor(tensor, batch_size=batch_size, step=step)
    if bc:
        for key, value in bc.items():
            tensor = _to_float_tensor(value)
            if tensor is None:
                continue
            cond[f"bc_{key}"] = _condition_step_tensor(tensor, batch_size=batch_size, step=step)
    if param_vocab:
        param_signature = _presence_signature(param_vocab, params)
        param_presence = param_signature[:-1]
        cond["param_presence"] = param_presence.view(1, -1).expand(batch_size, -1).contiguous()
        cond["parameter_signature"] = param_signature.view(1, -1).expand(batch_size, -1).contiguous()
        cond["parameter_nodes"] = _expand_batch_static_condition_tensor(_presence_nodes(param_vocab, params), batch_size)
    if bc_vocab:
        bc_signature = _presence_signature(bc_vocab, bc)
        bc_presence = bc_signature[:-1]
        cond["bc_presence"] = bc_presence.view(1, -1).expand(batch_size, -1).contiguous()
        cond["boundary_signature"] = bc_signature.view(1, -1).expand(batch_size, -1).contiguous()
        cond["boundary_nodes"] = _expand_batch_static_condition_tensor(_presence_nodes(bc_vocab, bc), batch_size)
    if extras:
        for key, value in extras.items():
            tensor = _to_float_tensor(value)
            if tensor is None:
                continue
            cond[key] = _expand_batch_static_condition_tensor(tensor, batch_size)
    return cond


def _fields_to_latent_batch(
    encoder: GridEncoder,
    fields: torch.Tensor,
    coords: torch.Tensor,
    grid_shape: Tuple[int, int],
    *,
    field_name: str,
    params: Optional[Dict[str, torch.Tensor]] = None,
    bc: Optional[Dict[str, torch.Tensor]] = None,
) -> torch.Tensor:
    T = fields.shape[0]
    H, W = grid_shape

    if fields.dim() == 4:
        if tuple(fields.shape[-2:]) == (H, W):
            data = fields
        elif tuple(fields.shape[1:3]) == (H, W):
            data = fields.permute(0, 3, 1, 2)
        else:
            if fields.shape[-1] <= 8:
                data = fields.permute(0, 3, 1, 2)
            else:
                data = fields
    elif fields.dim() == 3:
        if tuple(fields.shape[-2:]) == (H, W):
            data = fields.unsqueeze(1)
        elif H == 1 and fields.shape[-1] == W and fields.shape[1] != W:
            data = fields.unsqueeze(2)
        elif H == 1 and fields.shape[1] == W:
            data = fields.permute(0, 2, 1).unsqueeze(2)
        else:
            if fields.shape[-1] <= 8:
                data = fields.permute(0, 2, 1).unsqueeze(2)
            else:
                data = fields.unsqueeze(2)
    elif fields.dim() == 2:
        data = fields.unsqueeze(1).unsqueeze(2)
    else:
        raise ValueError(f"Unsupported PDEBench field shape {tuple(fields.shape)}")

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

    params_device: Optional[Dict[str, torch.Tensor]] = None
    bc_device: Optional[Dict[str, torch.Tensor]] = None
    if params is not None:
        params_device = {k: v.to(encoder_device, non_blocking=True) for k, v in params.items()}
    if bc is not None:
        bc_device = {k: v.to(encoder_device, non_blocking=True) for k, v in bc.items()}

    meta = {"grid_shape": grid_shape}
    field_inputs = {field_name: flattened}

    with torch.no_grad():
        latent = encoder(field_inputs, coord_batch, params=params_device, bc=bc_device, meta=meta)
    return latent.cpu()


def _encode_grid_sample(encoder: GridEncoder, sample: Dict[str, Any], device: Optional[torch.device] = None) -> torch.Tensor:
    meta = dict(sample.get("meta", {}))
    if "grid_shape" not in meta:
        raise ValueError("Grid sample meta must include 'grid_shape'")
    H, W = map(int, meta["grid_shape"])

    field_batches: Dict[str, torch.Tensor] = {}
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
    cond: Dict[str, torch.Tensor]
    z_seq: Optional[torch.Tensor] = None


class GridLatentPairDataset(Dataset):
    """Wrap a PDEBench dataset and emit latent (t, t+1) token pairs."""

    def __init__(
        self,
        base: PDEBenchDataset,
        encoder: GridEncoder,
        coords: torch.Tensor,
        grid_shape: Tuple[int, int],
        field_name: str = "u",
        conditioning_extras: Optional[Mapping[str, Any]] = None,
        param_vocab: Optional[Sequence[str]] = None,
        bc_vocab: Optional[Sequence[str]] = None,
        *,
        device: torch.device | None = None,
        cache_dir: Optional[Path] = None,
        cache_dtype: Optional[torch.dtype] = torch.float16,
        time_stride: int = 1,
    ) -> None:
        super().__init__()
        self.base = base
        self.encoder = encoder
        self.device = device or torch.device("cpu")
        self.coords = coords.to(self.device)
        self.grid_shape = grid_shape
        self.field_name = field_name
        self.conditioning_extras = dict(conditioning_extras or {})
        self.param_vocab = tuple(param_vocab if param_vocab is not None else getattr(base, "param_keys", ()))
        self.bc_vocab = tuple(bc_vocab if bc_vocab is not None else getattr(base, "bc_keys", ()))
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dtype = cache_dtype
        self.time_stride = max(1, int(time_stride))

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> LatentPair:
        cache_hit = False
        latent_seq: Optional[torch.Tensor] = None
        params_cpu = None
        bc_cpu = None
        if self.cache_dir:
            cache_path = self.cache_dir / f"sample_{idx:05d}.pt"
            if cache_path.exists():
                try:
                    data = torch.load(cache_path, map_location="cpu")
                except (RuntimeError, EOFError):  # corrupted file
                    cache_path.unlink(missing_ok=True)
                else:
                    latent_seq = data["latent"].float()
                    params_cpu = data.get("params")
                    bc_cpu = data.get("bc")
                    cache_hit = True

        if latent_seq is None:
            sample = self.base[idx]
            fields_cpu = sample["fields"].float()
            params_cpu = sample.get("params")
            bc_cpu = sample.get("bc")
            fields = fields_cpu.to(self.device, non_blocking=True)
            params_device = None
            if params_cpu is not None:
                params_device = {k: v.to(self.device, non_blocking=True) for k, v in params_cpu.items()}
            bc_device = None
            if bc_cpu is not None:
                bc_device = {k: v.to(self.device, non_blocking=True) for k, v in bc_cpu.items()}
            latent_seq = _fields_to_latent_batch(
                self.encoder,
                fields,
                self.coords,
                self.grid_shape,
                params=params_device,
                bc=bc_device,
                field_name=self.field_name,
            )
            if self.cache_dir and not cache_hit:
                to_store = latent_seq.to(self.cache_dtype) if self.cache_dtype is not None else latent_seq
                tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
                tmp_path.unlink(missing_ok=True)
                payload = {
                    "latent": to_store.cpu(),
                    "params": params_cpu,
                    "bc": bc_cpu,
                }
                buffer = io.BytesIO()
                torch.save(payload, buffer)
                tmp_path.write_bytes(buffer.getvalue())
                tmp_path.replace(cache_path)

        # Optionally downsample the time dimension to accelerate epochs
        if self.time_stride > 1:
            latent_seq = latent_seq[:: self.time_stride]

        if latent_seq.shape[0] < 2:
            raise ValueError("Need at least two time steps to form latent pairs")
        pair_len = latent_seq.shape[0] - 1
        cond = prepare_conditioning(
            params_cpu,
            bc_cpu,
            pair_len,
            extras=self.conditioning_extras,
            param_vocab=self.param_vocab,
            bc_vocab=self.bc_vocab,
        )
        return LatentPair(latent_seq[:-1], latent_seq[1:], cond, z_seq=latent_seq)


class GridZarrLatentPairDataset(Dataset):
    """Encode consecutive time steps from a GridZarrDataset into latent pairs."""

    def __init__(self, base: GridZarrDataset, encoder: GridEncoder) -> None:
        super().__init__()
        self.base = base
        self.encoder = encoder
        params = list(encoder.parameters())
        self.device = params[0].device if params else torch.device("cpu")

    def __len__(self) -> int:
        return max(len(self.base) - 1, 0)

    def __getitem__(self, idx: int) -> LatentPair:
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        sample0 = self.base[idx]
        sample1 = self.base[idx + 1]

        latent0 = _encode_grid_sample(self.encoder, sample0, self.device).unsqueeze(0)
        latent1 = _encode_grid_sample(self.encoder, sample1, self.device).unsqueeze(0)
        cond = prepare_conditioning(
            sample0.get("params"),
            sample0.get("bc"),
            1,
            time=sample0.get("time"),
            dt=sample0.get("dt"),
        )
        return LatentPair(latent0, latent1, cond, z_seq=torch.cat([latent0, latent1], dim=0))


def _graph_coords_at_time(
    sample: Dict[str, Any],
    step_fields: Dict[str, torch.Tensor],
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

    def __init__(self, base: Dataset, encoder: MeshParticleEncoder, kind: str = "mesh") -> None:
        super().__init__()
        self.base = base
        self.encoder = encoder
        self.kind = kind
        params = list(encoder.parameters())
        self.device = params[0].device if params else torch.device("cpu")

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> LatentPair:
        sample = self.base[idx]
        fields = sample.get("fields", {})
        if not fields:
            raise ValueError("Graph samples must provide fields with time dimension")

        step_fields: Dict[str, torch.Tensor] = {}
        time_dim: Optional[int] = None
        for name, tensor in fields.items():
            if tensor.dim() == 2:
                tensor = tensor.unsqueeze(0)
            if tensor.dim() != 3:
                raise ValueError(f"Field '{name}' must have shape (T, N, C); got {tensor.shape}")
            if time_dim is None:
                time_dim = tensor.shape[0]
            elif tensor.shape[0] != time_dim:
                raise ValueError("All fields must share the same time dimension")
            step_fields[name] = tensor.float().to(self.device, non_blocking=True)

        if time_dim is None or time_dim < 2:
            raise ValueError("Need at least two time steps to form latent pairs")

        latents: List[torch.Tensor] = []
        for step in range(time_dim):
            field_batch = {name: tensor[step].unsqueeze(0) for name, tensor in step_fields.items()}
            coords = _graph_coords_at_time(sample, step_fields, step).to(self.device, non_blocking=True)
            connect = sample.get("connect")
            if connect is not None:
                connect = connect.to(self.device, dtype=torch.long, non_blocking=True)
            meta = sample.get("meta")
            with torch.no_grad():
                latent_step = self.encoder(field_batch, coords, connect=connect, meta=meta)
            latents.append(latent_step.squeeze(0).cpu())

        latent_seq = torch.stack(latents, dim=0)
        cond = prepare_conditioning(
            sample.get("params"),
            sample.get("bc"),
            time_dim - 1,
            time=sample.get("time"),
            dt=sample.get("dt"),
        )
        return LatentPair(latent_seq[:-1], latent_seq[1:], cond, z_seq=latent_seq)


def collate_latent_pairs(batch_items: Iterable[LatentPair]) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    items = list(batch_items)
    z0_chunks = []
    z1_chunks = []
    cond_list = []
    lengths = []
    for item in items:
        z0_chunks.append(item.z0)
        z1_chunks.append(item.z1)
        cond_list.append(item.cond)
        lengths.append(int(item.z0.shape[0]))
    z0 = torch.cat(z0_chunks, dim=0)
    z1 = torch.cat(z1_chunks, dim=0)
    cond = collate_conditions(cond_list, lengths=lengths)
    return z0, z1, cond


def _pad_sequence_tensors(tensors: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    if not tensors:
        raise ValueError("Expected at least one tensor sequence to pad")
    lengths = torch.tensor([tensor.shape[0] for tensor in tensors], dtype=torch.long)
    max_len = int(lengths.max().item())
    padded = tensors[0].new_zeros((len(tensors), max_len, *tensors[0].shape[1:]))
    for idx, tensor in enumerate(tensors):
        padded[idx, : tensor.shape[0]] = tensor
    return padded, lengths


def _pad_condition_sequences(conditions: List[Dict[str, torch.Tensor]], max_len: int) -> Dict[str, torch.Tensor]:
    keys = set()
    for cond in conditions:
        keys.update(cond.keys())
    padded: Dict[str, torch.Tensor] = {}
    for key in keys:
        exemplar = next(cond[key] for cond in conditions if key in cond)
        bucket = exemplar.new_zeros((len(conditions), max_len, *exemplar.shape[1:]))
        for idx, cond in enumerate(conditions):
            if key not in cond:
                continue
            value = cond[key]
            bucket[idx, : value.shape[0]] = value
        padded[key] = bucket
    return padded


def collate_latent_pairs_with_sequences(batch_items: Iterable[LatentPair]) -> Dict[str, Any]:
    items = list(batch_items)
    z0, z1, cond = collate_latent_pairs(items)
    sequences = [item.z_seq if item.z_seq is not None else torch.cat([item.z0, item.z1[-1:]], dim=0) for item in items]
    z_seq, seq_lens = _pad_sequence_tensors(sequences)
    cond_seq = _pad_condition_sequences([item.cond for item in items], max(int(seq_lens.max().item()) - 1, 0))
    return {
        "z0": z0,
        "z1": z1,
        "cond": cond,
        "cond_seq": cond_seq,
        "z_seq": z_seq,
        "seq_lens": seq_lens,
    }


def _build_pdebench_dataset(data_cfg: Dict[str, Any]) -> Tuple[Dataset, GridEncoder, Tuple[int, int], str]:
    dataset = PDEBenchDataset(
        PDEBenchConfig(
            task=data_cfg["task"],
            split=data_cfg.get("split", "train"),
            root=data_cfg.get("root"),
            param_keys=tuple(data_cfg.get("param_keys", ())),
            bc_keys=tuple(data_cfg.get("bc_keys", ())),
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


def build_latent_pair_loader(cfg: Dict[str, Any], *, encoder_override: Optional[Any] = None) -> DataLoader:
    """Construct a DataLoader that yields latent pairs from data config."""

    data_cfg = cfg.get("data", {})
    latent_cfg = cfg.get("latent", {})
    train_cfg = cfg.get("training", {})
    batch = train_cfg.get("batch_size", 16)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    default_workers = max(1, (os.cpu_count() or 4) // 4)
    num_workers = int(train_cfg.get("num_workers", default_workers))
    pin_memory = bool(train_cfg.get("pin_memory", torch.cuda.is_available()))
    prefetch_factor = train_cfg.get("prefetch_factor")
    cache_dir_cfg = train_cfg.get("latent_cache_dir")
    cache_root: Optional[Path] = Path(cache_dir_cfg) if cache_dir_cfg else None
    cache_dtype_str = train_cfg.get("latent_cache_dtype", "float16") or None
    cache_dtype = getattr(torch, cache_dtype_str) if cache_dtype_str and hasattr(torch, cache_dtype_str) else None
    split_name = data_cfg.get("split", "train")
    time_stride = int(train_cfg.get("time_stride", 1))

    preserve_sequences = bool(train_cfg.get("preserve_sequences", False))

    loader_kwargs: Dict[str, Any] = {
        "batch_size": batch,
        "shuffle": True,
        "collate_fn": collate_latent_pairs_with_sequences if preserve_sequences else collate_latent_pairs,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }
    if prefetch_factor is not None and num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    # Support single task or a list of tasks for multi-dataset mixing
    tasks = data_cfg.get("task")
    if tasks:
        auto_conditioning = bool(train_cfg.get("auto_conditioning", False))
        if isinstance(tasks, (list, tuple)):
            datasets: List[Dataset] = []
            task_vocab = tuple(str(task_name) for task_name in tasks)
            for task_name in tasks:
                ds_cfg = {**data_cfg, "task": task_name}
                dataset, encoder, grid_shape, field_name = _build_pdebench_dataset(
                    {**ds_cfg, "latent_dim": latent_cfg.get("dim", 32), "latent_len": latent_cfg.get("tokens", 16)}
                )
                if encoder_override is not None:
                    encoder = encoder_override
                encoder = encoder.to(device)
                coords = make_grid_coords(grid_shape, device)
                ds_cache = cache_root / f"{task_name}_{split_name}" if cache_root else None
                extras = (
                    pdebench_conditioning_extras(task_name=str(task_name), grid_shape=grid_shape, task_vocab=task_vocab)
                    if auto_conditioning
                    else None
                )
                latent_ds = GridLatentPairDataset(
                    dataset,
                    encoder,
                    coords,
                    grid_shape,
                    field_name=field_name,
                    conditioning_extras=extras,
                    device=device,
                    cache_dir=ds_cache,
                    cache_dtype=cache_dtype,
                    time_stride=time_stride,
                )
                datasets.append(latent_ds)
            mixed = ConcatDataset(datasets)
            return DataLoader(mixed, **loader_kwargs)
        else:
            dataset, encoder, grid_shape, field_name = _build_pdebench_dataset(
                {**data_cfg, "latent_dim": latent_cfg.get("dim", 32), "latent_len": latent_cfg.get("tokens", 16)}
            )
            if encoder_override is not None:
                encoder = encoder_override
            encoder = encoder.to(device)
            coords = make_grid_coords(grid_shape, device)
            ds_cache = cache_root / f"{tasks}_{split_name}" if cache_root and isinstance(tasks, str) else cache_root
            extras = (
                pdebench_conditioning_extras(task_name=str(tasks), grid_shape=grid_shape)
                if auto_conditioning and isinstance(tasks, str)
                else None
            )
            latent_dataset = GridLatentPairDataset(
                dataset,
                encoder,
                coords,
                grid_shape,
                field_name=field_name,
                conditioning_extras=extras,
                device=device,
                cache_dir=ds_cache,
                cache_dtype=cache_dtype,
                time_stride=time_stride,
            )
            return DataLoader(latent_dataset, **loader_kwargs)

    kind = data_cfg.get("kind")
    if kind == "grid":
        dataset = GridZarrDataset(data_cfg["path"], group=data_cfg.get("group"))
        if len(dataset) < 2:
            raise ValueError("GridZarrDataset must contain at least two time steps to form latent pairs")
        sample0 = dataset[0]
        field_channels = {name: tensor.shape[1] for name, tensor in sample0["fields"].items()}
        encoder_cfg = GridEncoderConfig(
            patch_size=data_cfg.get("patch_size", 4),
            latent_dim=latent_cfg.get("dim", 32),
            latent_len=latent_cfg.get("tokens", 16),
            field_channels=field_channels,
        )
        grid_encoder = GridEncoder(encoder_cfg).eval().to(device)
        if encoder_override is not None:
            grid_encoder = encoder_override.to(device)
        latent_dataset = GridZarrLatentPairDataset(dataset, grid_encoder)
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
        latent_dataset = GraphLatentPairDataset(base_dataset, graph_encoder, kind=kind)
        return DataLoader(latent_dataset, **loader_kwargs)

    raise ValueError("Data configuration must specify either 'task' or 'kind'")


def unpack_batch(batch) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    if isinstance(batch, (list, tuple)) and len(batch) == 3 and isinstance(batch[2], dict):
        z0, z1, cond = batch
        return z0, z1, cond
    if isinstance(batch, (list, tuple)) and len(batch) == 2:
        z0, z1 = batch
        return z0, z1, {}
    if isinstance(batch, dict):
        return batch["z0"], batch["z1"], batch.get("cond", {})
    raise ValueError("Unexpected batch structure returned by DataLoader")
