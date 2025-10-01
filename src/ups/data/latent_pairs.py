from __future__ import annotations

"""Reusable latent pair datasets for grid, mesh, and particle sources."""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from ups.data.datasets import GridZarrDataset, MeshZarrDataset, ParticleZarrDataset
from ups.data.pdebench import PDEBenchConfig, PDEBenchDataset
from ups.io.enc_grid import GridEncoder, GridEncoderConfig
from ups.io.enc_mesh_particle import MeshParticleEncoder, MeshParticleEncoderConfig


def infer_grid_shape(fields: torch.Tensor) -> Tuple[int, int]:
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


def prepare_conditioning(
    params: Optional[Dict[str, Any]],
    bc: Optional[Dict[str, Any]],
    seq_len: int,
    *,
    time: Optional[Any] = None,
    dt: Optional[Any] = None,
    extras: Optional[Dict[str, Any]] = None,
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


def collate_conditions(conditions: Iterable[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    keys = set()
    for cond in conditions:
        keys.update(cond.keys())
    collated: Dict[str, torch.Tensor] = {}
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
    grid_shape: Tuple[int, int],
    *,
    field_name: str,
    params: Optional[Dict[str, torch.Tensor]] = None,
    bc: Optional[Dict[str, torch.Tensor]] = None,
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

    coord_batch = coords.expand(B, -1, -1)
    meta = {"grid_shape": grid_shape}
    field_inputs = {field_name: flattened}

    with torch.no_grad():
        latent = encoder(field_inputs, coord_batch, params=params, bc=bc, meta=meta)
    return latent


def _encode_grid_sample(encoder: GridEncoder, sample: Dict[str, Any]) -> torch.Tensor:
    meta = dict(sample.get("meta", {}))
    if "grid_shape" not in meta:
        raise ValueError("Grid sample meta must include 'grid_shape'")
    H, W = map(int, meta["grid_shape"])

    field_batches: Dict[str, torch.Tensor] = {}
    fields = sample.get("fields", {})
    if not fields:
        raise ValueError("Grid sample requires non-empty fields for encoding")
    for name, tensor in fields.items():
        if tensor.dim() != 2:
            raise ValueError(f"Field '{name}' expected shape (N, C); got {tensor.shape}")
        field_batches[name] = tensor.unsqueeze(0).float()

    coords = sample["coords"].float()
    if coords.dim() == 2:
        coords_batch = coords.unsqueeze(0)
    elif coords.dim() == 3:
        coords_batch = coords
    else:
        raise ValueError("Coords tensor must have shape (N, d) or (B, N, d)")

    meta.setdefault("grid_shape", (H, W))

    with torch.no_grad():
        latent = encoder(field_batches, coords_batch, meta=meta)
    return latent.squeeze(0)


@dataclass
class LatentPair:
    z0: torch.Tensor
    z1: torch.Tensor
    cond: Dict[str, torch.Tensor]


class GridLatentPairDataset(Dataset):
    """Wrap a PDEBench dataset and emit latent (t, t+1) token pairs."""

    def __init__(
        self,
        base: PDEBenchDataset,
        encoder: GridEncoder,
        coords: torch.Tensor,
        grid_shape: Tuple[int, int],
        field_name: str = "u",
    ) -> None:
        super().__init__()
        self.base = base
        self.encoder = encoder
        self.coords = coords
        self.grid_shape = grid_shape
        self.field_name = field_name

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> LatentPair:
        sample = self.base[idx]
        fields = sample["fields"].float()
        params = sample.get("params")
        bc = sample.get("bc")
        latent_seq = _fields_to_latent_batch(
            self.encoder,
            fields,
            self.coords,
            self.grid_shape,
            params=params,
            bc=bc,
            field_name=self.field_name,
        )
        if latent_seq.shape[0] < 2:
            raise ValueError("Need at least two time steps to form latent pairs")
        pair_len = latent_seq.shape[0] - 1
        cond = prepare_conditioning(params, bc, pair_len)
        return LatentPair(latent_seq[:-1], latent_seq[1:], cond)


class GridZarrLatentPairDataset(Dataset):
    """Encode consecutive time steps from a GridZarrDataset into latent pairs."""

    def __init__(self, base: GridZarrDataset, encoder: GridEncoder) -> None:
        super().__init__()
        self.base = base
        self.encoder = encoder

    def __len__(self) -> int:
        return max(len(self.base) - 1, 0)

    def __getitem__(self, idx: int) -> LatentPair:
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        sample0 = self.base[idx]
        sample1 = self.base[idx + 1]

        latent0 = _encode_grid_sample(self.encoder, sample0).unsqueeze(0)
        latent1 = _encode_grid_sample(self.encoder, sample1).unsqueeze(0)
        cond = prepare_conditioning(
            sample0.get("params"),
            sample0.get("bc"),
            1,
            time=sample0.get("time"),
            dt=sample0.get("dt"),
        )
        return LatentPair(latent0, latent1, cond)


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
            step_fields[name] = tensor.float()

        if time_dim is None or time_dim < 2:
            raise ValueError("Need at least two time steps to form latent pairs")

        latents: List[torch.Tensor] = []
        for step in range(time_dim):
            field_batch = {name: tensor[step].unsqueeze(0) for name, tensor in step_fields.items()}
            coords = _graph_coords_at_time(sample, step_fields, step)
            connect = sample.get("connect")
            if connect is not None:
                connect = connect.to(coords.device, dtype=torch.long)
            meta = sample.get("meta")
            with torch.no_grad():
                latent_step = self.encoder(field_batch, coords, connect=connect, meta=meta)
            latents.append(latent_step.squeeze(0))

        latent_seq = torch.stack(latents, dim=0)
        cond = prepare_conditioning(
            sample.get("params"),
            sample.get("bc"),
            time_dim - 1,
            time=sample.get("time"),
            dt=sample.get("dt"),
        )
        return LatentPair(latent_seq[:-1], latent_seq[1:], cond)


def collate_latent_pairs(batch_items: Iterable[LatentPair]) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    z0_chunks = []
    z1_chunks = []
    cond_list = []
    for item in batch_items:
        z0_chunks.append(item.z0)
        z1_chunks.append(item.z1)
        cond_list.append(item.cond)
    z0 = torch.cat(z0_chunks, dim=0)
    z1 = torch.cat(z1_chunks, dim=0)
    cond = collate_conditions(cond_list)
    return z0, z1, cond


def _build_pdebench_dataset(data_cfg: Dict[str, Any]) -> Tuple[Dataset, GridEncoder, Tuple[int, int], str]:
    dataset = PDEBenchDataset(
        PDEBenchConfig(task=data_cfg["task"], split=data_cfg.get("split", "train"), root=data_cfg.get("root")),
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


def build_latent_pair_loader(cfg: Dict[str, Any]) -> DataLoader:
    """Construct a DataLoader that yields latent pairs from data config."""

    data_cfg = cfg.get("data", {})
    latent_cfg = cfg.get("latent", {})
    train_cfg = cfg.get("training", {})
    batch = train_cfg.get("batch_size", 16)

    if data_cfg.get("task"):
        dataset, encoder, grid_shape, field_name = _build_pdebench_dataset(
            {**data_cfg, "latent_dim": latent_cfg.get("dim", 32), "latent_len": latent_cfg.get("tokens", 16)}
        )
        coords = make_grid_coords(grid_shape, torch.device("cpu"))
        latent_dataset = GridLatentPairDataset(dataset, encoder, coords, grid_shape, field_name=field_name)
        return DataLoader(latent_dataset, batch_size=batch, shuffle=True, collate_fn=collate_latent_pairs)

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
        grid_encoder = GridEncoder(encoder_cfg).eval()
        latent_dataset = GridZarrLatentPairDataset(dataset, grid_encoder)
        return DataLoader(latent_dataset, batch_size=batch, shuffle=True, collate_fn=collate_latent_pairs)

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
        graph_encoder = MeshParticleEncoder(encoder_cfg).eval()
        latent_dataset = GraphLatentPairDataset(base_dataset, graph_encoder, kind=kind)
        return DataLoader(latent_dataset, batch_size=batch, shuffle=True, collate_fn=collate_latent_pairs)

    raise ValueError("Data configuration must specify either 'task' or 'kind'")


def unpack_batch(batch) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    if isinstance(batch, (list, tuple)) and len(batch) == 3 and isinstance(batch[2], dict):
        z0, z1, cond = batch
        return z0, z1, cond
    if isinstance(batch, (list, tuple)) and len(batch) == 2:
        z0, z1 = batch
        return z0, z1, {}
    raise ValueError("Unexpected batch structure returned by DataLoader")
