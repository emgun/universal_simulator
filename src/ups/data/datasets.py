from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .schemas import Sample


def _flatten_grid(arr: np.ndarray) -> np.ndarray:
    # Accept (H, W, C) -> (N, C)
    if arr.ndim == 2:  # (H, W) -> (N, 1)
        h, w = arr.shape
        return arr.reshape(h * w, 1)
    if arr.ndim == 3:
        h, w, c = arr.shape
        return arr.reshape(h * w, c)
    raise ValueError(f"Unsupported grid array shape {arr.shape}")


class GridZarrDataset(Dataset):
    """
    Lightweight Zarr-backed grid dataset that yields Sample dicts per time index.

    Expected Zarr layout inside a group (e.g., group='diffusion2d'):
      - 'coords': (N, 2) float32
      - 'time': (T,) float32
      - 'dt': scalar attribute on group or array 'dt' with shape (1,)
      - 'fields/<name>': (T, H, W, C)
      - group attrs: {'kind': 'grid', 'H': int, 'W': int}
    """

    def __init__(self, path: str, group: Optional[str] = None, device: Optional[torch.device] = None):
        import zarr  # imported lazily

        self.path = path
        self.device = device
        self.store = zarr.open(path, mode="r")

        if group is None:
            # Choose first subgroup if present, else root
            subgroups = [k for k, v in self.store.groups()]
            self.group = self.store[subgroups[0]] if subgroups else self.store
        else:
            self.group = self.store[group]

        g = self.group
        self.kind = g.attrs.get("kind", "grid")
        if self.kind != "grid":
            raise ValueError(f"Unsupported kind '{self.kind}' in GridZarrDataset")

        self.coords = np.asarray(g["coords"])  # (N, 2)
        self.time = np.asarray(g["time"]) if "time" in g else np.arange(len(next(g["fields"].values())))
        self.dt = float(g.attrs.get("dt", 0.0))
        self.H = int(g.attrs.get("H", 0))
        self.W = int(g.attrs.get("W", 0))
        if self.H == 0 or self.W == 0:
            # Infer from coords assuming regular grid
            n = self.coords.shape[0]
            side = int(np.sqrt(n))
            if side * side != n:
                raise ValueError("Unable to infer grid shape; please annotate attrs H and W")
            self.H = side
            self.W = side

        self.fields_names: List[str] = []
        if "fields" in g:
            for name, arr in g["fields"].arrays():
                self.fields_names.append(name)
        else:
            # Back-compat: look for any arrays at top-level except coords/time
            for name, arr in g.arrays():
                if name not in ("coords", "time"):
                    self.fields_names.append(name)

        # Determine length from first field
        if self.fields_names:
            first = self._get_field_arr(self.fields_names[0])
            self.T = first.shape[0]
        else:
            self.T = len(self.time)

    def _get_field_arr(self, name: str) -> np.ndarray:
        if "fields" in self.group:
            return self.group["fields"][name]
        return self.group[name]

    def __len__(self) -> int:
        return self.T

    def __getitem__(self, idx: int) -> Sample:
        if idx < 0 or idx >= self.T:
            raise IndexError(idx)

        fields: Dict[str, torch.Tensor] = {}
        for name in self.fields_names:
            arr = np.asarray(self._get_field_arr(name)[idx])  # (H, W, C) or (H, W)
            flat = _flatten_grid(arr)  # (N, C)
            fields[name] = torch.from_numpy(flat).to(dtype=torch.float32)

        coords = torch.from_numpy(self.coords).to(dtype=torch.float32)
        kind = "grid"
        connect = None
        bc: Dict[str, Any] = {"type": "periodic"}
        params: Dict[str, float] = {}
        geom: Dict[str, Any] = {"domain": "unit_square"}
        time = torch.tensor(float(self.time[idx]) if len(self.time) > idx else float(idx), dtype=torch.float32)
        dt = torch.tensor(self.dt if self.dt > 0 else 0.0, dtype=torch.float32)
        meta: Dict[str, Any] = {
            "zarr_path": self.path,
            "grid_shape": (self.H, self.W),
            "field_channels": {name: fields[name].shape[1] for name in fields},
        }

        sample: Sample = {
            "kind": kind,
            "coords": coords if self.device is None else coords.to(self.device),
            "connect": None,
            "fields": fields if self.device is None else {k: v.to(self.device) for k, v in fields.items()},
            "bc": bc,
            "params": params,
            "geom": geom,
            "time": time if self.device is None else time.to(self.device),
            "dt": dt if self.device is None else dt.to(self.device),
            "meta": meta,
        }
        return sample


class MeshZarrDataset(Dataset):
    """Iterable dataset that yields mesh samples with cached Laplacians."""

    def __init__(self, path: str, group: str = "mesh_poisson", device: Optional[torch.device] = None):
        try:
            import zarr
        except ImportError as exc:
            raise ImportError("MeshZarrDataset requires the 'zarr' package") from exc
        try:
            import scipy.sparse as sp
        except ImportError as exc:
            raise ImportError("MeshZarrDataset requires scipy.sparse") from exc

        self.path = path
        self.group_name = group
        self.device = device

        store = zarr.open(path, mode="r")
        if group not in store:
            raise ValueError(f"Group '{group}' not found in Zarr store {path}")
        self._group = store[group]
        self._keys = sorted(self._group.group_keys())
        self._lap_cache = {}  # cache scipy matrices keyed by sample name
        self._sp = sp

    def __len__(self) -> int:
        return len(self._keys)

    def __getitem__(self, idx: int) -> Sample:
        if idx < 0 or idx >= len(self._keys):
            raise IndexError(idx)

        name = self._keys[idx]
        grp = self._group[name]

        coords = torch.from_numpy(grp["coords"][:]).to(dtype=torch.float32)
        edges = torch.from_numpy(grp["edges"][:]).to(dtype=torch.int64)
        cells = torch.from_numpy(grp["cells"][:]).to(dtype=torch.int64)

        if name not in self._lap_cache:
            data = grp["laplacian"]["data"][:]
            indices = grp["laplacian"]["indices"][:]
            indptr = grp["laplacian"]["indptr"][:]
            n = coords.shape[0]
            lap = self._sp.csr_matrix((data, indices, indptr), shape=(n, n))
            self._lap_cache[name] = lap
        else:
            lap = self._lap_cache[name]

        sample: Sample = {
            "kind": "mesh",
            "coords": coords if self.device is None else coords.to(self.device),
            "connect": edges if self.device is None else edges.to(self.device),
            "fields": {},
            "bc": {"type": "dirichlet_zero"},
            "params": {},
            "geom": {"cells": cells},
            "time": torch.tensor(float(idx), dtype=torch.float32),
            "dt": torch.tensor(0.0, dtype=torch.float32),
            "meta": {"zarr_path": self.path, "laplacian": lap},
        }
        return sample


class ParticleZarrDataset(Dataset):
    """Dataset for particle advection samples with cached neighbour graphs."""

    def __init__(self, path: str, group: str = "particles_advect", device: Optional[torch.device] = None):
        try:
            import zarr
        except ImportError as exc:
            raise ImportError("ParticleZarrDataset requires the 'zarr' package") from exc

        self.path = path
        self.group_name = group
        self.device = device

        store = zarr.open(path, mode="r")
        if group not in store:
            raise ValueError(f"Group '{group}' not found in Zarr store {path}")
        self._group = store[group]
        self._keys = sorted(self._group.group_keys())

    def __len__(self) -> int:
        return len(self._keys)

    def __getitem__(self, idx: int) -> Sample:
        if idx < 0 or idx >= len(self._keys):
            raise IndexError(idx)

        name = self._keys[idx]
        grp = self._group[name]

        positions = torch.from_numpy(grp["positions"][:]).to(dtype=torch.float32)
        velocities = torch.from_numpy(grp["velocities"][:]).to(dtype=torch.float32)

        nbr = grp["neighbors"]
        indices = torch.from_numpy(nbr["indices"][:]).to(dtype=torch.int64)
        indptr = torch.from_numpy(nbr["indptr"][:]).to(dtype=torch.int64)
        edges = torch.from_numpy(nbr["edges"][:]).to(dtype=torch.int64)

        meta = {
            "zarr_path": self.path,
            "indices_csr": indices,
            "indptr_csr": indptr,
            "edges": edges,
            "radius": float(nbr.attrs.get("radius", 0.0)),
        }

        sample: Sample = {
            "kind": "particles",
            "coords": positions[0] if self.device is None else positions[0].to(self.device),
            "connect": edges if self.device is None else edges.to(self.device),
            "fields": {
                "positions": positions if self.device is None else positions.to(self.device),
                "velocities": velocities if self.device is None else velocities.to(self.device),
            },
            "bc": {"type": "periodic"},
            "params": {"radius": meta["radius"]},
            "geom": None,
            "time": torch.tensor(float(idx), dtype=torch.float32),
            "dt": torch.tensor(0.0, dtype=torch.float32),
            "meta": meta,
        }
        return sample
