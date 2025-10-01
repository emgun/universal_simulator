from __future__ import annotations

"""Lightweight PDEBench dataset adapters used for benchmarking."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import h5py
import torch
from torch.utils.data import Dataset


@dataclass
class PDEBenchSpec:
    field_key: str
    target_key: Optional[str] = None
    param_keys: Tuple[str, ...] = ()
    bc_keys: Tuple[str, ...] = ()


TASK_SPECS: Dict[str, PDEBenchSpec] = {
    "burgers1d": PDEBenchSpec(field_key="data"),
    "advection1d": PDEBenchSpec(field_key="data"),
    "darcy2d": PDEBenchSpec(field_key="data"),
    "navier_stokes2d": PDEBenchSpec(field_key="data"),
}


@dataclass
class PDEBenchConfig:
    task: str
    split: str = "train"
    root: Optional[str] = None
    normalize: bool = True


def _normalise_fields(fields: torch.Tensor) -> torch.Tensor:
    mean = fields.mean()
    std = fields.std()
    if std < 1e-6:
        std = torch.tensor(1.0)
    return (fields - mean) / std


class PDEBenchDataset(Dataset):
    """Loader for PDEBench HDF5 dumps (with fallback tensor data for tests)."""

    def __init__(
        self,
        cfg: PDEBenchConfig,
        tensor_data: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        if tensor_data is not None:
            self.fields = tensor_data["fields"].float()
            self.targets = tensor_data.get("targets", tensor_data["fields"]).float()
            self.params = tensor_data.get("params")
            self.bc = tensor_data.get("bc")
        else:
            if cfg.root is None:
                raise ValueError("Either tensor_data or cfg.root must be provided")
            spec = TASK_SPECS.get(cfg.task)
            if spec is None:
                raise KeyError(f"Unknown PDEBench task '{cfg.task}'")
            base = Path(cfg.root)
            file_path = base / f"{cfg.task}_{cfg.split}.h5"
            if not file_path.exists():
                raise FileNotFoundError(file_path)
            with h5py.File(file_path, "r") as f:
                fields = torch.from_numpy(f[spec.field_key][...]).float()
                if cfg.normalize:
                    fields = _normalise_fields(fields)
                self.fields = fields
                if spec.target_key and spec.target_key in f:
                    self.targets = torch.from_numpy(f[spec.target_key][...]).float()
                else:
                    self.targets = self.fields
                if spec.param_keys:
                    self.params = {key: torch.from_numpy(f[key][...]).float() for key in spec.param_keys if key in f}
                else:
                    self.params = None
                if spec.bc_keys:
                    self.bc = {key: torch.from_numpy(f[key][...]).float() for key in spec.bc_keys if key in f}
                else:
                    self.bc = None
        if self.fields.shape != self.targets.shape:
            raise ValueError("Fields and targets must share shape")

    def __len__(self) -> int:
        return self.fields.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = {
            "fields": self.fields[idx],
            "targets": self.targets[idx],
        }
        if self.params is not None:
            sample["params"] = {k: v[idx] for k, v in self.params.items()}
        if self.bc is not None:
            sample["bc"] = {k: v[idx] for k, v in self.bc.items()}
        return sample
