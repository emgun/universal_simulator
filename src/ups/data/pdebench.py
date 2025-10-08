from __future__ import annotations

"""Lightweight PDEBench dataset adapters used for benchmarking."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import h5py
import torch
from torch.utils.data import Dataset
import os


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
                # Allow environment override for convenience in remote runs
                env_root = os.environ.get("PDEBENCH_ROOT")
                if env_root:
                    cfg.root = env_root
                else:
                    raise ValueError("Either tensor_data or cfg.root must be provided")
            spec = TASK_SPECS.get(cfg.task)
            if spec is None:
                raise KeyError(f"Unknown PDEBench task '{cfg.task}'")
            base = Path(cfg.root)
            file_path = base / f"{cfg.task}_{cfg.split}.h5"
            shard_paths = []
            if file_path.exists():
                shard_paths = [file_path]
            else:
                shard_paths = sorted(base.glob(f"{cfg.task}_{cfg.split}_*.h5"))
                if not shard_paths:
                    raise FileNotFoundError(file_path)

            fields_list = []
            targets_list = []
            params_accum = None
            bc_accum = None

            for path in shard_paths:
                with h5py.File(path, "r") as f:
                    f_fields = torch.from_numpy(f[spec.field_key][...]).float()
                    if cfg.normalize:
                        f_fields = _normalise_fields(f_fields)
                    fields_list.append(f_fields)
                    if spec.target_key and spec.target_key in f:
                        targets_list.append(torch.from_numpy(f[spec.target_key][...]).float())
                    else:
                        targets_list.append(f_fields)
                    # Parameter/BC aggregation (if present): concatenate along first axis
                    if spec.param_keys:
                        p = {key: torch.from_numpy(f[key][...]).float() for key in spec.param_keys if key in f}
                        if p:
                            if params_accum is None:
                                params_accum = {k: v.clone() for k, v in p.items()}
                            else:
                                for k, v in p.items():
                                    if k in params_accum:
                                        params_accum[k] = torch.cat([params_accum[k], v], dim=0)
                    if spec.bc_keys:
                        b = {key: torch.from_numpy(f[key][...]).float() for key in spec.bc_keys if key in f}
                        if b:
                            if bc_accum is None:
                                bc_accum = {k: v.clone() for k, v in b.items()}
                            else:
                                for k, v in b.items():
                                    if k in bc_accum:
                                        bc_accum[k] = torch.cat([bc_accum[k], v], dim=0)

            self.fields = torch.cat(fields_list, dim=0)
            self.targets = torch.cat(targets_list, dim=0)
            self.params = params_accum
            self.bc = bc_accum
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
