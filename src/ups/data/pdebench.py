from __future__ import annotations

"""Lightweight PDEBench dataset adapters used for benchmarking."""

import os
import signal
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import ContextManager

import h5py
import torch
from torch.utils.data import Dataset


@contextmanager
def hdf5_timeout(seconds: int) -> ContextManager:
    """Context manager for HDF5 operation timeout.

    Uses SIGALRM to interrupt blocking HDF5 operations.
    Only works on Unix systems.

    Args:
        seconds: Timeout in seconds (0=disabled)

    Raises:
        TimeoutError: If operation exceeds timeout
    """
    if seconds <= 0 or os.name == 'nt':  # Disabled or Windows
        yield
        return

    def timeout_handler(signum, frame):
        raise TimeoutError(
            f"HDF5 operation timed out after {seconds}s. "
            f"Possible causes: network storage lag, file corruption, or large file size."
        )

    # Set alarm
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        # Cancel alarm
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


@dataclass
class PDEBenchSpec:
    field_key: str
    target_key: str | None = None
    param_keys: tuple[str, ...] = ()
    bc_keys: tuple[str, ...] = ()
    kind: str = "grid"  # one of {"grid", "mesh", "particles"}


TASK_SPECS: dict[str, PDEBenchSpec] = {
    # Grid-based PDEBench tasks (HDF5)
    "burgers1d": PDEBenchSpec(field_key="data"),
    "advection1d": PDEBenchSpec(field_key="data"),
    "diffusion_sorption1d": PDEBenchSpec(field_key="data"),
    "reaction_diffusion1d": PDEBenchSpec(field_key="data"),
    "cfd1d_shocktube": PDEBenchSpec(field_key="data"),
    "darcy2d": PDEBenchSpec(field_key="data"),
    "navier_stokes2d": PDEBenchSpec(field_key="data"),
    "cfd2d_rand": PDEBenchSpec(field_key="data"),
    "cfd2d_turb": PDEBenchSpec(field_key="data"),
    "allen_cahn2d": PDEBenchSpec(field_key="data"),
    "cahn_hilliard2d": PDEBenchSpec(field_key="data"),
    "reaction_diffusion2d": PDEBenchSpec(field_key="data"),
    "shallow_water2d": PDEBenchSpec(field_key="data"),
    "compressible_ns1d": PDEBenchSpec(field_key="data"),
    "compressible_ns3d": PDEBenchSpec(field_key="data"),
    "cfd3d": PDEBenchSpec(field_key="data"),
    # Mesh / particle variants (Zarr)
    "darcy2d_mesh": PDEBenchSpec(field_key="data", kind="mesh"),
    "particles_advect": PDEBenchSpec(field_key="data", kind="particles"),
}


def get_pdebench_spec(task: str) -> PDEBenchSpec:
    spec = TASK_SPECS.get(task)
    if spec is None:
        raise KeyError(f"Unknown PDEBench task '{task}'")
    return spec


def resolve_pdebench_root(root: str | None) -> Path:
    env_root = os.environ.get("PDEBENCH_ROOT")
    if env_root:
        return Path(env_root)
    if root is None:
        raise ValueError("Either specify data.root or set PDEBENCH_ROOT")
    return Path(root)


@dataclass
class PDEBenchConfig:
    task: str
    split: str = "train"
    root: str | None = None
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
        tensor_data: dict[str, torch.Tensor] | None = None,
        hdf5_timeout_sec: int = 0,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        if tensor_data is not None:
            self.fields = tensor_data["fields"].float()
            self.targets = tensor_data.get("targets", tensor_data["fields"]).float()
            self.params = tensor_data.get("params")
            self.bc = tensor_data.get("bc")
        else:
            # Allow environment override for convenience in remote runs.
            # If PDEBENCH_ROOT is set, it takes precedence over cfg.root to
            # avoid brittle symlink requirements on remote instances.
            spec = get_pdebench_spec(cfg.task)
            if spec.kind != "grid":
                raise ValueError(f"PDEBenchDataset only supports grid tasks; got '{cfg.task}'")
            base = resolve_pdebench_root(cfg.root)
            file_path = base / f"{cfg.task}_{cfg.split}.h5"
            shard_paths = []
            if file_path.exists():
                shard_paths = [file_path]
            else:
                shard_paths = sorted(base.glob(f"{cfg.task}_{cfg.split}_*.h5"))
                if not shard_paths:
                    raise FileNotFoundError(
                        f"Missing data for task '{cfg.task}' split '{cfg.split}'. "
                        f"Checked: {file_path} and {base}/{cfg.task}_{cfg.split}_*.h5. "
                        "Run scripts/setup_vast_data.sh or scripts/remote_preprocess_pdebench.sh to download/convert."
                    )

            fields_list = []
            targets_list = []
            params_accum = None
            bc_accum = None

            for path in shard_paths:
                try:
                    with hdf5_timeout(hdf5_timeout_sec):
                        # Use SWMR (Single-Writer-Multiple-Reader) mode for parallel safety
                        with h5py.File(path, "r", libver='latest', swmr=True) as f:
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
                except TimeoutError as e:
                    raise RuntimeError(
                        f"HDF5 read timeout for {path}. "
                        f"Try: 1) Copy data to local storage, or 2) Increase --hdf5-timeout"
                    ) from e
                except OSError as e:
                    # HDF5 file errors (permission, corruption, network timeout)
                    raise RuntimeError(
                        f"Failed to read HDF5 file {path}: {e}. "
                        f"If using network storage, try copying data to local disk first."
                    ) from e
                except Exception as e:
                    # Unexpected errors
                    raise RuntimeError(f"Unexpected error reading {path}: {e}") from e

            self.fields = torch.cat(fields_list, dim=0)
            self.targets = torch.cat(targets_list, dim=0)
            self.params = params_accum
            self.bc = bc_accum
        if self.fields.shape != self.targets.shape:
            raise ValueError("Fields and targets must share shape")

    def __len__(self) -> int:
        return self.fields.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = {
            "fields": self.fields[idx],
            "targets": self.targets[idx],
        }
        if self.params is not None:
            sample["params"] = {k: v[idx] for k, v in self.params.items()}
        if self.bc is not None:
            sample["bc"] = {k: v[idx] for k, v in self.bc.items()}
        return sample
