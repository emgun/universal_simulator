from __future__ import annotations

"""Lightweight PDEBench dataset adapters used for benchmarking."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

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
    family: str = "generic"
    traits: Tuple[str, ...] = ()


TASK_SPECS: Dict[str, PDEBenchSpec] = {
    "burgers1d": PDEBenchSpec(
        field_key="data",
        family="conservation",
        traits=("scalar", "time_dependent", "transport", "nonlinear", "dissipative"),
    ),
    "advection1d": PDEBenchSpec(
        field_key="data",
        family="transport",
        traits=("scalar", "time_dependent", "transport", "linear"),
    ),
    "darcy2d": PDEBenchSpec(
        field_key="data",
        family="elliptic",
        traits=("scalar", "steady_state", "elliptic", "heterogeneous_medium"),
    ),
    "navier_stokes2d": PDEBenchSpec(
        field_key="data",
        family="fluid",
        traits=(
            "vector",
            "time_dependent",
            "transport",
            "nonlinear",
            "dissipative",
            "incompressible",
        ),
    ),
}


def get_pdebench_spec(task: str) -> PDEBenchSpec:
    spec = TASK_SPECS.get(task)
    if spec is None:
        raise KeyError(f"Unknown PDEBench task '{task}'")
    return spec


def _ordered_unique(values: Sequence[str]) -> Tuple[str, ...]:
    seen = set()
    ordered = []
    for value in values:
        value = str(value)
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return tuple(ordered)


def pdebench_family_vocab(tasks: Optional[Sequence[str]] = None) -> Tuple[str, ...]:
    task_names = tuple(str(task) for task in (tasks or TASK_SPECS.keys()))
    return _ordered_unique(tuple(get_pdebench_spec(task).family for task in task_names))


def pdebench_trait_vocab(tasks: Optional[Sequence[str]] = None) -> Tuple[str, ...]:
    task_names = tuple(str(task) for task in (tasks or TASK_SPECS.keys()))
    traits = []
    for task in task_names:
        traits.extend(get_pdebench_spec(task).traits)
    return _ordered_unique(tuple(traits))


def pdebench_task_semantics(
    task: str,
    *,
    task_vocab: Optional[Sequence[str]] = None,
    family_vocab: Optional[Sequence[str]] = None,
    trait_vocab: Optional[Sequence[str]] = None,
) -> Dict[str, torch.Tensor]:
    spec = get_pdebench_spec(task)
    semantics: Dict[str, torch.Tensor] = {}

    if task_vocab is not None and len(task_vocab) > 1 and task in task_vocab:
        task_id = torch.zeros(len(task_vocab), dtype=torch.float32)
        task_id[list(task_vocab).index(task)] = 1.0
        semantics["task_id"] = task_id

    resolved_family_vocab: Tuple[str, ...]
    if family_vocab is not None:
        resolved_family_vocab = tuple(str(name) for name in family_vocab)
    elif task_vocab is not None and len(task_vocab) > 1:
        resolved_family_vocab = pdebench_family_vocab(task_vocab)
    else:
        resolved_family_vocab = ()
    if len(resolved_family_vocab) > 1 and spec.family in resolved_family_vocab:
        family_id = torch.zeros(len(resolved_family_vocab), dtype=torch.float32)
        family_id[resolved_family_vocab.index(spec.family)] = 1.0
        semantics["task_family"] = family_id

    resolved_trait_vocab: Tuple[str, ...]
    if trait_vocab is not None:
        resolved_trait_vocab = tuple(str(name) for name in trait_vocab)
    elif task_vocab is not None and len(task_vocab) > 1:
        resolved_trait_vocab = pdebench_trait_vocab(task_vocab)
    else:
        resolved_trait_vocab = ()
    if resolved_trait_vocab:
        trait_id = torch.zeros(len(resolved_trait_vocab), dtype=torch.float32)
        active_traits = set(spec.traits)
        for idx, name in enumerate(resolved_trait_vocab):
            if name in active_traits:
                trait_id[idx] = 1.0
        semantics["equation_traits"] = trait_id

    return semantics


def pdebench_equation_signature(
    task: str,
    *,
    family_vocab: Optional[Sequence[str]] = None,
    trait_vocab: Optional[Sequence[str]] = None,
) -> torch.Tensor:
    resolved_family_vocab = tuple(str(name) for name in (family_vocab or pdebench_family_vocab()))
    resolved_trait_vocab = tuple(str(name) for name in (trait_vocab or pdebench_trait_vocab()))
    semantics = pdebench_task_semantics(
        task,
        family_vocab=resolved_family_vocab,
        trait_vocab=resolved_trait_vocab,
    )
    parts = []
    family = semantics.get("task_family")
    if family is not None:
        parts.append(family)
    traits = semantics.get("equation_traits")
    if traits is not None:
        parts.append(traits)
    if not parts:
        return torch.zeros(0, dtype=torch.float32)
    return torch.cat(parts, dim=0)


def _indicator_node_set(size: int, active_indices: Sequence[int]) -> torch.Tensor:
    if size <= 0:
        return torch.zeros((0, 0), dtype=torch.float32)
    eye = torch.eye(size, dtype=torch.float32)
    active = torch.zeros(size, 1, dtype=torch.float32)
    for index in active_indices:
        if 0 <= int(index) < size:
            active[int(index), 0] = 1.0
    return torch.cat([eye, active], dim=-1)


def pdebench_equation_nodes(
    task: str,
    *,
    family_vocab: Optional[Sequence[str]] = None,
    trait_vocab: Optional[Sequence[str]] = None,
) -> torch.Tensor:
    spec = get_pdebench_spec(task)
    resolved_family_vocab = tuple(str(name) for name in (family_vocab or pdebench_family_vocab()))
    resolved_trait_vocab = tuple(str(name) for name in (trait_vocab or pdebench_trait_vocab()))
    family_index = (
        resolved_family_vocab.index(spec.family) if spec.family in resolved_family_vocab else -1
    )
    family_nodes = _indicator_node_set(
        len(resolved_family_vocab), [family_index] if family_index >= 0 else []
    )
    trait_indices = [
        resolved_trait_vocab.index(name) for name in spec.traits if name in resolved_trait_vocab
    ]
    trait_nodes = _indicator_node_set(len(resolved_trait_vocab), trait_indices)
    if family_nodes.numel() == 0:
        return trait_nodes
    if trait_nodes.numel() == 0:
        return family_nodes
    max_dim = max(family_nodes.shape[-1], trait_nodes.shape[-1])
    if family_nodes.shape[-1] < max_dim:
        family_nodes = torch.cat(
            [
                family_nodes,
                family_nodes.new_zeros(family_nodes.shape[0], max_dim - family_nodes.shape[-1]),
            ],
            dim=-1,
        )
    if trait_nodes.shape[-1] < max_dim:
        trait_nodes = torch.cat(
            [
                trait_nodes,
                trait_nodes.new_zeros(trait_nodes.shape[0], max_dim - trait_nodes.shape[-1]),
            ],
            dim=-1,
        )
    return torch.cat([family_nodes, trait_nodes], dim=0)


@dataclass
class PDEBenchConfig:
    task: str
    split: str = "train"
    root: Optional[str] = None
    normalize: bool = True
    param_keys: Tuple[str, ...] = ()
    bc_keys: Tuple[str, ...] = ()
    max_samples: Optional[int] = None


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
        max_samples = int(cfg.max_samples) if cfg.max_samples is not None else None
        if max_samples is not None and max_samples <= 0:
            raise ValueError("PDEBenchConfig.max_samples must be positive when set")
        if tensor_data is not None:
            self.spec = get_pdebench_spec(cfg.task)
            self.param_keys = tuple(cfg.param_keys or self.spec.param_keys)
            self.bc_keys = tuple(cfg.bc_keys or self.spec.bc_keys)
            sample_slice = slice(0, max_samples) if max_samples is not None else slice(None)
            self.fields = tensor_data["fields"][sample_slice].float()
            self.targets = tensor_data.get("targets", tensor_data["fields"])[sample_slice].float()
            self.params = (
                {key: value[sample_slice] for key, value in tensor_data["params"].items()}
                if tensor_data.get("params") is not None
                else None
            )
            self.bc = (
                {key: value[sample_slice] for key, value in tensor_data["bc"].items()}
                if tensor_data.get("bc") is not None
                else None
            )
        else:
            if cfg.root is None:
                # Allow environment override for convenience in remote runs
                env_root = os.environ.get("PDEBENCH_ROOT")
                if env_root:
                    cfg.root = env_root
                else:
                    raise ValueError("Either tensor_data or cfg.root must be provided")
            spec = get_pdebench_spec(cfg.task)
            self.spec = spec
            param_keys = tuple(cfg.param_keys or spec.param_keys)
            bc_keys = tuple(cfg.bc_keys or spec.bc_keys)
            self.param_keys = param_keys
            self.bc_keys = bc_keys
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
            remaining = max_samples

            for path in shard_paths:
                if remaining is not None and remaining <= 0:
                    break
                with h5py.File(path, "r") as f:
                    available = int(f[spec.field_key].shape[0])
                    take = available if remaining is None else min(remaining, available)
                    if take <= 0:
                        continue
                    sample_slice = slice(0, take)
                    f_fields = torch.from_numpy(f[spec.field_key][sample_slice]).float()
                    if cfg.normalize:
                        f_fields = _normalise_fields(f_fields)
                    fields_list.append(f_fields)
                    if spec.target_key and spec.target_key in f:
                        targets_list.append(
                            torch.from_numpy(f[spec.target_key][sample_slice]).float()
                        )
                    else:
                        targets_list.append(f_fields)
                    # Parameter/BC aggregation (if present): concatenate along first axis
                    if param_keys:
                        p = {
                            key: torch.from_numpy(f[key][sample_slice]).float()
                            for key in param_keys
                            if key in f
                        }
                        if p:
                            if params_accum is None:
                                params_accum = {k: v.clone() for k, v in p.items()}
                            else:
                                for k, v in p.items():
                                    if k in params_accum:
                                        params_accum[k] = torch.cat([params_accum[k], v], dim=0)
                    if bc_keys:
                        b = {
                            key: torch.from_numpy(f[key][sample_slice]).float()
                            for key in bc_keys
                            if key in f
                        }
                        if b:
                            if bc_accum is None:
                                bc_accum = {k: v.clone() for k, v in b.items()}
                            else:
                                for k, v in b.items():
                                    if k in bc_accum:
                                        bc_accum[k] = torch.cat([bc_accum[k], v], dim=0)
                    if remaining is not None:
                        remaining -= take

            if not fields_list:
                raise RuntimeError(
                    f"No samples loaded for PDEBench task '{cfg.task}' split '{cfg.split}'"
                )
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
