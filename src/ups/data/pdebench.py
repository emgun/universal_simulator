from __future__ import annotations

"""Lightweight PDEBench dataset adapters used for benchmarking."""

import os
import re
from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import torch
from torch.utils.data import Dataset

from ups.data.manifests import RunDataLock, canonical_sha256, load_data_lock
from ups.data.normalization import NormalizationStats


@dataclass
class PDEBenchSpec:
    field_key: str
    target_key: str | None = None
    mapping_kind: str = "trajectory"
    param_keys: tuple[str, ...] = ()
    bc_keys: tuple[str, ...] = ()
    family: str = "generic"
    traits: tuple[str, ...] = ()


TASK_SPECS: dict[str, PDEBenchSpec] = {
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
        target_key="targets",
        mapping_kind="steady_operator",
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


def _ordered_unique(values: Sequence[str]) -> tuple[str, ...]:
    seen = set()
    ordered = []
    for value in values:
        value = str(value)
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return tuple(ordered)


def pdebench_family_vocab(tasks: Sequence[str] | None = None) -> tuple[str, ...]:
    task_names = tuple(str(task) for task in (tasks or TASK_SPECS.keys()))
    return _ordered_unique(tuple(get_pdebench_spec(task).family for task in task_names))


def pdebench_trait_vocab(tasks: Sequence[str] | None = None) -> tuple[str, ...]:
    task_names = tuple(str(task) for task in (tasks or TASK_SPECS.keys()))
    traits = []
    for task in task_names:
        traits.extend(get_pdebench_spec(task).traits)
    return _ordered_unique(tuple(traits))


def pdebench_task_semantics(
    task: str,
    *,
    task_vocab: Sequence[str] | None = None,
    family_vocab: Sequence[str] | None = None,
    trait_vocab: Sequence[str] | None = None,
) -> dict[str, torch.Tensor]:
    spec = get_pdebench_spec(task)
    semantics: dict[str, torch.Tensor] = {}

    if task_vocab is not None and len(task_vocab) > 1 and task in task_vocab:
        task_id = torch.zeros(len(task_vocab), dtype=torch.float32)
        task_id[list(task_vocab).index(task)] = 1.0
        semantics["task_id"] = task_id

    resolved_family_vocab: tuple[str, ...]
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

    resolved_trait_vocab: tuple[str, ...]
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
    family_vocab: Sequence[str] | None = None,
    trait_vocab: Sequence[str] | None = None,
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
    family_vocab: Sequence[str] | None = None,
    trait_vocab: Sequence[str] | None = None,
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
    root: str | None = None
    normalize: bool = False
    normalization_path: str | None = None
    target_normalization_path: str | None = None
    data_lock_path: str | None = None
    data_lock_sha256: str | None = None
    selection_sha256: str | None = None
    param_keys: tuple[str, ...] = ()
    bc_keys: tuple[str, ...] = ()
    max_samples: int | None = None


_BETA_PATTERN = re.compile(r"beta(?P<beta>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)")


def _h5_attr_strings(value) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        values = (value,)
    else:
        values = tuple(value)
    result = []
    for item in values:
        if isinstance(item, bytes):
            result.append(item.decode("utf-8"))
        else:
            result.append(str(item))
    return tuple(result)


def _beta_from_source_path(source_path: str) -> float:
    match = _BETA_PATTERN.search(source_path)
    if not match:
        raise ValueError(f"Could not parse beta from source path: {source_path}")
    return float(match.group("beta"))


def _derive_param_from_source_provenance(
    handle: h5py.File,
    *,
    key: str,
    sample_slice: slice,
) -> torch.Tensor | None:
    if key != "beta" or "source_file_index" not in handle or "source_paths" not in handle.attrs:
        return None
    source_paths = _h5_attr_strings(handle.attrs.get("source_paths"))
    beta_by_source = {
        source_index: _beta_from_source_path(source_path)
        for source_index, source_path in enumerate(source_paths)
    }
    source_file_index = torch.from_numpy(handle["source_file_index"][sample_slice]).long()
    values = []
    for source_index in source_file_index.tolist():
        if int(source_index) not in beta_by_source:
            raise ValueError(f"source_file_index {source_index} has no source_paths beta metadata")
        values.append(beta_by_source[int(source_index)])
    return torch.tensor(values, dtype=torch.float32).view(-1, 1)


class PDEBenchDataset(Dataset):
    """Worker-safe lazy loader for local PDEBench HDF5 shards.

    Files are indexed at construction, but physical field arrays are read only
    from ``__getitem__``. Remote objects must be staged locally before this
    class is constructed.
    """

    def __init__(
        self,
        cfg: PDEBenchConfig,
        tensor_data: dict[str, torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.spec = get_pdebench_spec(cfg.task)
        self.param_keys = tuple(cfg.param_keys or self.spec.param_keys)
        self.bc_keys = tuple(cfg.bc_keys or self.spec.bc_keys)
        self._sample_param_keys = self.param_keys
        self._sample_bc_keys = self.bc_keys
        self._handles: dict[Path, h5py.File] = {}
        self._handle_pid: int | None = None
        self._shard_paths: list[Path] = []
        self._shard_lengths: list[int] = []
        self._shard_ends: list[int] = []
        self._length = 0
        self._tensor_fields: torch.Tensor | None = None
        self._tensor_targets: torch.Tensor | None = None
        self._tensor_params: dict[str, torch.Tensor] | None = None
        self._tensor_bc: dict[str, torch.Tensor] | None = None
        self._normalization: NormalizationStats | None = None
        self._target_normalization: NormalizationStats | None = None
        self._data_lock: RunDataLock | None = None
        data_lock_path = cfg.data_lock_path or os.environ.get("DATA_LOCK")
        expected_data_lock_sha256 = cfg.data_lock_sha256
        expected_selection_sha256 = cfg.selection_sha256
        if data_lock_path:
            self._data_lock = load_data_lock(data_lock_path)
            role = "valid" if cfg.split in {"val", "valid", "validation"} else cfg.split
            if role not in self._data_lock.requested_roles:
                raise PermissionError(
                    f"Run data lock does not authorize the requested {cfg.split!r} split"
                )
            if (
                expected_data_lock_sha256
                and expected_data_lock_sha256 != self._data_lock.lock_sha256
            ):
                raise ValueError("Configured data_lock_sha256 does not match data_lock_path")
            expected_data_lock_sha256 = self._data_lock.lock_sha256
            resolved_selection_sha256 = canonical_sha256(self._data_lock.selection)
            if expected_selection_sha256 and expected_selection_sha256 != resolved_selection_sha256:
                raise ValueError("Configured selection_sha256 does not match data_lock_path")
            expected_selection_sha256 = resolved_selection_sha256
        max_samples = int(cfg.max_samples) if cfg.max_samples is not None else None
        if max_samples is not None and max_samples <= 0:
            raise ValueError("PDEBenchConfig.max_samples must be positive when set")
        if cfg.normalize:
            if not cfg.normalization_path:
                raise ValueError(
                    "normalize=True requires checksum-bound training statistics via "
                    "PDEBenchConfig.normalization_path"
                )
            self._normalization = NormalizationStats.load(
                Path(cfg.normalization_path),
                expected_data_lock_sha256=expected_data_lock_sha256,
                expected_selection_sha256=expected_selection_sha256,
            )
            if self.spec.target_key is not None:
                if not cfg.target_normalization_path:
                    raise ValueError(
                        f"normalize=True for {cfg.task} requires separately fitted target "
                        "statistics via PDEBenchConfig.target_normalization_path"
                    )
                self._target_normalization = NormalizationStats.load(
                    Path(cfg.target_normalization_path),
                    expected_data_lock_sha256=expected_data_lock_sha256,
                    expected_selection_sha256=expected_selection_sha256,
                )
        if tensor_data is not None:
            if not self._sample_param_keys and tensor_data.get("params") is not None:
                self._sample_param_keys = tuple(tensor_data["params"])
            if not self._sample_bc_keys and tensor_data.get("bc") is not None:
                self._sample_bc_keys = tuple(tensor_data["bc"])
            sample_slice = slice(0, max_samples) if max_samples is not None else slice(None)
            self._tensor_fields = tensor_data["fields"][sample_slice].float()
            self._tensor_targets = tensor_data.get("targets", tensor_data["fields"])[
                sample_slice
            ].float()
            self._tensor_params = (
                {key: value[sample_slice] for key, value in tensor_data["params"].items()}
                if tensor_data.get("params") is not None
                else None
            )
            self._tensor_bc = (
                {key: value[sample_slice] for key, value in tensor_data["bc"].items()}
                if tensor_data.get("bc") is not None
                else None
            )
            self._length = int(self._tensor_fields.shape[0])
            if self._tensor_fields.shape != self._tensor_targets.shape:
                raise ValueError("Fields and targets must share shape")
            if self.spec.target_key is not None and "targets" not in tensor_data:
                raise ValueError(
                    f"{cfg.task} requires explicit coefficient inputs and solution targets"
                )
        else:
            resolved_root = cfg.root or os.environ.get("PDEBENCH_ROOT")
            if resolved_root is None:
                raise ValueError("Either tensor_data, cfg.root, or PDEBENCH_ROOT must be provided")
            if "://" in str(resolved_root):
                raise ValueError("PDEBenchDataset requires a verified local staged root")
            base = Path(resolved_root)
            file_path = base / f"{cfg.task}_{cfg.split}.h5"
            shard_paths: list[Path]
            if file_path.exists():
                shard_paths = [file_path]
            else:
                shard_paths = sorted(base.glob(f"{cfg.task}_{cfg.split}_*.h5"))
                if not shard_paths:
                    raise FileNotFoundError(file_path)
            remaining = max_samples
            for path in shard_paths:
                if remaining is not None and remaining <= 0:
                    break
                with h5py.File(path, "r") as handle:
                    if self.spec.field_key not in handle:
                        raise ValueError(
                            f"{path} is missing explicit field dataset {self.spec.field_key!r}"
                        )
                    if self.spec.target_key is not None and self.spec.target_key not in handle:
                        raise ValueError(
                            f"{path} is missing required target dataset {self.spec.target_key!r}; "
                            f"{cfg.task} cannot be represented as a solution trajectory"
                        )
                    available = int(handle[self.spec.field_key].shape[0])
                    if self.spec.target_key is not None:
                        targets = handle[self.spec.target_key]
                        if int(targets.shape[0]) != available:
                            raise ValueError(f"{path} input and target sample counts differ")
                        if tuple(targets.shape[1:]) != tuple(handle[self.spec.field_key].shape[1:]):
                            raise ValueError(f"{path} input and target canonical shapes differ")
                    take = available if remaining is None else min(remaining, available)
                    if take <= 0:
                        continue
                self._shard_paths.append(path)
                self._shard_lengths.append(take)
                self._length += take
                self._shard_ends.append(self._length)
                if remaining is not None:
                    remaining -= take
            if not self._shard_paths:
                raise RuntimeError(
                    f"No samples loaded for PDEBench task '{cfg.task}' split '{cfg.split}'"
                )
            if self._data_lock is not None:
                locked_names = {
                    Path(item.path).name for item in self._data_lock.objects if item.role == role
                }
                unlocked = [
                    path.name for path in self._shard_paths if path.name not in locked_names
                ]
                if unlocked:
                    raise PermissionError(
                        "Dataset root contains selected files outside the run data lock: "
                        + ", ".join(unlocked)
                    )

    def __len__(self) -> int:
        return self._length

    def _normalise_index(self, idx: int) -> int:
        if idx < 0:
            idx += self._length
        if idx < 0 or idx >= self._length:
            raise IndexError(idx)
        return idx

    def _locate(self, idx: int) -> tuple[Path, int]:
        idx = self._normalise_index(idx)
        shard_index = bisect_right(self._shard_ends, idx)
        start = 0 if shard_index == 0 else self._shard_ends[shard_index - 1]
        return self._shard_paths[shard_index], idx - start

    def _get_handle(self, path: Path) -> h5py.File:
        pid = os.getpid()
        if self._handle_pid != pid:
            self.close()
            self._handle_pid = pid
        handle = self._handles.get(path)
        if handle is None:
            handle = h5py.File(path, "r")
            self._handles[path] = handle
        return handle

    def _read_param(self, idx: int, key: str) -> torch.Tensor | None:
        if self._tensor_fields is not None:
            if self._tensor_params is None or key not in self._tensor_params:
                return None
            return self._tensor_params[key][self._normalise_index(idx)]
        path, local_idx = self._locate(idx)
        handle = self._get_handle(path)
        if key in handle:
            return torch.as_tensor(handle[key][local_idx]).float()
        derived = _derive_param_from_source_provenance(
            handle, key=key, sample_slice=slice(local_idx, local_idx + 1)
        )
        return None if derived is None else derived[0]

    def _read_bc(self, idx: int, key: str) -> torch.Tensor | None:
        if self._tensor_fields is not None:
            if self._tensor_bc is None or key not in self._tensor_bc:
                return None
            return self._tensor_bc[key][self._normalise_index(idx)]
        path, local_idx = self._locate(idx)
        handle = self._get_handle(path)
        return torch.as_tensor(handle[key][local_idx]).float() if key in handle else None

    def __getitem__(self, idx: int) -> dict[str, Any]:
        idx = self._normalise_index(idx)
        if self._tensor_fields is not None:
            fields = self._tensor_fields[idx]
            assert self._tensor_targets is not None
            targets = self._tensor_targets[idx]
        else:
            path, local_idx = self._locate(idx)
            handle = self._get_handle(path)
            fields = torch.as_tensor(handle[self.spec.field_key][local_idx]).float()
            if self.spec.target_key and self.spec.target_key in handle:
                targets = torch.as_tensor(handle[self.spec.target_key][local_idx]).float()
            else:
                targets = fields.clone()
        if self._normalization is not None:
            fields = self._normalization.apply(fields)
            target_normalization = self._target_normalization or self._normalization
            targets = target_normalization.apply(targets)
        sample = {
            "fields": fields,
            "targets": targets,
        }
        params = {
            key: value
            for key in self._sample_param_keys
            if (value := self._read_param(idx, key)) is not None
        }
        bc = {
            key: value
            for key in self._sample_bc_keys
            if (value := self._read_bc(idx, key)) is not None
        }
        if params:
            sample["params"] = params
        if bc:
            sample["bc"] = bc
        return sample

    def _stack_component(self, component: str, item: int | slice) -> torch.Tensor:
        if isinstance(item, slice):
            indices = range(*item.indices(self._length))
            values = [self[index][component] for index in indices]
            if not values:
                raise IndexError("Cannot materialize an empty lazy slice")
            return torch.stack(values)
        return self[item][component]

    @property
    def fields(self) -> _LazyTensorView:
        return _LazyTensorView(self, "fields")

    @property
    def targets(self) -> _LazyTensorView:
        return _LazyTensorView(self, "targets")

    @property
    def params(self) -> _LazyMappingView | None:
        return (
            _LazyMappingView(self, "params", self._sample_param_keys)
            if self._sample_param_keys
            else None
        )

    @property
    def bc(self) -> _LazyMappingView | None:
        return _LazyMappingView(self, "bc", self._sample_bc_keys) if self._sample_bc_keys else None

    def close(self) -> None:
        for handle in getattr(self, "_handles", {}).values():
            try:
                handle.close()
            except Exception:
                pass
        self._handles = {}

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_handles"] = {}
        state["_handle_pid"] = None
        return state

    def __del__(self) -> None:
        self.close()


class _LazyTensorView:
    """Compatibility view that materializes only explicitly requested rows."""

    def __init__(self, dataset: PDEBenchDataset, component: str) -> None:
        self._dataset = dataset
        self._component = component

    @property
    def shape(self) -> torch.Size:
        sample = self._dataset[0][self._component]
        return torch.Size((len(self._dataset), *sample.shape))

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, item: int | slice) -> torch.Tensor:
        return self._dataset._stack_component(self._component, item)


class _LazyMappingView:
    def __init__(self, dataset: PDEBenchDataset, component: str, keys: Sequence[str]) -> None:
        self._dataset = dataset
        self._component = component
        self._keys = tuple(keys)

    def __getitem__(self, key: str) -> torch.Tensor:
        if key not in self._keys:
            raise KeyError(key)
        values = []
        for index in range(len(self._dataset)):
            mapping = self._dataset[index].get(self._component, {})
            if key not in mapping:
                raise KeyError(key)
            values.append(mapping[key])
        return torch.stack(values)
