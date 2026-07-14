from __future__ import annotations

"""Lazy adapter from The Well's native loader to the UPS trajectory contract."""

import os
from bisect import bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import torch
from torch.utils.data import Dataset

from ups.data.manifests import load_data_lock
from ups.data.trajectory import (
    TrajectoryIdentity,
    TrajectoryMetadata,
    TrajectorySample,
    validate_trajectory_sample,
)


class NativeWellDataset(Protocol):
    """Structural interface used to keep ``the_well`` an optional dependency."""

    metadata: Any

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class WellConfig:
    dataset_name: str
    root: str
    split: str = "train"
    n_steps_input: int = 1
    n_steps_output: int = 1
    time_stride: int = 1
    max_samples: int | None = None
    cache_small: bool = False
    data_lock_path: str | None = None

    def __post_init__(self) -> None:
        if not self.dataset_name.strip():
            raise ValueError("dataset_name cannot be empty")
        if self.split not in {"train", "valid", "test"}:
            raise ValueError("split must be one of: train, valid, test")
        if self.n_steps_input <= 0 or self.n_steps_output <= 0:
            raise ValueError("input and output step counts must be positive")
        if self.time_stride <= 0:
            raise ValueError("time_stride must be positive")
        if self.max_samples is not None and self.max_samples <= 0:
            raise ValueError("max_samples must be positive when set")
        if "://" in self.root:
            raise ValueError("The Well data must be staged locally before constructing the adapter")


def _as_tensor(value: Any, *, name: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    try:
        return torch.as_tensor(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"The Well sample value {name!r} is not tensor-like") from exc


def _named_tensors(
    value: Any,
    *,
    default_name: str,
    names: Sequence[str] = (),
) -> dict[str, torch.Tensor]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return {str(name): _as_tensor(item, name=str(name)) for name, item in value.items()}
    tensor = _as_tensor(value, name=default_name)
    if names and tensor.ndim > 0 and tensor.shape[-1] == len(names):
        return {str(name): tensor[..., index] for index, name in enumerate(names)}
    return {default_name: tensor}


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        value = (value,)
    if not isinstance(value, Sequence):
        return ()
    return tuple(item.decode() if isinstance(item, bytes) else str(item) for item in value)


class WellTrajectoryDataset(Dataset[TrajectorySample]):
    """Map-style, non-materializing adapter over ``the_well.data.WellDataset``.

    The native loader owns HDF5/fsspec handle lifecycle and chunk reads. This
    adapter only translates the single sample requested by ``__getitem__``.
    Native normalization is intentionally disabled so training-only,
    checksum-bound normalization can be applied by the shared pipeline.
    """

    def __init__(
        self,
        config: WellConfig,
        *,
        native_dataset: NativeWellDataset | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        lock_path = config.data_lock_path or os.environ.get("DATA_LOCK")
        self._data_lock = load_data_lock(lock_path) if lock_path else None
        if self._data_lock is not None and config.split not in self._data_lock.requested_roles:
            raise PermissionError(
                f"Run data lock does not authorize The Well {config.split!r} split"
            )
        self._native = native_dataset if native_dataset is not None else self._build_native(config)
        if self._data_lock is not None:
            locked_paths = {
                Path(item.path).as_posix()
                for item in self._data_lock.objects
                if item.role == config.split
            }
            native_paths = tuple(getattr(self._native, "files_paths", ()))
            unlocked = [
                str(path)
                for path in native_paths
                if not any(Path(str(path)).as_posix().endswith(locked) for locked in locked_paths)
            ]
            if unlocked:
                raise PermissionError(
                    "The Well native loader selected files outside the run data lock: "
                    + ", ".join(unlocked)
                )
        native_length = len(self._native)
        self._length = min(native_length, config.max_samples or native_length)
        if self._length <= 0:
            raise ValueError("The Well dataset contains no selected samples")

    @staticmethod
    def _build_native(config: WellConfig) -> NativeWellDataset:
        try:
            from the_well.data import WellDataset
        except ImportError as exc:
            raise ImportError(
                "The Well adapter requires the optional 'the_well' package; "
                "install it only on workers that read The Well data"
            ) from exc

        return cast(
            NativeWellDataset,
            WellDataset(
                well_base_path=str(Path(config.root)),
                well_dataset_name=config.dataset_name,
                well_split_name=config.split,
                n_steps_input=config.n_steps_input,
                n_steps_output=config.n_steps_output,
                min_dt_stride=config.time_stride,
                max_dt_stride=config.time_stride,
                flatten_tensors=True,
                cache_small=config.cache_small,
                return_grid=True,
                normalize_time_grid=False,
                boundary_return_type="padding",
                use_normalization=False,
            ),
        )

    def __len__(self) -> int:
        return self._length

    def _identity(self, index: int) -> TrajectoryIdentity:
        identity: TrajectoryIdentity = {
            "source": "the-well",
            "dataset_name": self.config.dataset_name,
            "split": self.config.split,
            "sample_index": index,
            "source_file": None,
            "trajectory_index": None,
            "window_start": None,
        }
        offsets = getattr(self._native, "file_index_offsets", None)
        windows = getattr(self._native, "n_windows_per_trajectory", None)
        paths = getattr(self._native, "files_paths", None)
        if isinstance(offsets, Sequence) and isinstance(windows, Sequence) and offsets:
            file_index = bisect_right(offsets, index) - 1
            if 0 <= file_index < len(windows) and int(windows[file_index]) > 0:
                local_index = index - max(int(offsets[file_index]), 0)
                identity["trajectory_index"] = local_index // int(windows[file_index])
                identity["window_start"] = local_index % int(windows[file_index])
                if isinstance(paths, Sequence) and file_index < len(paths):
                    identity["source_file"] = str(paths[file_index])
        return identity

    def _field_names(self, channel_count: int) -> tuple[str, ...]:
        names_by_order = getattr(self._native, "field_names", None)
        if isinstance(names_by_order, Mapping):
            field_names = tuple(str(name) for names in names_by_order.values() for name in names)
        else:
            field_names = _string_tuple(getattr(self._native, "core_field_names", None))
        return field_names if len(field_names) == channel_count else ()

    def _metadata(self, fields: torch.Tensor, index: int) -> TrajectoryMetadata:
        native_metadata = getattr(self._native, "metadata", None)
        field_names = self._field_names(fields.shape[-1])
        scalar_names = _string_tuple(getattr(self._native, "constant_scalar_names", None))
        return {
            "identity": self._identity(index),
            "layout": "T...C",
            "channel_axis": -1,
            "n_spatial_dims": fields.ndim - 2,
            "field_names": field_names,
            "attributes": {
                "native_adapter": type(self._native).__name__,
                "native_metadata_type": type(native_metadata).__name__,
                "constant_scalar_names": scalar_names,
            },
        }

    def __getitem__(self, index: int) -> TrajectorySample:
        if index < 0:
            index += self._length
        if index < 0 or index >= self._length:
            raise IndexError(index)
        native = self._native[index]
        missing = {"input_fields", "output_fields"}.difference(native)
        if missing:
            raise ValueError(f"The Well sample is missing keys: {', '.join(sorted(missing))}")

        fields = _as_tensor(native["input_fields"], name="input_fields")
        targets = _as_tensor(native["output_fields"], name="output_fields")
        input_time_grid = native.get("input_time_grid")
        output_time_grid = native.get("output_time_grid")
        sample: TrajectorySample = {
            "fields": fields,
            "targets": targets,
            "params": _named_tensors(
                native.get("constant_scalars"),
                default_name="constants",
                names=_string_tuple(getattr(self._native, "constant_scalar_names", None)),
            ),
            "boundary_conditions": _named_tensors(
                native.get("boundary_conditions"), default_name="padding"
            ),
            "auxiliary": {
                name: _as_tensor(native[name], name=name)
                for name in ("constant_fields", "input_scalars", "output_scalars")
                if native.get(name) is not None
            },
            "space_grid": (
                None
                if native.get("space_grid") is None
                else _as_tensor(native["space_grid"], name="space_grid")
            ),
            "input_time_grid": (
                None
                if input_time_grid is None
                else _as_tensor(input_time_grid, name="input_time_grid")
            ),
            "target_time_grid": (
                None
                if output_time_grid is None
                else _as_tensor(output_time_grid, name="output_time_grid")
            ),
            "metadata": self._metadata(fields, index),
        }
        validate_trajectory_sample(sample)
        return sample
