from __future__ import annotations

"""Fail-closed runtime bridge from the frozen strat-v1 lock to baseline loaders."""

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .manifests import RunDataLock, canonical_sha256, load_data_lock
from .pdebench import PDEBenchConfig
from .staging import file_checksum

FROZEN_STRAT_V1_TRAINING_LOCK_SHA256 = (
    "5666fa8bb646b270d23077d1936c8ee80fa1cfeaf8f5870c55103fc58dd80afd"
)


class BaselineRuntimeError(ValueError):
    """Raised before a baseline can observe unauthorized or incompatible data."""


@dataclass(frozen=True)
class TaskSemantics:
    task: str
    field_kind: str
    mapping_kind: str
    time_axis: int
    regime_dataset: str
    param_keys: tuple[str, ...]
    expected_regimes: tuple[float, ...]


STRAT_V1_TASKS: dict[str, TaskSemantics] = {
    "advection1d": TaskSemantics(
        task="advection1d",
        field_kind="temporal",
        mapping_kind="trajectory",
        time_axis=1,
        regime_dataset="beta",
        param_keys=("beta",),
        expected_regimes=(0.1, 0.2, 0.4, 0.7, 1.0, 2.0, 4.0, 7.0),
    ),
    "burgers1d": TaskSemantics(
        task="burgers1d",
        field_kind="temporal",
        mapping_kind="trajectory",
        time_axis=1,
        regime_dataset="nu",
        param_keys=("nu",),
        expected_regimes=(0.001, 0.002, 0.004, 0.01, 0.02, 0.04, 0.1, 0.2, 0.4, 1.0, 2.0, 4.0),
    ),
    "darcy2d": TaskSemantics(
        task="darcy2d",
        field_kind="steady",
        mapping_kind="steady_operator",
        time_axis=-1,
        regime_dataset="beta",
        param_keys=("beta",),
        expected_regimes=(0.01, 0.1, 1.0, 10.0, 100.0),
    ),
}


@dataclass(frozen=True)
class RegimeCount:
    value: float
    count: int


@dataclass(frozen=True)
class BaselineSplitRuntime:
    role: str
    loader_split: str
    path: Path
    sample_count: int
    regimes: tuple[RegimeCount, ...]
    balanced_indices: tuple[int, ...]


@dataclass(frozen=True)
class BaselineTaskRuntime:
    semantics: TaskSemantics
    root: Path
    train: BaselineSplitRuntime
    valid: BaselineSplitRuntime


@dataclass(frozen=True)
class StratV1BaselineRuntime:
    """Exact train/validation runtime inputs safe to hand to baseline code."""

    lock_path: Path
    run_root: Path
    lock: RunDataLock
    selection_sha256: str
    tasks: dict[str, BaselineTaskRuntime]

    def dataset_config(
        self,
        task: str,
        split: str,
        *,
        max_samples: int | None = None,
        condition_on_regime: bool = False,
    ) -> PDEBenchConfig:
        runtime = self.tasks.get(task)
        if runtime is None:
            raise BaselineRuntimeError(f"Task {task!r} is not authorized by this baseline runtime")
        role = "valid" if split in {"val", "valid", "validation"} else split
        if role not in {"train", "valid"}:
            raise PermissionError("Baseline runtime permits train and validation only")
        return PDEBenchConfig(
            task=task,
            split="val" if role == "valid" else "train",
            root=str(runtime.root),
            data_lock_path=str(self.lock_path),
            data_lock_sha256=self.lock.lock_sha256,
            selection_sha256=self.selection_sha256,
            param_keys=runtime.semantics.param_keys if condition_on_regime else (),
            max_samples=max_samples,
        )

    def apply_to_runner_config(
        self, cfg: dict[str, Any], *, condition_on_regime: bool = False
    ) -> dict[str, Any]:
        """Return a config overlay consumed by the shared external-baseline loader."""

        result = copy.deepcopy(cfg)
        data = result.setdefault("data", {})
        data.update(
            {
                "data_lock_path": str(self.lock_path),
                "data_lock_sha256": self.lock.lock_sha256,
                "selection_sha256": self.selection_sha256,
                "runtime_identity": {
                    "lock_sha256": self.lock.lock_sha256,
                    "source_manifest_sha256": self.lock.source_manifest_sha256,
                    "protocol_manifest_sha256": self.lock.protocol_manifest_sha256,
                    "selection_sha256": self.selection_sha256,
                },
                "task_roots": {
                    task: str(runtime.root) for task, runtime in sorted(self.tasks.items())
                },
                "task_param_keys": {
                    task: list(runtime.semantics.param_keys) if condition_on_regime else []
                    for task, runtime in sorted(self.tasks.items())
                },
                "physical_parameter_conditioning": bool(condition_on_regime),
                "task_semantics": {
                    task: {
                        "field_kind": runtime.semantics.field_kind,
                        "mapping_kind": runtime.semantics.mapping_kind,
                        "time_axis": runtime.semantics.time_axis,
                        "regime_dataset": runtime.semantics.regime_dataset,
                    }
                    for task, runtime in sorted(self.tasks.items())
                },
                "regime_counts": {
                    task: {
                        "train": [vars(item) for item in runtime.train.regimes],
                        "valid": [vars(item) for item in runtime.valid.regimes],
                    }
                    for task, runtime in sorted(self.tasks.items())
                },
                "balanced_sample_indices": {
                    task: {
                        "train": list(runtime.train.balanced_indices),
                        "valid": list(runtime.valid.balanced_indices),
                    }
                    for task, runtime in sorted(self.tasks.items())
                },
            }
        )
        return result


def _attribute(handle: h5py.File, name: str, path: Path) -> Any:
    if name not in handle.attrs:
        raise BaselineRuntimeError(f"{path} is missing required attribute {name!r}")
    value = handle.attrs[name]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _inspect_split(path: Path, *, role: str, semantics: TaskSemantics) -> BaselineSplitRuntime:
    loader_split = "val" if role == "valid" else "train"
    try:
        with h5py.File(path, "r") as handle:
            if _attribute(handle, "task", path) != semantics.task:
                raise BaselineRuntimeError(f"{path} task attribute does not match {semantics.task}")
            if _attribute(handle, "selection_protocol", path) != "strat-v1":
                raise BaselineRuntimeError(f"{path} is not a strat-v1 shard")
            if _attribute(handle, "field_kind", path) != semantics.field_kind:
                raise BaselineRuntimeError(f"{path} field_kind does not match the frozen contract")
            if int(_attribute(handle, "time_axis", path)) != semantics.time_axis:
                raise BaselineRuntimeError(f"{path} time_axis does not match the frozen contract")
            if (
                "regime_dataset" in handle.attrs
                and _attribute(handle, "regime_dataset", path) != semantics.regime_dataset
            ):
                raise BaselineRuntimeError(
                    f"{path} regime_dataset does not match the frozen contract"
                )
            required = {
                "data",
                semantics.regime_dataset,
                "source_file_index",
                "source_sample_index",
            }
            if semantics.mapping_kind == "steady_operator":
                required.add("targets")
            missing = sorted(required.difference(handle.keys()))
            if missing:
                raise BaselineRuntimeError(f"{path} is missing required datasets: {missing}")
            sample_count = int(handle["data"].shape[0])
            if sample_count <= 0:
                raise BaselineRuntimeError(f"{path} contains no samples")
            for name in required.difference({"targets"}):
                if int(handle[name].shape[0]) != sample_count:
                    raise BaselineRuntimeError(f"{path} dataset {name!r} is not sample-aligned")
            if "targets" in required and handle["targets"].shape != handle["data"].shape:
                raise BaselineRuntimeError(
                    f"{path} steady inputs and targets have different shapes"
                )
            values = np.asarray(handle[semantics.regime_dataset][...], dtype=np.float64)
    except OSError as exc:
        raise BaselineRuntimeError(f"Unable to inspect staged HDF5 object {path}") from exc

    if values.ndim != 1 or not np.isfinite(values).all():
        raise BaselineRuntimeError(f"{path} regime metadata must be finite and one-dimensional")
    unique, counts = np.unique(values, return_counts=True)
    regimes = tuple(RegimeCount(float(value), int(count)) for value, count in zip(unique, counts))
    observed = tuple(item.value for item in regimes)
    if observed != semantics.expected_regimes:
        raise BaselineRuntimeError(f"{path} regimes {observed!r} do not match the frozen contract")
    if len({item.count for item in regimes}) != 1:
        raise BaselineRuntimeError(f"{path} is not balanced across physical-parameter regimes")
    grouped = [np.flatnonzero(values == value).tolist() for value in unique]
    balanced_indices = tuple(
        index
        for offset in range(max(len(indices) for indices in grouped))
        for indices in grouped
        if offset < len(indices)
        for index in (int(indices[offset]),)
    )
    return BaselineSplitRuntime(
        role=role,
        loader_split=loader_split,
        path=path,
        sample_count=sample_count,
        regimes=regimes,
        balanced_indices=balanced_indices,
    )


def bounded_balanced_indices(
    indices: tuple[int, ...] | list[int], *, limit: int | None, regime_count: int
) -> tuple[int, ...]:
    """Return a deterministic balanced prefix, rejecting regime under-coverage."""

    result = tuple(int(index) for index in indices)
    if limit is None:
        return result
    limit = int(limit)
    if limit <= 0:
        raise BaselineRuntimeError("max_samples must be positive")
    if limit < regime_count:
        raise BaselineRuntimeError(
            f"max_samples={limit} would omit at least one of {regime_count} regimes"
        )
    return result[: min(limit, len(result))]


def load_strat_v1_baseline_runtime(
    lock_path: str | Path,
    run_root: str | Path,
    *,
    expected_lock_sha256: str = FROZEN_STRAT_V1_TRAINING_LOCK_SHA256,
) -> StratV1BaselineRuntime:
    """Validate and expose the frozen universal train/validation run view.

    Lock authorization is checked before the run view is enumerated or any HDF5
    object is opened, so a measurement/test lock cannot disclose held-out bytes.
    """

    lock_path = Path(lock_path)
    lock = load_data_lock(lock_path)
    if lock.purpose != "training":
        raise PermissionError("Baseline runtime rejects measurement locks")
    if set(lock.requested_roles) != {"train", "valid"}:
        raise PermissionError("Baseline runtime requires exactly train and valid roles")
    if lock.measurement_contract_id is not None or any(
        item.role == "test" for item in lock.objects
    ):
        raise PermissionError("Baseline runtime rejects all held-out test authorization")
    if lock.lock_sha256 != expected_lock_sha256:
        raise BaselineRuntimeError("Baseline runtime lock does not match the frozen A4 lock")
    if lock.dataset_id != "pdebench-strat-v1-universal":
        raise BaselineRuntimeError("Baseline runtime requires the universal strat-v1 dataset")
    if lock.protocol_id != "pdebench-strat-v1-universal":
        raise BaselineRuntimeError("Baseline runtime requires the universal strat-v1 protocol")
    if lock.adapter != "pdebench_hdf5" or lock.selection.get("protocol") != "strat-v1":
        raise BaselineRuntimeError("Baseline runtime requires the frozen strat-v1 HDF5 adapter")

    expected_ids = {f"{task}-{role}" for task in STRAT_V1_TASKS for role in ("train", "valid")}
    by_id = {item.object_id: item for item in lock.objects}
    if set(by_id) != expected_ids:
        raise BaselineRuntimeError("Baseline runtime lock does not contain the exact A4 task set")

    run_root = Path(run_root)
    # Filename-only inspection protects the physical cache boundary without
    # opening or hashing any possible held-out object.
    if run_root.is_dir() and any(
        "test" in path.name.lower() or "heldout" in path.name.lower()
        for path in run_root.rglob("*.h5")
    ):
        raise PermissionError("Baseline run view must not contain held-out HDF5 objects")

    task_objects: dict[str, dict[str, Path]] = {}
    for task in STRAT_V1_TASKS:
        paths: dict[str, Path] = {}
        for role in ("train", "valid"):
            item = by_id[f"{task}-{role}"]
            expected_name = f"{task}_{'val' if role == 'valid' else 'train'}.h5"
            if Path(item.path).name != expected_name:
                raise BaselineRuntimeError(
                    f"Locked {task} {role} object does not use canonical filename {expected_name}"
                )
            path = run_root / item.path
            if not path.is_file():
                raise BaselineRuntimeError(f"Staged run view is missing {item.path}")
            if path.stat().st_size != item.size_bytes:
                raise BaselineRuntimeError(f"Staged object size does not match lock: {item.path}")
            sha256 = item.checksums.get("sha256")
            if not sha256 or file_checksum(path, "sha256") != sha256:
                raise BaselineRuntimeError(
                    f"Staged object checksum does not match lock: {item.path}"
                )
            paths[role] = path
        if paths["train"].parent != paths["valid"].parent:
            raise BaselineRuntimeError(f"{task} train and validation objects require one task root")
        task_objects[task] = paths

    tasks: dict[str, BaselineTaskRuntime] = {}
    for task, semantics in STRAT_V1_TASKS.items():
        paths = task_objects[task]
        train = _inspect_split(paths["train"], role="train", semantics=semantics)
        valid = _inspect_split(paths["valid"], role="valid", semantics=semantics)
        if tuple(item.value for item in train.regimes) != tuple(
            item.value for item in valid.regimes
        ):
            raise BaselineRuntimeError(f"{task} train and validation regime coverage differs")
        tasks[task] = BaselineTaskRuntime(
            semantics=semantics,
            root=paths["train"].parent,
            train=train,
            valid=valid,
        )

    return StratV1BaselineRuntime(
        lock_path=lock_path,
        run_root=run_root,
        lock=lock,
        selection_sha256=canonical_sha256(lock.selection),
        tasks=tasks,
    )
