from __future__ import annotations

"""Resolve the shared PDE parameter-conditioning contract.

The resolver keeps dataset selection, conditioner construction, training, and
evaluation on one deterministic parameter vocabulary.  Legacy single-task
configs continue to use ``data.param_keys`` and ``data.root`` unchanged.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch

_TRANSFORM_KINDS = {"identity", "log10", "log10_zscore", "log1p"}


@dataclass(frozen=True)
class ParameterTransform:
    kind: str = "identity"
    mean: float | None = None
    std: float | None = None
    count: int | None = None
    source_sha256: str | None = None


@dataclass(frozen=True)
class ParameterConditioningContract:
    task_names: tuple[str, ...]
    task_param_keys: Mapping[str, tuple[str, ...]]
    task_roots: Mapping[str, str]
    param_vocab: tuple[str, ...]
    task_transforms: Mapping[str, Mapping[str, ParameterTransform]]
    default_param_keys: tuple[str, ...]
    default_root: str | None

    def param_keys_for(self, task_name: str) -> tuple[str, ...]:
        if task_name not in self.task_names:
            raise ValueError(f"Unknown task {task_name!r} for parameter-conditioning contract")
        return self.task_param_keys.get(task_name, self.default_param_keys)

    def root_for(self, task_name: str) -> str | None:
        if task_name not in self.task_names:
            raise ValueError(f"Unknown task {task_name!r} for parameter-conditioning contract")
        return self.task_roots.get(task_name, self.default_root)

    def transforms_for(self, task_name: str) -> Mapping[str, ParameterTransform]:
        if task_name not in self.task_names:
            raise ValueError(f"Unknown task {task_name!r} for parameter-conditioning contract")
        return self.task_transforms.get(task_name, {})


def _string_tuple(raw: Any, *, setting: str) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str) or not isinstance(raw, Sequence):
        raise ValueError(f"{setting} must be a sequence of parameter names")
    values = tuple(str(value) for value in raw)
    if any(not value for value in values):
        raise ValueError(f"{setting} must not contain empty parameter names")
    if len(set(values)) != len(values):
        raise ValueError(f"{setting} must not contain duplicate parameter names")
    return values


def _task_mapping(raw: Any, *, setting: str) -> Mapping[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"{setting} must be a mapping keyed by task name")
    return {str(key): value for key, value in raw.items()}


def _parse_transform(raw: Any, *, setting: str) -> ParameterTransform:
    if isinstance(raw, str):
        spec = ParameterTransform(kind=raw)
    elif isinstance(raw, Mapping):
        unknown = set(raw) - {"kind", "mean", "std", "count", "source_sha256"}
        if unknown:
            raise ValueError(f"{setting} contains unknown fields: {sorted(unknown)}")
        spec = ParameterTransform(
            kind=str(raw.get("kind", "identity")),
            mean=float(raw["mean"]) if raw.get("mean") is not None else None,
            std=float(raw["std"]) if raw.get("std") is not None else None,
            count=int(raw["count"]) if raw.get("count") is not None else None,
            source_sha256=(
                str(raw["source_sha256"]) if raw.get("source_sha256") is not None else None
            ),
        )
    else:
        raise ValueError(f"{setting} must be a transform name or mapping")
    if spec.kind not in _TRANSFORM_KINDS:
        raise ValueError(f"{setting}.kind must be one of {sorted(_TRANSFORM_KINDS)}")
    if spec.std is not None and spec.std <= 0.0:
        raise ValueError(f"{setting}.std must be positive")
    if spec.kind == "log10_zscore":
        if spec.mean is None or spec.std is None or spec.count is None or not spec.source_sha256:
            raise ValueError(
                f"{setting} log10_zscore requires frozen mean, std, count, and source_sha256"
            )
        if spec.count <= 0:
            raise ValueError(f"{setting}.count must be positive")
        if len(spec.source_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in spec.source_sha256.lower()
        ):
            raise ValueError(f"{setting}.source_sha256 must be a 64-character SHA-256")
    return spec


def resolve_parameter_conditioning(
    data_cfg: Mapping[str, Any], *, task_names: Sequence[str] | None = None
) -> ParameterConditioningContract:
    """Validate and resolve task roots, parameter keys, vocab, and transforms."""

    if task_names is None:
        raw_tasks = data_cfg.get("task")
        if isinstance(raw_tasks, str):
            task_names = (raw_tasks,)
        elif isinstance(raw_tasks, Sequence) and not isinstance(raw_tasks, str):
            task_names = tuple(str(task) for task in raw_tasks)
        else:
            raise ValueError("data.task must name one task or a non-empty task sequence")
    tasks = tuple(str(task) for task in task_names)
    if not tasks or any(not task for task in tasks) or len(set(tasks)) != len(tasks):
        raise ValueError("Resolved task names must be non-empty and unique")

    default_keys = _string_tuple(data_cfg.get("param_keys"), setting="data.param_keys")
    raw_task_keys = _task_mapping(data_cfg.get("task_param_keys"), setting="data.task_param_keys")
    unknown_tasks = set(raw_task_keys) - set(tasks)
    if unknown_tasks:
        raise ValueError(
            f"data.task_param_keys contains unknown task entries: {sorted(unknown_tasks)}"
        )
    if raw_task_keys and set(raw_task_keys) != set(tasks):
        missing = sorted(set(tasks) - set(raw_task_keys))
        raise ValueError(f"data.task_param_keys is missing selected tasks: {missing}")
    task_param_keys = {
        task: _string_tuple(keys, setting=f"data.task_param_keys[{task!r}]")
        for task, keys in raw_task_keys.items()
    }

    raw_roots = _task_mapping(data_cfg.get("task_roots"), setting="data.task_roots")
    unknown_root_tasks = set(raw_roots) - set(tasks)
    if unknown_root_tasks:
        raise ValueError(
            f"data.task_roots contains unknown task entries: {sorted(unknown_root_tasks)}"
        )
    task_roots = {task: str(root) for task, root in raw_roots.items()}
    if any(not root for root in task_roots.values()):
        raise ValueError("data.task_roots values must be non-empty paths")

    all_keys = (
        default_keys
        if not task_param_keys
        else tuple(key for task in tasks for key in task_param_keys[task])
    )
    param_vocab = tuple(sorted(set(all_keys)))

    raw_transforms_by_task = _task_mapping(
        data_cfg.get("parameter_transforms"), setting="data.parameter_transforms"
    )
    unknown_transform_tasks = set(raw_transforms_by_task) - set(tasks)
    if unknown_transform_tasks:
        raise ValueError(
            "data.parameter_transforms contains unknown task entries: "
            f"{sorted(unknown_transform_tasks)}"
        )
    task_transforms: dict[str, dict[str, ParameterTransform]] = {}
    for task, raw_transforms in raw_transforms_by_task.items():
        if not isinstance(raw_transforms, Mapping):
            raise ValueError(f"data.parameter_transforms[{task!r}] must map keys to transforms")
        expected_keys = set(task_param_keys.get(task, default_keys))
        unknown_params = set(raw_transforms) - expected_keys
        if unknown_params:
            raise ValueError(
                f"data.parameter_transforms[{task!r}] contains keys outside the task schema: "
                f"{sorted(unknown_params)}"
            )
        task_transforms[task] = {
            str(key): _parse_transform(
                value, setting=f"data.parameter_transforms[{task!r}][{key!r}]"
            )
            for key, value in raw_transforms.items()
        }
    _validate_transform_source_bindings(data_cfg, task_transforms)
    return ParameterConditioningContract(
        task_names=tasks,
        task_param_keys=task_param_keys,
        task_roots=task_roots,
        param_vocab=param_vocab,
        task_transforms=task_transforms,
        default_param_keys=default_keys,
        default_root=str(data_cfg["root"]) if data_cfg.get("root") is not None else None,
    )


def _validate_transform_source_bindings(
    data_cfg: Mapping[str, Any],
    task_transforms: Mapping[str, Mapping[str, ParameterTransform]],
) -> None:
    """Bind frozen transform statistics to training objects in a supplied lock."""

    lock_path_raw = data_cfg.get("data_lock_path")
    if not task_transforms or lock_path_raw is None:
        return
    lock_path = Path(str(lock_path_raw)).resolve()
    training_sources = _training_sources_from_lock(str(lock_path))
    for task, transforms in task_transforms.items():
        task_sources = training_sources.get(task, frozenset())
        for key, spec in transforms.items():
            if spec.source_sha256 is not None and spec.source_sha256 not in task_sources:
                raise ValueError(
                    f"data.parameter_transforms[{task!r}][{key!r}].source_sha256 "
                    "does not match the task's training object in data.data_lock_path"
                )


@lru_cache(maxsize=32)
def _training_sources_from_lock(lock_path_str: str) -> Mapping[str, frozenset[str]]:
    """Read immutable lock bindings once per process."""

    lock_path = Path(lock_path_str)
    if not lock_path.is_file():
        raise ValueError(f"data.data_lock_path does not exist: {lock_path}")
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read parameter transform source lock: {lock_path}") from exc
    objects = lock.get("objects")
    if not isinstance(objects, list):
        raise ValueError("Parameter transform source lock must contain an objects list")
    sources: dict[str, set[str]] = {}
    for obj in objects:
        if not isinstance(obj, Mapping) or obj.get("role") != "train":
            continue
        object_id = str(obj.get("object_id", ""))
        path_name = Path(str(obj.get("path", ""))).name
        task = object_id.removesuffix("-train") if object_id.endswith("-train") else None
        if task is None and path_name.endswith("_train.h5"):
            task = path_name.removesuffix("_train.h5")
        checksum = obj.get("checksums", {}).get("sha256")
        if task and checksum:
            sources.setdefault(task, set()).add(str(checksum))
    return {task: frozenset(checksums) for task, checksums in sources.items()}


def transform_parameter(
    key: str, tensor: torch.Tensor, transforms: Mapping[str, ParameterTransform] | None
) -> torch.Tensor:
    """Apply an explicitly configured transform; unconfigured values stay raw."""

    spec = transforms.get(key) if transforms else None
    if spec is None or spec.kind == "identity":
        result = tensor.float()
    elif spec.kind in {"log10", "log10_zscore"}:
        if torch.any(tensor <= 0):
            raise ValueError(f"Parameter {key!r} must be positive for log10 conditioning")
        result = torch.log10(tensor.float())
    elif spec.kind == "log1p":
        if torch.any(tensor <= -1):
            raise ValueError(f"Parameter {key!r} must be greater than -1 for log1p conditioning")
        result = torch.log1p(tensor.float())
    else:  # pragma: no cover - specs are validated by the resolver
        raise ValueError(f"Unsupported parameter transform {spec.kind!r}")
    if spec is not None and spec.mean is not None:
        result = result - spec.mean
    if spec is not None and spec.std is not None:
        result = result / spec.std
    if not torch.isfinite(result).all():
        raise ValueError(f"Parameter transform for {key!r} produced non-finite values")
    return result
