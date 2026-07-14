from __future__ import annotations

"""Checksum-bound, training-only normalization statistics."""

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


@dataclass(frozen=True)
class NormalizationStats:
    mean: tuple[float, ...]
    std: tuple[float, ...]
    count: int
    channel_axis: int = -1
    method: str = "zscore"
    fit_role: str = "train"
    data_lock_sha256: str | None = None
    selection_sha256: str | None = None
    version: int = 1

    def __post_init__(self) -> None:
        if self.version != 1:
            raise ValueError(f"Unsupported normalization version: {self.version}")
        if self.method != "zscore":
            raise ValueError(f"Unsupported normalization method: {self.method}")
        if self.fit_role != "train":
            raise ValueError("Normalization statistics must be fitted on training data only")
        if self.count <= 0:
            raise ValueError("Normalization count must be positive")
        if not self.mean or len(self.mean) != len(self.std):
            raise ValueError("Normalization mean/std must have equal non-zero channel counts")
        if any(value <= 0 for value in self.std):
            raise ValueError("Normalization standard deviations must be positive")

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "version": self.version,
            "method": self.method,
            "fit_role": self.fit_role,
            "channel_axis": self.channel_axis,
            "count": self.count,
            "mean": list(self.mean),
            "std": list(self.std),
            "data_lock_sha256": self.data_lock_sha256,
            "selection_sha256": self.selection_sha256,
        }
        if include_sha256:
            payload["sha256"] = self.sha256
        return payload

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.to_dict(include_sha256=False))).hexdigest()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(path)

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        expected_data_lock_sha256: str | None = None,
        expected_selection_sha256: str | None = None,
    ) -> NormalizationStats:
        payload = json.loads(path.read_text(encoding="utf-8"))
        expected_sha256 = payload.pop("sha256", None)
        stats = cls(
            mean=tuple(float(value) for value in payload["mean"]),
            std=tuple(float(value) for value in payload["std"]),
            count=int(payload["count"]),
            channel_axis=int(payload.get("channel_axis", -1)),
            method=str(payload.get("method", "zscore")),
            fit_role=str(payload.get("fit_role", "train")),
            data_lock_sha256=payload.get("data_lock_sha256"),
            selection_sha256=payload.get("selection_sha256"),
            version=int(payload.get("version", 1)),
        )
        if expected_sha256 is not None and expected_sha256 != stats.sha256:
            raise ValueError(f"Normalization statistics checksum mismatch: {path}")
        if (
            expected_data_lock_sha256 is not None
            and stats.data_lock_sha256 != expected_data_lock_sha256
        ):
            raise ValueError("Normalization statistics do not match the resolved data lock")
        if (
            expected_selection_sha256 is not None
            and stats.selection_sha256 != expected_selection_sha256
        ):
            raise ValueError("Normalization statistics do not match the selected sample index")
        return stats

    def apply(self, fields: torch.Tensor) -> torch.Tensor:
        axis = self.channel_axis if self.channel_axis >= 0 else fields.ndim + self.channel_axis
        if axis < 0 or axis >= fields.ndim:
            raise ValueError(
                f"Normalization channel axis {self.channel_axis} is invalid for {fields.shape}"
            )
        if fields.shape[axis] != len(self.mean):
            raise ValueError(
                f"Normalization has {len(self.mean)} channels but fields have {fields.shape[axis]}"
            )
        shape = [1] * fields.ndim
        shape[axis] = len(self.mean)
        mean = fields.new_tensor(self.mean).reshape(shape)
        std = fields.new_tensor(self.std).reshape(shape)
        return (fields - mean) / std


def fit_normalization_stats(
    fields: Iterable[torch.Tensor],
    *,
    channel_axis: int = -1,
    data_lock_sha256: str | None = None,
    selection_sha256: str | None = None,
    min_std: float = 1e-6,
) -> NormalizationStats:
    """Fit per-channel statistics incrementally without materializing the dataset."""

    count = 0
    channel_sum: torch.Tensor | None = None
    channel_sq_sum: torch.Tensor | None = None
    for raw in fields:
        tensor = raw.detach().to(dtype=torch.float64, device="cpu")
        axis = channel_axis if channel_axis >= 0 else tensor.ndim + channel_axis
        if axis < 0 or axis >= tensor.ndim:
            raise ValueError(f"Invalid channel axis {channel_axis} for {tuple(tensor.shape)}")
        moved = tensor.movedim(axis, -1).reshape(-1, tensor.shape[axis])
        if not bool(torch.isfinite(moved).all()):
            raise ValueError("Cannot fit normalization statistics on non-finite fields")
        current_sum = moved.sum(dim=0)
        current_sq_sum = moved.square().sum(dim=0)
        if channel_sum is None:
            channel_sum = current_sum
            channel_sq_sum = current_sq_sum
        else:
            if channel_sum.shape != current_sum.shape:
                raise ValueError("Inconsistent channel counts while fitting normalization")
            channel_sum += current_sum
            assert channel_sq_sum is not None
            channel_sq_sum += current_sq_sum
        count += int(moved.shape[0])
    if count == 0 or channel_sum is None or channel_sq_sum is None:
        raise ValueError("Cannot fit normalization statistics on an empty training dataset")
    mean = channel_sum / count
    variance = (channel_sq_sum / count - mean.square()).clamp_min(0.0)
    std = variance.sqrt().clamp_min(float(min_std))
    return NormalizationStats(
        mean=tuple(float(value) for value in mean.tolist()),
        std=tuple(float(value) for value in std.tolist()),
        count=count,
        channel_axis=channel_axis,
        data_lock_sha256=data_lock_sha256,
        selection_sha256=selection_sha256,
    )
