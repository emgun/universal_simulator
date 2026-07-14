"""Atomic, integrity-bound checkpoints for exactly resumable training.

The checkpoint and its record are deliberately separate.  The binary checkpoint
is first atomically replaced, then a canonical JSON record is atomically replaced
to commit the checkpoint.  A checkpoint without a matching record is incomplete
and cannot be loaded.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

FORMAT_VERSION = "ups-resumable-checkpoint-v3"
RECORD_SUFFIX = ".record.json"


class CheckpointError(RuntimeError):
    """Base class for checkpoint failures."""


class CheckpointIntegrityError(CheckpointError):
    """Raised when checkpoint bytes or their record fail integrity checks."""


class CheckpointCompatibilityError(CheckpointError):
    """Raised when a checkpoint does not match the requested training contract."""


@dataclass(frozen=True)
class CheckpointBindings:
    """Specs and fingerprints that must match before state can be restored."""

    model_spec: Mapping[str, Any]
    optimizer_spec: Mapping[str, Any]
    normalizer_spec: Mapping[str, Any]
    plan_fingerprint: str
    data_fingerprint: str
    source_fingerprint: str
    runtime_fingerprint: str


@dataclass(frozen=True)
class TrainingProgress:
    """Counters and JSON-compatible history accumulated through a checkpoint."""

    completed_epoch: int
    steps: int
    examples: int
    history: Sequence[Any]


@dataclass(frozen=True)
class CheckpointRecord:
    """Integrity record written beside a binary checkpoint."""

    format_version: str
    checkpoint_sha256: str
    checkpoint_bytes: int
    parent_checkpoint_sha256: str | None
    self_hash: Mapping[str, str]


@dataclass(frozen=True)
class LoadedCheckpoint:
    """Verified progress and lineage returned after restoration."""

    progress: TrainingProgress
    record: CheckpointRecord


def checkpoint_record_path(path: str | Path) -> Path:
    """Return the commit-record path corresponding to ``path``."""

    checkpoint_path = Path(path)
    return checkpoint_path.with_name(checkpoint_path.name + RECORD_SUFFIX)


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("checkpoint metadata must be finite and JSON-compatible") from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_torch_save(payload: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_write_bytes(value: bytes, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _bindings_payload(bindings: CheckpointBindings) -> dict[str, Any]:
    for label in ("model_spec", "optimizer_spec", "normalizer_spec"):
        if not isinstance(getattr(bindings, label), Mapping):
            raise TypeError(f"{label} must be a mapping")
    for label in (
        "plan_fingerprint",
        "data_fingerprint",
        "source_fingerprint",
        "runtime_fingerprint",
    ):
        if not isinstance(getattr(bindings, label), str) or not getattr(bindings, label):
            raise ValueError(f"{label} must be a non-empty string")
    payload = asdict(bindings)
    # Validate eagerly so a checkpoint can never be written with metadata that
    # cannot later be compared canonically.
    _canonical_json(payload)
    return payload


def _progress_payload(progress: TrainingProgress) -> dict[str, Any]:
    if any(
        not isinstance(value, int) or isinstance(value, bool)
        for value in (progress.completed_epoch, progress.steps, progress.examples)
    ):
        raise TypeError("training progress counters must be integers")
    if progress.completed_epoch < 0 or progress.steps < 0 or progress.examples < 0:
        raise ValueError("training progress counters must be non-negative")
    payload = {
        "completed_epoch": progress.completed_epoch,
        "steps": progress.steps,
        "examples": progress.examples,
        "history": list(progress.history),
    }
    _canonical_json(payload)
    return payload


def _state_dict(value: Any, label: str) -> Mapping[str, Any]:
    state_dict = getattr(value, "state_dict", None)
    if not callable(state_dict):
        raise TypeError(f"{label} must provide state_dict()")
    state = state_dict()
    if not isinstance(state, Mapping):
        raise TypeError(f"{label}.state_dict() must return a mapping")
    return state


def _tensor_only_model_state(model: torch.nn.Module) -> OrderedDict[str, torch.Tensor]:
    """Return a weights-only-safe model state without framework configuration objects.

    Some model libraries place constructor metadata, including Python callables and
    classes, directly in ``state_dict()``.  Those objects are not model weights and
    cannot be decoded by PyTorch's restricted ``weights_only`` loader.  Checkpoints
    deliberately keep only tensors and fail closed for any unexpected extra state.
    NeuralOperator's redundant top-level ``_metadata`` entry is the one supported
    exception; the standard PyTorch ``state_dict._metadata`` attribute is retained.
    """

    raw_state = _state_dict(model, "model")
    safe_state: OrderedDict[str, torch.Tensor] = OrderedDict()
    unsupported: list[str] = []
    for key, value in raw_state.items():
        if isinstance(value, torch.Tensor):
            safe_state[str(key)] = value
        elif key != "_metadata":
            unsupported.append(f"{key} ({type(value).__name__})")
    if unsupported:
        raise TypeError(
            "model.state_dict() contains unsupported non-tensor state: "
            + ", ".join(sorted(unsupported))
        )
    pytorch_metadata = getattr(raw_state, "_metadata", None)
    if pytorch_metadata is not None:
        if not isinstance(pytorch_metadata, Mapping):
            raise TypeError("model state metadata must be a mapping")
        # Module version metadata is made only of restricted-loader-safe primitives.
        _canonical_json(pytorch_metadata)
        safe_state._metadata = OrderedDict(pytorch_metadata)  # type: ignore[attr-defined]
    return safe_state


def save_training_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    progress: TrainingProgress,
    sampler_generator: torch.Generator,
    bindings: CheckpointBindings,
    normalizer: Any | None = None,
    parent_checkpoint_sha256: str | None = None,
) -> CheckpointRecord:
    """Atomically save a checkpoint and commit it with a self-hashed record."""

    if parent_checkpoint_sha256 is not None and (
        len(parent_checkpoint_sha256) != 64
        or any(character not in "0123456789abcdef" for character in parent_checkpoint_sha256)
    ):
        raise ValueError("parent_checkpoint_sha256 must be a lowercase SHA-256 digest")

    checkpoint_path = Path(path)
    payload: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "bindings": _bindings_payload(bindings),
        "progress": _progress_payload(progress),
        "parent_checkpoint_sha256": parent_checkpoint_sha256,
        "model_state": _tensor_only_model_state(model),
        "optimizer_state": optimizer.state_dict(),
        "normalizer_state": None if normalizer is None else _state_dict(normalizer, "normalizer"),
        "sampler_generator_state": sampler_generator.get_state(),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state_all": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
    }
    _atomic_torch_save(payload, checkpoint_path)

    record_without_hash: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "parent_checkpoint_sha256": parent_checkpoint_sha256,
        "self_hash": {
            "algorithm": "sha256",
            "excluded_field": "self_hash.value",
        },
    }
    self_hash = _sha256_bytes(_canonical_json(record_without_hash))
    record_payload = json.loads(_canonical_json(record_without_hash))
    record_payload["self_hash"]["value"] = self_hash
    _atomic_write_bytes(
        json.dumps(record_payload, indent=2, sort_keys=True).encode("utf-8") + b"\n",
        checkpoint_record_path(checkpoint_path),
    )
    return _record_from_payload(record_payload)


def _record_from_payload(payload: Mapping[str, Any]) -> CheckpointRecord:
    try:
        format_version = payload["format_version"]
        checkpoint_sha256 = payload["checkpoint_sha256"]
        checkpoint_bytes = payload["checkpoint_bytes"]
        parent_checkpoint_sha256 = payload.get("parent_checkpoint_sha256")
        self_hash = payload["self_hash"]
        if not isinstance(format_version, str):
            raise TypeError
        if not isinstance(checkpoint_sha256, str):
            raise TypeError
        if not isinstance(checkpoint_bytes, int) or isinstance(checkpoint_bytes, bool):
            raise TypeError
        if parent_checkpoint_sha256 is not None and not isinstance(parent_checkpoint_sha256, str):
            raise TypeError
        if not isinstance(self_hash, Mapping):
            raise TypeError
        return CheckpointRecord(
            format_version=format_version,
            checkpoint_sha256=checkpoint_sha256,
            checkpoint_bytes=checkpoint_bytes,
            parent_checkpoint_sha256=parent_checkpoint_sha256,
            self_hash=dict(self_hash),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CheckpointIntegrityError("checkpoint record has an invalid schema") from exc


def verify_checkpoint_record(
    path: str | Path, *, expected_checkpoint_sha256: str | None = None
) -> CheckpointRecord:
    """Verify a checkpoint's commit record and exact binary hash without loading state."""

    checkpoint_path = Path(path)
    record_path = checkpoint_record_path(checkpoint_path)
    try:
        raw_payload = json.loads(record_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
        raise CheckpointIntegrityError("checkpoint has no readable commit record") from exc
    if not isinstance(raw_payload, dict):
        raise CheckpointIntegrityError("checkpoint record must be a JSON object")

    record = _record_from_payload(raw_payload)
    if record.format_version != FORMAT_VERSION:
        raise CheckpointCompatibilityError(
            f"unsupported checkpoint format {record.format_version!r}"
        )
    unhashed = dict(raw_payload)
    self_hash = dict(record.self_hash)
    recorded_hash = self_hash.pop("value", None)
    unhashed["self_hash"] = self_hash
    if self_hash != {"algorithm": "sha256", "excluded_field": "self_hash.value"}:
        raise CheckpointIntegrityError("checkpoint record has an invalid self-hash contract")
    if recorded_hash != _sha256_bytes(_canonical_json(unhashed)):
        raise CheckpointIntegrityError("checkpoint record self-hash does not match")
    if (
        len(record.checkpoint_sha256) != 64
        or any(character not in "0123456789abcdef" for character in record.checkpoint_sha256)
        or record.checkpoint_bytes < 0
    ):
        raise CheckpointIntegrityError("checkpoint record has invalid binary metadata")
    if not checkpoint_path.is_file():
        raise CheckpointIntegrityError("checkpoint binary is missing")
    if checkpoint_path.stat().st_size != record.checkpoint_bytes:
        raise CheckpointIntegrityError("checkpoint byte count does not match its record")
    if _sha256_file(checkpoint_path) != record.checkpoint_sha256:
        raise CheckpointIntegrityError("checkpoint SHA-256 does not match its record")
    if (
        expected_checkpoint_sha256 is not None
        and record.checkpoint_sha256 != expected_checkpoint_sha256
    ):
        raise CheckpointCompatibilityError("checkpoint SHA-256 does not match expected value")
    return record


def _require_equal(label: str, actual: Any, expected: Any) -> None:
    if _canonical_json(actual) != _canonical_json(expected):
        raise CheckpointCompatibilityError(f"checkpoint {label} does not match expected value")


def load_training_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    sampler_generator: torch.Generator,
    expected_bindings: CheckpointBindings,
    normalizer: Any | None = None,
    expected_parent_checkpoint_sha256: str | None = None,
    expected_checkpoint_sha256: str | None = None,
    map_location: str | torch.device = "cpu",
) -> LoadedCheckpoint:
    """Verify compatibility and integrity, then restore a training checkpoint.

    No supplied object is mutated until the record, binary hash, format,
    lineage, specs, and fingerprints have all been checked.
    """

    checkpoint_path = Path(path)
    record = verify_checkpoint_record(
        checkpoint_path, expected_checkpoint_sha256=expected_checkpoint_sha256
    )
    try:
        payload = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    except Exception as exc:
        raise CheckpointIntegrityError("checkpoint binary could not be decoded safely") from exc
    if not isinstance(payload, dict):
        raise CheckpointIntegrityError("checkpoint payload must be a mapping")
    if payload.get("format_version") != FORMAT_VERSION:
        raise CheckpointCompatibilityError("checkpoint payload format does not match")
    _require_equal("bindings", payload.get("bindings"), _bindings_payload(expected_bindings))
    _require_equal(
        "parent checkpoint SHA-256",
        payload.get("parent_checkpoint_sha256"),
        expected_parent_checkpoint_sha256,
    )
    _require_equal(
        "record parent checkpoint SHA-256",
        record.parent_checkpoint_sha256,
        expected_parent_checkpoint_sha256,
    )
    normalizer_state = payload.get("normalizer_state")
    if (normalizer is None) != (normalizer_state is None):
        raise CheckpointCompatibilityError("checkpoint normalizer state presence does not match")

    try:
        progress_payload = payload["progress"]
        progress = TrainingProgress(
            completed_epoch=int(progress_payload["completed_epoch"]),
            steps=int(progress_payload["steps"]),
            examples=int(progress_payload["examples"]),
            history=list(progress_payload["history"]),
        )
        _progress_payload(progress)
        model_state = payload["model_state"]
        optimizer_state = payload["optimizer_state"]
        sampler_state = payload["sampler_generator_state"]
        cpu_rng_state = payload["torch_cpu_rng_state"]
        cuda_rng_state = payload["torch_cuda_rng_state_all"]
    except (KeyError, TypeError, ValueError) as exc:
        raise CheckpointIntegrityError("checkpoint payload has an invalid schema") from exc

    if normalizer is not None and not callable(getattr(normalizer, "load_state_dict", None)):
        raise TypeError("normalizer must provide load_state_dict()")
    if cuda_rng_state and not torch.cuda.is_available():
        raise CheckpointCompatibilityError(
            "checkpoint contains CUDA RNG state but CUDA is unavailable"
        )
    if cuda_rng_state and len(cuda_rng_state) != torch.cuda.device_count():
        raise CheckpointCompatibilityError(
            "checkpoint CUDA RNG device count differs from the current runtime"
        )

    # All fail-closed checks above precede mutation.  RNG is restored last so
    # deserialization and state loading cannot alter the resumed random stream.
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)
    if normalizer is not None:
        load_state_dict = normalizer.load_state_dict
        load_state_dict(normalizer_state)
    sampler_generator.set_state(sampler_state)
    torch.set_rng_state(cpu_rng_state.cpu())
    if cuda_rng_state:
        torch.cuda.set_rng_state_all(cuda_rng_state)
    return LoadedCheckpoint(progress=progress, record=record)
