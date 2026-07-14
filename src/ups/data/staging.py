from __future__ import annotations

"""Verified, content-addressed staging for immutable dataset objects.

Objects are plain mappings so this layer does not depend on a particular
manifest implementation. Required fields are ``uri`` (or ``uris``), ``size``,
``role``, and a checksum expressed as ``{algorithm, value}``. ``name`` controls
the optional flat run-view filename.
"""

import fcntl
import hashlib
import json
import os
import shutil
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .manifests import RunDataLock
from .transports import TransportError, transfer_to_partial

_DEFAULT_ROLES = frozenset({"train", "valid", "validation"})
_TEST_ROLES = frozenset({"test", "heldout", "held_out"})


class StagingError(RuntimeError):
    """Base error for invalid or unverifiable staging operations."""


class IntegrityError(StagingError):
    """An object's bytes do not match its declared immutable identity."""


class InsufficientSpaceError(StagingError):
    """Scratch space cannot hold the planned working set."""


def staging_objects_from_lock(lock: RunDataLock) -> list[dict[str, Any]]:
    """Adapt a verified run lock into the transport layer's narrow object schema."""

    lock.verify()
    objects = []
    run_names: set[str] = set()
    for item in lock.objects:
        if "sha256" in item.checksums:
            algorithm = "sha256"
        else:
            algorithm = sorted(item.checksums)[0]
        run_name = item.path
        if run_name in run_names:
            raise StagingError(f"Run lock has duplicate run-view path: {run_name}")
        run_names.add(run_name)
        objects.append(
            {
                "id": item.object_id,
                "name": run_name,
                "role": item.role,
                "size": item.size_bytes,
                "checksum": {
                    "algorithm": algorithm,
                    "value": item.checksums[algorithm],
                },
                "uris": list(item.uris),
            }
        )
    return objects


def _checksum_spec(obj: Mapping[str, Any]) -> tuple[str, str]:
    checksum = obj.get("checksum")
    if isinstance(checksum, Mapping):
        algorithm = str(checksum.get("algorithm", "sha256")).lower()
        value = str(checksum.get("value", "")).lower()
    else:
        algorithm = str(obj.get("checksum_algorithm", "sha256")).lower()
        value = str(checksum or obj.get("checksum_value", "")).lower()
    if not value:
        raise StagingError("Every staged object must declare a checksum")
    try:
        expected_length = hashlib.new(algorithm).digest_size * 2
    except ValueError as exc:
        raise StagingError(f"Unsupported checksum algorithm: {algorithm}") from exc
    if expected_length == 0:
        raise StagingError(f"Variable-length checksum algorithm is unsupported: {algorithm}")
    if len(value) != expected_length or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise StagingError(
            f"Invalid {algorithm} digest: expected {expected_length} lowercase hexadecimal characters"
        )
    return algorithm, value


def file_checksum(path: Path, algorithm: str, *, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _verify(path: Path, *, size: int, algorithm: str, digest: str) -> None:
    actual_size = path.stat().st_size
    if actual_size != size:
        raise IntegrityError(f"Size mismatch for {path}: expected {size}, got {actual_size}")
    actual_digest = file_checksum(path, algorithm)
    if actual_digest.lower() != digest.lower():
        raise IntegrityError(
            f"{algorithm} mismatch for {path}: expected {digest}, got {actual_digest}"
        )


def _uris(obj: Mapping[str, Any]) -> tuple[str, ...]:
    values = obj.get("uris")
    if values is None:
        values = (obj.get("uri"),)
    elif isinstance(values, str):
        values = (values,)
    result = tuple(str(value) for value in values if value)
    if not result:
        raise StagingError("Every staged object must declare at least one URI")
    return result


def _run_name(obj: Mapping[str, Any], digest: str) -> str:
    name = str(obj.get("run_name") or obj.get("name") or obj.get("id") or digest)
    path = Path(name)
    if not name or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise StagingError(f"Run-view paths must be safe relative paths, got {name!r}")
    return name


def _selected_objects(
    objects: Iterable[Mapping[str, Any]],
    *,
    allowed_roles: set[str] | frozenset[str] | None,
    allow_test: bool,
) -> list[Mapping[str, Any]]:
    selected = []
    roles = _DEFAULT_ROLES if allowed_roles is None else frozenset(allowed_roles)
    for obj in objects:
        role = str(obj.get("role", "")).lower()
        if role in _TEST_ROLES and not allow_test:
            raise PermissionError(f"Refusing to stage test-role object {obj.get('id', '')!r}")
        if role not in roles and not (allow_test and role in _TEST_ROLES):
            continue
        selected.append(obj)
    return selected


def _cache_path(cache_dir: Path, algorithm: str, digest: str) -> Path:
    # The algorithm namespace permits generic declared checksums without collisions.
    return cache_dir / "objects" / algorithm / digest[:2] / digest


def plan_staging(
    objects: Iterable[Mapping[str, Any]],
    cache_dir: str | Path,
    *,
    allowed_roles: set[str] | frozenset[str] | None = None,
    allow_test: bool = False,
    reserve_bytes: int = 0,
) -> dict[str, Any]:
    """Return an exact byte plan and fail early if scratch space is insufficient."""

    cache_dir = Path(cache_dir)
    selected = _selected_objects(objects, allowed_roles=allowed_roles, allow_test=allow_test)
    missing_bytes = 0
    cached_bytes = 0
    largest_object = 0
    entries = []
    for obj in selected:
        try:
            size = int(obj["size"])
        except (KeyError, TypeError, ValueError) as exc:
            raise StagingError("Every staged object must declare an integer size") from exc
        if size < 0:
            raise StagingError("Object size cannot be negative")
        algorithm, digest = _checksum_spec(obj)
        target = _cache_path(cache_dir, algorithm, digest)
        is_cached = False
        if target.is_file():
            try:
                _verify(target, size=size, algorithm=algorithm, digest=digest)
                is_cached = True
            except IntegrityError:
                # Do not mutate during planning, but reserve enough room to repair it.
                pass
        if is_cached:
            cached_bytes += size
        else:
            partial = target.with_name(f".{target.name}.partial")
            partial_size = min(partial.stat().st_size, size) if partial.exists() else 0
            missing_bytes += size - partial_size
            largest_object = max(largest_object, size)
        entries.append(
            {
                "id": str(obj.get("id", obj.get("name", digest))),
                "role": str(obj.get("role", "")),
                "size": size,
                "cached": is_cached,
                "cache_path": str(target),
            }
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(cache_dir).free
    required_bytes = missing_bytes + max(0, int(reserve_bytes))
    if free_bytes < required_bytes:
        raise InsufficientSpaceError(
            f"Insufficient space in {cache_dir}: need {required_bytes} bytes "
            f"({missing_bytes} data + {reserve_bytes} reserve), have {free_bytes}"
        )
    return {
        "object_count": len(entries),
        "cached_bytes": cached_bytes,
        "missing_bytes": missing_bytes,
        "largest_object_bytes": largest_object,
        "reserve_bytes": max(0, int(reserve_bytes)),
        "free_bytes": free_bytes,
        "entries": entries,
    }


def _atomic_run_link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        try:
            if os.path.samefile(source, destination):
                return
        except OSError:
            pass
        raise StagingError(f"Run-view destination already exists with other content: {destination}")
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.link")
    try:
        os.link(source, temporary)
    except OSError:
        temporary.symlink_to(source.resolve())
    os.replace(temporary, destination)


def stage_objects(
    objects: Iterable[Mapping[str, Any]],
    cache_dir: str | Path,
    *,
    run_dir: str | Path | None = None,
    allowed_roles: set[str] | frozenset[str] | None = None,
    allow_test: bool = False,
    reserve_bytes: int = 0,
    report_path: str | Path | None = None,
    http_headers: Mapping[str, str] | None = None,
    timeout: float = 60.0,
) -> dict[str, Any]:
    """Stage locked objects and optionally create a flat compatibility run view."""

    started = time.monotonic()
    object_list = list(objects)
    selected = _selected_objects(object_list, allowed_roles=allowed_roles, allow_test=allow_test)
    plan = plan_staging(
        selected,
        cache_dir,
        allowed_roles=(allowed_roles or _DEFAULT_ROLES) | (_TEST_ROLES if allow_test else set()),
        allow_test=allow_test,
        reserve_bytes=reserve_bytes,
    )
    cache_dir = Path(cache_dir)
    staged = []
    total_transferred = 0
    cache_hits = 0

    for obj in selected:
        size = int(obj["size"])
        algorithm, digest = _checksum_spec(obj)
        destination = _cache_path(cache_dir, algorithm, digest)
        destination.parent.mkdir(parents=True, exist_ok=True)
        lock_path = destination.with_name(f".{destination.name}.lock")
        with lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            hit = False
            if destination.is_file():
                try:
                    _verify(destination, size=size, algorithm=algorithm, digest=digest)
                    hit = True
                    cache_hits += 1
                except IntegrityError:
                    destination.unlink()

            used_uri = None
            transferred = 0
            if not hit:
                partial = destination.with_name(f".{destination.name}.partial")
                if partial.is_file() and partial.stat().st_size == size:
                    try:
                        _verify(partial, size=size, algorithm=algorithm, digest=digest)
                        used_uri = "completed-partial"
                    except IntegrityError:
                        partial.unlink()
                errors = []
                if used_uri is None:
                    for uri in _uris(obj):
                        try:
                            transferred += transfer_to_partial(
                                uri,
                                partial,
                                headers=http_headers,
                                timeout=timeout,
                            )
                            used_uri = uri
                            break
                        except TransportError as exc:
                            errors.append(str(exc))
                if used_uri is None:
                    raise StagingError("; ".join(errors))
                # This order is the core integrity guarantee: no unverified bytes
                # ever receive their immutable content-addressed final path.
                try:
                    _verify(partial, size=size, algorithm=algorithm, digest=digest)
                except IntegrityError:
                    partial.unlink(missing_ok=True)
                    raise
                os.replace(partial, destination)
                # Persist the directory entry before reporting success.
                directory_fd = os.open(destination.parent, os.O_DIRECTORY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
                total_transferred += transferred

        name = _run_name(obj, digest)
        if run_dir is not None:
            _atomic_run_link(destination, Path(run_dir) / name)
        staged.append(
            {
                "id": str(obj.get("id", obj.get("name", digest))),
                "name": name,
                "role": str(obj.get("role", "")),
                "size": size,
                "checksum": {"algorithm": algorithm, "value": digest},
                "cache_path": str(destination),
                "cache_hit": hit,
                "source_uri": used_uri,
                "bytes_transferred": transferred,
            }
        )

    report = {
        "schema_version": 1,
        "status": "complete",
        "duration_seconds": round(time.monotonic() - started, 6),
        "object_count": len(staged),
        "cache_hits": cache_hits,
        "bytes_transferred": total_transferred,
        "plan": plan,
        "objects": staged,
    }
    if report_path is not None:
        output = Path(report_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, output)
    return report
