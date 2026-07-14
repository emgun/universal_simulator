from __future__ import annotations

"""Command-line control plane for immutable, staged scientific data."""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ups.data.manifests import (
    RunDataLock,
    canonical_sha256,
    load_data_lock,
    load_protocol_manifest,
    load_source_manifest,
    resolve_data_lock,
    write_data_lock,
)
from ups.data.normalization import fit_normalization_stats
from ups.data.pdebench import PDEBenchConfig, PDEBenchDataset
from ups.data.staging import (
    IntegrityError,
    file_checksum,
    plan_staging,
    stage_objects,
    staging_objects_from_lock,
)


def _objects(lock: RunDataLock) -> list[dict[str, Any]]:
    return staging_objects_from_lock(lock)


def _roles(lock: RunDataLock) -> set[str]:
    return set(lock.requested_roles)


def _allow_test(lock: RunDataLock) -> bool:
    return lock.purpose == "measurement"


def _cache_path(cache_dir: Path, obj: dict[str, Any]) -> Path:
    checksum = obj["checksum"]
    digest = checksum["value"]
    return cache_dir / "objects" / checksum["algorithm"] / digest[:2] / digest


def verify_lock_cache(lock: RunDataLock, cache_dir: Path) -> dict[str, Any]:
    """Verify every locked object currently exists in the content-addressed cache."""

    verified = []
    for obj in _objects(lock):
        path = _cache_path(cache_dir, obj)
        if not path.is_file():
            raise FileNotFoundError(f"Locked object is not staged: {obj['id']} ({path})")
        size = path.stat().st_size
        if size != obj["size"]:
            raise IntegrityError(
                f"Size mismatch for {obj['id']}: expected {obj['size']}, got {size}"
            )
        checksum = obj["checksum"]
        digest = file_checksum(path, checksum["algorithm"])
        if digest != checksum["value"]:
            raise IntegrityError(f"Checksum mismatch for locked object {obj['id']}")
        verified.append({"id": obj["id"], "path": str(path), "size": size})
    return {"status": "verified", "lock_sha256": lock.lock_sha256, "objects": verified}


def pinned_cache_paths(cache_dir: Path, locks: Sequence[RunDataLock]) -> set[Path]:
    return {
        _cache_path(cache_dir, obj) for lock in locks for obj in staging_objects_from_lock(lock)
    }


def evict_unpinned_cache(
    cache_dir: Path,
    locks: Sequence[RunDataLock],
    *,
    apply: bool = False,
) -> dict[str, Any]:
    """Remove only complete cache objects not pinned by supplied run locks."""

    pinned = pinned_cache_paths(cache_dir, locks)
    candidates = []
    objects_root = cache_dir / "objects"
    if objects_root.exists():
        for path in objects_root.glob("*/*/*"):
            if path.is_file() and not path.name.startswith(".") and path not in pinned:
                candidates.append(path)
    candidates.sort(key=lambda path: (path.stat().st_mtime_ns, str(path)))
    bytes_reclaimable = sum(path.stat().st_size for path in candidates)
    if apply:
        for path in candidates:
            path.unlink()
        for directory in sorted(objects_root.glob("*/*"), reverse=True):
            try:
                directory.rmdir()
            except OSError:
                pass
    return {
        "status": "evicted" if apply else "dry-run",
        "object_count": len(candidates),
        "bytes_reclaimable": bytes_reclaimable,
        "objects": [str(path) for path in candidates],
    }


def _emit(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _command_resolve(args: argparse.Namespace) -> None:
    lock = resolve_data_lock(
        load_source_manifest(args.source),
        load_protocol_manifest(args.protocol),
        requested_roles=tuple(args.roles),
        purpose=args.purpose,
        measurement_contract_id=args.measurement_contract_id,
    )
    write_data_lock(args.output, lock)
    _emit({"lock": str(args.output), "lock_sha256": lock.lock_sha256})


def _command_plan(args: argparse.Namespace) -> None:
    lock = load_data_lock(args.lock)
    _emit(
        plan_staging(
            _objects(lock),
            args.cache,
            allowed_roles=_roles(lock),
            allow_test=_allow_test(lock),
            reserve_bytes=args.reserve_bytes,
        )
    )


def _command_stage(args: argparse.Namespace) -> None:
    lock = load_data_lock(args.lock)
    report = stage_objects(
        _objects(lock),
        args.cache,
        run_dir=args.run_dir,
        allowed_roles=_roles(lock),
        allow_test=_allow_test(lock),
        reserve_bytes=args.reserve_bytes,
        report_path=args.report,
    )
    report["lock_sha256"] = lock.lock_sha256
    if args.report is not None:
        # stage_objects writes the transfer report before this lock-specific
        # field is attached. Replace it atomically so persisted evidence and
        # stdout carry the same immutable run identity.
        report_path = Path(args.report)
        temporary = report_path.with_name(f".{report_path.name}.tmp")
        temporary.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(report_path)
    _emit(report)


def _command_verify(args: argparse.Namespace) -> None:
    _emit(verify_lock_cache(load_data_lock(args.lock), Path(args.cache)))


def _command_fit_stats(args: argparse.Namespace) -> None:
    lock = load_data_lock(args.lock)
    if lock.purpose != "training" or "train" not in lock.requested_roles:
        raise ValueError("Normalization statistics require a training lock containing train data")
    dataset = PDEBenchDataset(
        PDEBenchConfig(
            task=args.task,
            split="train",
            root=args.root,
            normalize=False,
            data_lock_path=str(args.lock),
            max_samples=args.max_samples,
        )
    )
    component = str(args.component)
    stats = fit_normalization_stats(
        (dataset[index][component] for index in range(len(dataset))),
        channel_axis=args.channel_axis,
        data_lock_sha256=lock.lock_sha256,
        selection_sha256=canonical_sha256(lock.selection),
    )
    stats.save(Path(args.output))
    _emit(
        {
            "normalization": str(args.output),
            "component": component,
            "sha256": stats.sha256,
            "count": stats.count,
        }
    )


def _command_evict(args: argparse.Namespace) -> None:
    locks = [load_data_lock(path) for path in args.lock]
    _emit(evict_unpinned_cache(Path(args.cache), locks, apply=args.apply))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ups-data")
    commands = parser.add_subparsers(dest="command", required=True)

    resolve = commands.add_parser("resolve", help="resolve manifests into an immutable run lock")
    resolve.add_argument("--source", required=True, type=Path)
    resolve.add_argument("--protocol", required=True, type=Path)
    resolve.add_argument("--output", required=True, type=Path)
    resolve.add_argument("--roles", nargs="+", default=["train", "valid"])
    resolve.add_argument("--purpose", choices=("training", "measurement"), default="training")
    resolve.add_argument("--measurement-contract-id")
    resolve.set_defaults(handler=_command_resolve)

    for name, handler in (("plan", _command_plan), ("stage", _command_stage)):
        command = commands.add_parser(name)
        command.add_argument("--lock", required=True, type=Path)
        command.add_argument("--cache", required=True, type=Path)
        command.add_argument("--reserve-bytes", type=int, default=0)
        if name == "stage":
            command.add_argument("--run-dir", required=True, type=Path)
            command.add_argument("--report", type=Path)
        command.set_defaults(handler=handler)

    verify = commands.add_parser("verify")
    verify.add_argument("--lock", required=True, type=Path)
    verify.add_argument("--cache", required=True, type=Path)
    verify.set_defaults(handler=_command_verify)

    stats = commands.add_parser("fit-stats")
    stats.add_argument("--lock", required=True, type=Path)
    stats.add_argument("--root", required=True, type=Path)
    stats.add_argument("--task", required=True)
    stats.add_argument("--output", required=True, type=Path)
    stats.add_argument("--component", choices=("fields", "targets"), default="fields")
    stats.add_argument("--channel-axis", type=int, default=-1)
    stats.add_argument("--max-samples", type=int)
    stats.set_defaults(handler=_command_fit_stats)

    evict = commands.add_parser("evict")
    evict.add_argument("--cache", required=True, type=Path)
    evict.add_argument("--lock", action="append", default=[], type=Path)
    evict.add_argument("--apply", action="store_true", help="apply the default dry-run plan")
    evict.set_defaults(handler=_command_evict)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.handler(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
