"""Immutable dataset manifests and deterministic run data locks.

This module is the data control plane.  It intentionally does not download or
open dataset bytes; transports and loaders consume the exact objects resolved
here.  The types are dependency-light so manifests can be validated before a
training environment is provisioned.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

DataRole = Literal["train", "valid", "test"]
AccessPurpose = Literal["training", "measurement"]

_ROLES = frozenset({"train", "valid", "test"})
_MUTABLE_REVISIONS = frozenset({"head", "latest", "main", "master", "stable", "current"})
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/+-]*$")
_CHECKSUM_LENGTHS = {"md5": 32, "sha1": 40, "sha256": 64, "sha512": 128}


class ManifestError(ValueError):
    """Raised when a data manifest or resolution request is unsafe or invalid."""


def _plain(value: Any) -> Any:
    if is_dataclass(value):
        value = asdict(value)  # type: ignore[call-overload]
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def canonical_json_bytes(value: Any) -> bytes:
    """Return the one canonical JSON encoding used for manifest identities."""

    try:
        return json.dumps(
            _plain(value),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ManifestError(f"value is not canonically serializable: {exc}") from exc


def canonical_sha256(value: Any) -> str:
    """Hash a value after canonical JSON serialization."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ManifestError(f"{name} must be a mapping")
    return value


def _required_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{name} must be a non-empty string")
    return value.strip()


def _identifier(value: Any, name: str) -> str:
    result = _required_string(value, name)
    if not _ID_RE.fullmatch(result):
        raise ManifestError(f"{name} contains unsupported characters: {result!r}")
    return result


def _revision(value: Any, name: str) -> str:
    result = _required_string(value, name)
    revision_tokens = set(re.split(r"[/:\s]+", result.lower()))
    if revision_tokens & _MUTABLE_REVISIONS:
        raise ManifestError(f"{name} must be immutable, not {result!r}")
    return result


def _required_digest(value: Any, name: str, algorithm: str) -> str:
    digest = _required_string(value, name).lower()
    expected_length = _CHECKSUM_LENGTHS[algorithm]
    if len(digest) != expected_length or not re.fullmatch(r"[0-9a-f]+", digest):
        raise ManifestError(f"{name} must be a valid {algorithm} digest")
    return digest


def _role(value: Any, name: str) -> DataRole:
    result = _required_string(value, name)
    if result not in _ROLES:
        raise ManifestError(f"{name} must be one of {sorted(_ROLES)}, got {result!r}")
    return result  # type: ignore[return-value]


def _string_tuple(value: Any, name: str, *, nonempty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ManifestError(f"{name} must be a list of strings")
    result = tuple(_required_string(item, f"{name}[]") for item in value)
    if nonempty and not result:
        raise ManifestError(f"{name} must not be empty")
    if len(result) != len(set(result)):
        raise ManifestError(f"{name} contains duplicates")
    return result


def _freeform_mapping(value: Any, name: str) -> dict[str, Any]:
    result = dict(_require_mapping(value, name))
    # Validate now, rather than discovering unsupported values while hashing a run.
    canonical_json_bytes(result)
    return result


@dataclass(frozen=True)
class SourceObject:
    """One independently addressable immutable source object."""

    object_id: str
    path: str
    size_bytes: int
    checksums: Mapping[str, str]
    uris: tuple[str, ...]
    declared_roles: tuple[DataRole, ...]
    media_type: str = "application/x-hdf5"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> SourceObject:
        raw = _require_mapping(raw, "source object")
        object_id = _identifier(raw.get("object_id"), "source object.object_id")
        path = _required_string(raw.get("path"), f"source object {object_id}.path")
        if path.startswith("/") or ".." in Path(path).parts:
            raise ManifestError(f"source object {object_id}.path must be a safe relative path")

        size_bytes = raw.get("size_bytes")
        if not isinstance(size_bytes, int) or isinstance(size_bytes, bool) or size_bytes <= 0:
            raise ManifestError(f"source object {object_id}.size_bytes must be a positive integer")

        checksums_raw = _require_mapping(
            raw.get("checksums"), f"source object {object_id}.checksums"
        )
        if not checksums_raw:
            raise ManifestError(f"source object {object_id}.checksums must not be empty")
        checksums: dict[str, str] = {}
        for algorithm, digest in checksums_raw.items():
            algorithm = _required_string(algorithm, "checksum algorithm").lower()
            if algorithm not in _CHECKSUM_LENGTHS:
                raise ManifestError(f"unsupported checksum algorithm {algorithm!r}")
            digest = _required_string(digest, f"checksum {algorithm}").lower()
            if len(digest) != _CHECKSUM_LENGTHS[algorithm] or not re.fullmatch(
                r"[0-9a-f]+", digest
            ):
                raise ManifestError(f"invalid {algorithm} checksum for source object {object_id}")
            checksums[algorithm] = digest

        uris = _string_tuple(raw.get("uris", []), f"source object {object_id}.uris", nonempty=True)
        roles_raw = _string_tuple(
            raw.get("declared_roles", []),
            f"source object {object_id}.declared_roles",
            nonempty=True,
        )
        roles = tuple(
            _role(role, f"source object {object_id}.declared_roles") for role in roles_raw
        )
        return cls(
            object_id=object_id,
            path=path,
            size_bytes=size_bytes,
            checksums=checksums,
            uris=uris,
            declared_roles=roles,
            media_type=_required_string(
                raw.get("media_type", "application/x-hdf5"),
                f"source object {object_id}.media_type",
            ),
            metadata=_freeform_mapping(
                raw.get("metadata", {}), f"source object {object_id}.metadata"
            ),
        )


@dataclass(frozen=True)
class SourceManifest:
    """Immutable upstream identity and byte inventory for one dataset revision."""

    schema_version: int
    dataset_id: str
    provider: str
    revision: str
    native_format: str
    license: str
    citation: str
    objects: tuple[SourceObject, ...]
    metadata_only: bool = False
    inventory_reference: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> SourceManifest:
        raw = _require_mapping(raw, "source manifest")
        version = raw.get("schema_version")
        if version != 1:
            raise ManifestError(f"source manifest.schema_version must be 1, got {version!r}")
        dataset_id = _identifier(raw.get("dataset_id"), "source manifest.dataset_id")
        objects_raw = raw.get("objects", [])
        if not isinstance(objects_raw, list):
            raise ManifestError("source manifest.objects must be a list")
        objects = tuple(
            sorted(
                (SourceObject.from_dict(item) for item in objects_raw),
                key=lambda item: item.object_id,
            )
        )
        metadata_only = raw.get("metadata_only", False)
        if not isinstance(metadata_only, bool):
            raise ManifestError("source manifest.metadata_only must be boolean")
        inventory_reference_raw = raw.get("inventory_reference")
        inventory_reference = (
            _required_string(inventory_reference_raw, "source manifest.inventory_reference")
            if inventory_reference_raw is not None
            else None
        )
        if not objects and not metadata_only:
            raise ManifestError(
                "source manifest.objects must not be empty unless metadata_only is true"
            )
        if metadata_only and not inventory_reference:
            raise ManifestError("metadata-only source manifest requires inventory_reference")

        object_ids = [item.object_id for item in objects]
        paths = [item.path for item in objects]
        if len(object_ids) != len(set(object_ids)):
            raise ManifestError("source manifest contains duplicate object_id values")
        if len(paths) != len(set(paths)):
            raise ManifestError("source manifest contains duplicate object paths")

        return cls(
            schema_version=version,
            dataset_id=dataset_id,
            provider=_required_string(raw.get("provider"), "source manifest.provider"),
            revision=_revision(raw.get("revision"), "source manifest.revision"),
            native_format=_required_string(
                raw.get("native_format"), "source manifest.native_format"
            ),
            license=_required_string(raw.get("license"), "source manifest.license"),
            citation=_required_string(raw.get("citation"), "source manifest.citation"),
            objects=objects,
            metadata_only=metadata_only,
            inventory_reference=inventory_reference,
            metadata=_freeform_mapping(raw.get("metadata", {}), "source manifest.metadata"),
        )

    @property
    def manifest_sha256(self) -> str:
        return canonical_sha256(self)


@dataclass(frozen=True)
class ProtocolManifest:
    """Scientific split, identity, selection, and normalization policy."""

    schema_version: int
    protocol_id: str
    dataset_id: str
    source_revision: str
    adapter: str
    adapter_revision: str
    split_authority: str
    splits: Mapping[DataRole, tuple[str, ...]]
    identity_fields: tuple[str, ...]
    selection: Mapping[str, Any]
    normalization: Mapping[str, Any]
    test_access: Literal["forbidden", "measurement_contract_required"]
    coverage_dimensions: tuple[str, ...] = ()
    metadata_only: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ProtocolManifest:
        raw = _require_mapping(raw, "protocol manifest")
        version = raw.get("schema_version")
        if version != 1:
            raise ManifestError(f"protocol manifest.schema_version must be 1, got {version!r}")
        splits_raw = _require_mapping(raw.get("splits", {}), "protocol manifest.splits")
        splits: dict[DataRole, tuple[str, ...]] = {}
        all_ids: list[str] = []
        for role_raw, ids_raw in splits_raw.items():
            role = _role(role_raw, "protocol manifest split role")
            ids = _string_tuple(ids_raw, f"protocol manifest.splits.{role}")
            splits[role] = tuple(
                sorted(_identifier(item, f"protocol split {role} object id") for item in ids)
            )
            all_ids.extend(splits[role])
        if len(all_ids) != len(set(all_ids)):
            raise ManifestError(
                "a source object cannot be assigned to more than one protocol split"
            )

        metadata_only = raw.get("metadata_only", False)
        if not isinstance(metadata_only, bool):
            raise ManifestError("protocol manifest.metadata_only must be boolean")
        if not splits and not metadata_only:
            raise ManifestError(
                "protocol manifest.splits must not be empty unless metadata_only is true"
            )

        test_access = raw.get("test_access", "measurement_contract_required")
        if test_access not in {"forbidden", "measurement_contract_required"}:
            raise ManifestError(
                "protocol manifest.test_access must be 'forbidden' or "
                "'measurement_contract_required'"
            )

        normalization = _freeform_mapping(
            raw.get("normalization", {}), "protocol manifest.normalization"
        )
        fit_role = normalization.get("fit_role")
        if fit_role is not None and fit_role != "train":
            raise ManifestError("normalization.fit_role must be 'train'")

        return cls(
            schema_version=version,
            protocol_id=_identifier(raw.get("protocol_id"), "protocol manifest.protocol_id"),
            dataset_id=_identifier(raw.get("dataset_id"), "protocol manifest.dataset_id"),
            source_revision=_revision(
                raw.get("source_revision"), "protocol manifest.source_revision"
            ),
            adapter=_identifier(raw.get("adapter"), "protocol manifest.adapter"),
            adapter_revision=_revision(
                raw.get("adapter_revision"), "protocol manifest.adapter_revision"
            ),
            split_authority=_required_string(
                raw.get("split_authority"), "protocol manifest.split_authority"
            ),
            splits=splits,
            identity_fields=_string_tuple(
                raw.get("identity_fields", []),
                "protocol manifest.identity_fields",
                nonempty=True,
            ),
            selection=_freeform_mapping(raw.get("selection", {}), "protocol manifest.selection"),
            normalization=normalization,
            test_access=test_access,
            coverage_dimensions=_string_tuple(
                raw.get("coverage_dimensions", []), "protocol manifest.coverage_dimensions"
            ),
            metadata_only=metadata_only,
            metadata=_freeform_mapping(raw.get("metadata", {}), "protocol manifest.metadata"),
        )

    @property
    def manifest_sha256(self) -> str:
        return canonical_sha256(self)


@dataclass(frozen=True)
class LockedObject:
    """Exact source object and role embedded in a run lock."""

    object_id: str
    role: DataRole
    path: str
    size_bytes: int
    checksums: Mapping[str, str]
    uris: tuple[str, ...]
    media_type: str

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> LockedObject:
        raw = _require_mapping(raw, "locked object")
        object_id = _identifier(raw.get("object_id"), "locked object.object_id")
        checksums = _require_mapping(raw.get("checksums"), f"locked object {object_id}.checksums")
        # Reuse SourceObject's checksum, URI, path, and size validation.
        source = SourceObject.from_dict(
            {
                "object_id": object_id,
                "path": raw.get("path"),
                "size_bytes": raw.get("size_bytes"),
                "checksums": checksums,
                "uris": raw.get("uris"),
                "declared_roles": [raw.get("role")],
                "media_type": raw.get("media_type", "application/x-hdf5"),
            }
        )
        return cls(
            object_id=source.object_id,
            role=_role(raw.get("role"), f"locked object {object_id}.role"),
            path=source.path,
            size_bytes=source.size_bytes,
            checksums=source.checksums,
            uris=source.uris,
            media_type=source.media_type,
        )


@dataclass(frozen=True)
class RunDataLock:
    """Deterministic, self-verifying resolution of bytes permitted for one run."""

    schema_version: int
    dataset_id: str
    source_revision: str
    source_manifest_sha256: str
    protocol_id: str
    protocol_manifest_sha256: str
    adapter: str
    adapter_revision: str
    purpose: AccessPurpose
    requested_roles: tuple[DataRole, ...]
    measurement_contract_id: str | None
    objects: tuple[LockedObject, ...]
    selection: Mapping[str, Any]
    normalization: Mapping[str, Any]
    lock_sha256: str

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> RunDataLock:
        raw = _require_mapping(raw, "run data lock")
        if raw.get("schema_version") != 1:
            raise ManifestError("run data lock.schema_version must be 1")
        purpose = raw.get("purpose")
        if purpose not in {"training", "measurement"}:
            raise ManifestError(f"unsupported run data lock purpose {purpose!r}")
        role_values = _string_tuple(raw.get("requested_roles"), "requested_roles", nonempty=True)
        roles = tuple(sorted(_role(item, "requested role") for item in role_values))
        objects_raw = raw.get("objects")
        if not isinstance(objects_raw, list):
            raise ManifestError("run data lock.objects must be a list")
        objects = tuple(
            sorted(
                (LockedObject.from_dict(item) for item in objects_raw),
                key=lambda item: (item.role, item.object_id),
            )
        )
        if any(item.role not in roles for item in objects):
            raise ManifestError("run data lock contains an object outside requested_roles")
        if purpose == "training" and (
            "test" in roles or any(item.role == "test" for item in objects)
        ):
            raise ManifestError("training run locks cannot contain test-role objects")
        measurement_contract_raw = raw.get("measurement_contract_id")
        measurement_contract_id = (
            _identifier(measurement_contract_raw, "measurement_contract_id")
            if measurement_contract_raw is not None
            else None
        )
        if purpose == "measurement" and ("test" not in roles or not measurement_contract_id):
            raise ManifestError("measurement lock requires test role and measurement_contract_id")
        lock = cls(
            schema_version=1,
            dataset_id=_identifier(raw.get("dataset_id"), "run data lock.dataset_id"),
            source_revision=_revision(raw.get("source_revision"), "run data lock.source_revision"),
            source_manifest_sha256=_required_digest(
                raw.get("source_manifest_sha256"), "source_manifest_sha256", "sha256"
            ),
            protocol_id=_identifier(raw.get("protocol_id"), "run data lock.protocol_id"),
            protocol_manifest_sha256=_required_digest(
                raw.get("protocol_manifest_sha256"), "protocol_manifest_sha256", "sha256"
            ),
            adapter=_identifier(raw.get("adapter"), "run data lock.adapter"),
            adapter_revision=_revision(raw.get("adapter_revision"), "adapter_revision"),
            purpose=purpose,
            requested_roles=roles,
            measurement_contract_id=measurement_contract_id,
            objects=objects,
            selection=_freeform_mapping(raw.get("selection", {}), "run data lock.selection"),
            normalization=_freeform_mapping(
                raw.get("normalization", {}), "run data lock.normalization"
            ),
            lock_sha256=_required_digest(raw.get("lock_sha256"), "lock_sha256", "sha256"),
        )
        lock.verify()
        return lock

    def payload(self) -> dict[str, Any]:
        """Return the hash-covered payload (everything except the digest itself)."""

        result = asdict(self)
        result.pop("lock_sha256")
        return result

    def verify(self) -> None:
        actual = canonical_sha256(self.payload())
        if actual != self.lock_sha256:
            raise ManifestError(
                f"run data lock digest mismatch: recorded {self.lock_sha256}, computed {actual}"
            )

    def to_dict(self) -> dict[str, Any]:
        return _plain(self)


def resolve_data_lock(
    source: SourceManifest,
    protocol: ProtocolManifest,
    *,
    requested_roles: Sequence[DataRole] = ("train", "valid"),
    purpose: AccessPurpose = "training",
    measurement_contract_id: str | None = None,
) -> RunDataLock:
    """Resolve a deterministic run lock, enforcing split and test-byte boundaries."""

    if source.metadata_only or protocol.metadata_only:
        raise ManifestError("metadata-only manifests cannot be resolved into a run data lock")
    if source.dataset_id != protocol.dataset_id:
        raise ManifestError("source and protocol dataset_id values do not match")
    if source.revision != protocol.source_revision:
        raise ManifestError("source revision does not match protocol source_revision")
    if purpose not in {"training", "measurement"}:
        raise ManifestError(f"unsupported access purpose {purpose!r}")

    roles_raw = _string_tuple(requested_roles, "requested_roles", nonempty=True)
    roles = tuple(_role(role, "requested role") for role in roles_raw)
    includes_test = "test" in roles
    if purpose == "training" and includes_test:
        raise ManifestError("training run locks cannot contain test-role objects")
    if purpose == "measurement":
        if not includes_test:
            raise ManifestError("measurement run locks must explicitly request the test role")
        if protocol.test_access == "forbidden":
            raise ManifestError(f"protocol {protocol.protocol_id} forbids test access")
        if not measurement_contract_id:
            raise ManifestError("test access requires an explicit measurement_contract_id")
        measurement_contract_id = _identifier(measurement_contract_id, "measurement_contract_id")
    elif measurement_contract_id is not None:
        raise ManifestError("measurement_contract_id is only valid for measurement locks")

    source_by_id = {item.object_id: item for item in source.objects}
    locked: list[LockedObject] = []
    for role in roles:
        if role not in protocol.splits:
            raise ManifestError(f"requested role {role!r} is not defined by the protocol")
        for object_id in protocol.splits[role]:
            item = source_by_id.get(object_id)
            if item is None:
                raise ManifestError(
                    f"protocol split {role!r} references unknown source object {object_id!r}"
                )
            if role not in item.declared_roles:
                raise ManifestError(
                    f"source object {object_id!r} is not declared for protocol role {role!r}"
                )
            locked.append(
                LockedObject(
                    object_id=item.object_id,
                    role=role,
                    path=item.path,
                    size_bytes=item.size_bytes,
                    checksums=dict(item.checksums),
                    uris=item.uris,
                    media_type=item.media_type,
                )
            )

    # Object order in source files and split lists must not affect a lock identity.
    locked.sort(key=lambda item: (item.role, item.object_id))
    payload = {
        "schema_version": 1,
        "dataset_id": source.dataset_id,
        "source_revision": source.revision,
        "source_manifest_sha256": source.manifest_sha256,
        "protocol_id": protocol.protocol_id,
        "protocol_manifest_sha256": protocol.manifest_sha256,
        "adapter": protocol.adapter,
        "adapter_revision": protocol.adapter_revision,
        "purpose": purpose,
        "requested_roles": sorted(roles),
        "measurement_contract_id": measurement_contract_id,
        "objects": locked,
        "selection": dict(protocol.selection),
        "normalization": dict(protocol.normalization),
    }
    digest = canonical_sha256(payload)
    lock = RunDataLock(
        schema_version=1,
        dataset_id=source.dataset_id,
        source_revision=source.revision,
        source_manifest_sha256=source.manifest_sha256,
        protocol_id=protocol.protocol_id,
        protocol_manifest_sha256=protocol.manifest_sha256,
        adapter=protocol.adapter,
        adapter_revision=protocol.adapter_revision,
        purpose=purpose,
        requested_roles=tuple(sorted(roles)),
        measurement_contract_id=measurement_contract_id,
        objects=tuple(locked),
        selection=dict(protocol.selection),
        normalization=dict(protocol.normalization),
        lock_sha256=digest,
    )
    lock.verify()
    return lock


def _load_yaml(path: str | Path) -> Mapping[str, Any]:
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ManifestError(f"could not load manifest {path}: {exc}") from exc
    return _require_mapping(raw, f"manifest {path}")


def load_source_manifest(path: str | Path) -> SourceManifest:
    return SourceManifest.from_dict(_load_yaml(path))


def load_protocol_manifest(path: str | Path) -> ProtocolManifest:
    return ProtocolManifest.from_dict(_load_yaml(path))


def write_data_lock(path: str | Path, lock: RunDataLock) -> None:
    """Atomically write a byte-deterministic JSON run lock."""

    lock.verify()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(canonical_json_bytes(lock) + b"\n")
    temporary.replace(path)


def load_data_lock(path: str | Path) -> RunDataLock:
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"could not load run data lock {path}: {exc}") from exc
    return RunDataLock.from_dict(_require_mapping(raw, f"run data lock {path}"))
