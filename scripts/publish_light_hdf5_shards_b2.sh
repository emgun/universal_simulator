#!/usr/bin/env bash
set -euo pipefail

# Publish already-built, protocol-gated PDEBench HDF5 shards to Backblaze B2.
#
# Safe dry run over already-built shards:
#   DRY_RUN=1 OUT_ROOT=data/pdebench_light VERSION=light-v1 bash scripts/publish_light_hdf5_shards_b2.sh
#
# Construction is intentionally separate. This script refuses manifests without
# passing protocol gates.

read_env_key() {
  local file="$1"; shift
  local key="$1"; shift || true
  if [ ! -f "$file" ]; then
    return 1
  fi
  local line
  while IFS= read -r line; do
    line="${line#${line%%[![:space:]]*}}"
    [ -z "$line" ] && continue
    [ "${line:0:1}" = "#" ] && continue
    if [[ "$line" =~ ^[[:space:]]*$key[[:space:]]*[:=][[:space:]]*(.*)$ ]]; then
      local val="${BASH_REMATCH[1]}"
      if [[ "$val" =~ ^"(.*)"$ ]]; then
        echo "${BASH_REMATCH[1]}"
      elif [[ "$val" =~ ^'(.*)'$ ]]; then
        echo "${BASH_REMATCH[1]}"
      else
        echo "$val"
      fi
      return 0
    fi
  done < "$file"
  return 1
}

load_optional_env() {
  local env_file="$1"
  [ -f "$env_file" ] || return 0
  : "${B2_KEY_ID:=$(read_env_key "$env_file" B2_KEY_ID || read_env_key "$env_file" B2_ACCOUNT_ID || true)}"
  : "${B2_APP_KEY:=$(read_env_key "$env_file" B2_APP_KEY || read_env_key "$env_file" B2_APPLICATION_KEY || true)}"
  : "${B2_BUCKET:=$(read_env_key "$env_file" B2_BUCKET || read_env_key "$env_file" B2_BUCKET_NAME || true)}"
  : "${B2_S3_ENDPOINT:=$(read_env_key "$env_file" B2_S3_ENDPOINT || true)}"
  : "${B2_S3_REGION:=$(read_env_key "$env_file" B2_S3_REGION || true)}"
  export B2_KEY_ID B2_APP_KEY B2_BUCKET B2_S3_ENDPOINT B2_S3_REGION
}

configure_rclone() {
  if [ "$DRY_RUN" -eq 1 ]; then
    : "${B2_BUCKET:=example-bucket}"
  else
    : "${B2_KEY_ID:?Set B2_KEY_ID in ENV_FILE or environment}"
    : "${B2_APP_KEY:?Set B2_APP_KEY in ENV_FILE or environment}"
    : "${B2_BUCKET:?Set B2_BUCKET in ENV_FILE or environment}"
    if ! command -v rclone >/dev/null 2>&1; then
      echo "rclone is required for B2 publishing. Install rclone or run with DRY_RUN=1." >&2
      exit 1
    fi
  fi

  if [ -n "${B2_S3_ENDPOINT:-}" ] || [ -n "${B2_S3_REGION:-}" ]; then
    export RCLONE_CONFIG_UPSB2_TYPE=s3
    export RCLONE_CONFIG_UPSB2_PROVIDER=Other
    export RCLONE_CONFIG_UPSB2_ACCESS_KEY_ID="${B2_KEY_ID:-dry-run-key-id}"
    export RCLONE_CONFIG_UPSB2_SECRET_ACCESS_KEY="${B2_APP_KEY:-dry-run-app-key}"
    [ -n "${B2_S3_ENDPOINT:-}" ] && export RCLONE_CONFIG_UPSB2_ENDPOINT="${B2_S3_ENDPOINT}"
    [ -n "${B2_S3_REGION:-}" ] && export RCLONE_CONFIG_UPSB2_REGION="${B2_S3_REGION}"
  else
    export RCLONE_CONFIG_UPSB2_TYPE=b2
    export RCLONE_CONFIG_UPSB2_ACCOUNT="${B2_KEY_ID:-dry-run-key-id}"
    export RCLONE_CONFIG_UPSB2_KEY="${B2_APP_KEY:-dry-run-app-key}"
  fi
}

publish_immutable_file() {
  local local_path="$1"
  local remote_key="$2"
  local expected_sha256="$3"
  local remote="UPSB2:${B2_BUCKET}/${remote_key}"
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "DRY_RUN: rclone copyto ${local_path} ${remote}"
    return
  fi

  # Content-addressed keys must never be overwritten.  If the key already
  # exists, prove that it contains the named bytes and treat publication as
  # idempotent; a divergent object is a hard failure.
  local parent="${remote%/*}"
  local name="${remote##*/}"
  if rclone lsf "$parent" --files-only --include "$name" 2>/dev/null | grep -Fxq "$name"; then
    local observed_sha256
    observed_sha256="$(rclone cat "$remote" | sha256sum | awk '{print $1}')"
    if [ "$observed_sha256" != "$expected_sha256" ]; then
      echo "Refusing to overwrite divergent immutable object: ${remote}" >&2
      exit 1
    fi
    echo "Verified existing immutable object: ${remote}"
    return
  fi
  rclone copyto "$local_path" "$remote"
}

ENV_FILE=${ENV_FILE:-.env}
VERSION=${VERSION:-strat-v1}
REMOTE_PREFIX=${REMOTE_PREFIX:-$VERSION}
OUT_ROOT=${OUT_ROOT:-data/pdebench_strat_v1}
MANIFEST=${MANIFEST:-docs/strat_v1_manifest.yaml}
SOURCE_MANIFEST=${SOURCE_MANIFEST:-${MANIFEST%.yaml}.source.yaml}
PROTOCOL_MANIFEST=${PROTOCOL_MANIFEST:-${MANIFEST%.yaml}.protocol.yaml}
TRAINING_LOCK=${TRAINING_LOCK:-}
MEASUREMENT_LOCK=${MEASUREMENT_LOCK:-}
CANONICAL_SOURCE=${CANONICAL_SOURCE:-}
HYDRATION_RECORD=${HYDRATION_RECORD:-}
DRY_RUN=${DRY_RUN:-1}

load_optional_env "$ENV_FILE"
configure_rclone

for required in "$MANIFEST" "$SOURCE_MANIFEST" "$PROTOCOL_MANIFEST"; do
  if [ ! -f "$required" ]; then
    echo "Required publication manifest not found: ${required}" >&2
    exit 1
  fi
done
if { [ -n "$CANONICAL_SOURCE" ] && [ -z "$HYDRATION_RECORD" ]; } || \
   { [ -z "$CANONICAL_SOURCE" ] && [ -n "$HYDRATION_RECORD" ]; }; then
  echo "CANONICAL_SOURCE and HYDRATION_RECORD must be supplied together." >&2
  exit 1
fi
for optional in "$TRAINING_LOCK" "$MEASUREMENT_LOCK" "$CANONICAL_SOURCE" "$HYDRATION_RECORD"; do
  if [ -n "$optional" ] && [ ! -f "$optional" ]; then
    echo "Optional publication input not found: ${optional}" >&2
    exit 1
  fi
done

publication_plan="$(PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}src" python - "$MANIFEST" "$SOURCE_MANIFEST" "$PROTOCOL_MANIFEST" \
  "$OUT_ROOT" "$REMOTE_PREFIX" "$B2_BUCKET" "$TRAINING_LOCK" "$MEASUREMENT_LOCK" \
  "$CANONICAL_SOURCE" "$HYDRATION_RECORD" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
import yaml

from ups.data.manifests import ProtocolManifest, SourceManifest

path = Path(sys.argv[1])
source_path = Path(sys.argv[2])
protocol_path = Path(sys.argv[3])
out_root = Path(sys.argv[4])
remote_prefix = sys.argv[5].rstrip("/")
bucket = sys.argv[6]
training_lock_path = Path(sys.argv[7]) if sys.argv[7] else None
measurement_lock_path = Path(sys.argv[8]) if sys.argv[8] else None
canonical_path = Path(sys.argv[9]) if sys.argv[9] else None
hydration_path = Path(sys.argv[10]) if sys.argv[10] else None

def canonical_sha256(value):
    encoded = json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

def file_sha256(candidate):
    hasher = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

if remote_prefix in {"smoke-v1", "light-v1", "medium-v1"}:
    raise SystemExit(f"Refusing to publish to frozen legacy prefix: {remote_prefix}")
payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
if payload.get("protocol_mode") != "strat-v1":
    raise SystemExit(f"Refusing to publish non-strat-v1 manifest: {path}")
if payload.get("version") in {"smoke-v1", "light-v1", "medium-v1"}:
    raise SystemExit(f"Refusing to publish reserved legacy version: {payload.get('version')}")
if payload.get("remote_prefix") != remote_prefix:
    raise SystemExit("Manifest remote_prefix does not match requested publish prefix")
gates = payload.get("protocol_gates")
if not isinstance(gates, dict) or not gates:
    raise SystemExit(f"Refusing to publish manifest without protocol gates: {path}")
failed = sorted(task for task, gate in gates.items() if not isinstance(gate, dict) or gate.get("status") != "passed")
if failed:
    raise SystemExit(f"Refusing to publish failed protocol gates for: {', '.join(failed)}")
tasks = payload.get("tasks")
if not isinstance(tasks, list) or set(tasks) != set(gates) or len(tasks) != len(gates):
    raise SystemExit("Manifest tasks and protocol_gates do not agree")
if len(tasks) != 1:
    raise SystemExit("Construction manifest must contain exactly one task")
task = tasks[0]
records = payload.get("records")
if not isinstance(records, list) or not records:
    raise SystemExit(f"Refusing to publish manifest without artifact records: {path}")
by_name = {}
splits_by_task = {task: set() for task in gates}
for record in records:
    if not isinstance(record, dict):
        raise SystemExit("Manifest artifact records must be mappings")
    name = Path(str(record.get("output_path", ""))).name
    if not name or name in by_name:
        raise SystemExit(f"Manifest contains an invalid or duplicate output name: {name!r}")
    if record.get("split") not in {"train", "val", "test"}:
        raise SystemExit(f"Manifest record has invalid split: {record.get('split')!r}")
    if record.get("task") not in gates:
        raise SystemExit(f"Manifest record task lacks a passed gate: {record.get('task')!r}")
    expected_name = f"{record['task']}_{record['split']}.h5"
    if name != expected_name:
        raise SystemExit(f"Manifest record filename is not canonical: {name!r} != {expected_name!r}")
    splits_by_task[record["task"]].add(record["split"])
    gate = record.get("protocol_gate")
    if gate != gates[record["task"]]:
        raise SystemExit(f"Manifest record gate differs from its task gate: {name}")
    by_name[name] = record
incomplete = sorted(task for task, splits in splits_by_task.items() if splits != {"train", "val", "test"})
if incomplete:
    raise SystemExit(f"Manifest does not contain exactly train/val/test for: {', '.join(incomplete)}")
if {record["task"] for record in records} != set(tasks):
    raise SystemExit("Manifest tasks and artifact records do not agree")
files = {candidate.name: candidate for candidate in out_root.glob("*.h5")}
if set(files) != set(by_name):
    missing = sorted(set(by_name) - set(files))
    extra = sorted(set(files) - set(by_name))
    raise SystemExit(f"Manifest/artifact set mismatch; missing={missing}, extra={extra}")
for name, candidate in files.items():
    record = by_name[name]
    digest = file_sha256(candidate)
    if candidate.stat().st_size != record.get("bytes") or digest != record.get("sha256"):
        raise SystemExit(f"Artifact bytes/hash do not match manifest record: {name}")

construction_digest = canonical_sha256(payload)
source = yaml.safe_load(source_path.read_text(encoding="utf-8")) or {}
protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8")) or {}
release_id = str(source.get("revision", "")).removeprefix("sha256:")
if release_id != construction_digest:
    raise SystemExit("Source revision does not match construction manifest digest")
if source.get("metadata", {}).get("construction_manifest_sha256") != construction_digest:
    raise SystemExit("Source manifest does not match construction manifest")
if protocol.get("metadata", {}).get("construction_manifest_sha256") != construction_digest:
    raise SystemExit("Protocol manifest does not match construction manifest")
if protocol.get("source_revision") != source.get("revision"):
    raise SystemExit("Protocol source_revision does not match source manifest")
if source.get("dataset_id") != protocol.get("dataset_id"):
    raise SystemExit("Source and protocol dataset_id values do not match")
if protocol.get("selection") != payload.get("selection"):
    raise SystemExit("Protocol selection does not match construction manifest")

source_objects = source.get("objects")
if not isinstance(source_objects, list) or len(source_objects) != 3:
    raise SystemExit("Source manifest must contain exactly the three task shard objects")
source_by_path = {str(item.get("path")): item for item in source_objects if isinstance(item, dict)}
if len(source_by_path) != len(source_objects):
    raise SystemExit("Source manifest contains invalid or duplicate object paths")
source_by_id = {str(item.get("object_id")): item for item in source_objects}
expected_protocol_splits = {"train": [], "valid": [], "test": []}
object_plan = []
control_plan = []
for name, record in by_name.items():
    item = source_by_path.get(name)
    if item is None:
        raise SystemExit(f"Source manifest lacks construction record: {name}")
    split_role = "valid" if record["split"] == "val" else record["split"]
    digest = record["sha256"]
    object_key = f"{remote_prefix}/immutable/sha256/{digest}/{name}"
    exact_uri = f"b2://{bucket}/{object_key}"
    if record.get("remote_key") != object_key:
        raise SystemExit(f"Construction record remote_key is not the immutable publish key: {name}")
    if item.get("size_bytes") != record["bytes"] or item.get("checksums", {}).get("sha256") != digest:
        raise SystemExit(f"Source object bytes/hash differ from construction record: {name}")
    uris = item.get("uris")
    if not isinstance(uris, list) or not uris or uris[0] != exact_uri:
        raise SystemExit(f"Source object first b2:// URI is not the immutable publish URI: {name}")
    if item.get("declared_roles") != [split_role]:
        raise SystemExit(f"Source object role differs from construction record: {name}")
    expected_protocol_splits[split_role].append(item.get("object_id"))
    object_plan.append(f"SHARD\t{out_root / name}\t{object_key}\t{digest}")
if protocol.get("splits") != {role: sorted(ids) for role, ids in expected_protocol_splits.items()}:
    raise SystemExit("Protocol splits do not exactly match source objects")

# Locks hash the validated, normalized control-plane models rather than YAML
# presentation order.  Use the same authority here so a valid generated lock
# cannot be rejected merely because source objects were serialized train/val/test.
source_digest = SourceManifest.from_dict(source).manifest_sha256
protocol_digest = ProtocolManifest.from_dict(protocol).manifest_sha256
def validate_lock(lock_path, expected_purpose):
    if lock_path is None:
        return
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    expected_roles = ["train", "valid"] if expected_purpose == "training" else ["test"]
    if lock.get("schema_version") != 1 or lock.get("requested_roles") != expected_roles:
        raise SystemExit(f"{expected_purpose} lock roles/schema do not match its purpose")
    if lock.get("purpose") != expected_purpose:
        raise SystemExit(f"{expected_purpose} lock has the wrong purpose")
    if lock.get("source_manifest_sha256") != source_digest or lock.get("protocol_manifest_sha256") != protocol_digest:
        raise SystemExit(f"{expected_purpose} lock manifest hashes do not match")
    if lock.get("source_revision") != source.get("revision") or lock.get("protocol_id") != protocol.get("protocol_id"):
        raise SystemExit(f"{expected_purpose} lock identity does not match manifests")
    if lock.get("dataset_id") != source.get("dataset_id") or lock.get("adapter") != protocol.get("adapter") or lock.get("adapter_revision") != protocol.get("adapter_revision"):
        raise SystemExit(f"{expected_purpose} lock adapter identity does not match manifests")
    if lock.get("selection") != protocol.get("selection") or lock.get("normalization") != protocol.get("normalization"):
        raise SystemExit(f"{expected_purpose} lock policy does not match protocol")
    measurement_contract = lock.get("measurement_contract_id")
    if (expected_purpose == "training" and measurement_contract is not None) or (expected_purpose == "measurement" and not isinstance(measurement_contract, str)):
        raise SystemExit(f"{expected_purpose} lock measurement contract is invalid")
    lock_payload = dict(lock)
    recorded_lock_digest = lock_payload.pop("lock_sha256", None)
    if recorded_lock_digest != canonical_sha256(lock_payload):
        raise SystemExit(f"{expected_purpose} lock digest does not verify")
    requested = set(expected_roles)
    expected_objects = []
    for role in requested:
        for object_id in protocol["splits"].get(role, []):
            item = source_by_id[object_id]
            expected_objects.append({
                "object_id": object_id,
                "role": role,
                "path": item["path"],
                "size_bytes": item["size_bytes"],
                "checksums": item["checksums"],
                "uris": item["uris"],
                "media_type": item.get("media_type", "application/x-hdf5"),
            })
    key = lambda item: (item["role"], item["object_id"])
    if sorted(lock.get("objects", []), key=key) != sorted(expected_objects, key=key):
        raise SystemExit(f"{expected_purpose} lock objects do not match manifests")
    control_plan.append(f"LOCK\t{lock_path}\t{remote_prefix}/releases/{task}/{release_id}/{expected_purpose}.lock.json\t{file_sha256(lock_path)}")

validate_lock(training_lock_path, "training")
validate_lock(measurement_lock_path, "measurement")

release_root = f"{remote_prefix}/releases/{task}/{release_id}"
control_plan.append(f"CONTROL\t{path}\t{release_root}/construction.manifest.yaml\t{file_sha256(path)}")
control_plan.append(f"CONTROL\t{source_path}\t{release_root}/source.manifest.yaml\t{file_sha256(source_path)}")
control_plan.append(f"CONTROL\t{protocol_path}\t{release_root}/protocol.manifest.yaml\t{file_sha256(protocol_path)}")
if canonical_path is not None:
    hydration = json.loads(hydration_path.read_text(encoding="utf-8"))
    canonical_digest = file_sha256(canonical_path)
    canonical_key = f"{remote_prefix}/immutable/sha256/{canonical_digest}/{canonical_path.name}"
    if hydration.get("task") != task:
        raise SystemExit("Hydration record task does not match construction task")
    if hydration.get("output_bytes") != canonical_path.stat().st_size or hydration.get("output_sha256") != canonical_digest:
        raise SystemExit("Canonical source bytes/hash do not match hydration record")
    if hydration.get("remote_key") != canonical_key or hydration.get("uri") != f"b2://{bucket}/{canonical_key}":
        raise SystemExit("Hydration record does not point to the immutable canonical object")
    object_plan.append(f"CANONICAL\t{canonical_path}\t{canonical_key}\t{canonical_digest}")
    control_plan.append(f"CONTROL\t{hydration_path}\t{release_root}/canonical/source.json\t{file_sha256(hydration_path)}")
print("\n".join(object_plan + control_plan))
PY
)"

if ! compgen -G "${OUT_ROOT}/*.h5" >/dev/null; then
  echo "No protocol-gated HDF5 shards found under ${OUT_ROOT}." >&2
  exit 1
fi

while IFS=$'\t' read -r kind local_path remote_key digest; do
  case "$kind" in
    SHARD|CANONICAL) publish_immutable_file "$local_path" "$remote_key" "$digest" ;;
    LOCK|CONTROL) publish_immutable_file "$local_path" "$remote_key" "$digest" ;;
    *) echo "Invalid publication plan entry: ${kind}" >&2; exit 1 ;;
  esac
done <<< "$publication_plan"

echo "Publish plan complete for ${REMOTE_PREFIX}."
