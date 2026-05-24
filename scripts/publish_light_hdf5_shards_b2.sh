#!/usr/bin/env bash
set -euo pipefail

# Build and/or publish small PDEBench HDF5 shards to Backblaze B2.
#
# Safe dry run over already-built shards:
#   DRY_RUN=1 OUT_ROOT=data/pdebench_light VERSION=light-v1 bash scripts/publish_light_hdf5_shards_b2.sh
#
# Build locally, write a manifest, then publish:
#   ENV_FILE=/path/to/.env DRY_RUN=0 BUILD_SHARDS=1 VERSION=light-v1 bash scripts/publish_light_hdf5_shards_b2.sh

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
    export RCLONE_CONFIG_UPSB2_PROVIDER=B2
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

publish_file() {
  local local_path="$1"
  local remote_key="$2"
  local remote="UPSB2:${B2_BUCKET}/${remote_key}"
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "DRY_RUN: rclone copyto ${local_path} ${remote}"
  else
    rclone copyto "$local_path" "$remote"
  fi
}

ENV_FILE=${ENV_FILE:-.env}
VERSION=${VERSION:-light-v1}
REMOTE_PREFIX=${REMOTE_PREFIX:-$VERSION}
SOURCE_ROOT=${SOURCE_ROOT:-data/pdebench}
OUT_ROOT=${OUT_ROOT:-data/pdebench_light}
MANIFEST=${MANIFEST:-docs/demo_data_manifest.yaml}
TASKS=${TASKS:-"burgers1d advection1d darcy2d"}
TRAIN_COUNT=${TRAIN_COUNT:-128}
VAL_COUNT=${VAL_COUNT:-32}
TEST_COUNT=${TEST_COUNT:-32}
START_INDEX=${START_INDEX:-0}
BUILD_SHARDS=${BUILD_SHARDS:-0}
OVERWRITE=${OVERWRITE:-1}
DRY_RUN=${DRY_RUN:-1}

load_optional_env "$ENV_FILE"
configure_rclone

if [ "$BUILD_SHARDS" -eq 1 ]; then
  overwrite_args=()
  if [ "$OVERWRITE" -eq 1 ]; then
    overwrite_args+=(--overwrite)
  fi
  python scripts/make_light_hdf5_shards.py \
    --root "$SOURCE_ROOT" \
    --out-root "$OUT_ROOT" \
    --tasks $TASKS \
    --train-count "$TRAIN_COUNT" \
    --val-count "$VAL_COUNT" \
    --test-count "$TEST_COUNT" \
    --start-index "$START_INDEX" \
    --version "$VERSION" \
    --remote-prefix "$REMOTE_PREFIX" \
    --manifest "$MANIFEST" \
    "${overwrite_args[@]}"
fi

if ! compgen -G "${OUT_ROOT}/*.h5" >/dev/null; then
  echo "No HDF5 shards found under ${OUT_ROOT}. Set BUILD_SHARDS=1 or provide existing shards." >&2
  exit 1
fi

for shard in "${OUT_ROOT}"/*.h5; do
  base="$(basename "$shard")"
  task="${base%_*}"
  publish_file "$shard" "${REMOTE_PREFIX}/${task}/${base}"
done

if [ -f "$MANIFEST" ]; then
  publish_file "$MANIFEST" "${REMOTE_PREFIX}/manifest.yaml"
fi

echo "Publish plan complete for ${REMOTE_PREFIX}."
