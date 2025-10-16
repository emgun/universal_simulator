#!/usr/bin/env bash
set -euo pipefail

# Fetch UPS datasets from Backblaze B2 using rclone with .env-provided credentials.
#
# Usage examples:
#   WANDB_DATASETS="burgers1d_subset_v1" ./scripts/fetch_datasets_b2.sh
#   DATA_ROOT=/path/to/data ./scripts/fetch_datasets_b2.sh burgers1d_subset_v1 advection1d_subset_v1
#
# Environment/.env variables:
#   B2_KEY_ID=<Backblaze B2 application key ID>
#   B2_APP_KEY=<Backblaze B2 application key secret>
#   B2_BUCKET=<Backblaze B2 bucket name>
#   B2_PREFIX=pdebench   # optional object prefix (folder) inside the bucket
#   B2_S3_ENDPOINT=https://s3.us-west-004.backblazeb2.com  # optional S3 endpoint
#   B2_S3_REGION=us-west-004                             # optional S3 region
#
# Optional toggles:
#   DRY_RUN=1            # do not contact B2; just print the command
#   CLEAN_OLD_SPLITS=1   # delete existing local *.h5 before extraction (default: 1)
#   WORKDIR=$PWD         # working directory (defaults to current)
#   DATA_ROOT=$WORKDIR/data/pdebench

# Helper: read a key from .env supporting both KEY=value and KEY: value
read_env_key() {
  local file="$1"; shift
  local key="$1"; shift || true
  if [ ! -f "$file" ]; then
    return 1
  fi
  local line
  while IFS= read -r line; do
    line="${line#${line%%[![:space:]]*}}"  # trim leading whitespace
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

REMOTE_NAME="UPSB2"

append_candidate() {
  local value="$1"
  for existing in "${CANDIDATES[@]-}"; do
    if [ "$existing" = "$value" ]; then
      return 0
    fi
  done
  CANDIDATES+=("$value")
}

build_candidates() {
  local key="$1"
  CANDIDATES=()

  append_candidate "$key"
  append_candidate "${key}.h5"
  append_candidate "${key}.tar.gz"
  append_candidate "${key}.tar"

  local group="${key%%_*}"
  if [ -n "$group" ] && [ "$group" != "$key" ]; then
    append_candidate "${group}/${key}"
    append_candidate "${group}/${key}.h5"
    append_candidate "${group}/${key}.tar.gz"
    append_candidate "${group}/${key}.tar"
  fi

  if [[ "$key" == */* ]]; then
    local base="${key##*/}"
    append_candidate "$key"
    append_candidate "${key}.h5"
    append_candidate "${key}.tar.gz"
    append_candidate "${key}.tar"
    append_candidate "${key}/${base}.h5"
    append_candidate "${key}/${base}.tar.gz"
    append_candidate "${key}/${base}.tar"
  fi
}

configure_rclone() {
  if [ "${DRY_RUN:-0}" -eq 1 ]; then
    : "${B2_BUCKET:=example-bucket}"
    return
  fi

  : "${B2_KEY_ID:?Set B2_KEY_ID in .env or environment}"
  : "${B2_APP_KEY:?Set B2_APP_KEY in .env or environment}"
  : "${B2_BUCKET:?Set B2_BUCKET (or BUCKET/BUCKET_NAME) in .env or environment}"

  if ! command -v rclone >/dev/null 2>&1; then
    echo "rclone is required but not found. Install via: brew install rclone" >&2
    exit 1
  fi

  if [ -n "${B2_S3_ENDPOINT:-}" ] || [ -n "${B2_S3_REGION:-}" ]; then
    export RCLONE_CONFIG_${REMOTE_NAME}_TYPE=s3
    export RCLONE_CONFIG_${REMOTE_NAME}_PROVIDER=B2
    export RCLONE_CONFIG_${REMOTE_NAME}_ACCESS_KEY_ID="${B2_KEY_ID}"
    export RCLONE_CONFIG_${REMOTE_NAME}_SECRET_ACCESS_KEY="${B2_APP_KEY}"
    [ -n "${B2_S3_ENDPOINT:-}" ] && export RCLONE_CONFIG_${REMOTE_NAME}_ENDPOINT="${B2_S3_ENDPOINT}"
    [ -n "${B2_S3_REGION:-}" ] && export RCLONE_CONFIG_${REMOTE_NAME}_REGION="${B2_S3_REGION}"
  else
    export RCLONE_CONFIG_${REMOTE_NAME}_TYPE=b2
    export RCLONE_CONFIG_${REMOTE_NAME}_ACCOUNT="${B2_KEY_ID}"
    export RCLONE_CONFIG_${REMOTE_NAME}_KEY="${B2_APP_KEY}"
  fi
}

remote_root_path() {
  local bucket="$1"
  local prefix="$2"
  prefix="${prefix#/}"
  prefix="${prefix%/}"
  if [ -n "$prefix" ]; then
    echo "${REMOTE_NAME}:${bucket}/${prefix}"
  else
    echo "${REMOTE_NAME}:${bucket}"
  fi
}

fetch_file() {
  local remote_path="$1"
  local destination="$2"
  local pretty_path="${remote_path#${REMOTE_NAME}:}"
  if [ "${DRY_RUN:-0}" -eq 1 ]; then
    echo "DRY_RUN: copy ${pretty_path} -> ${destination}"
    return 0
  fi

  echo "Copying ${pretty_path} -> ${destination}"
  case "$remote_path" in
    *.tar.gz)
      rclone cat "$remote_path" | tar -xz -C "$destination" ;;
    *.tar)
      rclone cat "$remote_path" | tar -x -C "$destination" ;;
    *)
      rclone copy --copy-links "$remote_path" "$destination" ;;
  esac
}

fetch_directory() {
  local remote_path="$1"
  local destination="$2"
  local pretty_path="${remote_path#${REMOTE_NAME}:}"
  if [ "${DRY_RUN:-0}" -eq 1 ]; then
    echo "DRY_RUN: copy directory ${pretty_path} -> ${destination}"
    return 0
  fi

  echo "Copying directory ${pretty_path} -> ${destination}"
  mkdir -p "$destination"
  rclone copy --copy-links "$remote_path" "$destination" --create-empty-src-dirs
}

download_dataset() {
  local key="$1"
  local remote_root="$2"
  build_candidates "$key"

  if [ "${DRY_RUN:-0}" -eq 1 ]; then
    local candidate="${CANDIDATES[0]}"
    local remote_candidate="$remote_root"
    [ -n "$candidate" ] && remote_candidate="${remote_candidate}/${candidate}"
    echo "DRY_RUN: would fetch ${remote_candidate#${REMOTE_NAME}:} -> ${DATA_ROOT}"
    return 0
  fi

  for candidate in "${CANDIDATES[@]}"; do
    local remote_candidate="$remote_root"
    [ -n "$candidate" ] && remote_candidate="${remote_candidate}/${candidate}"

    if rclone lsjson --files-only "$remote_candidate" >/dev/null 2>&1; then
      fetch_file "$remote_candidate" "$DATA_ROOT"
      return 0
    fi

    if rclone lsjson --dirs-only "$remote_candidate" >/dev/null 2>&1; then
      fetch_directory "$remote_candidate" "$DATA_ROOT/$key"
      return 0
    fi
  done

  echo "Error: could not locate dataset '$key' under ${remote_root#${REMOTE_NAME}:}" >&2
  return 1
}

B2_ENV_FILE=".env"
if [ -f "$B2_ENV_FILE" ]; then
  : "${B2_KEY_ID:=$(read_env_key "$B2_ENV_FILE" B2_KEY_ID || read_env_key "$B2_ENV_FILE" B2_ACCOUNT_ID || read_env_key "$B2_ENV_FILE" ACCOUNT_ID || read_env_key "$B2_ENV_FILE" KEY_ID || true)}"
  : "${B2_APP_KEY:=$(read_env_key "$B2_ENV_FILE" B2_APP_KEY || read_env_key "$B2_ENV_FILE" B2_APPLICATION_KEY || read_env_key "$B2_ENV_FILE" APPLICATION_KEY || read_env_key "$B2_ENV_FILE" APP_KEY || true)}"
  : "${B2_BUCKET:=$(read_env_key "$B2_ENV_FILE" B2_BUCKET || read_env_key "$B2_ENV_FILE" B2_BUCKET_NAME || read_env_key "$B2_ENV_FILE" BUCKET || read_env_key "$B2_ENV_FILE" BUCKET_NAME || read_env_key "$B2_ENV_FILE" BUCKET_ID || true)}"
  : "${B2_PREFIX:=$(read_env_key "$B2_ENV_FILE" B2_PREFIX || true)}"
  : "${B2_S3_ENDPOINT:=$(read_env_key "$B2_ENV_FILE" B2_S3_ENDPOINT || true)}"
  : "${B2_S3_REGION:=$(read_env_key "$B2_ENV_FILE" B2_S3_REGION || true)}"
fi

WORKDIR=${WORKDIR:-$PWD}
DATA_ROOT=${DATA_ROOT:-$WORKDIR/data/pdebench}
B2_PREFIX=${B2_PREFIX:-pdebench}
mkdir -p "$DATA_ROOT" "$WORKDIR/artifacts/cache"

configure_rclone

REMOTE_ROOT=$(remote_root_path "$B2_BUCKET" "$B2_PREFIX")

if [ "${CLEAN_OLD_SPLITS:-1}" -eq 1 ]; then
  find "$DATA_ROOT" -maxdepth 1 -type f -name "*.h5" -delete || true
fi

declare -a KEYS
if [ "$#" -gt 0 ]; then
  KEYS=("$@")
elif [ -n "${WANDB_DATASETS:-}" ]; then
  IFS=',' read -r -a KEYS <<< "${WANDB_DATASETS}"
  declare -a CLEANED_KEYS=()
  for item in "${KEYS[@]}"; do
    item="${item#${item%%[![:space:]]*}}"
    item="${item%${item##*[![:space:]]}}"
    [ -n "$item" ] && CLEANED_KEYS+=("$item")
  done
  KEYS=("${CLEANED_KEYS[@]}")
  unset CLEANED_KEYS
else
  echo "Provide dataset keys as args or set WANDB_DATASETS" >&2
  exit 1
fi

if [ "${#KEYS[@]}" -eq 0 ]; then
  echo "No dataset keys provided after parsing." >&2
  exit 1
fi

status=0
for key in "${KEYS[@]}"; do
  if ! download_dataset "$key" "$REMOTE_ROOT"; then
    status=1
  fi
done

if [ $status -ne 0 ]; then
  exit $status
fi

echo "Done. PDEBENCH_ROOT=${DATA_ROOT}"
