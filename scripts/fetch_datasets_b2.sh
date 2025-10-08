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
    # Trim leading spaces
    line="${line#${line%%[![:space:]]*}}"
    # Skip empty or comment lines
    [ -z "$line" ] && continue
    [ "${line:0:1}" = "#" ] && continue
    if [[ "$line" =~ ^[[:space:]]*$key[[:space:]]*[:=][[:space:]]*(.*)$ ]]; then
      local val="${BASH_REMATCH[1]}"
      # Strip surrounding quotes if present
      if [[ "$val" =~ ^\"(.*)\"$ ]]; then
        echo "${BASH_REMATCH[1]}"
      elif [[ "$val" =~ ^\'(.*)\'$ ]]; then
        echo "${BASH_REMATCH[1]}"
      else
        echo "$val"
      fi
      return 0
    fi
  done < "$file"
  return 1
}

# Load required B2 variables from .env if present (supports KEY=value and KEY: value)
B2_ENV_FILE=".env"
if [ -f "$B2_ENV_FILE" ]; then
  : "${B2_KEY_ID:=$(read_env_key "$B2_ENV_FILE" B2_KEY_ID || read_env_key "$B2_ENV_FILE" B2_ACCOUNT_ID || read_env_key "$B2_ENV_FILE" ACCOUNT_ID || read_env_key "$B2_ENV_FILE" KEY_ID || true)}"
  : "${B2_APP_KEY:=$(read_env_key "$B2_ENV_FILE" B2_APP_KEY || read_env_key "$B2_ENV_FILE" B2_APPLICATION_KEY || read_env_key "$B2_ENV_FILE" APPLICATION_KEY || read_env_key "$B2_ENV_FILE" APP_KEY || true)}"
  : "${B2_BUCKET:=$(read_env_key "$B2_ENV_FILE" B2_BUCKET || read_env_key "$B2_ENV_FILE" B2_BUCKET_NAME || read_env_key "$B2_ENV_FILE" BUCKET || read_env_key "$B2_ENV_FILE" BUCKET_NAME || read_env_key "$B2_ENV_FILE" BUCKET_ID || true)}"
  : "${B2_PREFIX:=$(read_env_key "$B2_ENV_FILE" B2_PREFIX || true)}"
fi

WORKDIR=${WORKDIR:-$PWD}
DATA_ROOT=${DATA_ROOT:-$WORKDIR/data/pdebench}
B2_PREFIX=${B2_PREFIX:-pdebench}
mkdir -p "$DATA_ROOT" "$WORKDIR/artifacts/cache"

if [ "${DRY_RUN:-0}" -eq 1 ]; then
  : "${B2_BUCKET:=example-bucket}"
else
  : "${B2_KEY_ID:?Set B2_KEY_ID in .env or environment}"
  : "${B2_APP_KEY:?Set B2_APP_KEY in .env or environment}"
  : "${B2_BUCKET:?Set B2_BUCKET (or BUCKET/BUCKET_NAME) in .env or environment}"
  if ! command -v rclone >/dev/null 2>&1; then
    echo "rclone is required but not found. Install via: brew install rclone" >&2
    exit 1
  fi
  export RCLONE_CONFIG_UPSB2_TYPE=b2
  export RCLONE_CONFIG_UPSB2_ACCOUNT="${B2_KEY_ID}"
  export RCLONE_CONFIG_UPSB2_KEY="${B2_APP_KEY}"
fi

# Gather dataset keys
declare -a KEYS
if [ "$#" -gt 0 ]; then
  KEYS=("$@")
elif [ -n "${WANDB_DATASETS:-}" ]; then
  IFS=', ' read -r -a KEYS <<< "${WANDB_DATASETS}"
else
  echo "Provide dataset keys as args or set WANDB_DATASETS" >&2
  exit 1
fi

# Stream-extract <key>.tar.gz from B2 into DATA_ROOT without creating a local temp copy
stream_extract() {
  local key="$1"
  local object="${key}.tar.gz"
  local src="UPSB2:${B2_BUCKET}/${B2_PREFIX}/${object}"
  echo "Streaming ${src} -> ${DATA_ROOT}"
  if [ "${DRY_RUN:-0}" -eq 1 ]; then
    echo "DRY_RUN: rclone cat \"${src}\" | tar -xz -C \"${DATA_ROOT}\""
  else
    rclone cat "${src}" | tar -xz -C "${DATA_ROOT}"
  fi
}

# Optional cleanup of old local splits to prevent storage bloat
if [ "${CLEAN_OLD_SPLITS:-1}" -eq 1 ]; then
  find "${DATA_ROOT}" -maxdepth 1 -type f -name "*.h5" -delete || true
fi

for key in "${KEYS[@]}"; do
  stream_extract "${key}"
done

echo "Done. PDEBENCH_ROOT=${DATA_ROOT}"


