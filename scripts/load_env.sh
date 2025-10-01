#!/usr/bin/env bash
set -euo pipefail

ENV_FILE=${1:-.env}
if [ ! -f "$ENV_FILE" ]; then
  echo "[load_env] missing $ENV_FILE (copy .env.example to $ENV_FILE and customize)." >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

echo "[load_env] Loaded environment from $ENV_FILE"
