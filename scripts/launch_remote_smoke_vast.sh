#!/usr/bin/env bash
set -euo pipefail

# Dry-run-first Vast launcher for the remote smoke pipeline.
#
# Defaults do not launch paid compute. Set DRY_RUN=0 only after reviewing the
# generated Vast command and onstart script.

ENV_FILE=${ENV_FILE:-.env}
DRY_RUN=${DRY_RUN:-1}
GIT_REF=${GIT_REF:-$(git rev-parse --abbrev-ref HEAD)}
DISK_GB=${DISK_GB:-32}
GPU=${GPU:-RTX_4090}
NUM_GPUS=${NUM_GPUS:-1}
ORDER=${ORDER:-dph_total}
LIMIT=${LIMIT:-10}
OFFER_ID=${OFFER_ID:-}
WORKDIR=${WORKDIR:-/workspace}
PIPELINE_ROOT=${PIPELINE_ROOT:-reports/demo/remote_smoke_pipeline}
REMOTE_SCRIPT=${REMOTE_SCRIPT:-scripts/run_remote_smoke_pipeline.sh}
AUTO_SHUTDOWN=${AUTO_SHUTDOWN:-1}
SSH=${SSH:-1}
ARGS_MODE=${ARGS_MODE:-0}
# 'experiment' includes matplotlib/wandb, which run_light_experiment.py needs
# whenever the queue actually executes (RUN_EXPERIMENTS=1).
INSTALL_MODE=${INSTALL_MODE:-experiment}
BOOTSTRAP_MODE=${BOOTSTRAP_MODE:-inline}
EXTRA_PIPELINE_ARGS=${EXTRA_PIPELINE_ARGS:-}

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

if [ -f "$ENV_FILE" ]; then
  : "${B2_KEY_ID:=$(read_env_key "$ENV_FILE" B2_KEY_ID || read_env_key "$ENV_FILE" B2_ACCOUNT_ID || true)}"
  : "${B2_APP_KEY:=$(read_env_key "$ENV_FILE" B2_APP_KEY || read_env_key "$ENV_FILE" B2_APPLICATION_KEY || true)}"
  : "${B2_BUCKET:=$(read_env_key "$ENV_FILE" B2_BUCKET || read_env_key "$ENV_FILE" B2_BUCKET_NAME || true)}"
  : "${B2_S3_ENDPOINT:=$(read_env_key "$ENV_FILE" B2_S3_ENDPOINT || true)}"
  : "${B2_S3_REGION:=$(read_env_key "$ENV_FILE" B2_S3_REGION || true)}"
fi

args=(
  python scripts/vast_launch.py launch
  --gpu "$GPU"
  --num-gpus "$NUM_GPUS"
  --disk "$DISK_GB"
  --git-ref "$GIT_REF"
  --workdir "$WORKDIR"
  --remote-script "$REMOTE_SCRIPT"
  --skip-prefetch
  --install-mode "$INSTALL_MODE"
  --bootstrap-mode "$BOOTSTRAP_MODE"
  --script-args "DRY_RUN=0 ENV_FILE=.env PIPELINE_ROOT=$PIPELINE_ROOT $EXTRA_PIPELINE_ARGS"
)

if [ -n "$OFFER_ID" ]; then
  args+=(--offer-id "$OFFER_ID")
else
  args+=(--order "$ORDER" --limit "$LIMIT")
fi

if [ "$DRY_RUN" -eq 1 ]; then
  args+=(--dry-run)
fi

if [ "$AUTO_SHUTDOWN" -eq 1 ]; then
  args+=(--auto-shutdown)
fi

if [ "$SSH" -eq 0 ]; then
  args+=(--no-ssh)
fi

if [ "$ARGS_MODE" -eq 1 ]; then
  args+=(--args-mode)
fi

[ -n "${B2_KEY_ID:-}" ] && args+=(--b2-key-id "$B2_KEY_ID")
[ -n "${B2_APP_KEY:-}" ] && args+=(--b2-app-key "$B2_APP_KEY")
[ -n "${B2_BUCKET:-}" ] && args+=(--b2-bucket "$B2_BUCKET")
[ -n "${B2_S3_ENDPOINT:-}" ] && args+=(--b2-s3-endpoint "$B2_S3_ENDPOINT")
[ -n "${B2_S3_REGION:-}" ] && args+=(--b2-s3-region "$B2_S3_REGION")

"${args[@]}"
