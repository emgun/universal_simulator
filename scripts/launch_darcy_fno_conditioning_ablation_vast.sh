#!/usr/bin/env bash
set -euo pipefail

ENV_FILE=${ENV_FILE:-.env}
DRY_RUN=${DRY_RUN:-1}
GIT_REF=${GIT_REF:-$(git rev-parse --abbrev-ref HEAD)}
GPU=${GPU:-RTX_4090}
DISK_GB=${DISK_GB:-64}
MAX_DPH=${MAX_DPH:-0.45}
OFFER_ID=${OFFER_ID:-}
REMOTE_SCRIPT=${REMOTE_SCRIPT:-scripts/run_remote_darcy_fno_conditioning_ablation.sh}
ARTIFACT_PREFIX=${ARTIFACT_PREFIX:-remote-runs/darcy-fno-conditioning-ablation}

read_env_key() {
  local file="$1" key="$2" line val
  [ -f "$file" ] || return 1
  while IFS= read -r line; do
    line="${line#${line%%[![:space:]]*}}"
    [ -z "$line" ] && continue
    [ "${line:0:1}" = "#" ] && continue
    if [[ "$line" =~ ^[[:space:]]*$key[[:space:]]*[:=][[:space:]]*(.*)$ ]]; then
      val="${BASH_REMATCH[1]}"; val="${val%\"}"; val="${val#\"}"; val="${val%\'}"; val="${val#\'}"
      echo "$val"; return 0
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

case "$DRY_RUN" in 0|1) ;; *) echo "DRY_RUN must be 0 or 1" >&2; exit 2 ;; esac
if [ "$DRY_RUN" -eq 0 ]; then
  [ -n "${B2_KEY_ID:-}" ] && [ -n "${B2_APP_KEY:-}" ] && [ -n "${B2_BUCKET:-}" ] || { echo "Missing B2 credentials" >&2; exit 2; }
  [ "$GIT_REF" != main ] && [ "$GIT_REF" != HEAD ] || { echo "Use an explicit pushed non-main ref" >&2; exit 2; }
  local_commit=$(git rev-parse "${GIT_REF}^{commit}")
  remote_commit=$(git ls-remote origin "refs/heads/${GIT_REF}" | awk 'NR==1 {print $1}')
  [ -n "$remote_commit" ] && [ "$local_commit" = "$remote_commit" ] || { echo "GIT_REF is not pushed at local commit" >&2; exit 2; }
  offers=$(vastai search offers "gpu_name=${GPU} num_gpus=1 rentable=true verified=true disk_space>=${DISK_GB} dph_total<=${MAX_DPH}" -o dph_total --limit 200 --raw)
  resolved=$(python -c '
import json,sys
rows=json.load(sys.stdin); requested=sys.argv[2]
if requested: rows=[r for r in rows if str(r.get("id") or r.get("ask_contract_id")) == requested]
if not rows: raise SystemExit("no bounded offer")
r=rows[0]; price=float(r["dph_total"]); cap=float(sys.argv[1])
if price > cap: raise SystemExit("offer exceeds cap")
print(r.get("id") or r.get("ask_contract_id"), price)
' "$MAX_DPH" "$OFFER_ID" <<<"$offers")
  read -r OFFER_ID price <<<"$resolved"
  echo "Cost preflight selected Vast offer $OFFER_ID at \$$price/hr (cap \$$MAX_DPH/hr)."
fi

args=(python scripts/vast_launch.py launch --gpu "$GPU" --num-gpus 1 --disk "$DISK_GB" --git-ref "$GIT_REF" --workdir /workspace --remote-script "$REMOTE_SCRIPT" --skip-prefetch --install-mode experiment --bootstrap-mode tracked-script --script-args "DRY_RUN=0 ARTIFACT_PREFIX=$ARTIFACT_PREFIX" --auto-shutdown)
[ -n "$OFFER_ID" ] && args+=(--offer-id "$OFFER_ID") || args+=(--order dph_total --limit 10)
[ "$DRY_RUN" -eq 1 ] && args+=(--dry-run)
[ -n "${B2_KEY_ID:-}" ] && args+=(--b2-key-id "$B2_KEY_ID")
[ -n "${B2_APP_KEY:-}" ] && args+=(--b2-app-key "$B2_APP_KEY")
[ -n "${B2_BUCKET:-}" ] && args+=(--b2-bucket "$B2_BUCKET")
[ -n "${B2_S3_ENDPOINT:-}" ] && args+=(--b2-s3-endpoint "$B2_S3_ENDPOINT")
[ -n "${B2_S3_REGION:-}" ] && args+=(--b2-s3-region "$B2_S3_REGION")
"${args[@]}"
