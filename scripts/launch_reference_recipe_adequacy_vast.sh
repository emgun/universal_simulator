#!/usr/bin/env bash
set -euo pipefail

# Dry-run-first, bounded Vast launcher for reference-recipe adequacy.

ENV_FILE=${ENV_FILE:-.env}
DRY_RUN=${DRY_RUN:-1}
GIT_REF=${GIT_REF:-$(git rev-parse --abbrev-ref HEAD)}
GPU=${GPU:-RTX_4090}
NUM_GPUS=${NUM_GPUS:-1}
DISK_GB=${DISK_GB:-64}
MAX_DISK_GB=${MAX_DISK_GB:-96}
MAX_DPH=${MAX_DPH:-0.45}
OFFER_ID=${OFFER_ID:-}
ORDER=${ORDER:-dph_total}
LIMIT=${LIMIT:-10}
WORKDIR=${WORKDIR:-/workspace}
REMOTE_SCRIPT=${REMOTE_SCRIPT:-scripts/run_remote_reference_recipe_adequacy.sh}
ARTIFACT_PREFIX=${ARTIFACT_PREFIX:-remote-runs/reference-recipe-adequacy}

read_env_key() {
  local file="$1" key="$2" line val
  [ -f "$file" ] || return 1
  while IFS= read -r line; do
    line="${line#${line%%[![:space:]]*}}"
    [ -z "$line" ] && continue
    [ "${line:0:1}" = "#" ] && continue
    if [[ "$line" =~ ^[[:space:]]*$key[[:space:]]*[:=][[:space:]]*(.*)$ ]]; then
      val="${BASH_REMATCH[1]}"
      val="${val%\"}"; val="${val#\"}"; val="${val%\'}"; val="${val#\'}"
      echo "$val"
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

case "$DRY_RUN" in 0|1) ;; *) echo "DRY_RUN must be 0 or 1." >&2; exit 2 ;; esac
if ! [[ "$NUM_GPUS" =~ ^[0-9]+$ && "$DISK_GB" =~ ^[0-9]+$ && "$MAX_DISK_GB" =~ ^[0-9]+$ ]]; then
  echo "NUM_GPUS, DISK_GB, and MAX_DISK_GB must be integers." >&2
  exit 2
fi
if [ "$NUM_GPUS" -ne 1 ]; then
  echo "Reference-recipe adequacy is bounded to exactly one GPU." >&2
  exit 2
fi
if [ "$DISK_GB" -le 0 ]; then
  echo "DISK_GB must be a positive integer." >&2
  exit 2
fi
if [ "$DISK_GB" -gt "$MAX_DISK_GB" ]; then
  echo "Refusing DISK_GB=${DISK_GB}; bounded maximum is ${MAX_DISK_GB} GB." >&2
  exit 2
fi
# Validate the cap without relying on shell floating-point arithmetic.
if ! "${PYTHON:-python}" -c 'import sys; value=float(sys.argv[1]); raise SystemExit(0 if value > 0 else 1)' "$MAX_DPH"; then
  echo "MAX_DPH must be a positive number." >&2
  exit 2
fi

if [ "$DRY_RUN" -eq 0 ]; then
  missing=()
  [ -n "${B2_KEY_ID:-}" ] || missing+=(B2_KEY_ID)
  [ -n "${B2_APP_KEY:-}" ] || missing+=(B2_APP_KEY)
  [ -n "${B2_BUCKET:-}" ] || missing+=(B2_BUCKET)
  if [ "${#missing[@]}" -gt 0 ]; then
    echo "Refusing paid launch; missing B2 credentials: ${missing[*]}." >&2
    exit 2
  fi
  if [ "$GIT_REF" = "HEAD" ] || [ "$GIT_REF" = "main" ]; then
    echo "Refusing paid launch without an explicit pushed non-main ref." >&2
    exit 2
  fi
  local_commit=$(git rev-parse "${GIT_REF}^{commit}")
  remote_commit=$(git ls-remote origin "refs/heads/${GIT_REF}" | awk 'NR==1 {print $1}')
  if [ -z "$remote_commit" ] || [ "$local_commit" != "$remote_commit" ]; then
    echo "Refusing paid launch: GIT_REF=${GIT_REF} is not pushed at the local commit." >&2
    exit 2
  fi
  if [ ! -f "$REMOTE_SCRIPT" ] || ! git cat-file -e "${remote_commit}:${REMOTE_SCRIPT}" 2>/dev/null; then
    echo "Refusing paid launch: ${REMOTE_SCRIPT} is not present in pushed GIT_REF=${GIT_REF}." >&2
    exit 2
  fi

  if ! command -v vastai >/dev/null 2>&1; then
    echo "Refusing paid launch: vastai CLI is unavailable for cost preflight." >&2
    exit 2
  fi
  requested_offer_id=$OFFER_ID
  offer_query="gpu_name=${GPU} num_gpus=${NUM_GPUS} rentable=true verified=true disk_space>=${DISK_GB} dph_total<=${MAX_DPH}"
  search_limit=$LIMIT
  if [ -n "$requested_offer_id" ]; then
    # Vast's offer query language does not reliably filter by offer ID. Fetch
    # a wider bounded set and select the requested ID from the returned rows.
    search_limit=200
  fi
  offers_json=$(vastai search offers "$offer_query" -o dph_total --limit "$search_limit" --raw)
  resolved=$("${PYTHON:-python}" -c '
import json, sys
offers=json.load(sys.stdin)
requested=sys.argv[2]
if requested:
    offers=[offer for offer in offers if str(offer.get("id") or offer.get("ask_contract_id")) == requested]
if not isinstance(offers, list) or not offers:
    raise SystemExit("no offer satisfies the bounded paid-launch preflight")
offer=offers[0]
price=float(offer["dph_total"])
cap=float(sys.argv[1])
if price > cap:
    raise SystemExit(f"offer price ${price}/hr exceeds MAX_DPH=${cap}/hr")
offer_id=offer.get("id") or offer.get("ask_contract_id")
if offer_id is None:
    raise SystemExit("bounded offer lacks an id")
print(f"{offer_id} {price}")
' "$MAX_DPH" "$requested_offer_id" <<<"$offers_json") || {
    echo "Refusing paid launch: Vast offer did not pass the hourly-cost cap." >&2
    exit 2
  }
  read -r OFFER_ID resolved_dph <<<"$resolved"
  echo "Cost preflight selected Vast offer ${OFFER_ID} at \$${resolved_dph}/hr (cap \$${MAX_DPH}/hr)."
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
  --install-mode experiment
  --bootstrap-mode tracked-script
  --script-args "DRY_RUN=0 ARTIFACT_PREFIX=$ARTIFACT_PREFIX"
  --auto-shutdown
)

if [ -n "$OFFER_ID" ]; then
  args+=(--offer-id "$OFFER_ID")
else
  args+=(--order "$ORDER" --limit "$LIMIT")
fi
if [ "$DRY_RUN" -eq 1 ]; then
  echo "DRY RUN: paid launch will preflight offers at MAX_DPH=\$${MAX_DPH}/hr."
  args+=(--dry-run)
fi
[ -n "${B2_KEY_ID:-}" ] && args+=(--b2-key-id "$B2_KEY_ID")
[ -n "${B2_APP_KEY:-}" ] && args+=(--b2-app-key "$B2_APP_KEY")
[ -n "${B2_BUCKET:-}" ] && args+=(--b2-bucket "$B2_BUCKET")
[ -n "${B2_S3_ENDPOINT:-}" ] && args+=(--b2-s3-endpoint "$B2_S3_ENDPOINT")
[ -n "${B2_S3_REGION:-}" ] && args+=(--b2-s3-region "$B2_S3_REGION")

"${args[@]}"
