#!/usr/bin/env bash
set -euo pipefail

ENV_FILE=${ENV_FILE:-.env}
DRY_RUN=${DRY_RUN:-1}
GIT_REF=${GIT_REF:-$(git rev-parse HEAD)}
GPU=${GPU:-RTX_4090}
DISK_GB=${DISK_GB:-96}
MAX_DPH=${MAX_DPH:-0.45}
MAX_RUNTIME_MINUTES=${MAX_RUNTIME_MINUTES:-600}
STARTUP_TIMEOUT_MINUTES=${STARTUP_TIMEOUT_MINUTES:-15}
OFFER_ID=${OFFER_ID:-}
REMOTE_SCRIPT=${REMOTE_SCRIPT:-scripts/run_remote_strat_v1_modular_shared_trunk.sh}
ARTIFACT_PREFIX=${ARTIFACT_PREFIX:-remote-runs/strat-v1-modular-shared-trunk}
TRAINING_LOCK=${TRAINING_LOCK:-docs/data/releases/strat-v1/universal/9d43d283f04f5b8d17cf6126ad189075c53307e715d7d4f61af440c2fed155c1/training.lock.json}
PLAN=${PLAN:-docs/research/artifacts/strat_v1_modular_shared_trunk_plan_v5.json}

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
[[ "$GIT_REF" =~ ^[0-9a-f]{40}$ ]] || { echo "GIT_REF must be a full lowercase commit" >&2; exit 2; }
[ "$MAX_RUNTIME_MINUTES" -gt 0 ] && [ "$MAX_RUNTIME_MINUTES" -le 600 ] || { echo "MAX_RUNTIME_MINUTES must be in 1..600" >&2; exit 2; }
[ "$STARTUP_TIMEOUT_MINUTES" -gt 0 ] && [ "$STARTUP_TIMEOUT_MINUTES" -lt "$MAX_RUNTIME_MINUTES" ] || { echo "STARTUP_TIMEOUT_MINUTES must be positive and below MAX_RUNTIME_MINUTES" >&2; exit 2; }
python - "$MAX_DPH" "$MAX_RUNTIME_MINUTES" <<'PY'
import sys
price=float(sys.argv[1]); minutes=float(sys.argv[2])
if price <= 0 or price > .45: raise SystemExit("MAX_DPH must be in (0, 0.45]")
if price * minutes / 60 > 4.50 + 1e-12: raise SystemExit("maximum run cost exceeds $4.50")
PY
implementation_commit=$(python - "$PLAN" <<'PY'
import hashlib,json,sys
path=sys.argv[1]
p=json.load(open(path,encoding="utf-8"))
recorded=p.get("plan_sha256")
unsigned={k:v for k,v in p.items() if k != "plan_sha256"}
computed=hashlib.sha256(json.dumps(unsigned,sort_keys=True,separators=(",",":")).encode()).hexdigest()
retired={
  "ec36aead4c537267fae78c71de8d14156fba253899f90ec72fe867dd6bce80e8",
  "f00f2031be4138954cffc21fe5793aeeb0edbf9197b11e6290534e176897267d",
  "017af9eab7c60250d0825f91eea83f25a212e49c09368e96a3037e889f70290e",
  "88bcb9c70eefa1f7bda97577ff65dcd82e080022594cb9a3b5181b9418b06487",
}
if recorded != computed: raise SystemExit("D6 plan self hash does not match")
if recorded in retired: raise SystemExit("refusing retired D6 plan")
if p.get("plan_id") != "strat-v1-modular-shared-trunk-d6-v5": raise SystemExit("refusing superseded D6 plan identity")
if p.get("mode") != "validation_only" or p.get("heldout_access") != "forbidden" or p.get("measurement_lock_access") != "forbidden": raise SystemExit("refusing non-validation D6 plan")
recovery=p.get("recovery", {})
if recovery.get("prior_vast_contract") != 45126713: raise SystemExit("refusing unbound D6 recovery history")
if recovery.get("scientific_attempt_consumed") is not False: raise SystemExit("refusing consumed D6 scientific attempt")
if recovery.get("authorized_recovery_provider_launches") != 1: raise SystemExit("refusing unbounded D6 recovery")
if recovery.get("startup_timeout_minutes") != 15: raise SystemExit("refusing altered D6 startup timeout")
print(p["bindings"]["source"]["implementation_commit"])
PY
)
git cat-file -e "${GIT_REF}^{commit}"
git merge-base --is-ancestor "$implementation_commit" "$GIT_REF" || { echo "GIT_REF must descend from the D6 implementation commit" >&2; exit 2; }

if [ "$DRY_RUN" -eq 0 ]; then
  [ -n "${B2_KEY_ID:-}" ] && [ -n "${B2_APP_KEY:-}" ] && [ -n "${B2_BUCKET:-}" ] || { echo "Missing B2 credentials" >&2; exit 2; }
  git fetch origin --quiet
  git ls-remote origin | awk -v commit="$GIT_REF" '$1 == commit {found=1} END {exit !found}' || { echo "GIT_REF must be the exact commit of a pushed ref" >&2; exit 2; }
  offers=$(vastai search offers "gpu_name=${GPU} num_gpus=1 rentable=true verified=true disk_space>=${DISK_GB} dph_total<=${MAX_DPH}" -o dph_total --limit 200 --raw)
  resolved=$(python -c 'import json,sys
rows=json.load(sys.stdin); requested=sys.argv[2]
if requested: rows=[r for r in rows if str(r.get("id") or r.get("ask_contract_id")) == requested]
if not rows: raise SystemExit("no verified bounded offer")
r=rows[0]; price=float(r["dph_total"]); cap=float(sys.argv[1])
if price > cap: raise SystemExit("offer exceeds cap")
print(r.get("id") or r.get("ask_contract_id"), price)' "$MAX_DPH" "$OFFER_ID" <<<"$offers")
  read -r OFFER_ID price <<<"$resolved"
  echo "Cost preflight selected verified Vast offer $OFFER_ID at \$$price/hr."
fi

TRANSFER_MANIFEST=.vast/d6-transfer-${GIT_REF:0:12}-$$.json
TRANSFER_URL_RECEIPT=.vast/d6-transfer-url-${GIT_REF:0:12}-$$.json
VAST_RECEIPT=${VAST_RECEIPT:-.vast/receipts/d6-v5-recovery.json}
[ "$DRY_RUN" -eq 1 ] || [ ! -e "$VAST_RECEIPT" ] || { echo "refusing a second D6 v5 recovery provider launch: $VAST_RECEIPT already exists" >&2; exit 2; }
finalization_complete=$DRY_RUN
cleanup_transfer_files() {
  if [ "$finalization_complete" -eq 1 ]; then
    rm -f "$TRANSFER_MANIFEST" "$TRANSFER_URL_RECEIPT"
  else
    echo "Retained private D6 transfer receipts for recovery: $TRANSFER_MANIFEST $TRANSFER_URL_RECEIPT" >&2
  fi
}
trap cleanup_transfer_files EXIT
transfer_token=DRY_RUN_CAPABILITY
if [ "$DRY_RUN" -eq 0 ]; then
  python scripts/generate_b2_presigned_bundle.py \
    --lock "$TRAINING_LOCK" --plan "$PLAN" --artifact-prefix "$ARTIFACT_PREFIX" \
    --max-runtime-minutes "$MAX_RUNTIME_MINUTES" --env-file "$ENV_FILE" \
    --output "$TRANSFER_MANIFEST" --upload-control --url-output "$TRANSFER_URL_RECEIPT"
  transfer_token=$(python - "$TRANSFER_URL_RECEIPT" <<'PY'
import base64,json,sys
url=json.load(open(sys.argv[1],encoding="utf-8"))["TRANSFER_MANIFEST_URL"]
print(base64.urlsafe_b64encode(url.encode("utf-8")).decode("ascii").rstrip("="))
PY
)
fi

args=(python scripts/vast_launch.py launch --gpu "$GPU" --num-gpus 1 --disk "$DISK_GB" --git-ref "$GIT_REF" --workdir /workspace --remote-script "$REMOTE_SCRIPT" --skip-prefetch --skip-rclone-install --install-mode experiment --bootstrap-mode tracked-script --script-args "DRY_RUN=0 ARTIFACT_PREFIX=$ARTIFACT_PREFIX TRANSFER_MANIFEST_URL_B64=$transfer_token" --auto-shutdown --managed --max-runtime-minutes "$MAX_RUNTIME_MINUTES" --startup-timeout-minutes "$STARTUP_TIMEOUT_MINUTES" --success-marker "Uploaded verified D6 ingress artifact:" --receipt "$VAST_RECEIPT" --launch-retries 0)
[ -n "$OFFER_ID" ] && args+=(--offer-id "$OFFER_ID") || args+=(--order dph_total --limit 10)
[ "$DRY_RUN" -eq 1 ] && args+=(--dry-run)
env -u B2_KEY_ID -u B2_ACCOUNT_ID -u B2_APP_KEY -u B2_APPLICATION_KEY \
  -u B2_BUCKET -u B2_BUCKET_NAME -u B2_S3_ENDPOINT -u B2_S3_REGION "${args[@]}"
if [ "$DRY_RUN" -eq 0 ]; then
  python scripts/finalize_d5_presigned_transfer.py \
    --manifest "$TRANSFER_MANIFEST" --env-file "$ENV_FILE" --receipt "$VAST_RECEIPT" \
    --archive-stem strat_v1_modular_shared_trunk --workflow-label D6
  finalization_complete=1
fi
