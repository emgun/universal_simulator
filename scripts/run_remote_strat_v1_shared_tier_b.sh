#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python}
PLAN=${PLAN:-docs/research/artifacts/strat_v1_shared_tier_b_plan.json}
CONFIG=${CONFIG:-configs/d5_strat_v1_shared_tier_b.yaml}
TRAINING_LOCK=${TRAINING_LOCK:-docs/data/releases/strat-v1/universal/9d43d283f04f5b8d17cf6126ad189075c53307e715d7d4f61af440c2fed155c1/training.lock.json}
CACHE=${CACHE:-reports/research/strat_v1_shared_tier_b_scratch/cache}
DATA_ROOT=${DATA_ROOT:-reports/research/strat_v1_shared_tier_b_scratch/data}
STAGE_REPORT=${STAGE_REPORT:-reports/research/strat_v1_shared_tier_b_stage.json}
OUTPUT_DIR=${OUTPUT_DIR:-reports/research/strat_v1_shared_tier_b}
RESULT=${RESULT:-reports/research/strat_v1_shared_tier_b_result.json}
ARTIFACT_PREFIX=${ARTIFACT_PREFIX:-remote-runs/strat-v1-shared-tier-b}
RESERVE_BYTES=${RESERVE_BYTES:-8589934592}
DRY_RUN=${DRY_RUN:-1}

if [ "$DRY_RUN" = 1 ]; then
  echo "$PYTHON -m ups.data.cli stage --lock $TRAINING_LOCK --cache $CACHE --run-dir $DATA_ROOT"
  echo "$PYTHON scripts/run_strat_v1_shared_tier_b.py --training-lock $TRAINING_LOCK --data-root $DATA_ROOT --config $CONFIG --output-dir $OUTPUT_DIR --plan-path $PLAN --plan-sha256 <from-plan> --device cuda"
  echo "$PYTHON scripts/materialize_strat_v1_shared_tier_b.py --plan $PLAN --summary $OUTPUT_DIR/summary.json --output $RESULT"
  exit 0
fi

: "${B2_KEY_ID:?Set B2_KEY_ID}"
: "${B2_APP_KEY:?Set B2_APP_KEY}"
: "${B2_BUCKET:?Set B2_BUCKET}"
command -v rclone >/dev/null || { echo "rclone is required" >&2; exit 1; }
export RCLONE_CONFIG_UPSB2_TYPE=s3 RCLONE_CONFIG_UPSB2_PROVIDER=Other
export RCLONE_CONFIG_UPSB2_ACCESS_KEY_ID="$B2_KEY_ID"
export RCLONE_CONFIG_UPSB2_SECRET_ACCESS_KEY="$B2_APP_KEY"
[ -n "${B2_S3_ENDPOINT:-}" ] && export RCLONE_CONFIG_UPSB2_ENDPOINT="$B2_S3_ENDPOINT"
[ -n "${B2_S3_REGION:-}" ] && export RCLONE_CONFIG_UPSB2_REGION="$B2_S3_REGION"

plan_sha=$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["plan_sha256"])' "$PLAN")
temporary_prefix="${ARTIFACT_PREFIX%/}/resumable/${plan_sha}"
success=0
preserve_resume() {
  status=$?
  if [ "$success" -ne 1 ] && [ -d "$OUTPUT_DIR" ]; then
    rclone sync "$OUTPUT_DIR" "UPSB2:${B2_BUCKET}/${temporary_prefix}/output" || true
    echo "Preserved resumable D5 arms: b2://${B2_BUCKET}/${temporary_prefix}/output" >&2
  fi
  exit "$status"
}
trap preserve_resume EXIT

resume_arg=()
if rclone lsf "UPSB2:${B2_BUCKET}/${temporary_prefix}/output" --files-only --recursive 2>/dev/null | grep -q .; then
  mkdir -p "$OUTPUT_DIR"
  rclone sync "UPSB2:${B2_BUCKET}/${temporary_prefix}/output" "$OUTPUT_DIR"
  test -f "$OUTPUT_DIR/run_identity.json"
  resume_arg=(--resume)
fi

$PYTHON -m pip install -e .
$PYTHON -m ups.data.cli plan --lock "$TRAINING_LOCK" --cache "$CACHE" --reserve-bytes "$RESERVE_BYTES"
$PYTHON -m ups.data.cli stage --lock "$TRAINING_LOCK" --cache "$CACHE" --run-dir "$DATA_ROOT" --reserve-bytes "$RESERVE_BYTES" --report "$STAGE_REPORT"
$PYTHON -m ups.data.cli verify --lock "$TRAINING_LOCK" --cache "$CACHE"

$PYTHON - "$PLAN" <<'PY'
import hashlib,json,pathlib,subprocess,sys
p=json.load(open(sys.argv[1],encoding="utf-8"))
if p.get("mode") != "validation_only" or p.get("heldout_access") != "forbidden" or p.get("measurement_lock_access") != "forbidden": raise SystemExit("refusing non-validation D5 plan")
current=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
commit=p["bindings"]["source"]["implementation_commit"]
if current != commit and subprocess.run(["git","merge-base","--is-ancestor",commit,current]).returncode: raise SystemExit("D5 implementation commit is not in remote HEAD")
for relative,expected in p["bindings"]["source"]["files"].items():
    if hashlib.sha256(pathlib.Path(relative).read_bytes()).hexdigest() != expected: raise SystemExit(f"source binding mismatch: {relative}")
PY

$PYTHON scripts/run_strat_v1_shared_tier_b.py \
  --training-lock "$TRAINING_LOCK" --data-root "$DATA_ROOT" --config "$CONFIG" \
  --output-dir "$OUTPUT_DIR" --plan-path "$PLAN" --plan-sha256 "$plan_sha" \
  --device cuda "${resume_arg[@]}"
$PYTHON scripts/materialize_strat_v1_shared_tier_b.py \
  --plan "$PLAN" --summary "$OUTPUT_DIR/summary.json" --output "$RESULT"

stamp=$(date -u +%Y%m%dT%H%M%SZ)
archive_name="strat_v1_shared_tier_b_${stamp}.tar.gz"
archive_path="/tmp/${archive_name}"
tar -czf "$archive_path" "$PLAN" "$CONFIG" "$STAGE_REPORT" "$OUTPUT_DIR" "$RESULT"
digest=$(sha256sum "$archive_path" | awk '{print $1}')
remote_key="${ARTIFACT_PREFIX%/}/immutable/sha256/${digest}/${archive_name}"
rclone copyto "$archive_path" "UPSB2:${B2_BUCKET}/${remote_key}"
remote_digest=$(rclone cat "UPSB2:${B2_BUCKET}/${remote_key}" | sha256sum | awk '{print $1}')
[ "$remote_digest" = "$digest" ] || { echo "Immutable D5 artifact read-back mismatch" >&2; exit 1; }
rclone purge "UPSB2:${B2_BUCKET}/${temporary_prefix}" >/dev/null 2>&1 || true
success=1
trap - EXIT
echo "Published immutable D5 artifact: b2://${B2_BUCKET}/${remote_key}"
