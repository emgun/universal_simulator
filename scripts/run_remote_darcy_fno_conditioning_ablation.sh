#!/usr/bin/env bash
set -euo pipefail

for assignment in "$@"; do
  case "$assignment" in *=*) export "$assignment" ;; "") ;; *) echo "Unexpected argument: $assignment" >&2; exit 2 ;; esac
done

DRY_RUN=${DRY_RUN:-1}
PYTHON=${PYTHON:-python}
PLAN=${PLAN:-docs/research/artifacts/strat_v1_1_darcy_fno_conditioning_ablation_plan.json}
TRAINING_LOCK=${TRAINING_LOCK:-docs/data/releases/strat-v1/universal/9d43d283f04f5b8d17cf6126ad189075c53307e715d7d4f61af440c2fed155c1/training.lock.json}
SCRATCH_ROOT=${SCRATCH_ROOT:-reports/research/darcy_conditioning_scratch}
CACHE=${CACHE:-$SCRATCH_ROOT/cache}
DATA_ROOT=${DATA_ROOT:-$SCRATCH_ROOT/data}
OUTPUT_DIR=${OUTPUT_DIR:-reports/research/darcy_fno_conditioning_ablation}
STAGE_REPORT=${STAGE_REPORT:-reports/research/darcy_fno_conditioning_stage.json}
RESULT=${RESULT:-reports/research/darcy_fno_conditioning_result.json}
ARTIFACT_PREFIX=${ARTIFACT_PREFIX:-remote-runs/darcy-fno-conditioning-ablation}
RESERVE_BYTES=${RESERVE_BYTES:-8589934592}
NEURALOPERATOR_VERSION=${NEURALOPERATOR_VERSION:-2.0.0}

case "$DRY_RUN" in 0|1) ;; *) echo "DRY_RUN must be 0 or 1" >&2; exit 2 ;; esac

if [ "$DRY_RUN" -eq 1 ]; then
  echo "DRY RUN: validation-only Darcy U/K ablation; no test or measurement lock is accepted."
  echo "$PYTHON -m pip install neuraloperator==$NEURALOPERATOR_VERSION"
  echo "$PYTHON -m ups.data.cli stage --lock $TRAINING_LOCK --cache $CACHE --run-dir $DATA_ROOT --report $STAGE_REPORT"
  echo "$PYTHON executes the canonical command in $PLAN"
  echo "$PYTHON scripts/materialize_darcy_fno_conditioning_ablation.py --plan $PLAN --summary $OUTPUT_DIR/summary.json --output $RESULT"
  exit 0
fi

: "${B2_KEY_ID:?Set B2_KEY_ID}"
: "${B2_APP_KEY:?Set B2_APP_KEY}"
: "${B2_BUCKET:?Set B2_BUCKET}"
command -v rclone >/dev/null || { echo "rclone is required" >&2; exit 1; }

if [ -n "${B2_S3_ENDPOINT:-}" ] || [ -n "${B2_S3_REGION:-}" ]; then
  export RCLONE_CONFIG_UPSB2_TYPE=s3
  export RCLONE_CONFIG_UPSB2_PROVIDER=Other
  export RCLONE_CONFIG_UPSB2_ACCESS_KEY_ID="$B2_KEY_ID"
  export RCLONE_CONFIG_UPSB2_SECRET_ACCESS_KEY="$B2_APP_KEY"
  [ -n "${B2_S3_ENDPOINT:-}" ] && export RCLONE_CONFIG_UPSB2_ENDPOINT="$B2_S3_ENDPOINT"
  [ -n "${B2_S3_REGION:-}" ] && export RCLONE_CONFIG_UPSB2_REGION="$B2_S3_REGION"
else
  export RCLONE_CONFIG_UPSB2_TYPE=b2
  export RCLONE_CONFIG_UPSB2_ACCOUNT="$B2_KEY_ID"
  export RCLONE_CONFIG_UPSB2_KEY="$B2_APP_KEY"
fi

test -f "$PLAN"
test ! -e "$OUTPUT_DIR"
test ! -e "$RESULT"
mkdir -p "$CACHE" "$DATA_ROOT" "$(dirname "$STAGE_REPORT")"
"$PYTHON" -m pip install "neuraloperator==$NEURALOPERATOR_VERSION"
"$PYTHON" -m ups.data.cli plan --lock "$TRAINING_LOCK" --cache "$CACHE" --reserve-bytes "$RESERVE_BYTES"
"$PYTHON" -m ups.data.cli stage --lock "$TRAINING_LOCK" --cache "$CACHE" --run-dir "$DATA_ROOT" --reserve-bytes "$RESERVE_BYTES" --report "$STAGE_REPORT"
"$PYTHON" -m ups.data.cli verify --lock "$TRAINING_LOCK" --cache "$CACHE"

"$PYTHON" - "$PLAN" <<'PY'
import hashlib, json, pathlib, subprocess, sys
plan = json.load(open(sys.argv[1], encoding="utf-8"))
if plan.get("mode") != "validation_only" or plan.get("heldout_access") != "forbidden":
    raise SystemExit("refusing non-validation plan")
binding = plan.get("bindings", {}).get("source", {})
current = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
if subprocess.run(["git", "merge-base", "--is-ancestor", binding.get("implementation_commit", ""), current]).returncode:
    raise SystemExit("implementation commit is not an ancestor of remote HEAD")
for relative, expected in binding.get("files", {}).items():
    observed = hashlib.sha256(pathlib.Path(relative).read_bytes()).hexdigest()
    if observed != expected:
        raise SystemExit(f"source binding mismatch: {relative}")
command = plan.get("command")
if not isinstance(command, list) or any("test" in str(item).lower() for item in command):
    raise SystemExit("refusing ambiguous or test-capable command")
subprocess.run(command, check=True)
PY

"$PYTHON" scripts/materialize_darcy_fno_conditioning_ablation.py \
  --plan "$PLAN" --summary "$OUTPUT_DIR/summary.json" --output "$RESULT"

stamp=$(date -u +%Y%m%dT%H%M%SZ)
archive_name="darcy_fno_conditioning_ablation_${stamp}.tar.gz"
archive_path="/tmp/$archive_name"
tar -czf "$archive_path" "$PLAN" "$STAGE_REPORT" "$OUTPUT_DIR" "$RESULT"
digest=$(sha256sum "$archive_path" | awk '{print $1}')
remote_key="${ARTIFACT_PREFIX%/}/immutable/sha256/${digest}/${archive_name}"
rclone copyto "$archive_path" "UPSB2:${B2_BUCKET}/${remote_key}"
echo "Published immutable Darcy ablation artifact: b2://${B2_BUCKET}/${remote_key}"
