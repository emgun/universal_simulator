#!/usr/bin/env bash
set -euo pipefail

for assignment in "$@"; do
  case "$assignment" in *=*) export "$assignment" ;; "") ;; *) echo "Unexpected argument: $assignment" >&2; exit 2 ;; esac
done

DRY_RUN=${DRY_RUN:-1}
PYTHON=${PYTHON:-python}
PLAN=${PLAN:-docs/research/artifacts/strat_v1_darcy_conditioned_uno_ablation_plan.json}
TRAINING_LOCK=${TRAINING_LOCK:-docs/data/releases/strat-v1/universal/9d43d283f04f5b8d17cf6126ad189075c53307e715d7d4f61af440c2fed155c1/training.lock.json}
SCRATCH_ROOT=${SCRATCH_ROOT:-reports/research/darcy_conditioned_uno_scratch}
CACHE=${CACHE:-$SCRATCH_ROOT/cache}
DATA_ROOT=${DATA_ROOT:-$SCRATCH_ROOT/data}
OUTPUT_DIR=${OUTPUT_DIR:-reports/research/darcy_conditioned_uno_ablation}
STAGE_REPORT=${STAGE_REPORT:-reports/research/darcy_conditioned_uno_stage.json}
RESULT=${RESULT:-reports/research/darcy_conditioned_uno_result.json}
RUNNER_LOG=${RUNNER_LOG:-$SCRATCH_ROOT/remote_runner.log}
ARTIFACT_PREFIX=${ARTIFACT_PREFIX:-remote-runs/darcy-conditioned-uno-ablation}
RESERVE_BYTES=${RESERVE_BYTES:-8589934592}
NEURALOPERATOR_VERSION=${NEURALOPERATOR_VERSION:-2.0.0}

case "$DRY_RUN" in 0|1) ;; *) echo "DRY_RUN must be 0 or 1" >&2; exit 2 ;; esac
if [ "$DRY_RUN" -eq 1 ]; then
  echo "DRY RUN: validation-only Darcy conditioned UNO D4; held-out data is inaccessible."
  echo "$PYTHON -m ups.data.cli stage --lock $TRAINING_LOCK --cache $CACHE --run-dir $DATA_ROOT --report $STAGE_REPORT"
  echo "$PYTHON executes the exact plan command, adding --resume only after plan-bound hydration."
  echo "$PYTHON scripts/materialize_darcy_conditioned_uno_ablation.py --plan $PLAN --summary $OUTPUT_DIR/summary.json --output $RESULT"
  exit 0
fi

test -f "$PLAN"
test ! -e "$RESULT"
plan_sha=$($PYTHON - "$PLAN" <<'PY'
import hashlib,json,sys
def canonical(v): return hashlib.sha256(json.dumps(v,allow_nan=False,ensure_ascii=False,separators=(",",":"),sort_keys=True).encode()).hexdigest()
p=json.load(open(sys.argv[1],encoding="utf-8"))
if p.get("schema_version") != 2: raise SystemExit("D4 plan schema must be 2")
if p.get("mode") != "validation_only" or p.get("heldout_access") != "forbidden" or p.get("measurement_lock_access") != "forbidden": raise SystemExit("refusing non-validation plan")
v=p.get("plan_sha256")
if v != canonical({k:x for k,x in p.items() if k != "plan_sha256"}): raise SystemExit("invalid plan self hash")
c=p.get("command")
if not isinstance(c,list) or p.get("command_sha256") != canonical(c): raise SystemExit("invalid command hash")
if any("test" in str(x).lower() for x in c): raise SystemExit("refusing test-capable command")
print(v)
PY
)
plan_file_sha=$(sha256sum "$PLAN" | awk '{print $1}')
: "${B2_KEY_ID:?Set B2_KEY_ID}"
: "${B2_APP_KEY:?Set B2_APP_KEY}"
: "${B2_BUCKET:?Set B2_BUCKET}"
command -v rclone >/dev/null || { echo "rclone is required" >&2; exit 1; }
if [ -n "${B2_S3_ENDPOINT:-}" ] || [ -n "${B2_S3_REGION:-}" ]; then
  export RCLONE_CONFIG_UPSB2_TYPE=s3 RCLONE_CONFIG_UPSB2_PROVIDER=Other
  export RCLONE_CONFIG_UPSB2_ACCESS_KEY_ID="$B2_KEY_ID" RCLONE_CONFIG_UPSB2_SECRET_ACCESS_KEY="$B2_APP_KEY"
  [ -n "${B2_S3_ENDPOINT:-}" ] && export RCLONE_CONFIG_UPSB2_ENDPOINT="$B2_S3_ENDPOINT"
  [ -n "${B2_S3_REGION:-}" ] && export RCLONE_CONFIG_UPSB2_REGION="$B2_S3_REGION"
else
  export RCLONE_CONFIG_UPSB2_TYPE=b2 RCLONE_CONFIG_UPSB2_ACCOUNT="$B2_KEY_ID" RCLONE_CONFIG_UPSB2_KEY="$B2_APP_KEY"
fi
temporary_prefix="${ARTIFACT_PREFIX%/}/resumable/${plan_sha}"
success=0
preserve_resume() {
  status=$?
  if [ -f "$RUNNER_LOG" ] && [ -d "$OUTPUT_DIR" ]; then cp "$RUNNER_LOG" "$OUTPUT_DIR/remote_runner.log"; fi
  if [ "$success" -ne 1 ] && [ -d "$OUTPUT_DIR" ]; then
    rclone sync "$OUTPUT_DIR" "UPSB2:${B2_BUCKET}/${temporary_prefix}/output" --exclude 'summary.json' || true
    echo "Preserved resumable D4 checkpoints: b2://${B2_BUCKET}/${temporary_prefix}/output" >&2
  fi
  exit "$status"
}
trap preserve_resume EXIT

mkdir -p "$CACHE" "$DATA_ROOT" "$(dirname "$STAGE_REPORT")"
resume_arg=()
resume_listing=$(rclone lsf "UPSB2:${B2_BUCKET}/${temporary_prefix}/output" --files-only --recursive 2>/dev/null || true)
if [ -n "$resume_listing" ]; then
  mkdir -p "$OUTPUT_DIR"
  rclone sync "UPSB2:${B2_BUCKET}/${temporary_prefix}/output" "$OUTPUT_DIR"
  test -f "$OUTPUT_DIR/run_identity.json"
  checkpoint=$(find "$OUTPUT_DIR/checkpoints" -type f -name '*.pt' -print -quit)
  [ -n "$checkpoint" ] && test -f "${checkpoint}.record.json"
  resume_arg=(--resume)
fi

$PYTHON -m pip install "neuraloperator==$NEURALOPERATOR_VERSION"
$PYTHON -m ups.data.cli plan --lock "$TRAINING_LOCK" --cache "$CACHE" --reserve-bytes "$RESERVE_BYTES"
$PYTHON -m ups.data.cli stage --lock "$TRAINING_LOCK" --cache "$CACHE" --run-dir "$DATA_ROOT" --reserve-bytes "$RESERVE_BYTES" --report "$STAGE_REPORT"
$PYTHON -m ups.data.cli verify --lock "$TRAINING_LOCK" --cache "$CACHE"
[ "$(sha256sum "$PLAN" | awk '{print $1}')" = "$plan_file_sha" ] || { echo "D4 plan bytes changed" >&2; exit 1; }

$PYTHON - "$PLAN" "${resume_arg[@]}" 2>&1 <<'PY' | tee "$RUNNER_LOG"
import hashlib,json,pathlib,subprocess,sys
p=json.load(open(sys.argv[1],encoding="utf-8"))
if p.get("mode") != "validation_only" or p.get("heldout_access") != "forbidden" or p.get("measurement_lock_access") != "forbidden": raise SystemExit("refusing non-validation plan")
current=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
commit=p["bindings"]["source"]["implementation_commit"]
if current != commit and subprocess.run(["git","merge-base","--is-ancestor",commit,current]).returncode: raise SystemExit("implementation commit is not in remote HEAD")
for relative,expected in p["bindings"]["source"]["files"].items():
    if hashlib.sha256(pathlib.Path(relative).read_bytes()).hexdigest() != expected: raise SystemExit(f"source binding mismatch: {relative}")
command=p["command"][:]
if any("test" in str(x).lower() for x in command): raise SystemExit("refusing test-capable command")
if len(sys.argv) > 2: command.append("--resume")
subprocess.run(command,check=True)
PY
cp "$RUNNER_LOG" "$OUTPUT_DIR/remote_runner.log"

$PYTHON scripts/materialize_darcy_conditioned_uno_ablation.py --plan "$PLAN" --summary "$OUTPUT_DIR/summary.json" --output "$RESULT"
stamp=$(date -u +%Y%m%dT%H%M%SZ)
archive_name="darcy_conditioned_uno_ablation_${stamp}.tar.gz"
archive_path="/tmp/$archive_name"
tar -czf "$archive_path" "$PLAN" "$STAGE_REPORT" "$OUTPUT_DIR" "$RESULT"
digest=$(sha256sum "$archive_path" | awk '{print $1}')
remote_key="${ARTIFACT_PREFIX%/}/immutable/sha256/${digest}/${archive_name}"
rclone copyto "$archive_path" "UPSB2:${B2_BUCKET}/${remote_key}"
remote_digest=$(rclone cat "UPSB2:${B2_BUCKET}/${remote_key}" | sha256sum | awk '{print $1}')
[ "$remote_digest" = "$digest" ] || { echo "Immutable D4 artifact read-back hash mismatch" >&2; exit 1; }
rclone purge "UPSB2:${B2_BUCKET}/${temporary_prefix}" >/dev/null 2>&1 || true
success=1
trap - EXIT
echo "Published immutable Darcy D4 artifact: b2://${B2_BUCKET}/${remote_key}"
