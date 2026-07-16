#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python}
PLAN=${PLAN:-docs/research/artifacts/strat_v1_modular_shared_trunk_plan_v4.json}
CONFIG=${CONFIG:-configs/d6_strat_v1_modular_shared_trunk.yaml}
TRAINING_LOCK=${TRAINING_LOCK:-docs/data/releases/strat-v1/universal/9d43d283f04f5b8d17cf6126ad189075c53307e715d7d4f61af440c2fed155c1/training.lock.json}
CACHE=${CACHE:-reports/research/strat_v1_modular_shared_trunk_scratch/cache}
DATA_ROOT=${DATA_ROOT:-reports/research/strat_v1_modular_shared_trunk_scratch/data}
STAGE_REPORT=${STAGE_REPORT:-reports/research/strat_v1_modular_shared_trunk_stage.json}
OUTPUT_DIR=${OUTPUT_DIR:-reports/research/strat_v1_modular_shared_trunk}
RUN_LOG=${RUN_LOG:-reports/research/strat_v1_modular_shared_trunk.remote.log}
TRANSFER_MANIFEST=${TRANSFER_MANIFEST:-/tmp/d6_transfer_manifest.json}
TRANSFER_MANIFEST_URL_B64=${TRANSFER_MANIFEST_URL_B64:-}
RESULT=${RESULT:-reports/research/strat_v1_modular_shared_trunk_result.json}
ARTIFACT_PREFIX=${ARTIFACT_PREFIX:-remote-runs/strat-v1-modular-shared-trunk}
RESERVE_BYTES=${RESERVE_BYTES:-8589934592}
DRY_RUN=${DRY_RUN:-1}

for override in "$@"; do
  case "$override" in
    DRY_RUN=0|DRY_RUN=1) DRY_RUN=${override#DRY_RUN=} ;;
    ARTIFACT_PREFIX=*) ARTIFACT_PREFIX=${override#ARTIFACT_PREFIX=} ;;
    TRANSFER_MANIFEST_URL_B64=*) TRANSFER_MANIFEST_URL_B64=${override#TRANSFER_MANIFEST_URL_B64=} ;;
    *) echo "Unsupported D6 remote override: $override" >&2; exit 2 ;;
  esac
done
case "$ARTIFACT_PREFIX" in
  ""|/*|*..*) echo "ARTIFACT_PREFIX must be a nonempty relative B2 prefix" >&2; exit 2 ;;
esac

if [ "$DRY_RUN" = 1 ]; then
  echo "$PYTHON -m ups.data.cli stage --lock $TRAINING_LOCK --cache $CACHE --run-dir $DATA_ROOT"
  echo "$PYTHON scripts/run_strat_v1_modular_shared_trunk.py --training-lock $TRAINING_LOCK --data-root $DATA_ROOT --config $CONFIG --output-dir $OUTPUT_DIR --plan-path $PLAN --plan-sha256 <from-plan> --stage-report $STAGE_REPORT --device cuda"
  echo "$PYTHON scripts/materialize_strat_v1_modular_shared_trunk.py --plan $PLAN --summary $OUTPUT_DIR/summary.json --stage-report $STAGE_REPORT --output $RESULT"
  exit 0
fi

: "${TRANSFER_MANIFEST_URL_B64:?Set TRANSFER_MANIFEST_URL_B64 to the short-lived D6 control capability}"
TRANSFER_MANIFEST_URL=$($PYTHON - "$TRANSFER_MANIFEST_URL_B64" <<'PY'
import base64,sys
value=sys.argv[1]
value += "=" * (-len(value) % 4)
print(base64.urlsafe_b64decode(value).decode("utf-8"))
PY
)
$PYTHON scripts/d5_presigned_io.py fetch-manifest \
  --url "$TRANSFER_MANIFEST_URL" --output "$TRANSFER_MANIFEST"
unset TRANSFER_MANIFEST_URL TRANSFER_MANIFEST_URL_B64
export UPS_B2_PRESIGNED_URLS_FILE="$TRANSFER_MANIFEST"

plan_sha=$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["plan_sha256"])' "$PLAN")
$PYTHON - "$TRANSFER_MANIFEST" "$plan_sha" "$ARTIFACT_PREFIX" <<'PY'
import json,sys
p=json.load(open(sys.argv[1],encoding="utf-8"))
if p.get("plan_sha256") != sys.argv[2]: raise SystemExit("transfer manifest plan mismatch")
if p.get("artifact_prefix", "").rstrip("/") != sys.argv[3].rstrip("/"): raise SystemExit("transfer manifest artifact prefix mismatch")
PY
success=0
preserve_resume() {
  status=$?
  if [ "$success" -ne 1 ]; then
    $PYTHON scripts/d5_presigned_io.py preserve \
      --manifest "$TRANSFER_MANIFEST" --output-dir "$OUTPUT_DIR" --run-log "$RUN_LOG" || true
    echo "Attempted credential-free preservation of the sealed D6 resume/log slots." >&2
  fi
  exit "$status"
}
trap preserve_resume EXIT

resume_arg=()
set +e
$PYTHON scripts/d5_presigned_io.py fetch-resume \
  --manifest "$TRANSFER_MANIFEST" --output-dir "$OUTPUT_DIR"
resume_status=$?
set -e
if [ "$resume_status" -eq 0 ]; then
  resume_arg=(--resume)
elif [ "$resume_status" -ne 3 ]; then
  echo "D6 resume capability failed closed." >&2
  exit "$resume_status"
fi

$PYTHON -m pip install -e . --no-deps
$PYTHON -m ups.data.cli plan --lock "$TRAINING_LOCK" --cache "$CACHE" --reserve-bytes "$RESERVE_BYTES"
$PYTHON -m ups.data.cli stage --lock "$TRAINING_LOCK" --cache "$CACHE" --run-dir "$DATA_ROOT" --reserve-bytes "$RESERVE_BYTES" --report "$STAGE_REPORT"
$PYTHON -m ups.data.cli verify --lock "$TRAINING_LOCK" --cache "$CACHE"
$PYTHON - "$STAGE_REPORT" <<'PY'
import json,sys
from ups.data.manifests import canonical_sha256
path=sys.argv[1]
report=json.load(open(path,encoding="utf-8"))
report["artifact_sha256"]=canonical_sha256(report)
open(path,"w",encoding="utf-8").write(json.dumps(report,indent=2,sort_keys=True)+"\n")
PY

$PYTHON - "$PLAN" <<'PY'
import hashlib,json,pathlib,subprocess,sys
p=json.load(open(sys.argv[1],encoding="utf-8"))
if p.get("mode") != "validation_only" or p.get("heldout_access") != "forbidden" or p.get("measurement_lock_access") != "forbidden": raise SystemExit("refusing non-validation D6 plan")
current=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
commit=p["bindings"]["source"]["implementation_commit"]
if current != commit and subprocess.run(["git","merge-base","--is-ancestor",commit,current]).returncode: raise SystemExit("D6 implementation commit is not in remote HEAD")
for relative,expected in p["bindings"]["source"]["files"].items():
    if hashlib.sha256(pathlib.Path(relative).read_bytes()).hexdigest() != expected: raise SystemExit(f"source binding mismatch: {relative}")
PY

mkdir -p "$(dirname "$RUN_LOG")"
$PYTHON scripts/run_strat_v1_modular_shared_trunk.py \
  --training-lock "$TRAINING_LOCK" --data-root "$DATA_ROOT" --config "$CONFIG" \
  --output-dir "$OUTPUT_DIR" --plan-path "$PLAN" --plan-sha256 "$plan_sha" \
  --stage-report "$STAGE_REPORT" --device cuda "${resume_arg[@]}" 2>&1 | tee "$RUN_LOG"
$PYTHON scripts/materialize_strat_v1_modular_shared_trunk.py \
  --plan "$PLAN" --summary "$OUTPUT_DIR/summary.json" \
  --stage-report "$STAGE_REPORT" --output "$RESULT"

stamp=$(date -u +%Y%m%dT%H%M%SZ)
archive_name="strat_v1_modular_shared_trunk_${stamp}.tar.gz"
archive_path="/tmp/${archive_name}"
tar -czf "$archive_path" "$PLAN" "$CONFIG" "$STAGE_REPORT" "$OUTPUT_DIR" "$RESULT" "$RUN_LOG"
digest=$(sha256sum "$archive_path" | awk '{print $1}')
$PYTHON scripts/d5_presigned_io.py publish \
  --manifest "$TRANSFER_MANIFEST" --archive "$archive_path"
success=1
trap - EXIT
echo "Uploaded verified D6 ingress artifact: sha256=${digest}"
echo "D6 ingress upload complete; trusted local finalization is required."
