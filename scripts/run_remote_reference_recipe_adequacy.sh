#!/usr/bin/env bash
set -euo pipefail

# Validation-only remote workflow for the bounded FNO/UNO reference-recipe gate.
#
# The safe default is a preview. Vast launchers must explicitly pass DRY_RUN=0.
# The only data authority accepted here is the frozen train+valid lock; neither a
# measurement lock nor a test path is configurable through this wrapper.

apply_cli_assignments() {
  local assignment
  for assignment in "$@"; do
    case "$assignment" in
      *=*) export "$assignment" ;;
      "") ;;
      *)
        echo "Unexpected argument '${assignment}'. Pass KEY=VALUE assignments." >&2
        exit 2
        ;;
    esac
  done
}

print_command() {
  printf ' %q' "$@"
  echo
}

run_or_echo() {
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "DRY_RUN command:"
    print_command "$@"
  else
    "$@"
  fi
}

configure_b2_rclone() {
  : "${B2_KEY_ID:?Set B2_KEY_ID for frozen data staging and artifact publication}"
  : "${B2_APP_KEY:?Set B2_APP_KEY for frozen data staging and artifact publication}"
  : "${B2_BUCKET:?Set B2_BUCKET for frozen data staging and artifact publication}"
  if ! command -v rclone >/dev/null 2>&1; then
    echo "rclone is required for lock-based B2 staging and immutable publication." >&2
    exit 1
  fi
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
}

package_and_publish() {
  local stamp archive_name archive_path digest remote_key
  stamp=$(date -u +%Y%m%dT%H%M%SZ)
  archive_name="reference_recipe_adequacy_${stamp}.tar.gz"
  archive_path="/tmp/${archive_name}"

  # PIPELINE_ROOT contains the stage report, both plans, selection, and the
  # optional final artifact. OUTPUT_ROOT contains only summaries/checkpoints.
  tar -czf "$archive_path" "$PIPELINE_ROOT" "$OUTPUT_ROOT"
  digest=$(sha256sum "$archive_path" | awk '{print $1}')
  remote_key="${ARTIFACT_PREFIX%/}/immutable/sha256/${digest}/${archive_name}"
  configure_b2_rclone
  rclone copyto "$archive_path" "UPSB2:${B2_BUCKET}/${remote_key}"
  printf 'Published immutable adequacy artifact: b2://%s/%s\n' "$B2_BUCKET" "$remote_key"
}

apply_cli_assignments "$@"

DRY_RUN=${DRY_RUN:-1}
PYTHON=${PYTHON:-python}
NEURALOPERATOR_VERSION=${NEURALOPERATOR_VERSION:-2.0.0}
SCRATCH_ROOT=${SCRATCH_ROOT:-/workspace/reference_recipe_scratch}
DATA_CACHE=${DATA_CACHE:-$SCRATCH_ROOT/cache}
DATA_ROOT=${DATA_ROOT:-$SCRATCH_ROOT/data}
PIPELINE_ROOT=${PIPELINE_ROOT:-reports/research/reference_recipe_adequacy_remote}
OUTPUT_ROOT=${OUTPUT_ROOT:-reports/research/reference_recipe_adequacy}
TRAINING_LOCK=${TRAINING_LOCK:-docs/data/releases/strat-v1/universal/9d43d283f04f5b8d17cf6126ad189075c53307e715d7d4f61af440c2fed155c1/training.lock.json}
DISCOVERY_PLAN=${DISCOVERY_PLAN:-$PIPELINE_ROOT/discovery_plan.json}
SELECTION_ARTIFACT=${SELECTION_ARTIFACT:-$PIPELINE_ROOT/selection.json}
CONFIRMATION_PLAN=${CONFIRMATION_PLAN:-$PIPELINE_ROOT/confirmation_plan.json}
FINAL_ARTIFACT=${FINAL_ARTIFACT:-$PIPELINE_ROOT/claim_grade_recipe.json}
STAGE_REPORT=${STAGE_REPORT:-$PIPELINE_ROOT/training_stage_report.json}
FINALIZER_SCRIPT=${FINALIZER_SCRIPT:-scripts/finalize_reference_recipe_adequacy.py}
ARTIFACT_PREFIX=${ARTIFACT_PREFIX:-remote-runs/reference-recipe-adequacy}
RESERVE_BYTES=${RESERVE_BYTES:-8589934592}

case "$DRY_RUN" in 0|1) ;; *) echo "DRY_RUN must be 0 or 1." >&2; exit 2 ;; esac

if [ "$DRY_RUN" -eq 1 ]; then
  echo "DRY_RUN=1: validation-only preview; no packages, data, training, or B2 objects will be changed."
  echo "Boundary: frozen training lock only; test and measurement data are never staged or read."
  run_or_echo "$PYTHON" -m pip install "neuraloperator==${NEURALOPERATOR_VERSION}"
  run_or_echo "$PYTHON" -m ups.data.cli plan --lock "$TRAINING_LOCK" --cache "$DATA_CACHE" --reserve-bytes "$RESERVE_BYTES"
  run_or_echo "$PYTHON" -m ups.data.cli stage --lock "$TRAINING_LOCK" --cache "$DATA_CACHE" --run-dir "$DATA_ROOT" --reserve-bytes "$RESERVE_BYTES" --report "$STAGE_REPORT"
  run_or_echo "$PYTHON" scripts/plan_reference_recipe_adequacy.py --training-lock "$TRAINING_LOCK" --data-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT" --neuraloperator-version "$NEURALOPERATOR_VERSION" --device cuda --output-plan "$DISCOVERY_PLAN"
  run_or_echo "$PYTHON" scripts/execute_reference_recipe_adequacy.py --plan "$DISCOVERY_PLAN" --run-set discovery --confirm-validation-only
  echo "DRY_RUN conditional: materialize selection; stop cleanly and package if no architecture is eligible."
  echo "DRY_RUN conditional: otherwise bind confirmation to the selection, execute seeds 29/43, and finalize."
  echo "DRY_RUN publication: one content-addressed tarball under ${ARTIFACT_PREFIX%/}/immutable/sha256/<sha256>/."
  exit 0
fi

mkdir -p "$PIPELINE_ROOT" "$OUTPUT_ROOT" "$DATA_CACHE" "$DATA_ROOT"
configure_b2_rclone

"$PYTHON" -m pip install "neuraloperator==${NEURALOPERATOR_VERSION}"
"$PYTHON" -c 'import importlib.metadata, sys; expected=sys.argv[1]; observed=importlib.metadata.version("neuraloperator"); assert observed == expected, f"neuraloperator version {observed} != {expected}"' "$NEURALOPERATOR_VERSION"

# The control plane verifies the lock, stages only its train/valid objects into
# ephemeral local scratch, and persists checksums/roles in STAGE_REPORT.
"$PYTHON" -m ups.data.cli plan \
  --lock "$TRAINING_LOCK" --cache "$DATA_CACHE" --reserve-bytes "$RESERVE_BYTES"
"$PYTHON" -m ups.data.cli stage \
  --lock "$TRAINING_LOCK" --cache "$DATA_CACHE" --run-dir "$DATA_ROOT" \
  --reserve-bytes "$RESERVE_BYTES" --report "$STAGE_REPORT"
"$PYTHON" -m ups.data.cli verify --lock "$TRAINING_LOCK" --cache "$DATA_CACHE"

"$PYTHON" scripts/plan_reference_recipe_adequacy.py \
  --training-lock "$TRAINING_LOCK" --data-root "$DATA_ROOT" \
  --output-root "$OUTPUT_ROOT" --neuraloperator-version "$NEURALOPERATOR_VERSION" \
  --device cuda --output-plan "$DISCOVERY_PLAN"
"$PYTHON" scripts/execute_reference_recipe_adequacy.py \
  --plan "$DISCOVERY_PLAN" --run-set discovery --confirm-validation-only

"$PYTHON" scripts/materialize_reference_recipe_adequacy.py \
  --plan "$DISCOVERY_PLAN" \
  --summary "$OUTPUT_ROOT/r0_strat_v1_1_fno_all_e48_s17_discovery_val/summary.json" \
  --summary "$OUTPUT_ROOT/r0_strat_v1_1_uno_all_e48_s17_discovery_val/summary.json" \
  --output "$SELECTION_ARTIFACT"

if [ "$("$PYTHON" -c 'import json,sys; print("1" if json.load(open(sys.argv[1]))["no_eligible_architecture"] else "0")' "$SELECTION_ARTIFACT")" -eq 1 ]; then
  echo "No eligible validation architecture; stopping before confirmation and held-out access."
  package_and_publish
  exit 0
fi

"$PYTHON" scripts/plan_reference_recipe_adequacy.py \
  --training-lock "$TRAINING_LOCK" --data-root "$DATA_ROOT" \
  --output-root "$OUTPUT_ROOT" --neuraloperator-version "$NEURALOPERATOR_VERSION" \
  --device cuda --discovery-plan "$DISCOVERY_PLAN" \
  --selection-artifact "$SELECTION_ARTIFACT" --output-plan "$CONFIRMATION_PLAN"
"$PYTHON" scripts/execute_reference_recipe_adequacy.py \
  --plan "$CONFIRMATION_PLAN" --run-set confirmation --confirm-validation-only

selected_architecture=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selection"]["architecture"])' "$SELECTION_ARTIFACT")
selected_epoch=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selection"]["epoch"])' "$SELECTION_ARTIFACT")
discovery_summary="$OUTPUT_ROOT/r0_strat_v1_1_${selected_architecture}_all_e48_s17_discovery_val/summary.json"
confirmation_29="$OUTPUT_ROOT/r0_strat_v1_1_${selected_architecture}_all_e${selected_epoch}_s29_confirmation_val/summary.json"
confirmation_43="$OUTPUT_ROOT/r0_strat_v1_1_${selected_architecture}_all_e${selected_epoch}_s43_confirmation_val/summary.json"

if [ -f "$FINALIZER_SCRIPT" ]; then
  "$PYTHON" "$FINALIZER_SCRIPT" \
    --plan "$CONFIRMATION_PLAN" --selection-artifact "$SELECTION_ARTIFACT" \
    --discovery-summary "$discovery_summary" \
    --confirmation-summary "$confirmation_29" \
    --confirmation-summary "$confirmation_43" \
    --output "$FINAL_ARTIFACT"
else
  echo "Confirmation completed, but finalizer is unavailable at ${FINALIZER_SCRIPT}." >&2
  exit 1
fi

package_and_publish
