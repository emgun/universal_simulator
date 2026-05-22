#!/usr/bin/env bash
set -euo pipefail

# Remote official Advection hydration entrypoint.
#
# This wrapper matches the Vast launcher contract, which invokes remote scripts
# as `bash <script> KEY=VALUE ...`.

apply_cli_assignments() {
  local assignment
  for assignment in "$@"; do
    case "$assignment" in
      *=*) export "$assignment" ;;
      "")
        ;;
      *)
        echo "Unexpected argument '${assignment}'. Pass KEY=VALUE assignments." >&2
        exit 2
        ;;
    esac
  done
}

apply_cli_assignments "$@"

PLAN_JSON=${PLAN_JSON:-reports/research/sota_loop/official_advection_hydration_plan.json}
VALIDATION_JSON=${VALIDATION_JSON:-reports/research/sota_loop/official_advection_hydration_plan_validation.json}
RUN_JSON=${RUN_JSON:-reports/research/sota_loop/official_advection_hydration_plan_run.json}
OBJECTIVE_STATUS_JSON=${OBJECTIVE_STATUS_JSON:-reports/research/sota_loop/transport_objective_status.json}
POST_VALIDATION_TEST_JSON=${POST_VALIDATION_TEST_JSON:-reports/research/sota_loop/official_hydrated_post_validation_test_run.json}
OFFICIAL_HYDRATED_GATE_JSON=${OFFICIAL_HYDRATED_GATE_JSON:-reports/research/sota_loop/official_hydrated_transport_shift_gate.json}
OFFICIAL_HYDRATED_TEST_LEDGER_JSON=${OFFICIAL_HYDRATED_TEST_LEDGER_JSON:-reports/research/sota_loop/official_hydrated_transport_shift_test_ledger.json}
SEQUENTIAL_HYDRATION_JSON=${SEQUENTIAL_HYDRATION_JSON:-reports/research/sota_loop/official_advection_sequential_hydration_run.json}
MIN_DOWNLOAD_BYTES=${MIN_DOWNLOAD_BYTES:-60000000000}
EXECUTE=${EXECUTE:-1}
EXECUTE_DOWNLOADS=${EXECUTE_DOWNLOADS:-1}
SEQUENTIAL_HYDRATION=${SEQUENTIAL_HYDRATION:-0}
SEQUENTIAL_CLEANUP_RAW=${SEQUENTIAL_CLEANUP_RAW:-1}
SEQUENTIAL_USE_EXISTING_RAW=${SEQUENTIAL_USE_EXISTING_RAW:-0}
RUN_POST_VALIDATION_TEST=${RUN_POST_VALIDATION_TEST:-0}
EXECUTE_TEST=${EXECUTE_TEST:-0}
PUBLISH_ARTIFACTS=${PUBLISH_ARTIFACTS:-0}
ARTIFACT_PREFIX=${ARTIFACT_PREFIX:-remote-runs/official-hydration}
ARTIFACT_NAME=${ARTIFACT_NAME:-}
INSTALL_RCLONE=${INSTALL_RCLONE:-1}

# Dataverse returns short-lived S3 URLs for official files. Resolve that
# redirect once so ranged downloads do not repeatedly depend on Darus DNS.
export PDEBENCH_DOWNLOAD_RESOLVE_REDIRECT=${PDEBENCH_DOWNLOAD_RESOLVE_REDIRECT:-1}
export PDEBENCH_DOWNLOAD_REDIRECT_RETRIES=${PDEBENCH_DOWNLOAD_REDIRECT_RETRIES:-8}

export OBJECTIVE_STATUS_JSON

ensure_artifact_rclone() {
  if command -v rclone >/dev/null 2>&1; then
    return 0
  fi
  if [ "$INSTALL_RCLONE" -ne 1 ]; then
    echo "rclone is required to publish official hydration artifacts; set INSTALL_RCLONE=1 or preinstall rclone." >&2
    exit 1
  fi
  if command -v apt-get >/dev/null 2>&1; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y rclone
    return 0
  fi
  echo "rclone is required to publish official hydration artifacts and automatic install is only implemented for apt-get hosts." >&2
  exit 1
}

configure_artifact_rclone() {
  : "${B2_KEY_ID:?Set B2_KEY_ID to publish official hydration artifacts}"
  : "${B2_APP_KEY:?Set B2_APP_KEY to publish official hydration artifacts}"
  : "${B2_BUCKET:?Set B2_BUCKET to publish official hydration artifacts}"
  ensure_artifact_rclone
  if [ -n "${B2_S3_ENDPOINT:-}" ] || [ -n "${B2_S3_REGION:-}" ]; then
    export RCLONE_CONFIG_UPSB2_TYPE=s3
    export RCLONE_CONFIG_UPSB2_PROVIDER=Other
    export RCLONE_CONFIG_UPSB2_ACCESS_KEY_ID="${B2_KEY_ID}"
    export RCLONE_CONFIG_UPSB2_SECRET_ACCESS_KEY="${B2_APP_KEY}"
    [ -n "${B2_S3_ENDPOINT:-}" ] && export RCLONE_CONFIG_UPSB2_ENDPOINT="${B2_S3_ENDPOINT}"
    [ -n "${B2_S3_REGION:-}" ] && export RCLONE_CONFIG_UPSB2_REGION="${B2_S3_REGION}"
  else
    export RCLONE_CONFIG_UPSB2_TYPE=b2
    export RCLONE_CONFIG_UPSB2_ACCOUNT="${B2_KEY_ID}"
    export RCLONE_CONFIG_UPSB2_KEY="${B2_APP_KEY}"
  fi
}

publish_official_hydration_artifacts() {
  local stamp artifact_name artifact_path remote_key
  local -a existing_paths=()
  stamp=$(date -u +%Y%m%dT%H%M%SZ)
  artifact_name=${ARTIFACT_NAME:-official_hydration_${stamp}.tar.gz}
  artifact_path="/tmp/${artifact_name}"
  remote_key="${ARTIFACT_PREFIX%/}/${artifact_name}"

  for path in \
    "$PLAN_JSON" \
    "$VALIDATION_JSON" \
    "$RUN_JSON" \
    "$SEQUENTIAL_HYDRATION_JSON" \
    "$OBJECTIVE_STATUS_JSON" \
    "$POST_VALIDATION_TEST_JSON" \
    "$OFFICIAL_HYDRATED_GATE_JSON" \
    "$OFFICIAL_HYDRATED_TEST_LEDGER_JSON"; do
    [ -f "$path" ] && existing_paths+=("$path")
  done

  if [ "${#existing_paths[@]}" -eq 0 ]; then
    echo "No official hydration report artifacts found to publish." >&2
    exit 1
  fi

  tar -czf "$artifact_path" "${existing_paths[@]}"
  configure_artifact_rclone
  rclone copyto "$artifact_path" "UPSB2:${B2_BUCKET}/${remote_key}"
  echo "Published official hydration artifacts: b2://${B2_BUCKET}/${remote_key}"
}

if [ ! -f "$PLAN_JSON" ]; then
  python scripts/plan_transport_official_hydration.py --output-json "$PLAN_JSON"
fi

args=(
  python scripts/run_transport_official_hydration_plan.py
  --plan-json "$PLAN_JSON"
  --validation-json "$VALIDATION_JSON"
  --min-download-bytes "$MIN_DOWNLOAD_BYTES"
  --output-json "$RUN_JSON"
)

if [ "$EXECUTE" -eq 1 ]; then
  args+=(--execute)
fi

if [ "$EXECUTE_DOWNLOADS" -eq 1 ]; then
  args+=(--execute-downloads)
fi

if [ "$SEQUENTIAL_HYDRATION" -eq 1 ]; then
  sequential_args=(
    python scripts/hydrate_official_advection_source_sequential.py
    --plan-json "$PLAN_JSON"
    --output-json "$SEQUENTIAL_HYDRATION_JSON"
    --overwrite
  )
  if [ "$EXECUTE" -eq 1 ]; then
    sequential_args+=(--execute)
  fi
  if [ "$EXECUTE_DOWNLOADS" -eq 1 ]; then
    sequential_args+=(--execute-downloads)
  fi
  if [ "$SEQUENTIAL_USE_EXISTING_RAW" -eq 1 ]; then
    sequential_args+=(--use-existing-raw)
  fi
  if [ "$SEQUENTIAL_CLEANUP_RAW" -eq 1 ]; then
    sequential_args+=(--cleanup-raw)
  fi
  "${sequential_args[@]}"

  shard_validate_args=(
    python scripts/run_transport_official_hydration_plan.py
    --plan-json "$PLAN_JSON"
    --validation-json "$VALIDATION_JSON"
    --min-download-bytes "$MIN_DOWNLOAD_BYTES"
    --output-json "$RUN_JSON"
    --stage shard
    --stage validate
    --stage audit
  )
  if [ "$EXECUTE" -eq 1 ]; then
    shard_validate_args+=(--execute)
  fi
  "${shard_validate_args[@]}"
else
  "${args[@]}"
fi

if [ "$RUN_POST_VALIDATION_TEST" -eq 1 ]; then
  test_args=(
    python scripts/run_official_hydrated_post_validation_test.py
    --objective-status-json "$OBJECTIVE_STATUS_JSON"
    --output-json "$POST_VALIDATION_TEST_JSON"
  )
  if [ "$EXECUTE" -eq 1 ]; then
    test_args+=(--execute)
  fi
  if [ "$EXECUTE_TEST" -eq 1 ]; then
    test_args+=(--execute-test)
  fi
  "${test_args[@]}"
fi

if [ "$PUBLISH_ARTIFACTS" -eq 1 ]; then
  publish_official_hydration_artifacts
fi
