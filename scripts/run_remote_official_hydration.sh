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
MIN_DOWNLOAD_BYTES=${MIN_DOWNLOAD_BYTES:-60000000000}
EXECUTE=${EXECUTE:-1}
EXECUTE_DOWNLOADS=${EXECUTE_DOWNLOADS:-1}
RUN_POST_VALIDATION_TEST=${RUN_POST_VALIDATION_TEST:-0}
EXECUTE_TEST=${EXECUTE_TEST:-0}

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

"${args[@]}"

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
