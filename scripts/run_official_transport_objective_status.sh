#!/usr/bin/env bash
set -euo pipefail

# Aggregate official transport-shift objective status without rerunning gates.
#
# Safe planning run:
#   DRY_RUN=1 bash scripts/run_official_transport_objective_status.sh
#
# Literal objective release check:
#   bash scripts/run_official_transport_objective_status.sh
#
# Observed-context policy acceptance check:
#   ACCEPT_OBSERVED_CONTEXT=1 REQUIRE_STATUS=observed-accepted bash scripts/run_official_transport_objective_status.sh
#
# Two-frame context transport policy acceptance check:
#   ACCEPT_CONTEXT_TRANSPORT=1 REQUIRE_STATUS=context-accepted bash scripts/run_official_transport_objective_status.sh

DRY_RUN=${DRY_RUN:-0}
OUTPUT_ROOT=${OUTPUT_ROOT:-reports/research/sota_loop}
CONSTANT_AUDIT_JSON=${CONSTANT_AUDIT_JSON:-${OUTPUT_ROOT}/transport_shift_goal_audit.json}
OBSERVED_AUDIT_JSON=${OBSERVED_AUDIT_JSON:-${OUTPUT_ROOT}/observed_transport_shift_goal_audit.json}
CONTEXT_AUDIT_JSON=${CONTEXT_AUDIT_JSON:-${OUTPUT_ROOT}/context_transport_shift_goal_audit.json}
TRAIN_FEATURE_DIAGNOSTIC_JSON=${TRAIN_FEATURE_DIAGNOSTIC_JSON:-${OUTPUT_ROOT}/train_only_transport_feature_diagnostic_full.json}
TRAIN_IDENTIFIABILITY_AUDIT_JSON=${TRAIN_IDENTIFIABILITY_AUDIT_JSON:-${OUTPUT_ROOT}/train_only_transport_identifiability_audit.json}
HYDRATION_OPTIONS_JSON=${HYDRATION_OPTIONS_JSON:-${OUTPUT_ROOT}/transport_data_hydration_options.json}
HYDRATION_PLAN_JSON=${HYDRATION_PLAN_JSON:-${OUTPUT_ROOT}/official_advection_hydration_plan.json}
HYDRATION_PLAN_VALIDATION_JSON=${HYDRATION_PLAN_VALIDATION_JSON:-${OUTPUT_ROOT}/official_advection_hydration_plan_validation.json}
HYDRATION_PLAN_RUN_JSON=${HYDRATION_PLAN_RUN_JSON:-${OUTPUT_ROOT}/official_advection_hydration_plan_run.json}
HYDRATION_PREFLIGHT_JSON=${HYDRATION_PREFLIGHT_JSON:-${OUTPUT_ROOT}/official_advection_hydration_preflight.json}
HYDRATION_STORAGE_JSON=${HYDRATION_STORAGE_JSON:-${OUTPUT_ROOT}/official_advection_hydration_storage_recommendation.json}
OBJECTIVE_STATUS_JSON=${OBJECTIVE_STATUS_JSON:-${OUTPUT_ROOT}/transport_objective_status.json}
REQUIRE_STATUS=${REQUIRE_STATUS:-literal-achieved}
ACCEPT_OBSERVED_CONTEXT=${ACCEPT_OBSERVED_CONTEXT:-0}
ACCEPT_CONTEXT_TRANSPORT=${ACCEPT_CONTEXT_TRANSPORT:-0}

observed_args=()
if [ "$ACCEPT_OBSERVED_CONTEXT" -eq 1 ]; then
  observed_args+=(--accept-observed-context)
fi

context_args=()
if [ "$ACCEPT_CONTEXT_TRANSPORT" -eq 1 ]; then
  context_args+=(--accept-context-transport)
fi

if [ "$DRY_RUN" -eq 1 ]; then
  echo "DRY_RUN: aggregate transport objective status"
  echo "DRY_RUN: constant_audit=${CONSTANT_AUDIT_JSON}"
  echo "DRY_RUN: context_audit=${CONTEXT_AUDIT_JSON}"
  echo "DRY_RUN: observed_audit=${OBSERVED_AUDIT_JSON}"
  echo "DRY_RUN: train_feature_diagnostic=${TRAIN_FEATURE_DIAGNOSTIC_JSON}"
  echo "DRY_RUN: train_identifiability_audit=${TRAIN_IDENTIFIABILITY_AUDIT_JSON}"
  echo "DRY_RUN: hydration_options=${HYDRATION_OPTIONS_JSON}"
  echo "DRY_RUN: hydration_plan=${HYDRATION_PLAN_JSON}"
  echo "DRY_RUN: hydration_plan_validation=${HYDRATION_PLAN_VALIDATION_JSON}"
  echo "DRY_RUN: hydration_plan_run=${HYDRATION_PLAN_RUN_JSON}"
  echo "DRY_RUN: hydration_preflight=${HYDRATION_PREFLIGHT_JSON}"
  echo "DRY_RUN: hydration_storage=${HYDRATION_STORAGE_JSON}"
  echo "DRY_RUN: output=${OBJECTIVE_STATUS_JSON}"
  echo "DRY_RUN: require_status=${REQUIRE_STATUS}"
  echo "DRY_RUN: accept_context_transport=${ACCEPT_CONTEXT_TRANSPORT}"
  echo "DRY_RUN: accept_observed_context=${ACCEPT_OBSERVED_CONTEXT}"
  exit 0
fi

python scripts/audit_transport_objective_status.py \
  --constant-audit-json "$CONSTANT_AUDIT_JSON" \
  --context-audit-json "$CONTEXT_AUDIT_JSON" \
  --observed-audit-json "$OBSERVED_AUDIT_JSON" \
  --train-feature-diagnostic-json "$TRAIN_FEATURE_DIAGNOSTIC_JSON" \
  --train-identifiability-audit-json "$TRAIN_IDENTIFIABILITY_AUDIT_JSON" \
  --hydration-options-json "$HYDRATION_OPTIONS_JSON" \
  --hydration-plan-json "$HYDRATION_PLAN_JSON" \
  --hydration-plan-validation-json "$HYDRATION_PLAN_VALIDATION_JSON" \
  --hydration-plan-run-json "$HYDRATION_PLAN_RUN_JSON" \
  --hydration-preflight-json "$HYDRATION_PREFLIGHT_JSON" \
  --hydration-storage-json "$HYDRATION_STORAGE_JSON" \
  --output-json "$OBJECTIVE_STATUS_JSON" \
  --require-status "$REQUIRE_STATUS" \
  "${context_args[@]+"${context_args[@]}"}" \
  "${observed_args[@]+"${observed_args[@]}"}"
