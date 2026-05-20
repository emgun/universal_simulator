#!/usr/bin/env bash
set -euo pipefail

# Promote the already-audited two-frame context transport result under an
# explicit benchmark-policy acceptance flag. This does not rerun the gate and
# does not touch the held-out test ledger.
#
# Safe planning run:
#   DRY_RUN=1 bash scripts/run_official_context_transport_objective_status.sh
#
# Context-accepted release check:
#   bash scripts/run_official_context_transport_objective_status.sh

DRY_RUN=${DRY_RUN:-0}
OUTPUT_ROOT=${OUTPUT_ROOT:-reports/research/sota_loop}
OBJECTIVE_STATUS_JSON=${OBJECTIVE_STATUS_JSON:-${OUTPUT_ROOT}/transport_objective_status_context_accepted.json}

if [ "$DRY_RUN" -eq 1 ]; then
  ACCEPT_CONTEXT_TRANSPORT=1 \
    REQUIRE_STATUS=context-accepted \
    OBJECTIVE_STATUS_JSON="$OBJECTIVE_STATUS_JSON" \
    DRY_RUN=1 \
    bash scripts/run_official_transport_objective_status.sh
  exit 0
fi

ACCEPT_CONTEXT_TRANSPORT=1 \
  REQUIRE_STATUS=context-accepted \
  OBJECTIVE_STATUS_JSON="$OBJECTIVE_STATUS_JSON" \
  bash scripts/run_official_transport_objective_status.sh
