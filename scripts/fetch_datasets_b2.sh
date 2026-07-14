#!/usr/bin/env bash
set -euo pipefail

# Compatibility entrypoint for older launch scripts. Dataset-name guessing,
# archive extraction, and destructive split cleanup have been retired. The
# transport URI (B2, HTTP, local file, or another exact mirror) now belongs in
# the immutable run lock and is verified before promotion into the cache.

WORKDIR=${WORKDIR:-$PWD}
DATA_ROOT=${DATA_ROOT:-$WORKDIR/data/pdebench}
DATA_CACHE=${DATA_CACHE:-$WORKDIR/data/cache}
DATA_LOCK=${DATA_LOCK:-}

if [ -z "$DATA_LOCK" ]; then
  echo "Error: DATA_LOCK is required; fuzzy B2 dataset-name hydration has been retired." >&2
  echo "Resolve a source manifest and protocol with: python -m ups.data.cli resolve ..." >&2
  exit 2
fi

if [ "$#" -gt 0 ]; then
  echo "Warning: positional dataset names are ignored; DATA_LOCK is the sole byte authority." >&2
fi

PYTHONPATH=${PYTHONPATH:-src} python -m ups.data.cli plan \
  --lock "$DATA_LOCK" --cache "$DATA_CACHE" --reserve-bytes "${DATA_RESERVE_BYTES:-0}"
PYTHONPATH=${PYTHONPATH:-src} python -m ups.data.cli stage \
  --lock "$DATA_LOCK" --cache "$DATA_CACHE" --run-dir "$DATA_ROOT" \
  --report "${DATA_STAGE_REPORT:-$WORKDIR/reports/data_stage.json}"
PYTHONPATH=${PYTHONPATH:-src} python -m ups.data.cli verify \
  --lock "$DATA_LOCK" --cache "$DATA_CACHE"
