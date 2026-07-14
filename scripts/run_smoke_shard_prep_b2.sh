#!/usr/bin/env bash
set -euo pipefail

PROTOCOL_ROOTS_PENDING=1
echo "Blocked: strat-v1-smoke requires canonical Burgers and Darcy provenance roots first." >&2
exit 2

# Prepare protocol-gated smoke shards from explicitly provenance-bearing sources.
# The universal integrity rules apply even to plumbing checks.

export VERSION=${VERSION:-strat-v1-smoke}
export REMOTE_PREFIX=${REMOTE_PREFIX:-$VERSION}
export MANIFEST=${MANIFEST:-docs/strat_v1_smoke_data_manifest.yaml}
export OUT_ROOT=${OUT_ROOT:-data/pdebench_strat_v1_smoke}
export TRAIN_COUNT=${TRAIN_COUNT:-8}
export VAL_COUNT=${VAL_COUNT:-4}
export TEST_COUNT=${TEST_COUNT:-4}
export TASKS=${TASKS:-"burgers1d advection1d darcy2d"}
export DRY_RUN=${DRY_RUN:-1}
export REQUIRED_GB=${REQUIRED_GB:-60}

export BURGERS1D_SOURCE_SPLITS=${BURGERS1D_SOURCE_SPLITS:-train}
export BURGERS1D_TRAIN_SOURCE_KEYS=${BURGERS1D_TRAIN_SOURCE_KEYS:-burgers1d/burgers1d_train_000.h5}
export BURGERS1D_PROVENANCE_DATASETS=${BURGERS1D_PROVENANCE_DATASETS:-source_file_index,source_sample_index}
export BURGERS1D_REGIME_DATASET=${BURGERS1D_REGIME_DATASET:-nu}
export BURGERS1D_FIELD_KIND=${BURGERS1D_FIELD_KIND:-temporal}
export BURGERS1D_TIME_AXIS=${BURGERS1D_TIME_AXIS:-1}
export ADVECTION1D_SOURCE_SPLITS=${ADVECTION1D_SOURCE_SPLITS:-train}
export ADVECTION1D_TRAIN_SOURCE_KEYS=${ADVECTION1D_TRAIN_SOURCE_KEYS:-advection1d/advection1d_train.h5}
export ADVECTION1D_PROVENANCE_DATASETS=${ADVECTION1D_PROVENANCE_DATASETS:-source_file_index,source_sample_index}
export ADVECTION1D_REGIME_DATASET=${ADVECTION1D_REGIME_DATASET:-beta}
export ADVECTION1D_FIELD_KIND=${ADVECTION1D_FIELD_KIND:-temporal}
export ADVECTION1D_TIME_AXIS=${ADVECTION1D_TIME_AXIS:-1}
export DARCY2D_SOURCE_SPLITS=${DARCY2D_SOURCE_SPLITS:-train}
export DARCY2D_TRAIN_SOURCE_KEYS=${DARCY2D_TRAIN_SOURCE_KEYS:-darcy2d/darcy2d_train.h5}
export DARCY2D_PROVENANCE_DATASETS=${DARCY2D_PROVENANCE_DATASETS:-source_file_index,source_sample_index}
export DARCY2D_REGIME_DATASET=${DARCY2D_REGIME_DATASET:-nu}
export DARCY2D_FIELD_KIND=${DARCY2D_FIELD_KIND:-steady}

bash scripts/run_remote_shard_prep_b2.sh "$@"
