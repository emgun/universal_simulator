#!/usr/bin/env bash
set -euo pipefail

# Prepare plumbing-only smoke shards with the smallest known B2 source set.
#
# This wrapper is for validating the remote data/experiment loop cheaply. It
# may derive validation/test slices from train sources, so outputs are not
# benchmark evidence.

export VERSION=${VERSION:-smoke-v1}
export REMOTE_PREFIX=${REMOTE_PREFIX:-$VERSION}
export MANIFEST=${MANIFEST:-docs/demo_smoke_data_manifest.yaml}
export TRAIN_COUNT=${TRAIN_COUNT:-8}
export VAL_COUNT=${VAL_COUNT:-4}
export TEST_COUNT=${TEST_COUNT:-4}
export TASKS=${TASKS:-"burgers1d advection1d darcy2d"}
export DRY_RUN=${DRY_RUN:-1}

# Keep smoke prep cheap by default. Advection currently has no known small
# source shard, so it still hydrates the native train file unless overridden.
export BURGERS1D_SOURCE_SPLITS=${BURGERS1D_SOURCE_SPLITS:-train}
export BURGERS1D_TRAIN_SOURCE_KEYS=${BURGERS1D_TRAIN_SOURCE_KEYS:-burgers1d/burgers1d_train_000.h5}
export ADVECTION1D_SOURCE_SPLITS=${ADVECTION1D_SOURCE_SPLITS:-train}
export DARCY2D_SOURCE_SPLITS=${DARCY2D_SOURCE_SPLITS:-train}

bash scripts/run_remote_shard_prep_b2.sh "$@"
