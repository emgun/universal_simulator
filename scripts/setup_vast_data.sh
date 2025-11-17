#!/bin/bash
# VastAI data setup script - downloads training data from B2
set -euo pipefail

TASKS="${1:-advection1d darcy2d}"
ROOT_DIR="${2:-data/pdebench}"

echo "📥 Downloading data for tasks: $TASKS"

# Configure rclone for B2
export RCLONE_CONFIG_B2TRAIN_TYPE=s3
export RCLONE_CONFIG_B2TRAIN_PROVIDER=Other
export RCLONE_CONFIG_B2TRAIN_ACCESS_KEY_ID="$B2_KEY_ID"
export RCLONE_CONFIG_B2TRAIN_SECRET_ACCESS_KEY="$B2_APP_KEY"
export RCLONE_CONFIG_B2TRAIN_ENDPOINT="$B2_S3_ENDPOINT"
export RCLONE_CONFIG_B2TRAIN_REGION="$B2_S3_REGION"
export RCLONE_CONFIG_B2TRAIN_ACL=private
export RCLONE_CONFIG_B2TRAIN_NO_CHECK_BUCKET=true

rm -rf "$ROOT_DIR" || true
mkdir -p "$ROOT_DIR"

# Download train and val splits for each task
for task in $TASKS; do
  if [ ! -f "$ROOT_DIR/${task}_train.h5" ]; then
    rclone copy "B2TRAIN:pdebench/full/$task/${task}_train.h5" "$ROOT_DIR/" --progress
  fi
  if [ ! -f "$ROOT_DIR/${task}_val.h5" ]; then
    rclone copy "B2TRAIN:pdebench/full/$task/${task}_val.h5" "$ROOT_DIR/" --progress
  fi
done

# Wait for all files to be ready
echo "⏳ Verifying downloads..."
for i in {1..60}; do
  all_ready=true
  for task in $TASKS; do
    [ ! -f "$ROOT_DIR/${task}_train.h5" ] && all_ready=false
    [ ! -f "$ROOT_DIR/${task}_val.h5" ] && all_ready=false
  done

  if $all_ready; then
    echo "✓ All data files ready"
    ls -lh "$ROOT_DIR/"
    exit 0
  fi

  [ $i -eq 60 ] && echo "❌ Download timeout" && ls -lh "$ROOT_DIR/" && exit 1
  sleep 5
done
