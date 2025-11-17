#!/bin/bash
# VastAI data setup script - downloads training data from B2
# Simplified and bombproof version with proper error handling

set -euo pipefail

TASKS="${1:-advection1d darcy2d}"
ROOT_DIR="${2:-data/pdebench}"

echo "📥 Downloading data for tasks: $TASKS"

# === FIX: VastAI env-vars are in container environment, not bash by default ===
# Check if B2 credentials are available, if not try to source them
if [ -z "${B2_KEY_ID:-}" ]; then
  echo "⚠️  B2 credentials not in environment, attempting to load..."

  # Try sourcing from container environment (VastAI stores them here)
  if [ -f /proc/1/environ ]; then
    # Extract env vars from container init process
    eval $(cat /proc/1/environ | tr '\0' '\n' | grep -E '^(B2_|WANDB_)' | sed 's/^/export /')
    echo "✓ Loaded credentials from container environment"
  elif [ -f ~/.bashrc ] && grep -q "B2_KEY_ID" ~/.bashrc; then
    set +u
    source ~/.bashrc 2>/dev/null || true
    set -u
    echo "✓ Loaded credentials from ~/.bashrc"
  else
    echo "❌ ERROR: B2 credentials not found"
    echo "   VastAI env-vars not accessible. Please check:"
    echo "   1. Run: vastai show env-vars"
    echo "   2. Ensure B2_KEY_ID, B2_APP_KEY, etc. are set"
    exit 1
  fi
fi

# Validate required credentials
for var in B2_KEY_ID B2_APP_KEY B2_S3_ENDPOINT B2_S3_REGION; do
  if [ -z "${!var:-}" ]; then
    echo "❌ ERROR: Required variable $var is not set"
    exit 1
  fi
done
echo "✓ All B2 credentials validated"

# Configure rclone for B2 using environment variables
export RCLONE_CONFIG_B2TRAIN_TYPE=s3
export RCLONE_CONFIG_B2TRAIN_PROVIDER=Other
export RCLONE_CONFIG_B2TRAIN_ACCESS_KEY_ID="$B2_KEY_ID"
export RCLONE_CONFIG_B2TRAIN_SECRET_ACCESS_KEY="$B2_APP_KEY"
export RCLONE_CONFIG_B2TRAIN_ENDPOINT="$B2_S3_ENDPOINT"
export RCLONE_CONFIG_B2TRAIN_REGION="$B2_S3_REGION"
export RCLONE_CONFIG_B2TRAIN_ACL=private
export RCLONE_CONFIG_B2TRAIN_NO_CHECK_BUCKET=true

# Create data directory
mkdir -p "$ROOT_DIR"

# Download train splits (val/test will come from WandB artifacts)
echo "Downloading training data files..."
for task in $TASKS; do
  target_file="$ROOT_DIR/${task}_train.h5"

  if [ -f "$target_file" ]; then
    echo "  ✓ $task (already exists, skipping)"
    continue
  fi

  echo "  → Downloading $task..."
  if rclone copy "B2TRAIN:pdebench/full/$task/${task}_train.h5" "$ROOT_DIR/" --progress --retries 3; then
    if [ -f "$target_file" ]; then
      echo "  ✓ $task ($(du -h "$target_file" | cut -f1))"
    else
      echo "  ✗ $task (download succeeded but file not found)"
      exit 1
    fi
  else
    echo "  ✗ $task (rclone failed)"
    exit 1
  fi
done

echo ""
echo "✓ All data files downloaded successfully"
ls -lh "$ROOT_DIR/"
exit 0
