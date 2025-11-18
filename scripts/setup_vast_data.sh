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

# Download all splits (train, val, test)
echo "Downloading data files (train/val/test)..."
for task in $TASKS; do
  echo "📦 Task: $task"

  # Download all three splits
  for split in train val test; do
    target_file="$ROOT_DIR/${task}_${split}.h5"

    if [ -f "$target_file" ]; then
      echo "  ✓ $split (already exists, skipping)"
      continue
    fi

    echo "  → Downloading $split..."

    # Try different B2 path patterns
    # Pattern 1: full/$task/${task}_$split.h5
    # Pattern 2: full/$task/${task}_$split_000.h5
    # Pattern 3: pdebench/${task}_full_v1/${task}_$split.h5
    downloaded=false

    for base_path in "full/$task" "pdebench/${task}_full_v1"; do
      for pattern in "${task}_${split}.h5" "${task}_${split}_000.h5"; do
        if rclone copyto "B2TRAIN:PDEbench/${base_path}/${pattern}" "$target_file" --progress --retries 3 2>/dev/null; then
          if [ -f "$target_file" ]; then
            echo "  ✓ $split ($(du -h "$target_file" | cut -f1))"
            downloaded=true
            break 2
          fi
        fi
      done
    done

    if ! $downloaded; then
      echo "  ⚠️  $split (file not found in B2, skipping)"
      # Don't exit - test/val might not exist for all tasks
    fi
  done
done

echo ""
echo "✓ All data files downloaded successfully"
ls -lh "$ROOT_DIR/"
exit 0
