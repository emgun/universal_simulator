#!/bin/bash
# VastAI data setup script - downloads training data from B2
# Simplified and bombproof version with proper error handling

set -euo pipefail

TASKS="${1:-advection1d darcy2d}"
ROOT_DIR="${2:-data/pdebench}"
CACHE_VERSION="${CACHE_VERSION:-""}"
CACHE_DIR="${CACHE_DIR:-data/latent_cache}"
CHECKSUM="${CHECKSUM:-0}"
BUCKET_ROOT="B2TRAIN:pdebench"

declare -A TASK_PATHS=(
  [advection1d]="1D/Advection/Train/"
  [burgers1d]="1D/Burgers/Train/"
  [diffusion_sorption1d]="1D/diffusion-sorption/"
  [reaction_diffusion1d]="1D/ReactionDiffusion/"
  [cfd1d_shocktube]="1D/CFD/"
  [darcy2d]="2D/DarcyFlow/"
  [reaction_diffusion2d]="2D/diffusion-reaction/"
  [navier_stokes2d]="2D/NS_incom/"
  [shallow_water2d]="2D/shallow-water/"
  [cfd2d_rand]="2D/CFD/2D_Train_Rand/"
  [cfd2d_turb]="2D/CFD/2D_Train_Turb/"
  [cfd3d]="3D/Train/"
)

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
checksum_flag=()
if [ "$CHECKSUM" -eq 1 ]; then
  checksum_flag=(--checksum)
fi
for task in $TASKS; do
  echo "📦 Task: $task"
  if [ -n "${TASK_PATHS[$task]:-}" ]; then
    echo "    expected path: ${TASK_PATHS[$task]}"
  fi

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
        if rclone copyto "${BUCKET_ROOT}/${base_path}/${pattern}" "$target_file" --progress --retries 3 "${checksum_flag[@]}" 2>/dev/null; then
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

if [ -n "$CACHE_VERSION" ]; then
  echo ""
  echo "📦 Downloading latent caches (version: $CACHE_VERSION) to $CACHE_DIR"
  mkdir -p "$CACHE_DIR"
  for task in $TASKS; do
    for split in train val test; do
      src="${BUCKET_ROOT}/latent_caches/${CACHE_VERSION}/${task}_${split}/"
      dst="${CACHE_DIR}/${task}_${split}/"
      if [ -d "$dst" ] && [ "$(ls -A "$dst" 2>/dev/null)" ]; then
        echo "  ✓ Cache exists for ${task}_${split}, skipping"
        continue
      fi
      echo "  → Downloading cache ${task}_${split}..."
      mkdir -p "$dst"
      if rclone copy "$src" "$dst" --progress --retries 3 "${checksum_flag[@]}" 2>/dev/null; then
        echo "    ✓ Downloaded cache for ${task}_${split}"
      else
        echo "    ⚠️  Cache missing for ${task}_${split} (continuing)"
      fi
    done
  done
  echo "✓ Cache download step complete (see warnings above for missing splits)"
fi

echo ""
echo "✓ All data files downloaded successfully"
ls -lh "$ROOT_DIR/"
exit 0
