#!/bin/bash
set -euo pipefail

# Remote preprocessing pipeline for PDEBench datasets
# Runs on VastAI/Vultr GPU instance, no local downloads required

echo "═══════════════════════════════════════════════════════"
echo "PDEBench Remote Preprocessing Pipeline"
echo "═══════════════════════════════════════════════════════"

# Parse arguments
TASKS=${1:-"advection1d darcy2d"}  # Space-separated task list
CACHE_DIM=${2:-""}  # Optional: latent dim for cache precomputation
CACHE_TOKENS=${3:-""}  # Optional: latent tokens for cache

echo "Tasks to process: $TASKS"
echo "Latent cache: ${CACHE_DIM:+${CACHE_DIM}d × ${CACHE_TOKENS}tok}${CACHE_DIM:-disabled}"
echo ""

# Install dependencies
apt-get update && apt-get install -y git rclone build-essential python3-dev
pip install -e .[dev]

# Setup B2 rclone
export RCLONE_CONFIG_B2TRAIN_TYPE=s3
export RCLONE_CONFIG_B2TRAIN_PROVIDER=Other
export RCLONE_CONFIG_B2TRAIN_ACCESS_KEY_ID="$B2_KEY_ID"
export RCLONE_CONFIG_B2TRAIN_SECRET_ACCESS_KEY="$B2_APP_KEY"
export RCLONE_CONFIG_B2TRAIN_ENDPOINT="$B2_S3_ENDPOINT"
export RCLONE_CONFIG_B2TRAIN_REGION="$B2_S3_REGION"
export RCLONE_CONFIG_B2TRAIN_ACL=private
export RCLONE_CONFIG_B2TRAIN_NO_CHECK_BUCKET=true

# Create working directories
mkdir -p data/pdebench_raw data/pdebench data/latent_cache
cd /workspace/universal_simulator

echo ""
echo "──────────────────────────────────────────────────────"
echo "Step 1: Download Raw PDEBench Data"
echo "──────────────────────────────────────────────────────"

# Clone PDEBench repository for download scripts
if [ ! -d /tmp/PDEBench ]; then
  git clone https://github.com/pdebench/PDEBench.git /tmp/PDEBench
fi
cd /tmp/PDEBench
pip install -e .

# Download each task's raw data IN PARALLEL
echo "Starting parallel downloads..."
download_pids=()

for task in $TASKS; do
  (
    echo "→ Downloading $task..."
    # Map UPS task names to PDEBench download names
    case $task in
      advection1d)
        python pdebench/data_download/download_direct.py \
          --root_folder /workspace/data/pdebench_raw --pde_name advection
        ;;
      burgers1d)
        python pdebench/data_download/download_direct.py \
          --root_folder /workspace/data/pdebench_raw --pde_name burgers
        ;;
      darcy2d)
        python pdebench/data_download/download_direct.py \
          --root_folder /workspace/data/pdebench_raw --pde_name darcy
        ;;
      reaction_diffusion2d)
        python pdebench/data_download/download_direct.py \
          --root_folder /workspace/data/pdebench_raw --pde_name 2d_reacdiff
        ;;
      navier_stokes2d)
        python pdebench/data_download/download_direct.py \
          --root_folder /workspace/data/pdebench_raw --pde_name ns_incom
        ;;
      *)
        echo "⚠️  Unknown task: $task (skipping)"
        ;;
    esac
    echo "✓ Download complete: $task"
  ) &
  download_pids+=($!)
done

# Wait for all downloads to complete
echo "Waiting for ${#download_pids[@]} parallel downloads..."
for pid in "${download_pids[@]}"; do
  wait $pid || echo "⚠️  Download process $pid failed (continuing)"
done
echo "✓ All downloads complete"

cd /workspace/universal_simulator

echo ""
echo "──────────────────────────────────────────────────────"
echo "Step 2: Convert to UPS Format"
echo "──────────────────────────────────────────────────────"

# Convert each task using convert_pdebench_multimodal.py IN PARALLEL
echo "Starting parallel conversions..."
convert_pids=()

for task in $TASKS; do
  (
    # IMPORTANT: cd into workspace for each subshell
    cd /workspace/universal_simulator || exit 1

    echo "→ Converting $task to UPS format..."

    PYTHONPATH=src python scripts/convert_pdebench_multimodal.py $task \
      --root /workspace/data/pdebench_raw \
      --out data/pdebench \
      --limit 100 \
      --samples 1000 || echo "⚠️  Conversion failed for $task (continuing)"

    # Verify output files exist
    if [ -f "data/pdebench/${task}_train.h5" ]; then
      echo "  ✓ ${task}_train.h5 created"
    else
      echo "  ✗ ${task}_train.h5 MISSING"
    fi
  ) &
  convert_pids+=($!)
done

# Wait for all conversions to complete
echo "Waiting for ${#convert_pids[@]} parallel conversions..."
for pid in "${convert_pids[@]}"; do
  wait $pid || echo "⚠️  Conversion process $pid failed (continuing)"
done
echo "✓ All conversions complete"

echo ""
echo "──────────────────────────────────────────────────────"
echo "Step 3: Upload Converted Data to B2"
echo "──────────────────────────────────────────────────────"

# Upload each converted dataset to B2 IN PARALLEL
echo "Starting parallel uploads to B2..."
upload_pids=()

for task in $TASKS; do
  (
    # IMPORTANT: cd into workspace for each subshell
    cd /workspace/universal_simulator || exit 1

    echo "→ Uploading $task to B2..."

    # Upload all splits (train, val, test)
    for split in train val test; do
      file="data/pdebench/${task}_${split}.h5"
      if [ -f "$file" ]; then
        rclone copy "$file" \
          "B2TRAIN:pdebench/full/${task}/" \
          --progress --transfers 4
        echo "  ✓ Uploaded ${task}_${split}.h5"
      fi
    done
  ) &
  upload_pids+=($!)
done

# Wait for all uploads to complete
echo "Waiting for ${#upload_pids[@]} parallel uploads..."
for pid in "${upload_pids[@]}"; do
  wait $pid || echo "⚠️  Upload process $pid failed (continuing)"
done
echo "✓ All uploads complete"

echo ""
echo "──────────────────────────────────────────────────────"
echo "Step 4: Verify B2 Uploads"
echo "──────────────────────────────────────────────────────"

for task in $TASKS; do
  echo "→ Verifying $task in B2..."
  rclone ls "B2TRAIN:pdebench/full/${task}/" || echo "  ⚠️  No files found for $task"
done

# Cleanup raw data to free space
echo ""
echo "→ Cleaning up raw data..."
rm -rf /workspace/data/pdebench_raw /tmp/PDEBench
du -sh data/pdebench

# Optional: Precompute latent caches
if [ -n "$CACHE_DIM" ] && [ -n "$CACHE_TOKENS" ]; then
  echo ""
  echo "──────────────────────────────────────────────────────"
  echo "Step 5: Precompute Latent Caches (${CACHE_DIM}d × ${CACHE_TOKENS}tok)"
  echo "──────────────────────────────────────────────────────"

  PYTHONPATH=src python scripts/precompute_latent_cache.py \
    --tasks $TASKS \
    --splits train val test \
    --cache-dir data/latent_cache \
    --latent-dim $CACHE_DIM \
    --latent-len $CACHE_TOKENS \
    --device cuda \
    --batch-size 16 \
    --num-workers 4 \
    --cache-dtype float16 \
    --pin-memory \
    --parallel || echo "⚠️  Cache precomputation failed (continuing)"

  echo ""
  echo "──────────────────────────────────────────────────────"
  echo "Step 6: Upload Latent Caches to B2"
  echo "──────────────────────────────────────────────────────"

  CACHE_VERSION="upt_${CACHE_DIM}d_${CACHE_TOKENS}tok"

  for task in $TASKS; do
    for split in train val test; do
      cache_dir="data/latent_cache/${task}_${split}"
      if [ -d "$cache_dir" ]; then
        echo "→ Uploading ${task}_${split} cache..."
        rclone copy "$cache_dir/" \
          "B2TRAIN:pdebench/latent_caches/$CACHE_VERSION/${task}_${split}/" \
          --progress --transfers 8
      fi
    done
  done

  echo "✓ Latent cache uploaded: $CACHE_VERSION"
else
  echo ""
  echo "→ Skipping latent cache precomputation (not requested)"
fi

echo ""
echo "═══════════════════════════════════════════════════════"
echo "✓ Remote Preprocessing Complete!"
echo "═══════════════════════════════════════════════════════"
echo ""
echo "Uploaded to B2:"
for task in $TASKS; do
  echo "  • B2TRAIN:pdebench/full/${task}/"
done
if [ -n "$CACHE_DIM" ]; then
  echo "  • B2TRAIN:pdebench/latent_caches/upt_${CACHE_DIM}d_${CACHE_TOKENS}tok/"
fi
echo ""
echo "Next steps:"
echo "  1. Verify data in B2: rclone ls B2TRAIN:pdebench/full/"
echo "  2. Launch training: python scripts/vast_launch.py launch --config <config>"
