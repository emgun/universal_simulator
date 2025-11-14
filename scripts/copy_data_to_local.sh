#!/bin/bash
# Copy PDEBench data from network storage to local SSD on VastAI/Vultr instances

set -e

REMOTE_DATA_DIR="${1:-/root/data/pdebench}"
LOCAL_DATA_DIR="${2:-/workspace/data_local/pdebench}"

echo "📦 Copying PDEBench data to local storage..."
echo "   Remote: $REMOTE_DATA_DIR"
echo "   Local:  $LOCAL_DATA_DIR"

# Check if remote exists
if [ ! -d "$REMOTE_DATA_DIR" ]; then
    echo "❌ Remote data directory not found: $REMOTE_DATA_DIR"
    exit 1
fi

# Create local directory
mkdir -p "$LOCAL_DATA_DIR"

# Check available space
REQUIRED_GB=20
AVAILABLE_GB=$(df -BG "$LOCAL_DATA_DIR" | awk 'NR==2 {print $4}' | sed 's/G//')
if [ "$AVAILABLE_GB" -lt "$REQUIRED_GB" ]; then
    echo "⚠️  Warning: Low disk space (${AVAILABLE_GB}GB available, ${REQUIRED_GB}GB recommended)"
fi

# Copy with progress
rsync -avh --progress "$REMOTE_DATA_DIR/" "$LOCAL_DATA_DIR/"

echo "✅ Data copied successfully!"
echo "   Use --root $LOCAL_DATA_DIR when running precompute script"
