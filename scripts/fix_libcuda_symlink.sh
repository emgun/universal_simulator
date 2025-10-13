#!/usr/bin/env bash
#
# Ensure the CUDA driver library is visible to torch.compile / torch._dynamo.
# Some container images only ship libcuda.so.1; Inductor expects libcuda.so.
# Run once with sudo/root privileges on the target machine (e.g., Vast.ai VM).

set -euo pipefail

CUDA_REAL_PATH="/lib/x86_64-linux-gnu/libcuda.so.1"
CUDA_LINK_PATH="/usr/lib/libcuda.so"

if [ ! -f "$CUDA_REAL_PATH" ]; then
  echo "Could not find $CUDA_REAL_PATH. Nothing to link."
  exit 1
fi

if [ -L "$CUDA_LINK_PATH" ] || [ -f "$CUDA_LINK_PATH" ]; then
  echo "$CUDA_LINK_PATH already exists. Skipping."
  exit 0
fi

echo "Creating symlink $CUDA_LINK_PATH -> $CUDA_REAL_PATH"
ln -s "$CUDA_REAL_PATH" "$CUDA_LINK_PATH"
echo "Done."
