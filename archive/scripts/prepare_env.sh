#!/usr/bin/env bash
set -euo pipefail

# Deterministic environment & convenience flags
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 || true
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TOKENIZERS_PARALLELISM=false

python - <<'PY'
import os
import random
import numpy as np
import torch

seed = 17
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.benchmark = False
print("Env OK | torch=", torch.__version__, " cuda=", torch.cuda.is_available())
PY

echo "To install: pip install -e .[dev]"

