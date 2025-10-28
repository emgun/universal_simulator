#!/usr/bin/env python
"""Training script with memory optimizations for large models (UPT-17M+)."""

import os
import torch

# Set memory optimization environment variables
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")

# Enable gradient checkpointing for applicable modules (opt-in via env)
os.environ.setdefault("ENABLE_GRADIENT_CHECKPOINTING", "1")

from train import main

if __name__ == "__main__":
    print("=" * 60)
    print("Memory-optimized training mode enabled")
    print("  - Gradient checkpointing: ON (via env)")
    print("  - CUDA memory allocator: optimized")
    print("  - Expected slowdown: ~20-30%")
    print("=" * 60)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    main()

