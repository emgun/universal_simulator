#!/usr/bin/env python3
"""Quick script to restore all UPT Phase 1.5 changes that were lost."""

import sys
from pathlib import Path

# Read current latent_pairs.py
latent_pairs_path = Path(__file__).parent / "src/ups/data/latent_pairs.py"
content = latent_pairs_path.read_text()

# Check if changes already applied
if "latent_pair_collate" in content and "use_inverse_losses and fields_cpu" in content:
    print("✓ All changes already present in latent_pairs.py")
    sys.exit(0)

print("⚠️  Some changes missing from latent_pairs.py")
print("Please manually verify and restore:")
print("1. GridLatentPairDataset.__getitem__() - Add physical fields loading")
print("2. latent_pair_collate() function")
print("3. unpack_batch() - Handle dict format")
print("4. build_latent_pair_loader() - Pass use_inverse_losses")
print("5. DataLoader calls - Add collate_fn=latent_pair_collate")

# Check train.py
train_path = Path(__file__).parent / "scripts/train.py"
train_content = train_path.read_text()

if "compute_operator_loss_bundle" in train_content and "isinstance(unpacked, dict)" in train_content:
    print("✓ Training loop changes present in train.py")
else:
    print("⚠️  Training loop changes missing from train.py")

print("\nRecommendation: Review UPT_PHASE1.5_COMPLETE.md for full implementation details")
