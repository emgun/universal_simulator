#!/usr/bin/env python3
"""Check if model predictions contain negative density values.

This script helps diagnose why ARM TTC is failing by analyzing whether
the baseline predictions (operator-only) or TTC candidates produce negative
density values.

Usage:
    python scripts/check_prediction_negativity.py <config_path>

Example:
    python scripts/check_prediction_negativity.py configs/ablation_upt_256tokens_fixed.yaml
"""

import sys
from pathlib import Path

import torch
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ups.utils.config_loader import load_config_with_includes
from ups.data.datasets import create_data_module
from ups.models.latent_operator import LatentOperator
from ups.io.decoder_anypoint import AnyPointDecoder


def load_model(config_path: str, device: str = "cpu"):
    """Load operator model from checkpoint."""
    print(f"Loading config: {config_path}")
    cfg = load_config_with_includes(config_path)

    # Find operator checkpoint
    ckpt_dir = Path(cfg.get("checkpoint", {}).get("dir", "checkpoints"))
    ckpt_path = ckpt_dir / "operator.pt"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Operator checkpoint not found: {ckpt_path}")

    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)

    # Extract model
    latent_dim = cfg["latent"]["dim"]
    operator = LatentOperator(
        latent_dim=latent_dim,
        cond_dim=1,
        hidden_dim=cfg["operator"]["pdet"]["hidden_dim"],
        depths=cfg["operator"]["pdet"]["depths"],
        group_size=cfg["operator"]["pdet"]["group_size"],
        num_heads=cfg["operator"]["pdet"]["num_heads"],
    )

    if "model" in checkpoint:
        operator.load_state_dict(checkpoint["model"])
    else:
        operator.load_state_dict(checkpoint)

    operator = operator.to(device).eval()

    return operator, cfg


def load_decoder(cfg: dict, device: str = "cpu"):
    """Load decoder from config."""
    decoder_cfg = cfg.get("ttc", {}).get("decoder", {})

    decoder = AnyPointDecoder(
        latent_dim=decoder_cfg["latent_dim"],
        query_dim=decoder_cfg["query_dim"],
        hidden_dim=decoder_cfg["hidden_dim"],
        mlp_hidden_dim=decoder_cfg["mlp_hidden_dim"],
        num_layers=decoder_cfg["num_layers"],
        num_heads=decoder_cfg["num_heads"],
        frequencies=decoder_cfg["frequencies"],
        output_channels=decoder_cfg["output_channels"],
    )

    return decoder.to(device).eval()


def create_query_grid(height: int = 64, width: int = 64, device: str = "cpu"):
    """Create 2D query grid for decoding."""
    y = torch.linspace(0, 1, height, device=device)
    x = torch.linspace(0, 1, width, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    query_points = torch.stack([yy, xx], dim=-1)  # [H, W, 2]
    return query_points


def check_negativity(config_path: str, num_samples: int = 10):
    """Check for negative values in predictions."""
    device = "cpu"

    operator, cfg = load_model(config_path, device)
    decoder = load_decoder(cfg, device)

    # Load test data
    print("Loading test dataset...")
    data_module = create_data_module(cfg, split="test")
    data_module.setup("test")
    test_loader = data_module.test_dataloader()

    # Create query grid
    query_grid = create_query_grid(device=device)

    # Analyze samples
    negative_counts = []
    negative_magnitudes = []
    min_values = []

    print(f"\\nAnalyzing {num_samples} test samples...")
    print("=" * 70)

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= num_samples:
                break

            # Get initial state (assuming batch has latent state)
            # This depends on your data format
            latent_state = batch  # Adjust based on actual batch structure

            # Forward pass through operator
            dt = torch.tensor([0.1], device=device)
            next_state = operator(latent_state, dt)

            # Decode to physical space
            decoded = decoder(
                latent_state=next_state,
                query_points=query_grid.unsqueeze(0).expand(next_state.z.size(0), -1, -1, -1),
            )

            # Check rho field (density)
            rho = decoded["rho"].cpu().numpy()

            # Count negatives
            negatives = (rho < 0).sum()
            negative_counts.append(negatives)

            if negatives > 0:
                neg_values = rho[rho < 0]
                negative_magnitudes.append(np.abs(neg_values).sum())
                min_val = rho.min()
                min_values.append(min_val)

                print(f"Sample {i+1}:")
                print(f"  Negative pixels: {negatives} / {rho.size} ({100*negatives/rho.size:.2f}%)")
                print(f"  Min value: {min_val:.6f}")
                print(f"  Total |negative|: {np.abs(neg_values).sum():.2f}")
                print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    samples_with_negatives = sum(1 for c in negative_counts if c > 0)
    print(f"Samples with negatives: {samples_with_negatives} / {num_samples}")

    if samples_with_negatives > 0:
        print(f"\\nAvg negatives per sample: {np.mean(negative_counts):.0f}")
        print(f"Avg total |negative|: {np.mean(negative_magnitudes):.2f}")
        print(f"Min value across all samples: {min(min_values):.6f}")
        print()
        print("⚠️  MODEL PRODUCES NEGATIVE DENSITY VALUES")
        print("   This explains ARM TTC failure (all candidates penalized similarly)")
        print()
        print("RECOMMENDATIONS:")
        print("1. Add non-negativity constraint to decoder (ReLU/Softplus)")
        print("2. Retrain with non-negativity loss")
        print("3. Post-process predictions to clamp negatives")
    else:
        print("✅ No negative values found in baseline predictions!")
        print()
        print("   This means TTC sampling/diffusion is introducing negatives.")
        print()
        print("RECOMMENDATIONS:")
        print("1. Check diffusion residual model")
        print("2. Reduce noise_std in TTC sampler")
        print("3. Check tau_range sampling")

    return samples_with_negatives > 0


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/check_prediction_negativity.py <config_path>")
        print("Example: python scripts/check_prediction_negativity.py configs/ablation_upt_256tokens_fixed.yaml")
        sys.exit(1)

    config_path = sys.argv[1]
    has_negatives = check_negativity(config_path, num_samples=10)

    sys.exit(1 if has_negatives else 0)


if __name__ == "__main__":
    main()
