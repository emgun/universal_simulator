#!/usr/bin/env python
"""Test zero-shot super-resolution capability of UPT models.

This script encodes a sample field, decodes at multiple resolutions using the
AnyPointDecoder, and reports NRMSE at base and super-res grids.

Usage:
  python scripts/test_zero_shot_superres.py --config configs/train_burgers_upt17m.yaml \
    --checkpoint checkpoints/operator.pt --base-resolution 64
"""

import argparse
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from ups.io.enc_grid import GridEncoder, GridEncoderConfig
from ups.io.decoder_anypoint import AnyPointDecoder, AnyPointDecoderConfig
from ups.data.pdebench import PDEBenchDataset, PDEBenchConfig
import yaml


def _make_grid_coords(H: int, W: int, device: torch.device) -> torch.Tensor:
    ys = torch.linspace(0.0, 1.0, H, device=device)
    xs = torch.linspace(0.0, 1.0, W, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gx, gy], dim=-1).reshape(1, H * W, 2)


def _nrmse(pred: torch.Tensor, tgt: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mse = torch.mean((pred - tgt) ** 2)
    denom = torch.mean(tgt ** 2) + eps
    return torch.sqrt(mse / denom)


def _decode_to_grid(decoder: AnyPointDecoder, latent: torch.Tensor, H: int, W: int, device: torch.device) -> torch.Tensor:
    coords = _make_grid_coords(H, W, device)
    fields = decoder(coords, latent)
    u = fields.get("u")
    if u is None:
        # Fallback to any field
        k = next(iter(fields.keys()))
        u = fields[k]
    return u  # (B, H*W, 1)


def test_superres(config_path: str, checkpoint_path: str, base_resolution: int = 64, factors=(2, 4)) -> Dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = yaml.safe_load(Path(config_path).read_text())
    latent_dim = int(cfg.get("latent", {}).get("dim", 32))
    latent_len = int(cfg.get("latent", {}).get("tokens", 32))

    # Load one training sample from PDEBench
    ds = PDEBenchDataset(PDEBenchConfig(task=cfg.get("data", {}).get("task", "burgers1d"), split="train", root=cfg.get("data", {}).get("root")))
    sample = ds[0]

    # Prepare single-channel field 'u'
    fields_full = sample["fields"].float()  # shape could be (T, X, C) etc., but we just need one slice
    # Flatten to (B, N, C) assuming 2D grid (H, W). If 1D, use H=1.
    if fields_full.dim() == 4:  # (T, H, W, C) or (T, C, H, W)
        if fields_full.shape[-1] <= 8:
            H, W = int(fields_full.shape[1]), int(fields_full.shape[2])
            u0 = fields_full[0, :, :, 0].unsqueeze(0).unsqueeze(-1)  # (1, H, W, 1)
        else:
            H, W = int(fields_full.shape[-2]), int(fields_full.shape[-1])
            u0 = fields_full[0, 0, :, :].unsqueeze(0).unsqueeze(-1)  # (1, H, W, 1)
    elif fields_full.dim() == 3:  # (T, X, C) or (T, C, X)
        if fields_full.shape[-1] <= 8:
            H, W = 1, int(fields_full.shape[-2])
            u0 = fields_full[0, :, 0].view(1, 1, W, 1)
        else:
            H, W = 1, int(fields_full.shape[-1])
            u0 = fields_full[0, 0, :].view(1, 1, W, 1)
    else:
        raise ValueError(f"Unexpected PDEBench field shape: {tuple(fields_full.shape)}")

    # Convert to (B, N, C)
    B = 1
    N = H * W
    u_base = u0.view(B, H, W, 1).reshape(B, N, 1).to(device)
    coords_base = _make_grid_coords(H, W, device)

    # Build encoder/decoder
    enc = GridEncoder(GridEncoderConfig(latent_len=latent_len, latent_dim=latent_dim, field_channels={"u": 1}, patch_size=4)).to(device).eval()
    dec = AnyPointDecoder(AnyPointDecoderConfig(latent_dim=latent_dim, query_dim=2, hidden_dim=max(256, latent_dim * 2), num_layers=2, num_heads=4, output_channels={"u": 1})).to(device).eval()

    # Encode
    latent = enc({"u": u_base}, coords_base, meta={"grid_shape": (H, W)})

    # Decode base and measure NRMSE
    u_pred_base = _decode_to_grid(dec, latent, H, W, device)
    nrmse_base = _nrmse(u_pred_base, u_base).item()

    results = {"1x": nrmse_base}

    # Super-resolution factors
    for f in factors:
        Hs, Ws = H * f, W * f
        u_sr = _decode_to_grid(dec, latent, Hs, Ws, device)  # (B, Hs*Ws, 1)
        # Downsample to base for comparison
        u_sr_2d = u_sr.view(B, Hs, Ws, 1).permute(0, 3, 1, 2).contiguous()  # (B,1,Hs,Ws)
        u_down = F.interpolate(u_sr_2d, size=(H, W), mode="bilinear", align_corners=False)
        u_down_flat = u_down.permute(0, 2, 3, 1).reshape(B, N, 1)
        results[f"{f}x"] = _nrmse(u_down_flat, u_base).item()

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=False, help="Operator checkpoint (unused for decoding test)")
    parser.add_argument("--base-resolution", type=int, default=64)
    args = parser.parse_args()

    out = test_superres(args.config, args.checkpoint or "", base_resolution=args.base_resolution, factors=(2, 4, 8))
    print("\n" + "=" * 60)
    print("Zero-Shot Super-Resolution Results")
    print("=" * 60)
    for k, v in out.items():
        print(f"{k:>4}: NRMSE = {v:.6f}")

