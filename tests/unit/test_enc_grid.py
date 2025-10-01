import math

import torch

from ups.io import GridEncoder, GridEncoderConfig


def make_sample(batch: int = 2, height: int = 16, width: int = 16):
    N = height * width
    u = torch.linspace(0, 1, N).view(1, N, 1).repeat(batch, 1, 1)
    v = torch.linspace(1, 2, N).view(1, N, 1).repeat(batch, 1, 1)
    coords = torch.stack(torch.meshgrid(
        torch.linspace(0, 1, height),
        torch.linspace(0, 1, width),
        indexing="ij"
    ), dim=-1).reshape(1, N, 2).repeat(batch, 1, 1)
    fields = {"u": u, "v": v}
    meta = {"grid_shape": (height, width)}
    return fields, coords, meta


def test_grid_encoder_shapes_and_identity():
    fields, coords, meta = make_sample()
    cfg = GridEncoderConfig(
        latent_len=(meta["grid_shape"][0] // 4) * (meta["grid_shape"][1] // 4),
        latent_dim=32,
        field_channels={"u": 1, "v": 1},
        patch_size=4,
        use_fourier_features=False,
    )
    encoder = GridEncoder(cfg)

    latent = encoder(fields, coords, meta=meta)
    assert latent.shape[0] == coords.shape[0]
    assert latent.shape[1] == cfg.latent_len
    assert latent.shape[2] == cfg.latent_dim

    recon = encoder.reconstruct(latent, meta)
    for name in fields:
        mse = (recon[name] - fields[name]).pow(2).mean().item()
        assert mse < 1e-6
