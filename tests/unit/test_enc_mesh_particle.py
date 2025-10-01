import torch

from ups.io import MeshParticleEncoder, MeshParticleEncoderConfig


def make_graph(batch: int = 1, nodes: int = 12, features: int = 8):
    feats = torch.linspace(0, 1, steps=nodes * features).view(1, nodes, features)
    coords = torch.rand(nodes, 3)
    edges = torch.stack(
        [torch.arange(nodes - 1), torch.arange(1, nodes)], dim=1
    )  # simple chain
    return feats.repeat(batch, 1, 1), coords, edges


def test_encoder_identity_path():
    feats, coords, edges = make_graph(nodes=10, features=6)
    cfg = MeshParticleEncoderConfig(
        latent_len=10,
        latent_dim=6,
        hidden_dim=6,
        message_passing_steps=0,
        supernodes=1024,
        use_coords=False,
    )
    encoder = MeshParticleEncoder(cfg)
    fields = {"feat": feats}

    latent = encoder(fields, coords, connect=edges)
    recon = encoder.reconstruct(latent)

    assert latent.shape == (1, 10, 6)
    assert torch.allclose(recon, feats, atol=1e-6)


def test_encoder_reduces_tokens():
    feats, coords, edges = make_graph(nodes=32, features=4)
    cfg = MeshParticleEncoderConfig(
        latent_len=8,
        latent_dim=16,
        hidden_dim=16,
        message_passing_steps=2,
        supernodes=16,
        use_coords=True,
    )
    encoder = MeshParticleEncoder(cfg)
    fields = {"feat": feats}

    latent = encoder(fields, coords, connect=edges)
    assert latent.shape == (1, cfg.latent_len, cfg.latent_dim)
