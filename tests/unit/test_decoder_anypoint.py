import torch

from ups.io import AnyPointDecoder, AnyPointDecoderConfig


def test_decoder_shapes_and_grads():
    cfg = AnyPointDecoderConfig(
        latent_dim=64,
        query_dim=3,
        hidden_dim=64,
        num_layers=2,
        num_heads=8,
        mlp_hidden_dim=128,
        frequencies=(1.0, 2.0),
        output_channels={"u": 2, "p": 1},
    )
    decoder = AnyPointDecoder(cfg)
    latent = torch.randn(4, 32, cfg.latent_dim, requires_grad=True)
    points = torch.rand(4, 17, cfg.query_dim)

    outputs = decoder(points, latent)

    assert set(outputs.keys()) == {"u", "p"}
    assert outputs["u"].shape == (4, 17, 2)
    assert outputs["p"].shape == (4, 17, 1)

    loss = outputs["u"].sum() + outputs["p"].sum()
    loss.backward()
    assert latent.grad is not None


def test_decoder_constant_output_via_bias():
    cfg = AnyPointDecoderConfig(
        latent_dim=8,
        query_dim=2,
        hidden_dim=8,
        num_layers=0,
        num_heads=1,
        mlp_hidden_dim=8,
        frequencies=(),
        output_channels={"phi": 1},
    )
    decoder = AnyPointDecoder(cfg)
    with torch.no_grad():
        decoder.query_embed.weight.zero_()
        decoder.query_embed.bias.zero_()
        head = decoder.heads["phi"]
        linear1, _, linear2 = head
        linear1.weight.zero_()
        linear1.bias.zero_()
        linear2.weight.zero_()
        linear2.bias.fill_(2.5)

    latent = torch.zeros(1, 5, cfg.latent_dim)
    points = torch.rand(1, 9, cfg.query_dim)
    outputs = decoder(points, latent)
    assert torch.allclose(outputs["phi"], torch.full((1, 9, 1), 2.5, device=outputs["phi"].device))
