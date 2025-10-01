import torch

from ups.core.blocks_pdet import PDETransformerBlock, PDETransformerConfig


def test_pde_transformer_block_roundtrip():
    cfg = PDETransformerConfig(
        input_dim=48,
        hidden_dim=64,
        depths=(2, 2, 2),
        group_size=16,
        num_heads=4,
        mlp_ratio=2.0,
    )
    block = PDETransformerBlock(cfg)
    x = torch.randn(3, 64, cfg.input_dim, requires_grad=True)
    out = block(x)
    assert out.shape == x.shape
    loss = out.pow(2).mean()
    loss.backward()
    assert torch.isfinite(x.grad).all()


def test_pde_transformer_block_training_loop():
    cfg = PDETransformerConfig(
        input_dim=32,
        hidden_dim=64,
        depths=(1, 2, 2),
        group_size=16,
        num_heads=4,
        mlp_ratio=2.0,
    )
    block = PDETransformerBlock(cfg)
    optimizer = torch.optim.Adam(block.parameters(), lr=1e-3)
    x = torch.randn(2, 128, cfg.input_dim)
    target = torch.zeros_like(x)

    for _ in range(20):
        optimizer.zero_grad()
        out = block(x)
        loss = (out - target).pow(2).mean()
        loss.backward()
        assert torch.isfinite(loss)
        optimizer.step()
