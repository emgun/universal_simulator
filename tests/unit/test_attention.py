import pytest
import torch
from src.ups.core.attention import StandardSelfAttention


def test_standard_attention_shape():
    """Standard attention should preserve input shape."""
    attn = StandardSelfAttention(dim=192, num_heads=6)
    x = torch.randn(4, 256, 192)
    out = attn(x)
    assert out.shape == x.shape


def test_standard_attention_heads_divisibility():
    """Should raise error if dim not divisible by num_heads."""
    with pytest.raises(AssertionError):
        StandardSelfAttention(dim=192, num_heads=7)  # 192 not divisible by 7


def test_standard_attention_with_qk_norm():
    """QK normalization should stabilize attention."""
    attn = StandardSelfAttention(dim=192, num_heads=6, qk_norm=True)
    x = torch.randn(4, 256, 192)
    out = attn(x)
    assert out.shape == x.shape
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_standard_attention_dropout():
    """Dropout should be applied in training mode."""
    attn = StandardSelfAttention(dim=192, num_heads=6, attn_drop=0.5, proj_drop=0.5)
    attn.train()

    x = torch.randn(4, 256, 192)
    out1 = attn(x)
    out2 = attn(x)

    # With dropout, outputs should differ
    assert not torch.allclose(out1, out2)


def test_standard_attention_inference():
    """Dropout should be disabled in eval mode."""
    attn = StandardSelfAttention(dim=192, num_heads=6, attn_drop=0.5, proj_drop=0.5)
    attn.eval()

    x = torch.randn(4, 256, 192)
    out1 = attn(x)
    out2 = attn(x)

    # Without dropout, outputs should be identical
    assert torch.allclose(out1, out2)


def test_standard_attention_vs_channel_separated():
    """Verify both attention mechanisms produce valid outputs."""
    from src.ups.core.blocks_pdet import ChannelSeparatedSelfAttention

    dim = 192
    num_heads = 6
    x = torch.randn(4, 256, dim)

    # Standard attention
    attn_std = StandardSelfAttention(dim=dim, num_heads=num_heads)
    out_std = attn_std(x)

    # Channel-separated attention (group_size must be divisible by num_heads)
    attn_chan = ChannelSeparatedSelfAttention(dim=dim, group_size=24, num_heads=num_heads)
    out_chan = attn_chan(x)

    # Both should produce valid outputs with same shape
    assert out_std.shape == out_chan.shape
    assert not torch.isnan(out_std).any()
    assert not torch.isnan(out_chan).any()
    assert not torch.isinf(out_std).any()
    assert not torch.isinf(out_chan).any()


def test_standard_attention_no_bias():
    """Test without QKV bias."""
    attn = StandardSelfAttention(dim=192, num_heads=6, qkv_bias=False)
    x = torch.randn(4, 256, 192)
    out = attn(x)
    assert out.shape == x.shape


def test_standard_attention_batch_size_one():
    """Test with batch size 1."""
    attn = StandardSelfAttention(dim=128, num_heads=4)
    x = torch.randn(1, 100, 128)
    out = attn(x)
    assert out.shape == x.shape


def test_standard_attention_different_seq_lengths():
    """Test with various sequence lengths."""
    attn = StandardSelfAttention(dim=64, num_heads=4)

    for seq_len in [16, 32, 64, 128, 256]:
        x = torch.randn(2, seq_len, 64)
        out = attn(x)
        assert out.shape == x.shape
