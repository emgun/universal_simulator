import torch

from torch import nn
from ups.training.losses import (
    LossBundle,
    compute_loss_bundle,
    compute_operator_loss_bundle,
    consistency_loss,
    edge_total_variation,
    inverse_decoding_loss,
    inverse_encoding_loss,
    one_step_loss,
    rollout_loss,
    spectral_loss,
)


def test_individual_losses_shapes():
    # Create simple stubs so inverse losses can run
    class MeanProjectDecoder(nn.Module):
        def forward(self, points, latent):
            # Map latent to a single-channel field by mean over tokens
            b, q, _ = points.shape
            m = latent.mean(dim=1, keepdim=True)  # (B, 1, D)
            u = m[..., :1].expand(b, q, 1)  # (B, Q, 1)
            return {"u": u}

    class SumLatentEncoder(nn.Module):
        def forward(self, fields, coords, meta=None):
            # Re-encode by averaging over queries and linearly projecting back to latent_dim
            u = fields["u"]  # (B, Q, 1)
            b, q, _ = u.shape
            # Produce a simple latent with fixed length and dim for test predictability
            latent_len = 8
            latent_dim = 16
            mean_u = u.mean(dim=1, keepdim=True)  # (B, 1, 1)
            return mean_u.expand(b, latent_len, 1).repeat(1, 1, latent_dim)

    B, T, D = 2, 8, 16
    N = 32
    latent = torch.randn(B, T, D)
    points = torch.randn(B, N, 2)
    decoder = MeanProjectDecoder()
    encoder = SumLatentEncoder()

    # For inverse encoding, construct input_fields to match decoder(latent)
    expected_fields = decoder(points, latent)
    inv_enc = inverse_encoding_loss(expected_fields, latent, decoder, points)
    assert inv_enc.shape == () and inv_enc == torch.tensor(0.0)

    # Inverse decoding: ensure it returns a scalar >= 0
    inv_dec = inverse_decoding_loss(latent, decoder, encoder, points, points, {"grid_shape": (1, N)})
    assert inv_dec.shape == () and inv_dec >= 0

    pred_next = torch.randn(B, T, D)
    target_next = pred_next.clone()
    assert one_step_loss(pred_next, target_next) == torch.tensor(0.0)

    rollout = torch.randn(B, 3, T, D)
    assert rollout_loss(rollout, rollout.clone()) == torch.tensor(0.0)

    spec = spectral_loss(pred_next, target_next)
    assert spec == torch.tensor(0.0)

    cons = consistency_loss(pred_next, target_next)
    assert cons == torch.tensor(0.0)

    latent_tv = torch.randn(B, 10, 6)
    edges = torch.tensor([[0, 1], [2, 3]])
    tv = edge_total_variation(latent_tv, edges)
    assert tv >= 0


def test_compute_loss_bundles():
    # New operator loss bundle
    pred_next = torch.randn(2, 6, 12)
    target_next = pred_next.clone()
    roll_pred = torch.randn(2, 2, 6, 12)
    roll_tgt = roll_pred.clone()
    spec_pred = torch.randn(2, 6, 12)
    spec_tgt = spec_pred.clone()

    op_bundle = compute_operator_loss_bundle(
        pred_next=pred_next,
        target_next=target_next,
        pred_rollout=roll_pred,
        target_rollout=roll_tgt,
        spectral_pred=spec_pred,
        spectral_target=spec_tgt,
        weights={"lambda_forward": 1.0, "lambda_rollout": 1.0, "lambda_spectral": 1.0},
    )
    assert isinstance(op_bundle, LossBundle)
    assert op_bundle.total.shape == ()
    assert torch.isfinite(op_bundle.total)

    # Deprecated wrapper remains usable
    dep_bundle = compute_loss_bundle(
        pred_next=pred_next,
        target_next=target_next,
        pred_rollout=roll_pred,
        target_rollout=roll_tgt,
        spectral_pred=spec_pred,
        spectral_target=spec_tgt,
        weights={"lambda_forward": 1.0, "lambda_rollout": 1.0, "lambda_spectral": 1.0},
    )
    assert isinstance(dep_bundle, LossBundle)
    assert dep_bundle.total.shape == ()
    assert torch.isfinite(dep_bundle.total)
