import torch

from ups.training.losses import (
    LossBundle,
    compute_loss_bundle,
    consistency_loss,
    edge_total_variation,
    inverse_decoding_loss,
    inverse_encoding_loss,
    one_step_loss,
    rollout_loss,
    spectral_loss,
)


def test_individual_losses_shapes():
    encoded = torch.randn(4, 8, 16)
    reconstructed = encoded + 0.1 * torch.randn_like(encoded)
    inv_enc = inverse_encoding_loss(encoded, reconstructed)
    assert inv_enc.shape == ()

    preds = {"u": torch.randn(2, 32, 3)}
    targets = {"u": preds["u"].clone()}
    inv_dec = inverse_decoding_loss(preds, targets)
    assert inv_dec == torch.tensor(0.0)

    pred_next = torch.randn(2, 8, 16)
    target_next = pred_next.clone()
    assert one_step_loss(pred_next, target_next) == torch.tensor(0.0)

    rollout = torch.randn(2, 5, 8, 16)
    assert rollout_loss(rollout, rollout.clone()) == torch.tensor(0.0)

    spec = spectral_loss(pred_next, target_next)
    assert spec == torch.tensor(0.0)

    cons = consistency_loss(pred_next, target_next)
    assert cons == torch.tensor(0.0)

    latent = torch.randn(2, 10, 6)
    edges = torch.tensor([[0, 1], [2, 3]])
    tv = edge_total_variation(latent, edges)
    assert tv >= 0


def test_compute_loss_bundle():
    encoded = torch.randn(2, 6, 12)
    reconstructed = encoded + 0.05 * torch.randn_like(encoded)
    decoded_pred = {"p": torch.randn(2, 20, 1)}
    decoded_target = {"p": torch.randn(2, 20, 1)}
    pred_next = torch.randn(2, 6, 12)
    target_next = torch.randn(2, 6, 12)
    pred_rollout = torch.randn(2, 3, 6, 12)
    target_rollout = torch.randn(2, 3, 6, 12)
    spectral_pred = torch.randn(2, 6, 12)
    spectral_target = torch.randn(2, 6, 12)
    consistency_pred = torch.randn(2, 6, 12)
    consistency_target = torch.randn(2, 6, 12)
    latent_for_tv = torch.randn(2, 10, 12)
    edges = torch.tensor([[0, 1], [1, 2], [3, 4]])

    bundle = compute_loss_bundle(
        encoded=encoded,
        reconstructed=reconstructed,
        decoded_pred=decoded_pred,
        decoded_target=decoded_target,
        pred_next=pred_next,
        target_next=target_next,
        pred_rollout=pred_rollout,
        target_rollout=target_rollout,
        spectral_pred=spectral_pred,
        spectral_target=spectral_target,
        consistency_pred=consistency_pred,
        consistency_target=consistency_target,
        latent_for_tv=latent_for_tv,
        edges=edges,
        weights={"L_inv_enc": 0.5},
    )

    assert isinstance(bundle, LossBundle)
    assert bundle.total.shape == ()
    assert len(bundle.components) == 7
    assert torch.isfinite(bundle.total)

