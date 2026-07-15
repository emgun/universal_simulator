import pytest
import torch

from scripts.train import (
    _canonical_raw_supervision,
    _decoded_field_loss,
    _decoded_rollout_training_loss,
    _decoded_rollout_training_window,
    _source_sample_balanced_loss,
    _task_loss_weight,
    _transport_shift_consistency_loss,
)
from ups.training.losses import (
    LossBundle,
    compute_loss_bundle,
    consistency_loss,
    edge_total_variation,
    inverse_decoding_loss,
    inverse_encoding_loss,
    one_step_loss,
    rollout_loss,
    semigroup_consistency_loss,
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
    assert semigroup_consistency_loss(pred_next, target_next) >= 0

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


def test_semigroup_consistency_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        semigroup_consistency_loss(torch.randn(2, 3), torch.randn(2, 4))


def test_decoded_field_loss_can_weight_persistence_residual():
    previous = torch.zeros(1, 4, 1)
    target = torch.ones(1, 4, 1)
    pred = torch.full((1, 4, 1), 0.5)

    base = _decoded_field_loss(pred, target, previous, stage_cfg={})
    residual = _decoded_field_loss(
        pred,
        target,
        previous,
        stage_cfg={"lambda_persistence_residual": 1.0},
    )

    assert residual > base


def test_task_loss_weight_uses_explicit_task_weight():
    stage_cfg = {"task_loss_weights": {"advection1d": 3.0, "burgers1d": 0.5}}

    assert _task_loss_weight(stage_cfg, "advection1d") == 3.0
    assert _task_loss_weight(stage_cfg, "burgers1d") == 0.5
    assert _task_loss_weight(stage_cfg, "darcy2d") == 1.0


def test_decoded_rollout_training_loss_preserves_uniform_default():
    decoded_losses = [torch.tensor(1.0), torch.tensor(2.0), torch.tensor(4.0)]

    loss = _decoded_rollout_training_loss(
        decoded_losses,
        stage_cfg={},
        lambda_rollout=0.5,
    )

    assert loss == torch.tensor(2.5)


def test_decoded_rollout_training_loss_can_emphasize_late_horizons():
    decoded_losses = [torch.tensor(1.0), torch.tensor(2.0), torch.tensor(4.0)]

    uniform = _decoded_rollout_training_loss(
        decoded_losses,
        stage_cfg={},
        lambda_rollout=1.0,
    )
    weighted = _decoded_rollout_training_loss(
        decoded_losses,
        stage_cfg={"rollout_loss_horizon_power": 2.0},
        lambda_rollout=1.0,
    )

    assert weighted > uniform


def test_decoded_rollout_training_loss_rejects_negative_horizon_power():
    with pytest.raises(ValueError):
        _decoded_rollout_training_loss(
            [torch.tensor(1.0), torch.tensor(2.0)],
            stage_cfg={"rollout_loss_horizon_power": -1.0},
            lambda_rollout=1.0,
        )


def test_decoded_rollout_training_window_preserves_zero_start_default():
    fields = torch.arange(6).view(1, 6, 1)

    window, start = _decoded_rollout_training_window(
        fields,
        rollout_steps=3,
        stage_cfg={},
    )

    assert start == 0
    assert torch.equal(window[:, :, 0], torch.tensor([[0, 1, 2, 3]]))


def test_decoded_rollout_training_window_can_select_latest_window():
    fields = torch.arange(6).view(1, 6, 1)

    window, start = _decoded_rollout_training_window(
        fields,
        rollout_steps=3,
        stage_cfg={"rollout_start_strategy": "latest"},
    )

    assert start == 2
    assert torch.equal(window[:, :, 0], torch.tensor([[2, 3, 4, 5]]))


def test_decoded_rollout_training_window_rejects_unknown_strategy():
    with pytest.raises(ValueError):
        _decoded_rollout_training_window(
            torch.arange(4).view(1, 4, 1),
            rollout_steps=2,
            stage_cfg={"rollout_start_strategy": "middle"},
        )


def test_canonical_raw_supervision_maps_darcy_coefficient_to_solution_once():
    coefficients = torch.zeros(2, 1, 1, 4, 4)
    solutions = torch.ones(2, 1, 1, 4, 4)

    supervision, is_steady = _canonical_raw_supervision(
        coefficients,
        solutions,
        task_name="darcy2d",
        enabled=True,
    )

    assert is_steady is True
    assert supervision.shape == (2, 2, 1, 4, 4)
    assert torch.equal(supervision[:, :1], coefficients)
    assert torch.equal(supervision[:, 1:], solutions)


def test_canonical_raw_supervision_preserves_temporal_and_legacy_inputs():
    trajectory = torch.arange(6, dtype=torch.float32).view(1, 3, 1, 2)

    temporal, temporal_is_steady = _canonical_raw_supervision(
        trajectory,
        trajectory.clone(),
        task_name="advection1d",
        enabled=True,
    )
    legacy, legacy_is_steady = _canonical_raw_supervision(
        trajectory[:, :1],
        torch.ones_like(trajectory[:, :1]),
        task_name="darcy2d",
        enabled=False,
    )

    assert temporal_is_steady is False
    assert torch.equal(temporal, trajectory)
    assert legacy_is_steady is False
    assert torch.equal(legacy, trajectory[:, :1])


def test_source_sample_balanced_loss_does_not_overweight_long_trajectories():
    # Sample one has three unit-error transitions; sample two has one transition
    # with error 3. Equal-source weighting is (1 + 9) / 2 = 5, rather than the
    # flattened transition mean (1 + 1 + 1 + 9) / 4 = 3.
    prediction = torch.tensor([[1.0], [1.0], [1.0], [3.0]])
    target = torch.zeros_like(prediction)

    loss = _source_sample_balanced_loss(
        prediction,
        target,
        torch.tensor([4, 2]),
        torch.nn.functional.mse_loss,
    )

    assert loss == torch.tensor(5.0)


def test_source_sample_balanced_loss_rejects_inconsistent_grouping():
    with pytest.raises(ValueError, match="transition counts"):
        _source_sample_balanced_loss(
            torch.zeros(3, 1),
            torch.zeros(3, 1),
            torch.tensor([2, 2]),
            torch.nn.functional.mse_loss,
        )


def test_transport_shift_consistency_loss_is_default_off():
    previous = torch.arange(4, dtype=torch.float32).view(1, 4, 1)
    pred = torch.roll(previous, shifts=1, dims=1)

    loss = _transport_shift_consistency_loss(
        pred,
        previous,
        stage_cfg={},
        task_name="advection1d",
    )

    assert loss is None


def test_transport_shift_consistency_loss_uses_task_shift():
    previous = torch.arange(4, dtype=torch.float32).view(1, 4, 1)
    pred = torch.roll(previous, shifts=1, dims=1)

    loss = _transport_shift_consistency_loss(
        pred,
        previous,
        stage_cfg={
            "transport_shift_consistency_lambda": 2.0,
            "transport_shift_consistency_by_task": {"advection1d": 1},
        },
        task_name="advection1d",
    )

    assert loss == torch.tensor(0.0)


def test_transport_shift_consistency_loss_ignores_unconfigured_task():
    previous = torch.arange(4, dtype=torch.float32).view(1, 4, 1)

    loss = _transport_shift_consistency_loss(
        previous,
        previous,
        stage_cfg={
            "transport_shift_consistency_lambda": 1.0,
            "transport_shift_consistency_by_task": {"advection1d": 1},
        },
        task_name="burgers1d",
    )

    assert loss is None


def test_transport_shift_consistency_loss_rejects_negative_weight():
    with pytest.raises(ValueError):
        _transport_shift_consistency_loss(
            torch.zeros(1, 4, 1),
            torch.zeros(1, 4, 1),
            stage_cfg={"transport_shift_consistency_lambda": -1.0},
            task_name="advection1d",
        )
