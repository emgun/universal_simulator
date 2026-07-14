from __future__ import annotations

import pytest
import torch

from ups.eval.regime_metrics import (
    aligned_element_count,
    global_scale_regime_nrmse,
    regime_spread_ratio,
    weighted_reconstructed_nrmse,
)


def test_aligned_element_count_counts_chunks_and_rejects_mismatch():
    assert (
        aligned_element_count([torch.ones(2), torch.ones(3)], [torch.zeros(2), torch.zeros(3)]) == 5
    )
    with pytest.raises(ValueError, match="shapes"):
        aligned_element_count([torch.ones(2)], [torch.zeros(3)])


def test_weighted_reconstruction_and_spread_gate_boundary():
    reconstructed = weighted_reconstructed_nrmse([1.0, 2.0], [3, 1])
    assert reconstructed == pytest.approx((7 / 4) ** 0.5)
    assert regime_spread_ratio(1.5, 1.0) == pytest.approx(1.5)
    assert regime_spread_ratio(1.500001, 1.0) > 1.5


def test_weighted_reconstruction_rejects_bad_counts_and_metrics():
    with pytest.raises(ValueError, match="positive"):
        weighted_reconstructed_nrmse([1.0], [0])
    with pytest.raises(ValueError, match="finite"):
        weighted_reconstructed_nrmse([float("nan")], [1])
    with pytest.raises(ValueError, match="positive"):
        regime_spread_ratio(1.0, 0.0)


def test_global_scale_metric_uses_one_task_denominator_across_regimes():
    low_scale_target = torch.full((4,), 0.01)
    high_scale_target = torch.full((4,), 10.0)
    low_scale_prediction = low_scale_target + 0.1
    high_scale_prediction = high_scale_target + 0.1
    task_targets = [low_scale_target, high_scale_target]

    low = global_scale_regime_nrmse([low_scale_prediction], [low_scale_target], task_targets)
    high = global_scale_regime_nrmse([high_scale_prediction], [high_scale_target], task_targets)

    # Slice-normalized NRMSE differs by 1000x here. One task denominator makes
    # equal absolute errors comparable across regimes.
    assert low == pytest.approx(high, rel=1e-5)


def test_global_scale_metric_preserves_real_error_differences():
    task_targets = [torch.ones(4), torch.full((4,), 2.0)]
    small_error = global_scale_regime_nrmse([torch.full((4,), 1.1)], [torch.ones(4)], task_targets)
    large_error = global_scale_regime_nrmse([torch.full((4,), 1.4)], [torch.ones(4)], task_targets)
    assert large_error == pytest.approx(4 * small_error)


def test_global_scale_metric_is_invariant_to_task_chunk_boundaries():
    prediction = torch.tensor([1.25, 1.75], dtype=torch.float32)
    target = torch.tensor([1.0, 2.0], dtype=torch.float32)
    task_target = torch.tensor([1.0e-4, 1.0, 2.0, 1000.0], dtype=torch.float32)

    contiguous = global_scale_regime_nrmse([prediction], [target], [task_target])
    chunked = global_scale_regime_nrmse(
        [prediction[:1], prediction[1:]],
        [target[:1], target[1:]],
        [task_target[:1], task_target[1:3], task_target[3:]],
    )

    assert contiguous == pytest.approx(chunked, rel=1e-12)


def test_global_scale_metric_rejects_nonfinite_values():
    with pytest.raises(ValueError, match="finite"):
        global_scale_regime_nrmse(
            [torch.tensor([float("nan")])],
            [torch.ones(1)],
            [torch.ones(1)],
        )


def test_weighted_regime_global_metrics_reconstruct_task_nrmse():
    first_target = torch.tensor([1.0, 2.0], dtype=torch.float64)
    second_target = torch.tensor([3.0, 4.0, 5.0], dtype=torch.float64)
    first_prediction = first_target + torch.tensor([0.2, -0.1])
    second_prediction = second_target + torch.tensor([-0.3, 0.4, -0.2])
    task_targets = [first_target, second_target]

    first = global_scale_regime_nrmse([first_prediction], [first_target], task_targets)
    second = global_scale_regime_nrmse([second_prediction], [second_target], task_targets)
    reconstructed_squared = (2 * first**2 + 3 * second**2) / 5
    task_error = torch.cat([first_prediction - first_target, second_prediction - second_target])
    task_target = torch.cat(task_targets)
    expected_squared = float(task_error.square().mean() / (task_target.square().mean() + 1e-8))

    assert reconstructed_squared == pytest.approx(expected_squared, rel=1e-12)


@pytest.mark.parametrize(
    "predictions,targets,task_targets",
    [
        ([], [], [torch.ones(1)]),
        ([torch.ones(1)], [], [torch.ones(1)]),
        ([torch.ones(2)], [torch.ones(1)], [torch.ones(1)]),
        ([torch.ones(1)], [torch.ones(1)], []),
    ],
)
def test_global_scale_metric_rejects_invalid_inputs(predictions, targets, task_targets):
    with pytest.raises(ValueError):
        global_scale_regime_nrmse(predictions, targets, task_targets)
