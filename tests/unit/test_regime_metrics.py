from __future__ import annotations

import pytest
import torch

from ups.eval.regime_metrics import global_scale_regime_nrmse


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
