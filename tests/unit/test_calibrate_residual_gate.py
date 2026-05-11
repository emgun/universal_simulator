from __future__ import annotations

from scripts.calibrate_residual_gate import _alpha_override, _safe_alpha, _select_best


def test_safe_alpha_makes_run_name_fragment():
    assert _safe_alpha(0.42) == "0p42"
    assert _safe_alpha(-0.1) == "m0p1"


def test_alpha_override_serializes_family_and_task_maps():
    assert _alpha_override("family", "transport", 0.42) == (
        'evaluation.decoded_persistence_residual_alpha_by_family={"transport":0.42}'
    )
    assert _alpha_override("task", "advection1d", 0.4) == (
        'evaluation.decoded_persistence_residual_alpha_by_task={"advection1d":0.4}'
    )


def test_select_best_supports_min_and_max():
    rows = [
        {"alpha": 0.1, "decoded_rollout_nrmse": 0.9},
        {"alpha": 0.4, "decoded_rollout_nrmse": 0.5},
        {"alpha": 0.8, "decoded_rollout_nrmse": 0.7},
    ]

    assert _select_best(rows, metric="decoded_rollout_nrmse", mode="min")["alpha"] == 0.4
    assert _select_best(rows, metric="decoded_rollout_nrmse", mode="max")["alpha"] == 0.1
