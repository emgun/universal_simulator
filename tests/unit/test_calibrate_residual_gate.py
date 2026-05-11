from __future__ import annotations

from scripts.calibrate_residual_gate import (
    _alpha_override,
    _is_better,
    _relative_improvement,
    _safe_alpha,
    _schedule_override,
    _select_best,
    _select_horizon_schedule,
)


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


def test_schedule_override_serializes_family_and_task_horizon_maps():
    assert _schedule_override("family", "transport", {2: 0.3, 1: 0.2}) == (
        'evaluation.decoded_persistence_residual_alpha_by_family_horizon={"transport":{"1":0.2,"2":0.3}}'
    )
    assert _schedule_override("task", "advection1d", {1: 0.4}) == (
        'evaluation.decoded_persistence_residual_alpha_by_task_horizon={"advection1d":{"1":0.4}}'
    )


def test_select_best_supports_min_and_max():
    rows = [
        {"alpha": 0.1, "decoded_rollout_nrmse": 0.9},
        {"alpha": 0.4, "decoded_rollout_nrmse": 0.5},
        {"alpha": 0.8, "decoded_rollout_nrmse": 0.7},
    ]

    assert _select_best(rows, metric="decoded_rollout_nrmse", mode="min")["alpha"] == 0.4
    assert _select_best(rows, metric="decoded_rollout_nrmse", mode="max")["alpha"] == 0.1


def test_is_better_supports_min_and_max():
    assert _is_better(0.4, 0.5, mode="min")
    assert not _is_better(0.5, 0.4, mode="min")
    assert _is_better(0.5, 0.4, mode="max")
    assert not _is_better(0.4, 0.5, mode="max")


def test_relative_improvement_supports_min_and_max():
    assert round(_relative_improvement(0.45, 0.5, mode="min"), 6) == 0.1
    assert round(_relative_improvement(0.55, 0.5, mode="max"), 6) == 0.1


def test_select_horizon_schedule_uses_per_horizon_family_metrics():
    rows = [
        {
            "alpha": 0.1,
            "run_name": "alpha01",
            "summary": "a.json",
            "metrics": {
                "family_transport_decoded_h1_nrmse": 0.3,
                "family_transport_decoded_h2_nrmse": 0.8,
            },
        },
        {
            "alpha": 0.4,
            "run_name": "alpha04",
            "summary": "b.json",
            "metrics": {
                "family_transport_decoded_h1_nrmse": 0.5,
                "family_transport_decoded_h2_nrmse": 0.2,
            },
        },
    ]

    schedule, selections = _select_horizon_schedule(rows, kind="family", key="transport", mode="min")

    assert schedule == {1: 0.1, 2: 0.4}
    assert [selection["run_name"] for selection in selections] == ["alpha01", "alpha04"]
