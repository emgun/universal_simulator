from __future__ import annotations

from scripts.calibrate_roll_shift import (
    _candidate_shifts,
    _safe_shift,
    _schedule_override,
    _select_horizon_schedule,
    _shift_override,
)


def test_safe_shift_makes_run_name_fragment():
    assert _safe_shift(40) == "40"
    assert _safe_shift(-2) == "m2"


def test_candidate_shifts_use_default_or_explicit_values():
    assert 40 in _candidate_shifts(None)
    assert _candidate_shifts([1, -2, 3]) == [1, -2, 3]


def test_shift_override_serializes_task_and_family_maps():
    assert _shift_override("task", "advection1d", 40) == 'evaluation.decoded_roll_shift_by_task={"advection1d":40}'
    assert _shift_override("family", "transport", -2) == 'evaluation.decoded_roll_shift_by_family={"transport":-2}'


def test_schedule_override_serializes_horizon_maps():
    assert _schedule_override("task", "advection1d", {2: 4, 1: 3}) == (
        'evaluation.decoded_roll_shift_by_task_horizon={"advection1d":{"1":3,"2":4}}'
    )
    assert _schedule_override("family", "transport", {4: -1}) == (
        'evaluation.decoded_roll_shift_by_family_horizon={"transport":{"4":-1}}'
    )


def test_select_horizon_schedule_uses_matching_task_metrics():
    rows = [
        {
            "shift": 2,
            "run_name": "shift2",
            "summary": "a.json",
            "metrics": {
                "task_advection1d_decoded_h1_nrmse": 0.3,
                "task_advection1d_decoded_h2_nrmse": 0.8,
            },
        },
        {
            "shift": 4,
            "run_name": "shift4",
            "summary": "b.json",
            "metrics": {
                "task_advection1d_decoded_h1_nrmse": 0.5,
                "task_advection1d_decoded_h2_nrmse": 0.2,
            },
        },
    ]

    schedule, selections = _select_horizon_schedule(rows, kind="task", key="advection1d", mode="min")

    assert schedule == {1: 2, 2: 4}
    assert [selection["run_name"] for selection in selections] == ["shift2", "shift4"]
