from __future__ import annotations

import argparse
import json

import pytest

from scripts.calibrate_residual_gate import (
    _alpha_override,
    _gate_candidates,
    _gate_config_override,
    _gate_overrides,
    _gate_suffix,
    _guard_test_measurement,
    _is_better,
    _merge_gate_config,
    _parse_float_map,
    _parse_json_mapping,
    _record_test_measurement,
    _relative_improvement,
    _safe_alpha,
    _schedule_override,
    _select_best,
    _select_horizon_schedule,
    _test_guard_result,
    _test_measurement_key,
)


def _ledger_args() -> argparse.Namespace:
    return argparse.Namespace(
        checkpoint_source="reports/light_experiments_remote/best",
        config="configs/train_multitask_heterogeneous_light_best.yaml",
        data_root="data/pdebench",
        decoded_rollout_steps=16,
        default_alpha=0.0,
        eval_max_samples=32,
        eval_override=[],
        kind="global",
        key="all",
        metric="decoded_rollout_nrmse",
        mode="min",
        override=['operator.conditioning.sources={"task_id":3}'],
        promotion_rule=["decoded_rollout_nrmse<=1.0"],
        reference_metric_value=0.3567910081081011,
        test_min_relative_improvement=0.01,
        test_split="test",
        val_split="val",
    )


def test_safe_alpha_makes_run_name_fragment():
    assert _safe_alpha(0.42) == "0p42"
    assert _safe_alpha(-0.1) == "m0p1"


def test_alpha_override_serializes_family_and_task_maps():
    assert _alpha_override("global", "all", 0.25) == (
        "evaluation.decoded_persistence_residual_alpha=0.25"
    )
    assert _alpha_override("family", "transport", 0.42) == (
        'evaluation.decoded_persistence_residual_alpha_by_family={"transport":0.42}'
    )
    assert _alpha_override("task", "advection1d", 0.4) == (
        'evaluation.decoded_persistence_residual_alpha_by_task={"advection1d":0.4}'
    )


def test_schedule_override_serializes_family_and_task_horizon_maps():
    assert _schedule_override("global", "all", {2: 0.3, 1: 0.2}) == (
        'evaluation.decoded_persistence_residual_alpha_by_horizon={"1":0.2,"2":0.3}'
    )
    assert _schedule_override("family", "transport", {2: 0.3, 1: 0.2}) == (
        'evaluation.decoded_persistence_residual_alpha_by_family_horizon={"transport":{"1":0.2,"2":0.3}}'
    )
    assert _schedule_override("task", "advection1d", {1: 0.4}) == (
        'evaluation.decoded_persistence_residual_alpha_by_task_horizon={"advection1d":{"1":0.4}}'
    )


def test_gate_config_override_serializes_stable_json():
    assert _gate_config_override({"max_alpha": 0.9, "min_alpha": 0.1}) == (
        'evaluation.decoded_persistence_residual_gate={"max_alpha":0.9,"min_alpha":0.1}'
    )


def test_gate_overrides_include_static_and_optional_decoded_gate():
    static = _alpha_override("family", "transport", 0.2)

    assert _gate_overrides(static, None) == [static]
    assert _gate_overrides(static, {"feature_weights": {"residual_rms": -0.1}}) == [
        static,
        'evaluation.decoded_persistence_residual_gate={"feature_weights":{"residual_rms":-0.1}}',
    ]


def test_gate_candidate_helpers_parse_and_merge():
    assert _parse_json_mapping('{"bias": 0.1}', setting="candidate") == {"bias": 0.1}
    assert _parse_float_map(["residual_rms=-0.25"], setting="weights") == {"residual_rms": -0.25}
    assert _merge_gate_config(
        {"feature_weights": {"residual_rms": -0.1}, "min_alpha": 0.0},
        {"feature_weights": {"horizon_norm": 0.2}},
    ) == {"feature_weights": {"residual_rms": -0.1, "horizon_norm": 0.2}, "min_alpha": 0.0}


def test_gate_candidates_build_default_and_json_sweep_configs():
    args = argparse.Namespace(
        use_decoded_residual_gate=True,
        gate_config_candidate=['{"bias": 0.2}', '{"feature_weights": {"horizon_norm": 0.3}}'],
        gate_min_alpha=0.05,
        gate_max_alpha=0.95,
        gate_bias=0.0,
        gate_feature_weight=["residual_rms=-0.1"],
    )

    candidates = _gate_candidates(args)

    assert candidates == [
        {
            "bias": 0.2,
            "feature_weights": {"residual_rms": -0.1},
            "min_alpha": 0.05,
            "max_alpha": 0.95,
        },
        {
            "feature_weights": {"residual_rms": -0.1, "horizon_norm": 0.3},
            "min_alpha": 0.05,
            "max_alpha": 0.95,
        },
    ]


def test_gate_suffix_is_stable_for_single_and_multiple_candidates():
    assert _gate_suffix(0, 1, None) == ""
    assert _gate_suffix(0, 1, {"min_alpha": 0.0}) == "_gate"
    assert _gate_suffix(2, 3, {"min_alpha": 0.0}) == "_gate2"


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


def test_test_guard_result_skips_when_validation_gain_is_too_small():
    result = _test_guard_result(
        value=0.3556, reference=0.3568, min_relative_improvement=0.01, mode="min"
    )

    assert result["enabled"]
    assert not result["passed"]
    assert result["relative_improvement"] < 0.01


def test_test_guard_result_passes_when_disabled():
    assert _test_guard_result(
        value=0.4, reference=None, min_relative_improvement=0.01, mode="min"
    ) == {
        "enabled": False,
        "passed": True,
    }


def test_test_measurement_key_is_stable_and_tracks_selected_gate():
    args = _ledger_args()
    selected_gate = {
        "selected_gate": "horizon_schedule",
        "schedule": {"2": 0.1, "1": 0.4},
    }
    selected_overrides = [
        'evaluation.decoded_persistence_residual_alpha_by_horizon={"1":0.4,"2":0.1}'
    ]

    key = _test_measurement_key(
        args=args, selected_overrides=selected_overrides, selected_gate=selected_gate
    )
    repeat = _test_measurement_key(
        args=args, selected_overrides=list(selected_overrides), selected_gate=dict(selected_gate)
    )
    changed = _test_measurement_key(
        args=args,
        selected_overrides=[
            'evaluation.decoded_persistence_residual_alpha_by_horizon={"1":0.1,"2":0.1}'
        ],
        selected_gate=selected_gate,
    )

    assert len(key) == 64
    assert repeat == key
    assert changed != key


def test_guard_test_measurement_blocks_repeated_key(tmp_path):
    ledger_path = tmp_path / "test-ledger.json"
    ledger_path.write_text(
        json.dumps({"measurements": [{"measurement_key": "abc"}]}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="held-out test measurement already recorded"):
        _guard_test_measurement(
            ledger_path=str(ledger_path),
            measurement_key="abc",
            allow_repeat_test=False,
        )

    policy = _guard_test_measurement(
        ledger_path=str(ledger_path), measurement_key="abc", allow_repeat_test=True
    )

    assert policy["already_recorded"] is True
    assert policy["allow_repeat_test"] is True


def test_record_test_measurement_appends_once_and_skips_debug_repeats(tmp_path):
    ledger_path = tmp_path / "test-ledger.json"

    recorded = _record_test_measurement(
        ledger_path=str(ledger_path),
        measurement_key="abc",
        allow_repeat_test=False,
        metric="decoded_rollout_nrmse",
        test_metric_value=0.31,
        validation_metric_value=0.35,
        test_split="test",
        selected_gate="horizon_schedule",
        selected_overrides=["evaluation.decoded_persistence_residual_alpha_by_horizon={}"],
        run_name="candidate_test",
        summary="candidate_test/summary.json",
    )
    debug_recorded = _record_test_measurement(
        ledger_path=str(ledger_path),
        measurement_key="abc",
        allow_repeat_test=True,
        metric="decoded_rollout_nrmse",
        test_metric_value=0.3,
        validation_metric_value=0.35,
        test_split="test",
        selected_gate="horizon_schedule",
        selected_overrides=["evaluation.decoded_persistence_residual_alpha_by_horizon={}"],
        run_name="candidate_test_repeat",
        summary="candidate_test_repeat/summary.json",
    )

    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert recorded is True
    assert debug_recorded is False
    assert len(ledger["measurements"]) == 1
    assert ledger["measurements"][0]["selected_overrides"] == [
        "evaluation.decoded_persistence_residual_alpha_by_horizon={}"
    ]
    assert ledger["measurements"][0]["test_metric_value"] == 0.31


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

    schedule, selections = _select_horizon_schedule(
        rows, kind="family", key="transport", mode="min"
    )

    assert schedule == {1: 0.1, 2: 0.4}
    assert [selection["run_name"] for selection in selections] == ["alpha01", "alpha04"]


def test_select_horizon_schedule_uses_global_horizon_metrics():
    rows = [
        {
            "alpha": 0.1,
            "run_name": "alpha01",
            "summary": "a.json",
            "metrics": {
                "decoded_h1_nrmse": 0.3,
                "decoded_h2_nrmse": 0.8,
            },
        },
        {
            "alpha": 0.4,
            "run_name": "alpha04",
            "summary": "b.json",
            "metrics": {
                "decoded_h1_nrmse": 0.5,
                "decoded_h2_nrmse": 0.2,
            },
        },
    ]

    schedule, selections = _select_horizon_schedule(rows, kind="global", key="all", mode="min")

    assert schedule == {1: 0.1, 2: 0.4}
    assert [selection["metric"] for selection in selections] == [
        "decoded_h1_nrmse",
        "decoded_h2_nrmse",
    ]
