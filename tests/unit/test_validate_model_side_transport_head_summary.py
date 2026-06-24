from __future__ import annotations

from scripts.validate_model_side_transport_head_summary import validate_summary


def _summary():
    return {
        "held_out_test_used": False,
        "held_out_test_data_read": False,
        "metrics": {
            "decoded_rollout_nrmse": 0.1,
            "task_advection1d_decoded_rollout_nrmse": 0.1,
            "task_advection1d_decoded_h16_nrmse": 0.1,
            "task_burgers1d_decoded_rollout_nrmse": 0.1,
            "task_darcy2d_decoded_rollout_nrmse": 0.1,
        },
        "extra": {
            "model_side_transport_head": {
                "enabled": True,
                "mode": "periodic_roll",
                "apply_at": "decoded_rollout",
                "tasks": ["advection1d"],
                "required_params": ["beta"],
            },
            "model_side_transport_head_metrics": {
                "beta_missing_count": 0,
                "applied_sample_count": 8,
                "skipped_sample_count": 0,
            },
            "decoded_context_roll_shift_estimator": {},
            "decoded_observed_roll_shift_estimator": {},
            "decoded_prediction_roll_shift_estimator": {},
            "decoded_data_conditioned_roll_shift_estimator": {},
        },
    }


def test_validate_model_side_transport_head_summary_accepts_clean_summary():
    assert validate_summary(_summary()) == []


def test_validate_model_side_transport_head_summary_rejects_heldout_and_sidecar():
    summary = _summary()
    summary["held_out_test_used"] = True
    summary["extra"]["decoded_data_conditioned_roll_shift_estimator"] = {"enabled": True}

    errors = validate_summary(summary)

    assert "held_out_test_used must be false" in errors
    assert (
        "extra.decoded_data_conditioned_roll_shift_estimator must be empty for model-side "
        "transport-head evidence"
    ) in errors


def test_validate_model_side_transport_head_summary_rejects_missing_beta_and_task_scope():
    summary = _summary()
    summary["extra"]["model_side_transport_head"]["tasks"] = ["advection1d", "burgers1d"]
    summary["extra"]["model_side_transport_head"]["required_params"] = []
    summary["extra"]["model_side_transport_head_metrics"]["beta_missing_count"] = 1

    errors = validate_summary(summary)

    assert "extra.model_side_transport_head.tasks must be exactly ['advection1d']" in errors
    assert "extra.model_side_transport_head.required_params must include beta" in errors
    assert "extra.model_side_transport_head_metrics.beta_missing_count must be 0" in errors
