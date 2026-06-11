from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.summarize_data_budget_sweep import summarize_data_budget_sweep


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _summary(run_name: str, metric: float, *, split: str = "val") -> dict:
    return {
        "run_name": run_name,
        "duration_sec": 12.0,
        "extra": {
            "decoded_split": split,
            "decoded_task": ["burgers1d", "advection1d", "darcy2d"],
            "decoded_task_roots": {},
            "decoded_decoded_context_roll_shift_estimator": {},
            "decoded_decoded_data_conditioned_roll_shift_estimator": {},
            "decoded_decoded_observed_roll_shift_estimator": {},
            "decoded_decoded_prediction_roll_shift_estimator": {},
        },
        "extra_evaluations": {},
        "metrics": {
            "decoded_rollout_nrmse": metric,
            "decoded_step1_nrmse": 0.41,
            "decoded_h16_nrmse": metric + 0.1,
            "task_advection1d_decoded_rollout_nrmse": metric + 0.2,
            "task_burgers1d_decoded_rollout_nrmse": metric - 0.1,
            "task_darcy2d_decoded_rollout_nrmse": metric - 0.05,
        },
    }


def _baseline_payload() -> dict:
    return {
        "measurement_type": "p1_capacity_sweep_results",
        "held_out_test_data_read": False,
        "runs": {
            "persistence_medium_v1_val": {
                "metrics": {
                    "decoded_rollout_nrmse": 0.38260034902058476,
                    "decoded_h16_nrmse": 0.3709625398109224,
                }
            },
            "ups_medium_capacity_tier_b": {
                "metrics": {
                    "decoded_rollout_nrmse": 0.7449,
                    "decoded_h16_nrmse": 0.7723,
                }
            },
        },
    }


def _contract_payload() -> dict:
    return {
        "measurement_type": "p1_data_budget_sweep_contract",
        "status": "pre_registered",
        "held_out_test_data_read": False,
        "test_ledger_writes": [],
        "protocol": {
            "data": "medium-v1",
            "eval_split": "val",
            "train_sample_budgets": [128, 512],
            "eval_samples": 128,
            "decoded_rollout_steps": 16,
            "fixed_capacity_tier": "tier_b",
            "estimators": "none",
        },
        "runner": {
            "run_name_prefix": "ups_medium_data_budget",
        },
        "gate": {
            "metric": "decoded_rollout_nrmse",
            "reference_run": "persistence_medium_v1_val",
            "diagnostic_reference_run": "ups_medium_capacity_tier_b",
        },
    }


def test_summarize_data_budget_sweep_selects_best_and_records_curve(tmp_path):
    output_root = tmp_path / "reports"
    _write_json(
        output_root / "ups_medium_data_budget_n128" / "summary.json",
        _summary("ups_medium_data_budget_n128", 0.8),
    )
    _write_json(
        output_root / "ups_medium_data_budget_n512" / "summary.json",
        _summary("ups_medium_data_budget_n512", 0.7),
    )
    baseline_json = _write_json(tmp_path / "capacity.json", _baseline_payload())
    contract_json = _write_json(tmp_path / "contract.json", _contract_payload())

    record = summarize_data_budget_sweep(
        output_root=output_root,
        baseline_json=baseline_json,
        contract_json=contract_json,
        artifact="b2://bucket/data.tar.gz",
    )

    assert record["measurement_type"] == "p1_data_budget_sweep_results"
    assert record["held_out_test_data_read"] is False
    assert record["artifact"] == "b2://bucket/data.tar.gz"
    assert record["complete_budget_curve"] is True
    assert record["decision"]["best_run"] == "ups_medium_data_budget_n512"
    assert record["decision"]["best_train_samples"] == 512
    assert record["decision"]["best_metric_value"] == 0.7
    assert record["decision"]["improves_tier_b_capacity"] is True
    assert record["decision"]["improves_persistence"] is False
    assert [row["train_samples"] for row in record["budget_curve"]] == [128, 512]


def test_summarize_data_budget_sweep_records_missing_budget_without_promotion(tmp_path):
    output_root = tmp_path / "reports"
    _write_json(
        output_root / "ups_medium_data_budget_n128" / "summary.json",
        _summary("ups_medium_data_budget_n128", 0.8),
    )
    baseline_json = _write_json(tmp_path / "capacity.json", _baseline_payload())
    contract_json = _write_json(tmp_path / "contract.json", _contract_payload())

    record = summarize_data_budget_sweep(
        output_root=output_root,
        baseline_json=baseline_json,
        contract_json=contract_json,
    )

    assert record["complete_budget_curve"] is False
    assert record["missing_train_sample_budgets"] == [512]
    assert record["decision"]["held_out_test_allowed_by_this_artifact"] is False


def test_summarize_data_budget_sweep_rejects_test_split_or_estimator(tmp_path):
    output_root = tmp_path / "reports"
    leaked = _summary("ups_medium_data_budget_n128", 0.4, split="test")
    leaked["extra"]["decoded_decoded_context_roll_shift_estimator"] = {"enabled": True}
    _write_json(output_root / "ups_medium_data_budget_n128" / "summary.json", leaked)
    baseline_json = _write_json(tmp_path / "capacity.json", _baseline_payload())
    contract_json = _write_json(tmp_path / "contract.json", _contract_payload())

    with pytest.raises(ValueError, match="test"):
        summarize_data_budget_sweep(
            output_root=output_root,
            baseline_json=baseline_json,
            contract_json=contract_json,
        )


def test_summarize_data_budget_sweep_rejects_unexpected_run_name(tmp_path):
    output_root = tmp_path / "reports"
    _write_json(
        output_root / "other_run" / "summary.json",
        _summary("other_run", 0.8),
    )
    baseline_json = _write_json(tmp_path / "capacity.json", _baseline_payload())
    contract_json = _write_json(tmp_path / "contract.json", _contract_payload())

    with pytest.raises(ValueError, match="run_name"):
        summarize_data_budget_sweep(
            output_root=output_root,
            baseline_json=baseline_json,
            contract_json=contract_json,
        )
