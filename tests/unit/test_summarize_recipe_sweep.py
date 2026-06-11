from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.summarize_recipe_sweep import summarize_recipe_sweep


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _summary(
    run_name: str,
    metric: float,
    *,
    split: str = "val",
    decoded_metadata: bool = False,
) -> dict:
    extra = {
        "split": split,
        "task": ["burgers1d", "advection1d", "darcy2d"],
        "task_roots": {},
        "decoded_context_roll_shift_estimator": {},
        "decoded_data_conditioned_roll_shift_estimator": {},
        "decoded_observed_roll_shift_estimator": {},
        "decoded_prediction_roll_shift_estimator": {},
    }
    if decoded_metadata:
        extra = {
            "decoded_split": split,
            "decoded_task": ["burgers1d", "advection1d", "darcy2d"],
            "decoded_task_roots": {},
            "decoded_decoded_context_roll_shift_estimator": {},
            "decoded_decoded_data_conditioned_roll_shift_estimator": {},
            "decoded_decoded_observed_roll_shift_estimator": {},
            "decoded_decoded_prediction_roll_shift_estimator": {},
        }
    return {
        "run_name": run_name,
        "duration_sec": 12.0,
        "extra": extra,
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
        "protocol": {
            "data": "medium-v1",
            "eval_split": "val",
            "train_samples": 512,
            "eval_samples": 128,
            "decoded_rollout_steps": 16,
            "decoded_persistence_residual_alpha": 1.0,
            "estimators": "none",
        },
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
        "measurement_type": "p1_rollout_stability_recipe_sweep_contract",
        "status": "pre_registered",
        "held_out_test_data_read": False,
        "test_ledger_writes": [],
        "protocol": {
            "data": "medium-v1",
            "eval_split": "val",
            "train_samples": 512,
            "eval_samples": 128,
            "decoded_rollout_steps": 16,
            "fixed_capacity_tier": "tier_b",
            "estimators": "none",
        },
        "recipes": ["r_rollout8", "r_hpower"],
        "gate": {
            "beat_persistence_metric": "decoded_rollout_nrmse",
            "reference_run": "persistence_medium_v1_val",
            "continue_if_best_below": 0.38260034902058476,
        },
    }


def test_summarize_recipe_sweep_selects_best_and_compares_to_baselines(tmp_path):
    output_root = tmp_path / "reports"
    _write_json(
        output_root / "ups_medium_recipe_r_rollout8" / "summary.json",
        _summary("ups_medium_recipe_r_rollout8", 0.51),
    )
    _write_json(
        output_root / "ups_medium_recipe_r_hpower" / "summary.json",
        _summary("ups_medium_recipe_r_hpower", 0.49),
    )
    baseline_json = _write_json(tmp_path / "capacity.json", _baseline_payload())
    contract_json = _write_json(tmp_path / "contract.json", _contract_payload())

    record = summarize_recipe_sweep(
        output_root=output_root,
        baseline_json=baseline_json,
        contract_json=contract_json,
        artifact="b2://bucket/recipe.tar.gz",
    )

    assert record["measurement_type"] == "p1_rollout_stability_recipe_sweep_results"
    assert record["held_out_test_data_read"] is False
    assert record["artifact"] == "b2://bucket/recipe.tar.gz"
    assert record["decision"]["best_run"] == "ups_medium_recipe_r_hpower"
    assert record["decision"]["best_metric_value"] == 0.49
    assert record["decision"]["improves_tier_b_capacity"] is True
    assert record["decision"]["improves_persistence"] is False
    assert (
        record["baselines"]["persistence_medium_v1_val"]["decoded_rollout_nrmse"]
        == 0.38260034902058476
    )
    assert sorted(record["runs"]) == ["ups_medium_recipe_r_hpower", "ups_medium_recipe_r_rollout8"]


def test_summarize_recipe_sweep_rejects_test_split_or_extra_test_eval(tmp_path):
    output_root = tmp_path / "reports"
    leaked = _summary("ups_medium_recipe_r_rollout8", 0.4, split="test")
    leaked["extra_evaluations"] = {"test": {"summary": "summary_test.json"}}
    _write_json(output_root / "ups_medium_recipe_r_rollout8" / "summary.json", leaked)
    baseline_json = _write_json(tmp_path / "capacity.json", _baseline_payload())
    contract_json = _write_json(tmp_path / "contract.json", _contract_payload())

    with pytest.raises(ValueError, match="test"):
        summarize_recipe_sweep(
            output_root=output_root,
            baseline_json=baseline_json,
            contract_json=contract_json,
        )


def test_summarize_recipe_sweep_accepts_decoded_runner_metadata(tmp_path):
    output_root = tmp_path / "reports"
    _write_json(
        output_root / "ups_medium_recipe_r_rollout8" / "summary.json",
        _summary("ups_medium_recipe_r_rollout8", 0.51, decoded_metadata=True),
    )
    baseline_json = _write_json(tmp_path / "capacity.json", _baseline_payload())
    contract_json = _write_json(tmp_path / "contract.json", _contract_payload())

    record = summarize_recipe_sweep(
        output_root=output_root,
        baseline_json=baseline_json,
        contract_json=contract_json,
    )

    assert record["decision"]["best_run"] == "ups_medium_recipe_r_rollout8"
    assert (
        record["runs"]["ups_medium_recipe_r_rollout8"]["metrics"]["decoded_rollout_nrmse"] == 0.51
    )


def test_summarize_recipe_sweep_rejects_decoded_runner_estimator_metadata(tmp_path):
    output_root = tmp_path / "reports"
    leaked = _summary("ups_medium_recipe_r_rollout8", 0.4, decoded_metadata=True)
    leaked["extra"]["decoded_decoded_context_roll_shift_estimator"] = {"enabled": True}
    _write_json(output_root / "ups_medium_recipe_r_rollout8" / "summary.json", leaked)
    baseline_json = _write_json(tmp_path / "capacity.json", _baseline_payload())
    contract_json = _write_json(tmp_path / "contract.json", _contract_payload())

    with pytest.raises(ValueError, match="roll_shift_estimator"):
        summarize_recipe_sweep(
            output_root=output_root,
            baseline_json=baseline_json,
            contract_json=contract_json,
        )


def test_summarize_recipe_sweep_rejects_empty_output_root(tmp_path):
    baseline_json = _write_json(tmp_path / "capacity.json", _baseline_payload())
    contract_json = _write_json(tmp_path / "contract.json", _contract_payload())

    with pytest.raises(FileNotFoundError, match="summary.json"):
        summarize_recipe_sweep(
            output_root=tmp_path / "empty",
            baseline_json=baseline_json,
            contract_json=contract_json,
        )
