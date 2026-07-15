from __future__ import annotations

import copy
import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from scripts import materialize_strat_v1_shared_tier_b as materialize
from scripts import plan_strat_v1_shared_tier_b as planner
from scripts import run_strat_v1_shared_tier_b as runner
from scripts import train as train_script
from ups.data.manifests import canonical_sha256

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/d5_strat_v1_shared_tier_b.yaml"
TASKS = ("advection1d", "burgers1d", "darcy2d")


def _self_hash(payload: dict, key: str) -> dict:
    result = copy.deepcopy(payload)
    result[key] = canonical_sha256(result)
    return result


def _plan() -> dict:
    return _self_hash(
        {
            "schema_version": 1,
            "plan_id": "strat-v1-shared-tier-b-d5",
            "mode": "validation_only",
            "heldout_access": "forbidden",
            "measurement_lock_access": "forbidden",
            "gates": {
                "shared_macro_ratio_to_specialist_oracle_maximum": 1.05,
                "shared_per_task_ratio_to_specialist_maximum": 1.10,
                "persistence_maximum_by_task": {
                    "advection1d": 0.7,
                    "burgers1d": 0.7,
                    "darcy2d": 1.0,
                },
                "darcy_primary_maximum": 0.14,
                "maximum_corrected_regime_spread": 1.5,
                "shuffled_parameter_nrmse_degradation_minimum": 0.05,
                "shared_checkpoint_bytes_less_than_specialist_ensemble": True,
                "heldout_reads": 0,
            },
            "command": ["python", "scripts/run_strat_v1_shared_tier_b.py"],
        },
        "plan_sha256",
    )


def _metrics(tasks: tuple[str, ...], *, primary: float = 0.1, spread: float = 1.0) -> dict:
    metrics = {"macro_primary_nrmse": primary}
    for task in tasks:
        suffix = "decoded_solution_nrmse" if task == "darcy2d" else "decoded_rollout_nrmse"
        metrics[f"task_{task}_{suffix}"] = primary
        metrics[f"task_{task}_maximum_corrected_regime_spread_ratio"] = spread
    return metrics


def _summary(*, shuffled_degradation: float = 0.10, heldout_reads: int = 0) -> dict:
    arms = {
        "shared": {
            "tasks": list(TASKS),
            "metrics": _metrics(TASKS),
            "checkpoints": {"total_checkpoint_bytes": 100},
        }
    }
    for task in TASKS:
        arms[f"specialist-{task}"] = {
            "tasks": [task],
            "metrics": _metrics((task,)),
            "checkpoints": {"total_checkpoint_bytes": 100},
        }
    plan = _plan()
    return _self_hash(
        {
            "schema_version": 1,
            "artifact_id": "strat-v1-shared-tier-b-d5-summary",
            "status": "complete_validation_only",
            "plan_sha256": plan["plan_sha256"],
            "heldout_reads": heldout_reads,
            "arms": arms,
            "conditioning_diagnostics": {
                "relative_nrmse_degradation": shuffled_degradation,
            },
        },
        "artifact_sha256",
    )


def test_d5_config_freezes_schema_training_contract_and_stages() -> None:
    payload = planner._checked_config(CONFIG)

    assert payload["data"]["conditioning_schema"] == {
        "task_vocab": list(TASKS),
        "param_vocab": ["beta", "nu"],
    }
    assert payload["training"]["sample_balanced_operator_loss"] is True
    assert payload["training"]["canonical_steady_operator_mapping"] is True
    assert payload["training"]["lambda_semigroup"] == 0.0
    assert payload["training"]["batch_size"] == 16
    assert {
        name: payload["stages"][name]["batch_size"]
        for name in ("decoder", "operator_decoded", "joint_codec_operator")
    } == {"decoder": 2, "operator_decoded": 2, "joint_codec_operator": 2}
    assert train_script._stage_batch_size(payload, "operator") == 16
    assert train_script._stage_batch_size(payload, "decoder") == 2
    assert payload["evaluation"]["strict_stratified_metrics"] is True
    assert {
        name: payload["stages"][name]["epochs"]
        for name in ("operator", "decoder", "operator_decoded", "joint_codec_operator")
    } == {"operator": 12, "decoder": 6, "operator_decoded": 6, "joint_codec_operator": 4}


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("data", "conditioning_schema", "task_vocab"), ["darcy2d"], "task vocabulary"),
        (("training", "sample_balanced_operator_loss"), False, "sample-balanced"),
        (("training", "canonical_steady_operator_mapping"), False, "canonical steady"),
        (("training", "lambda_semigroup"), 0.1, "semigroup"),
        (("training", "batch_size"), 8, "latent-operator batch size"),
        (("stages", "decoder", "batch_size"), 4, "decoded-stage batch sizes"),
    ],
)
def test_d5_config_checker_fails_closed(
    tmp_path: Path, path: tuple[str, ...], value, match: str
) -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    cursor = payload
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    candidate = tmp_path / "candidate.yaml"
    candidate.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        planner._checked_config(candidate)


def test_stage_batch_size_rejects_nonpositive_override() -> None:
    with pytest.raises(ValueError, match="stages.decoder.batch_size must be positive"):
        train_script._stage_batch_size(
            {"training": {"batch_size": 16}, "stages": {"decoder": {"batch_size": 0}}},
            "decoder",
        )


def test_runner_plan_checker_accepts_only_untampered_validation_plan(tmp_path: Path) -> None:
    plan = _plan()
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")

    assert runner._checked_plan(path)["plan_sha256"] == plan["plan_sha256"]

    plan["mode"] = "measurement"
    path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ValueError, match="self hash"):
        runner._checked_plan(path)

    plan = _plan()
    plan["heldout_access"] = "allowed"
    plan["plan_sha256"] = canonical_sha256(
        {key: value for key, value in plan.items() if key != "plan_sha256"}
    )
    path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(PermissionError, match="held-out"):
        runner._checked_plan(path)

    plan = _plan()
    plan["measurement_lock_access"] = "allowed"
    plan["plan_sha256"] = canonical_sha256(
        {key: value for key, value in plan.items() if key != "plan_sha256"}
    )
    path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(PermissionError, match="measurement-lock"):
        runner._checked_plan(path)


def test_runner_verifies_live_config_lock_and_source_bindings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config.yaml"
    lock = tmp_path / "training.lock.json"
    source = tmp_path / "source.py"
    config.write_text("training: {}\n", encoding="utf-8")
    lock.write_text("{}\n", encoding="utf-8")
    source.write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    relative_source = source.name
    runtime = SimpleNamespace(lock=SimpleNamespace(lock_sha256="lock-contract"))
    args = Namespace(config=str(config), training_lock=str(lock))
    plan = {
        "bindings": {
            "training_lock": {
                "lock_sha256": "lock-contract",
                "file_sha256": runner._sha256(lock),
            },
            "config": {"file_sha256": runner._sha256(config)},
            "source": {
                "implementation_commit": "a" * 40,
                "files": {relative_source: runner._sha256(source)},
            },
        }
    }

    runner._verify_plan_bindings(plan, args, runtime)

    config.write_text("training: {seed: 99}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="live config"):
        runner._verify_plan_bindings(plan, args, runtime)

    config.write_text("training: {}\n", encoding="utf-8")
    source.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="live source"):
        runner._verify_plan_bindings(plan, args, runtime)


def test_runner_summary_checker_requires_validation_stratification(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    summary = {
        "extra": {"split": "val", "strict_stratified_metrics": True},
        "metrics": _metrics(TASKS),
    }
    path.write_text(json.dumps(summary), encoding="utf-8")

    checked = runner._validate_arm_summary(path, TASKS)
    assert checked["metrics"]["macro_primary_nrmse"] == pytest.approx(0.1)

    summary["extra"]["split"] = "test"
    path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(PermissionError, match="validation-only"):
        runner._validate_arm_summary(path, TASKS)

    summary["extra"] = {"split": "val", "strict_stratified_metrics": True}
    del summary["metrics"]["task_darcy2d_decoded_solution_nrmse"]
    path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="lacks strict metrics"):
        runner._validate_arm_summary(path, TASKS)


def test_materializer_passes_complete_synthetic_evidence() -> None:
    plan = _plan()
    summary = _summary()

    result = materialize.build_result(plan, summary)

    assert result["all_gates_passed"] is True
    assert all(result["gate_checks"].values())
    assert result["interpretation"] == "shared_tier_b_validated"
    assert result["metrics"]["shared_macro_ratio_to_specialist_oracle"] == pytest.approx(1.0)
    assert result["artifact_sha256"] == canonical_sha256(
        {key: value for key, value in result.items() if key != "artifact_sha256"}
    )


def test_materializer_reports_failed_parameter_use_gate() -> None:
    plan = _plan()
    summary = _summary(shuffled_degradation=0.01)

    result = materialize.build_result(plan, summary)

    assert result["all_gates_passed"] is False
    assert result["gate_checks"]["shared_parameter_use"] is False
    assert result["interpretation"] == "shared_tier_b_not_validated"


def test_materializer_rejects_macro_that_is_not_equal_task_mean() -> None:
    plan = _plan()
    summary = _summary()
    summary["arms"]["shared"]["metrics"]["macro_primary_nrmse"] = 0.05
    summary["artifact_sha256"] = canonical_sha256(
        {key: value for key, value in summary.items() if key != "artifact_sha256"}
    )

    with pytest.raises(ValueError, match="equal-task primary mean"):
        materialize.build_result(plan, summary)


def test_materializer_rejects_heldout_or_tampered_evidence() -> None:
    plan = _plan()
    heldout = _summary(heldout_reads=1)
    with pytest.raises(PermissionError, match="validation-only"):
        materialize.build_result(plan, heldout)

    summary = _summary()
    summary["conditioning_diagnostics"]["relative_nrmse_degradation"] = 0.9
    with pytest.raises(ValueError, match="artifact_sha256"):
        materialize.build_result(plan, summary)
