from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from scripts import materialize_strat_v1_modular_shared_trunk as materialize
from scripts import plan_strat_v1_modular_shared_trunk as planner
from ups.data.manifests import canonical_sha256

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/d6_strat_v1_modular_shared_trunk.yaml"
TASKS = ("advection1d", "burgers1d", "darcy2d")
ARMS = (
    "joint-modular",
    "ablation-advection1d",
    "ablation-burgers1d",
    "ablation-darcy2d",
)
OBJECTS = materialize.FROZEN_STAGE_OBJECTS


def _self_hash(payload: dict, key: str) -> dict:
    result = copy.deepcopy(payload)
    result[key] = canonical_sha256(result)
    return result


def _plan() -> dict:
    return _self_hash(
        {
            "schema_version": 1,
            "plan_id": "strat-v1-modular-shared-trunk-d6",
            "mode": "validation_only",
            "heldout_access": "forbidden",
            "measurement_lock_access": "forbidden",
            "bindings": {
                "training_lock": {"lock_sha256": "training-lock", "objects": OBJECTS},
                "config": {"file_sha256": "config-file"},
            },
            "design": {
                "arms": list(ARMS),
                "adapter_inventory_by_arm": {arm: list(TASKS) for arm in ARMS},
            },
            "frozen_references": {
                "d5_specialist_by_task": {task: 0.1 for task in TASKS},
                "d5_specialist_macro_primary_nrmse": 0.1,
                "d5_specialist_ensemble_checkpoint_bytes": 1_000,
            },
            "gates": {
                "u1": {
                    "joint_macro_ratio_to_frozen_d5_specialist_maximum": 1.10,
                    "joint_per_task_ratio_to_frozen_d5_specialist_maximum": 1.20,
                    "persistence_maximum_by_task": {task: 0.2 for task in TASKS},
                    "darcy_primary_maximum": 0.14,
                    "maximum_corrected_regime_spread": 1.5,
                    "shuffled_parameter_nrmse_degradation_minimum": 0.05,
                    "joint_checkpoint_bytes_less_than_frozen_d5_ensemble": True,
                    "joint_initialized_tensor_elements_less_than_matched_ablation_ensemble": True,
                    "heldout_reads": 0,
                },
                "u2": {
                    "joint_macro_ratio_to_matched_ablation_macro_maximum": 1.05,
                    "joint_per_task_ratio_to_matched_ablation_maximum": 1.10,
                    "update_parity_required": True,
                },
            },
        },
        "plan_sha256",
    )


def _metrics(tasks: tuple[str, ...], primary: float = 0.1) -> dict:
    metrics = {"macro_primary_nrmse": primary}
    for task in tasks:
        suffix = "decoded_solution_nrmse" if task == "darcy2d" else "decoded_rollout_nrmse"
        metrics[f"task_{task}_{suffix}"] = primary
        metrics[f"task_{task}_maximum_corrected_regime_spread_ratio"] = 1.0
    return metrics


def _stage_report() -> dict:
    return _self_hash(
        {
            "schema_version": 1,
            "status": "complete",
            "lock_sha256": "training-lock",
            "object_count": len(OBJECTS),
            "objects": [
                {
                    "id": object_id,
                    "role": object_id.rsplit("-", 1)[1],
                    "checksum": {"algorithm": "sha256", "value": digest},
                }
                for object_id, digest in OBJECTS.items()
            ],
        },
        "artifact_sha256",
    )


def _attempt(summary_sha: str, wall: float, rss: int) -> dict:
    return _self_hash(
        {
            "schema_version": 1,
            "summary_file_sha256": summary_sha,
            "wall_time_sec_observed_by_orchestrator": wall,
            "child_process_family_max_rss_kib_high_watermark": rss,
            "rss_scope": (
                "cumulative RUSAGE_CHILDREN process-family high-water mark; "
                "not attributable to this arm alone"
            ),
        },
        "artifact_sha256",
    )


def _summary(*, joint: float = 0.1, ablation: float = 0.1) -> dict:
    arms = {
        "joint-modular": {
            "tasks": list(TASKS),
            "adapter_inventory": list(TASKS),
            "adapter_bottleneck_dim": 16,
            "summary_file_sha256": "joint-summary",
            "attempt_evidence": _attempt("joint-summary", 2.5, 2048),
            "metrics": _metrics(TASKS, joint),
            "checkpoints": {
                "total_checkpoint_bytes": 500,
                "total_initialized_tensor_elements": 600,
                "total_adapter_tensor_elements": 60,
            },
            "resources": {
                "wall_time_sec_observed_by_orchestrator": 2.5,
                "child_process_family_max_rss_kib_high_watermark": 2048,
                "training_log": {
                    "present": True,
                    "records": 10,
                    "reported_epoch_time_sec": 2.0,
                },
            },
        }
    }
    for task in TASKS:
        arms[f"ablation-{task}"] = {
            "tasks": [task],
            "adapter_inventory": list(TASKS),
            "adapter_bottleneck_dim": 16,
            "summary_file_sha256": f"{task}-summary",
            "attempt_evidence": _attempt(f"{task}-summary", 1.5, 1024),
            "metrics": _metrics((task,), ablation),
            "checkpoints": {
                "total_checkpoint_bytes": 500,
                "total_initialized_tensor_elements": 300,
                "total_adapter_tensor_elements": 30,
            },
            "resources": {
                "wall_time_sec_observed_by_orchestrator": 1.5,
                "child_process_family_max_rss_kib_high_watermark": 1024,
                "training_log": {
                    "present": True,
                    "records": 10,
                    "reported_epoch_time_sec": 1.0,
                },
            },
        }
    exposure = {
        task: {
            "source_examples": 100,
            "scheduled_compute_units": 10,
        }
        for task in TASKS
    }
    return _self_hash(
        {
            "schema_version": 1,
            "artifact_id": "strat-v1-modular-shared-trunk-d6-summary",
            "status": "complete_validation_only",
            "plan_sha256": _plan()["plan_sha256"],
            "training_lock_sha256": "training-lock",
            "config_sha256": "config-file",
            "stage_report_artifact_sha256": _stage_report()["artifact_sha256"],
            "heldout_reads": 0,
            "heldout_evidence": {
                "requested_roles": ["train", "valid"],
                "contains_test_object": False,
                "evaluation_splits": ["val"],
            },
            "duration_sec": 5.0,
            "arms": arms,
            "update_parity": {
                "comparison": "joint_task_to_matching_ablation",
                "joint_by_task": exposure,
                "ablation_by_task": copy.deepcopy(exposure),
                "total_scheduled_optimizer_updates_by_arm": {
                    "joint-modular": 30,
                    "ablation-advection1d": 10,
                    "ablation-burgers1d": 10,
                    "ablation-darcy2d": 10,
                },
            },
            "conditioning_diagnostics": {
                "reference_macro_primary_nrmse": joint,
                "shuffled_macro_primary_nrmse": joint * 1.10,
                "relative_nrmse_degradation": 0.10,
            },
        },
        "artifact_sha256",
    )


def _rehash(summary: dict) -> None:
    summary["artifact_sha256"] = canonical_sha256(
        {key: value for key, value in summary.items() if key != "artifact_sha256"}
    )


def test_config_freezes_four_arms_bottleneck_and_full_inventory() -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    modular = payload["modular_shared_trunk"]

    assert tuple(modular["arms"]) == ARMS
    assert modular["adapter_bottleneck_dim"] == 16
    assert all(tuple(arm["adapter_inventory"]) == TASKS for arm in modular["arms"].values())
    assert payload["operator"]["routed_adapters"] == {
        "enabled": True,
        "route_source": "task_id",
        "route_vocab": list(TASKS),
        "bottleneck_dim": 16,
        "input_enabled": True,
        "output_enabled": True,
        "zero_init": True,
    }
    assert payload["update_parity"]["fail_closed_on_missing_or_mismatch"] is True
    assert payload["update_parity"]["dimensions"] == [
        "source_examples",
        "scheduled_compute_units",
    ]
    assert (
        payload["update_parity"]["efficiency_reporting"]
        == "total_scheduled_optimizer_updates_by_arm"
    )
    assert all("trainable_adapters" not in arm for arm in modular["arms"].values())
    assert payload["training"]["patience"] is None
    assert payload["training"]["fail_on_oom"] is True
    assert planner._checked_config(CONFIG) == payload


def test_planner_binds_complete_runtime_and_remote_surface() -> None:
    paths = set(planner.source_paths())

    assert {
        "configs/d6_strat_v1_modular_shared_trunk.yaml",
        "configs/d5_strat_v1_shared_tier_b.yaml",
        "configs/a4_strat_v1_baselines.yaml",
        "configs/train_multitask_heterogeneous_light_best.yaml",
        "configs/defaults.yaml",
        "scripts/run_light_experiment.py",
        "scripts/run_remote_strat_v1_modular_shared_trunk.sh",
        "scripts/launch_strat_v1_modular_shared_trunk_vast.sh",
        "scripts/generate_b2_presigned_bundle.py",
        "scripts/d5_presigned_io.py",
        "scripts/finalize_d5_presigned_transfer.py",
    } <= paths
    assert "src/ups/models/latent_operator.py" in paths
    assert all(path.startswith("src/ups/") for path in paths if path.startswith("src/"))


def test_materializer_passes_separate_u1_and_u2() -> None:
    result = materialize.build_result(_plan(), _summary(), _stage_report())

    assert result["u1_passed"] is True
    assert result["u2_passed"] is True
    assert result["all_preregistered_gates_passed"] is True
    assert result["interpretation"] == "modular_shared_trunk_validated"
    assert result["u1_checks"]["initialized_tensor_consolidation"] is True
    assert result["resource_evidence"]["run_duration_sec"] == pytest.approx(5.0)
    assert result["update_parity"]["optimizer_update_efficiency"][
        "joint_to_ablation_ensemble_ratio"
    ] == pytest.approx(1.0)
    assert "gpu" not in str(result).lower()


def test_materializer_reports_u1_and_u2_independently() -> None:
    u1_failure = materialize.build_result(
        _plan(), _summary(joint=0.15, ablation=0.15), _stage_report()
    )
    assert u1_failure["u1_passed"] is False
    assert u1_failure["u2_passed"] is True
    assert u1_failure["interpretation"] == "u1_failed"

    u2_failure = materialize.build_result(
        _plan(), _summary(joint=0.1, ablation=0.08), _stage_report()
    )
    assert u2_failure["u1_passed"] is True
    assert u2_failure["u2_passed"] is False
    assert u2_failure["interpretation"] == "u2_negative_transfer"


def test_materializer_fails_closed_on_update_mismatch() -> None:
    summary = _summary()
    summary["update_parity"]["ablation_by_task"]["burgers1d"]["source_examples"] = 99
    _rehash(summary)

    with pytest.raises(ValueError, match="update parity mismatch"):
        materialize.build_result(_plan(), summary, _stage_report())


def test_materializer_recomputes_shuffle_degradation() -> None:
    summary = _summary()
    summary["conditioning_diagnostics"]["relative_nrmse_degradation"] = 0.50
    _rehash(summary)
    with pytest.raises(ValueError, match="internally inconsistent"):
        materialize.build_result(_plan(), summary, _stage_report())

    summary = _summary()
    summary["conditioning_diagnostics"]["reference_macro_primary_nrmse"] = 0.2
    _rehash(summary)
    with pytest.raises(ValueError, match="reference differs"):
        materialize.build_result(_plan(), summary, _stage_report())


def test_materializer_requires_exact_stage_report_binding() -> None:
    stage = _stage_report()
    stage["objects"][0]["checksum"]["value"] = "f" * 64
    stage["artifact_sha256"] = canonical_sha256(
        {key: value for key, value in stage.items() if key != "artifact_sha256"}
    )
    with pytest.raises(ValueError, match="objects differ"):
        materialize.build_result(_plan(), _summary(), stage)

    stage = _stage_report()
    summary = _summary()
    summary["stage_report_artifact_sha256"] = "0" * 64
    _rehash(summary)
    with pytest.raises(ValueError, match="not bound"):
        materialize.build_result(_plan(), summary, stage)

    stage = _stage_report()
    stage["objects"].append(copy.deepcopy(stage["objects"][0]))
    stage["artifact_sha256"] = canonical_sha256(
        {key: value for key, value in stage.items() if key != "artifact_sha256"}
    )
    with pytest.raises(ValueError, match="exactly"):
        materialize.build_result(_plan(), _summary(), stage)

    stage = _stage_report()
    next(item for item in stage["objects"] if item["id"].endswith("-valid"))["role"] = "train"
    stage["artifact_sha256"] = canonical_sha256(
        {key: value for key, value in stage.items() if key != "artifact_sha256"}
    )
    with pytest.raises(ValueError, match="role differs"):
        materialize.build_result(_plan(), _summary(), stage)

    reduced_plan = _plan()
    reduced_plan["bindings"]["training_lock"]["objects"] = dict(list(OBJECTS.items())[:2])
    reduced_plan["plan_sha256"] = canonical_sha256(
        {key: value for key, value in reduced_plan.items() if key != "plan_sha256"}
    )
    reduced_summary = _summary()
    reduced_summary["plan_sha256"] = reduced_plan["plan_sha256"]
    _rehash(reduced_summary)
    with pytest.raises(ValueError, match="exact frozen six-object"):
        materialize.build_result(reduced_plan, reduced_summary, _stage_report())


def test_optimizer_updates_are_efficiency_not_parity() -> None:
    summary = _summary()
    summary["update_parity"]["total_scheduled_optimizer_updates_by_arm"]["joint-modular"] = 21
    _rehash(summary)

    result = materialize.build_result(_plan(), summary, _stage_report())

    efficiency = result["update_parity"]["optimizer_update_efficiency"]
    assert efficiency["joint_total"] == 21
    assert efficiency["ablation_ensemble_total"] == 30
    assert result["u2_checks"]["update_parity"] is True


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda s: s.update(duration_sec=0), "run duration"),
        (
            lambda s: s["arms"]["joint-modular"]["resources"]["training_log"].update(records=0),
            "training-log evidence",
        ),
        (
            lambda s: s["arms"]["joint-modular"]["checkpoints"].update(
                total_initialized_tensor_elements=0
            ),
            "checkpoint tensor and adapter evidence",
        ),
        (
            lambda s: s["arms"]["joint-modular"]["checkpoints"].update(
                total_adapter_tensor_elements=0
            ),
            "checkpoint tensor and adapter evidence",
        ),
    ],
)
def test_materializer_requires_positive_resource_evidence(mutation, match: str) -> None:
    summary = _summary()
    mutation(summary)
    _rehash(summary)

    with pytest.raises(ValueError, match=match):
        materialize.build_result(_plan(), summary, _stage_report())


def test_initialized_tensor_consolidation_is_a_gate() -> None:
    summary = _summary()
    summary["arms"]["joint-modular"]["checkpoints"]["total_initialized_tensor_elements"] = 1_000
    _rehash(summary)

    result = materialize.build_result(_plan(), summary, _stage_report())

    assert result["u1_checks"]["initialized_tensor_consolidation"] is False
    assert result["u1_passed"] is False


def test_materializer_rejects_incomplete_adapter_inventory_and_heldout() -> None:
    summary = _summary()
    summary["arms"]["ablation-advection1d"]["adapter_inventory"] = ["advection1d"]
    _rehash(summary)
    with pytest.raises(ValueError, match="full three-task adapter inventory"):
        materialize.build_result(_plan(), summary, _stage_report())

    summary = _summary()
    summary["heldout_reads"] = 1
    _rehash(summary)
    with pytest.raises(PermissionError, match="validation-only"):
        materialize.build_result(_plan(), summary, _stage_report())


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda s: s.update(training_lock_sha256="wrong"), "training lock"),
        (lambda s: s.update(config_sha256="wrong"), "config differs"),
        (
            lambda s: s["heldout_evidence"].update(evaluation_splits=["test"]),
            "held-out evidence",
        ),
    ],
)
def test_materializer_rechecks_boundaries_independently(mutation, match: str) -> None:
    summary = _summary()
    mutation(summary)
    _rehash(summary)

    with pytest.raises((ValueError, PermissionError), match=match):
        materialize.build_result(_plan(), summary, _stage_report())
