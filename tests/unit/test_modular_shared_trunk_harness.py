from __future__ import annotations

import copy
import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from scripts import run_strat_v1_modular_shared_trunk as runner
from ups.data.manifests import canonical_sha256

TASKS = ("advection1d", "burgers1d", "darcy2d")


def _self_hash(payload: dict, key: str) -> dict:
    result = copy.deepcopy(payload)
    result[key] = canonical_sha256(result)
    return result


def _plan() -> dict:
    command = ["python", "scripts/run_strat_v1_modular_shared_trunk.py"]
    return _self_hash(
        {
            "schema_version": 1,
            "plan_id": runner.EXPECTED_PLAN_ID,
            "mode": "validation_only",
            "heldout_access": "forbidden",
            "measurement_lock_access": "forbidden",
            "design": {
                "seed": 17,
                "architecture": "modular_shared_trunk_task_adapters",
                "arms": list(runner.ARM_NAMES),
            },
            "command": command,
            "command_sha256": canonical_sha256(command),
        },
        "plan_sha256",
    )


def _stage_report(lock_sha256: str, objects: dict[str, str]) -> dict:
    return _self_hash(
        {
            "schema_version": 1,
            "status": "complete",
            "lock_sha256": lock_sha256,
            "object_count": len(objects),
            "objects": [
                {
                    "id": object_id,
                    "role": "valid" if object_id.endswith("-valid") else "train",
                    "checksum": {"algorithm": "sha256", "value": digest},
                }
                for object_id, digest in objects.items()
            ],
        },
        "artifact_sha256",
    )


def _metrics(tasks: tuple[str, ...], primary: float = 0.1) -> dict[str, float]:
    result = {"macro_primary_nrmse": primary}
    for task in tasks:
        suffix = "decoded_solution_nrmse" if task == "darcy2d" else "decoded_rollout_nrmse"
        result[f"task_{task}_{suffix}"] = primary
        result[f"task_{task}_maximum_corrected_regime_spread_ratio"] = 1.0
    return result


def _base_config() -> dict:
    return {
        "data": {
            "conditioning_schema": {
                "task_vocab": list(TASKS),
                "param_vocab": ["beta", "nu"],
            }
        },
        "operator": {
            "conditioning": {"sources": None},
            "task_adapters": {"enabled": True, "bottleneck_dim": 16},
        },
        "modular_shared_trunk": {
            "adapter_type": "residual_bottleneck",
            "adapter_bottleneck_dim": 16,
            "adapter_inventory": list(TASKS),
            "arms": {
                arm: {
                    "tasks": list(runner.ARM_TASKS[arm]),
                    "adapter_inventory": list(TASKS),
                }
                for arm in runner.ARM_NAMES
            },
        },
        "training": {
            "batch_size": 16,
            "sample_balanced_operator_loss": True,
            "canonical_steady_operator_mapping": True,
            "lambda_semigroup": 0.0,
            "patience": None,
            "fail_on_oom": True,
        },
        "stages": {
            "operator": {"epochs": 1},
            "decoder": {"epochs": 1, "batch_size": 2},
            "operator_decoded": {"epochs": 1, "batch_size": 2},
            "joint_codec_operator": {"epochs": 1, "batch_size": 2, "rollout_steps": 2},
        },
    }


def test_plan_checker_requires_exact_four_arm_validation_contract(tmp_path: Path) -> None:
    plan = _plan()
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    assert runner._checked_plan(path)["plan_sha256"] == plan["plan_sha256"]

    plan["design"]["arms"] = ["joint-modular"]
    plan["plan_sha256"] = runner._unsigned_hash(plan, "plan_sha256")
    path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ValueError, match="arm mismatch"):
        runner._checked_plan(path)

    plan = _plan()
    plan["heldout_access"] = "allowed"
    plan["plan_sha256"] = runner._unsigned_hash(plan, "plan_sha256")
    path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(PermissionError, match="held-out"):
        runner._checked_plan(path)

    plan = _plan()
    plan["measurement_lock_access"] = "allowed"
    plan["plan_sha256"] = runner._unsigned_hash(plan, "plan_sha256")
    path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(PermissionError, match="measurement-lock"):
        runner._checked_plan(path)

    plan = _plan()
    plan["plan_id"] = "strat-v1-modular-shared-trunk-d6"
    plan["plan_sha256"] = runner._unsigned_hash(plan, "plan_sha256")
    path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(PermissionError, match="superseded"):
        runner._checked_plan(path)


def test_stage_report_rejects_duplicate_ids_and_wrong_roles(tmp_path: Path) -> None:
    objects = {"advection1d-train": "1" * 64, "advection1d-valid": "2" * 64}
    plan = {"bindings": {"training_lock": {"objects": objects}}}
    runtime = SimpleNamespace(lock=SimpleNamespace(lock_sha256="lock"))
    path = tmp_path / "stage.json"

    duplicate = _stage_report("lock", objects)
    duplicate["objects"].append(copy.deepcopy(duplicate["objects"][0]))
    duplicate["object_count"] = len(objects)
    duplicate["artifact_sha256"] = runner._unsigned_hash(duplicate, "artifact_sha256")
    path.write_text(json.dumps(duplicate), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly"):
        runner._checked_stage_report(path, plan, runtime)

    wrong_role = _stage_report("lock", objects)
    wrong_role["objects"][1]["role"] = "train"
    wrong_role["artifact_sha256"] = runner._unsigned_hash(wrong_role, "artifact_sha256")
    path.write_text(json.dumps(wrong_role), encoding="utf-8")
    with pytest.raises(ValueError, match="role differs"):
        runner._checked_stage_report(path, plan, runtime)


def test_config_and_arm_builder_preserve_identical_modular_schema() -> None:
    base = _base_config()
    runner._validate_base_config(base)

    class Runtime:
        def apply_to_runner_config(self, cfg, *, condition_on_regime):
            assert condition_on_regime is True
            cfg.setdefault("data", {})["root"] = "/sealed"
            return cfg

    shared = runner._arm_config(base, Runtime(), TASKS, "train")
    specialist = runner._arm_config(base, Runtime(), ("darcy2d",), "val")
    assert shared["operator"] == specialist["operator"] == base["operator"]
    assert shared["data"]["conditioning_schema"] == specialist["data"]["conditioning_schema"]
    assert shared["seed"] == shared["training"]["seed"] == 17
    assert specialist["seed"] == specialist["training"]["seed"] == 17
    assert shared["data"]["task"] == list(TASKS)
    assert specialist["data"]["task"] == "darcy2d"

    with pytest.raises(PermissionError, match="split"):
        runner._arm_config(base, Runtime(), TASKS, "test")

    no_adapter = _base_config()
    del no_adapter["operator"]["task_adapters"]
    with pytest.raises(ValueError, match="adapter"):
        runner._validate_base_config(no_adapter)


def test_checkpoint_evidence_records_adapter_ownership_and_fails_closed(tmp_path: Path) -> None:
    torch.save(
        {
            "core.weight": torch.ones(3, 3),
            "task_adapters.advection1d.input.weight": torch.ones(2, 3),
        },
        tmp_path / "operator.pt",
    )
    evidence = runner._checkpoint_evidence(tmp_path)
    assert evidence["total_initialized_tensor_elements"] == 15
    assert evidence["total_adapter_tensor_elements"] == 6
    assert evidence["total_shared_tensor_elements"] == 9
    assert evidence["files"]["operator.pt"]["sha256"] == runner._sha256(tmp_path / "operator.pt")

    plain = tmp_path / "plain"
    plain.mkdir()
    torch.save({"core.weight": torch.ones(1)}, plain / "operator.pt")
    with pytest.raises(ValueError, match="no modular adapter"):
        runner._checkpoint_evidence(plain)


def test_resource_evidence_requires_positive_duration_and_labels_cumulative_rss(
    tmp_path: Path,
) -> None:
    evidence = runner._resource_evidence(
        run_dir=tmp_path,
        summary={"duration_sec": 1.25},
        wall_time_sec=1.5,
        child_max_rss_kib=2048,
    )
    assert evidence["duration_sec_reported_by_runner"] == 1.25
    assert evidence["child_process_family_max_rss_kib_high_watermark"] == 2048
    assert "not attributable to this arm alone" in evidence["rss_scope"]

    with pytest.raises(ValueError, match="duration must be finite and positive"):
        runner._resource_evidence(
            run_dir=tmp_path,
            summary={"duration_sec": 0.0},
            wall_time_sec=0.0,
            child_max_rss_kib=0,
        )


def test_summary_checker_rejects_test_split_task_drift_and_missing_metrics(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    payload = {
        "extra": {
            "split": "val",
            "strict_stratified_metrics": True,
            "task": list(TASKS),
        },
        "metrics": _metrics(TASKS),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert runner._validate_arm_summary(path, TASKS)["metrics"]["macro_primary_nrmse"] == 0.1

    payload["extra"]["split"] = "test"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PermissionError, match="validation-only"):
        runner._validate_arm_summary(path, TASKS)

    payload["extra"]["split"] = "val"
    payload["extra"]["task"] = ["darcy2d"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="task mismatch"):
        runner._validate_arm_summary(path, TASKS)

    payload["extra"]["task"] = list(TASKS)
    del payload["metrics"]["task_darcy2d_decoded_solution_nrmse"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="lacks strict metrics"):
        runner._validate_arm_summary(path, TASKS)


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
    args = Namespace(config=str(config), training_lock=str(lock))
    runtime = SimpleNamespace(lock=SimpleNamespace(lock_sha256="lock-contract"))
    plan = {
        "bindings": {
            "training_lock": {
                "lock_sha256": "lock-contract",
                "file_sha256": runner._sha256(lock),
            },
            "config": {"file_sha256": runner._sha256(config)},
            "source": {
                "implementation_commit": "a" * 40,
                "files": {source.name: runner._sha256(source)},
            },
        }
    }
    runner._verify_plan_bindings(plan, args, runtime)
    source.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="live source"):
        runner._verify_plan_bindings(plan, args, runtime)


def test_full_local_harness_runs_four_arms_and_joint_parameter_shuffle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    config = repo / "d6.yaml"
    lock = repo / "training.lock.json"
    source = repo / "source.py"
    config.write_text(yaml.safe_dump(_base_config()), encoding="utf-8")
    lock.write_text("{}\n", encoding="utf-8")
    source.write_text("VALUE = 1\n", encoding="utf-8")

    plan = _plan()
    objects = {f"{task}-train": f"{index:064x}" for index, task in enumerate(TASKS, 1)}
    objects.update({f"{task}-valid": f"{index:064x}" for index, task in enumerate(TASKS, 4)})
    plan["bindings"] = {
        "training_lock": {
            "lock_sha256": "frozen-lock",
            "file_sha256": runner._sha256(lock),
            "objects": objects,
        },
        "config": {"file_sha256": runner._sha256(config)},
        "source": {
            "implementation_commit": "a" * 40,
            "files": {source.name: runner._sha256(source)},
        },
    }
    plan["plan_sha256"] = runner._unsigned_hash(plan, "plan_sha256")
    plan_path = repo / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    stage_path = repo / "stage.json"
    stage_path.write_text(json.dumps(_stage_report("frozen-lock", objects)), encoding="utf-8")

    class Runtime:
        lock = SimpleNamespace(
            lock_sha256="frozen-lock",
            requested_roles=("train", "valid"),
            objects=(SimpleNamespace(role="train"), SimpleNamespace(role="valid")),
        )

        def apply_to_runner_config(self, cfg, *, condition_on_regime):
            cfg.setdefault("data", {})["root"] = str(repo / "data")
            return cfg

        tasks = {task: SimpleNamespace(train=SimpleNamespace(sample_count=2)) for task in TASKS}

    monkeypatch.setattr(runner, "REPO_ROOT", repo)
    monkeypatch.setattr(runner, "load_strat_v1_baseline_runtime", lambda *args, **kwargs: Runtime())
    monkeypatch.setattr(runner, "load_config_with_includes", lambda _path: _base_config())

    seen_names: list[str] = []
    exposure = {
        task: {
            "source_examples": 8,
            "scheduled_compute_units": 4,
        }
        for task in TASKS
    }

    def fake_run(command: list[str]) -> tuple[float, int]:
        name = command[command.index("--name") + 1]
        output_root = Path(command[command.index("--output-root") + 1])
        seen_names.append(name)
        run_dir = output_root / name
        run_dir.mkdir(parents=True, exist_ok=True)
        if name == "joint-modular-parameter-shuffled":
            tasks = TASKS
            primary = 0.11
        else:
            tasks = runner.ARM_TASKS[name]
            primary = 0.10
            checkpoints = run_dir / "checkpoints"
            checkpoints.mkdir()
            torch.save(
                {
                    "core.weight": torch.ones(2, 2),
                    "task_adapters.all.weight": torch.ones(2, 2),
                },
                checkpoints / "operator.pt",
            )
            logs = run_dir / "logs"
            logs.mkdir()
            (logs / "training.jsonl").write_text(
                "\n".join(
                    json.dumps({f"{stage}/epoch": 0, f"{stage}/epoch_time_sec": 0.2})
                    for stage in runner.STAGES
                )
                + "\n",
                encoding="utf-8",
            )
        summary = {
            "extra": {
                "split": "val",
                "strict_stratified_metrics": True,
                "task": list(tasks) if len(tasks) > 1 else tasks[0],
            },
            "metrics": _metrics(tasks, primary),
            "duration_sec": 1.25,
            "resource_accounting": {
                "update_parity_by_task": {task: exposure[task] for task in tasks}
            },
        }
        (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        return 1.5, 2048

    monkeypatch.setattr(runner, "_run_arm_command", fake_run)
    output = repo / "output"
    args = Namespace(
        training_lock=str(lock),
        data_root=str(repo / "data"),
        config=str(config),
        output_dir=str(output),
        plan_path=str(plan_path),
        plan_sha256=plan["plan_sha256"],
        stage_report=str(stage_path),
        device="cpu",
        resume=False,
    )
    summary_path = runner.run(args)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert seen_names == [*runner.ARM_NAMES, "joint-modular-parameter-shuffled"]
    assert tuple(summary["arms"]) == runner.ARM_NAMES
    assert summary["heldout_reads"] == 0
    assert summary["heldout_evidence"] == {
        "requested_roles": ["train", "valid"],
        "contains_test_object": False,
        "evaluation_splits": ["val"],
    }
    assert summary["conditioning_diagnostics"]["relative_nrmse_degradation"] == pytest.approx(0.1)
    assert summary["arms"]["joint-modular"]["resources"]["runner_reported"] == {
        "update_parity_by_task": exposure,
    }
    assert summary["update_parity"]["joint_by_task"] == exposure
    assert summary["arms"]["ablation-darcy2d"]["adapter_inventory"] == list(TASKS)
    assert summary["arms"]["ablation-darcy2d"]["adapter_bottleneck_dim"] == 16
    joint_schedule = summary["arms"]["joint-modular"]["resources"]["optimizer_update_schedule"]
    assert joint_schedule["scope"] == "whole_arm_combined_task_loader"
    assert joint_schedule["combined_source_samples"] == 6
    assert summary["update_parity"]["total_scheduled_optimizer_updates_by_arm"] == {
        arm: summary["arms"][arm]["resources"]["optimizer_update_schedule"][
            "total_scheduled_optimizer_updates"
        ]
        for arm in runner.ARM_NAMES
    }
    assert "optimizer_updates" not in summary["update_parity"]["joint_by_task"]["advection1d"]
    assert summary["artifact_sha256"] == canonical_sha256(
        {key: value for key, value in summary.items() if key != "artifact_sha256"}
    )


def test_run_rejects_training_lock_with_test_object(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan()
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    runtime = SimpleNamespace(
        lock=SimpleNamespace(
            lock_sha256="x",
            requested_roles=("train", "valid", "test"),
            objects=(SimpleNamespace(role="test"),),
        )
    )
    monkeypatch.setattr(runner, "load_strat_v1_baseline_runtime", lambda *a, **k: runtime)
    args = Namespace(
        training_lock=str(tmp_path / "lock"),
        data_root=str(tmp_path),
        config=str(tmp_path / "config"),
        output_dir=str(tmp_path / "output"),
        plan_path=str(plan_path),
        plan_sha256=plan["plan_sha256"],
        device="cpu",
        resume=False,
    )
    with pytest.raises(PermissionError, match="held-out"):
        runner.run(args)


def test_update_parity_fails_closed_on_missing_or_mismatched_counts() -> None:
    def record(task_values):
        return {
            "tasks": list(task_values),
            "resources": {
                "optimizer_update_schedule": {"total_scheduled_optimizer_updates": 5},
                "runner_reported": {
                    "update_parity_by_task": {
                        task: {
                            "source_examples": 10,
                            "scheduled_compute_units": 2,
                        }
                        for task in task_values
                    }
                },
            },
        }

    arms = {"joint-modular": record(TASKS)}
    arms.update({f"ablation-{task}": record((task,)) for task in TASKS})
    assert runner._update_parity_evidence(arms)["comparison"] == ("joint_task_to_matching_ablation")

    del arms["ablation-burgers1d"]["resources"]["runner_reported"]["update_parity_by_task"]
    with pytest.raises(ValueError, match="lacks exact"):
        runner._update_parity_evidence(arms)

    arms["ablation-burgers1d"] = record(("burgers1d",))
    arms["ablation-burgers1d"]["resources"]["runner_reported"]["update_parity_by_task"][
        "burgers1d"
    ]["scheduled_compute_units"] = 1
    with pytest.raises(ValueError, match="update parity mismatch"):
        runner._update_parity_evidence(arms)


def test_runner_derives_exposure_from_sealed_counts_and_observed_epochs() -> None:
    base = _base_config()
    base["training"]["batch_size"] = 4
    base["training"]["accum_steps"] = 2
    base["stages"] = {
        "operator": {"epochs": 3},
        "decoder": {"epochs": 2, "batch_size": 2},
        "operator_decoded": {"epochs": 2, "batch_size": 2, "rollout_steps": 2},
        "joint_codec_operator": {"epochs": 1, "batch_size": 2, "rollout_steps": 3},
    }
    runtime = SimpleNamespace(
        tasks={
            "burgers1d": SimpleNamespace(train=SimpleNamespace(sample_count=10)),
        }
    )
    record = {
        "resources": {
            "runner_reported": {},
            "training_log": {
                "stage_epochs": {
                    "operator": 3,
                    "decoder": 2,
                    "operator_decoded": 2,
                    "joint_codec_operator": 1,
                }
            },
        }
    }
    runner._ensure_update_exposure(record=record, base=base, runtime=runtime, tasks=("burgers1d",))
    # Per-task exposure excludes optimizer steps because joint batches can mix tasks.
    assert record["resources"]["runner_reported"]["update_parity_by_task"]["burgers1d"] == {
        "source_examples": 80,
        "scheduled_compute_units": 120,
    }
    assert (
        record["resources"]["optimizer_update_schedule"]["total_scheduled_optimizer_updates"] == 31
    )
    assert record["resources"]["update_parity_derivation"]["epoch_count_source"] == (
        "arm_training_log"
    )

    truncated = copy.deepcopy(record)
    truncated["resources"]["training_log"]["stage_epochs"]["operator"] = 2
    with pytest.raises(ValueError, match="complete all 3 scheduled operator epochs"):
        runner._ensure_update_exposure(
            record=truncated,
            base=base,
            runtime=runtime,
            tasks=("burgers1d",),
        )

    record["resources"]["runner_reported"] = {}
    record["resources"]["training_log"]["stage_epochs"].pop("decoder")
    with pytest.raises(ValueError, match="completed decoder epochs"):
        runner._ensure_update_exposure(
            record=record, base=base, runtime=runtime, tasks=("burgers1d",)
        )
