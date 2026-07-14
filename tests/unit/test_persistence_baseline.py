from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import h5py
import pytest
import torch
import yaml

from scripts import run_persistence_baseline as baseline_script
from ups.data.manifests import ProtocolManifest, SourceManifest, resolve_data_lock, write_data_lock
from ups.eval.persistence_baselines import evaluate_persistence_decoded


def test_evaluate_persistence_decoded_constant_sequence_is_zero(tmp_path):
    with h5py.File(tmp_path / "burgers1d_test.h5", "w") as handle:
        handle.create_dataset("data", data=torch.ones(2, 4, 8).numpy())

    cfg = {
        "data": {
            "task": "burgers1d",
            "split": "test",
            "root": str(tmp_path),
            "max_samples": 1,
        }
    }

    report = evaluate_persistence_decoded(cfg, rollout_steps=2)

    assert report.metrics["task_burgers1d_decoded_rollout_nrmse"] == 0.0
    assert report.metrics["task_burgers1d_decoded_h1_nrmse"] == 0.0
    assert report.extra["baseline"] == "persistence"
    assert report.extra["samples_by_task"] == {"burgers1d": 1}


def test_darcy_persistence_is_identity_operator_not_fake_time_rollout(tmp_path):
    with h5py.File(tmp_path / "darcy2d_test.h5", "w") as handle:
        handle.create_dataset("data", data=torch.zeros(1, 1, 4, 4, 1).numpy())
        handle.create_dataset("targets", data=torch.ones(1, 1, 4, 4, 1).numpy())
    cfg = {
        "data": {
            "task": "darcy2d",
            "split": "test",
            "root": str(tmp_path),
        }
    }

    report = evaluate_persistence_decoded(cfg, rollout_steps=16)

    assert report.metrics["task_darcy2d_decoded_solution_nrmse"] > 0.999
    assert not any("darcy2d_decoded_h" in key for key in report.metrics)
    assert report.extra["samples_by_task"] == {"darcy2d": 1}


def test_temporal_persistence_holds_initial_state_for_every_horizon(tmp_path):
    values = torch.arange(17, dtype=torch.float32).view(1, 17, 1, 1)
    with h5py.File(tmp_path / "burgers1d_val.h5", "w") as handle:
        handle.create_dataset("data", data=values.numpy())
        handle.create_dataset("nu", data=[0.1])
    cfg = {"data": {"task": "burgers1d", "split": "val", "root": str(tmp_path)}}

    report = evaluate_persistence_decoded(cfg, rollout_steps=16)

    assert report.metrics["task_burgers1d_decoded_h1_nrmse"] > 0.999
    assert report.metrics["task_burgers1d_decoded_h16_nrmse"] > 0.999
    assert report.extra["temporal_rollout_semantics"] == "initial_state_held_constant"


def _write_official_fixture(root: Path) -> Path:
    objects = []
    splits = {"valid": []}
    for task, regime_key, regime_values in (
        ("advection1d", "beta", (0.1, 0.2)),
        ("burgers1d", "nu", (0.01, 0.02)),
    ):
        path = root / f"{task}_val.h5"
        fields = torch.stack(
            [torch.arange(17, dtype=torch.float32).view(17, 1, 1) + offset for offset in (0, 1)]
        )
        with h5py.File(path, "w") as handle:
            handle.create_dataset("data", data=fields.numpy())
            handle.create_dataset(regime_key, data=regime_values)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        object_id = f"{task}-valid"
        objects.append(
            {
                "object_id": object_id,
                "path": path.name,
                "size_bytes": path.stat().st_size,
                "checksums": {"sha256": digest},
                "uris": [path.as_uri()],
                "declared_roles": ["valid"],
            }
        )
        splits["valid"].append(object_id)

    darcy = root / "darcy2d_val.h5"
    with h5py.File(darcy, "w") as handle:
        handle.create_dataset("data", data=torch.zeros(2, 1, 2, 2, 1).numpy())
        handle.create_dataset("targets", data=torch.ones(2, 1, 2, 2, 1).numpy())
        handle.create_dataset("beta", data=[1.0, 10.0])
    digest = hashlib.sha256(darcy.read_bytes()).hexdigest()
    objects.append(
        {
            "object_id": "darcy2d-valid",
            "path": darcy.name,
            "size_bytes": darcy.stat().st_size,
            "checksums": {"sha256": digest},
            "uris": [darcy.as_uri()],
            "declared_roles": ["valid"],
        }
    )
    splits["valid"].append("darcy2d-valid")
    source = SourceManifest.from_dict(
        {
            "schema_version": 1,
            "dataset_id": "pdebench-strat-v1-test",
            "provider": "fixture",
            "revision": "sha256:" + "1" * 64,
            "native_format": "HDF5",
            "license": "CC0-1.0",
            "citation": "fixture",
            "objects": objects,
        }
    )
    protocol = ProtocolManifest.from_dict(
        {
            "schema_version": 1,
            "protocol_id": "pdebench-strat-v1-test",
            "dataset_id": source.dataset_id,
            "source_revision": source.revision,
            "adapter": "pdebench_hdf5",
            "adapter_revision": "1.0.0",
            "split_authority": "fixture",
            "splits": splits,
            "identity_fields": ["task", "sample"],
            "selection": {"protocol": "strat-v1", "seed": 0},
            "normalization": {"fit_role": "train", "method": "none"},
            "test_access": "measurement_contract_required",
        }
    )
    lock = resolve_data_lock(source, protocol, requested_roles=("valid",))
    lock_path = root / "training.lock.json"
    write_data_lock(lock_path, lock)
    return lock_path


def test_run_persistence_baseline_writes_light_summary(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    output_root = tmp_path / "runs"
    data_root.mkdir()
    lock_path = _write_official_fixture(data_root)

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "task": ["advection1d", "burgers1d", "darcy2d"],
                    "split": "val",
                    "root": str(data_root),
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_persistence_baseline",
            "--config",
            str(cfg_path),
            "--name",
            "persistence_burgers",
            "--output-root",
            str(output_root),
            "--rollout-steps",
            "16",
            "--data-lock",
            str(lock_path),
        ],
    )

    baseline_script.main()

    summary_path = output_root / "persistence_burgers" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["run_name"] == "persistence_burgers"
    assert summary["stages"] == ["persistence"]
    assert "macro_primary_nrmse" in summary["metrics"]
    assert summary["split"] == "val"
    assert summary["details"]["protocol"]["data_lock_sha256"]
    assert summary["details"]["protocol"]["code_files_sha256"]
    assert summary["details"]["protocol"]["normalization"] == "none_physical_space"
    assert summary["details"]["tasks"]["darcy2d"]["per_horizon_nrmse"] == {}
    assert summary["details"]["tasks"]["darcy2d"]["per_regime_global_scale_nrmse"]
    assert summary["details"]["tasks"]["darcy2d"]["per_regime_element_count"]
    assert abs(summary["details"]["tasks"]["darcy2d"]["global_scale_reconstruction_delta"]) < 1e-6
    assert (output_root / "results.tsv").exists()


def test_strict_contract_reports_all_horizons_regimes_and_macro(tmp_path):
    lock_path = _write_official_fixture(tmp_path)
    cfg = {
        "data": {
            "task": list(baseline_script.STRAT_V1_TASKS),
            "split": "val",
            "root": str(tmp_path),
            "data_lock_path": str(lock_path),
        }
    }

    report = evaluate_persistence_decoded(cfg, rollout_steps=16, strict_contract=True)

    task_values = [
        report.metrics["task_advection1d_decoded_rollout_nrmse"],
        report.metrics["task_burgers1d_decoded_rollout_nrmse"],
        report.metrics["task_darcy2d_decoded_solution_nrmse"],
    ]
    assert report.metrics["macro_primary_nrmse"] == pytest.approx(sum(task_values) / 3)
    for task in ("advection1d", "burgers1d"):
        assert all(
            f"task_{task}_decoded_h{horizon}_nrmse" in report.metrics for horizon in range(1, 17)
        )
    assert any(key.startswith("task_advection1d_regime_") for key in report.metrics)
    assert any(key.startswith("task_burgers1d_regime_") for key in report.metrics)
    assert any(key.startswith("task_darcy2d_regime_") for key in report.metrics)


def test_strict_contract_rejects_short_temporal_samples(tmp_path):
    for task, regime_key in (("advection1d", "beta"), ("burgers1d", "nu")):
        with h5py.File(tmp_path / f"{task}_val.h5", "w") as handle:
            handle.create_dataset("data", data=torch.ones(1, 16, 1, 1).numpy())
            handle.create_dataset(regime_key, data=[0.1])
    with h5py.File(tmp_path / "darcy2d_val.h5", "w") as handle:
        handle.create_dataset("data", data=torch.zeros(1, 1, 2, 2, 1).numpy())
        handle.create_dataset("targets", data=torch.ones(1, 1, 2, 2, 1).numpy())
        handle.create_dataset("beta", data=[1.0])
    cfg = {
        "data": {
            "task": list(baseline_script.STRAT_V1_TASKS),
            "split": "val",
            "root": str(tmp_path),
        }
    }

    with pytest.raises(ValueError, match="16 required"):
        evaluate_persistence_decoded(cfg, rollout_steps=16, strict_contract=True)


def test_official_preflight_rejects_test_bytes_in_run_view(tmp_path):
    lock_path = _write_official_fixture(tmp_path)
    with h5py.File(tmp_path / "burgers1d_test.h5", "w") as handle:
        handle.create_dataset("data", data=torch.ones(1, 17, 1, 1).numpy())
    data_cfg = {
        "task": list(baseline_script.STRAT_V1_TASKS),
        "split": "val",
        "root": str(tmp_path),
    }

    with pytest.raises(PermissionError, match="contains test"):
        baseline_script._validate_official_inputs(
            data_cfg=data_cfg,
            data_lock_path=lock_path,
            expected_lock_sha256=None,
            rollout_steps=16,
        )


def test_append_results_row_preserves_existing_tracking_columns(tmp_path):
    results_path = tmp_path / "runs" / "results.tsv"
    results_path.parent.mkdir()
    results_path.write_text(
        "\t".join(
            [
                "run_name",
                "timestamp",
                "stages",
                "decoded",
                "train_split",
                "eval_split",
                "transfer_tasks",
                "promotion_passed",
                "main_metric_name",
                "main_metric_value",
                "summary_json",
                "wandb_run_ids",
                "wandb_urls",
            ]
        )
        + "\n"
        + "\t".join(
            [
                "candidate",
                "123",
                "train,eval",
                "True",
                "train",
                "test",
                "",
                "True",
                "decoded_rollout_nrmse",
                "0.3",
                "candidate/summary.json",
                "run-1",
                "https://wandb.invalid/run-1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    baseline_script._append_results_row(
        results_path,
        {
            "run_name": "persistence",
            "timestamp": 456,
            "stages": "persistence",
            "decoded": True,
            "train_split": "",
            "eval_split": "test",
            "transfer_tasks": "",
            "promotion_passed": "",
            "main_metric_name": "decoded_rollout_nrmse",
            "main_metric_value": 0.6,
            "summary_json": "persistence/summary.json",
        },
    )

    lines = results_path.read_text(encoding="utf-8").strip().splitlines()
    header = lines[0].split("\t")
    assert "wandb_run_ids" in header
    assert "wandb_urls" in header
    assert len(lines) == 3

    candidate = dict(zip(header, lines[1].split("\t", maxsplit=len(header) - 1)))
    persistence = dict(zip(header, lines[2].split("\t", maxsplit=len(header) - 1)))
    assert candidate["wandb_run_ids"] == "run-1"
    assert candidate["wandb_urls"] == "https://wandb.invalid/run-1"
    assert persistence["run_name"] == "persistence"
