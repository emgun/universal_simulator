from __future__ import annotations

import json

import h5py

from scripts.check_demo_readiness import readiness_payload


def _write_summary(path, *, run_name: str, decoded_rollout_nrmse: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "run_name": run_name,
                "metrics": {"decoded_rollout_nrmse": decoded_rollout_nrmse},
                "extra": {},
            }
        ),
        encoding="utf-8",
    )


def _write_h5(path, *, sample_count: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "data",
            data=[[float(row), float(row + 1)] for row in range(sample_count)],
        )


def test_readiness_reports_missing_candidate_and_baseline(tmp_path):
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        """
version: light-v1
remote_prefix: light-v1
tasks: [burgers1d]
splits:
  train: {samples: 8}
  test: {samples: 4}
records: []
""",
        encoding="utf-8",
    )

    payload = readiness_payload(
        manifest=manifest,
        summary_patterns=[str(tmp_path / "missing" / "*" / "summary.json")],
        baseline_run="persistence_light_v1_test",
        candidate_run="ups_light_v1_task_signature_only",
        check_b2=False,
        env_file=tmp_path / ".env",
    )

    assert payload["ready"] is False
    assert payload["manifest"]["expected_key_count"] == 2
    assert "Missing baseline summary: persistence_light_v1_test" in payload["blockers"]
    assert "Missing candidate summary: ups_light_v1_task_signature_only" in payload["blockers"]


def test_readiness_is_ready_with_manifest_and_required_summaries(tmp_path):
    manifest = tmp_path / "manifest.yaml"
    summaries = tmp_path / "runs"
    manifest.write_text(
        """
version: light-v1
remote_prefix: light-v1
tasks: [burgers1d]
splits:
  train: {samples: 8}
records: []
""",
        encoding="utf-8",
    )
    _write_summary(
        summaries / "persistence_light_v1_test" / "summary.json",
        run_name="persistence_light_v1_test",
        decoded_rollout_nrmse=1.0,
    )
    _write_summary(
        summaries / "ups_light_v1_task_signature_only" / "summary.json",
        run_name="ups_light_v1_task_signature_only",
        decoded_rollout_nrmse=0.75,
    )

    payload = readiness_payload(
        manifest=manifest,
        summary_patterns=[str(summaries / "*" / "summary.json")],
        baseline_run="persistence_light_v1_test",
        candidate_run="ups_light_v1_task_signature_only",
        check_b2=False,
        env_file=tmp_path / ".env",
    )

    assert payload["ready"] is True
    assert payload["summaries"]["has_baseline"] is True
    assert payload["summaries"]["has_candidate"] is True
    assert (
        "Build reports/demo/light_latest with scripts/build_demo_report.py."
        in payload["next_steps"]
    )


def test_readiness_reports_missing_local_source_data(tmp_path):
    manifest = tmp_path / "manifest.yaml"
    data_root = tmp_path / "data"
    summaries = tmp_path / "runs"
    manifest.write_text(
        f"""
version: light-v1
remote_prefix: light-v1
source_root: {data_root}
tasks: [burgers1d, darcy2d]
splits:
  train: {{samples: 3, preferred_source_split: train}}
records: []
""",
        encoding="utf-8",
    )
    _write_h5(data_root / "burgers1d_train.h5", sample_count=3)
    _write_summary(
        summaries / "ups_light_v1_task_signature_only" / "summary.json",
        run_name="ups_light_v1_task_signature_only",
        decoded_rollout_nrmse=0.75,
    )

    payload = readiness_payload(
        manifest=manifest,
        summary_patterns=[str(summaries / "*" / "summary.json")],
        baseline_run="",
        candidate_run="ups_light_v1_task_signature_only",
        check_b2=False,
        env_file=tmp_path / ".env",
        check_local_data=True,
        data_root=data_root,
    )

    assert payload["ready"] is False
    assert payload["local_data"]["ok"] is False
    assert payload["local_data"]["missing_count"] == 1
    assert payload["local_data"]["missing"][0]["task"] == "darcy2d"
    assert payload["local_data"]["missing"][0]["split"] == "train"
    assert payload["local_data"]["missing"][0]["candidate_paths"] == [
        str(data_root / "darcy2d_train.h5")
    ]
    assert "Local data missing 1 expected demo source shard(s)." in payload["blockers"]


def test_readiness_is_ready_with_local_source_data(tmp_path):
    manifest = tmp_path / "manifest.yaml"
    data_root = tmp_path / "data"
    summaries = tmp_path / "runs"
    manifest.write_text(
        f"""
version: light-v1
remote_prefix: light-v1
source_root: {data_root}
tasks: [burgers1d, darcy2d]
splits:
  train: {{samples: 3, preferred_source_split: train}}
  val: {{samples: 2, preferred_source_split: val, fallback_source_split: train}}
records: []
""",
        encoding="utf-8",
    )
    for task in ("burgers1d", "darcy2d"):
        _write_h5(data_root / f"{task}_train.h5", sample_count=3)
        _write_h5(data_root / f"{task}_val.h5", sample_count=2)
    _write_summary(
        summaries / "persistence_light_v1_test" / "summary.json",
        run_name="persistence_light_v1_test",
        decoded_rollout_nrmse=1.0,
    )
    _write_summary(
        summaries / "ups_light_v1_task_signature_only" / "summary.json",
        run_name="ups_light_v1_task_signature_only",
        decoded_rollout_nrmse=0.75,
    )

    payload = readiness_payload(
        manifest=manifest,
        summary_patterns=[str(summaries / "*" / "summary.json")],
        baseline_run="persistence_light_v1_test",
        candidate_run="ups_light_v1_task_signature_only",
        check_b2=False,
        env_file=tmp_path / ".env",
        check_local_data=True,
        data_root=data_root,
    )

    assert payload["ready"] is True
    assert payload["local_data"]["ok"] is True
    assert payload["local_data"]["present_count"] == 4
    assert payload["local_data"]["short_count"] == 0
