from __future__ import annotations

from argparse import Namespace

import yaml

from scripts.plan_transport_official_hydration import create_plan


def _write_manifest(path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "files": [
                    {
                        "path": "1D/Advection/Train/1D_Advection_Sols_beta0.1.hdf5",
                        "file_id": 1,
                        "size_bytes": 100,
                        "checksum": "a",
                        "checksum_type": "MD5",
                    },
                    {
                        "path": "1D/Burgers/Train/ignored.hdf5",
                        "file_id": 2,
                        "size_bytes": 999,
                    },
                    {
                        "path": "1D/Advection/Train/1D_Advection_Sols_beta0.2.hdf5",
                        "file_id": 3,
                        "size_bytes": 200,
                        "checksum": "b",
                        "checksum_type": "MD5",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )


def _args(tmp_path):
    manifest = tmp_path / "manifest.yaml"
    _write_manifest(manifest)
    return Namespace(
        manifest=str(manifest),
        raw_out="data/pdebench/raw",
        hydrated_source_root="data/hydrated_source",
        hydrated_light_root="data/hydrated_light",
        output_root="reports/research/sota_loop",
        train_count=10,
        val_count=4,
        reserved_test_count=4,
        rollout_steps=3,
        shift=[0, 1],
        max_files=None,
        reference_metric_value=0.5,
        val_min_relative_improvement=0.0,
        output_json=str(tmp_path / "plan.json"),
    )


def test_hydration_plan_uses_only_official_advection_train_entries(tmp_path):
    plan = create_plan(_args(tmp_path))

    assert plan["status"] == "ready_for_explicit_hydration"
    assert plan["selected_file_count"] == 2
    assert plan["estimated_download_bytes"] == 300
    assert len(plan["commands"]["download_official_train_files"]) == 2
    assert "1D/Advection/Train/1D_Advection_Sols_beta0.1.hdf5" in plan["commands"]["download_official_train_files"][0]
    assert "test-count 0" in plan["commands"]["build_light_train_val_shards"]
    assert "--samples 9" in plan["commands"]["build_train_val_source"]
    assert "--split-block-size 9" in plan["commands"]["build_light_train_val_shards"]
    assert "--split-block-offset train=0" in plan["commands"]["build_light_train_val_shards"]
    assert "--split-block-offset val=5" in plan["commands"]["build_light_train_val_shards"]
    assert plan["stratified_split_policy"]["train_per_file"] == 5
    assert plan["stratified_split_policy"]["val_per_file"] == 2
    assert plan["stratified_split_policy"]["reserved_test_per_file"] == 2
    assert "REQUIRE_STATUS=literal-test-ready" in plan["commands"]["objective_audit_after_validation"]
    assert plan["held_out_test_policy"]["test_split_downloaded"] is False


def test_hydration_plan_can_limit_files_for_dry_run(tmp_path):
    args = _args(tmp_path)
    args.max_files = 1

    plan = create_plan(args)

    assert plan["selected_file_count"] == 1
    assert plan["estimated_download_bytes"] == 100
