from __future__ import annotations

from argparse import Namespace
import json

from scripts.run_official_hydrated_post_validation_test import run_post_validation_test


def _write_status(tmp_path, status: str):
    path = tmp_path / "objective.json"
    path.write_text(json.dumps({"status": status}), encoding="utf-8")
    return path


def _args(tmp_path, *, status: str, execute: bool = False, execute_test: bool = False):
    return Namespace(
        objective_status_json=str(_write_status(tmp_path, status)),
        hydrated_source_root="data/source",
        hydrated_light_root="data/light",
        output_root="reports/research/sota_loop",
        official_hydrated_gate_json="reports/research/sota_loop/official_hydrated_transport_shift_gate.json",
        train_count=10,
        val_count=4,
        test_count=3,
        split_block_size=17,
        test_block_offset=14,
        rollout_steps=2,
        shift=[0, 1],
        reference_metric_value=0.5,
        val_min_relative_improvement=0.0,
        execute=execute,
        execute_test=execute_test,
        output_json=str(tmp_path / "run.json"),
    )


def test_post_validation_test_runner_blocks_before_literal_test_ready(tmp_path):
    record = run_post_validation_test(_args(tmp_path, status="literal_blocked", execute=True, execute_test=True))

    assert record["status"] == "blocked"
    assert any("expected literal_test_ready" in blocker for blocker in record["blockers"])
    assert all(row["executed"] is False for row in record["executed"])


def test_post_validation_test_runner_dry_run_requires_explicit_test_execution(tmp_path):
    record = run_post_validation_test(_args(tmp_path, status="literal_test_ready"))

    assert record["status"] == "dry_run"
    assert record["blockers"] == ["held-out test execution requires --execute-test"]
    assert record["held_out_test_policy"]["test_start_index"] == 14
    assert record["held_out_test_policy"]["split_block_size"] == 17
    assert record["held_out_test_policy"]["test_block_offset"] == 14
    assert all(row["executed"] is False for row in record["executed"])


def test_post_validation_test_runner_command_shape_is_gated(tmp_path):
    record = run_post_validation_test(_args(tmp_path, status="literal_test_ready", execute_test=True))
    commands = {row["stage"]: row["command"] for row in record["executed"]}

    assert record["status"] == "dry_run"
    assert record["blockers"] == []
    assert "--split-source test=train" in commands["build_test_shard"]
    assert "--split-block-size 17" in commands["build_test_shard"]
    assert "--split-block-offset test=14" in commands["build_test_shard"]
    assert "--train-count 0 --val-count 0" in commands["build_test_shard"]
    assert "--test-count 3" in commands["build_test_shard"]
    assert "--test-split test" in commands["run_gated_test"]
    assert "--test-max-samples 3" in commands["run_gated_test"]
    assert "official_hydrated_transport_shift_gate.json" in commands["run_gated_test"]
