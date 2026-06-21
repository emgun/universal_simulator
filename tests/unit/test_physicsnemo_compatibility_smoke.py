from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.run_physicsnemo_compatibility_smoke import (
    build_parser,
    run_compatibility_smoke,
    validate_physicsnemo_smoke_summary,
)

ROOT = Path(__file__).resolve().parents[2]


def test_physicsnemo_compatibility_smoke_writes_non_metric_manifest(tmp_path):
    evidence_json = tmp_path / "physicsnemo_smoke.json"
    args = build_parser().parse_args(
        [
            "--name",
            "physicsnemo_smoke_test",
            "--output-root",
            str(tmp_path / "out"),
            "--evidence-json",
            str(evidence_json),
            "--tasks",
            "advection1d",
            "burgers1d",
            "darcy2d",
            "--train-split",
            "train",
            "--eval-split",
            "val",
        ]
    )

    summary_path = run_compatibility_smoke(args)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    evidence = json.loads(evidence_json.read_text(encoding="utf-8"))

    assert summary["status"] == "compatibility_smoke_ready"
    assert summary["measurement_type"] == "physicsnemo_compatibility_smoke"
    assert summary["held_out_test_used"] is False
    assert summary["held_out_test_data_read"] is False
    assert summary["claim_comparable"] is False
    assert summary["published_numbers_directly_comparable"] is False
    assert "decoded_rollout_nrmse" not in summary["metrics"]
    assert summary["metrics"] == {}
    assert summary["inspected_splits"] == ["train", "val"]
    assert summary["details"]["package"]["pip_name"] == "nvidia-physicsnemo"
    assert summary["details"]["package"]["import_name"] == "physicsnemo"
    assert summary["details"]["recipe_contract"]["tasks"] == [
        "advection1d",
        "burgers1d",
        "darcy2d",
    ]
    assert summary["details"]["recipe_contract"]["live_metric_allowed"] is False
    assert summary["details"]["recipe_contract"]["next_gate"].startswith(
        "Run a live PhysicsNeMo recipe adapter"
    )
    assert evidence == summary
    assert validate_physicsnemo_smoke_summary(summary) == []


def test_physicsnemo_compatibility_smoke_blocks_test_split_before_output(tmp_path):
    output_root = tmp_path / "out"
    evidence_json = tmp_path / "physicsnemo_smoke.json"

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_physicsnemo_compatibility_smoke.py",
            "--name",
            "blocked_physicsnemo_smoke",
            "--output-root",
            str(output_root),
            "--evidence-json",
            str(evidence_json),
            "--eval-split",
            "test",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode != 0
    assert "must not inspect split=test" in proc.stderr
    assert not output_root.exists()
    assert not evidence_json.exists()


def test_physicsnemo_smoke_validator_rejects_metric_overclaim():
    summary = {
        "schema_version": 1,
        "status": "compatibility_smoke_ready",
        "measurement_type": "physicsnemo_compatibility_smoke",
        "run_name": "physicsnemo_smoke_test",
        "split": "val",
        "inspected_splits": ["train", "val"],
        "metrics": {"decoded_rollout_nrmse": 0.1},
        "claim_comparable": False,
        "published_numbers_directly_comparable": False,
        "held_out_test_used": False,
        "held_out_test_data_read": False,
        "details": {
            "package": {"pip_name": "nvidia-physicsnemo", "import_name": "physicsnemo"},
            "recipe_contract": {
                "tasks": ["advection1d"],
                "live_metric_allowed": False,
                "next_gate": "Run a live PhysicsNeMo recipe adapter on train/val.",
            },
        },
    }

    assert "compatibility smoke must not report decoded_rollout_nrmse" in (
        validate_physicsnemo_smoke_summary(summary)
    )
