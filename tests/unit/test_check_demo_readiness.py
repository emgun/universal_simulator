from __future__ import annotations

import json

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
        candidate_run="ups_light_v1_current_best",
        check_b2=False,
        env_file=tmp_path / ".env",
    )

    assert payload["ready"] is False
    assert payload["manifest"]["expected_key_count"] == 2
    assert "Missing baseline summary: persistence_light_v1_test" in payload["blockers"]
    assert "Missing candidate summary: ups_light_v1_current_best" in payload["blockers"]


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
        summaries / "ups_light_v1_current_best" / "summary.json",
        run_name="ups_light_v1_current_best",
        decoded_rollout_nrmse=0.75,
    )

    payload = readiness_payload(
        manifest=manifest,
        summary_patterns=[str(summaries / "*" / "summary.json")],
        baseline_run="persistence_light_v1_test",
        candidate_run="ups_light_v1_current_best",
        check_b2=False,
        env_file=tmp_path / ".env",
    )

    assert payload["ready"] is True
    assert payload["summaries"]["has_baseline"] is True
    assert payload["summaries"]["has_candidate"] is True
    assert "Build reports/demo/latest with scripts/build_demo_report.py." in payload["next_steps"]
