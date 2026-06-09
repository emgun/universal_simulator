from __future__ import annotations

import json
from argparse import Namespace

from scripts.audit_universal_sota_status import run_audit


def _write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _scorecard(rows):
    return {"rows": rows, "metric_keys": sorted({key for row in rows for key in row})}


def _row(
    run_name: str,
    metric: float,
    *,
    split: str = "test",
    advection: float = 0.7,
    burgers: float = 0.2,
    darcy: float = 0.3,
    spectral: float = 0.1,
    duration_sec: float | None = 1.0,
    wandb_urls: str = "",
):
    return {
        "run_name": run_name,
        "split": split,
        "duration_sec": duration_sec,
        "tracking_wandb_urls": wandb_urls,
        "metric:decoded_rollout_nrmse": metric,
        "metric:task_advection1d_decoded_rollout_nrmse": advection,
        "metric:task_burgers1d_decoded_rollout_nrmse": burgers,
        "metric:task_darcy2d_decoded_rollout_nrmse": darcy,
        "metric:decoded_rollout_spectral_energy_error": spectral,
    }


def test_universal_sota_audit_fails_closed_when_light_gate_is_below_threshold(tmp_path):
    scorecard = _write_json(
        tmp_path / "scorecard.json",
        _scorecard(
            [
                _row("persistence_light_v1_test", 0.570),
                _row("candidate", 0.528),
            ]
        ),
    )
    transport_status = _write_json(
        tmp_path / "transport_status.json", {"status": "literal_achieved"}
    )
    transfer_scorecard = _write_json(
        tmp_path / "transfer_scorecard.json",
        {
            "status": "partial_transfer_validated",
            "calibration_scope": "shared_1d_transport",
            "evaluated_task_count": 2,
            "skipped_task_count": 1,
        },
    )

    record = run_audit(
        Namespace(
            light_scorecard_json=scorecard,
            transport_status_json=transport_status,
            transfer_scorecard_json=transfer_scorecard,
            baseline_run_name="persistence_light_v1_test",
            metric_name="decoded_rollout_nrmse",
            min_improvement=0.2,
            medium_confirmed=False,
            strong_baseline_compared=False,
            artifact_handles_confirmed=False,
            documentation_confirmed=False,
            candidate_summary_glob=[],
            output_json=tmp_path / "out.json",
        )
    )

    assert record["status"] == "not_sota_ready"
    assert record["transport_objective"]["literal_achieved"] is True
    assert record["light_v1"]["best_run_name"] == "candidate"
    assert record["light_v1"]["passes_min_improvement_gate"] is False
    assert "light_v1_min_improvement" in record["blocking_reasons"]
    assert record["transfer"]["status"] == "partial_transfer_validated"
    assert record["transfer"]["calibration_scope"] == "shared_1d_transport"
    assert "learned general PDE operator" in record["next_recommended_path"]
    assert json.loads((tmp_path / "out.json").read_text())["status"] == "not_sota_ready"


def test_universal_sota_audit_marks_ready_only_when_all_claim_criteria_pass(tmp_path):
    scorecard = _write_json(
        tmp_path / "scorecard.json",
        _scorecard(
            [
                _row("persistence_light_v1_test", 1.0),
                _row("candidate", 0.7, wandb_urls="https://wandb.example/run"),
            ]
        ),
    )
    transport_status = _write_json(
        tmp_path / "transport_status.json", {"status": "literal_achieved"}
    )
    transfer_scorecard = _write_json(
        tmp_path / "transfer_scorecard.json",
        {"status": "transfer_validated", "evaluated_task_count": 3, "skipped_task_count": 0},
    )

    record = run_audit(
        Namespace(
            light_scorecard_json=scorecard,
            transport_status_json=transport_status,
            transfer_scorecard_json=transfer_scorecard,
            baseline_run_name="persistence_light_v1_test",
            metric_name="decoded_rollout_nrmse",
            min_improvement=0.2,
            medium_confirmed=True,
            strong_baseline_compared=True,
            artifact_handles_confirmed=True,
            documentation_confirmed=True,
            candidate_summary_glob=[],
            output_json="",
        )
    )

    assert record["status"] == "sota_ready"
    assert record["sota_ready"] is True
    assert record["blocking_reasons"] == []
    assert all(check["passed"] for check in record["readiness_checks"])


def test_universal_sota_audit_can_scan_candidate_summaries_not_in_scorecard(tmp_path):
    scorecard = _write_json(
        tmp_path / "scorecard.json",
        _scorecard([_row("persistence_light_v1_test", 1.0)]),
    )
    summary_dir = tmp_path / "reports" / "ups_light_task_signature_trained_residual"
    summary_dir.mkdir(parents=True)
    _write_json(
        summary_dir / "summary.json",
        {
            "run_name": "ups_light_task_signature_trained_residual",
            "split": "test",
            "duration_sec": 2.0,
            "metrics": {
                "decoded_rollout_nrmse": 0.8,
                "task_advection1d_decoded_rollout_nrmse": 0.9,
                "task_burgers1d_decoded_rollout_nrmse": 0.7,
                "task_darcy2d_decoded_rollout_nrmse": 0.6,
                "decoded_rollout_spectral_energy_error": 0.1,
            },
        },
    )
    diagnostic_dir = tmp_path / "reports" / "ups_light_gate_hook_transport_base_val"
    diagnostic_dir.mkdir(parents=True)
    _write_json(
        diagnostic_dir / "summary.json",
        {
            "run_name": "ups_light_gate_hook_transport_base_val",
            "split": "val",
            "duration_sec": 1.0,
            "metrics": {
                "decoded_rollout_nrmse": 0.1,
                "task_advection1d_decoded_rollout_nrmse": 0.1,
                "task_burgers1d_decoded_rollout_nrmse": 0.1,
                "task_darcy2d_decoded_rollout_nrmse": 0.1,
                "decoded_rollout_spectral_energy_error": 0.1,
            },
        },
    )
    transport_status = _write_json(
        tmp_path / "transport_status.json", {"status": "literal_achieved"}
    )
    transfer_scorecard = _write_json(
        tmp_path / "transfer_scorecard.json",
        {"status": "partial_transfer_validated", "evaluated_task_count": 2},
    )

    record = run_audit(
        Namespace(
            light_scorecard_json=scorecard,
            transport_status_json=transport_status,
            transfer_scorecard_json=transfer_scorecard,
            baseline_run_name="persistence_light_v1_test",
            metric_name="decoded_rollout_nrmse",
            min_improvement=0.2,
            medium_confirmed=False,
            strong_baseline_compared=False,
            artifact_handles_confirmed=False,
            documentation_confirmed=False,
            candidate_summary_glob=[str(tmp_path / "reports" / "ups_light*" / "summary.json")],
            output_json="",
        )
    )

    assert record["light_v1"]["best_run_name"] == "ups_light_task_signature_trained_residual"
    assert record["light_v1"]["best_metric_value"] == 0.8
    assert record["light_v1"]["claim_eligible_run_count"] == 1
    assert record["light_v1"]["passes_min_improvement_gate"] is False


def test_universal_sota_audit_uses_summary_cost_and_artifact_handles(tmp_path):
    scorecard = _write_json(
        tmp_path / "scorecard.json",
        _scorecard([_row("persistence_light_v1_test", 1.0)]),
    )
    summary_dir = tmp_path / "reports" / "ups_light_claim_candidate"
    summary_dir.mkdir(parents=True)
    _write_json(
        summary_dir / "summary_test.json",
        {
            "run_name": "ups_light_claim_candidate",
            "split": "test",
            "artifact_handles": "b2://pdebench/remote-runs/light/claim.tar.gz",
            "cost": {
                "provider": "vast",
                "gpu_type": "RTX 4090",
                "gpu_count": 1,
                "wall_clock_hours": 0.5,
                "hourly_usd": 0.8,
            },
            "metrics": {
                "decoded_rollout_nrmse": 0.7,
                "task_advection1d_decoded_rollout_nrmse": 0.8,
                "task_burgers1d_decoded_rollout_nrmse": 0.6,
                "task_darcy2d_decoded_rollout_nrmse": 0.7,
                "decoded_rollout_spectral_energy_error": 0.1,
            },
        },
    )
    transport_status = _write_json(
        tmp_path / "transport_status.json", {"status": "literal_achieved"}
    )
    transfer_scorecard = _write_json(
        tmp_path / "transfer_scorecard.json",
        {"status": "partial_transfer_validated", "evaluated_task_count": 2},
    )

    record = run_audit(
        Namespace(
            light_scorecard_json=scorecard,
            transport_status_json=transport_status,
            transfer_scorecard_json=transfer_scorecard,
            baseline_run_name="persistence_light_v1_test",
            metric_name="decoded_rollout_nrmse",
            min_improvement=0.2,
            medium_confirmed=False,
            strong_baseline_compared=False,
            artifact_handles_confirmed=False,
            documentation_confirmed=False,
            candidate_summary_glob=[str(summary_dir / "summary_test.json")],
            claim_split="test",
            output_json="",
        )
    )

    assert record["light_v1"]["cost_or_throughput_present"] is True
    assert record["light_v1"]["wandb_or_artifact_handles_present"] is True
    scorecard_check = next(
        check
        for check in record["readiness_checks"]
        if check["key"] == "scorecard_metrics_complete"
    )
    handles_check = next(
        check for check in record["readiness_checks"] if check["key"] == "wandb_or_artifact_handles"
    )
    assert scorecard_check["passed"] is True
    assert handles_check["passed"] is True


def test_universal_sota_audit_preserves_scorecard_metadata_when_summary_overlaps(tmp_path):
    scorecard = _write_json(
        tmp_path / "scorecard.json",
        _scorecard(
            [
                _row("persistence_light_v1_test", 1.0),
                {
                    **_row("ups_light_claim_candidate", 0.7, duration_sec=""),
                    "artifact_handles": "b2://pdebench/remote-runs/light/claim.tar.gz",
                    "cost_estimated_usd": 0.4,
                },
            ]
        ),
    )
    summary_dir = tmp_path / "reports" / "ups_light_claim_candidate"
    summary_dir.mkdir(parents=True)
    _write_json(
        summary_dir / "summary_test.json",
        {
            "run_name": "ups_light_claim_candidate",
            "split": "test",
            "metrics": {
                "decoded_rollout_nrmse": 0.7,
                "task_advection1d_decoded_rollout_nrmse": 0.8,
                "task_burgers1d_decoded_rollout_nrmse": 0.6,
                "task_darcy2d_decoded_rollout_nrmse": 0.7,
                "decoded_rollout_spectral_energy_error": 0.1,
            },
        },
    )
    transport_status = _write_json(
        tmp_path / "transport_status.json", {"status": "literal_achieved"}
    )
    transfer_scorecard = _write_json(
        tmp_path / "transfer_scorecard.json",
        {"status": "partial_transfer_validated", "evaluated_task_count": 2},
    )

    record = run_audit(
        Namespace(
            light_scorecard_json=scorecard,
            transport_status_json=transport_status,
            transfer_scorecard_json=transfer_scorecard,
            baseline_run_name="persistence_light_v1_test",
            metric_name="decoded_rollout_nrmse",
            min_improvement=0.2,
            medium_confirmed=False,
            strong_baseline_compared=False,
            artifact_handles_confirmed=False,
            documentation_confirmed=False,
            candidate_summary_glob=[str(summary_dir / "summary_test.json")],
            claim_split="test",
            output_json="",
        )
    )

    assert record["light_v1"]["cost_or_throughput_present"] is True
    assert record["light_v1"]["wandb_or_artifact_handles_present"] is True


def test_universal_sota_audit_excludes_non_test_candidate_summaries(tmp_path):
    scorecard = _write_json(
        tmp_path / "scorecard.json",
        _scorecard([_row("persistence_light_v1_test", 1.0)]),
    )
    val_dir = tmp_path / "reports" / "ups_light_candidate_val"
    val_dir.mkdir(parents=True)
    _write_json(
        val_dir / "summary.json",
        {
            "run_name": "ups_light_candidate_val",
            "split": "val",
            "duration_sec": 2.0,
            "metrics": {
                "decoded_rollout_nrmse": 0.1,
                "task_advection1d_decoded_rollout_nrmse": 0.1,
                "task_burgers1d_decoded_rollout_nrmse": 0.1,
                "task_darcy2d_decoded_rollout_nrmse": 0.1,
                "decoded_rollout_spectral_energy_error": 0.1,
            },
        },
    )
    test_dir = tmp_path / "reports" / "ups_light_candidate_test"
    test_dir.mkdir(parents=True)
    _write_json(
        test_dir / "summary_test.json",
        {
            "run_name": "ups_light_candidate_test",
            "split": "test",
            "duration_sec": 3.0,
            "metrics": {
                "decoded_rollout_nrmse": 0.8,
                "task_advection1d_decoded_rollout_nrmse": 0.9,
                "task_burgers1d_decoded_rollout_nrmse": 0.7,
                "task_darcy2d_decoded_rollout_nrmse": 0.6,
                "decoded_rollout_spectral_energy_error": 0.1,
            },
        },
    )
    transport_status = _write_json(
        tmp_path / "transport_status.json", {"status": "literal_achieved"}
    )
    transfer_scorecard = _write_json(
        tmp_path / "transfer_scorecard.json",
        {"status": "partial_transfer_validated", "evaluated_task_count": 2},
    )

    record = run_audit(
        Namespace(
            light_scorecard_json=scorecard,
            transport_status_json=transport_status,
            transfer_scorecard_json=transfer_scorecard,
            baseline_run_name="persistence_light_v1_test",
            metric_name="decoded_rollout_nrmse",
            min_improvement=0.2,
            medium_confirmed=False,
            strong_baseline_compared=False,
            artifact_handles_confirmed=False,
            documentation_confirmed=False,
            candidate_summary_glob=[str(tmp_path / "reports" / "ups_light*" / "summary*.json")],
            claim_split="test",
            output_json="",
        )
    )

    assert record["light_v1"]["best_run_name"] == "ups_light_candidate_test"
    assert record["light_v1"]["best_metric_value"] == 0.8


def test_universal_sota_audit_uses_claim_evidence_for_docs_cost_and_artifacts(tmp_path):
    scorecard = _write_json(
        tmp_path / "scorecard.json",
        _scorecard([_row("persistence_light_v1_test", 1.0)]),
    )
    summary_dir = tmp_path / "reports" / "ups_light_claim_candidate"
    summary_dir.mkdir(parents=True)
    summary_path = _write_json(
        summary_dir / "summary_test.json",
        {
            "run_name": "ups_light_claim_candidate",
            "split": "test",
            "metrics": {
                "decoded_rollout_nrmse": 0.7,
                "task_advection1d_decoded_rollout_nrmse": 0.8,
                "task_burgers1d_decoded_rollout_nrmse": 0.6,
                "task_darcy2d_decoded_rollout_nrmse": 0.7,
                "decoded_rollout_spectral_energy_error": 0.1,
            },
        },
    )
    claim_evidence = _write_json(
        tmp_path / "claim_evidence.json",
        {
            "candidate_evidence": [
                {
                    "run_name": "ups_light_claim_candidate",
                    "split": "test",
                    "summary_json": str(summary_path),
                    "duration_sec": 12.5,
                    "artifact_handles": ["git:docs/claim_evidence/artifacts/claim.tar.gz"],
                    "cost": {
                        "provider": "local",
                        "wall_clock_hours": 12.5 / 3600.0,
                    },
                }
            ],
            "claim_documentation": {
                "status": "complete",
                "run_name": "ups_light_claim_candidate",
                "split": "test",
                "summary_json": str(summary_path),
                "metric_name": "decoded_rollout_nrmse",
                "metric_value": 0.7,
                "commit": "abc123",
                "command": "python scripts/run_light_experiment.py ...",
                "checkpoints": {
                    "operator": "checkpoints/operator.pt",
                    "encoder": "checkpoints/encoder.pt",
                    "decoder": "checkpoints/decoder.pt",
                },
                "artifact_handles": ["git:docs/claim_evidence/artifacts/claim.tar.gz"],
            },
            "strong_baseline_comparison": {
                "status": "not_satisfied",
                "reason": "validation-only Fourier baseline is not a claim-level test comparison",
            },
        },
    )
    transport_status = _write_json(
        tmp_path / "transport_status.json", {"status": "literal_achieved"}
    )
    transfer_scorecard = _write_json(
        tmp_path / "transfer_scorecard.json",
        {"status": "partial_transfer_validated", "evaluated_task_count": 2},
    )

    record = run_audit(
        Namespace(
            light_scorecard_json=scorecard,
            transport_status_json=transport_status,
            transfer_scorecard_json=transfer_scorecard,
            claim_evidence_json=claim_evidence,
            baseline_run_name="persistence_light_v1_test",
            metric_name="decoded_rollout_nrmse",
            min_improvement=0.2,
            medium_confirmed=True,
            strong_baseline_compared=False,
            artifact_handles_confirmed=False,
            documentation_confirmed=False,
            candidate_summary_glob=[str(summary_path)],
            claim_split="test",
            output_json="",
        )
    )

    assert record["light_v1"]["cost_or_throughput_present"] is True
    assert record["light_v1"]["wandb_or_artifact_handles_present"] is True
    assert record["claim_documentation"]["validated"] is True
    assert "claim_documentation_confirmed" not in record["blocking_reasons"]
    assert "strong_baseline_comparison" in record["blocking_reasons"]


def test_universal_sota_audit_accepts_validated_medium_confirmation_evidence(tmp_path):
    scorecard = _write_json(
        tmp_path / "scorecard.json",
        _scorecard([_row("persistence_light_v1_test", 1.0)]),
    )
    medium_evidence = _write_json(
        tmp_path / "medium_confirmation.json",
        {
            "artifact": {
                "handle": "b2://pdebench/remote-runs/medium/medium-confirmation.tar.gz",
                "published": True,
            },
            "candidate": {
                "metric_value": 0.3,
                "run_name": "ups_medium_shared_context_transport",
            },
            "comparison_to_persistence": {
                "absolute_improvement": 0.3,
                "baseline_metric_value": 0.6,
                "candidate_metric_value": 0.3,
                "improvement_fraction": 0.5,
                "metric_name": "decoded_rollout_nrmse",
                "persistence_run_name": "persistence_medium_v1_test",
            },
            "confirmation_scope": {
                "data_root": "data/pdebench_medium_v1",
                "decoded_rollout_steps": 16,
                "split": "test",
                "test_samples": 128,
                "train_samples": 512,
                "val_samples": 128,
                "version": "medium-v1",
            },
            "measurement_type": "medium_or_larger_confirmation_evidence",
            "selection_policy": {
                "selected_from_light_v1": True,
                "test_tuned": False,
            },
            "status": "complete",
        },
    )
    claim_evidence = _write_json(
        tmp_path / "claim_evidence.json",
        {
            "candidate_evidence": [
                {
                    "artifact_handles": ["repo:docs/claim_evidence/artifacts/claim.tar.gz"],
                    "duration_sec": 12.5,
                    "metrics": {
                        "decoded_rollout_nrmse": 0.7,
                        "decoded_rollout_spectral_energy_error": 0.1,
                        "task_advection1d_decoded_rollout_nrmse": 0.8,
                        "task_burgers1d_decoded_rollout_nrmse": 0.6,
                        "task_darcy2d_decoded_rollout_nrmse": 0.7,
                    },
                    "run_name": "ups_light_claim_candidate",
                    "split": "test",
                    "summary_json": "reports/claim/summary_test.json",
                }
            ],
            "claim_documentation": {
                "artifact_handles": ["repo:docs/claim_evidence/artifacts/claim.tar.gz"],
                "checkpoints": {
                    "decoder": "checkpoints/decoder.pt",
                    "encoder": "checkpoints/encoder.pt",
                    "operator": "checkpoints/operator.pt",
                },
                "command": "python scripts/run_light_experiment.py --decoded-rollout-steps 16",
                "commit": "abc123",
                "metric_name": "decoded_rollout_nrmse",
                "metric_value": 0.7,
                "run_name": "ups_light_claim_candidate",
                "split": "test",
                "status": "complete",
                "summary_json": "reports/claim/summary_test.json",
            },
            "medium_or_larger_confirmation": {
                "evidence_json": str(medium_evidence),
                "status": "complete",
            },
            "strong_baseline_comparison": {
                "artifact_handles": ["repo:baseline.tar.gz"],
                "baseline_metric_value": 0.9,
                "baseline_run_name": "physical_fourier_test",
                "candidate_metric_value": 0.7,
                "claim_run_name": "ups_light_claim_candidate",
                "metric_name": "decoded_rollout_nrmse",
                "split": "test",
                "status": "complete",
            },
        },
    )
    transport_status = _write_json(
        tmp_path / "transport_status.json",
        {"status": "literal_achieved"},
    )
    transfer_scorecard = _write_json(
        tmp_path / "transfer_scorecard.json",
        {"evaluated_task_count": 3, "skipped_task_count": 0, "status": "transfer_validated"},
    )

    record = run_audit(
        Namespace(
            artifact_handles_confirmed=False,
            baseline_run_name="persistence_light_v1_test",
            candidate_summary_glob=[],
            claim_evidence_json=claim_evidence,
            claim_split="test",
            documentation_confirmed=False,
            light_scorecard_json=scorecard,
            medium_confirmed=False,
            metric_name="decoded_rollout_nrmse",
            min_improvement=0.2,
            output_json="",
            strong_baseline_compared=False,
            transfer_scorecard_json=transfer_scorecard,
            transport_status_json=transport_status,
        )
    )

    assert record["status"] == "sota_ready"
    assert record["medium_or_larger_confirmation"]["validated"] is True
    assert "medium_or_larger_confirmation" not in record["blocking_reasons"]


def test_universal_sota_audit_reports_scoped_ct1_variant_separately(tmp_path):
    scorecard = _write_json(
        tmp_path / "scorecard.json",
        _scorecard([_row("persistence_light_v1_test", 1.0)]),
    )
    claim_evidence = _write_json(
        tmp_path / "claim_evidence.json",
        {
            "candidate_evidence": [
                {
                    "run_name": "ups_light_shared_context_transport_guarded",
                    "split": "test",
                    "summary_json": "reports/ct8/summary_test.json",
                    "duration_sec": 12.5,
                    "artifact_handles": ["repo:docs/claim_evidence/artifacts/ct8.tar.gz"],
                    "metrics": {
                        "decoded_rollout_nrmse": 0.4165820594268877,
                        "task_advection1d_decoded_rollout_nrmse": 0.5765863333379032,
                        "task_burgers1d_decoded_rollout_nrmse": 0.17446857017795178,
                        "task_darcy2d_decoded_rollout_nrmse": 0.20909553062258152,
                        "decoded_rollout_spectral_energy_error": 0.06721625502425291,
                    },
                }
            ],
            "claim_documentation": {
                "status": "complete",
                "run_name": "ups_light_shared_context_transport_guarded",
                "split": "test",
                "summary_json": "reports/ct8/summary_test.json",
                "metric_name": "decoded_rollout_nrmse",
                "metric_value": 0.4165820594268877,
                "commit": "abc123",
                "command": "python scripts/run_light_experiment.py --decoded-rollout-steps 16",
                "checkpoints": {
                    "operator": "checkpoints/operator.pt",
                    "encoder": "checkpoints/encoder.pt",
                    "decoder": "checkpoints/decoder.pt",
                },
                "artifact_handles": ["repo:docs/claim_evidence/artifacts/ct8.tar.gz"],
            },
            "strong_baseline_comparison": {
                "status": "complete",
                "claim_run_name": "ups_light_shared_context_transport_guarded",
                "baseline_run_name": "physical_fourier_test",
                "split": "test",
                "metric_name": "decoded_rollout_nrmse",
                "candidate_metric_value": 0.4165820594268877,
                "baseline_metric_value": 0.5636730976415197,
                "artifact_handles": ["repo:baseline.tar.gz"],
            },
            "scoped_claim_variants": [
                {
                    "variant_id": "light_v1_ct1_online_transport_context",
                    "status": "held_out_complete",
                    "claim_contract_label": "light-v1 CT1 online transport-context UPS variant",
                    "run_name": "ups_light_advection_context_transport_only_ct1_guarded",
                    "split": "test",
                    "metric_name": "decoded_rollout_nrmse",
                    "metric_value": 0.20177292896682064,
                    "evidence_json": (
                        "docs/claim_evidence/" "ups_advection_ct1_heldout_light_v1_evidence.json"
                    ),
                    "artifact_handles": [
                        "repo:docs/claim_evidence/artifacts/"
                        "ups_advection_ct1_heldout_light_v1.tar.gz"
                    ],
                    "artifact_sha256": "b3b0809afc58085433ba0bbe1efbfa87deb1c227",
                    "same_exact_inference_contract_as_primary": False,
                    "not_autonomous_rollout_claim": True,
                    "published_numbers_directly_comparable": False,
                    "external_paper_reproduction": False,
                }
            ],
        },
    )
    transport_status = _write_json(
        tmp_path / "transport_status.json", {"status": "literal_achieved"}
    )
    transfer_scorecard = _write_json(
        tmp_path / "transfer_scorecard.json",
        {"status": "partial_transfer_validated", "evaluated_task_count": 2},
    )

    record = run_audit(
        Namespace(
            light_scorecard_json=scorecard,
            transport_status_json=transport_status,
            transfer_scorecard_json=transfer_scorecard,
            claim_evidence_json=claim_evidence,
            baseline_run_name="persistence_light_v1_test",
            metric_name="decoded_rollout_nrmse",
            min_improvement=0.2,
            medium_confirmed=True,
            strong_baseline_compared=False,
            artifact_handles_confirmed=False,
            documentation_confirmed=False,
            candidate_summary_glob=[],
            claim_split="test",
            output_json="",
        )
    )

    assert record["light_v1"]["best_run_name"] == "ups_light_shared_context_transport_guarded"
    assert record["scoped_claim_variants"]["validated_count"] == 1
    assert record["scoped_claim_variants"]["best_valid_variant"]["variant_id"] == (
        "light_v1_ct1_online_transport_context"
    )
    assert record["scoped_claim_variants"]["best_valid_variant"]["metric_value"] == (
        0.20177292896682064
    )
    assert (
        record["scoped_claim_variants"]["best_valid_variant"][
            "same_exact_inference_contract_as_primary"
        ]
        is False
    )


def test_universal_sota_audit_rejects_mismatched_claim_documentation(tmp_path):
    scorecard = _write_json(
        tmp_path / "scorecard.json",
        _scorecard([_row("persistence_light_v1_test", 1.0)]),
    )
    summary_dir = tmp_path / "reports" / "ups_light_claim_candidate"
    summary_dir.mkdir(parents=True)
    summary_path = _write_json(
        summary_dir / "summary_test.json",
        {
            "run_name": "ups_light_claim_candidate",
            "split": "test",
            "duration_sec": 2.0,
            "metrics": {
                "decoded_rollout_nrmse": 0.7,
                "task_advection1d_decoded_rollout_nrmse": 0.8,
                "task_burgers1d_decoded_rollout_nrmse": 0.6,
                "task_darcy2d_decoded_rollout_nrmse": 0.7,
                "decoded_rollout_spectral_energy_error": 0.1,
            },
        },
    )
    claim_evidence = _write_json(
        tmp_path / "claim_evidence.json",
        {
            "claim_documentation": {
                "status": "complete",
                "run_name": "different_candidate",
                "split": "test",
                "summary_json": str(summary_path),
                "metric_name": "decoded_rollout_nrmse",
                "metric_value": 0.7,
                "commit": "abc123",
                "command": "python scripts/run_light_experiment.py ...",
                "checkpoints": {
                    "operator": "checkpoints/operator.pt",
                    "encoder": "checkpoints/encoder.pt",
                    "decoder": "checkpoints/decoder.pt",
                },
                "artifact_handles": ["git:docs/claim_evidence/artifacts/claim.tar.gz"],
            }
        },
    )
    transport_status = _write_json(
        tmp_path / "transport_status.json", {"status": "literal_achieved"}
    )
    transfer_scorecard = _write_json(
        tmp_path / "transfer_scorecard.json",
        {"status": "partial_transfer_validated", "evaluated_task_count": 2},
    )

    record = run_audit(
        Namespace(
            light_scorecard_json=scorecard,
            transport_status_json=transport_status,
            transfer_scorecard_json=transfer_scorecard,
            claim_evidence_json=claim_evidence,
            baseline_run_name="persistence_light_v1_test",
            metric_name="decoded_rollout_nrmse",
            min_improvement=0.2,
            medium_confirmed=True,
            strong_baseline_compared=True,
            artifact_handles_confirmed=True,
            documentation_confirmed=False,
            candidate_summary_glob=[str(summary_path)],
            claim_split="test",
            output_json="",
        )
    )

    assert record["claim_documentation"]["validated"] is False
    assert "claim_documentation_confirmed" in record["blocking_reasons"]


def test_universal_sota_audit_validates_complete_strong_baseline_evidence(tmp_path):
    summary_dir = tmp_path / "reports" / "ups_light_claim_candidate"
    summary_dir.mkdir(parents=True)
    summary_path = _write_json(
        summary_dir / "summary_test.json",
        {
            "run_name": "ups_light_claim_candidate",
            "split": "test",
            "duration_sec": 2.0,
            "artifact_handles": "repo:claim.tar.gz",
            "metrics": {
                "decoded_rollout_nrmse": 0.7,
                "task_advection1d_decoded_rollout_nrmse": 0.8,
                "task_burgers1d_decoded_rollout_nrmse": 0.6,
                "task_darcy2d_decoded_rollout_nrmse": 0.7,
                "decoded_rollout_spectral_energy_error": 0.1,
            },
        },
    )
    scorecard = _write_json(
        tmp_path / "scorecard.json",
        _scorecard([_row("persistence_light_v1_test", 1.0)]),
    )
    claim_evidence = _write_json(
        tmp_path / "claim_evidence.json",
        {
            "claim_documentation": {
                "status": "complete",
                "run_name": "ups_light_claim_candidate",
                "split": "test",
                "summary_json": str(summary_path),
                "metric_name": "decoded_rollout_nrmse",
                "metric_value": 0.7,
                "commit": "abc123",
                "command": "python scripts/run_light_experiment.py ...",
                "checkpoints": {
                    "operator": "checkpoints/operator.pt",
                    "encoder": "checkpoints/encoder.pt",
                    "decoder": "checkpoints/decoder.pt",
                },
                "artifact_handles": ["repo:claim.tar.gz"],
            },
            "strong_baseline_comparison": {
                "status": "complete",
                "claim_run_name": "ups_light_claim_candidate",
                "baseline_run_name": "physical_fourier_test",
                "split": "test",
                "metric_name": "decoded_rollout_nrmse",
                "candidate_metric_value": 0.7,
                "baseline_metric_value": 0.9,
                "artifact_handles": ["repo:baseline.tar.gz"],
            },
        },
    )
    transport_status = _write_json(
        tmp_path / "transport_status.json", {"status": "literal_achieved"}
    )
    transfer_scorecard = _write_json(
        tmp_path / "transfer_scorecard.json",
        {"status": "partial_transfer_validated", "evaluated_task_count": 2},
    )

    record = run_audit(
        Namespace(
            light_scorecard_json=scorecard,
            transport_status_json=transport_status,
            transfer_scorecard_json=transfer_scorecard,
            claim_evidence_json=claim_evidence,
            baseline_run_name="persistence_light_v1_test",
            metric_name="decoded_rollout_nrmse",
            min_improvement=0.2,
            medium_confirmed=True,
            strong_baseline_compared=False,
            artifact_handles_confirmed=False,
            documentation_confirmed=False,
            candidate_summary_glob=[str(summary_path)],
            claim_split="test",
            output_json="",
        )
    )

    assert record["status"] == "sota_ready"
    assert record["strong_baseline_comparison"]["validated"] is True
    assert (
        record["strong_baseline_comparison"]["baseline_improvement_fraction"] == (0.9 - 0.7) / 0.9
    )
    assert record["blocking_reasons"] == []


def test_universal_sota_audit_rejects_losing_strong_baseline_evidence(tmp_path):
    summary_dir = tmp_path / "reports" / "ups_light_claim_candidate"
    summary_dir.mkdir(parents=True)
    summary_path = _write_json(
        summary_dir / "summary_test.json",
        {
            "run_name": "ups_light_claim_candidate",
            "split": "test",
            "duration_sec": 2.0,
            "artifact_handles": "repo:claim.tar.gz",
            "metrics": {
                "decoded_rollout_nrmse": 0.95,
                "task_advection1d_decoded_rollout_nrmse": 0.8,
                "task_burgers1d_decoded_rollout_nrmse": 0.6,
                "task_darcy2d_decoded_rollout_nrmse": 0.7,
                "decoded_rollout_spectral_energy_error": 0.1,
            },
        },
    )
    scorecard = _write_json(
        tmp_path / "scorecard.json",
        _scorecard([_row("persistence_light_v1_test", 1.0)]),
    )
    claim_evidence = _write_json(
        tmp_path / "claim_evidence.json",
        {
            "claim_documentation": {
                "status": "complete",
                "run_name": "ups_light_claim_candidate",
                "split": "test",
                "summary_json": str(summary_path),
                "metric_name": "decoded_rollout_nrmse",
                "metric_value": 0.95,
                "commit": "abc123",
                "command": "python scripts/run_light_experiment.py ...",
                "checkpoints": {
                    "operator": "checkpoints/operator.pt",
                    "encoder": "checkpoints/encoder.pt",
                    "decoder": "checkpoints/decoder.pt",
                },
                "artifact_handles": ["repo:claim.tar.gz"],
            },
            "strong_baseline_comparison": {
                "status": "complete",
                "claim_run_name": "ups_light_claim_candidate",
                "baseline_run_name": "physical_fourier_test",
                "split": "test",
                "metric_name": "decoded_rollout_nrmse",
                "candidate_metric_value": 0.95,
                "baseline_metric_value": 0.9,
                "artifact_handles": ["repo:baseline.tar.gz"],
            },
        },
    )
    transport_status = _write_json(
        tmp_path / "transport_status.json", {"status": "literal_achieved"}
    )
    transfer_scorecard = _write_json(
        tmp_path / "transfer_scorecard.json",
        {"status": "partial_transfer_validated", "evaluated_task_count": 2},
    )

    record = run_audit(
        Namespace(
            light_scorecard_json=scorecard,
            transport_status_json=transport_status,
            transfer_scorecard_json=transfer_scorecard,
            claim_evidence_json=claim_evidence,
            baseline_run_name="persistence_light_v1_test",
            metric_name="decoded_rollout_nrmse",
            min_improvement=0.0,
            medium_confirmed=True,
            strong_baseline_compared=False,
            artifact_handles_confirmed=False,
            documentation_confirmed=False,
            candidate_summary_glob=[str(summary_path)],
            claim_split="test",
            output_json="",
        )
    )

    assert record["strong_baseline_comparison"]["validated"] is False
    assert "candidate_beats_baseline" in record["strong_baseline_comparison"]["reason"]
    assert "strong_baseline_comparison" in record["blocking_reasons"]
