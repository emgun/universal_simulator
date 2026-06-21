from __future__ import annotations

import json
import tarfile

from scripts.build_showcase_assets import (
    build_benchmark_readiness_rows,
    build_benchmark_rows,
    build_external_matrix_rows,
    build_horizon_rows,
    build_metric_suite_rows,
    build_reproducibility_card_rows,
    build_rollout_preview_status_rows,
    build_task_rows,
    build_transfer_rows,
    build_transport_ablation_rows,
    sha256_file,
)


def _fixture_payloads():
    claim_evidence = {
        "candidate_evidence": [
            {
                "run_name": "ups_primary",
                "split": "test",
                "metrics": {
                    "decoded_rollout_nrmse": 0.4,
                    "decoded_rollout_spectral_energy_error": 0.05,
                    "task_advection1d_decoded_rollout_nrmse": 0.5,
                    "task_burgers1d_decoded_rollout_nrmse": 0.2,
                    "task_darcy2d_decoded_rollout_nrmse": 0.3,
                },
                "artifact_sha256": "primary-sha",
            }
        ],
        "claim_documentation": {
            "metric_name": "decoded_rollout_nrmse",
            "metric_value": 0.4,
            "artifact_sha256": "primary-doc-sha",
        },
        "scoped_claim_variants": [
            {
                "variant_id": "light_v1_ct1",
                "claim_contract_label": "CT1 online context",
                "run_name": "ups_ct1",
                "split": "test",
                "metric_name": "decoded_rollout_nrmse",
                "metric_value": 0.25,
                "metrics": {
                    "decoded_rollout_nrmse": 0.25,
                    "task_advection1d_decoded_rollout_nrmse": 0.1,
                    "task_burgers1d_decoded_rollout_nrmse": 0.2,
                    "task_darcy2d_decoded_rollout_nrmse": 0.3,
                },
                "artifact_sha256": "ct1-sha",
                "published_numbers_directly_comparable": False,
                "same_exact_inference_contract_as_primary": False,
            }
        ],
        "strong_baseline_comparison": {
            "baseline_run_name": "physical_fourier",
            "baseline_family": "Physical Fourier",
            "baseline_metric_value": 0.55,
            "baseline_artifact_sha256": "fourier-sha",
            "baseline_metrics": {
                "decoded_rollout_nrmse": 0.55,
                "task_advection1d_decoded_rollout_nrmse": 0.7,
                "task_burgers1d_decoded_rollout_nrmse": 0.25,
                "task_darcy2d_decoded_rollout_nrmse": 0.35,
            },
        },
    }
    external_mapping = {
        "baseline_candidates": [
            {
                "candidate_id": "fno_light_v1",
                "model_family": "FNO",
                "status": "selected_primary_reproduction_path",
                "source_refs": ["pdebench_official_repo", "neuraloperator_official_repo"],
                "test_measurements": [
                    {
                        "run_name": "external_fno",
                        "split": "test",
                        "metric_name": "decoded_rollout_nrmse",
                        "metric_value": 0.7,
                        "claim_comparable": True,
                        "held_out_test_used": True,
                        "published_numbers_directly_comparable": False,
                        "evidence_json": "docs/claim_evidence/fno.json",
                        "artifact_handle": "repo:fno.tar.gz",
                    }
                ],
            },
            {
                "candidate_id": "poseidon_transfer",
                "model_family": "Poseidon",
                "status": "foundation_transfer_validation_measured_finetuning_required",
                "why_not_primary": "Validation-only transfer path did not clear the gate.",
            },
        ],
        "external_sources": [
            {
                "source_id": "pdebench_official_repo",
                "title": "PDEBench official repository",
                "url": "https://github.com/pdebench/PDEBench",
            },
            {
                "source_id": "neuraloperator_official_repo",
                "title": "NeuralOperator official repository",
                "url": "https://github.com/neuraloperator/neuraloperator",
            },
        ],
    }
    durable_scorecard = {
        "rows": [
            {
                "run_name": "persistence_light_v1_test",
                "metric:decoded_rollout_nrmse": 0.6,
                "metric:decoded_rollout_mae": 0.3,
                "metric:decoded_rollout_mse": 0.36,
                "metric:decoded_rollout_spectral_energy_error": 0.05,
                "metric:decoded_step1_nrmse": 0.7,
                "metric:decoded_h4_nrmse": 0.55,
                "metric:decoded_h16_nrmse": 0.56,
                "metric:task_advection1d_decoded_rollout_nrmse": 0.8,
                "metric:task_burgers1d_decoded_rollout_nrmse": 0.2,
                "metric:task_darcy2d_decoded_rollout_nrmse": 0.3,
            }
        ]
    }
    return claim_evidence, external_mapping, durable_scorecard


def test_build_benchmark_rows_labels_claim_comparability_and_external_results():
    claim_evidence, external_mapping, durable_scorecard = _fixture_payloads()

    rows = build_benchmark_rows(claim_evidence, external_mapping, durable_scorecard)
    by_label = {row["label"]: row for row in rows}

    assert by_label["UPS primary claim"]["metric_value"] == 0.4
    assert by_label["UPS primary claim"]["claim_comparable"] is True
    assert by_label["Persistence baseline"]["metric_value"] == 0.6
    assert by_label["Physical Fourier"]["metric_value"] == 0.55
    assert by_label["FNO"]["metric_value"] == 0.7
    assert by_label["FNO"]["source_refs"] == "pdebench_official_repo,neuraloperator_official_repo"
    assert by_label["FNO"]["published_numbers_directly_comparable"] is False
    assert by_label["CT1 online context"]["claim_comparable"] is False
    assert by_label["FNO"]["primary_improvement_fraction"] == (0.7 - 0.4) / 0.7


def test_build_task_rows_emits_one_row_per_task_for_available_metrics():
    claim_evidence, external_mapping, durable_scorecard = _fixture_payloads()

    rows = build_task_rows(claim_evidence, external_mapping, durable_scorecard)
    labels = {(row["label"], row["task"]): row["metric_value"] for row in rows}

    assert labels[("UPS primary claim", "advection1d")] == 0.5
    assert labels[("UPS primary claim", "burgers1d")] == 0.2
    assert labels[("UPS primary claim", "darcy2d")] == 0.3
    assert labels[("Persistence baseline", "advection1d")] == 0.8
    assert labels[("Physical Fourier", "advection1d")] == 0.7
    assert labels[("CT1 online context", "advection1d")] == 0.1


def test_build_metric_suite_rows_reads_primary_tar_metrics_and_compares_persistence(tmp_path):
    claim_evidence, _, durable_scorecard = _fixture_payloads()
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    summary_path = artifact_dir / "summary_test.json"
    summary_path.write_text(
        json.dumps(
            {
                "metrics": {
                    "decoded_rollout_nrmse": 0.4,
                    "decoded_rollout_mae": 0.18,
                    "decoded_rollout_mse": 0.16,
                    "decoded_rollout_spectral_energy_error": 0.05,
                    "decoded_step1_nrmse": 0.7,
                    "decoded_h4_nrmse": 0.55,
                    "decoded_h16_nrmse": 0.08,
                }
            }
        ),
        encoding="utf-8",
    )
    tar_path = artifact_dir / "primary.tar.gz"
    with tarfile.open(tar_path, "w:gz") as archive:
        archive.add(summary_path, arcname="ups_primary/summary_test.json")
    claim_evidence["candidate_evidence"][0]["artifact_handles"] = ["repo:artifacts/primary.tar.gz"]

    rows = build_metric_suite_rows(claim_evidence, durable_scorecard, artifact_root=tmp_path)
    by_metric = {row["metric_name"]: row for row in rows}

    assert by_metric["decoded_rollout_nrmse"]["ups_value"] == 0.4
    assert by_metric["decoded_rollout_mae"]["persistence_value"] == 0.3
    assert by_metric["decoded_rollout_mse"]["relative_improvement_fraction"] == (0.36 - 0.16) / 0.36
    assert by_metric["decoded_h16_nrmse"]["relative_improvement_fraction"] == (0.56 - 0.08) / 0.56
    assert by_metric["decoded_rollout_nrmse"]["claim_role"] == "primary"
    assert by_metric["decoded_h16_nrmse"]["claim_role"] == "diagnostic"


def test_build_horizon_rows_emits_primary_and_persistence_horizons(tmp_path):
    claim_evidence, _, durable_scorecard = _fixture_payloads()
    claim_evidence["candidate_evidence"][0]["metrics"].update(
        {
            "decoded_step1_nrmse": 0.7,
            "decoded_h4_nrmse": 0.55,
            "decoded_h16_nrmse": 0.08,
        }
    )

    rows = build_horizon_rows(claim_evidence, durable_scorecard, artifact_root=tmp_path)
    values = {(row["series"], row["horizon"]): row["metric_value"] for row in rows}

    assert values[("UPS primary claim", "step1")] == 0.7
    assert values[("UPS primary claim", "h16")] == 0.08
    assert values[("Persistence baseline", "h4")] == 0.55
    assert all(row["metric_name"].endswith("_nrmse") for row in rows)


def test_build_transport_ablation_rows_keeps_validation_variants_separate():
    ablation_matrix = {
        "split": "val",
        "metric_name": "nrmse",
        "held_out_test_used": False,
        "variants": {
            "full_context_shift": {
                "metrics": {"validation_nrmse": 0.001},
                "context_transitions": 1,
                "candidate_shift_min": -80,
                "candidate_shift_max": 80,
                "held_out_test_used": False,
            },
            "weaker_context_shift": {
                "metrics": {"validation_nrmse": 0.4},
                "context_transitions": 1,
                "candidate_shift_min": -8,
                "candidate_shift_max": 8,
                "held_out_test_used": False,
            },
        },
    }

    rows = build_transport_ablation_rows(ablation_matrix)
    by_variant = {row["variant_id"]: row for row in rows}

    assert by_variant["full_context_shift"]["label"] == "Full context shift"
    assert by_variant["full_context_shift"]["metric_value"] == 0.001
    assert by_variant["weaker_context_shift"]["claim_boundary"] == "validation-only diagnostic"
    assert by_variant["full_context_shift"]["held_out_test_used"] is False


def test_build_transfer_rows_marks_skipped_tasks_without_metric():
    transfer_scorecard = {
        "status": "partial_transfer_validated",
        "metric": "nrmse",
        "tasks": {
            "advection1d": {
                "status": "validated",
                "validation_nrmse": 0.002,
                "train_nrmse": 0.001,
                "test_touched": False,
            },
            "darcy2d": {
                "status": "skipped",
                "reason": "missing train split",
            },
        },
    }

    rows = build_transfer_rows(transfer_scorecard)
    by_task = {row["task"]: row for row in rows}

    assert by_task["advection1d"]["metric_value"] == 0.002
    assert by_task["advection1d"]["claim_boundary"] == "train/validation transfer diagnostic"
    assert by_task["darcy2d"]["metric_value"] is None
    assert by_task["darcy2d"]["status"] == "skipped"


def test_build_external_matrix_rows_keeps_future_surfaces_separate():
    _, external_mapping, _ = _fixture_payloads()

    rows = build_external_matrix_rows(external_mapping)
    by_surface = {row["surface"]: row for row in rows}

    assert by_surface["FNO"]["status"] == "measured"
    assert by_surface["FNO"]["claim_boundary"] == "Matched light-v1 repo protocol"
    assert by_surface["Poseidon"]["status"] == "future_or_partial"
    assert "Validation-only transfer path" in by_surface["Poseidon"]["next_step"]


def test_build_reproducibility_card_rows_marks_missing_cost_as_not_recorded(tmp_path):
    claim_evidence, _, durable_scorecard = _fixture_payloads()
    source_path = tmp_path / "claim.json"
    source_path.write_text("{}", encoding="utf-8")

    rows = build_reproducibility_card_rows(
        claim_evidence,
        durable_scorecard,
        source_paths=[source_path],
        generated_output_count=4,
    )
    by_key = {row["key"]: row for row in rows}

    assert by_key["showcase_check"]["value"] == "python scripts/build_showcase_assets.py --check"
    assert by_key["showcase_gpu_required"]["value"] == "no"
    assert by_key["evidence_input_count"]["value"] == "1"
    assert by_key["generated_output_count"]["value"] == "4"
    assert by_key["benchmark_cost_status"]["status"] == "not_recorded"
    assert by_key["primary_metric"]["value"] == "decoded_rollout_nrmse"


def test_build_benchmark_readiness_rows_splits_measured_protocols_and_ecosystem():
    rows = build_benchmark_readiness_rows(
        [
            {
                "surface": "FNO",
                "status": "measured",
                "metric_value": 0.7,
                "next_step": "Keep in table.",
                "claim_boundary": "Matched light-v1 repo protocol",
            },
            {
                "surface": "PDEArena",
                "status": "future_or_partial",
                "metric_value": None,
                "next_step": "Add adapter.",
                "claim_boundary": "External protocol, not directly comparable to light-v1.",
            },
            {
                "surface": "PhysicsNeMo",
                "status": "future_or_partial",
                "metric_value": None,
                "next_step": "Track compatibility.",
                "claim_boundary": "Compatibility surface; no current UPS metric.",
            },
        ]
    )
    by_surface = {row["surface"]: row for row in rows}

    assert by_surface["FNO"]["readiness_lane"] == "matched third-party baseline"
    assert by_surface["FNO"]["readiness"] == "measured"
    assert by_surface["PDEArena"]["readiness_lane"] == "official external protocol"
    assert by_surface["PhysicsNeMo"]["readiness_lane"] == "ecosystem compatibility"


def test_build_rollout_preview_status_rows_excludes_ignored_local_preview():
    rows = build_rollout_preview_status_rows(local_preview_exists=True)
    by_key = {row["key"]: row for row in rows}

    assert by_key["claim_linked_preview_artifact"]["status"] == "missing"
    assert by_key["ignored_local_preview"]["status"] == "excluded"
    assert "not public evidence" in by_key["ignored_local_preview"]["claim_boundary"]


def test_build_rows_are_json_serializable():
    claim_evidence, external_mapping, durable_scorecard = _fixture_payloads()
    external_rows = build_external_matrix_rows(external_mapping)

    json.dumps(
        {
            "benchmark": build_benchmark_rows(claim_evidence, external_mapping, durable_scorecard),
            "tasks": build_task_rows(claim_evidence, external_mapping, durable_scorecard),
            "external": external_rows,
            "metric_suite": build_metric_suite_rows(claim_evidence, durable_scorecard),
            "horizons": build_horizon_rows(claim_evidence, durable_scorecard),
            "transport_ablation": build_transport_ablation_rows({"variants": {}}),
            "transfer": build_transfer_rows({"tasks": {}}),
            "reproducibility": build_reproducibility_card_rows(
                claim_evidence,
                durable_scorecard,
                source_paths=[],
                generated_output_count=0,
            ),
            "benchmark_readiness": build_benchmark_readiness_rows(external_rows),
            "rollout_preview_status": build_rollout_preview_status_rows(),
        }
    )


def test_sha256_file_is_stable_for_repeatability_manifest(tmp_path):
    path = tmp_path / "artifact.txt"
    path.write_text("showcase\n", encoding="utf-8")

    assert sha256_file(path) == sha256_file(path)
    assert sha256_file(path) == "7ed52c567af52b74d1782bf769c055fa23e62037c62f758600ae41a9fc972d60"
