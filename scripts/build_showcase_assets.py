#!/usr/bin/env python
from __future__ import annotations

"""Build public showcase tables and figures from committed UPS evidence."""

import argparse
import csv
import filecmp
import hashlib
import json
import math
import sys
import tarfile
import tempfile
import textwrap
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

DEFAULT_CLAIM_EVIDENCE = Path("docs/claim_evidence/universal_sota_claim_evidence.json")
DEFAULT_EXTERNAL_MAPPING = Path("docs/claim_evidence/external_baseline_mapping.json")
DEFAULT_DURABLE_SCORECARD = Path("docs/claim_evidence/artifacts/light_v1_demo_scorecard.json")
DEFAULT_TRANSPORT_ABLATION = Path(
    "docs/claim_evidence/artifacts/ups_advection_data_conditioned_ablation_matrix.json"
)
DEFAULT_TRANSFER_SCORECARD = Path(
    "docs/claim_evidence/artifacts/inferred_transport_transfer_scorecard.json"
)
DEFAULT_ROLLOUT_PREVIEW_MANIFEST = Path("docs/claim_evidence/rollout_preview_manifest.json")
DEFAULT_OUTPUT_DIR = Path("docs/showcase/generated")

TASK_METRICS = {
    "advection1d": "task_advection1d_decoded_rollout_nrmse",
    "burgers1d": "task_burgers1d_decoded_rollout_nrmse",
    "darcy2d": "task_darcy2d_decoded_rollout_nrmse",
}

METRIC_SUITE = (
    {
        "label": "Rollout NRMSE",
        "metric_name": "decoded_rollout_nrmse",
        "metric_family": "aggregate error",
        "claim_role": "primary",
    },
    {
        "label": "Rollout MAE",
        "metric_name": "decoded_rollout_mae",
        "metric_family": "aggregate error",
        "claim_role": "diagnostic",
    },
    {
        "label": "Rollout MSE",
        "metric_name": "decoded_rollout_mse",
        "metric_family": "aggregate error",
        "claim_role": "diagnostic",
    },
    {
        "label": "Spectral energy error",
        "metric_name": "decoded_rollout_spectral_energy_error",
        "metric_family": "spectral shape",
        "claim_role": "diagnostic",
    },
    {
        "label": "Step-1 NRMSE",
        "metric_name": "decoded_step1_nrmse",
        "metric_family": "horizon profile",
        "claim_role": "diagnostic",
    },
    {
        "label": "H4 NRMSE",
        "metric_name": "decoded_h4_nrmse",
        "metric_family": "horizon profile",
        "claim_role": "diagnostic",
    },
    {
        "label": "H16 NRMSE",
        "metric_name": "decoded_h16_nrmse",
        "metric_family": "horizon profile",
        "claim_role": "diagnostic",
    },
)

HORIZON_METRICS = (
    ("step1", "Step 1", "decoded_step1_nrmse"),
    ("h4", "H4", "decoded_h4_nrmse"),
    ("h16", "H16", "decoded_h16_nrmse"),
)

BENCHMARK_FIELDS = (
    "label",
    "run_name",
    "category",
    "metric_name",
    "metric_value",
    "split",
    "primary_metric_value",
    "primary_improvement_fraction",
    "claim_comparable",
    "published_numbers_directly_comparable",
    "artifact_sha256",
    "artifact_handle",
    "evidence_json",
    "source_refs",
    "notes",
)

TASK_FIELDS = (
    "label",
    "run_name",
    "category",
    "task",
    "metric_name",
    "metric_value",
    "claim_comparable",
)

EXTERNAL_FIELDS = (
    "surface",
    "candidate_id",
    "status",
    "model_family",
    "source_refs",
    "metric_name",
    "metric_value",
    "what_it_proves",
    "next_step",
    "claim_boundary",
)

ECOSYSTEM_COMPATIBILITY_FIELDS = (
    "surface",
    "candidate_id",
    "status",
    "readiness_lane",
    "source_refs",
    "adapter_entrypoint",
    "validation_command",
    "evidence_json",
    "metric_name",
    "metric_value",
    "test_metric_value",
    "next_step",
    "claim_boundary",
)

METRIC_SUITE_FIELDS = (
    "label",
    "metric_name",
    "metric_family",
    "claim_role",
    "ups_value",
    "persistence_value",
    "relative_improvement_fraction",
    "claim_boundary",
)

HORIZON_FIELDS = (
    "series",
    "horizon",
    "horizon_label",
    "metric_name",
    "metric_value",
    "claim_boundary",
)

TRANSPORT_ABLATION_FIELDS = (
    "variant_id",
    "label",
    "split",
    "metric_name",
    "metric_value",
    "context_transitions",
    "candidate_shift_min",
    "candidate_shift_max",
    "held_out_test_used",
    "claim_boundary",
)

TRANSFER_FIELDS = (
    "task",
    "status",
    "metric_name",
    "metric_value",
    "train_metric_value",
    "test_touched",
    "reason",
    "claim_boundary",
)

REPRODUCIBILITY_FIELDS = (
    "key",
    "label",
    "value",
    "status",
    "claim_boundary",
)

BENCHMARK_READINESS_FIELDS = (
    "surface",
    "readiness_lane",
    "readiness",
    "metric_value",
    "next_step",
    "claim_boundary",
)

ROLLOUT_PREVIEW_FIELDS = (
    "key",
    "label",
    "status",
    "next_step",
    "claim_boundary",
)

ROLLOUT_PREVIEW_SUMMARY_FIELDS = (
    "run_name",
    "split",
    "task",
    "metric_name",
    "metric_value",
    "sample_count",
    "frame_count",
    "source_summary_json",
    "artifact_path",
    "artifact_sha256",
    "access_boundary",
    "claim_boundary",
)


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return payload


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_artifact_path(artifact_root: Path, artifact_path: str | Path) -> Path:
    path = Path(artifact_path)
    if path.is_absolute():
        return path
    return artifact_root / path


def write_json(payload: Mapping[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_tsv(rows: Sequence[Mapping[str, Any]], path: str | Path, fields: Sequence[str]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(fields), delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _stringify(row.get(field)) for field in fields})


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return ""
        return f"{value:.16g}"
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True)


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _metric(metrics: Mapping[str, Any], key: str) -> float | None:
    return _as_float(metrics.get(key))


def _task_key_from_scorecard(task: str) -> str:
    return f"metric:{TASK_METRICS[task]}"


def _scorecard_row(
    durable_scorecard: Mapping[str, Any],
    run_name: str,
) -> Mapping[str, Any] | None:
    for row in durable_scorecard.get("rows", []):
        if isinstance(row, Mapping) and row.get("run_name") == run_name:
            return row
    return None


def _primary_candidate(claim_evidence: Mapping[str, Any]) -> Mapping[str, Any] | None:
    candidates = claim_evidence.get("candidate_evidence", [])
    if candidates and isinstance(candidates[0], Mapping):
        return candidates[0]
    return None


def _repo_path_from_handle(handle: Any, artifact_root: Path) -> Path | None:
    raw = str(handle or "")
    if not raw.startswith("repo:"):
        return None
    relative = raw.removeprefix("repo:")
    path = Path(relative)
    if path.is_absolute():
        return path
    return artifact_root / path


def _read_summary_metrics_from_tar(path: Path, run_name: str) -> dict[str, Any]:
    if not path.exists():
        return {}
    preferred_members = [
        f"{run_name}/summary_test.json",
        f"{run_name}/summary.json",
    ]
    with tarfile.open(path, "r:*") as archive:
        members = {member.name: member for member in archive.getmembers() if member.isfile()}
        selected_name = next((name for name in preferred_members if name in members), None)
        if selected_name is None:
            selected_name = next(
                (
                    name
                    for name in sorted(members)
                    if name.endswith("/summary_test.json") or name.endswith("/summary.json")
                ),
                None,
            )
        if selected_name is None:
            return {}
        extracted = archive.extractfile(members[selected_name])
        if extracted is None:
            return {}
        payload = json.load(extracted)
    if not isinstance(payload, Mapping):
        return {}
    metrics = payload.get("metrics", {})
    return dict(metrics) if isinstance(metrics, Mapping) else {}


def _primary_metrics(
    claim_evidence: Mapping[str, Any],
    *,
    artifact_root: Path = Path("."),
) -> dict[str, Any]:
    candidate = _primary_candidate(claim_evidence)
    if candidate is None:
        return {}
    metrics = candidate.get("metrics", {})
    resolved: dict[str, Any] = dict(metrics) if isinstance(metrics, Mapping) else {}
    run_name = str(candidate.get("run_name", ""))
    for handle in candidate.get("artifact_handles", []):
        path = _repo_path_from_handle(handle, artifact_root)
        if path is None:
            continue
        artifact_metrics = _read_summary_metrics_from_tar(path, run_name)
        if artifact_metrics:
            resolved.update(artifact_metrics)
            break
    return resolved


def _persistence_metrics(durable_scorecard: Mapping[str, Any]) -> dict[str, Any]:
    row = _scorecard_row(durable_scorecard, "persistence_light_v1_test")
    if row is None:
        return {}
    metrics: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(key, str) and key.startswith("metric:"):
            metrics[key.removeprefix("metric:")] = value
    return metrics


def _short_baseline_label(value: Any) -> str:
    label = str(value or "Local strong baseline")
    lowered = label.lower()
    if "fourier" in lowered and len(label) > 24:
        return "Fourier baseline"
    return label


def _surface_label(candidate: Mapping[str, Any]) -> str:
    candidate_id = str(candidate.get("candidate_id", ""))
    model_family = str(candidate.get("model_family", candidate_id))
    lowered = candidate_id.lower()
    if "poseidon" in lowered:
        return "Poseidon"
    if "pdeformer" in lowered:
        return "PDEformer-2"
    if lowered.startswith("cfo") or "continuous_time" in lowered:
        return "CFO"
    if "realpdebench" in lowered:
        return "RealPDEBench"
    return model_family


def _variant_label(variant: Mapping[str, Any]) -> str:
    raw = str(variant.get("claim_contract_label", variant.get("variant_id", "")))
    lowered = raw.lower()
    if "ct1" in lowered and len(raw) > 25:
        return "CT1 scoped UPS"
    if ("data-conditioned" in lowered or "data_conditioned" in lowered) and len(raw) > 25:
        return "Data-conditioned scoped UPS"
    return raw


def _primary_metric(claim_evidence: Mapping[str, Any]) -> float:
    docs = claim_evidence.get("claim_documentation", {})
    if isinstance(docs, Mapping):
        value = _as_float(docs.get("metric_value"))
        if value is not None:
            return value
    candidates = claim_evidence.get("candidate_evidence", [])
    if candidates and isinstance(candidates[0], Mapping):
        metrics = candidates[0].get("metrics", {})
        if isinstance(metrics, Mapping):
            value = _metric(metrics, "decoded_rollout_nrmse")
            if value is not None:
                return value
    raise ValueError("Could not resolve primary decoded_rollout_nrmse")


def _primary_improvement_fraction(primary_value: float, metric_value: float | None) -> float | None:
    if metric_value is None or metric_value == 0:
        return None
    return (metric_value - primary_value) / metric_value


def _benchmark_row(
    *,
    label: str,
    run_name: str,
    category: str,
    metric_name: str,
    metric_value: float | None,
    split: str,
    primary_metric_value: float,
    claim_comparable: bool,
    published_numbers_directly_comparable: bool,
    artifact_sha256: str = "",
    artifact_handle: str = "",
    evidence_json: str = "",
    source_refs: Iterable[str] = (),
    notes: str = "",
    sort_key: tuple[int, str] = (99, ""),
) -> dict[str, Any]:
    return {
        "label": label,
        "run_name": run_name,
        "category": category,
        "metric_name": metric_name,
        "metric_value": metric_value,
        "split": split,
        "primary_metric_value": primary_metric_value,
        "primary_improvement_fraction": _primary_improvement_fraction(
            primary_metric_value, metric_value
        ),
        "claim_comparable": claim_comparable,
        "published_numbers_directly_comparable": published_numbers_directly_comparable,
        "artifact_sha256": artifact_sha256,
        "artifact_handle": artifact_handle,
        "evidence_json": evidence_json,
        "source_refs": ",".join(source_refs),
        "notes": notes,
        "_sort_key": sort_key,
    }


def build_benchmark_rows(
    claim_evidence: Mapping[str, Any],
    external_mapping: Mapping[str, Any],
    durable_scorecard: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Build aggregate benchmark rows for README-grade scorecards."""
    primary_value = _primary_metric(claim_evidence)
    rows: list[dict[str, Any]] = []

    persistence = _scorecard_row(durable_scorecard, "persistence_light_v1_test")
    if persistence is not None:
        rows.append(
            _benchmark_row(
                label="Persistence baseline",
                run_name=str(persistence.get("run_name", "")),
                category="baseline",
                metric_name="decoded_rollout_nrmse",
                metric_value=_as_float(persistence.get("metric:decoded_rollout_nrmse")),
                split=str(persistence.get("split", "test") or "test"),
                primary_metric_value=primary_value,
                claim_comparable=True,
                published_numbers_directly_comparable=False,
                notes="Non-learned held-out light-v1 reference.",
                sort_key=(10, "persistence"),
            )
        )

    strong_baseline = claim_evidence.get("strong_baseline_comparison", {})
    if isinstance(strong_baseline, Mapping) and strong_baseline.get("baseline_metric_value"):
        rows.append(
            _benchmark_row(
                label=_short_baseline_label(strong_baseline.get("baseline_family")),
                run_name=str(strong_baseline.get("baseline_run_name", "")),
                category="local neural baseline",
                metric_name=str(strong_baseline.get("metric_name", "decoded_rollout_nrmse")),
                metric_value=_as_float(strong_baseline.get("baseline_metric_value")),
                split=str(strong_baseline.get("split", "test")),
                primary_metric_value=primary_value,
                claim_comparable=True,
                published_numbers_directly_comparable=False,
                artifact_sha256=str(strong_baseline.get("baseline_artifact_sha256", "")),
                artifact_handle=str(strong_baseline.get("baseline_artifact_handles", [""])[0]),
                notes="Repo-native neural baseline measured under the same claim protocol.",
                sort_key=(20, "local"),
            )
        )

    for candidate in external_mapping.get("baseline_candidates", []):
        if not isinstance(candidate, Mapping):
            continue
        measurements = candidate.get("test_measurements", [])
        if not isinstance(measurements, list):
            continue
        for measurement in measurements:
            if not isinstance(measurement, Mapping):
                continue
            if measurement.get("split") != "test":
                continue
            if measurement.get("claim_comparable") is not True:
                continue
            rows.append(
                _benchmark_row(
                    label=str(candidate.get("model_family", candidate.get("candidate_id", ""))),
                    run_name=str(measurement.get("run_name", "")),
                    category="external matched baseline",
                    metric_name=str(measurement.get("metric_name", "decoded_rollout_nrmse")),
                    metric_value=_as_float(measurement.get("metric_value")),
                    split=str(measurement.get("split", "")),
                    primary_metric_value=primary_value,
                    claim_comparable=True,
                    published_numbers_directly_comparable=bool(
                        measurement.get("published_numbers_directly_comparable", False)
                    ),
                    artifact_handle=str(measurement.get("artifact_handle", "")),
                    evidence_json=str(measurement.get("evidence_json", "")),
                    source_refs=candidate.get("source_refs", []),
                    notes="Third-party model measured under the repo light-v1 protocol.",
                    sort_key=(30, str(candidate.get("model_family", ""))),
                )
            )

    candidates = claim_evidence.get("candidate_evidence", [])
    if candidates and isinstance(candidates[0], Mapping):
        candidate = candidates[0]
        metrics = candidate.get("metrics", {})
        rows.append(
            _benchmark_row(
                label="UPS primary claim",
                run_name=str(candidate.get("run_name", "")),
                category="ups primary",
                metric_name="decoded_rollout_nrmse",
                metric_value=(
                    _metric(metrics, "decoded_rollout_nrmse")
                    if isinstance(metrics, Mapping)
                    else primary_value
                ),
                split=str(candidate.get("split", "test")),
                primary_metric_value=primary_value,
                claim_comparable=True,
                published_numbers_directly_comparable=False,
                artifact_sha256=str(candidate.get("artifact_sha256", "")),
                notes="Primary guarded held-out light-v1 UPS claim.",
                sort_key=(40, "ups-primary"),
            )
        )

    for variant in claim_evidence.get("scoped_claim_variants", []):
        if not isinstance(variant, Mapping):
            continue
        rows.append(
            _benchmark_row(
                label=_variant_label(variant),
                run_name=str(variant.get("run_name", "")),
                category="ups scoped variant",
                metric_name=str(variant.get("metric_name", "decoded_rollout_nrmse")),
                metric_value=_as_float(variant.get("metric_value")),
                split=str(variant.get("split", "test")),
                primary_metric_value=primary_value,
                claim_comparable=bool(
                    variant.get("same_exact_inference_contract_as_primary", False)
                ),
                published_numbers_directly_comparable=bool(
                    variant.get("published_numbers_directly_comparable", False)
                ),
                artifact_sha256=str(variant.get("artifact_sha256", "")),
                artifact_handle=str(variant.get("artifact_handles", [""])[0]),
                evidence_json=str(variant.get("evidence_json", "")),
                notes=str(
                    variant.get(
                        "claim_contract_label",
                        "Scoped variant; not the same inference contract as primary.",
                    )
                ),
                sort_key=(50, str(variant.get("variant_id", ""))),
            )
        )

    rows.sort(key=lambda row: row["_sort_key"])
    for row in rows:
        row.pop("_sort_key", None)
    return rows


def _row_task_metrics(
    *,
    label: str,
    run_name: str,
    category: str,
    metrics: Mapping[str, Any],
    claim_comparable: bool,
    scorecard_style: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task, metric_key in TASK_METRICS.items():
        lookup_key = f"metric:{metric_key}" if scorecard_style else metric_key
        value = _as_float(metrics.get(lookup_key))
        if value is None:
            continue
        rows.append(
            {
                "label": label,
                "run_name": run_name,
                "category": category,
                "task": task,
                "metric_name": metric_key,
                "metric_value": value,
                "claim_comparable": claim_comparable,
            }
        )
    return rows


def build_task_rows(
    claim_evidence: Mapping[str, Any],
    external_mapping: Mapping[str, Any],
    durable_scorecard: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Build per-task breakdown rows from available committed metrics."""
    rows: list[dict[str, Any]] = []

    persistence = _scorecard_row(durable_scorecard, "persistence_light_v1_test")
    if persistence is not None:
        rows.extend(
            _row_task_metrics(
                label="Persistence baseline",
                run_name=str(persistence.get("run_name", "")),
                category="baseline",
                metrics=persistence,
                claim_comparable=True,
                scorecard_style=True,
            )
        )

    strong_baseline = claim_evidence.get("strong_baseline_comparison", {})
    if isinstance(strong_baseline, Mapping):
        metrics = strong_baseline.get("baseline_metrics", {})
        if isinstance(metrics, Mapping):
            rows.extend(
                _row_task_metrics(
                    label=_short_baseline_label(strong_baseline.get("baseline_family")),
                    run_name=str(strong_baseline.get("baseline_run_name", "")),
                    category="local neural baseline",
                    metrics=metrics,
                    claim_comparable=True,
                )
            )

    for candidate in claim_evidence.get("candidate_evidence", []):
        if not isinstance(candidate, Mapping):
            continue
        metrics = candidate.get("metrics", {})
        if isinstance(metrics, Mapping):
            rows.extend(
                _row_task_metrics(
                    label="UPS primary claim",
                    run_name=str(candidate.get("run_name", "")),
                    category="ups primary",
                    metrics=metrics,
                    claim_comparable=True,
                )
            )

    for variant in claim_evidence.get("scoped_claim_variants", []):
        if not isinstance(variant, Mapping):
            continue
        metrics = variant.get("metrics", {})
        if isinstance(metrics, Mapping):
            rows.extend(
                _row_task_metrics(
                    label=_variant_label(variant),
                    run_name=str(variant.get("run_name", "")),
                    category="ups scoped variant",
                    metrics=metrics,
                    claim_comparable=False,
                )
            )

    for candidate in external_mapping.get("baseline_candidates", []):
        if not isinstance(candidate, Mapping):
            continue
        for measurement in candidate.get("test_measurements", []):
            if not isinstance(measurement, Mapping):
                continue
            evidence_path = measurement.get("evidence_json")
            if not evidence_path:
                continue
            path = Path(str(evidence_path))
            if not path.exists():
                continue
            evidence = load_json(path)
            task_metrics = evidence.get("task_metrics", {})
            if not isinstance(task_metrics, Mapping):
                continue
            rows.extend(
                _row_task_metrics(
                    label=_surface_label(candidate),
                    run_name=str(measurement.get("run_name", "")),
                    category="external matched baseline",
                    metrics=task_metrics,
                    claim_comparable=measurement.get("claim_comparable") is True,
                )
            )

    order = {
        "Persistence baseline": 0,
        "Fourier baseline": 1,
        "FNO": 2,
        "UNO": 3,
        "U-Net": 4,
        "CNO1d": 5,
        "UPS primary claim": 6,
    }
    rows.sort(key=lambda row: (order.get(str(row["label"]), 20), str(row["task"])))
    return rows


def build_metric_suite_rows(
    claim_evidence: Mapping[str, Any],
    durable_scorecard: Mapping[str, Any],
    *,
    artifact_root: Path = Path("."),
) -> list[dict[str, Any]]:
    """Build secondary metric rows for the primary UPS claim versus persistence."""
    ups_metrics = _primary_metrics(claim_evidence, artifact_root=artifact_root)
    persistence_metrics = _persistence_metrics(durable_scorecard)
    rows: list[dict[str, Any]] = []
    for definition in METRIC_SUITE:
        metric_name = str(definition["metric_name"])
        ups_value = _as_float(ups_metrics.get(metric_name))
        persistence_value = _as_float(persistence_metrics.get(metric_name))
        if ups_value is None or persistence_value is None:
            continue
        relative_improvement = (
            None if persistence_value == 0 else (persistence_value - ups_value) / persistence_value
        )
        rows.append(
            {
                "label": str(definition["label"]),
                "metric_name": metric_name,
                "metric_family": str(definition["metric_family"]),
                "claim_role": str(definition["claim_role"]),
                "ups_value": ups_value,
                "persistence_value": persistence_value,
                "relative_improvement_fraction": relative_improvement,
                "claim_boundary": (
                    "Primary public claim metric"
                    if definition["claim_role"] == "primary"
                    else "Secondary diagnostic; not a standalone headline claim"
                ),
            }
        )
    return rows


def build_horizon_rows(
    claim_evidence: Mapping[str, Any],
    durable_scorecard: Mapping[str, Any],
    *,
    artifact_root: Path = Path("."),
) -> list[dict[str, Any]]:
    """Build step/horizon profile rows for UPS and persistence."""
    ups_metrics = _primary_metrics(claim_evidence, artifact_root=artifact_root)
    persistence_metrics = _persistence_metrics(durable_scorecard)
    rows: list[dict[str, Any]] = []
    for series, metrics, boundary in (
        ("UPS primary claim", ups_metrics, "Primary UPS artifact metrics"),
        ("Persistence baseline", persistence_metrics, "Held-out light-v1 persistence scorecard"),
    ):
        for horizon, horizon_label, metric_name in HORIZON_METRICS:
            value = _as_float(metrics.get(metric_name))
            if value is None:
                continue
            rows.append(
                {
                    "series": series,
                    "horizon": horizon,
                    "horizon_label": horizon_label,
                    "metric_name": metric_name,
                    "metric_value": value,
                    "claim_boundary": boundary,
                }
            )
    return rows


def _transport_ablation_label(variant_id: str) -> str:
    labels = {
        "full_context_shift": "Full context shift",
        "weaker_context_shift": "Bounded context shift",
        "no_data_conditioning": "No data conditioning",
    }
    return labels.get(variant_id, variant_id.replace("_", " ").title())


def build_transport_ablation_rows(ablation_matrix: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build validation-only transport-context ablation rows."""
    variants = ablation_matrix.get("variants", {})
    if not isinstance(variants, Mapping):
        return []
    rows: list[dict[str, Any]] = []
    for variant_id, variant in variants.items():
        if not isinstance(variant, Mapping):
            continue
        metrics = variant.get("metrics", {})
        value = _as_float(metrics.get("validation_nrmse")) if isinstance(metrics, Mapping) else None
        rows.append(
            {
                "variant_id": str(variant_id),
                "label": _transport_ablation_label(str(variant_id)),
                "split": str(ablation_matrix.get("split", "val")),
                "metric_name": str(ablation_matrix.get("metric_name", "nrmse")),
                "metric_value": value,
                "context_transitions": variant.get("context_transitions", ""),
                "candidate_shift_min": _as_float(variant.get("candidate_shift_min")),
                "candidate_shift_max": _as_float(variant.get("candidate_shift_max")),
                "held_out_test_used": bool(
                    variant.get(
                        "held_out_test_used", ablation_matrix.get("held_out_test_used", False)
                    )
                ),
                "claim_boundary": "validation-only diagnostic",
            }
        )
    order = {"full_context_shift": 0, "weaker_context_shift": 1, "no_data_conditioning": 2}
    rows.sort(key=lambda row: (order.get(str(row["variant_id"]), 20), str(row["variant_id"])))
    return rows


def build_transfer_rows(transfer_scorecard: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build train/validation transfer diagnostic rows."""
    tasks = transfer_scorecard.get("tasks", {})
    if not isinstance(tasks, Mapping):
        return []
    rows: list[dict[str, Any]] = []
    for task, task_payload in tasks.items():
        if not isinstance(task_payload, Mapping):
            continue
        rows.append(
            {
                "task": str(task),
                "status": str(task_payload.get("status", "")),
                "metric_name": str(transfer_scorecard.get("metric", "nrmse")),
                "metric_value": _as_float(task_payload.get("validation_nrmse")),
                "train_metric_value": _as_float(task_payload.get("train_nrmse")),
                "test_touched": bool(task_payload.get("test_touched", False)),
                "reason": str(task_payload.get("reason", "")),
                "claim_boundary": "train/validation transfer diagnostic",
            }
        )
    task_order = {"advection1d": 0, "burgers1d": 1, "darcy2d": 2}
    rows.sort(key=lambda row: (task_order.get(str(row["task"]), 20), str(row["task"])))
    return rows


def _card_row(
    *,
    key: str,
    label: str,
    value: Any,
    status: str,
    claim_boundary: str,
) -> dict[str, Any]:
    return {
        "key": key,
        "label": label,
        "value": _stringify(value),
        "status": status,
        "claim_boundary": claim_boundary,
    }


def build_reproducibility_card_rows(
    claim_evidence: Mapping[str, Any],
    durable_scorecard: Mapping[str, Any],
    *,
    source_paths: Sequence[Path],
    generated_output_count: int,
) -> list[dict[str, Any]]:
    """Build a compact public reproducibility/cost card from committed evidence."""
    docs = claim_evidence.get("claim_documentation", {})
    primary_metric = (
        str(docs.get("metric_name", "decoded_rollout_nrmse"))
        if isinstance(docs, Mapping)
        else "decoded_rollout_nrmse"
    )
    artifact_sha256 = str(docs.get("artifact_sha256", "")) if isinstance(docs, Mapping) else ""
    rows_payload = durable_scorecard.get("rows", [])
    scorecard_rows = [row for row in rows_payload if isinstance(row, Mapping)]
    cost_values = [
        value
        for value in (_as_float(row.get("cost_estimated_usd")) for row in scorecard_rows)
        if value is not None
    ]
    duration_values = [
        value
        for value in (_as_float(row.get("duration_sec")) for row in scorecard_rows)
        if value is not None
    ]
    benchmark_cost_value = f"${sum(cost_values):.2f}" if cost_values else "not recorded"
    benchmark_cost_status = "recorded" if cost_values else "not_recorded"
    total_duration = f"{sum(duration_values):.1f}s" if duration_values else "not recorded"
    duration_status = "recorded" if duration_values else "not_recorded"
    return [
        _card_row(
            key="showcase_check",
            label="Showcase check",
            value="python scripts/build_showcase_assets.py --check",
            status="repeatable",
            claim_boundary="Regenerates public showcase assets from committed evidence.",
        ),
        _card_row(
            key="showcase_gpu_required",
            label="GPU for showcase",
            value="no",
            status="zero_gpu",
            claim_boundary="Showcase regeneration does not rerun benchmarks.",
        ),
        _card_row(
            key="showcase_data_required",
            label="Dataset hydration",
            value="no",
            status="zero_data_hydration",
            claim_boundary="Showcase regeneration reads committed evidence files only.",
        ),
        _card_row(
            key="evidence_input_count",
            label="Evidence inputs",
            value=str(len(source_paths)),
            status="tracked",
            claim_boundary="Inputs are listed and hashed in showcase_manifest.json.",
        ),
        _card_row(
            key="generated_output_count",
            label="Generated outputs",
            value=str(generated_output_count),
            status="tracked",
            claim_boundary=(
                "Generated assets are listed and hashed in showcase_manifest.json; "
                "this count also includes the manifest."
            ),
        ),
        _card_row(
            key="primary_metric",
            label="Primary metric",
            value=primary_metric,
            status="claim_metric",
            claim_boundary="Primary held-out claim metric; secondary metrics are diagnostic.",
        ),
        _card_row(
            key="primary_artifact_hash",
            label="Primary artifact hash",
            value=artifact_sha256[:12] if artifact_sha256 else "not recorded",
            status="recorded" if artifact_sha256 else "not_recorded",
            claim_boundary="Full artifact hash remains in claim evidence.",
        ),
        _card_row(
            key="benchmark_cost_status",
            label="Benchmark dollar cost",
            value=benchmark_cost_value,
            status=benchmark_cost_status,
            claim_boundary="Dollar cost is shown only when recorded in committed scorecards.",
        ),
        _card_row(
            key="recorded_eval_duration",
            label="Recorded eval duration",
            value=total_duration,
            status=duration_status,
            claim_boundary="Duration comes from committed scorecard rows, not a fresh run.",
        ),
    ]


def _readiness_lane(surface: str, status: str) -> str:
    if status == "measured":
        return "matched third-party baseline"
    if surface in {"PDEArena", "RealPDEBench"}:
        return "official external protocol"
    if surface == "PhysicsNeMo":
        return "ecosystem compatibility"
    return "future model or recipe surface"


def build_ecosystem_compatibility_rows(external_mapping: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build official-protocol and ecosystem compatibility rows from evidence."""
    rows: list[dict[str, Any]] = []
    for item in external_mapping.get("ecosystem_compatibility", []):
        if not isinstance(item, Mapping):
            continue
        source_refs = item.get("source_refs", [])
        source_refs_text = (
            source_refs
            if isinstance(source_refs, str)
            else ",".join(str(ref) for ref in source_refs)
        )
        rows.append(
            {
                "surface": str(item.get("surface", "")),
                "candidate_id": str(item.get("candidate_id", "")),
                "status": str(item.get("status", "planned")),
                "readiness_lane": str(item.get("readiness_lane", "")),
                "source_refs": source_refs_text,
                "adapter_entrypoint": str(item.get("adapter_entrypoint", "")),
                "validation_command": str(item.get("validation_command", "")),
                "evidence_json": str(item.get("evidence_json", "")),
                "metric_name": str(item.get("metric_name", "")),
                "metric_value": _as_float(item.get("metric_value")),
                "test_metric_value": _as_float(item.get("test_metric_value")),
                "next_step": str(item.get("next_step", "")),
                "claim_boundary": str(item.get("claim_boundary", "")),
            }
        )
    lane_order = {
        "official architecture adapter": 0,
        "validation-only transfer": 1,
        "official external protocol": 2,
        "ecosystem compatibility": 3,
        "future model or recipe surface": 4,
    }
    rows.sort(
        key=lambda row: (
            lane_order.get(str(row["readiness_lane"]), 9),
            str(row["surface"]),
        )
    )
    return rows


def build_benchmark_readiness_rows(
    external_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build public benchmark-readiness card rows from the external matrix."""
    rows: list[dict[str, Any]] = []
    for row in external_rows:
        surface = str(row.get("surface", ""))
        status = str(row.get("status", ""))
        if status == "measured":
            readiness = "measured"
        elif status == "validation":
            readiness = "validation"
        elif status == "smoke_ready":
            readiness = "smoke_ready"
        else:
            readiness = "planned"
        rows.append(
            {
                "surface": surface,
                "readiness_lane": _readiness_lane(surface, status),
                "readiness": readiness,
                "metric_value": _as_float(row.get("metric_value")),
                "next_step": str(row.get("next_step", "")),
                "claim_boundary": str(row.get("claim_boundary", "")),
            }
        )
    lane_order = {
        "matched third-party baseline": 0,
        "official external protocol": 1,
        "ecosystem compatibility": 2,
        "future model or recipe surface": 3,
    }
    rows.sort(key=lambda item: (lane_order.get(str(item["readiness_lane"]), 9), item["surface"]))
    return rows


def load_rollout_preview_manifest(
    preview_manifest_path: Path | None,
    *,
    artifact_root: Path = Path("."),
) -> dict[str, Any] | None:
    """Load and validate an optional claim-linked rollout preview manifest."""
    if preview_manifest_path is None or not preview_manifest_path.exists():
        return None
    manifest = load_json(preview_manifest_path)
    required = {
        "command",
        "run_name",
        "split",
        "metric_name",
        "metric_value",
        "task",
        "sample_count",
        "frame_count",
        "source_summary_json",
        "artifact_path",
        "artifact_sha256",
        "access_boundary",
    }
    missing = sorted(required - manifest.keys())
    if missing:
        raise ValueError(f"{preview_manifest_path} missing required keys: {', '.join(missing)}")

    artifact_path = _resolve_artifact_path(artifact_root, str(manifest["artifact_path"]))
    if not artifact_path.exists():
        raise FileNotFoundError(f"rollout preview artifact not found: {artifact_path}")
    actual_sha = sha256_file(artifact_path)
    expected_sha = str(manifest["artifact_sha256"])
    if actual_sha != expected_sha:
        raise ValueError(
            f"rollout preview artifact hash mismatch for {artifact_path}: "
            f"{actual_sha} != {expected_sha}"
        )

    import numpy as np

    with np.load(artifact_path) as preview:
        files = set(preview.files)
        for key in ("target", "prediction", "time_index"):
            if key not in files:
                raise ValueError(f"{artifact_path} missing required array: {key}")
        target = preview["target"]
        prediction = preview["prediction"]
        time_index = preview["time_index"]
        if target.shape != prediction.shape:
            raise ValueError(
                f"{artifact_path} target/prediction shape mismatch: "
                f"{target.shape} vs {prediction.shape}"
            )
        if target.ndim < 4:
            raise ValueError(
                f"{artifact_path} arrays must follow sample x time x channel x spatial... shape"
            )
        if int(manifest["sample_count"]) != int(target.shape[0]):
            raise ValueError(f"{artifact_path} sample_count does not match target shape")
        if int(manifest["frame_count"]) != int(target.shape[1]):
            raise ValueError(f"{artifact_path} frame_count does not match target shape")
        if len(time_index) != int(target.shape[1]):
            raise ValueError(f"{artifact_path} time_index length does not match frame count")
        if "baseline" in files and preview["baseline"].shape != target.shape:
            raise ValueError(f"{artifact_path} baseline shape does not match target shape")

    return manifest


def build_rollout_preview_summary_rows(
    rollout_preview_manifest: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    if rollout_preview_manifest is None:
        return []
    access_boundary = str(rollout_preview_manifest["access_boundary"])
    return [
        {
            "run_name": str(rollout_preview_manifest["run_name"]),
            "split": str(rollout_preview_manifest["split"]),
            "task": str(rollout_preview_manifest["task"]),
            "metric_name": str(rollout_preview_manifest["metric_name"]),
            "metric_value": _as_float(rollout_preview_manifest["metric_value"]),
            "sample_count": int(rollout_preview_manifest["sample_count"]),
            "frame_count": int(rollout_preview_manifest["frame_count"]),
            "source_summary_json": str(rollout_preview_manifest["source_summary_json"]),
            "artifact_path": str(rollout_preview_manifest["artifact_path"]),
            "artifact_sha256": str(rollout_preview_manifest["artifact_sha256"]),
            "access_boundary": access_boundary,
            "claim_boundary": (
                f"{access_boundary}; qualitative preview only, numeric claims remain source-of-truth"
            ),
        }
    ]


def build_rollout_preview_status_rows(
    *,
    local_preview_exists: bool | None = None,
    preview_manifest_path: Path | None = None,
    artifact_root: Path = Path("."),
) -> list[dict[str, Any]]:
    """Build rollout-preview status rows without treating ignored reports as evidence."""
    if local_preview_exists is None:
        local_preview_exists = False
    ignored_status = "excluded" if local_preview_exists else "absent"
    rollout_preview_manifest = load_rollout_preview_manifest(
        preview_manifest_path,
        artifact_root=artifact_root,
    )
    if rollout_preview_manifest is None:
        artifact_status = "missing"
        artifact_next_step = (
            "Add a compact prediction/target preview artifact with command, split, "
            "metric, and SHA-256 before rendering qualitative rollout panels."
        )
        artifact_boundary = "No qualitative rollout panel is currently claim-linked."
    else:
        artifact_status = "available"
        artifact_next_step = (
            f"Render generated/rollout_preview_panel.png from "
            f"{rollout_preview_manifest['artifact_path']}."
        )
        artifact_boundary = (
            f"{rollout_preview_manifest['access_boundary']}; qualitative preview only."
        )
    return [
        {
            "key": "claim_linked_preview_artifact",
            "label": "Claim-linked preview artifact",
            "status": artifact_status,
            "next_step": artifact_next_step,
            "claim_boundary": artifact_boundary,
        },
        {
            "key": "ignored_local_preview",
            "label": "Ignored local preview",
            "status": ignored_status,
            "next_step": "Do not use ignored reports as public showcase evidence.",
            "claim_boundary": "Ignored local reports are not public evidence.",
        },
        {
            "key": "preview_contract",
            "label": "Preview contract",
            "status": "defined",
            "next_step": "Use docs/showcase/rollout_preview_artifact_contract.md.",
            "claim_boundary": "Contract defines future artifact shape; it is not a result.",
        },
    ]


def build_external_matrix_rows(external_mapping: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build a public matrix of external benchmark surfaces and readiness."""
    rows: list[dict[str, Any]] = []
    seen_candidate_ids: set[str] = set()
    for candidate in external_mapping.get("baseline_candidates", []):
        if not isinstance(candidate, Mapping):
            continue
        candidate_id = str(candidate.get("candidate_id", ""))
        measurements = [
            item
            for item in candidate.get("test_measurements", [])
            if isinstance(item, Mapping) and item.get("claim_comparable") is True
        ]
        measured = bool(measurements)
        metric_value = _as_float(measurements[0].get("metric_value")) if measured else None
        status = "measured" if measured else "future_or_partial"
        model_family = str(candidate.get("model_family", candidate.get("candidate_id", "")))
        surface = _surface_label(candidate)
        why = str(
            candidate.get(
                "why_selected",
                candidate.get("why_not_primary", "Tracked as an external benchmark surface."),
            )
        )
        next_step = (
            "Keep in matched-protocol table; do not mix with published-paper values."
            if measured
            else str(
                candidate.get(
                    "why_not_primary",
                    candidate.get(
                        "required_next_step",
                        "Define adapter and validation gate before held-out test use.",
                    ),
                )
            )
        )
        rows.append(
            {
                "surface": surface,
                "candidate_id": candidate_id,
                "status": status,
                "model_family": model_family,
                "source_refs": ",".join(candidate.get("source_refs", [])),
                "metric_name": str(
                    measurements[0].get("metric_name", "decoded_rollout_nrmse") if measured else ""
                ),
                "metric_value": metric_value,
                "what_it_proves": (
                    "Measured under the same light-v1 split, horizon, and metric."
                    if measured
                    else why
                ),
                "next_step": next_step,
                "claim_boundary": (
                    "Matched light-v1 repo protocol"
                    if measured
                    else "Not a current held-out claim-comparable benchmark"
                ),
            }
        )
        seen_candidate_ids.add(candidate_id)

    for item in build_ecosystem_compatibility_rows(external_mapping):
        candidate_id = str(item["candidate_id"])
        if candidate_id in seen_candidate_ids:
            continue
        measured = (
            str(item["status"]) == "matched_protocol_measured"
            and item.get("test_metric_value") is not None
        )
        metric_value = item.get("test_metric_value") if measured else item.get("metric_value")
        status = "measured" if measured else "future_or_partial"
        if str(item["status"]) == "compatibility_smoke_ready":
            status = "smoke_ready"
        elif str(item["status"]) == "validation_recipe_adapter_complete":
            status = "validation"
        rows.append(
            {
                "surface": str(item["surface"]),
                "candidate_id": candidate_id,
                "status": status,
                "model_family": str(item.get("readiness_lane", "")),
                "source_refs": str(item["source_refs"]),
                "metric_name": str(item["metric_name"] if metric_value is not None else ""),
                "metric_value": metric_value,
                "what_it_proves": str(
                    item.get("claim_boundary")
                    or item.get("next_step")
                    or "Tracked as an ecosystem compatibility surface."
                ),
                "next_step": str(item["next_step"]),
                "claim_boundary": str(item["claim_boundary"]),
            }
        )
    rows.sort(key=lambda row: (row["status"] != "measured", row["surface"]))
    return rows


def _plot_bar(
    rows: Sequence[Mapping[str, Any]],
    *,
    path: Path,
    title: str,
    label_key: str = "label",
    value_key: str = "metric_value",
    color_by_category: bool = True,
) -> None:
    import matplotlib.pyplot as plt

    labels = [str(row[label_key]) for row in rows if _as_float(row.get(value_key)) is not None]
    values = [
        _as_float(row[value_key]) for row in rows if _as_float(row.get(value_key)) is not None
    ]
    if not labels or not values:
        return
    palette = {
        "baseline": "#6b7280",
        "local neural baseline": "#7c3aed",
        "external matched baseline": "#2563eb",
        "ups primary": "#15803d",
        "ups scoped variant": "#0f766e",
    }
    colors = [
        palette.get(str(row.get("category", "")), "#334155") if color_by_category else "#2563eb"
        for row in rows
        if _as_float(row.get(value_key)) is not None
    ]
    width = max(8.0, min(15.0, 1.05 * len(labels)))
    fig, ax = plt.subplots(figsize=(width, 5.0), constrained_layout=True)
    bars = ax.bar(labels, values, color=colors)
    ax.set_title(title)
    ax.set_ylabel("decoded rollout NRMSE (lower is better)")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values, strict=True):
        if value is None:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    _save_figure(fig, path)
    plt.close(fig)


def _save_figure(fig: Any, path: str | Path) -> None:
    fig.savefig(
        path,
        dpi=180,
        metadata={"Software": "universal_simulator showcase generator"},
    )


def render_claim_scorecard(rows: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    visible = [
        row
        for row in rows
        if row.get("category")
        in {"baseline", "local neural baseline", "external matched baseline", "ups primary"}
    ]
    _plot_bar(
        visible,
        path=Path(path),
        title="UPS light-v1 matched-protocol scorecard",
    )


def render_external_benchmarks(rows: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    measured = [row for row in rows if row.get("status") == "measured"]
    _plot_bar(
        measured,
        path=Path(path),
        title="Measured third-party baselines under light-v1",
        label_key="surface",
        color_by_category=False,
    )


def render_task_breakdown(rows: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    import matplotlib.pyplot as plt

    labels = []
    for row in rows:
        label = str(row["label"])
        if label not in labels:
            labels.append(label)
    tasks = list(TASK_METRICS)
    values_by_task = {
        task: {
            str(row["label"]): _as_float(row["metric_value"]) for row in rows if row["task"] == task
        }
        for task in tasks
    }
    if not labels:
        return
    width = max(9.0, min(15.5, 1.15 * len(labels)))
    fig, ax = plt.subplots(figsize=(width, 5.2), constrained_layout=True)
    x = list(range(len(labels)))
    bar_width = 0.24
    colors = {"advection1d": "#dc2626", "burgers1d": "#2563eb", "darcy2d": "#16a34a"}
    offsets = {"advection1d": -bar_width, "burgers1d": 0.0, "darcy2d": bar_width}
    for task in tasks:
        values = [
            (
                values_by_task[task].get(label, float("nan"))
                if values_by_task[task].get(label) is not None
                else float("nan")
            )
            for label in labels
        ]
        ax.bar(
            [item + offsets[task] for item in x],
            values,
            width=bar_width,
            label=task,
            color=colors[task],
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("task decoded rollout NRMSE (lower is better)")
    ax.set_title("Per-task light-v1 breakdown")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    _save_figure(fig, path)
    plt.close(fig)


def render_metric_suite(rows: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    import matplotlib.pyplot as plt

    visible = [
        row for row in rows if _as_float(row.get("relative_improvement_fraction")) is not None
    ]
    if not visible:
        return
    labels = [str(row["label"]) for row in visible]
    values = [100.0 * float(row["relative_improvement_fraction"]) for row in visible]
    colors = ["#15803d" if value >= 0 else "#b91c1c" for value in values]
    fig, ax = plt.subplots(figsize=(10.0, 5.2), constrained_layout=True)
    bars = ax.bar(labels, values, color=colors)
    ax.axhline(0.0, color="#111827", linewidth=0.8)
    ax.set_title("UPS vs persistence across secondary metrics")
    ax.set_ylabel("relative reduction vs persistence, % (higher is better)")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values, strict=True):
        va = "bottom" if value >= 0 else "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.1f}%",
            ha="center",
            va=va,
            fontsize=8,
        )
    _save_figure(fig, path)
    plt.close(fig)


def render_horizon_profile(rows: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    import matplotlib.pyplot as plt

    horizons = [item[0] for item in HORIZON_METRICS]
    horizon_labels = {item[0]: item[1] for item in HORIZON_METRICS}
    series_order = ["Persistence baseline", "UPS primary claim"]
    values = {
        (str(row["series"]), str(row["horizon"])): _as_float(row["metric_value"]) for row in rows
    }
    if not values:
        return
    fig, ax = plt.subplots(figsize=(8.5, 5.0), constrained_layout=True)
    x = list(range(len(horizons)))
    colors = {"Persistence baseline": "#6b7280", "UPS primary claim": "#15803d"}
    offsets = {"Persistence baseline": -0.18, "UPS primary claim": 0.18}
    bar_width = 0.34
    for series in series_order:
        y = [values.get((series, horizon), float("nan")) for horizon in horizons]
        bar_x = [item + offsets[series] for item in x]
        bars = ax.bar(bar_x, y, width=bar_width, label=series, color=colors[series])
        for bar, item_y in zip(bars, y, strict=True):
            if item_y is None or math.isnan(float(item_y)):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                item_y,
                f"{item_y:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_xticks(x)
    ax.set_xticklabels([horizon_labels[horizon] for horizon in horizons])
    ax.set_ylabel("decoded NRMSE (lower is better)")
    ax.set_title("Primary claim horizon profile")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    _save_figure(fig, path)
    plt.close(fig)


def render_transport_ablation(rows: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    import matplotlib.pyplot as plt

    visible = [row for row in rows if _as_float(row.get("metric_value")) is not None]
    if not visible:
        return
    labels = [str(row["label"]) for row in visible]
    values = [float(row["metric_value"]) for row in visible]
    colors = ["#15803d", "#f59e0b", "#b91c1c", "#6b7280"][: len(values)]
    fig, ax = plt.subplots(figsize=(9.0, 5.0), constrained_layout=True)
    bars = ax.bar(labels, values, color=colors)
    ax.set_title("Validation-only transport context ablation")
    ax.set_ylabel("validation NRMSE (lower is better)")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    _save_figure(fig, path)
    plt.close(fig)


def render_transfer_validation(rows: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    import matplotlib.pyplot as plt

    visible = [row for row in rows if _as_float(row.get("metric_value")) is not None]
    if not visible:
        return
    labels = [str(row["task"]) for row in visible]
    values = [float(row["metric_value"]) for row in visible]
    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    bars = ax.bar(labels, values, color="#2563eb")
    ax.set_title("Train/validation inferred transport transfer")
    ax.set_ylabel("validation NRMSE (lower is better)")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    _save_figure(fig, path)
    plt.close(fig)


def _render_text_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    path: str | Path,
    title: str,
    columns: Sequence[tuple[str, str, int]],
) -> None:
    import matplotlib.pyplot as plt

    if not rows:
        return
    height = max(4.2, 1.0 + 0.55 * len(rows))
    fig, ax = plt.subplots(figsize=(12.0, height), constrained_layout=True)
    ax.axis("off")
    ax.text(0.0, 1.02, title, transform=ax.transAxes, fontsize=16, fontweight="bold", va="bottom")
    total_width = sum(width for _, _, width in columns)
    x_positions: list[float] = []
    cursor = 0.0
    for _, _, width in columns:
        x_positions.append(cursor / total_width)
        cursor += width
    for x, (_, heading, _) in zip(x_positions, columns, strict=True):
        ax.text(x, 0.94, heading, transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")
    y = 0.86
    row_gap = 0.76 / max(len(rows), 1)
    for row in rows:
        for x, (field, _, wrap_width) in zip(x_positions, columns, strict=True):
            text = _stringify(row.get(field))
            if field in {"status", "readiness"}:
                text = text.replace("_", "-")
            ax.text(
                x,
                y,
                textwrap.fill(text, width=wrap_width),
                transform=ax.transAxes,
                fontsize=9,
                va="top",
            )
        y -= row_gap
    _save_figure(fig, path)
    plt.close(fig)


def render_reproducibility_card(rows: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    _render_text_rows(
        rows,
        path=path,
        title="Showcase cost and reproducibility card",
        columns=(
            ("label", "Item", 22),
            ("value", "Value", 32),
            ("status", "Status", 18),
            ("claim_boundary", "Boundary", 42),
        ),
    )


def render_benchmark_readiness(rows: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    _render_text_rows(
        rows,
        path=path,
        title="Benchmark and ecosystem readiness",
        columns=(
            ("surface", "Surface", 18),
            ("readiness_lane", "Lane", 26),
            ("readiness", "Readiness", 14),
            ("claim_boundary", "Boundary", 42),
        ),
    )


def render_ecosystem_compatibility(rows: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    display_rows: list[dict[str, Any]] = []
    for row in rows:
        status = str(row.get("status", ""))
        if status == "matched_protocol_measured":
            boundary = "Matched light-v1 adapter; published tables unmapped."
        elif status == "validation_stopped":
            boundary = "Validation-only transfer; no held-out test."
        elif status == "compatibility_smoke_ready":
            boundary = "Recipe smoke ready; no UPS metric yet."
        elif status == "validation_recipe_adapter_complete":
            boundary = "Validation recipe metric; no held-out test."
        elif str(row.get("readiness_lane")) == "official external protocol":
            boundary = "Planned external protocol; not light-v1 comparable."
        elif str(row.get("readiness_lane")) == "ecosystem compatibility":
            boundary = "Planned compatibility gate; no UPS metric yet."
        else:
            boundary = str(row.get("claim_boundary", ""))
        metric_value = _as_float(row.get("metric_value"))
        display_rows.append(
            {
                "surface": str(row.get("surface", "")),
                "status": status,
                "readiness_lane": str(row.get("readiness_lane", "")),
                "metric_value": "" if metric_value is None else f"{metric_value:.4f}",
                "claim_boundary": boundary,
            }
        )
    _render_text_rows(
        display_rows,
        path=path,
        title="Official and ecosystem compatibility",
        columns=(
            ("surface", "Surface", 18),
            ("status", "Status", 18),
            ("readiness_lane", "Lane", 24),
            ("metric_value", "Val metric", 12),
            ("claim_boundary", "Boundary", 32),
        ),
    )


def render_rollout_preview_status(rows: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    _render_text_rows(
        rows,
        path=path,
        title="Qualitative rollout preview status",
        columns=(
            ("label", "Item", 24),
            ("status", "Status", 16),
            ("next_step", "Next step", 42),
            ("claim_boundary", "Boundary", 36),
        ),
    )


def _preview_frame_for_display(array: Any) -> Any:
    import numpy as np

    frame = np.asarray(array[0, -1, 0], dtype=float)
    if frame.ndim == 1:
        return frame.reshape(1, -1)
    if frame.ndim == 2:
        return frame
    return frame.reshape(frame.shape[0], -1)


def render_rollout_preview_panel(
    rollout_preview_manifest: Mapping[str, Any],
    path: str | Path,
    *,
    artifact_root: Path = Path("."),
) -> None:
    """Render a qualitative target/prediction/error panel from a validated preview artifact."""
    import matplotlib.pyplot as plt
    import numpy as np

    artifact_path = _resolve_artifact_path(
        artifact_root,
        str(rollout_preview_manifest["artifact_path"]),
    )
    with np.load(artifact_path) as preview:
        target = preview["target"]
        prediction = preview["prediction"]
        baseline = preview["baseline"] if "baseline" in preview.files else None
        time_index = preview["time_index"]

    target_frame = _preview_frame_for_display(target)
    prediction_frame = _preview_frame_for_display(prediction)
    error_frame = prediction_frame - target_frame
    panels: list[tuple[str, Any, str]] = [
        (f"Target t={time_index[-1]}", target_frame, "viridis"),
        ("UPS prediction", prediction_frame, "viridis"),
        ("Prediction error", error_frame, "coolwarm"),
    ]
    if baseline is not None:
        panels.insert(2, ("Baseline", _preview_frame_for_display(baseline), "viridis"))

    comparable = np.concatenate(
        [np.asarray(panel[1]).reshape(-1) for panel in panels if panel[2] == "viridis"]
    )
    value_min = float(np.nanmin(comparable))
    value_max = float(np.nanmax(comparable))
    error_max = float(max(abs(np.nanmin(error_frame)), abs(np.nanmax(error_frame)), 1e-12))

    width = 3.7 * len(panels)
    fig, axes = plt.subplots(1, len(panels), figsize=(width, 3.6), constrained_layout=True)
    if len(panels) == 1:
        axes = [axes]
    fig.suptitle(
        (
            f"{rollout_preview_manifest['run_name']} | {rollout_preview_manifest['task']} | "
            f"{rollout_preview_manifest['split']} | "
            f"{rollout_preview_manifest['metric_name']}="
            f"{float(rollout_preview_manifest['metric_value']):.4g}"
        ),
        fontsize=12,
        fontweight="bold",
    )
    for ax, (title, data, cmap) in zip(axes, panels, strict=True):
        if cmap == "coolwarm":
            image = ax.imshow(data, aspect="auto", cmap=cmap, vmin=-error_max, vmax=error_max)
        else:
            image = ax.imshow(data, aspect="auto", cmap=cmap, vmin=value_min, vmax=value_max)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("space")
        ax.set_yticks([])
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_repeatability_manifest(
    *,
    input_paths: Sequence[Path],
    output_paths: Sequence[Path],
    output_dir: Path,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "generator": "scripts/build_showcase_assets.py",
        "check_command": "python scripts/build_showcase_assets.py --check",
        "inputs": [
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in input_paths
        ],
        "outputs": [
            {
                "path": str(path.relative_to(output_dir)),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in output_paths
        ],
    }


def build_showcase(
    *,
    claim_evidence_path: Path,
    external_mapping_path: Path,
    durable_scorecard_path: Path,
    transport_ablation_path: Path,
    transfer_scorecard_path: Path,
    rollout_preview_manifest_path: Path | None = DEFAULT_ROLLOUT_PREVIEW_MANIFEST,
    output_dir: Path,
    artifact_root: Path = Path("."),
) -> list[Path]:
    claim_evidence = load_json(claim_evidence_path)
    external_mapping = load_json(external_mapping_path)
    durable_scorecard = load_json(durable_scorecard_path)
    transport_ablation = load_json(transport_ablation_path)
    transfer_scorecard = load_json(transfer_scorecard_path)
    rollout_preview_manifest = load_rollout_preview_manifest(
        rollout_preview_manifest_path,
        artifact_root=artifact_root,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_rows = build_benchmark_rows(claim_evidence, external_mapping, durable_scorecard)
    task_rows = build_task_rows(claim_evidence, external_mapping, durable_scorecard)
    metric_suite_rows = build_metric_suite_rows(
        claim_evidence,
        durable_scorecard,
        artifact_root=artifact_root,
    )
    horizon_rows = build_horizon_rows(
        claim_evidence,
        durable_scorecard,
        artifact_root=artifact_root,
    )
    transport_ablation_rows = build_transport_ablation_rows(transport_ablation)
    transfer_rows = build_transfer_rows(transfer_scorecard)
    external_rows = build_external_matrix_rows(external_mapping)
    ecosystem_compatibility_rows = build_ecosystem_compatibility_rows(external_mapping)
    benchmark_readiness_rows = build_benchmark_readiness_rows(external_rows)

    paths = [
        output_dir / "benchmark_summary.json",
        output_dir / "benchmark_summary.tsv",
        output_dir / "per_task_summary.tsv",
        output_dir / "metric_suite_summary.tsv",
        output_dir / "horizon_summary.tsv",
        output_dir / "transport_ablation_summary.tsv",
        output_dir / "transfer_validation_summary.tsv",
        output_dir / "reproducibility_card.tsv",
        output_dir / "benchmark_readiness_summary.tsv",
        output_dir / "rollout_preview_status.tsv",
        output_dir / "external_benchmark_matrix.tsv",
        output_dir / "claim_scorecard.png",
        output_dir / "per_task_breakdown.png",
        output_dir / "primary_metric_suite.png",
        output_dir / "horizon_profile.png",
        output_dir / "transport_ablation.png",
        output_dir / "transfer_validation.png",
        output_dir / "reproducibility_card.png",
        output_dir / "benchmark_readiness.png",
        output_dir / "rollout_preview_status.png",
        output_dir / "external_benchmarks.png",
        output_dir / "ecosystem_compatibility_summary.tsv",
        output_dir / "ecosystem_compatibility.png",
    ]
    if rollout_preview_manifest is not None:
        paths.extend(
            [
                output_dir / "rollout_preview_summary.tsv",
                output_dir / "rollout_preview_panel.png",
            ]
        )
    source_paths = [
        claim_evidence_path,
        external_mapping_path,
        durable_scorecard_path,
        transport_ablation_path,
        transfer_scorecard_path,
    ]
    if rollout_preview_manifest is not None:
        if rollout_preview_manifest_path is not None:
            source_paths.append(rollout_preview_manifest_path)
        source_paths.append(
            _resolve_artifact_path(artifact_root, str(rollout_preview_manifest["artifact_path"]))
        )
    reproducibility_rows = build_reproducibility_card_rows(
        claim_evidence,
        durable_scorecard,
        source_paths=source_paths,
        generated_output_count=len(paths) + 1,
    )
    rollout_preview_rows = build_rollout_preview_status_rows(
        preview_manifest_path=rollout_preview_manifest_path,
        artifact_root=artifact_root,
    )
    rollout_preview_summary_rows = build_rollout_preview_summary_rows(rollout_preview_manifest)
    write_json(
        {
            "source_files": {
                "claim_evidence": str(claim_evidence_path),
                "external_mapping": str(external_mapping_path),
                "durable_scorecard": str(durable_scorecard_path),
                "transport_ablation": str(transport_ablation_path),
                "transfer_scorecard": str(transfer_scorecard_path),
            },
            "benchmark_rows": benchmark_rows,
            "task_rows": task_rows,
            "metric_suite_rows": metric_suite_rows,
            "horizon_rows": horizon_rows,
            "transport_ablation_rows": transport_ablation_rows,
            "transfer_rows": transfer_rows,
            "reproducibility_rows": reproducibility_rows,
            "benchmark_readiness_rows": benchmark_readiness_rows,
            "rollout_preview_status_rows": rollout_preview_rows,
            "rollout_preview_summary_rows": rollout_preview_summary_rows,
            "external_matrix_rows": external_rows,
            "ecosystem_compatibility_rows": ecosystem_compatibility_rows,
        },
        paths[0],
    )
    write_tsv(benchmark_rows, paths[1], BENCHMARK_FIELDS)
    write_tsv(task_rows, paths[2], TASK_FIELDS)
    write_tsv(metric_suite_rows, paths[3], METRIC_SUITE_FIELDS)
    write_tsv(horizon_rows, paths[4], HORIZON_FIELDS)
    write_tsv(transport_ablation_rows, paths[5], TRANSPORT_ABLATION_FIELDS)
    write_tsv(transfer_rows, paths[6], TRANSFER_FIELDS)
    write_tsv(reproducibility_rows, paths[7], REPRODUCIBILITY_FIELDS)
    write_tsv(benchmark_readiness_rows, paths[8], BENCHMARK_READINESS_FIELDS)
    write_tsv(rollout_preview_rows, paths[9], ROLLOUT_PREVIEW_FIELDS)
    write_tsv(external_rows, paths[10], EXTERNAL_FIELDS)
    write_tsv(ecosystem_compatibility_rows, paths[21], ECOSYSTEM_COMPATIBILITY_FIELDS)
    render_claim_scorecard(benchmark_rows, paths[11])
    render_task_breakdown(task_rows, paths[12])
    render_metric_suite(metric_suite_rows, paths[13])
    render_horizon_profile(horizon_rows, paths[14])
    render_transport_ablation(transport_ablation_rows, paths[15])
    render_transfer_validation(transfer_rows, paths[16])
    render_reproducibility_card(reproducibility_rows, paths[17])
    render_benchmark_readiness(benchmark_readiness_rows, paths[18])
    render_rollout_preview_status(rollout_preview_rows, paths[19])
    render_external_benchmarks(external_rows, paths[20])
    render_ecosystem_compatibility(ecosystem_compatibility_rows, paths[22])
    if rollout_preview_manifest is not None:
        write_tsv(rollout_preview_summary_rows, paths[23], ROLLOUT_PREVIEW_SUMMARY_FIELDS)
        render_rollout_preview_panel(
            rollout_preview_manifest,
            paths[24],
            artifact_root=artifact_root,
        )

    manifest_path = output_dir / "showcase_manifest.json"
    write_json(
        build_repeatability_manifest(
            input_paths=source_paths,
            output_paths=paths,
            output_dir=output_dir,
        ),
        manifest_path,
    )
    return [*paths, manifest_path]


def check_showcase_assets(
    *,
    claim_evidence_path: Path,
    external_mapping_path: Path,
    durable_scorecard_path: Path,
    transport_ablation_path: Path,
    transfer_scorecard_path: Path,
    rollout_preview_manifest_path: Path | None,
    output_dir: Path,
    artifact_root: Path = Path("."),
) -> bool:
    with tempfile.TemporaryDirectory() as tmpdir:
        generated_paths = build_showcase(
            claim_evidence_path=claim_evidence_path,
            external_mapping_path=external_mapping_path,
            durable_scorecard_path=durable_scorecard_path,
            transport_ablation_path=transport_ablation_path,
            transfer_scorecard_path=transfer_scorecard_path,
            rollout_preview_manifest_path=rollout_preview_manifest_path,
            output_dir=Path(tmpdir),
            artifact_root=artifact_root,
        )
        mismatches: list[str] = []
        for generated_path in generated_paths:
            committed_path = output_dir / generated_path.name
            if not committed_path.exists():
                mismatches.append(f"missing {committed_path}")
                continue
            if not filecmp.cmp(generated_path, committed_path, shallow=False):
                mismatches.append(f"stale {committed_path}")
        if mismatches:
            for mismatch in mismatches:
                print(mismatch, file=sys.stderr)
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claim-evidence", type=Path, default=DEFAULT_CLAIM_EVIDENCE)
    parser.add_argument("--external-mapping", type=Path, default=DEFAULT_EXTERNAL_MAPPING)
    parser.add_argument("--durable-scorecard", type=Path, default=DEFAULT_DURABLE_SCORECARD)
    parser.add_argument("--transport-ablation", type=Path, default=DEFAULT_TRANSPORT_ABLATION)
    parser.add_argument("--transfer-scorecard", type=Path, default=DEFAULT_TRANSFER_SCORECARD)
    parser.add_argument(
        "--rollout-preview-manifest",
        type=Path,
        default=DEFAULT_ROLLOUT_PREVIEW_MANIFEST,
        help=(
            "Optional manifest for a compact claim-linked rollout preview artifact. "
            "Missing default path leaves qualitative panels gated."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Regenerate into a temporary directory and fail if committed assets are stale.",
    )
    args = parser.parse_args()

    if args.check:
        if not check_showcase_assets(
            claim_evidence_path=args.claim_evidence,
            external_mapping_path=args.external_mapping,
            durable_scorecard_path=args.durable_scorecard,
            transport_ablation_path=args.transport_ablation,
            transfer_scorecard_path=args.transfer_scorecard,
            rollout_preview_manifest_path=args.rollout_preview_manifest,
            output_dir=args.output_dir,
            artifact_root=Path("."),
        ):
            sys.exit(1)
        print("showcase assets are up to date")
        return

    for path in build_showcase(
        claim_evidence_path=args.claim_evidence,
        external_mapping_path=args.external_mapping,
        durable_scorecard_path=args.durable_scorecard,
        transport_ablation_path=args.transport_ablation,
        transfer_scorecard_path=args.transfer_scorecard,
        rollout_preview_manifest_path=args.rollout_preview_manifest,
        output_dir=args.output_dir,
        artifact_root=Path("."),
    ):
        print(path)


if __name__ == "__main__":
    main()
