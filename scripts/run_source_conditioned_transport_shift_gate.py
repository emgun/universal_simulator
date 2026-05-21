#!/usr/bin/env python
from __future__ import annotations

"""Run a train-only source-conditioned transport-shift validation gate.

The rule fits one periodic shift per source_file_index using train rows only,
then validates the locked mapping on validation rows. This is intended for
official Advection shards built from multiple PDEBench beta files.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.calibrate_residual_gate import _relative_improvement, _test_guard_result
from scripts.fit_transport_shift_head import _candidate_scores, _select_best


def _max_samples(value: int | None) -> int | None:
    return None if value is not None and value < 0 else value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe_attr(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _split_source_record(data_root: str | Path, task: str, split: str) -> dict[str, Any]:
    path = Path(data_root) / f"{task}_{split}.h5"
    if not path.exists():
        return {"split": split, "path": str(path), "exists": False}
    with h5py.File(path, "r") as handle:
        source_paths = _json_safe_attr(handle.attrs.get("source_paths", []))
        datasets = {
            key: {"shape": [int(dim) for dim in value.shape], "dtype": str(value.dtype)}
            for key, value in handle.items()
            if isinstance(value, h5py.Dataset)
        }
    return {
        "split": split,
        "path": str(path),
        "exists": True,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "source_paths": source_paths,
        "datasets": datasets,
    }


def _load_series_and_source(
    *,
    root: str | Path,
    task: str,
    split: str,
    max_samples: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    path = Path(root) / f"{task}_{split}.h5"
    if not path.exists():
        raise FileNotFoundError(path)
    sample_slice = slice(0, _max_samples(max_samples)) if _max_samples(max_samples) is not None else slice(None)
    with h5py.File(path, "r") as handle:
        if "source_file_index" not in handle:
            raise KeyError(f"{path} does not contain source_file_index provenance")
        data = np.asarray(handle["data"][sample_slice], dtype=np.float32)
        source_file_index = np.asarray(handle["source_file_index"][sample_slice], dtype=np.int64)
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"Expected 1D task data shaped (samples, steps, width[, 1]), got {tuple(data.shape)}")
    if data.shape[0] != source_file_index.shape[0]:
        raise ValueError(f"{path} data/source_file_index sample counts differ")
    return data, source_file_index


def _periodic_shift(fields: np.ndarray, shift: float) -> np.ndarray:
    if abs(float(shift) - round(float(shift))) < 1e-9:
        return np.roll(fields, shift=int(round(float(shift))), axis=-1)
    width = int(fields.shape[-1])
    frequencies = np.fft.fftfreq(width)
    phase = np.exp(-2j * np.pi * frequencies * float(shift))
    shifted = np.fft.ifft(np.fft.fft(fields, axis=-1) * phase, axis=-1)
    return np.real(shifted).astype(fields.dtype, copy=False)


def _candidate_scores_periodic(
    fields: np.ndarray,
    shifts: Sequence[float],
    *,
    rollout_steps: int,
) -> list[dict[str, float]]:
    steps = min(int(rollout_steps), fields.shape[1] - 1)
    if steps <= 0:
        raise ValueError("Need at least two trajectory frames to score a transport shift")
    previous = fields[:, :steps]
    current = fields[:, 1 : steps + 1]
    rows = []
    for shift in shifts:
        shifted = _periodic_shift(previous, float(shift))
        squared_error = np.square(shifted - current)
        mse = float(np.mean(squared_error))
        nrmse = float(np.sqrt(np.mean(squared_error)) / max(float(np.std(current)), 1e-12))
        rows.append({"shift": float(shift), "mse": mse, "nrmse": nrmse})
    return rows


def _score_locked_shifts(
    fields: np.ndarray,
    source_file_index: np.ndarray,
    source_shift_map: Mapping[int, float],
    *,
    rollout_steps: int,
) -> dict[str, float]:
    steps = min(int(rollout_steps), fields.shape[1] - 1)
    if steps <= 0:
        raise ValueError("Need at least two trajectory frames to validate a transport shift")
    previous = fields[:, :steps]
    current = fields[:, 1 : steps + 1]
    shifted = np.empty_like(previous)
    for source_index in sorted(set(int(value) for value in source_file_index.tolist())):
        mask = source_file_index == source_index
        shifted[mask] = _periodic_shift(previous[mask], float(source_shift_map[source_index]))
    squared_error = np.square(shifted - current)
    mse = float(np.mean(squared_error))
    nrmse = float(np.sqrt(np.mean(squared_error)) / max(float(np.std(current)), 1e-12))
    return {"mse": mse, "nrmse": nrmse}


def _refined_shifts(base_shift: int, radius: int, *, width: int) -> list[int]:
    if radius <= 0:
        return [int(base_shift)]
    candidates = list(range(int(base_shift) - int(radius), int(base_shift) + int(radius) + 1))
    # Periodic shifts that differ by the domain width are equivalent; normalize
    # only for de-duplication while preserving signed shifts near the coarse mode.
    seen: set[int] = set()
    refined: list[int] = []
    for shift in candidates:
        key = int(shift) % int(width)
        if key in seen:
            continue
        seen.add(key)
        refined.append(int(shift))
    return refined


def _fractional_refined_shifts(base_shift: float, radius: int, step: float, *, width: int) -> list[float]:
    if radius <= 0 or step <= 0:
        return [float(base_shift)]
    count = int(round((2.0 * float(radius)) / float(step)))
    start = float(base_shift) - float(radius)
    candidates = [start + idx * float(step) for idx in range(count + 1)]
    seen: set[int] = set()
    refined: list[float] = []
    for shift in candidates:
        key = int(round((float(shift) % float(width)) / float(step)))
        if key in seen:
            continue
        seen.add(key)
        refined.append(float(shift))
    return refined


def _fit_source_shift_map(
    fields: np.ndarray,
    source_file_index: np.ndarray,
    shifts: Sequence[int],
    *,
    rollout_steps: int,
    metric: str,
    fit_strategy: str,
    refine_radius: int,
    fractional_refine_step: float,
) -> tuple[dict[int, float], dict[str, Any]]:
    selected: dict[int, float] = {}
    groups: dict[str, Any] = {}
    for source_index in sorted(set(int(value) for value in source_file_index.tolist())):
        mask = source_file_index == source_index
        group_fields = fields[mask]
        rows = _candidate_scores(group_fields, shifts, rollout_steps=rollout_steps)
        best = _select_best(rows, metric)
        selected_shift = float(best["shift"])
        refined_rows: list[dict[str, float | int]] = []
        refined_best: dict[str, Any] | None = None
        sample_votes: list[dict[str, Any]] = []
        if fit_strategy == "sample_mode":
            vote_scores: dict[float, list[float]] = {}
            for sample_idx in range(group_fields.shape[0]):
                sample_rows = _candidate_scores(group_fields[sample_idx : sample_idx + 1], shifts, rollout_steps=rollout_steps)
                sample_best = _select_best(sample_rows, metric)
                shift = float(sample_best["shift"])
                vote_scores.setdefault(shift, []).append(float(sample_best[metric]))
                sample_votes.append(
                    {
                        "sample_index": int(sample_idx),
                        "selected_shift": shift,
                        f"selected_{metric}": float(sample_best[metric]),
                    }
                )
            selected_shift = sorted(
                vote_scores,
                key=lambda shift: (-len(vote_scores[shift]), float(np.mean(vote_scores[shift])), abs(shift), shift),
            )[0]
            if refine_radius > 0:
                refined_vote_scores: dict[float, list[float]] = {}
                refined_votes: list[dict[str, Any]] = []
                for vote in sample_votes:
                    sample_idx = int(vote["sample_index"])
                    local_shifts = (
                        _fractional_refined_shifts(
                            float(vote["selected_shift"]),
                            refine_radius,
                            fractional_refine_step,
                            width=int(group_fields.shape[-1]),
                        )
                        if fractional_refine_step > 0
                        else _refined_shifts(
                            int(round(float(vote["selected_shift"]))),
                            refine_radius,
                            width=int(group_fields.shape[-1]),
                        )
                    )
                    sample_rows = _candidate_scores_periodic(
                        group_fields[sample_idx : sample_idx + 1],
                        local_shifts,
                        rollout_steps=rollout_steps,
                    )
                    sample_best = _select_best(sample_rows, metric)
                    shift = float(sample_best["shift"])
                    refined_vote_scores.setdefault(shift, []).append(float(sample_best[metric]))
                    refined_votes.append(
                        {
                            "sample_index": sample_idx,
                            "selected_shift": shift,
                            f"selected_{metric}": float(sample_best[metric]),
                            "candidate_shifts": local_shifts,
                        }
                    )
                selected_shift = float(
                    sorted(
                        refined_vote_scores,
                        key=lambda shift: (
                            -len(refined_vote_scores[shift]),
                            float(np.mean(refined_vote_scores[shift])),
                            abs(shift),
                            shift,
                        ),
                    )[0]
                )
                sample_votes = refined_votes
        elif fit_strategy != "aggregate":
            raise ValueError(f"Unsupported fit strategy: {fit_strategy}")
        if refine_radius > 0:
            local_shifts = (
                _fractional_refined_shifts(
                    selected_shift,
                    refine_radius,
                    fractional_refine_step,
                    width=int(group_fields.shape[-1]),
                )
                if fractional_refine_step > 0
                else _refined_shifts(int(round(selected_shift)), refine_radius, width=int(group_fields.shape[-1]))
            )
            refined_rows = _candidate_scores_periodic(group_fields, local_shifts, rollout_steps=rollout_steps)
            refined_best = _select_best(refined_rows, metric)
            selected_shift = float(refined_best["shift"])
        selected[source_index] = float(selected_shift)
        groups[str(source_index)] = {
            "sample_count": int(mask.sum()),
            "fit_strategy": fit_strategy,
            "selected_shift": float(selected_shift),
            "selected_train": best,
            "candidate_scores": rows,
            "refine_radius": int(refine_radius),
            "fractional_refine_step": float(fractional_refine_step),
            "refined_selected_train": refined_best,
            "refined_candidate_scores": refined_rows,
            "sample_votes": sample_votes,
        }
    return selected, groups


def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    shifts = list(args.shift if args.shift is not None else range(-96, 97, 8))
    train_fields, train_source = _load_series_and_source(
        root=args.data_root,
        task=args.task,
        split=args.train_split,
        max_samples=args.max_samples,
    )
    val_fields, val_source = _load_series_and_source(
        root=args.data_root,
        task=args.task,
        split=args.val_split,
        max_samples=args.val_max_samples if args.val_max_samples is not None else args.max_samples,
    )
    source_shift_map, train_groups = _fit_source_shift_map(
        train_fields,
        train_source,
        shifts,
        rollout_steps=args.rollout_steps,
        metric=args.metric,
        fit_strategy=args.fit_strategy,
        refine_radius=args.refine_radius,
        fractional_refine_step=args.fractional_refine_step,
    )
    train_sources = sorted(source_shift_map)
    val_sources = sorted(set(int(value) for value in val_source.tolist()))
    unsupported_val_sources = sorted(set(val_sources) - set(train_sources))

    blockers: list[str] = []
    selected_validation: dict[str, float] | None = None
    oracle_validation: dict[str, Any] | None = None
    validation_guard = {"passed": False, "blockers": ["validation contains unsupported source_file_index"]}
    if unsupported_val_sources:
        blockers.append(f"validation source_file_index values absent from train: {unsupported_val_sources}")
    else:
        selected_validation = _score_locked_shifts(
            val_fields,
            val_source,
            source_shift_map,
            rollout_steps=args.rollout_steps,
        )
        val_rows = _candidate_scores(val_fields, shifts, rollout_steps=args.rollout_steps)
        oracle_validation = _select_best(val_rows, args.metric)
        validation_guard = _test_guard_result(
            value=float(selected_validation[args.metric]),
            reference=args.reference_metric_value,
            min_relative_improvement=args.val_min_relative_improvement,
            mode="min",
        )
        if not bool(validation_guard["passed"]):
            blockers.append("source-conditioned train-fitted validation metric did not pass the configured SOTA guard")

    test_eligible = bool(selected_validation and validation_guard["passed"] and not blockers)
    test_record = None
    if test_eligible and args.test_split:
        test_fields, test_source = _load_series_and_source(
            root=args.data_root,
            task=args.task,
            split=args.test_split,
            max_samples=args.test_max_samples if args.test_max_samples is not None else args.max_samples,
        )
        unsupported_test_sources = sorted(set(int(value) for value in test_source.tolist()) - set(train_sources))
        if unsupported_test_sources:
            blockers.append(f"test source_file_index values absent from train: {unsupported_test_sources}")
            test_eligible = False
        else:
            test_record = {
                "split": args.test_split,
                "selected_test": _score_locked_shifts(
                    test_fields,
                    test_source,
                    source_shift_map,
                    rollout_steps=args.rollout_steps,
                ),
                "source_shift_map": {str(key): float(value) for key, value in source_shift_map.items()},
                "source_shift_map_float": {str(key): float(value) for key, value in source_shift_map.items()},
            }

    source_splits = [args.train_split, args.val_split]
    if args.test_split:
        source_splits.append(args.test_split)
    return {
        "task": args.task,
        "data_root": args.data_root,
        "data_sources": {
            split: _split_source_record(args.data_root, args.task, split)
            for split in dict.fromkeys(source_splits)
        },
        "train_split": args.train_split,
        "val_split": args.val_split,
        "metric": args.metric,
        "reference_metric_value": args.reference_metric_value,
        "val_min_relative_improvement": args.val_min_relative_improvement,
        "fit": {
            "model": "source_conditioned_periodic_shift",
            "fit_strategy": args.fit_strategy,
            "refine_radius": args.refine_radius,
            "fractional_refine_step": args.fractional_refine_step,
            "fit_uses": "train_split source_file_index groups only",
            "candidate_shifts": shifts,
            "source_shift_map": {str(key): float(value) for key, value in source_shift_map.items()},
            "train_groups": train_groups,
            "selected_train_shift": {str(key): float(value) for key, value in source_shift_map.items()},
            "selected_validation": selected_validation,
            "oracle_validation": oracle_validation,
            "validation_gap_to_oracle": (
                _relative_improvement(
                    float(selected_validation[args.metric]),
                    float(oracle_validation[args.metric]),
                    mode="min",
                )
                if selected_validation and oracle_validation
                else None
            ),
            "validation_guard": validation_guard,
        },
        "diagnostic": {
            "train_source_file_indices": train_sources,
            "val_source_file_indices": val_sources,
            "unsupported_val_source_file_indices": unsupported_val_sources,
            "best_shifts": {
                args.train_split: {str(key): float(value) for key, value in source_shift_map.items()},
                args.val_split: {str(key): float(source_shift_map[key]) for key in val_sources if key in source_shift_map},
            },
            "consistent_best_shift": not unsupported_val_sources,
        },
        "test_eligible": test_eligible,
        "test": test_record,
        "blockers": blockers,
        "next_action": (
            "held-out test measured" if test_record else "run exactly one held-out test with locked source_shift_map"
            if test_eligible and not test_record
            else "do not run held-out test; hydrate/train source support or improve train-fitted conditional rule first"
        ),
        "notes": [
            "Fits only on train_split rows and source_file_index provenance.",
            "Validation uses the locked train-fitted source_shift_map.",
            "Held-out test is optional and only evaluated after validation guard passes.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run source-conditioned train/val transport-shift gate")
    parser.add_argument("--data-root", default="data/pdebench")
    parser.add_argument("--task", default="advection1d")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--test-split", default="")
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--val-max-samples", type=int)
    parser.add_argument("--test-max-samples", type=int)
    parser.add_argument("--rollout-steps", type=int, default=16)
    parser.add_argument("--shift", action="append", type=int, default=None)
    parser.add_argument("--metric", choices=("mse", "nrmse"), default="nrmse")
    parser.add_argument("--fit-strategy", choices=("aggregate", "sample_mode"), default="aggregate")
    parser.add_argument("--refine-radius", type=int, default=0)
    parser.add_argument("--fractional-refine-step", type=float, default=0.0)
    parser.add_argument("--reference-metric-value", type=float, required=True)
    parser.add_argument("--val-min-relative-improvement", type=float, default=0.0)
    parser.add_argument("--output-json", default="reports/research/sota_loop/source_conditioned_transport_shift_gate.json")
    args = parser.parse_args()

    record = run_gate(args)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
