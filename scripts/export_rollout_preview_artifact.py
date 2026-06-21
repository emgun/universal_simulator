#!/usr/bin/env python
from __future__ import annotations

"""Export a compact decoded rollout preview artifact and public manifest."""

import argparse
import hashlib
import json
import shlex
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts import evaluate as evaluate_script
from scripts.run_light_experiment import _apply_overrides, _deep_merge, _set_dotpath
from ups.eval.pdebench_runner import evaluate_decoded_operator
from ups.utils.config_loader import load_config_with_includes


def _preferred_checkpoint(checkpoint_dir: Path, names: tuple[str, ...]) -> Path:
    source_dir = (
        checkpoint_dir / "checkpoints"
        if (checkpoint_dir / "checkpoints").is_dir()
        else checkpoint_dir
    )
    for name in names:
        candidate = source_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No checkpoint found in {source_dir} matching {names}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _command_text() -> str:
    return " ".join(shlex.quote(arg) for arg in sys.argv)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--eval-config")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--eval-override", action="append", default=[])
    parser.add_argument("--checkpoint-source", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--task", required=True, help="Task label to write into the manifest.")
    parser.add_argument(
        "--data-task",
        action="append",
        default=[],
        help="Evaluation task vocabulary. Defaults to --task; repeat to preserve multitask conditioning.",
    )
    parser.add_argument(
        "--skip-missing-tasks",
        action="store_true",
        help="Skip task-vocabulary shards that are not present in --data-root.",
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--rollout-steps", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--metric-name", default="decoded_rollout_nrmse")
    parser.add_argument(
        "--access-boundary",
        default="validation-only; no held-out test access",
        help="Evidence boundary written to the public manifest.",
    )
    parser.add_argument(
        "--output-artifact",
        default="docs/claim_evidence/artifacts/rollout_preview_ups_light_shared_context_transport_guarded_advection1d_val.npz",
    )
    parser.add_argument(
        "--output-summary-json",
        default="docs/claim_evidence/artifacts/rollout_preview_ups_light_shared_context_transport_guarded_advection1d_val_summary.json",
    )
    parser.add_argument(
        "--output-manifest",
        default="docs/claim_evidence/rollout_preview_manifest.json",
    )
    args = parser.parse_args()

    train_cfg = _apply_overrides(load_config_with_includes(args.config), args.override)
    eval_source = load_config_with_includes(args.eval_config) if args.eval_config else {}
    cfg = _deep_merge(train_cfg, _apply_overrides(eval_source, args.eval_override))
    _set_dotpath(cfg, "data.root", args.data_root)
    data_tasks = args.data_task or [args.task]
    _set_dotpath(cfg, "data.task", data_tasks if len(data_tasks) > 1 else data_tasks[0])
    _set_dotpath(cfg, "data.split", args.split)
    _set_dotpath(cfg, "data.max_samples", args.max_samples)
    if args.skip_missing_tasks:
        _set_dotpath(cfg, "evaluation.skip_missing_tasks", True)

    checkpoint_source = Path(args.checkpoint_source)
    operator_ckpt = _preferred_checkpoint(
        checkpoint_source, ("operator_joint.pt", "operator_decoded.pt", "operator.pt")
    )
    encoder_ckpt = _preferred_checkpoint(checkpoint_source, ("encoder_joint.pt", "encoder.pt"))
    decoder_ckpt = _preferred_checkpoint(checkpoint_source, ("decoder_joint.pt", "decoder.pt"))

    operator = evaluate_script.make_operator(cfg)
    encoder = evaluate_script.make_encoder(cfg)
    decoder = evaluate_script.make_decoder(cfg)
    evaluate_script._load_state_dict_compat(operator, str(operator_ckpt), prefix_to_strip="")
    evaluate_script._load_state_dict_compat(encoder, str(encoder_ckpt), prefix_to_strip="")
    evaluate_script._load_state_dict_compat(decoder, str(decoder_ckpt), prefix_to_strip="")

    report = evaluate_decoded_operator(
        cfg,
        encoder,
        operator,
        decoder,
        device=args.device,
        rollout_steps=args.rollout_steps,
        preview_sample_count=args.max_samples,
    )
    preview_records = [
        record
        for record in list((report.extra or {}).pop("rollout_preview", []))
        if record.get("task") == args.task
    ]
    if not preview_records:
        raise RuntimeError(
            f"Decoded evaluation produced no rollout preview records for task {args.task!r}"
        )

    first_time_index = preview_records[0]["time_index"].detach().cpu().numpy()
    targets = np.stack(
        [record["target"].detach().cpu().numpy() for record in preview_records],
        axis=0,
    ).astype(np.float32)
    predictions = np.stack(
        [record["prediction"].detach().cpu().numpy() for record in preview_records],
        axis=0,
    ).astype(np.float32)
    if targets.shape != predictions.shape:
        raise RuntimeError(
            f"Preview target/prediction shape mismatch: {targets.shape} vs {predictions.shape}"
        )
    if not all(
        np.array_equal(first_time_index, record["time_index"].detach().cpu().numpy())
        for record in preview_records
    ):
        raise RuntimeError("Preview records have inconsistent time indices")

    artifact_path = Path(args.output_artifact)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        artifact_path,
        target=targets,
        prediction=predictions,
        time_index=first_time_index.astype(np.float32),
    )
    artifact_sha = _sha256_file(artifact_path)

    summary_path = Path(args.output_summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    metric_value = float(report.metrics[args.metric_name])
    summary = {
        "run_name": args.run_name,
        "task": args.task,
        "split": args.split,
        "metrics": report.metrics,
        "extra": _json_safe(report.extra or {}),
        "preview": {
            "artifact_path": _relative_path(artifact_path),
            "artifact_sha256": artifact_sha,
            "sample_count": int(targets.shape[0]),
            "frame_count": int(targets.shape[1]),
            "shape": list(targets.shape),
        },
        "checkpoints": {
            "operator": str(operator_ckpt),
            "encoder": str(encoder_ckpt),
            "decoder": str(decoder_ckpt),
        },
        "command": _command_text(),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    manifest_path = Path(args.output_manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "command": _command_text(),
        "run_name": args.run_name,
        "split": args.split,
        "metric_name": args.metric_name,
        "metric_value": metric_value,
        "task": args.task,
        "sample_count": int(targets.shape[0]),
        "frame_count": int(targets.shape[1]),
        "source_summary_json": _relative_path(summary_path),
        "artifact_path": _relative_path(artifact_path),
        "artifact_sha256": artifact_sha,
        "access_boundary": args.access_boundary,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "artifact": _relative_path(artifact_path),
                "manifest": _relative_path(manifest_path),
                "summary": _relative_path(summary_path),
                "shape": list(targets.shape),
                args.metric_name: metric_value,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
