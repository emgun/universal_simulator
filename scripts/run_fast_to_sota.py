#!/usr/bin/env python3
"""Automated fast-to-SOTA iteration pipeline.

This script orchestrates the workflow described in
`docs/fast_to_sota_playbook.md`:

1. Validate configs and data.
2. Optionally run a cost dry-run.
3. Launch training for the supplied config.
4. Run small-eval and full-eval stages (with leaderboard updates).
5. Apply promotion gates and, when successful, mark the run as champion.

It complements the existing CLI utilities by wiring them together with
reproducible run IDs, gating logic, and artifact collection.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shlex
import shutil
import sys
import subprocess
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import yaml

try:  # Optional dependency
    import wandb  # type: ignore
except Exception:  # pragma: no cover - wandb not installed
    wandb = None

# Ensure repo root + src are importable before local imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ups.utils.config_loader import load_config_with_includes
from ups.utils.leaderboard import update_leaderboard

PYTHON = sys.executable


def _apply_overrides(cfg: Dict[str, object], overrides: List[str]) -> Dict[str, object]:
    """Apply dotted key overrides (key=value) to a config dictionary."""
    if not overrides:
        return cfg
    updated = copy.deepcopy(cfg)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override '{override}' must be formatted as key=value")
        path, raw_value = override.split("=", 1)
        keys = [key.strip() for key in path.split(".") if key.strip()]
        if not keys:
            raise ValueError(f"Invalid override path in '{override}'")
        value = yaml.safe_load(raw_value)
        target = updated
        for key in keys[:-1]:
            current = target.get(key)
            if not isinstance(current, dict):
                current = {}
                target[key] = current
            target = current
        target[keys[-1]] = value
    return updated


def _extract_arch_fingerprint(cfg: Dict[str, Any]) -> Dict[str, Any]:
    latent = cfg.get("latent", {}) if isinstance(cfg.get("latent"), dict) else {}
    operator_pdet = (
        cfg.get("operator", {}).get("pdet", {})
        if isinstance(cfg.get("operator"), dict)
        else {}
    )
    diffusion_cfg = cfg.get("diffusion", {}) if isinstance(cfg.get("diffusion"), dict) else {}
    return {
        "latent_dim": latent.get("dim"),
        "latent_tokens": latent.get("tokens"),
        "operator_hidden_dim": operator_pdet.get("hidden_dim"),
        "operator_num_heads": operator_pdet.get("num_heads"),
        "operator_depths": operator_pdet.get("depths"),
        "diffusion_hidden_dim": diffusion_cfg.get("hidden_dim"),
    }


def _write_checkpoint_metadata(
    cfg: Dict[str, Any],
    resolved_config: Path,
    checkpoint_dir: Path,
) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = checkpoint_dir / "metadata.json"
    config_hash = hashlib.sha256(resolved_config.read_bytes()).hexdigest()
    metadata = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_hash": config_hash,
        "config_path": str(resolved_config),
        "arch": _extract_arch_fingerprint(cfg),
        "trained": False,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata_path


def _update_checkpoint_metadata(metadata_path: Path, **fields: object) -> None:
    if not metadata_path.exists():
        return
    try:
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        existing = {}
    existing.update(fields)
    metadata_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")


def _persist_wandb_info(destination: Path, info: Mapping[str, Any]) -> None:
    """Write W&B metadata so evaluation stages can resume the training run."""
    if not info or not isinstance(info, Mapping):
        return
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(dict(info), indent=2), encoding="utf-8")
    except Exception:
        pass


def _write_config(cfg: Dict[str, object], destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return destination


def _log_metrics_to_wandb(run, namespace: str, metrics: Mapping[str, float]) -> None:  # pragma: no cover - optional
    if run is None or wandb is None:
        return
    scalar_payload: Dict[str, float] = {}
    table_rows: List[List[object]] = []
    for key, value in metrics.items():
        if not isinstance(value, (int, float)):
            continue
        sanitized = key.replace(":", "/")
        scalar_payload[f"{namespace}/{sanitized}"] = float(value)
        table_rows.append([key, float(value)])
    if not scalar_payload:
        return
    run.log(scalar_payload, commit=False)
    if table_rows:
        table = wandb.Table(columns=["metric", "value"], data=table_rows)
        run.log({f"{namespace}/metrics": table})


def _wandb_log_event(run, event: str, value: float = 1.0) -> None:  # pragma: no cover - optional
    if run is not None:
        run.log({f"fast_to_sota/{event}": value})


# Legacy function removed - evaluation subprocess logs metrics directly to training run
# via WANDB_CONTEXT_FILE mechanism (see WandBContext in src/ups/utils/wandb_context.py)


def _run_command(
    cmd: List[str],
    *,
    env: Optional[Mapping[str, str]] = None,
    desc: Optional[str] = None,
) -> None:
    """Execute a command, echoing it for transparency."""
    printable = shlex.join(cmd)
    header = f"[{desc}]" if desc else "[run]"
    print(f"{header} {printable}")
    base_env = os.environ.copy()
    if env:
        base_env.update(env)
    subprocess.run(cmd, cwd=ROOT, env=base_env, check=True)


def _load_metrics(path: Path) -> Dict[str, float]:
    """Flatten metrics/extra entries from an evaluate.py JSON payload."""
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics") or {}
    extra = payload.get("extra") or {}
    flattened: Dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            flattened[f"metric:{key}"] = float(value)
    for key, value in extra.items():
        if isinstance(value, (int, float)):
            flattened[f"extra:{key}"] = float(value)
    return flattened


def _read_leaderboard(csv_path: Path) -> List[Dict[str, str]]:
    if not csv_path.exists():
        return []
    rows: List[Dict[str, str]] = []
    import csv

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(dict(row))
    return rows


def _as_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _select_baseline(
    rows: List[Dict[str, str]],
    labels: List[str],
    metric_key: str,
) -> Tuple[Optional[Dict[str, str]], Dict[str, float]]:
    """Pick the best baseline row (lowest metric_key) among preferred labels."""
    for label in labels:
        candidates = [row for row in rows if row.get("label") == label]
        eligible = [
            row for row in candidates if _as_float(row.get(metric_key)) is not None
        ]
        if not eligible:
            continue
        best = min(eligible, key=lambda row: _as_float(row.get(metric_key)) or float("inf"))
        numeric = {
            key: _as_float(value)
            for key, value in best.items()
            if key.startswith("metric:") or key.startswith("extra:")
        }
        return best, {k: v for k, v in numeric.items() if v is not None}
    return None, {}


def _check_gates(
    candidate: Dict[str, float],
    baseline: Dict[str, float],
    *,
    improvement_metric: str,
    min_delta: float,
    ratio_limits: Mapping[str, float],
) -> Tuple[bool, List[str]]:
    """Return (passed, messages)."""
    messages: List[str] = []
    passed = True

    cand_val = candidate.get(improvement_metric)
    base_val = baseline.get(improvement_metric)
    if cand_val is None:
        passed = False
        messages.append(f"{improvement_metric}: missing in candidate metrics")
    elif base_val is None:
        messages.append(
            f"{improvement_metric}: no baseline available, treating as provisional pass"
        )
    else:
        improvement = base_val - cand_val
        if improvement >= min_delta:
            messages.append(
                f"{improvement_metric}: improved by {improvement:.4g} (>= {min_delta})"
            )
        else:
            passed = False
            messages.append(
                f"{improvement_metric}: improvement {improvement:.4g} < required {min_delta}"
            )

    for key, limit in ratio_limits.items():
        cand_metric = candidate.get(key)
        base_metric = baseline.get(key)
        if cand_metric is None or base_metric is None:
            messages.append(f"{key}: metric missing, skipping gate")
            continue
        allowed = base_metric * limit
        if abs(base_metric) < 1e-12:
            allowed = base_metric + 1e-6
        if cand_metric <= allowed:
            messages.append(
                f"{key}: {cand_metric:.4g} <= {allowed:.4g} (baseline {base_metric:.4g} * {limit})"
            )
        else:
            passed = False
            messages.append(
                f"{key}: {cand_metric:.4g} exceeds limit {allowed:.4g} (baseline {base_metric:.4g} * {limit})"
            )

    return passed, messages


def _parse_kv_pairs(values: List[str]) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    for entry in values:
        if "=" not in entry:
            raise ValueError(f"Expected key=value entry, got '{entry}'")
        key, value = entry.split("=", 1)
        parsed[key] = value
    return parsed


def _git_metadata() -> Dict[str, str]:
    meta: Dict[str, str] = {}
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        if rev:
            meta["commit"] = rev
    except Exception:
        pass
    try:
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        if branch:
            meta["branch"] = branch
    except Exception:
        pass
    return meta


def _find_checkpoint(directory: Path, names: List[str]) -> Optional[Path]:
    for name in names:
        candidate = directory / name
        if candidate.exists():
            return candidate
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the fast-to-SOTA automation pipeline.",
    )
    parser.add_argument("--train-config", required=True, help="Training config YAML")
    parser.add_argument(
        "--small-eval-config",
        help="Config to use for proxy evaluation (defaults to train config)",
    )
    parser.add_argument(
        "--full-eval-config",
        help="Config to use for full evaluation (defaults to train config)",
    )
    parser.add_argument("--run-id", help="Override generated run identifier")
    parser.add_argument("--train-stage", default="all", help="Training stage argument")
    parser.add_argument(
        "--train-extra-arg",
        action="append",
        default=[],
        help="Additional arguments to forward to train.py (may repeat)",
    )
    parser.add_argument(
        "--config-override",
        action="append",
        default=[],
        help="Apply dotted overrides to configs before running (e.g., latent.dim=48). May repeat.",
    )
    parser.add_argument("--skip-validation", action="store_true", help="Skip config checks")
    parser.add_argument("--skip-data-check", action="store_true", help="Skip dataset validation")
    parser.add_argument("--skip-dry-run", action="store_true", help="Skip dry-run estimate")
    parser.add_argument("--skip-training", action="store_true", help="Skip training step")
    parser.add_argument("--force-train", action="store_true", help="Force retraining even if existing checkpoints are marked complete")
    parser.add_argument("--skip-small-eval", action="store_true", help="Skip proxy evaluation")
    parser.add_argument("--skip-full-eval", action="store_true", help="Skip full evaluation")
    parser.add_argument("--redo-small-eval", action="store_true", help="Re-run proxy evaluation even if previous results exist in metadata")
    parser.add_argument("--redo-full-eval", action="store_true", help="Re-run full evaluation even if previous results exist in metadata")
    parser.add_argument("--skip-analysis", action="store_true", help="Skip post-run analysis report generation")
    parser.add_argument("--skip-comparison", action="store_true", help="Skip comparison reports against baseline")
    parser.add_argument(
        "--force-full-eval",
        action="store_true",
        help="Run full evaluation even if proxy gates fail",
    )
    parser.add_argument(
        "--min-small-delta",
        type=float,
        default=0.01,
        help="Required improvement (baseline - candidate) for proxy stage",
    )
    parser.add_argument(
        "--min-full-delta",
        type=float,
        default=0.01,
        help="Required improvement for full evaluation stage",
    )
    parser.add_argument(
        "--improvement-metric",
        default="metric:nrmse",
        help="Metric key used for improvement checks",
    )
    parser.add_argument(
        "--gate-ratio",
        action="append",
        default=[],
        metavar="KEY=FACTOR",
        help="Additional ratio gates (candidate <= factor * baseline)",
    )
    parser.add_argument(
        "--clear-default-ratio-gates",
        action="store_true",
        help="Do not apply default ratio gates from the playbook",
    )
    parser.add_argument(
        "--leaderboard-csv",
        default="reports/leaderboard.csv",
        help="Leaderboard CSV to update",
    )
    parser.add_argument(
        "--leaderboard-html",
        default="reports/leaderboard.html",
        help="Leaderboard HTML to update",
    )
    parser.add_argument(
        "--baseline-label",
        action="append",
        dest="baseline_labels",
        help="Labels (in preference order) to look for as the champion baseline",
    )
    parser.add_argument(
        "--champion-label",
        default="champion",
        help="Label used when a run is promoted",
    )
    parser.add_argument("--eval-device", default="cuda", help="Device for evaluation runs")
    parser.add_argument("--eval-tau", type=float, default=0.5, help="Tau for diffusion correction")
    parser.add_argument("--run-dir", default="artifacts/runs", help="Where to store run artifacts")
    parser.add_argument("--wandb-mode", default="offline", help="WANDB_MODE override during training/eval")
    parser.add_argument("--tag", action="append", default=[], help="Extra leaderboard tags (key=value)")
    parser.add_argument("--notes", help="Optional notes recorded on champion promotion")
    parser.add_argument(
        "--copy-checkpoints",
        action="store_true",
        help="Copy the produced checkpoint directory into the run artifact folder",
    )
    parser.add_argument(
        "--leaderboard-wandb",
        action="store_true",
        help="Forward leaderboard rows to W&B as well",
    )
    parser.add_argument("--leaderboard-wandb-project", help="W&B project for leaderboard rows")
    parser.add_argument("--leaderboard-wandb-entity", help="W&B entity for leaderboard rows")
    parser.add_argument(
        "--wandb-sync",
        action="store_true",
        help="Record orchestration summary, gates, and artifacts to W&B",
    )
    parser.add_argument("--wandb-project", help="W&B project for the orchestration run")
    parser.add_argument("--wandb-entity", help="W&B entity for the orchestration run")
    parser.add_argument("--wandb-run-name", help="Name to use for the orchestration W&B run")
    parser.add_argument("--wandb-group", help="Group to attach to the orchestration W&B run")
    parser.add_argument(
        "--wandb-tags",
        action="append",
        default=[],
        help="Additional W&B tags for the orchestration run (may repeat)",
    )
    parser.add_argument(
        "--analysis-max-rows",
        type=int,
        default=20000,
        help="Maximum number of history rows to fetch from W&B during run analysis",
    )
    parser.add_argument("--strict-exit", action="store_true", help="Exit non-zero when gates fail")

    args = parser.parse_args()

    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode

    train_config = Path(args.train_config).resolve()
    if not train_config.exists():
        raise FileNotFoundError(train_config)

    small_eval_config = Path(args.small_eval_config).resolve() if args.small_eval_config else train_config
    full_eval_config = Path(args.full_eval_config).resolve() if args.full_eval_config else train_config

    timestamp_dt = datetime.now(timezone.utc)
    run_id = args.run_id or f"run_{timestamp_dt.strftime('%Y%m%d_%H%M%S')}"
    timestamp = timestamp_dt.isoformat(timespec="seconds")
    run_root = (ROOT / args.run_dir / run_id).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    analysis_dir = run_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = run_root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    wandb_info_path = artifacts_dir / "wandb_run.json"

    # Persist the original training config for reference
    run_root.joinpath("train_config_snapshot.yaml").write_text(
        train_config.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    configs_dir = run_root / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    train_cfg_loaded = load_config_with_includes(train_config)
    cfg = _apply_overrides(train_cfg_loaded, args.config_override)
    resolved_train_config = _write_config(cfg, configs_dir / "train_resolved.yaml")

    small_cfg_loaded = load_config_with_includes(small_eval_config)
    small_cfg_resolved = _apply_overrides(small_cfg_loaded, args.config_override)
    resolved_small_config = _write_config(small_cfg_resolved, configs_dir / "small_eval_resolved.yaml")

    full_cfg_loaded = load_config_with_includes(full_eval_config)
    full_cfg_resolved = _apply_overrides(full_cfg_loaded, args.config_override)
    resolved_full_config = _write_config(full_cfg_resolved, configs_dir / "full_eval_resolved.yaml")

    ckpt_dir_value = cfg.get("checkpoint", {}).get("dir", "checkpoints")
    checkpoint_dir = Path(ckpt_dir_value)
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = (ROOT / checkpoint_dir).resolve()

    common_tags = {"pipeline": "fast_to_sota", "run": run_id}
    git_meta = _git_metadata()
    common_tags.update(git_meta)
    user_tags = _parse_kv_pairs(args.tag)

    default_ratio_limits = {
        "metric:conservation_gap": 1.0,
        "metric:bc_violation": 1.0,
        "metric:ece": 1.25,
        "metric:wall_clock": 1.0,
    }
    if args.clear_default_ratio_gates:
        ratio_limits: Dict[str, float] = {}
    else:
        ratio_limits = default_ratio_limits.copy()
    ratio_limits.update({key: float(value) for key, value in _parse_kv_pairs(args.gate_ratio).items()})

    baseline_labels = args.baseline_labels or ["champion", "full_eval", "baseline"]

    summary: Dict[str, object] = {
        "run_id": run_id,
        "timestamp": timestamp,
        "train_config_source": str(train_config),
        "small_eval_config_source": str(small_eval_config),
        "full_eval_config_source": str(full_eval_config),
        "train_config_resolved": str(resolved_train_config),
        "small_eval_config_resolved": str(resolved_small_config),
        "full_eval_config_resolved": str(resolved_full_config),
        "config_overrides": list(args.config_override),
        "checkpoint_dir": str(checkpoint_dir),
        "analysis_dir": str(analysis_dir),
        "artifacts_dir": str(artifacts_dir),
        "wandb_info_path": str(wandb_info_path),
    }

    leaderboard_csv = (ROOT / args.leaderboard_csv).resolve()
    leaderboard_html = (ROOT / args.leaderboard_html).resolve()

    summary["leaderboard_csv"] = str(leaderboard_csv)
    summary["leaderboard_html"] = str(leaderboard_html)

    metadata_path = checkpoint_dir / "metadata.json"
    metadata: Dict[str, Any] | None = None
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"⚠️  Failed to parse checkpoint metadata at {metadata_path}: {exc}. Recreating metadata.")
            metadata = None
    if metadata is None:
        metadata_path = _write_checkpoint_metadata(cfg, resolved_train_config, checkpoint_dir)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    summary["checkpoint_metadata"] = metadata

    # WandB tracking - let training script create the run
    # Orchestrator only tracks WandB info from training subprocess
    wandb_ctx = None
    wandb_context_file = run_root / "wandb_context.json"  # Path where training will save its context
    wandb_run = None  # Keep for backward compatibility with helper functions
    wandb_message_rows: List[List[str]] = []
    wandb_gate_rows: List[List[str]] = []

    # Note: We no longer create an orchestrator WandB run here.
    # The training subprocess creates the single WandB run for the entire pipeline.

    gate_results: Dict[str, Dict[str, object]] = {}
    promoted = False
    baseline_training_wandb: Optional[Dict[str, str]] = None
    training_wandb_info: Optional[Dict[str, Any]] = None

    def _normalize_tag(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.lower() in {"", "none", "null"}:
                return None
            return stripped
        return value

    def _ensure_dataset_symlinks(root: Path) -> None:
        mappings = [
            ("burgers1d_train.h5", "burgers1d_train_000.h5"),
        ]
        for target_name, source_name in mappings:
            target = root / target_name
            source = root / source_name
            if target.exists() or target.is_symlink():
                continue
            if not source.exists():
                continue
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(source.resolve())
            except Exception:
                continue

    def render_tags(stage: str) -> List[str]:
        tags = dict(common_tags)
        tags.update(user_tags)
        tags["stage"] = stage
        return [f"{k}={v}" for k, v in tags.items()]

    small_metrics_path: Optional[Path] = None
    full_metrics_path: Optional[Path] = None
    baseline_row: Optional[Dict[str, str]] = None
    baseline_metrics: Dict[str, float] = {}
    should_run_full = not args.skip_full_eval
    failed_gates = False

    summary["skip_validation"] = bool(args.skip_validation)
    summary["skip_data_check"] = bool(args.skip_data_check)
    summary["skip_dry_run"] = bool(args.skip_dry_run)
    summary["skip_training"] = bool(args.skip_training)
    summary["skip_small_eval"] = bool(args.skip_small_eval)
    summary["skip_full_eval"] = bool(args.skip_full_eval)
    summary["force_full_eval"] = bool(args.force_full_eval)
    summary["skip_analysis"] = bool(args.skip_analysis)
    summary["skip_comparison"] = bool(args.skip_comparison)
    summary["analysis_max_rows"] = int(args.analysis_max_rows)

    data_root_cfg = cfg.get("data", {}).get("root")
    if data_root_cfg:
        dataset_root = Path(data_root_cfg)
        if not dataset_root.is_absolute():
            dataset_root = (ROOT / dataset_root).resolve()
        _ensure_dataset_symlinks(dataset_root)

    try:
        # Validation and dry-run steps
        if not args.skip_validation:
            _run_command(
                [PYTHON, "scripts/validate_config.py", str(resolved_train_config)],
                env={"WANDB_MODE": args.wandb_mode},
                desc="validate-config",
            )
            _wandb_log_event(wandb_run, "validate_config_done")
        if not args.skip_data_check:
            _run_command(
                [PYTHON, "scripts/validate_data.py", str(resolved_train_config)],
                env={"WANDB_MODE": args.wandb_mode},
                desc="validate-data",
            )
            _wandb_log_event(wandb_run, "validate_data_done")
        if not args.skip_dry_run:
            _run_command(
                [PYTHON, "scripts/dry_run.py", str(resolved_train_config), "--estimate-only"],
                env={"WANDB_MODE": args.wandb_mode},
                desc="dry-run",
            )
            _wandb_log_event(wandb_run, "dry_run_done")

        # Training step (with metadata-aware skipping)
        metadata_trained = bool(metadata.get("trained")) if isinstance(metadata, dict) else False
        skip_training_due_to_metadata = metadata_trained and not args.force_train
        if skip_training_due_to_metadata:
            print("ℹ️  Skipping training: checkpoint metadata indicates training already completed. Use --force-train to override.")
        elif args.skip_training:
            print("ℹ️  Skipping training as requested via --skip-training")
        if not (args.skip_training or skip_training_due_to_metadata):
            train_cmd = [
                PYTHON,
                "scripts/train.py",
                "--config",
                str(resolved_train_config),
                "--stage",
                args.train_stage,
            ]
            train_cmd.extend(args.train_extra_arg)
            train_env = {
                "WANDB_MODE": args.wandb_mode,
                "FAST_TO_SOTA_WANDB_INFO": str(wandb_info_path),
                "WANDB_CONTEXT_FILE": str(wandb_context_file),  # Tell training where to save its context
            }
            _run_command(
                train_cmd,
                env=train_env,
                desc="train",
            )
            finished_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            _update_checkpoint_metadata(
                metadata_path,
                trained=True,
                trained_at=finished_at,
                last_small_eval=None,
                last_small_eval_at=None,
                last_full_eval=None,
                last_full_eval_at=None,
            )
            if isinstance(metadata, dict):
                metadata.update(
                    {
                        "trained": True,
                        "trained_at": finished_at,
                        "last_small_eval": None,
                        "last_small_eval_at": None,
                        "last_full_eval": None,
                        "last_full_eval_at": None,
                    }
                )

        training_wandb_info: Optional[Dict[str, Any]] = None
        if not (args.skip_training or skip_training_due_to_metadata):
            if wandb_info_path.exists():
                try:
                    training_wandb_info = json.loads(wandb_info_path.read_text(encoding="utf-8"))
                    summary["training_wandb"] = training_wandb_info
                    _update_checkpoint_metadata(metadata_path, training_wandb=training_wandb_info)
                    if isinstance(metadata, dict):
                        metadata["training_wandb"] = training_wandb_info
                    if training_wandb_info.get("id"):
                        common_tags["train_wandb_run"] = training_wandb_info.get("id")
                    if training_wandb_info.get("project"):
                        common_tags["train_wandb_project"] = training_wandb_info.get("project")
                    if training_wandb_info.get("entity"):
                        common_tags["train_wandb_entity"] = training_wandb_info.get("entity")
                    if training_wandb_info.get("url"):
                        summary["training_wandb_url"] = training_wandb_info.get("url")
                except Exception as exc:
                    summary["training_wandb_error"] = str(exc)
        else:
            summary["training_skipped"] = True
            if isinstance(metadata, dict):
                candidate = metadata.get("training_wandb")
                if isinstance(candidate, dict):
                    training_wandb_info = candidate
                    summary["training_wandb"] = training_wandb_info
            if isinstance(training_wandb_info, dict):
                if training_wandb_info.get("id"):
                    common_tags["train_wandb_run"] = training_wandb_info.get("id")
                if training_wandb_info.get("project"):
                    common_tags["train_wandb_project"] = training_wandb_info.get("project")
                if training_wandb_info.get("entity"):
                    common_tags["train_wandb_entity"] = training_wandb_info.get("entity")

        if isinstance(training_wandb_info, dict):
            _persist_wandb_info(wandb_info_path, training_wandb_info)

        operator_ckpt = _find_checkpoint(checkpoint_dir, ["operator_ema.pt", "operator.pt"])
        if operator_ckpt is None:
            raise FileNotFoundError(f"Operator checkpoint not found under {checkpoint_dir}")
        diffusion_ckpt = _find_checkpoint(
            checkpoint_dir,
            ["diffusion_residual_ema.pt", "diffusion_residual.pt"],
        )
        summary["operator_checkpoint"] = str(operator_ckpt)
        summary["diffusion_checkpoint"] = str(diffusion_ckpt) if diffusion_ckpt else None

        if args.copy_checkpoints and checkpoint_dir.exists():
            dest = run_root / "checkpoints"
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(checkpoint_dir, dest)

        # Pass WandB context to subprocesses (no more WANDB_MODE=disabled hack!)
        eval_env = {}
        if wandb_context_file and wandb_context_file.exists():
            eval_env["WANDB_CONTEXT_FILE"] = str(wandb_context_file)

        # Small evaluation stage
        small_flat: Optional[Dict[str, float]] = None
        metadata_last_small = metadata.get("last_small_eval") if isinstance(metadata, dict) else None
        if not args.skip_small_eval:
            reuse_small = isinstance(metadata_last_small, dict) and bool(metadata_last_small) and not args.redo_small_eval
            small_dir = run_root / "small_eval"
            small_prefix = small_dir / "results"
            small_metrics_path: Optional[Path] = small_prefix.with_suffix(".json")
            if reuse_small:
                print("ℹ️  Reusing previous small-eval metrics from checkpoint metadata")
                small_dir.mkdir(parents=True, exist_ok=True)
                small_flat = {
                    key: float(value)
                    for key, value in metadata_last_small.items()
                    if isinstance(value, (int, float))
                }
            else:
                small_dir.mkdir(parents=True, exist_ok=True)

                small_cmd = [
                    PYTHON,
                    "scripts/evaluate.py",
                    "--config",
                    str(resolved_small_config),
                    "--operator",
                    str(operator_ckpt),
                    "--device",
                    args.eval_device,
                    "--output-prefix",
                    str(small_prefix),
                    "--leaderboard-run-id",
                    f"{run_id}_small",
                    "--leaderboard-label",
                    "small_eval",
                    "--leaderboard-path",
                    str(leaderboard_csv),
                    "--leaderboard-html",
                    str(leaderboard_html),
                    "--leaderboard-notes",
                    f"run={run_id} stage=small",
                ]
                if diffusion_ckpt:
                    small_cmd.extend(["--diffusion", str(diffusion_ckpt), "--tau", str(args.eval_tau)])
                for tag in render_tags("small"):
                    small_cmd.extend(["--leaderboard-tag", tag])
                if args.leaderboard_wandb:
                    small_cmd.append("--leaderboard-wandb")
                    if args.leaderboard_wandb_project:
                        small_cmd.extend(["--leaderboard-wandb-project", args.leaderboard_wandb_project])
                    if args.leaderboard_wandb_entity:
                        small_cmd.extend(["--leaderboard-wandb-entity", args.leaderboard_wandb_entity])

                _run_command(
                    small_cmd,
                    env=dict(eval_env),
                    desc="evaluate-small",
                )
                small_metrics_path = small_prefix.with_suffix(".json")
                small_flat = _load_metrics(small_metrics_path)
                timestamp_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
                _update_checkpoint_metadata(
                    metadata_path,
                    last_small_eval=small_flat,
                    last_small_eval_at=timestamp_iso,
                )
                if isinstance(metadata, dict):
                    metadata["last_small_eval"] = small_flat
                    metadata["last_small_eval_at"] = timestamp_iso

            if reuse_small:
                try:
                    reuse_path = small_dir / "results.json"
                    if small_flat and not reuse_path.exists():
                        reuse_payload = {"reused": True, "source": "checkpoint_metadata", "metrics_flat": small_flat}
                        reuse_path.write_text(json.dumps(reuse_payload, indent=2), encoding="utf-8")
                    if small_metrics_path is None:
                        small_metrics_path = small_dir / "results.json"
                except Exception:
                    pass

            gate_results["small_eval"] = {"metrics": small_flat or {}, "reused": reuse_small}
            if small_flat and wandb_run is not None:
                _log_metrics_to_wandb(wandb_run, "small_eval", small_flat)
            # Evaluation subprocess logs metrics via WANDB_CONTEXT_FILE (no extra runs created)

            leaderboard_rows = _read_leaderboard(leaderboard_csv)
            baseline_row, baseline_metrics = _select_baseline(
                leaderboard_rows,
                baseline_labels,
                args.improvement_metric,
            )
            gate_results["baseline"] = baseline_row
            if baseline_row:
                baseline_training_wandb = {
                    "run": _normalize_tag(baseline_row.get("train_wandb_run")),
                    "project": _normalize_tag(baseline_row.get("train_wandb_project")),
                    "entity": _normalize_tag(baseline_row.get("train_wandb_entity")),
                }

                passed_small, small_messages = _check_gates(
                    small_flat or {},
                    baseline_metrics,
                    improvement_metric=args.improvement_metric,
                    min_delta=args.min_small_delta,
                    ratio_limits=ratio_limits,
                )
                gate_results["small_eval"]["passed"] = passed_small
                gate_results["small_eval"]["messages"] = small_messages
                # Gate results logged via orchestrator's wandb_run (if enabled)

                print("\n[gate] Small evaluation results:")
                for msg in small_messages:
                    print(f"  - {msg}")
                if wandb_run is not None:
                    wandb_message_rows.extend([["small_eval", msg] for msg in small_messages])
                    status_value = 1.0 if passed_small else 0.0
                    wandb_run.log({"fast_to_sota/small_eval_passed": status_value})
                    wandb_gate_rows.append(["small_eval", "passed" if passed_small else "failed", " | ".join(small_messages)])
                    _wandb_log_event(wandb_run, "small_eval_completed", status_value)
                if not passed_small:
                    print("Proxy evaluation gates failed.")
                    failed_gates = True
                    if not args.force_full_eval:
                        should_run_full = False
        else:
            print("Skipping small evaluation by request.")
            gate_results["small_eval"] = {"skipped": True}
            wandb_gate_rows.append(["small_eval", "skipped", ""])
            if wandb_run is not None:
                wandb_run.log({"fast_to_sota/small_eval_passed": -1.0})
                _wandb_log_event(wandb_run, "small_eval_completed", -1.0)

        if baseline_row is None:
            leaderboard_rows = _read_leaderboard(leaderboard_csv)
            baseline_row, baseline_metrics = _select_baseline(
                leaderboard_rows,
                baseline_labels,
                args.improvement_metric,
            )
            gate_results["baseline"] = baseline_row

        # Full evaluation stage
        metadata_last_full = metadata.get("last_full_eval") if isinstance(metadata, dict) else None
        if should_run_full and not args.skip_full_eval:
            full_dir = run_root / "full_eval"
            full_dir.mkdir(parents=True, exist_ok=True)
            full_prefix = full_dir / "results"
            full_metrics_path = full_prefix.with_suffix(".json")
            reuse_full = isinstance(metadata_last_full, dict) and bool(metadata_last_full) and not args.redo_full_eval
            if reuse_full:
                print("ℹ️  Reusing previous full-eval metrics from checkpoint metadata")
                full_flat = {
                    key: float(value)
                    for key, value in metadata_last_full.items()
                    if isinstance(value, (int, float))
                }
                full_dir.mkdir(parents=True, exist_ok=True)
            else:
                full_cmd = [
                    PYTHON,
                    "scripts/evaluate.py",
                    "--config",
                    str(resolved_full_config),
                    "--operator",
                    str(operator_ckpt),
                    "--device",
                    args.eval_device,
                    "--output-prefix",
                    str(full_prefix),
                    "--leaderboard-run-id",
                    f"{run_id}_full",
                    "--leaderboard-label",
                    "full_eval",
                    "--leaderboard-path",
                    str(leaderboard_csv),
                    "--leaderboard-html",
                    str(leaderboard_html),
                    "--leaderboard-notes",
                    f"run={run_id} stage=full",
                ]
                if diffusion_ckpt:
                    full_cmd.extend(["--diffusion", str(diffusion_ckpt), "--tau", str(args.eval_tau)])
                for tag in render_tags("full"):
                    full_cmd.extend(["--leaderboard-tag", tag])
                if args.leaderboard_wandb:
                    full_cmd.append("--leaderboard-wandb")
                    if args.leaderboard_wandb_project:
                        full_cmd.extend(["--leaderboard-wandb-project", args.leaderboard_wandb_project])
                    if args.leaderboard_wandb_entity:
                        full_cmd.extend(["--leaderboard-wandb-entity", args.leaderboard_wandb_entity])

                _run_command(
                    full_cmd,
                    env=dict(eval_env),
                    desc="evaluate-full",
                )
                full_flat = _load_metrics(full_metrics_path)
                timestamp_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
                _update_checkpoint_metadata(
                    metadata_path,
                    last_full_eval=full_flat,
                    last_full_eval_at=timestamp_iso,
                )
                if isinstance(metadata, dict):
                    metadata["last_full_eval"] = full_flat
                    metadata["last_full_eval_at"] = timestamp_iso

            if reuse_full:
                try:
                    full_dir.mkdir(parents=True, exist_ok=True)
                    reuse_path = full_dir / "results.json"
                    if full_flat and not reuse_path.exists():
                        reuse_payload = {"reused": True, "source": "checkpoint_metadata", "metrics_flat": full_flat}
                        reuse_path.write_text(json.dumps(reuse_payload, indent=2), encoding="utf-8")
                except Exception:
                    pass

            gate_results["full_eval"] = {"metrics": full_flat, "reused": reuse_full}
            if wandb_run is not None:
                _log_metrics_to_wandb(wandb_run, "full_eval", full_flat)
            # Evaluation subprocess logs metrics via WANDB_CONTEXT_FILE (no extra runs created)

            leaderboard_rows = _read_leaderboard(leaderboard_csv)
            baseline_row, baseline_metrics = _select_baseline(
                leaderboard_rows,
                baseline_labels,
                args.improvement_metric,
            )
            gate_results["baseline"] = baseline_row
            if baseline_row:
                baseline_training_wandb = {
                    "run": _normalize_tag(baseline_row.get("train_wandb_run")),
                    "project": _normalize_tag(baseline_row.get("train_wandb_project")),
                    "entity": _normalize_tag(baseline_row.get("train_wandb_entity")),
                }

            passed_full, full_messages = _check_gates(
                full_flat,
                baseline_metrics,
                improvement_metric=args.improvement_metric,
                min_delta=args.min_full_delta,
                ratio_limits=ratio_limits,
            )
            gate_results["full_eval"]["passed"] = passed_full
            gate_results["full_eval"]["messages"] = full_messages
            # Gate results logged via orchestrator's wandb_run (if enabled)

            print("\n[gate] Full evaluation results:")
            for msg in full_messages:
                print(f"  - {msg}")
            if wandb_run is not None:
                wandb_message_rows.extend([["full_eval", msg] for msg in full_messages])
                status_value = 1.0 if passed_full else 0.0
                wandb_run.log({"fast_to_sota/full_eval_passed": status_value})
                wandb_gate_rows.append(["full_eval", "passed" if passed_full else "failed", " | ".join(full_messages)])
                _wandb_log_event(wandb_run, "full_eval_completed", status_value)

            if passed_full:
                champion_run_id = f"{run_id}_champion"
                champion_tags = dict(common_tags)
                champion_tags.update(user_tags)
                champion_tags["stage"] = "champion"
                update_leaderboard(
                    metrics_path=full_metrics_path,
                    run_id=champion_run_id,
                    leaderboard_csv=leaderboard_csv,
                    leaderboard_html=leaderboard_html,
                    label=args.champion_label,
                    config=str(resolved_train_config),
                    notes=args.notes,
                    tags=champion_tags,
                    wandb_log=args.leaderboard_wandb,
                    wandb_project=args.leaderboard_wandb_project,
                    wandb_entity=args.leaderboard_wandb_entity,
                    wandb_run_name=champion_run_id,
                )
                promoted = True
                summary["champion_run_id"] = champion_run_id
                print(f"\n✅ Run promoted to champion ({champion_run_id}).")
            else:
                print("\nRun did not pass full-eval promotion gates.")
                failed_gates = True
        else:
            gate_results["full_eval"] = {"skipped": True}
            wandb_gate_rows.append(["full_eval", "skipped", ""])
            if wandb_run is not None:
                wandb_run.log({"fast_to_sota/full_eval_passed": -1.0})
                _wandb_log_event(wandb_run, "full_eval_completed", -1.0)
            print("Skipping full evaluation (not requested or gated).")

        analysis_report_path: Optional[Path] = None
        analysis_history_path: Optional[Path] = None
        comparison_report_path: Optional[Path] = None

        if (
            not args.skip_analysis
            and training_wandb_info
            and training_wandb_info.get("id")
            and training_wandb_info.get("project")
            and training_wandb_info.get("entity")
        ):
            print("📝 Running post-training analysis via analyze_run.py...")
            analysis_report_path = analysis_dir / "analysis.md"
            analysis_history_path = analysis_dir / "history.csv"
            analyze_cmd = [
                PYTHON,
                "scripts/analyze_run.py",
                training_wandb_info["id"],
                "--output",
                str(analysis_report_path),
                "--history-csv",
                str(analysis_history_path),
                "--max-rows",
                str(args.analysis_max_rows),
            ]
            if training_wandb_info.get("entity"):
                analyze_cmd.extend(["--entity", training_wandb_info["entity"]])
            if training_wandb_info.get("project"):
                analyze_cmd.extend(["--project", training_wandb_info["project"]])
            try:
                _run_command(analyze_cmd, desc="analyze-run")
                if analysis_report_path.exists():
                    summary["analysis_report"] = str(analysis_report_path)
                if analysis_history_path.exists():
                    summary["analysis_history"] = str(analysis_history_path)
            except subprocess.CalledProcessError as exc:
                summary["analysis_error"] = str(exc)
                analysis_report_path = None
                analysis_history_path = None

        if (
            promoted
            and not args.skip_comparison
            and baseline_training_wandb
            and baseline_training_wandb.get("run")
            and baseline_training_wandb.get("project")
            and baseline_training_wandb.get("entity")
            and training_wandb_info
            and training_wandb_info.get("id")
            and training_wandb_info.get("project")
            and training_wandb_info.get("entity")
            and baseline_training_wandb.get("run") != training_wandb_info.get("id")
        ):
            comparison_dir = run_root / "comparison"
            comparison_dir.mkdir(parents=True, exist_ok=True)
            comparison_report_path = comparison_dir / "comparison.md"
            print("🤝 Comparing new champion against baseline via compare_runs.py...")
            compare_cmd = [
                PYTHON,
                "scripts/compare_runs.py",
                baseline_training_wandb["run"],
                training_wandb_info["id"],
                "--output",
                str(comparison_report_path),
            ]
            entity_arg = training_wandb_info.get("entity") or baseline_training_wandb.get("entity")
            project_arg = training_wandb_info.get("project") or baseline_training_wandb.get("project")
            if entity_arg:
                compare_cmd.extend(["--entity", entity_arg])
            if project_arg:
                compare_cmd.extend(["--project", project_arg])
            try:
                _run_command(compare_cmd, desc="compare-runs")
                if comparison_report_path.exists():
                    summary["comparison_report"] = str(comparison_report_path)
            except subprocess.CalledProcessError as exc:
                summary["comparison_error"] = str(exc)
                comparison_report_path = None

        summary["gates"] = gate_results
        summary["promoted"] = promoted
        summary["small_eval_metrics"] = str(small_metrics_path) if small_metrics_path else None
        summary["full_eval_metrics"] = str(full_metrics_path) if full_metrics_path else None
        if baseline_row is not None:
            summary["baseline_row"] = baseline_row
        if baseline_training_wandb:
            summary["baseline_training_wandb"] = baseline_training_wandb
        summary["gates_failed"] = failed_gates
        summary["small_eval_attempted"] = not args.skip_small_eval
        summary["full_eval_attempted"] = bool(should_run_full and not args.skip_full_eval)

        summary_path = run_root / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nSummary written to {summary_path}")

        if wandb_run is not None and wandb is not None:
            if wandb_message_rows:
                message_table = wandb.Table(columns=["stage", "message"], data=wandb_message_rows)
                wandb_run.log({"fast_to_sota/messages": message_table})
            if wandb_gate_rows:
                gate_table = wandb.Table(columns=["stage", "status", "messages"], data=wandb_gate_rows)
                wandb_run.log({"fast_to_sota/gates": gate_table})
            if baseline_metrics:
                _log_metrics_to_wandb(wandb_run, "baseline", baseline_metrics)
            artifact = wandb.Artifact(name=f"fast_to_sota_summary_{run_id}", type="fast_to_sota")
            artifact.add_file(str(summary_path), name="summary.json")
            if leaderboard_csv.exists():
                artifact.add_file(str(leaderboard_csv), name="leaderboard.csv")
            if leaderboard_html.exists():
                artifact.add_file(str(leaderboard_html), name="leaderboard.html")
            if small_metrics_path and small_metrics_path.exists():
                artifact.add_file(str(small_metrics_path), name="small_eval.json")
            if full_metrics_path and full_metrics_path.exists():
                artifact.add_file(str(full_metrics_path), name="full_eval.json")
            if analysis_report_path and analysis_report_path.exists():
                artifact.add_file(str(analysis_report_path), name="run_analysis.md")
            if analysis_history_path and analysis_history_path.exists():
                artifact.add_file(str(analysis_history_path), name="run_history.csv")
            if comparison_report_path and comparison_report_path.exists():
                artifact.add_file(str(comparison_report_path), name="comparison.md")
            wandb_run.log_artifact(artifact)
            wandb_run.log({"fast_to_sota/promoted": float(promoted)})
            _wandb_log_event(wandb_run, "completed", 1.0 if not failed_gates else 0.0)

        if args.strict_exit and failed_gates:
            sys.exit(1)

    except Exception as exc:
        if wandb_run is not None:
            wandb_run.log({"fast_to_sota/error": str(exc)})
        raise
    finally:
        # Finish WandB context (clean way!)
        if wandb_ctx is not None:
            wandb_ctx.finish()
        elif wandb_run is not None:
            # Fallback for backward compatibility
            wandb_run.finish()

        # Force cleanup to ensure clean exit
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Give time for background processes to finish
        import time
        time.sleep(1)


if __name__ == "__main__":
    main()
